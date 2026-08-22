"""强制对齐：wav2vec2 CTC + torchaudio.functional.forced_align。

策略（评审反馈后确定）：
- 短音频（≤15min）：全局对齐——32s 窗/30s 步进分块前向，保留中央帧去边界伪影，
  log_probs 无缝拼接后一次 forced_align，帧索引 ×0.02s 还原全局时间轴。
- 长音频（>15min）：Silero-VAD 区间合并 ~10min 块（仅在 VAD 静音边界切割），
  行按字数↔语音时长比例分配到块，块间 ±2 行重叠缓冲；重复行取距块边更远者。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

from core.errors import TaskCancelled
from core.models import SubtitleLine, SubtitleWord

SAMPLE_RATE = 16000
FRAME_RATE = 50           # wav2vec2 输出帧率（步长 320 采样 @16k）
FRAME_SEC = 1.0 / FRAME_RATE
WINDOW = 32.0             # 前向窗口时长（秒）
STEP = 30.0               # 窗口步进；保留中央帧后各窗贡献无缝拼接
_GLOBAL_ALIGN_MAX = 900.0  # ≤15min 走全局对齐
_CHUNK_TARGET = 600.0      # 长音频 VAD 分块目标时长（~10min）
OVERLAP = 2                # 相邻块行重叠缓冲（±2 行）

_MODEL_IDS = {
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "en": "facebook/wav2vec2-base-960h",
}


class AlignmentError(RuntimeError):
    """强制对齐失败。"""


@dataclass
class TokenSpan:
    """一个对齐目标（字符）的帧区间；end 为不含端点。"""

    start: int
    end: int
    score: float = 1.0


@dataclass
class AlignResult:
    lines: list[SubtitleLine]
    low_confidence: list[int]  # 低置信行索引（对应传入歌词的行号）


# ---------------- 纯函数 ----------------

def _plan_windows(duration: float, window: float = WINDOW, step: float = STEP):
    """规划前向窗口：返回 [(seg_start, seg_end, keep_start, keep_end)]（秒）。

    每窗 32s、步进 30s；非首窗丢弃头部 1s、非末窗丢弃尾部 1s（边界伪影），
    保留区间按时间轴无缝拼接 [0,31),[31,61),[61,91)…
    """
    if duration <= 0:
        return []
    starts = []
    s = 0.0
    while s < duration:
        starts.append(s)
        s += step
    plans = []
    for i, seg_start in enumerate(starts):
        seg_end = min(seg_start + window, duration)
        keep_start = seg_start + 1.0 if i > 0 else 0.0
        if i + 1 < len(starts):
            # 窗 i 保留终点 = 下一窗保留起点（重叠区中点切分），保证无缝拼接
            keep_end = min(starts[i + 1] + 1.0, duration)
        else:
            keep_end = seg_end  # 末窗保留到结尾
        if keep_end > keep_start:
            plans.append((seg_start, seg_end, keep_start, keep_end))
    return plans


def _spans_from_alignments(alignments, blank_id: int, scores=None) -> list[TokenSpan]:
    """CTC 对齐输出（逐帧标签）→ 每个目标 token 的帧区间。

    规则：blank 断开运行；相同标签连续（无 blank）属同一 token（CTC 折叠）；
    不同标签相邻则各自成段。非 blank 运行按顺序对应目标序列。
    """
    spans: list[TokenSpan] = []

    def _close(end: int) -> None:
        idx = len(spans)
        score = scores[idx] if scores and idx < len(scores) else 1.0
        spans.append(TokenSpan(run_start, end, score))

    run_label = None
    run_start = 0
    for frame, label in enumerate(alignments):
        if label == blank_id:
            if run_label is not None:
                _close(frame)
                run_label = None
        elif label != run_label:
            if run_label is not None:
                _close(frame)
            run_label = label
            run_start = frame
    if run_label is not None:
        _close(len(alignments))
    return spans


def _build_chunks(vad_intervals, target: float = _CHUNK_TARGET):
    """VAD 区间合并为 ~target 时长的块；仅在静音边界切割，单条超长区间独立成块。"""
    chunks = []
    cur = None
    for a, b in vad_intervals:
        if cur is None:
            cur = [a, b]
        elif b - cur[0] <= target:
            cur[1] = b
        else:
            chunks.append((cur[0], cur[1]))
            cur = [a, b]
    if cur is not None:
        chunks.append((cur[0], cur[1]))
    return chunks


def _assign_lines_to_chunks(lines, vad_intervals, chunks, overlap: int = OVERLAP):
    """行按字数↔语音时长比例分配到块，再向外扩 overlap 行作缓冲。

    返回每块的 (line_start, line_end)（含重叠，左闭右开）；空块为 (0, 0)。
    """
    weights = [max(1, sum(len(t.align) for t in line)) for line in lines]
    total_weight = sum(weights)
    total_speech = sum(b - a for a, b in vad_intervals)

    # 各块在「语音时间轴」（剔除静音）上的区间
    speech_ranges = []
    cum = 0.0
    i = 0
    for t0, t1 in chunks:
        s0 = cum
        while (
            i < len(vad_intervals)
            and vad_intervals[i][0] >= t0 - 1e-9
            and vad_intervals[i][1] <= t1 + 1e-9
        ):
            cum += vad_intervals[i][1] - vad_intervals[i][0]
            i += 1
        speech_ranges.append((s0, cum))

    assigned = [[] for _ in chunks]
    if total_weight and total_speech > 0:
        cum_w = 0.0
        for li, w in enumerate(weights):
            center = (cum_w + w / 2.0) / total_weight * total_speech
            cum_w += w
            for ci, (s0, s1) in enumerate(speech_ranges):
                if s0 <= center < s1:
                    assigned[ci].append(li)
                    break
            else:  # 落点恰在块边界外（舍入）：挂到最近的非空块
                for ci in range(len(chunks) - 1, -1, -1):
                    if assigned[ci]:
                        assigned[ci].append(li)
                        break

    n = len(lines)
    ranges = []
    for ci in range(len(chunks)):
        if assigned[ci]:
            lo, hi = assigned[ci][0], assigned[ci][-1]
            ranges.append((max(0, lo - overlap), min(n, hi + 1 + overlap)))
        else:
            ranges.append((0, 0))
    return ranges


def _choose_candidate(candidates):
    """重复行的多个对齐候选中，取行中点距块边更远者（上下文更足、更可靠）。"""

    def margin(cand):
        line, t0, t1 = cand
        mid = (line.start + line.end) / 2.0
        return min(mid - t0, t1 - mid)

    return max(candidates, key=margin)[0]


def _expand_targets(lines, vocab, sep_id, unk_id):
    """歌词 token 展平为字符 id 流：中文一字一 id，英文按字符展开。

    词间（英文词与任意相邻 token 之间）插入词分隔符（若模型词表支持）。
    返回 (token_refs, char_ids)：token_refs[i] 为 char_ids[i] 所属 (行, 词)，
    分隔符为 None。
    """
    char_ids: list[int] = []
    token_refs: list[Optional[tuple[int, int]]] = []
    prev_multi = False
    for li, line in enumerate(lines):
        for ti, tok in enumerate(line):
            multi = len(tok.align) > 1
            if sep_id is not None and token_refs and (multi or prev_multi):
                char_ids.append(sep_id)
                token_refs.append(None)
            for ch in tok.align:
                char_ids.append(vocab.get(ch, unk_id))
                token_refs.append((li, ti))
            prev_multi = multi
    return token_refs, char_ids


# ---------------- 模型与音频接缝（测试可替换） ----------------

_model_cache: dict = {}
_cache_lock = threading.Lock()


def reset_model_cache() -> None:
    with _cache_lock:
        _model_cache.clear()


def _get_models(language: str, device: str):
    """懒加载 wav2vec2 CTC 模型；返回 (model, processor, vocab, blank, sep, unk)。"""
    key = (language, device)
    with _cache_lock:
        if key in _model_cache:
            return _model_cache[key]
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # 懒加载

    model_id = _MODEL_IDS.get(language)
    if model_id is None:
        raise AlignmentError(f"不支持的对齐语言：{language}（可选：{'/'.join(_MODEL_IDS)}）")
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
    vocab = processor.tokenizer.get_vocab()
    blank_id = vocab[processor.tokenizer.pad_token]  # wav2vec2 的 CTC blank 即 <pad>
    sep_token = getattr(processor.tokenizer, "word_delimiter_token", None)
    sep_id = vocab.get(sep_token) if sep_token else None
    unk_id = vocab[processor.tokenizer.unk_token]
    bundle = (model, processor, vocab, blank_id, sep_id, unk_id)
    with _cache_lock:
        _model_cache[key] = bundle
    return bundle


def _forward(waveform, model, processor, device):
    """单窗前向：波形 → (T, V) 归一化 log_probs（CPU float32，控制内存峰值）。"""
    import torch

    inputs = processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    return torch.log_softmax(logits, dim=-1).squeeze(0).to("cpu", torch.float32)


def _forced_align(log_probs, char_ids, blank_id) -> list[TokenSpan]:
    """CTC 强制对齐：逐字符帧区间（与 char_ids 一一对应）。"""
    import torch
    from torchaudio.functional import forced_align

    alignments, scores = forced_align(
        log_probs.unsqueeze(0),
        torch.tensor([char_ids], dtype=torch.int32),
        blank=blank_id,
    )
    return _spans_from_alignments(
        alignments.squeeze(0).tolist(), blank_id, scores.squeeze(0).tolist()
    )


def _vad_intervals(wav_path: str):
    """Silero-VAD 语音区间（秒）；复用 faster-whisper 内置实现。"""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    wav = _load_audio(wav_path, 0.0, None)
    speech = get_speech_timestamps(wav, vad_options=VadOptions())
    return [(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in speech]


def _load_audio(wav_path: str, t0: float, t1):
    """读取 [t0, t1) 秒音频（float32 单声道）；t1=None 读到结尾。"""
    import soundfile as sf

    with sf.SoundFile(wav_path) as f:
        if f.samplerate != SAMPLE_RATE:
            raise AlignmentError(
                f"期望 {SAMPLE_RATE}Hz 单声道 WAV，实际采样率 {f.samplerate}Hz"
            )
        f.seek(int(t0 * f.samplerate))
        frames = None if t1 is None else int(round((t1 - t0) * f.samplerate))
        return f.read(frames, dtype="float32")


def _probe_duration(wav_path: str) -> float:
    import soundfile as sf

    return float(sf.info(wav_path).duration)


# ---------------- 对齐器 ----------------

class Aligner:
    """wav2vec2 CTC 强制对齐器：短音频全局对齐，长音频 VAD 滑窗分块。"""

    def __init__(
        self,
        language: str = "zh",
        device: str = "cpu",
        on_progress: Optional[Callable[[float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        confidence_threshold: float = 0.5,
    ):
        self.language = language
        self.device = device
        self.on_progress = on_progress
        self.cancel_event = cancel_event
        self.confidence_threshold = confidence_threshold

    def align(self, wav_path: str, lyrics) -> AlignResult:
        """对齐歌词到音频，返回与歌词行一一对应的 SubtitleLine 列表。"""
        self._check_cancel()
        if not lyrics:
            raise AlignmentError("歌词为空，无法强制对齐")
        duration = _probe_duration(wav_path)
        if duration <= _GLOBAL_ALIGN_MAX:
            chunks = [(0.0, duration)]
            ranges = [(0, len(lyrics))]
        else:
            vad = _vad_intervals(wav_path)
            if not vad:
                vad = [(0.0, duration)]
            chunks = _build_chunks(vad, _CHUNK_TARGET)
            ranges = _assign_lines_to_chunks(lyrics, vad, chunks, OVERLAP)
        model, processor, vocab, blank_id, sep_id, unk_id = _get_models(
            self.language, self.device
        )

        aligned: dict[int, list] = {}
        low: set[int] = set()
        n_chunks = max(1, len(chunks))
        for ci, ((t0, t1), (li0, li1)) in enumerate(zip(chunks, ranges)):
            if li0 >= li1:
                continue
            self._check_cancel()
            chunk_lines, chunk_scores = self._align_range(
                wav_path, t0, t1, lyrics[li0:li1],
                model, processor, vocab, blank_id, sep_id, unk_id,
                progress_base=ci / n_chunks, progress_span=1.0 / n_chunks,
            )
            for k, (line, score) in enumerate(zip(chunk_lines, chunk_scores)):
                gi = li0 + k
                aligned.setdefault(gi, []).append((line, t0, t1))
                if score < self.confidence_threshold:
                    low.add(gi)
        if len(aligned) != len(lyrics):
            raise AlignmentError("存在未能对齐的歌词行，请检查音频与歌词是否匹配")
        final = [_choose_candidate(aligned[gi]) for gi in range(len(lyrics))]
        return AlignResult(lines=final, low_confidence=sorted(low))

    def _align_range(
        self, wav_path, t0, t1, lines, model, processor, vocab,
        blank_id, sep_id, unk_id, progress_base, progress_span,
    ):
        """块内全局对齐：窗口前向 → forced_align → 字符合并回词 → 绝对时间。"""
        waveform = _load_audio(wav_path, t0, t1)
        log_probs = self._forward_windows(
            waveform, model, processor, progress_base, progress_span
        )
        token_refs, char_ids = _expand_targets(lines, vocab, sep_id, unk_id)
        spans = _forced_align(log_probs, char_ids, blank_id)
        if len(spans) != len(char_ids):
            raise AlignmentError(
                f"对齐结果数量不匹配：期望 {len(char_ids)}，实际 {len(spans)}"
            )
        # 字符区间按 (行, 词) 聚合（英文词 = 多字符合并）
        agg: dict[tuple[int, int], list] = {}
        for span, ref in zip(spans, token_refs):
            if ref is None:
                continue
            cur = agg.get(ref)
            if cur is None:
                agg[ref] = [span.start, span.end, span.score, 1]
            else:
                cur[0] = min(cur[0], span.start)
                cur[1] = max(cur[1], span.end)
                cur[2] += span.score
                cur[3] += 1

        out_lines: list[SubtitleLine] = []
        out_scores: list[float] = []
        for li, line in enumerate(lines):
            words: list[SubtitleWord] = []
            scores: list[float] = []
            for ti, tok in enumerate(line):
                cur = agg.get((li, ti))
                if cur is None:
                    raise AlignmentError(f"第 {li + 1} 行存在未对齐的词：{tok.display!r}")
                words.append(SubtitleWord(
                    text=tok.display,
                    start=t0 + cur[0] * FRAME_SEC,
                    end=t0 + cur[1] * FRAME_SEC,
                ))
                scores.append(cur[2] / cur[3])
            out_lines.append(SubtitleLine(words=words))
            out_scores.append(sum(scores) / len(scores))
        return out_lines, out_scores

    def _forward_windows(self, waveform, model, processor, progress_base, progress_span):
        """32s 窗/30s 步进前向并保留中央帧，拼接为块内全局 log_probs。"""
        import torch

        duration = len(waveform) / SAMPLE_RATE
        plans = _plan_windows(duration)
        outs = []
        for wi, (seg_s, seg_e, keep_s, keep_e) in enumerate(plans):
            self._check_cancel()
            seg = waveform[int(seg_s * SAMPLE_RATE):int(seg_e * SAMPLE_RATE)]
            lp = _forward(seg, model, processor, self.device)
            f0 = int(round((keep_s - seg_s) * FRAME_RATE))
            f1 = min(int(round((keep_e - seg_s) * FRAME_RATE)), lp.shape[0])
            if f1 > f0:
                outs.append(lp[f0:f1])
            if self.on_progress:
                self.on_progress(progress_base + progress_span * (wi + 1) / len(plans))
        if not outs:
            raise AlignmentError("音频过短，无法提取有效帧")
        return torch.cat(outs, dim=0)

    def _check_cancel(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise TaskCancelled("强制对齐已取消")
