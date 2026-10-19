"""本地 whisper 后端：faster-whisper / CTranslate2。

关键实现决策（均来自本项目实测，见 docs/EVALUATION.md）：
1. 默认 **关闭 VAD 过滤**：Silero VAD 会把歌声与弱起句整段判为非语音，
   实测字符召回从 0.770 掉到 0.660；改用后处理剔除幻听更稳。
2. 长音频**分块 + 重叠**：单块 20 分钟、重叠 30 秒，
   避免整段解码的上下文漂移与显存抖动；块边界时间整体回偏移。
3. 词级时间戳是后续精修（core.timing）与切行的基础，始终开启。
4. 模型优先用本地目录（models/ct2/<name>），其次按 CT2 名称走 HF 缓存，
   最后回退到通用模型名——离线可用的关键。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable, Optional

from core.providers import ASRProvider, AsrError, AsrSegment, AsrWord, register_provider

# GPU 量化降档链（显存从高到低）
_COMPUTE_CHAIN = ("float16", "int8_float16", "int8")
_CACHE: dict[tuple, object] = {}
_CACHE_LOCK = threading.Lock()

# 常见别名 → CTranslate2 仓库名
_ALIASES = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


def reset_cache() -> None:
    """清空模型缓存（测试与显存回收）。"""
    with _CACHE_LOCK:
        _CACHE.clear()


def _models_root(models_dir: str) -> Path:
    return Path(models_dir) / "ct2"


def resolve_model_path(model: str, models_dir: str) -> str:
    """模型名/别名 → 可加载标识。

    优先级：绝对路径 → <models>/ct2/<name> → 别名仓库 → 原样（HF 缓存）。
    """
    p = Path(model)
    if p.is_dir():
        return str(p)
    local = _models_root(models_dir) / model
    if (local / "model.bin").is_file():
        return str(local)
    return _ALIASES.get(model, model)


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cublas" in text and "alloc" in text


def _device_chain(device: str, compute_type: str) -> list[tuple[str, str]]:
    """从当前配置出发的降档序列。"""
    if device == "cpu":
        chain = [("cpu", compute_type)]
        if compute_type != "int8":
            chain.append(("cpu", "int8"))
        return chain
    start = _COMPUTE_CHAIN.index(compute_type) if compute_type in _COMPUTE_CHAIN else 0
    return [("cuda", c) for c in _COMPUTE_CHAIN[start:]] + [("cpu", "int8")]


def _get_model(model_id: str, device: str, compute_type: str):
    key = (model_id, device, compute_type)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
    import faster_whisper

    try:
        model = faster_whisper.WhisperModel(
            model_id, device=device, compute_type=compute_type)
    except ValueError as exc:
        # 常见于 HF 名与本地目录混用；给出可操作的提示
        raise AsrError(
            f"加载 whisper 模型失败（{model_id} / {device}/{compute_type}）：{exc}\n"
            "提示：可先把模型放到 models/ct2/<名称>/，或在有网络的环境下载后复制。"
        ) from exc
    with _CACHE_LOCK:
        _CACHE[key] = model
    return model


@register_provider
class WhisperProvider(ASRProvider):
    """faster-whisper 后端。"""

    name = "whisper"

    def __init__(
        self,
        model: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        models_dir: str = "models",
        vad_filter: bool = False,
        word_timestamps: bool = True,
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
        condition_on_previous_text: bool = False,
        chunk_seconds: float = 1200.0,
        chunk_overlap: float = 30.0,
        initial_prompt: str = "",
        detect_default_language: str = "zh",
    ):
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.models_dir = models_dir
        self.vad_filter = vad_filter
        self.word_timestamps = word_timestamps
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature
        self.condition_on_previous_text = condition_on_previous_text
        self.chunk_seconds = max(60.0, chunk_seconds)
        self.chunk_overlap = max(0.0, chunk_overlap)
        self.initial_prompt = initial_prompt
        self.detect_default_language = detect_default_language
        self._model_id = resolve_model_path(model, models_dir)
        self._cancel = None

    # ---------------- 对外 ----------------

    def transcribe(self, wav_path, *, language="auto", total_duration=None,
                   on_progress=None, cancel_event=None) -> list[AsrSegment]:
        self._cancel = cancel_event
        duration = total_duration or _wav_duration(wav_path)
        plans = _plan_chunks(duration, self.chunk_seconds, self.chunk_overlap)
        out: list[AsrSegment] = []
        for idx, (start, end, keep_s, keep_e) in enumerate(plans):
            self._check_cancel()
            span = 1.0 / max(1, len(plans))
            base = idx * span

            def progress(r: float, _b=base, _s=span) -> None:
                if on_progress:
                    on_progress(min(1.0, _b + _s * r))

            # 只在「真正是子区间」时用 clip_timestamps：整段传 clip 会让
            # CTranslate2 的语言检测拿不到足够特征而 IndexError（实测）。
            whole = (start <= 0.0 and (duration <= 0 or end >= duration - 1e-3))
            segs = self._transcribe_span(
                wav_path, start, end, language, progress, whole=whole)
            if whole:
                out.extend(segs)
            else:
                out.extend(s for s in segs if _in_keep(s, keep_s, keep_e))
        out.sort(key=lambda s: s.start)
        return _dedupe(out)

    # ---------------- 内部 ----------------

    def _transcribe_span(self, wav_path, start, end, language, on_progress,
                         *, whole: bool = False) -> list[AsrSegment]:
        kwargs = {
            "language": None if language in (None, "auto") else language,
            "vad_filter": self.vad_filter,
            "word_timestamps": self.word_timestamps,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "condition_on_previous_text": self.condition_on_previous_text,
        }
        if self.initial_prompt:
            kwargs["initial_prompt"] = self.initial_prompt
        clip = None if whole else [start, end]

        last: Optional[BaseException] = None
        for device, compute_type in self._device_chain_cfg():
            try:
                model = _get_model(self._model_id, device, compute_type)
                if clip is not None:
                    kwargs["clip_timestamps"] = clip
                try:
                    segments, info = model.transcribe(wav_path, **kwargs)
                    collected = self._collect(segments, info, on_progress)
                except IndexError:
                    # 语言自动检测在本模型/本音频上会拿不到特征（实测
                    # large-v3 + 长段无 VAD 时 IndexError）；退回中文，
                    # 与直播场景的默认语言一致。
                    if kwargs.get("language") is not None:
                        raise
                    kwargs["language"] = self.detect_default_language or "zh"
                    self._warn("语言自动检测失败，已回退为中文（可在设置里指定语言）")
                    segments, info = model.transcribe(wav_path, **kwargs)
                    collected = self._collect(segments, info, on_progress)
                return collected
            except Exception as exc:  # noqa: BLE001 - 需要按 OOM 分类
                if not _is_oom(exc):
                    raise AsrError(f"识别失败：{exc}") from exc
                last = exc
                with _CACHE_LOCK:
                    _CACHE.pop((self._model_id, device, compute_type), None)
                self._warn(f"显存不足：{self.model}（{device}/{compute_type}）已降级重试")
        raise AsrError(f"显存不足：{self.model} 在所有量化档位均失败") from last

    def _device_chain_cfg(self):
        return _device_chain(self.device, self.compute_type)

    def _collect(self, segments, info, on_progress) -> list[AsrSegment]:
        out: list[AsrSegment] = []
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        for seg in segments:
            self._check_cancel()
            words = []
            for w in seg.words or []:
                text = (w.word or "")
                if not text.strip():
                    continue
                words.append(AsrWord(
                    text=text,
                    start=float(w.start if w.start is not None else seg.start),
                    end=float(w.end if w.end is not None else seg.end),
                    prob=float(getattr(w, "probability", 1.0) or 1.0),
                ))
            if not words:
                continue
            out.append(AsrSegment(
                start=float(seg.start), end=float(seg.end),
                text=(seg.text or "").strip(), words=words,
                avg_logprob=float(getattr(seg, "avg_logprob", 0.0) or 0.0),
                no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
                compression_ratio=float(getattr(seg, "compression_ratio", 0.0) or 0.0),
                language=str(getattr(info, "language", "") or ""),
            ))
            if on_progress and duration > 0:
                on_progress(min(1.0, float(seg.end) / duration))
        return out

    def _check_cancel(self) -> None:
        if self._cancel is not None and self._cancel.is_set():
            from core.errors import TaskCancelled

            raise TaskCancelled("识别已取消")

    def _warn(self, message: str) -> None:
        print(f"[asr] {message}")


def _plan_chunks(duration: float, chunk: float, overlap: float
                 ) -> list[tuple[float, float, float, float]]:
    """长音频分块计划：[(decode_start, decode_end, keep_start, keep_end)]。

    每块独立解码（含 overlap 上下文），但只保留「独占区间」内的结果，
    避免重叠区被重复转写两次。首块与末块的独占区间外扩到边界。
    """
    if duration <= chunk or chunk <= 0:
        return [(0.0, duration, 0.0, duration)]
    plans: list[tuple[float, float, float, float]] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + chunk)
        keep_start = 0.0 if start <= 0 else start + overlap / 2
        keep_end = duration if end >= duration else end - overlap / 2
        plans.append((start, end, keep_start, keep_end))
        if end >= duration:
            break
        start = end - overlap
    return plans


def _in_keep(seg, keep_start: float, keep_end: float) -> bool:
    """段中点是否落在独占区间内（跨界的段归后一块）。"""
    mid = (seg.start + seg.end) / 2
    return keep_start <= mid < keep_end


def _wav_duration(path: str) -> float:
    try:
        import soundfile as sf

        return float(sf.info(path).duration)
    except Exception:
        return 0.0


def _dedupe(segments: list[AsrSegment]) -> list[AsrSegment]:
    """重叠区残影去重：时间高度重叠且文本相同/包含的段只保留得分更高者。"""
    if len(segments) < 2:
        return segments
    kept: list[AsrSegment] = []
    for seg in segments:
        dup = False
        for i, prev in enumerate(kept):
            overlap = min(seg.end, prev.end) - max(seg.start, prev.start)
            if overlap <= 0:
                continue
            short = min(seg.end - seg.start, prev.end - prev.start) or 1e-6
            if overlap / short < 0.6:
                continue
            a, b = seg.text.replace(" ", ""), prev.text.replace(" ", "")
            if a == b or a in b or b in a:
                # 保留更长的文本（信息量更大），时间取并集起点
                if len(a) > len(b):
                    kept[i] = seg
                dup = True
                break
        if not dup:
            kept.append(seg)
    kept.sort(key=lambda s: s.start)
    return kept


# 兼容旧接口：pipeline 早期版本直接调 _get_model
def _legacy_model(model_size: str, device: str, compute_type: str):
    return _get_model(_ALIASES.get(model_size, model_size), device, compute_type)


__all__ = ["WhisperProvider", "resolve_model_path", "reset_cache",
           "_plan_chunks", "LOW_LOGPROB"]
