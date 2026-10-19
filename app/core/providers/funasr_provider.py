"""FunASR / Paraformer 后端（中文优化，离线可用）。

与 whisper 后端的分工：
- whisper large-v3：标点与语感更接近人工字幕，作为默认；
- paraformer-zh + ct-punc：中文专有名词与热词更强，长音频吞吐稳定，
  适合作为对比/回退（实测同一 10 分钟样本：whisper 34s / funasr 13s）。
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

from core.providers import ASRProvider, AsrError, AsrSegment, AsrWord, register_provider

_CACHE: dict[str, tuple] = {}
_LOCK = threading.Lock()
# ≤4GB 显存的 batch_size_s（300 会 OOM 回退 CPU）
_BATCH_4GB = 60
_BATCH_8GB = 150
_BATCH_DEFAULT = 300


def reset_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def _vram_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        return 0.0


def _batch_seconds(device: str) -> int:
    if device != "cuda":
        return _BATCH_DEFAULT
    total = _vram_gb()
    if total <= 0:
        return _BATCH_DEFAULT
    if total <= 5:
        return _BATCH_4GB
    if total <= 9:
        return _BATCH_8GB
    return _BATCH_DEFAULT


def _get_models(device: str):
    with _LOCK:
        if device in _CACHE:
            return _CACHE[device]
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise AsrError("未安装 funasr（pip install funasr），或改用 whisper 引擎") from exc
    asr = AutoModel(model="paraformer-zh", vad_model="fsmn-vad",
                    punc_model="ct-punc", device=device, disable_update=True)
    bundle = (asr,)
    with _LOCK:
        _CACHE[device] = bundle
    return bundle


@register_provider
class FunASRProvider(ASRProvider):
    """paraformer-zh（含 fsmn-vad 与 ct-punc）。"""

    name = "funasr"

    def __init__(self, device: str = "cuda", hotwords: str = "", **_ignored):
        self.device = device
        self.hotwords = (hotwords or "").strip()
        self._cancel = None

    def transcribe(self, wav_path, *, language="auto", total_duration=None,
                   on_progress=None, cancel_event=None) -> list[AsrSegment]:
        self._cancel = cancel_event
        devices = ["cuda", "cpu"] if self.device == "cuda" else ["cpu"]
        last: Optional[BaseException] = None
        for device in devices:
            try:
                (asr,) = _get_models(device)
                kwargs = {"input": wav_path, "batch_size_s": _batch_seconds(device),
                          "sentence_timestamp": True}
                if self.hotwords:
                    kwargs["hotword"] = self.hotwords
                result = asr.generate(**kwargs)[0]
                self._check_cancel()
                if on_progress:
                    on_progress(1.0)
                return _to_segments(result)
            except Exception as exc:  # noqa: BLE001
                if "out of memory" not in str(exc).lower():
                    raise AsrError(f"FunASR 识别失败：{exc}") from exc
                last = exc
                with _LOCK:
                    _CACHE.pop(device, None)
        raise AsrError("FunASR 在 GPU/CPU 均显存不足") from last

    def _check_cancel(self) -> None:
        if self._cancel is not None and self._cancel.is_set():
            from core.errors import TaskCancelled

            raise TaskCancelled("识别已取消")


def _to_segments(result: dict) -> list[AsrSegment]:
    """FunASR 结果 → AsrSegment 列表。

    优先用 sentence_info（含分句与字级时间）；退化时用整段 timestamp。
    文本与 timestamp 的对应关系按「非标点字符」逐字匹配（paraformer 的
    timestamp 只覆盖有效字符，标点由 ct-punc 补入）。
    """
    text = (result.get("text") or "").strip()
    stamps = result.get("timestamp") or []
    sentence = result.get("sentence_info") or []

    def build(src_text: str, times) -> tuple[list[AsrWord], int]:
        words: list[AsrWord] = []
        ti = 0
        for ch in src_text:
            if ch.isspace():
                if words:
                    words[-1].text += ch
                continue
            if not _is_word_char(ch):
                if words:
                    words[-1].text += ch
                continue
            if ti >= len(times):
                break
            s, e = times[ti][0], times[ti][1]
            ti += 1
            words.append(AsrWord(text=ch, start=s / 1000.0, end=e / 1000.0))
        return words, ti

    segments: list[AsrSegment] = []
    if sentence:
        cursor = 0
        for sent in sentence:
            stext = (sent.get("text") or "").strip()
            if not stext:
                continue
            # 分句文本在整段文本中的起点（用于切分 timestamp）
            idx = text.find(stext, cursor)
            if idx < 0:
                idx = cursor
            cursor = idx + len(stext)
            times = sent.get("timestamp") or []
            # 分句 timestamp 与该分句有效字符一一对应
            words, _ = build(stext, times)
            if not words:
                continue
            segments.append(AsrSegment(
                start=float(sent.get("start", words[0].start * 1000)) / 1000.0,
                end=float(sent.get("end", words[-1].end * 1000)) / 1000.0,
                text=stext, words=words,
            ))
        if segments:
            return segments
    words, _ = build(text, stamps)
    if not words:
        return []
    return [AsrSegment(start=words[0].start, end=words[-1].end, text=text,
                       words=words)]


def _is_word_char(ch: str) -> bool:
    return ch.isalnum()


__all__ = ["FunASRProvider", "reset_cache"]
