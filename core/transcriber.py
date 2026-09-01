"""盲识别：faster-whisper 封装（懒加载 + 模型缓存 + OOM 降档链 + 进度/取消）。"""
from __future__ import annotations

import re
import threading
from typing import Callable, Optional

from core.errors import TaskCancelled
from core.models import SubtitleLine, SubtitleWord

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

# GPU 批处理推理的批大小（int8_float16 small/medium 在 4GB 显存下的安全值）
_BATCH_SIZE = 8

# whisper 段平均 logprob 低于此值 → 行标注低置信度（经验值，供人工复核）
LOW_LOGPROB = -0.8


class TranscriptionError(RuntimeError):
    """盲识别失败（含所有量化档位均 OOM）。"""


# GPU 量化降档链：按显存占用从高到低
_COMPUTE_CHAIN = ["float16", "int8_float16", "int8"]

# 模型缓存：(model_size, device, compute_type) -> WhisperModel
_model_cache: dict[tuple[str, str, str], object] = {}
_cache_lock = threading.Lock()


def _get_model(model_size: str, device: str, compute_type: str):
    """懒加载 faster-whisper 模型并按配置缓存，避免重复加载。"""
    key = (model_size, device, compute_type)
    with _cache_lock:
        if key in _model_cache:
            return _model_cache[key]
    import faster_whisper  # 懒加载：仅在首次使用时导入

    model = faster_whisper.WhisperModel(model_size, device=device, compute_type=compute_type)
    with _cache_lock:
        _model_cache[key] = model
    return model


def _evict_model(model_size: str, device: str, compute_type: str) -> None:
    """将 OOM 的模型逐出缓存，防止坏配置被复用。"""
    with _cache_lock:
        _model_cache.pop((model_size, device, compute_type), None)


def reset_model_cache() -> None:
    """清空模型缓存（测试与显存回收用）。"""
    with _cache_lock:
        _model_cache.clear()


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _bridge_space(prev_text: str, cur_text: str) -> bool:
    """拉丁词间需要补空格；任一侧边缘为汉字时不补（避免中文夹杂空格）。"""
    if not prev_text or not cur_text:
        return False
    return not _CJK_RE.match(prev_text[-1]) and not _CJK_RE.match(cur_text[0])


def _get_batched_pipeline(model):
    """GPU 批处理流水线（faster-whisper >= 1.0）；不可用时返回 None。"""
    try:
        from faster_whisper import BatchedInferencePipeline
    except ImportError:
        return None
    return BatchedInferencePipeline(model=model)


class Transcriber:
    """faster-whisper 盲识别器：segments 逐段消费，支持进度回调与取消。"""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = "auto",
        on_progress: Optional[Callable[[float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.on_progress = on_progress
        self.cancel_event = cancel_event

    def transcribe(
        self,
        path: str,
        total_duration: Optional[float] = None,
        clip_timestamps: Optional[list[tuple[float, float]]] = None,
    ) -> list[SubtitleLine]:
        """转写音频，返回行列表。OOM 时沿降档链自动降级重试。

        clip_timestamps：仅转写指定区域（秒，全局时间轴，时间戳不受影响），
        None 表示整段转写。用于声纹过滤后的定向转写。
        """
        self._check_cancel()
        clips: Optional[list[float]] = None
        if clip_timestamps is not None:
            clips = [float(t) for region in clip_timestamps for t in region]
            if not clips:
                return []
        last_oom: BaseException | None = None
        for device, compute_type in self._fallback_chain():
            try:
                model = _get_model(self.model_size, device, compute_type)
                segments = self._decode(model, device, path, clips)
                return self._collect(segments, total_duration)
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                last_oom = exc
                _evict_model(self.model_size, device, compute_type)
        raise TranscriptionError(
            f"显存不足：模型 {self.model_size} 在所有量化档位（含 CPU int8）均 OOM，"
            "请减小模型尺寸或改用 CPU 模式"
        ) from last_oom

    def _decode(self, model, device: str, path: str, clips: Optional[list[float]] = None):
        """解码入口：GPU 优先批处理流水线（2~4x），OOM 或不可用退回逐段解码。"""
        kwargs = {
            "language": None if self.language in (None, "auto") else self.language,
            "vad_filter": True,
            "word_timestamps": True,
            "beam_size": 5,
            # 不携带上文 kv cache：长音频提速 10~20%，且显著减少重复幻觉
            "condition_on_previous_text": False,
        }
        if clips is not None:
            kwargs["clip_timestamps"] = clips
        if device == "cuda":
            batched = _get_batched_pipeline(model)
            if batched is not None:
                try:
                    segments, _info = batched.transcribe(path, batch_size=_BATCH_SIZE, **kwargs)
                    return segments
                except Exception as exc:
                    if not _is_oom(exc):
                        raise
        segments, _info = model.transcribe(path, **kwargs)
        return segments

    def _fallback_chain(self) -> list[tuple[str, str]]:
        """从当前配置出发的 (device, compute_type) 降档序列。"""
        if self.device == "cpu":
            chain = [("cpu", self.compute_type)]
            if self.compute_type != "int8":
                chain.append(("cpu", "int8"))
            return chain
        start = _COMPUTE_CHAIN.index(self.compute_type) if self.compute_type in _COMPUTE_CHAIN else 0
        chain = [("cuda", c) for c in _COMPUTE_CHAIN[start:]]
        chain.append(("cpu", "int8"))
        return chain

    def _collect(self, segments, total_duration: Optional[float]) -> list[SubtitleLine]:
        """逐段消费 segments：检查取消、聚合成行、按已处理时长报进度。

        whisper 词 token 的词间空白（前导空格）转移到前一个词的显示末尾，
        与对齐模式 display 约定一致；CJK 一侧为汉字时不补空格。
        """
        lines: list[SubtitleLine] = []
        prev_end = 0.0
        last_word: Optional[SubtitleWord] = None
        for seg in segments:
            self._check_cancel()
            words: list[SubtitleWord] = []
            for w in seg.words or []:
                raw = w.word or ""
                text = raw.strip()
                if not text:
                    continue
                if raw[:1].isspace() and last_word is not None and _bridge_space(
                    last_word.text, text
                ):
                    last_word.text += " "
                start = w.start if w.start is not None else prev_end
                end = w.end if w.end is not None else start
                last_word = SubtitleWord(text=text, start=start, end=end)
                words.append(last_word)
                prev_end = end
            if not words:
                continue
            logprob = getattr(seg, "avg_logprob", None)
            low_conf = logprob is not None and float(logprob) < LOW_LOGPROB
            lines.append(SubtitleLine(
                words=words,
                low_confidence=low_conf,
                low_confidence_reason="识别置信度低" if low_conf else "",
            ))
            if total_duration and self.on_progress:
                self.on_progress(min(1.0, seg.end / total_duration))
        return lines

    def _check_cancel(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise TaskCancelled("盲识别已取消")
