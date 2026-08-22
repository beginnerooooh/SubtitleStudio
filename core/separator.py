"""人声分离：Demucs 封装（懒加载 + 失败回退原始音频 + 显存显式释放）。"""
from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Callable, Optional

from core.errors import TaskCancelled


def _gc_collect() -> None:
    gc.collect()


def _empty_cuda_cache() -> None:
    """清空 CUDA 缓存；torch 缺失或无 CUDA 时静默跳过。"""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class VocalSeparator:
    """Demucs 人声分离器：htdemucs + two_stems=vocals，只出人声干声。

    分离完成后显式释放模型与 CUDA 缓存（Demucs 峰值 2~4GB 显存，
    防止与后续 faster-whisper / wav2vec2 争抢导致 OOM）。
    """

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str = "cpu",
        on_progress: Optional[Callable[[float], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.on_progress = on_progress
        self.on_warning = on_warning
        self.cancel_event = cancel_event
        self._separator = None  # 保留引用，结束时显式释放

    def separate(self, wav_path: str, output_dir: str) -> str:
        """分离人声并写出 vocals.wav，返回其路径。

        失败（OOM/依赖缺失/音频异常）时发警告并回退返回原始音频路径，
        任务不中断；用户取消（TaskCancelled）正常向上传播。
        """
        try:
            self._check_cancel()
            return self._do_separate(wav_path, output_dir)
        except TaskCancelled:
            raise
        except Exception as exc:
            self._warn(f"人声分离失败（{exc}），已回退使用原始音频继续任务")
            return wav_path
        finally:
            self._release()

    def _do_separate(self, wav_path: str, output_dir: str) -> str:
        from demucs.api import Separator  # 懒加载

        separator = Separator(
            model=self.model_name,
            device=self.device,
            two_stems="vocals",
            callback=self._demucs_callback,
        )
        self._separator = separator
        sources, sample_rate = separator.separate_audio_file(wav_path)
        vocals = self._pick_vocals(separator, sources)
        out_path = str(Path(output_dir) / "vocals.wav")
        _write_wav(vocals, sample_rate, out_path)
        return out_path

    def _demucs_callback(self, done: int, total: int) -> None:
        """Demucs 分块处理回调：先查取消，再上报阶段内进度。"""
        self._check_cancel()
        if self.on_progress and total:
            self.on_progress(min(1.0, done / total))

    @staticmethod
    def _pick_vocals(separator, sources):
        """按模型声部名取 vocals 轨；取不到时退回第一轨（two_stems 时首轨即人声）。"""
        names = getattr(getattr(separator, "model", None), "sources", None) or ["vocals", "no_vocals"]
        for i, name in enumerate(names):
            if name == "vocals":
                return sources[i]
        return sources[0]

    def _warn(self, message: str) -> None:
        if self.on_warning:
            self.on_warning(message)

    def _release(self) -> None:
        """显式释放：del 模型引用 → gc.collect → torch.cuda.empty_cache。"""
        self._separator = None
        _gc_collect()
        _empty_cuda_cache()

    def _check_cancel(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise TaskCancelled("人声分离已取消")


def _write_wav(tensor, sample_rate: int, path: str) -> None:
    """人声张量 → 16kHz 单声道 WAV，统一下游（whisper/wav2vec2）输入格式。"""
    import soundfile as sf

    wav = tensor
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = wav.mean(dim=0)  # 多声道 → 单声道
    if sample_rate != 16000:
        from torchaudio.functional import resample

        wav = resample(wav, sample_rate, 16000)
        sample_rate = 16000
    sf.write(path, wav.detach().cpu().numpy(), sample_rate)
