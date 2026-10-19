"""人声分离：Demucs 封装（懒加载 + 失败回退原始音频 + 显存显式释放）。"""
from __future__ import annotations

import contextlib
import gc
import threading
from pathlib import Path
from typing import Callable, Optional

from core.errors import TaskCancelled

# 长音频分块分离参数：demucs 内部虽按 segment 切分计算，但整轨输入/输出
# 张量驻留 mix.device，3 小时级音频约需 20GB+ 内存必然 OOM，故外层分块。
_CHUNK_SECONDS = 600.0   # 每块新增音频时长（秒）
_CHUNK_CONTEXT = 1.0     # 相邻块边界前后各保留的上下文（秒），分离后裁掉防边界伪影


def _use_half(device: str, half: Optional[bool]) -> bool:
    """是否启用 fp16 autocast。

    实测（RTX 3050 Laptop 4GB，120s 真实音频）：开启后 6.79s → 5.24s
    （1.37x），分离波形与 fp32 相关系数 0.9998、相对 RMS 误差 1.8%
    —— 人声分离只作为 ASR/声纹的前置，该量级差异不影响下游。
    显式指定优先；未指定时 CUDA 默认开启、CPU 关闭。
    """
    if half is not None:
        return bool(half) and str(device).startswith("cuda")
    if not str(device).startswith("cuda"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _autocast_ctx(device: str, half: Optional[bool]):
    """demucs 未暴露精度开关 → 用 autocast 在不改张量类型的情况下走 fp16。"""
    if not _use_half(device, half):
        return contextlib.nullcontext()
    import torch

    return torch.autocast(device_type="cuda", dtype=torch.float16)


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

    长音频自动分块流式分离（soundfile 增量读写，峰值内存与总时长无关），
    成功后原子改名 vocals.wav，失败不留半成品。
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
        half: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.on_progress = on_progress
        self.on_warning = on_warning
        self.cancel_event = cancel_event
        self.half = half          # None=自动（CUDA 开），True/False 强制
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
        """soundfile 可读 → 分块流式（短文件单块等价整轨）；不可读 → 整轨兜底。"""
        try:
            import soundfile as sf

            info = sf.info(wav_path)
            if info.frames <= 0:
                raise ValueError("输入音频为空")
        except Exception:
            return self._separate_whole(wav_path, output_dir)
        return self._separate_chunked(wav_path, output_dir, info)

    def _separate_chunked(self, wav_path: str, output_dir: str, info) -> str:
        """分块流式分离：逐块读 → separate_tensor → 裁掉上下文 → 增量写。

        写入 vocals.part.wav，全部成功后原子改名为 vocals.wav（失败不留半成品，
        重试时自动覆盖）。进度回调按块起点/新增时长映射为全局进度。
        """
        import soundfile as sf
        import torch
        from demucs.api import Separator

        sr_in = int(info.samplerate)
        total = int(info.frames)
        total_sec = total / sr_in
        chunk_frames = int(_CHUNK_SECONDS * sr_in)
        ctx_frames = int(_CHUNK_CONTEXT * sr_in)

        # 块内进度 → 全局进度（audio_length 含上下文，近似线性已足够）
        chunk_state = {"base": 0.0, "span": 0.0}

        def callback(d: dict) -> None:
            self._check_cancel()
            if not (self.on_progress and d.get("state") == "end"):
                return
            length = float(d.get("audio_length") or 0)
            offset = float(d.get("segment_offset") or 0)
            if length > 0:
                local = min(1.0, offset / length)
                done = chunk_state["base"] + chunk_state["span"] * local
                self.on_progress(min(1.0, done / total_sec))

        separator = Separator(model=self.model_name, device=self.device,
                              progress=False, callback=callback)
        self._separator = separator
        sr_model = int(separator.samplerate)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "vocals.wav")
        part_path = str(out_dir / "vocals.part.wav")

        with sf.SoundFile(wav_path) as fin, \
                sf.SoundFile(part_path, "w", samplerate=16000, channels=1,
                             subtype="PCM_16") as fout:
            pos = 0
            while pos < total:
                self._check_cancel()
                read_start = max(0, pos - ctx_frames)
                read_end = min(total, pos + chunk_frames + ctx_frames)
                keep_end = min(pos + chunk_frames, total)
                fin.seek(read_start)
                data = fin.read(read_end - read_start, dtype="float32",
                                always_2d=True)
                chunk_state["base"] = pos / sr_in
                chunk_state["span"] = (keep_end - pos) / sr_in
                with _autocast_ctx(self.device, self.half):
                    _, stems = separator.separate_tensor(
                        torch.from_numpy(data.T), sr=sr_in)
                vocals = self._pick_vocals(stems)
                # 裁掉前后上下文（模型采样率域），只保留本块新增区间
                lo = round((pos - read_start) / sr_in * sr_model)
                hi = round((keep_end - read_start) / sr_in * sr_model)
                fout.write(_to_mono_16k(vocals[:, lo:hi], sr_model))
                pos = keep_end
        Path(part_path).replace(out_path)
        return out_path

    def _separate_whole(self, wav_path: str, output_dir: str) -> str:
        """整轨一次性分离（soundfile 不可读格式的兜底路径）。"""
        from demucs.api import Separator

        separator = Separator(
            model=self.model_name,
            device=self.device,
            progress=False,
            callback=self._demucs_callback,
        )
        self._separator = separator
        with _autocast_ctx(self.device, self.half):
            _, stems = separator.separate_audio_file(wav_path)
        vocals = self._pick_vocals(stems)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "vocals.wav")
        _write_wav(vocals, separator.samplerate, out_path)
        return out_path

    def _demucs_callback(self, info: dict) -> None:
        """Demucs 分块处理回调：先查取消，再上报阶段内进度。

        demucs 4.x 回调收单个进度字典：state（start/end）与
        segment_offset / audio_length（均为模型采样率下的帧数）。
        """
        self._check_cancel()
        if self.on_progress and info.get("state") == "end":
            total = float(info.get("audio_length") or 0)
            done = float(info.get("segment_offset") or 0)
            if total > 0:
                self.on_progress(min(1.0, done / total))

    @staticmethod
    def _pick_vocals(stems: dict):
        """按声部名取 vocals 轨；模型无该声部时退回第一轨。"""
        if "vocals" in stems:
            return stems["vocals"]
        return next(iter(stems.values()))

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


def _to_mono_16k(tensor, sample_rate: int):
    """人声张量 → 16kHz 单声道 numpy（整轨写出与分块增量写出共用）。"""
    wav = tensor
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = wav.mean(dim=0)  # 多声道 → 单声道
    if sample_rate != 16000:
        from torchaudio.functional import resample

        wav = resample(wav, sample_rate, 16000)
    return wav.detach().cpu().numpy()


def _write_wav(tensor, sample_rate: int, path: str) -> None:
    """人声张量 → 16kHz 单声道 WAV，统一下游（whisper/wav2vec2）输入格式。"""
    import soundfile as sf

    sf.write(path, _to_mono_16k(tensor, sample_rate), 16000)
