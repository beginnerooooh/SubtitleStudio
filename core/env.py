"""环境探测：CUDA / FFmpeg / 显存 → 量化策略；结果进程内缓存。

量化策略表：
- CUDA 且显存 >= 8GB  → float16
- CUDA 且显存未知/<8GB → int8_float16
- CPU                  → int8
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

# shutil.which 未命中时的 Windows 常见安装位置回退
_WINDOWS_DIRS = (
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\Program Files (x86)\ffmpeg\bin",
)

# CPU + 长音频 + 开启人声分离的警告阈值（秒）
_LONG_AUDIO_SECONDS = 30 * 60


@dataclass
class EnvInfo:
    device: str              # "cuda" | "cpu"
    compute_type: str        # "float16" | "int8_float16" | "int8"
    vram_gb: float | None    # None 表示 CPU 或显存未知
    ffmpeg_path: str | None
    ffprobe_path: str | None

    @property
    def ffmpeg_available(self) -> bool:
        return self.ffmpeg_path is not None


_cache: EnvInfo | None = None


def reset_cache() -> None:
    """清空探测缓存（供测试与手动刷新）。"""
    global _cache
    _cache = None


def _find_executable(name: str) -> str | None:
    path = shutil.which(name)
    if path:
        return path
    if os.name == "nt":
        exe = f"{name}.exe"
        for directory in _WINDOWS_DIRS:
            candidate = os.path.join(directory, exe)
            if os.path.isfile(candidate):
                return candidate
    return None


def _probe_cuda() -> tuple[bool, float | None]:
    """惰性导入 torch 探测 CUDA；任何失败都安全回退 CPU。"""
    try:
        import torch
    except Exception:  # ImportError 或损坏的安装
        return False, None
    try:
        if not torch.cuda.is_available():
            return False, None
        props = torch.cuda.get_device_properties(0)
        return True, props.total_memory / (1024**3)
    except Exception:
        return True, None


def detect_env(force: bool = False) -> EnvInfo:
    """探测运行环境并给出量化策略；结果缓存，force=True 强制重探。"""
    global _cache
    if _cache is not None and not force:
        return _cache
    has_cuda, vram_gb = _probe_cuda()
    if has_cuda and vram_gb is not None and vram_gb >= 8:
        compute_type = "float16"
    elif has_cuda:
        compute_type = "int8_float16"
    else:
        compute_type = "int8"
    _cache = EnvInfo(
        device="cuda" if has_cuda else "cpu",
        compute_type=compute_type,
        vram_gb=vram_gb,
        ffmpeg_path=_find_executable("ffmpeg"),
        ffprobe_path=_find_executable("ffprobe"),
    )
    return _cache


def long_run_warning(info: EnvInfo, duration_sec: float, enable_separation: bool) -> str | None:
    """CPU + 时长超阈值 + 开启人声分离 → 耗时警告；其余情况返回 None。"""
    if info.device == "cpu" and enable_separation and duration_sec > _LONG_AUDIO_SECONDS:
        return (
            "当前为 CPU 模式且音频超过 30 分钟，人声分离（Demucs）将耗费极长时间，"
            "建议关闭伴奏分离或在有 NVIDIA GPU 的机器上运行。"
        )
    return None
