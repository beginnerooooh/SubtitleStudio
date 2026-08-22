"""FFmpeg 音频抽取：时长探测、转码为 16kHz/16bit/mono WAV。"""
from __future__ import annotations

import subprocess
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".webm",
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg",
}

_STDERR_TAIL = 500  # 报错时保留的 stderr 尾部长度


class AudioProcessError(RuntimeError):
    """FFmpeg/ffprobe 相关错误，message 面向用户可读。"""


def validate_input(path: Path | str) -> None:
    """校验输入文件存在且格式受支持；不合法抛 AudioProcessError。"""
    p = Path(path)
    if not p.is_file():
        raise AudioProcessError(f"输入文件不存在：{p}")
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise AudioProcessError(
            f"不支持的文件格式：{ext or '(无扩展名)'}"
            f"（支持 {'/'.join(sorted(SUPPORTED_EXTENSIONS))}）"
        )


def _tail(stderr: str) -> str:
    return stderr[-_STDERR_TAIL:].strip() or "(无 stderr 输出)"


def probe_duration(path: Path | str, ffprobe: str = "ffprobe", timeout: float = 60) -> float:
    """ffprobe 探测时长（秒）。"""
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise AudioProcessError(f"未找到 ffprobe（{ffprobe}），请检查 FFmpeg 安装") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessError(f"ffprobe 探测超时（>{timeout:g}s）") from exc
    if proc.returncode != 0:
        raise AudioProcessError(f"ffprobe 探测失败：{_tail(proc.stderr)}")
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise AudioProcessError(f"无法解析音频时长：{proc.stdout!r}") from exc


def extract_wav(
    src: Path | str,
    dst_dir: Path | str,
    ffmpeg: str = "ffmpeg",
    timeout: float = 3600,
) -> Path:
    """抽取/转码为 16kHz 16bit 单声道 WAV，返回输出文件路径。"""
    validate_input(src)
    src = Path(src)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / f"{src.stem}.wav"
    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        str(out),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise AudioProcessError(f"未找到 ffmpeg（{ffmpeg}），请检查 FFmpeg 安装") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessError(f"ffmpeg 转码超时（>{timeout:g}s）") from exc
    if proc.returncode != 0:
        raise AudioProcessError(f"ffmpeg 转码失败：{_tail(proc.stderr)}")
    if not out.is_file():
        raise AudioProcessError("ffmpeg 执行成功但未产出 WAV 文件")
    return out
