"""听歌识曲：shazamio 封装（同步接口）+ 演唱块合并 + 歌单时间戳导出。

识别链路：演唱块起点从**原始输入文件**（含伴奏）FFmpeg 截取高保真片段
（44.1kHz stereo）→ shazamio 异步识曲（asyncio.run + 超时）→ SongEntry。

所有识别失败（网络/无匹配/截取失败）均为尽力而为：日志提示 + 跳过，
任务不中断（识别失败的块由流水线保留转写兜底）。
"""
from __future__ import annotations

import asyncio
import csv
import io
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from core.errors import TaskCancelled

SONG_MIN_SECONDS = 30.0   # 演唱块最短总时长（过滤哼唱碎段）
SNIPPET_SECONDS = 12.0    # 送识别的片段时长（前 12s 足够高精度）
SNIPPET_TIMEOUT = 30.0    # Shazam 网络超时（秒）
BLOCK_GAP = 10.0          # 相邻语音段合并为同一演唱块的间隔阈值（VAD 间奏不碎歌）
MERGE_GAP = 30.0          # 相邻同名歌曲条目合并间隔


@dataclass
class SongEntry:
    """歌单时间戳条目。"""
    start: float
    end: float
    title: str
    artist: str
    confidence: Optional[float] = None


# ---------------- 纯函数 ----------------

def fmt_hms(seconds: float) -> str:
    """秒 → HH:MM:SS（截断到秒）。"""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def merge_blocks(
    segments: list[tuple[float, float]],
    gap: float = BLOCK_GAP,
) -> list[tuple[float, float]]:
    """相邻语音段（间隔 ≤ gap）合并为演唱块；输入排序不敏感。"""
    if not segments:
        return []
    spans = sorted(segments)
    merged: list[list[float]] = [list(spans[0])]
    for s, e in spans[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def merge_consecutive(
    entries: list[SongEntry],
    gap: float = MERGE_GAP,
) -> list[SongEntry]:
    """相邻同名歌曲条目合并（间奏/重复段）：起止取外沿，置信度取最大。"""
    if not entries:
        return []
    out: list[SongEntry] = []
    for e in sorted(entries, key=lambda x: x.start):
        prev = out[-1] if out else None
        if (
            prev is not None
            and e.start - prev.end <= gap
            and prev.title.casefold() == e.title.casefold()
            and prev.artist.casefold() == e.artist.casefold()
        ):
            prev.end = max(prev.end, e.end)
            prev.confidence = max(
                c for c in (prev.confidence, e.confidence) if c is not None
            ) if (prev.confidence is not None or e.confidence is not None) else None
        else:
            out.append(e)
    return out


def format_timeline_md(entries: list[SongEntry]) -> str:
    """SongEntry[] → songs_timeline.md 内容。"""
    lines = ["# 直播歌单时间戳索引", ""]
    if not entries:
        lines.append("未检测到歌曲。")
        return "\n".join(lines) + "\n"
    for e in sorted(entries, key=lambda x: x.start):
        text = f"《{e.title}》" + (f"- {e.artist}" if e.artist else "")
        conf = (
            f" (置信度: {int(round(e.confidence))}%)"
            if e.confidence is not None else ""
        )
        lines.append(f"- [{fmt_hms(e.start)} - {fmt_hms(e.end)}] {text}{conf}")
    return "\n".join(lines) + "\n"


def format_timeline_csv(entries: list[SongEntry]) -> str:
    """SongEntry[] → songs_timeline.csv 内容。"""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["start", "end", "title", "artist", "confidence"])
    for e in sorted(entries, key=lambda x: x.start):
        writer.writerow([
            fmt_hms(e.start), fmt_hms(e.end), e.title, e.artist,
            "" if e.confidence is None else int(round(e.confidence)),
        ])
    return buf.getvalue()


# ---------------- 识别 ----------------

def _cut_snippet(
    src: str | Path,
    start: float,
    seconds: float,
    dst: Path,
    ffmpeg: str = "ffmpeg",
    timeout: float = 60,
) -> Path:
    """从原始输入截取高保真片段（44.1kHz stereo，保留伴奏信息）。"""
    cmd = [
        ffmpeg, "-y",
        "-ss", str(start), "-t", str(seconds), "-i", str(src),
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RuntimeError(f"未找到 ffmpeg（{ffmpeg}）") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"片段截取超时（>{timeout:g}s）") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 截取失败：{(proc.stderr or '')[-300:].strip()}")
    return dst


def _shazam(snippet_path: Path, timeout: float) -> Optional[dict]:
    """shazamio 识曲（异步库同步包装）；任何失败返回 None。"""
    async def _run() -> dict:
        from shazamio import Shazam
        shazam = Shazam()
        return await shazam.recognize(str(snippet_path))

    try:
        result = asyncio.run(asyncio.wait_for(_run(), timeout=timeout))
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    track = result.get("track") or {}
    title = (track.get("title") or "").strip()
    if not title:
        return None
    conf = track.get("confidence")
    return {
        "title": title,
        "artist": (track.get("subtitle") or "").strip(),
        "confidence": float(conf) if isinstance(conf, (int, float)) else None,
    }


class SongRecognizer:
    """演唱块 → 歌曲条目（best-effort：失败跳过不中断）。"""

    def __init__(
        self,
        ffmpeg: str = "ffmpeg",
        snippet_seconds: float = SNIPPET_SECONDS,
        timeout: float = SNIPPET_TIMEOUT,
        on_log: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        self.ffmpeg = ffmpeg
        self.snippet_seconds = snippet_seconds
        self.timeout = timeout
        self.on_log = on_log or (lambda msg: None)
        self.on_progress = on_progress or (lambda r: None)
        self.cancel_event = cancel_event

    def recognize_blocks(
        self,
        source_path: str,
        blocks: list[tuple[float, float]],
        work_dir: Path | str,
    ) -> list[SongEntry]:
        """逐块截取片段识曲；返回识别成功的条目（未合并）。"""
        entries: list[SongEntry] = []
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        total = len(blocks)
        for i, (start, end) in enumerate(blocks):
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise TaskCancelled("用户取消")
            meta = self._recognize_snippet(source_path, start, i, work)
            if meta is not None:
                entries.append(SongEntry(start=start, end=end, **meta))
                self.on_log(f"识别到歌曲：《{meta['title']}》"
                            + (f" - {meta['artist']}" if meta["artist"] else ""))
            else:
                self.on_log("未能识别该演唱块（可能是清唱/翻唱或网络不可用）")
            self.on_progress((i + 1) / total if total else 1.0)
        return entries

    def _recognize_snippet(
        self, source_path: str, start: float, idx: int, work: Path,
    ) -> Optional[dict]:
        snippet = work / f"song_snippet_{idx}.wav"
        try:
            _cut_snippet(source_path, start, self.snippet_seconds,
                          snippet, ffmpeg=self.ffmpeg)
        except Exception as exc:
            self.on_log(f"歌曲片段截取失败：{exc}")
            return None
        meta = _shazam(snippet, self.timeout)
        return meta
