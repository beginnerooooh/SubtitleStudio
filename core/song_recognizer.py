"""听歌识曲：shazamio 封装（批量并行）+ 演唱块合并 + 歌单时间戳导出。

识别链路（v1.1 并行化）：
1. ThreadPoolExecutor 并行 FFmpeg 截取高保真片段（44.1kHz stereo）
2. 单事件循环 + 单 Shazam 客户端批量并发识曲（Semaphore 限流），
   较逐块串行提速 3~4 倍（网络耗时主导）
3. 识别结果回调逐条上报（完成即报，长任务不黑盒）

所有识别失败（网络/无匹配/截取失败）均为尽力而为：日志提示 + 跳过，
任务不中断（识别失败的块由流水线保留转写兜底）。
"""
from __future__ import annotations

import asyncio
import csv
import io
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from core.errors import TaskCancelled

SONG_MIN_SECONDS = 30.0   # 演唱块最短总时长（过滤哼唱碎段）
SNIPPET_SECONDS = 12.0    # 送识别的片段时长（前 12s 足够高精度）
SNIPPET_TIMEOUT = 30.0    # 单块 Shazam 网络超时（秒）
BLOCK_GAP = 10.0          # 相邻语音段合并为同一演唱块的间隔阈值（VAD 间奏不碎歌）
MERGE_GAP = 30.0          # 相邻同名歌曲条目合并间隔
CONCURRENCY = 4           # 并行截片 / 并发识曲的默认工作线程与信号量


@dataclass
class SongEntry:
    """歌单时间戳条目。lyrics_lrc 为拉取到的同步歌词原文（LRC，可空）。"""
    start: float
    end: float
    title: str
    artist: str
    confidence: Optional[float] = None
    lyrics_lrc: Optional[str] = None


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
    """相邻同名歌曲条目合并（间奏/重复段）：起止取外沿，置信度取最大。

    已拉取的同步歌词保留首个非空值（同名歌曲歌词相同）。
    """
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
            prev.lyrics_lrc = prev.lyrics_lrc or e.lyrics_lrc
        else:
            out.append(e)
    return out


def format_timeline_md(entries: list[SongEntry]) -> str:
    """SongEntry[] → songs_timeline.md 内容（含歌词获取状态标记）。"""
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
        lyric = " [已配同步歌词]" if e.lyrics_lrc else ""
        lines.append(f"- [{fmt_hms(e.start)} - {fmt_hms(e.end)}] {text}{conf}{lyric}")
    return "\n".join(lines) + "\n"


def format_timeline_csv(entries: list[SongEntry]) -> str:
    """SongEntry[] → songs_timeline.csv 内容（lyrics 列：1=有同步歌词）。"""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["start", "end", "title", "artist", "confidence", "lyrics"])
    for e in sorted(entries, key=lambda x: x.start):
        writer.writerow([
            fmt_hms(e.start), fmt_hms(e.end), e.title, e.artist,
            "" if e.confidence is None else int(round(e.confidence)),
            1 if e.lyrics_lrc else 0,
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


def _parse_track(result: object) -> Optional[dict]:
    """shazamio recognize 返回 → {title, artist, confidence}；无命中 None。"""
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


def _shazam_batch(
    snippet_paths: list[Path],
    timeout: float = SNIPPET_TIMEOUT,
    concurrency: int = CONCURRENCY,
    cancel_event: Optional[threading.Event] = None,
    on_result: Optional[Callable[[int, Optional[dict]], None]] = None,
) -> list[Optional[dict]]:
    """批量 shazam 识曲：单事件循环 + 单客户端 + Semaphore 并发限流。

    on_result(idx, meta) 在每个片段识别完成时同步回调（须轻量，
    如入队/更新浮点）。整体失败（shazamio 不可用）返回全 None 列表；
    取消事件置位时抛 TaskCancelled。
    """
    if not snippet_paths:
        return []

    async def run() -> list[Optional[dict]]:
        from shazamio import Shazam
        client = Shazam()
        sem = asyncio.Semaphore(max(1, concurrency))
        results: dict[int, Optional[dict]] = {}
        lock = asyncio.Lock()

        async def one(idx: int, path: Path) -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise TaskCancelled("用户取消")
            async with sem:
                if cancel_event is not None and cancel_event.is_set():
                    raise TaskCancelled("用户取消")
                meta: Optional[dict] = None
                try:
                    raw = await asyncio.wait_for(
                        client.recognize(str(path)), timeout=timeout)
                    meta = _parse_track(raw)
                except TaskCancelled:
                    raise
                except Exception:
                    meta = None
                async with lock:
                    results[idx] = meta
                if on_result is not None:
                    on_result(idx, meta)

        await asyncio.gather(*(one(i, p) for i, p in enumerate(snippet_paths)))
        return [results.get(i) for i in range(len(snippet_paths))]

    try:
        return asyncio.run(run())
    except TaskCancelled:
        raise
    except Exception:
        return [None] * len(snippet_paths)


class SongRecognizer:
    """演唱块 → 歌曲条目（best-effort：失败跳过不中断）。

    v1.1：截片与识曲并行化（默认并发 4），单块流程不变。
    """

    def __init__(
        self,
        ffmpeg: str = "ffmpeg",
        snippet_seconds: float = SNIPPET_SECONDS,
        timeout: float = SNIPPET_TIMEOUT,
        concurrency: int = CONCURRENCY,
        on_log: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        self.ffmpeg = ffmpeg
        self.snippet_seconds = snippet_seconds
        self.timeout = timeout
        self.concurrency = max(1, concurrency)
        self.on_log = on_log or (lambda msg: None)
        self.on_progress = on_progress or (lambda r: None)
        self.cancel_event = cancel_event

    def recognize_blocks(
        self,
        source_path: str,
        blocks: list[tuple[float, float]],
        work_dir: Path | str,
    ) -> list[SongEntry]:
        """并行截片 + 批量识曲；返回识别成功的条目（未合并）。"""
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        total = len(blocks)
        if total == 0:
            return []
        self._check_cancel()

        # 阶段 A：并行截片段（FFmpeg 子进程，各块互不依赖）
        snippets: list[Optional[Path]] = [None] * total
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {
                pool.submit(self._cut_block, source_path, start, i, work): i
                for i, (start, _end) in enumerate(blocks)
            }
            done = 0
            for fut, i in futures.items():
                snippets[i] = fut.result()  # 阻塞至对应片段完成（异常已在内部消化）
                done += 1
                self.on_progress(0.5 * done / total)
                self._check_cancel()

        # 阶段 B：单事件循环批量识曲（并发限流，逐块完成回调）
        valid = [(i, p) for i, p in enumerate(snippets) if p is not None]
        metas: list[Optional[dict]] = [None] * len(valid)
        if valid:
            reported = [0]

            def on_result(idx_in_batch: int, meta: Optional[dict]) -> None:
                reported[0] += 1
                self.on_progress(0.5 + 0.5 * reported[0] / total)

            metas = _shazam_batch(
                [p for _, p in valid],
                timeout=self.timeout,
                concurrency=self.concurrency,
                cancel_event=self.cancel_event,
                on_result=on_result,
            )

        entries: list[SongEntry] = []
        for (i, _p), meta in zip(valid, metas):
            start, end = blocks[i]
            if meta is not None:
                entries.append(SongEntry(start=start, end=end, **meta))
                self.on_log(f"识别到歌曲：《{meta['title']}》"
                            + (f" - {meta['artist']}" if meta["artist"] else ""))
            else:
                self.on_log("未能识别该演唱块（可能是清唱/翻唱或网络不可用）")
        if not valid:
            self.on_progress(1.0)
        return entries

    def _cut_block(
        self, source_path: str, start: float, idx: int, work: Path,
    ) -> Optional[Path]:
        """截取单个片段；失败记日志返回 None（不中断其余块）。"""
        snippet = work / f"song_snippet_{idx}.wav"
        try:
            return _cut_snippet(source_path, start, self.snippet_seconds,
                                snippet, ffmpeg=self.ffmpeg)
        except Exception as exc:
            self.on_log(f"歌曲片段截取失败：{exc}")
            return None

    def _check_cancel(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise TaskCancelled("用户取消")
