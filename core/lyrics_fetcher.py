"""在线歌词拉取：LRCLIB 客户端 + LRC 解析 + 歌词字幕行构造。

数据源：LRCLIB（https://lrclib.net）——免费、无需 API Key，提供
同步歌词（syncedLyrics，带 LRC 时间戳）与纯文本歌词（plainLyrics）。

链路：识曲得到 (title, artist) → fetch_lyrics 查询（进程内缓存去重）
→ parse_lrc 解析同步歌词 → build_lyric_lines 按演唱块起点偏移并
逐词均匀分布时间，生成与盲识别/对齐一致的 SubtitleLine。

所有失败（网络/无匹配/解析异常）均为尽力而为：返回 None 由调用方
回退到语音识别，任务不中断。
"""
from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

from core.models import SubtitleLine, SubtitleWord
from core.text import tokenize

LRCLIB_BASE = "https://lrclib.net"
USER_AGENT = "SubtitleStudio/1.1 (subtitle-studio; https://example.com)"
FETCH_TIMEOUT = 15.0            # 单次 HTTP 超时（秒）
DEFAULT_LINE_SPAN = 5.0         # 末行/无后续行时的默认行时长（秒）
MIN_WORD_DURATION = 0.04        # 均匀分布后的最短词时长（秒）

# 进程内缓存：(title, artist) casefold → LyricTrack | None（None 也缓存，避免重复请求）
_cache: dict[tuple[str, str], Optional["LyricTrack"]] = {}
_cache_lock = threading.Lock()

# LRC 行首时间戳（可叠加多个）：[mm:ss] [mm:ss.xx] [mm:ss.xxx] [mm:ss:xx]
_LRC_TS_RE = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]")
# [offset:+500] / [offset:-250]（毫秒；正值歌词提前出现）
_LRC_OFFSET_RE = re.compile(r"^\[offset:\s*([+-]?\d+)\s*\]", re.IGNORECASE)


@dataclass
class LyricTrack:
    """一次歌词查询结果（synced 与 plain 至少一项非空）。"""

    title: str
    artist: str
    synced: Optional[str] = None   # 原始 LRC 文本（带时间戳）
    plain: Optional[str] = None    # 纯文本歌词


# ---------------- HTTP ----------------

def _get_json(url: str, timeout: float = FETCH_TIMEOUT):
    """GET → JSON；404/超时/网络错误统一抛异常由上层捕获。"""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _track_from_payload(data: dict) -> Optional[LyricTrack]:
    """LRCLIB 记录 → LyricTrack；无任何歌词正文返回 None。"""
    synced = (data.get("syncedLyrics") or "").strip() or None
    plain = (data.get("plainLyrics") or "").strip() or None
    if not synced and not plain:
        return None
    return LyricTrack(
        title=(data.get("trackName") or "").strip(),
        artist=(data.get("artistName") or "").strip(),
        synced=synced,
        plain=plain,
    )


def _query_get(title: str, artist: str, timeout: float) -> Optional[LyricTrack]:
    """精确查询：/api/get?track_name=&artist_name=（未命中抛 HTTPError 404）。"""
    params = urllib.parse.urlencode({"track_name": title, "artist_name": artist})
    data = _get_json(f"{LRCLIB_BASE}/api/get?{params}", timeout=timeout)
    return _track_from_payload(data) if isinstance(data, dict) else None


def _query_search(title: str, artist: str, timeout: float) -> Optional[LyricTrack]:
    """模糊查询：/api/search → 优先取歌名精确匹配且带同步歌词的首条。"""
    params = urllib.parse.urlencode({"track_name": title, "artist_name": artist})
    rows = _get_json(f"{LRCLIB_BASE}/api/search?{params}", timeout=timeout)
    if not isinstance(rows, list) or not rows:
        return None
    fallback: Optional[dict] = None
    target = title.casefold()
    for row in rows:
        if not isinstance(row, dict):
            continue
        name_match = (row.get("trackName") or "").strip().casefold() == target
        has_synced = bool((row.get("syncedLyrics") or "").strip())
        if name_match and has_synced:
            return _track_from_payload(row)
        if fallback is None and has_synced:
            fallback = row
    return _track_from_payload(fallback) if fallback is not None else None


def reset_cache() -> None:
    """清空查询缓存（测试用）。"""
    with _cache_lock:
        _cache.clear()


def fetch_lyrics(
    title: str,
    artist: str = "",
    timeout: float = FETCH_TIMEOUT,
) -> Optional[LyricTrack]:
    """按歌名+歌手拉取歌词（先精确后模糊，带进程内缓存）。

    优先返回带同步歌词的结果；任何失败返回 None（best-effort）。
    """
    key = ((title or "").casefold(), (artist or "").casefold())
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    track: Optional[LyricTrack] = None
    for query in (_query_get, _query_search):
        try:
            track = query(title, artist, timeout)
        except Exception:
            continue  # 网络不可用/超时/404：尝试下一种查询方式
        if track is not None and track.synced:
            break  # 拿到同步歌词即收手
    # 精确命中但只有纯文本时也接受（调用方据 synced 判断能否生成字幕行）
    with _cache_lock:
        _cache[key] = track
    return track


# ---------------- LRC 解析 ----------------

def parse_lrc(text: str) -> list[tuple[float, str]]:
    """LRC 文本 → [(秒, 歌词行)]（升序）。

    - 支持一行多时间戳（[00:12.00][01:15.00]副歌）展开为多行
    - 支持 [offset:±ms]：正值使歌词整体提前（显示时间 = t - offset）
    - 跳过元信息行（[ti:] 等）与纯时间戳空行（间奏标记）
    """
    offset_ms = 0.0
    pairs: list[tuple[float, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if (m := _LRC_OFFSET_RE.match(line)) is not None:
            offset_ms = float(m.group(1))
            continue
        stamps: list[float] = []
        rest = line
        while (m := _LRC_TS_RE.match(rest)) is not None:
            mm, ss, frac = m.groups()
            t = int(mm) * 60 + int(ss)
            if frac:
                t += int(frac) / (10 ** len(frac))
            stamps.append(t)
            rest = rest[m.end():]
        content = rest.strip()
        if not stamps or not content:
            continue  # 元信息行 / 间奏空行
        for t in stamps:
            pairs.append((t - offset_ms / 1000.0, content))
    pairs.sort(key=lambda x: x[0])
    return pairs


# ---------------- 歌词 → 字幕行 ----------------

def distribute_words(text: str, start: float, end: float) -> list[SubtitleWord]:
    """单行歌词 → 逐词均匀分布的 SubtitleWord（按 token 字数加权）。

    词级时间为插值近似（供 ASS 逐字卡拉OK使用）；行级时间精确。
    """
    tokens = tokenize(text)
    if not tokens:
        return []
    weights = [max(1, len(tok.align)) for tok in tokens]
    total = sum(weights)
    span = max(MIN_WORD_DURATION * total, end - start)
    words: list[SubtitleWord] = []
    cur = start
    for tok, w in zip(tokens, weights):
        dur = span * w / total
        words.append(SubtitleWord(text=tok.display, start=cur, end=cur + dur))
        cur += dur
    words[-1].end = start + span
    return words


def build_lyric_lines(
    lrc_text: str,
    block_offset: float,
    until: Optional[float] = None,
) -> list[SubtitleLine]:
    """同步歌词 → 字幕行列表（叠加演唱块起点偏移）。

    - block_offset：演唱块在视频时间轴上的起点，对应歌曲 0 秒的近似
    - until：视频时间轴上的截断点（演唱块结束 + 容差），之后的歌词行
      视为未演唱而丢弃；None 表示不截断
    - 行结束时间 = 下一行开始；末行（及无后续行）用默认时长
    """
    pairs = parse_lrc(lrc_text)
    lines: list[SubtitleLine] = []
    for i, (t, text) in enumerate(pairs):
        start = block_offset + max(0.0, t)
        if until is not None and start > until:
            break
        if i + 1 < len(pairs):
            end = block_offset + max(0.0, pairs[i + 1][0])
        else:
            end = start + DEFAULT_LINE_SPAN
        if until is not None:
            end = min(end, until)
        if end <= start:
            end = start + MIN_WORD_DURATION
        words = distribute_words(text, start, end)
        if words:
            lines.append(SubtitleLine(words=words))
    return lines
