"""歌曲处理：演唱段检测 → 识曲/歌词检索 → 强制对齐 → 卡拉OK字幕行。

为什么需要这一层（实测结论）：
- 直播唱歌多为**翻唱**：Shazam 对翻唱/清唱识别率极低
  （本项目实测 3 个片段 0 命中），因此不能把识曲当作唯一入口；
- whisper 对歌声的转写**文本错误率高**（同段 CER 明显高于说话），
  但**词级时间相当准**（与强制对齐结果相差 0.1~0.3s）；
- 因此正确策略是：**先用低成本手段确定「唱的是哪首歌」，
  再用歌词文本 + 强制对齐生成字幕**，而不是让 ASR 直接产出歌词。

三级检索（成本递增，命中即停）：
1. 本地歌词库：用 ASR 文本片段与库内歌词做相似度匹配（离线、瞬时）；
2. 在线歌词库：用 ASR 文本片段检索 LRCLIB / NetEase（命中率最高）；
3. 听歌识曲：Shazam（仅原唱/热门曲目有效，作为补充）。

拿到歌词后：
- 有同步歌词（LRC）→ 直接用其行时间 + 演唱块偏移做初对齐；
- 无同步歌词 → 用 FunASR 时间戳预测模型对「音频 + 歌词文本」做强制对齐；
- 两条路都失败 → 保留 ASR 转写（不丢内容）。
"""
from __future__ import annotations

import json
import re
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable, Optional

from core.models import SubtitleLine, SubtitleWord

# ---------------- 数据结构 ----------------


@dataclass
class SongSegment:
    """一个演唱块（时间区间）及其识别结果。"""

    start: float
    end: float
    title: str = ""
    artist: str = ""
    lyrics_lrc: Optional[str] = None
    plain_lyrics: Optional[str] = None
    source: str = ""              # local / lrclib / netease / shazam
    score: float = 0.0            # 匹配得分 0~1
    lines: list[SubtitleLine] = field(default_factory=list)
    aligned: bool = False         # 是否已强制对齐（逐字时间可信）

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SongTuning:
    """演唱检测与检索参数。"""

    min_block_seconds: float = 25.0
    block_gap: float = 12.0
    min_segment_seconds: float = 35.0    # 短于此的块不检索（不足以判定）
    accept_score: float = 0.55
    search_window: float = 240.0
    probe_positions: tuple = (0.15, 0.45, 0.75)


# ---------------- 演唱检测 ----------------

def detect_song_blocks(
    segments: list[tuple[float, float]],
    tuning: SongTuning | None = None,
    is_singing: Optional[Callable[[float, float], bool]] = None,
) -> list[tuple[float, float]]:
    """语音段 → 候选演唱块。

    is_singing 可为外部判定（声纹/音高）；none 时退化为「按时长合并」。
    合并后块需要 ≥ min_block_seconds（过滤说话中的长段落噪声）。
    """
    tuning = tuning or SongTuning()
    if not segments:
        return []
    spans = sorted(segments)
    merged: list[list[float]] = [list(spans[0])]
    for s, e in spans[1:]:
        if s - merged[-1][1] <= tuning.block_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    blocks = [(s, e) for s, e in merged if e - s >= tuning.min_block_seconds]
    if is_singing is not None:
        blocks = _split_by_pitch(blocks, is_singing, tuning)
    return blocks


def _split_by_pitch(blocks, is_singing, tuning):
    """按外部「是否演唱」判定把长块切成演唱/说话子段。"""
    out: list[tuple[float, float]] = []
    for s, e in blocks:
        cur_start = None
        t = s
        while t < e:
            step = min(4.0, e - t)
            singing = is_singing(t, t + step)
            if singing and cur_start is None:
                cur_start = t
            elif not singing and cur_start is not None:
                if t - cur_start >= tuning.min_block_seconds:
                    out.append((cur_start, t))
                cur_start = None
            t += step
        if cur_start is not None and e - cur_start >= tuning.min_block_seconds:
            out.append((cur_start, e))
    return out


def merge_nearby(blocks: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    """相邻块（间隔 ≤ gap）合并为同一首（间奏不切歌）。"""
    if not blocks:
        return []
    out: list[list[float]] = [list(sorted(blocks)[0])]
    for s, e in sorted(blocks)[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


# ---------------- 歌词文本工具 ----------------

_LRC_TS = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]")
_LRC_META = re.compile(r"^\[(ti|ar|al|by|offset|re|ve|length|au):.*\]\s*$", re.I)
# 歌词文件头部的创作署名行：词/曲/编曲/制作人/演唱：XXX
_CREDIT_RE = re.compile(
    r"^\s*(作词|作曲|编曲|制作人|演唱|原唱|词|曲|OP|SP|出品|监制|统筹|混音|母带)"
    r"\s*[:：\-]\s*.{0,40}$", re.I)


def lrc_rows(text: str) -> list[tuple[float, str]]:
    """LRC → [(秒, 文本)]（展开一行多时间戳、跳过元信息与空行）。"""
    rows: list[tuple[float, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or _LRC_META.match(line):
            continue
        stamps = []
        rest = line
        while (m := _LRC_TS.match(rest)):
            mm, ss, frac = m.groups()
            t = int(mm) * 60 + int(ss)
            if frac:
                t += int(frac) / 10 ** len(frac)
            stamps.append(t)
            rest = rest[m.end():]
        content = rest.strip()
        if stamps and content:
            for t in stamps:
                rows.append((t, content))
    rows.sort(key=lambda r: r[0])
    return rows


def plain_rows(text: str, synced: bool = False) -> list[str]:
    """歌词文本 → 纯文本行（去 LRC 时间戳/元信息/创作署名/空行）。"""
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if synced or _LRC_TS.match(line):
            line = _LRC_TS.sub("", line).strip()
        if not line or _LRC_META.match(line):
            continue
        # 创作署名行（"词 : 张三" / "作曲：李四"）不是歌词
        if _CREDIT_RE.match(line):
            continue
        out.append(line)
    return out


def norm_text(text: str) -> str:
    """归一化（繁→简、去标点空白、小写）——检索与打分的统一口径。"""
    try:
        from core.text import t2s

        text = t2s(text)
    except Exception:
        pass
    return "".join(c for c in text if c.isalnum()).lower()


def score_similarity(a: str, b: str) -> float:
    """两段文本的相似度（0~1）：字符 LCS 覆盖率与序列相似度的折中。

    歌词检索里，ASR 文本常有错字，单纯 SequenceMatcher.ratio() 对长文本
    偏保守；用「ASR 字符被歌词覆盖的比例」更能反映「这段唱的是不是它」。
    """
    na, nb = norm_text(a), norm_text(b)
    if not na or not nb:
        return 0.0
    if na in nb or nb in na:
        return 1.0
    sm = SequenceMatcher(None, na, nb, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    cover = matched / len(na)
    ratio = sm.ratio()
    return round(max(cover, ratio), 4)


def rank_lyrics(
    transcript: str,
    candidates: Iterable[tuple[str, str, str]],
    *,
    top: int = 5,
) -> list[tuple[float, str, str, str]]:
    """用 ASR 文本给候选歌词打分。

    candidates: [(标题, 歌手, 歌词全文)]；返回 [(得分, 标题, 歌手, 全文)]。
    只在候选的某个窗口内比对（歌词通常比演唱片段长）。
    """
    query = norm_text(transcript)
    if not query:
        return []
    out = []
    for title, artist, lyrics in candidates:
        body = norm_text(lyrics)
        if not body:
            continue
        score = score_similarity(query, body)
        out.append((score, title, artist, lyrics))
    out.sort(key=lambda x: (-x[0], x[1]))
    return out[:top]


# ---------------- 在线歌词源 ----------------

_UA = "SubtitleStudio/2.0 (subtitle-studio; local tool)"


def _http_json(url: str, timeout: float = 15.0):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


class LyricProvider:
    """在线歌词源接口（可扩展：子类实现 search）。"""

    name = "base"

    def search(self, title: str = "", artist: str = "", query: str = "",
               limit: int = 10) -> list[tuple[str, str, str, str]]:
        """返回 [(标题, 歌手, LRC 或纯文本, 是否同步)]。"""
        raise NotImplementedError


class LrclibProvider(LyricProvider):
    """LRCLIB（免费、无需 key，返回结构化同步歌词）。"""

    name = "lrclib"
    BASE = "https://lrclib.net/api"

    def search(self, title="", artist="", query="", limit=10):
        params = {}
        if title:
            params["track_name"] = title
            if artist:
                params["artist_name"] = artist
        elif query:
            params["q"] = query
        else:
            return []
        try:
            rows = _http_json(f"{self.BASE}/search?{urllib.parse.urlencode(params)}")
        except Exception:
            return []
        out = []
        for row in (rows or [])[:limit]:
            if not isinstance(row, dict):
                continue
            body = (row.get("syncedLyrics") or "").strip() or (
                row.get("plainLyrics") or "").strip()
            if not body:
                continue
            out.append((row.get("trackName") or title,
                        row.get("artistName") or artist,
                        body,
                        bool((row.get("syncedLyrics") or "").strip())))
        return out


class NeteaseProvider(LyricProvider):
    """NetEase 云音乐（中文曲库覆盖最好，返回 LRC）。"""

    name = "netease"
    SEARCH = "https://music.163.com/api/search/get"
    LYRIC = "https://music.163.com/api/song/lyric"

    def search(self, title="", artist="", query="", limit=10):
        q = query or " ".join(x for x in (title, artist) if x)
        if not q:
            return []
        try:
            data = _http_json(
                f"{self.SEARCH}?s={urllib.parse.quote(q)}&type=1&limit={max(1, limit)}")
        except Exception:
            return []
        songs = ((data or {}).get("result") or {}).get("songs") or []
        out = []
        for song in songs[:limit]:
            sid = song.get("id")
            if not sid:
                continue
            try:
                lrc = _http_json(f"{self.LYRIC}?id={sid}&lv=-1&kv=-1&tv=-1")
            except Exception:
                continue
            body = (((lrc or {}).get("lrc") or {}).get("lyric") or "").strip()
            if not body:
                continue
            names = "、".join(a.get("name", "") for a in song.get("artists") or [])
            out.append((song.get("name") or title, names or artist, body, True))
        return out


class KugouProvider(LyricProvider):
    """酷狗音乐（国内曲库，翻唱/冷门歌覆盖较好）。

    两步：搜索拿 hash/duration → krcs 搜索拿歌词 id/accesskey →
    download 取 base64 的 LRC。
    """

    name = "kugou"
    SEARCH = "http://mobilecdn.kugou.com/api/v3/search/song"
    KRCS = "https://krcs.kugou.com/search"
    DOWNLOAD = "https://lyrics.kugou.com/download"

    def search(self, title="", artist="", query="", limit=10):
        q = " ".join(x for x in (title, artist) if x) or query
        if not q:
            return []
        ctx = _ssl_ctx()
        try:
            raw = _http_raw(
                f"{self.SEARCH}?keyword={urllib.parse.quote(q)}"
                f"&page=1&pagesize={max(1, min(limit, 10))}",
                ctx=ctx, referer="https://www.kugou.com/")
            data = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            return []
        songs = ((data or {}).get("data") or {}).get("info") or []
        out: list[tuple[str, str, str, bool]] = []
        for song in songs[:limit]:
            name = song.get("songname") or title
            singer = song.get("singername") or artist
            hsh, dur = song.get("hash"), song.get("duration")
            if not hsh:
                continue
            body = self._fetch(name, singer, hsh, dur, ctx)
            if body:
                out.append((name, singer, body, True))
        return out

    def _fetch(self, name: str, singer: str, hsh: str, dur, ctx) -> str:
        # krcs 对「歌手 - 歌名」的查询命中率明显更高（实测用纯歌名返回空）
        keywords = f"{singer} - {name}" if singer else name
        try:
            raw = _http_raw(
                f"{self.KRCS}?ver=1&man=yes&client=mobi"
                f"&keyword={urllib.parse.quote(keywords)}"
                f"&duration={dur or ''}&hash={hsh}",
                ctx=ctx, referer="https://www.kugou.com/")
            data = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            return ""
        cands = data.get("candidates") or []
        if not cands:
            return ""
        # 依次尝试前几个候选：个别候选的 accesskey 会过期/无版权，
        # 单看第一个容易「明明有歌词却拿不到」（实测存在这种失败）。
        for cand in cands[:4]:
            body = self._download(cand, ctx)
            if body:
                return body
        return ""

    def _download(self, cand: dict, ctx) -> str:
        try:
            raw = _http_raw(
                f"{self.DOWNLOAD}?ver=1&client=pc&id={cand['id']}"
                f"&accesskey={cand['accesskey']}&fmt=lrc&charset=utf8",
                ctx=ctx, referer="https://www.kugou.com/")
            payload = json.loads(raw.decode("utf-8", "replace"))
            if str(payload.get("status")) != "200":
                return ""
            import base64

            body = base64.b64decode(payload.get("content") or "").decode(
                "utf-8", "replace").strip()
            # 只接受含时间戳的同步歌词
            return body if _LRC_TS.search(body) else ""
        except Exception:
            return ""


class CustomLyricProvider(LyricProvider):
    """用户自建歌词服务（预设接口，对应「允许自定义接口」需求）。

    契约：`GET <endpoint>?title=..&artist=..&q=..` 返回
        {"lyrics": "LRC 文本", "title": "...", "artist": "...", "synced": true}
    或直接返回 LRC 纯文本（根据是否存在 `[mm:ss]` 自动判断 synced）。
    """

    name = "custom"

    def __init__(self, endpoint: str = "", api_key: str = "", timeout: float = 15.0):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout

    def search(self, title="", artist="", query="", limit=10):
        if not self.endpoint:
            return []
        params = urllib.parse.urlencode(
            {"title": title, "artist": artist, "q": query or title})
        sep = "&" if "?" in self.endpoint else "?"
        headers = {"User-Agent": _UA}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            req = urllib.request.Request(f"{self.endpoint}{sep}{params}",
                                         headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                text = resp.read().decode("utf-8", "replace")
        except Exception:
            return []
        body = text
        rtitle, rartist = title, artist
        synced = bool(_LRC_TS.search(text))
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                body = data.get("lyrics") or data.get("lrc") or ""
                rtitle = data.get("title") or title
                rartist = data.get("artist") or artist
                synced = bool(data.get("synced", synced))
        except Exception:
            pass
        body = (body or "").strip()
        return [(rtitle, rartist, body, synced)] if body else []


def _ssl_ctx():
    """部分音乐站点证书链不全（酷狗），这里放宽校验（仅下载公开歌词）。"""
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _http_raw(url: str, *, ctx=None, referer: str = "") -> bytes:
    headers = {"User-Agent": _UA}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
        return resp.read()


PROVIDERS: dict[str, LyricProvider] = {
    "lrclib": LrclibProvider(),
    "netease": NeteaseProvider(),
    "kugou": KugouProvider(),
}

# 默认检索顺序：先国内音源（中文曲库覆盖最好），再 LRCLIB（版本校对质量高）。
# 用户可在设置里改顺序或增删（对应「允许自定义或内置多个接口」）。
DEFAULT_PROVIDER_ORDER = ("netease", "kugou", "lrclib")

_CUSTOM: dict[str, LyricProvider] = {}


def register_lyric_provider(name: str, provider: LyricProvider) -> None:
    """注册自定义歌词源（插件或 UI 里配置的自建服务）。"""
    PROVIDERS[name] = provider
    _CUSTOM[name] = provider


def set_custom_endpoint(endpoint: str, api_key: str = "",
                        timeout: float = 15.0) -> str:
    """配置自建歌词服务（返回注册用的名称）。"""
    register_lyric_provider("custom",
                            CustomLyricProvider(endpoint, api_key, timeout))
    return "custom"


def available_lyric_providers() -> list[str]:
    return sorted(PROVIDERS)


def search_lyrics(
    title: str = "",
    artist: str = "",
    transcript: str = "",
    *,
    providers: Iterable[str] = DEFAULT_PROVIDER_ORDER,
    limit: int = 8,
    min_score: float = 0.45,
    min_sources: int = 1,
    on_log: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """多源歌词检索 + 版本打分 + **自动换源**。

    行为（对应「匹配不到时自动替换」）：
    1. 按 `providers` 顺序逐个源查询，**每个源都查完**（不是命中即停），
       这样才能按分数横向比较不同源的版本；
    2. 对每个候选版本用转写文本打「行序一致性」分（见 layout_score）；
    3. 若最佳得分低于 min_score，自动继续尝试后续源；
    4. 若所有内置源都不达标且配置了自建服务，最后尝试自建服务。

    返回 [{"title","artist","lyrics","synced","source","score"}]，
    按 score 降序。无网络/无命中返回空列表（调用方回退 ASR）。
    """
    query = _query_from_transcript(transcript)
    order = list(providers) or list(DEFAULT_PROVIDER_ORDER)
    if "custom" in _CUSTOM and "custom" not in order:
        order.append("custom")

    results: list[dict] = []
    for name in order:
        provider = PROVIDERS.get(name)
        if provider is None:
            if on_log:
                on_log(f"歌词源 {name} 未注册，跳过")
            continue
        try:
            rows = provider.search(title=title, artist=artist, query=query,
                                   limit=limit)
        except Exception as exc:  # noqa: BLE001 - 单源失败不影响其它源
            if on_log:
                on_log(f"歌词源 {name} 查询失败：{exc}")
            continue
        if on_log:
            on_log(f"歌词源 {name}：返回 {len(rows)} 条候选")
        for rtitle, rartist, body, synced in rows:
            # 版本判别：行序一致的一版得分最高（见 layout_score 说明）
            score = (layout_score(transcript, body) if transcript else 0.0)
            # 歌手偏好：同一首歌常有多条翻唱条目（歌词完全相同、分数并列），
            # 若查询带歌手名，与原唱歌手匹配者优先（避免把翻唱署名当原唱）。
            if artist and rartist:
                if norm_text(artist) in norm_text(rartist) or \
                        norm_text(rartist) in norm_text(artist):
                    score += 0.05
                elif _artist_mismatch(artist, rartist):
                    score -= 0.05
            results.append({
                "title": rtitle, "artist": rartist, "lyrics": body,
                "synced": synced, "source": provider.name, "score": score,
            })
        # 已经拿到足够好的结果就不再往下试（省时间也避免引入错版本）；
        # 但若已知歌手且当前最佳条目的歌手明显不符，继续换源找原版。
        if results and transcript:
            best_now = max(results, key=lambda r: r["score"])
            good = best_now["score"] >= max(min_score, 0.80)
            artist_ok = (not artist or not best_now["artist"]
                         or not _artist_mismatch(artist, best_now["artist"]))
            if good and artist_ok:
                if on_log:
                    on_log(f"已在 {name} 找到高置信版本"
                           f"（{best_now['title']}/{best_now['artist']}，"
                           f"{best_now['score']:.2f}），停止换源")
                break
            if good and not artist_ok and on_log:
                on_log(f"{name} 的最佳版本歌手为「{best_now['artist']}」，"
                       f"与「{artist}」不符，继续尝试其它源")

    # 排序键：得分 → 是否同步歌词 → 来源优先级
    _SRC_ORDER = {"lrclib": 0, "netease": 1, "kugou": 2, "custom": 3}
    results.sort(key=lambda r: (-r["score"], not r["synced"],
                                _SRC_ORDER.get(r["source"], 9)))
    kept = [r for r in results if r["score"] >= min_score or not transcript]
    if on_log:
        if kept:
            best = kept[0]
            on_log(f"采用歌词：{best['title']} / {best['artist']}"
                   f"（{best['source']}，得分 {best['score']:.2f}）")
        else:
            on_log(f"所有歌词源均未达标（阈值 {min_score:.2f}），保留语音识别")
    return kept


def _artist_mismatch(wanted: str, got: str) -> bool:
    """两个歌手名是否明显不是同一人（用于翻唱条目的降权）。

    只在两边都非空、且完全不互相包含时判为不匹配；不引入拼音/别名表，
    避免把「周杰伦」与「Jay Chou」这类别名误判（宁可不降权）。
    """
    a, b = norm_text(wanted), norm_text(got)
    if not a or not b:
        return False
    return a not in b and b not in a


def best_window_score(transcript: str, lyrics: str, window: int = 14) -> float:
    """转写文本 ↔ 歌词文本的最大连续覆盖得分（0~1）。

    逐行从每个起点取 window 行拼接，算转写有多少字被命中，取最大值。
    这是判断「这个歌词版本与本次演唱是否同一版」的可靠指标：
    版本不同时，行序与措辞都会变，得分显著下降。
    """
    from difflib import SequenceMatcher

    body = norm_text(transcript)
    lines = plain_rows(lyrics, synced=_LRC_TS.search(lyrics) is not None)
    if not body or not lines:
        return 0.0
    normed = [norm_text(x) for x in lines]
    best = 0.0
    step = max(1, window // 2)
    for i in range(0, len(normed), step):
        text = "".join(normed[i:i + window])
        if not text:
            continue
        sm = SequenceMatcher(None, body, text, autojunk=False)
        matched = sum(b.size for b in sm.get_matching_blocks())
        # 惩罚冗余窗口（歌词比转写长太多会导致虚高）
        score = matched / max(1, len(body)) - 0.03 * max(0, len(text) - 2 * len(body)) / 100
        best = max(best, score)
    return round(best, 4)


def layout_score(transcript: str, lyrics: str, window: int = 14) -> float:
    """版本判别得分：转写文本与歌词的**连续子串匹配**长度占比。

    与 best_window_score 的区别：这里要求匹配在转写中**连续成段**且
    顺序一致。不同编曲/现场版的歌词行序不同，连续匹配长度会显著变短，
    而任意版本都能拿到较高的「字符覆盖率」——因此用连续段长度做判别，
    能把「行序完全一致的那一版」挑出来。
    """
    from difflib import SequenceMatcher

    body = norm_text(transcript)
    lines = plain_rows(lyrics, synced=_LRC_TS.search(lyrics) is not None)
    if not body or not lines:
        return 0.0
    normed = [norm_text(x) for x in lines]
    best = 0.0
    step = max(1, window // 2)
    for i in range(0, len(normed), step):
        text = "".join(normed[i:i + window])
        if not text:
            continue
        sm = SequenceMatcher(None, body, text, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size >= 4]
        if not blocks:
            continue
        # 连续度：最长匹配块占比 + 匹配块数惩罚（碎片化说明行序不对）
        longest = max(b.size for b in blocks)
        coverage = sum(b.size for b in blocks) / max(1, len(body))
        score = 0.6 * coverage + 0.4 * (longest / max(1, len(body)))
        score -= 0.01 * max(0, len(blocks) - 3)
        best = max(best, score)
    return round(best, 4)


def _query_from_transcript(transcript: str, max_chars: int = 24) -> str:
    """从转写文本里挑一段最像歌词的连续片段（去掉语气词/口播）。"""
    if not transcript:
        return ""
    cleaned = re.sub(r"[嗯啊哦唉呀吧呢嘛]+", "", transcript)
    pieces = [p for p in re.split(r"[，。！？、,.!?\s]+", cleaned) if len(p) >= 6]
    if pieces:
        return max(pieces, key=len)[:max_chars]
    return cleaned[:max_chars]


# ---------------- 强制对齐 ----------------

_ALIGN_CACHE: dict[str, object] = {}
_ALIGN_LOCK = threading.Lock()


def _align_model(model_id: str, device: str):
    key = f"{model_id}|{device}"
    with _ALIGN_LOCK:
        if key in _ALIGN_CACHE:
            return _ALIGN_CACHE[key]
    try:
        from funasr import AutoModel
    except ImportError:
        return None
    try:
        model = AutoModel(model=model_id, device=device, disable_update=True)
    except Exception:
        return None
    with _ALIGN_LOCK:
        _ALIGN_CACHE[key] = model
    return model


def forced_align_text(
    wav_path: str,
    text: str,
    *,
    offset: float = 0.0,
    model_id: str = "iic/speech_timestamp_prediction-v1-16k-offline",
    device: str = "cuda",
) -> list[tuple[float, float, str]]:
    """音频 + 已知文本 → 逐字时间（秒，含 offset）。

    返回 [(start, end, char)]；模型不可用/失败返回空列表。
    """
    model = _align_model(model_id, device)
    if model is None or not text:
        return []
    body = norm_text(text)
    if not body:
        return []
    try:
        result = model.generate(
            input=[[wav_path], [body]], data_type=["sound", "text"],
            batch_size_s=120)
        stamps = result[0]["timestamp"]
    except Exception:
        return []
    out: list[tuple[float, float, str]] = []
    for (s, e), ch in zip(stamps, body):
        out.append((round(offset + s / 1000.0, 3), round(offset + e / 1000.0, 3), ch))
    return out


def lines_from_alignment(
    char_times: list[tuple[float, float, str]],
    lyric_lines: list[str],
) -> list[SubtitleLine]:
    """逐字时间 + 歌词行 → SubtitleLine（行边界按歌词行字符数切分）。"""
    lines: list[SubtitleLine] = []
    pos = 0
    for text in lyric_lines:
        n = len(norm_text(text))
        if n <= 0:
            continue
        chunk = char_times[pos:pos + n]
        pos += n
        if not chunk:
            break
        words = [SubtitleWord(text=ch, start=s, end=e) for s, e, ch in chunk]
        lines.append(SubtitleLine(words=words))
    return lines


__all__ = [
    "SongSegment", "SongTuning", "detect_song_blocks", "merge_nearby",
    "lrc_rows", "plain_rows", "norm_text", "score_similarity", "rank_lyrics",
    "LyricProvider", "LrclibProvider", "NeteaseProvider", "KugouProvider",
    "CustomLyricProvider", "search_lyrics", "register_lyric_provider",
    "set_custom_endpoint", "available_lyric_providers",
    "DEFAULT_PROVIDER_ORDER", "layout_score", "best_window_score",
    "forced_align_text", "lines_from_alignment",
]
