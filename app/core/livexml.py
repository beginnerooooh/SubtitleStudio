"""直播录制 XML 解析：弹幕 / 礼物 / SC / 舰长 / 发送者 ID。

支持两种实际格式：

1. **BililiveRecorder（录播姬）2.x** —— 本项目实测样本：
   ```xml
   <d p="6.081,1,25,14893055,1791374981019,0,234112129,0" user="昵称">正文</d>
   <gift ts="39.943" user="昵称" uid="1750602373" giftname="粉丝团灯牌" giftcount="1" />
   <sc ts="211.502" user="昵称" uid="379489989" price="50" time="120">正文</sc>
   <guard ts="112.018" user="昵称" uid="..." level="3" count="1" />
   <BililiveRecorderRecordInfo roomid="..." name="主播名" title="标题"
       start_time="2026-10-07T20:09:36..." />
   ```
   注意：`p` 的**第 8 段（index 7）是恒为 0 的占位**，昵称在 `user` 属性里；
   早期实现误把 index 7 当昵称，导致所有用户名变成 "0"。

2. **mikufans 主站弹幕 XML**：`<d p="时间,模式,字号,颜色,时间戳,池,用户ID,昵称哈希,...">`

这些数据对字幕生成的价值（本项目的实际用法见 pipeline）：

1. **热词**：高频昵称/礼物名/游戏术语 → ASR 热词（显著降低专名错误）；
2. **断句锚点**：礼物/SC/上舰事件时间点，主播通常随即口头致谢；
3. **歌曲线索**：SC/弹幕里的「想听《xxx》」「点歌 xxx」是**免费且准确**的
   歌曲候选（比 Shazam 对翻唱有效得多）；
4. **人声活动旁证**：长时间无弹幕 + 低能量 → 更可能是 BGM/间奏。
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# 礼物名/舰长/醒目留言等常见关键词（用于从弹幕里挑出点名事件）
_GIFT_WORDS = (
    "盲盒", "礼物", "舰长", "提督", "总督", "灯牌", "粉丝团", "醒目留言",
    "SC", "sc", "充电", "投喂", "小心心", "辣条", "火箭", "飞机", "鲜花",
    "点赞", "收藏", "分享", "关注", "打卡", "人气票", "星星之火", "薯条",
    "送花花", "心动卡", "泡泡机",
)
# 点歌/歌曲相关线索（门禁词：命中其一才去解析歌名）
# 注意必须与 _SONG_RE 的句式集合保持一致，否则会出现
# 「正则能解析、但门禁先拦掉」的漏检（实测踩过：「点一首X」「来首X」）。
_SONG_HINTS = ("点歌", "想听", "唱一", "唱首", "来一首", "来首", "歌名",
               "原唱", "能不能唱", "可以唱", "求唱", "点一首", "点首",
               "唱一首", "来首")


@dataclass
class Danmaku:
    """一条弹幕。"""

    time: float              # 相对录制开始的秒数
    text: str
    user_id: str = ""
    user_name: str = ""
    mode: int = 1
    color: int = 0
    raw_time: float = 0.0    # 原始发送时间戳（unix 毫秒，若有）


@dataclass
class GiftEvent:
    """礼物 / 上舰 / 醒目留言事件。"""

    time: float
    user_name: str
    gift: str
    count: int = 1
    price: float = 0.0
    user_id: str = ""
    kind: str = "gift"       # gift | guard | sc
    text: str = ""           # SC 附言


@dataclass
class LiveLog:
    """一次直播的完整旁路信息。"""

    danmaku: list[Danmaku] = field(default_factory=list)
    gifts: list[GiftEvent] = field(default_factory=list)
    title: str = ""
    streamer: str = ""
    room_id: str = ""
    started_at: float = 0.0     # unix 秒（录制起点，若能推断）
    duration: float = 0.0       # 由事件时间推断的录制时长（秒）

    # ---------------- 派生视图 ----------------

    def event_times(self, window: float = 0.5) -> list[float]:
        """礼物/SC/上舰时间点（供强制断句用，已按 window 合并接近点）。"""
        times = sorted(g.time for g in self.gifts)
        merged: list[float] = []
        for t in times:
            if not merged or t - merged[-1] > window:
                merged.append(t)
        return merged

    def hotwords(self, top: int = 60, min_count: int = 2) -> list[str]:
        """高频专名候选：昵称 + 礼物名 + 弹幕高频 2~8 字词。

        只做「轻量候选」：去停用词、去纯数字、去表情码（`[xx]`）。
        whisper 的 initial_prompt / funasr 的 hotword 都接受空格分隔串。
        """
        from collections import Counter

        counter: Counter[str] = Counter()
        for d in self.danmaku:
            if d.user_name:
                counter[d.user_name.strip()] += 3
            for w in _extract_terms(d.text):
                counter[w] += 1
        for g in self.gifts:
            if g.gift:
                counter[g.gift] += 2
            if g.user_name:
                counter[g.user_name] += 2
        blocked = _STOPWORDS
        ranked = [
            (w, c) for w, c in counter.most_common(top * 4)
            if c >= min_count and w not in blocked and not w.isdigit()
            and 1 < len(w) <= 12
        ]
        return [w for w, _c in ranked[:top]]

    def gift_thanks_windows(self) -> list[tuple[float, str, str]]:
        """礼物致谢窗口：[(时间, 用户, 礼物)]——主播通常在此后 1~4 秒内致谢。"""
        return [(g.time, g.user_name, g.gift) for g in self.gifts]

    def song_requests(self) -> list[tuple[float, str, str]]:
        """从弹幕/SC 里提取点歌请求：[(时间, 用户, 歌名或原话)]。

        这是**比识曲更可靠**的歌曲来源：SC 与点歌弹幕直接写了歌名。
        """
        out: list[tuple[float, str, str]] = []
        for d in self.danmaku:
            name = _song_name_from_text(d.text)
            if name:
                out.append((d.time, d.user_name, name))
        for g in self.gifts:
            if g.kind == "sc" and g.text:
                name = _song_name_from_text(g.text)
                if name:
                    out.append((g.time, g.user_name, name))
        return sorted(out)

    def density(self, window: float = 30.0) -> list[float]:
        """按窗口统计弹幕密度（旁证「是否在互动/唱歌」）。"""
        if self.duration <= 0 or window <= 0:
            return []
        n = int(self.duration / window) + 1
        bins = [0.0] * n
        for d in self.danmaku:
            i = int(d.time / window)
            if 0 <= i < n:
                bins[i] += 1
        return bins


# ---------------- 解析 ----------------

def parse_xml(path: str | Path) -> LiveLog:
    """解析 B 站录制弹幕 XML（BililiveRecorder 2.x / mikufans 主站格式）。"""
    p = Path(path)
    text = _read_text(p)
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        # 录制中断会产生截断 XML：抽出完整 <d>/<gift>/<sc>/<guard> 再包一层
        root = ET.fromstring(_repair_truncated(text))
    log = LiveLog()
    _read_record_info(root, log)
    for node in root.iter():
        tag = node.tag.split("}")[-1]
        if tag == "d":
            _parse_d(node, log)
        elif tag == "gift":
            _parse_gift(node, log)
        elif tag == "sc":
            _parse_sc(node, log)
        elif tag == "guard":
            _parse_guard(node, log)
    log.danmaku.sort(key=lambda d: d.time)
    log.gifts.sort(key=lambda g: g.time)
    _finalize(log)
    return log


def parse_jsonl(path: str | Path) -> LiveLog:
    """解析部分录制工具输出的 jsonl 事件流（字段名容错）。"""
    import json

    log = LiveLog()
    for line in _read_text(Path(path)).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = str(row.get("type") or row.get("cmd") or "").upper()
        t = _safe_float(row.get("time") or row.get("timestamp") or row.get("ts"))
        text = str(row.get("text") or row.get("content") or row.get("msg") or "")
        user = _user_of(row)
        if "GIFT" in kind or "SUPER" in kind or "GUARD" in kind:
            log.gifts.append(GiftEvent(
                time=round(t, 3), user_name=user,
                gift=str(row.get("gift") or row.get("gift_name") or text),
                count=int(_safe_float(row.get("count") or 1) or 1),
                price=_safe_float(row.get("price")),
                user_id=str(row.get("uid") or ""),
                kind="sc" if "SUPER" in kind else "gift",
                text=text))
        elif text:
            log.danmaku.append(Danmaku(
                time=round(t, 3), text=text, user_name=user,
                user_id=str(row.get("uid") or "")))
    log.danmaku.sort(key=lambda d: d.time)
    log.gifts.sort(key=lambda g: g.time)
    _finalize(log)
    return log


def load_log(path: str | Path) -> Optional[LiveLog]:
    """按扩展名自动解析；不支持/失败返回 None（调用方降级）。"""
    p = Path(path)
    try:
        if p.suffix.lower() == ".xml":
            return parse_xml(p)
        if p.suffix.lower() in (".jsonl", ".json"):
            return parse_jsonl(p)
    except Exception:
        return None
    return None


def merge_logs(*logs: LiveLog) -> LiveLog:
    """合并多个来源（XML + JSONL），按时间排序去重。"""
    out = LiveLog()
    seen: set[tuple[float, str]] = set()
    for log in logs:
        if log is None:
            continue
        out.title = out.title or log.title
        out.streamer = out.streamer or log.streamer
        out.room_id = out.room_id or log.room_id
        for d in log.danmaku:
            key = (round(d.time, 1), d.text)
            if key in seen:
                continue
            seen.add(key)
            out.danmaku.append(d)
        out.gifts.extend(log.gifts)
    out.danmaku.sort(key=lambda d: d.time)
    out.gifts.sort(key=lambda g: g.time)
    _finalize(out)
    return out


# ---------------- 各节点解析 ----------------

def _parse_d(node, log: LiveLog) -> None:
    """`<d p="...">` 弹幕节点。"""
    text = (node.text or "").strip()
    if not text:
        return
    meta = (node.get("p") or "").split(",")
    user = (node.get("user") or "").strip()
    if not user and len(meta) > 7:
        # 主站格式：index 7 是昵称（md5 哈希，不可读但可用作 ID）
        user = ""
    log.danmaku.append(Danmaku(
        time=round(_safe_float(meta[0] if meta else 0.0), 3),
        text=text,
        mode=int(_safe_float(meta[1] if len(meta) > 1 else 1)),
        color=int(_safe_float(meta[3] if len(meta) > 3 else 0)),
        raw_time=_safe_float(meta[4] if len(meta) > 4 else 0.0),
        user_id=(meta[6] if len(meta) > 6 else ""),
        user_name=user,
    ))


def _parse_gift(node, log: LiveLog) -> None:
    log.gifts.append(GiftEvent(
        time=round(_safe_float(node.get("ts")), 3),
        user_name=(node.get("user") or "").strip(),
        gift=(node.get("giftname") or "").strip(),
        count=int(_safe_float(node.get("giftcount") or 1) or 1),
        user_id=(node.get("uid") or "").strip(),
        kind="gift",
    ))


def _parse_sc(node, log: LiveLog) -> None:
    log.gifts.append(GiftEvent(
        time=round(_safe_float(node.get("ts")), 3),
        user_name=(node.get("user") or "").strip(),
        gift="醒目留言",
        price=_safe_float(node.get("price")),
        user_id=(node.get("uid") or "").strip(),
        kind="sc",
        text=(node.text or "").strip(),
    ))


def _parse_guard(node, log: LiveLog) -> None:
    level = int(_safe_float(node.get("level") or 3) or 3)
    names = {1: "总督", 2: "提督", 3: "舰长"}
    log.gifts.append(GiftEvent(
        time=round(_safe_float(node.get("ts")), 3),
        user_name=(node.get("user") or "").strip(),
        gift=names.get(level, "舰长"),
        count=int(_safe_float(node.get("count") or 1) or 1),
        user_id=(node.get("uid") or "").strip(),
        kind="guard",
    ))


def _read_record_info(root, log: LiveLog) -> None:
    """读取 `<BililiveRecorderRecordInfo>` 的标题/主播/房间号/起始时间。"""
    for node in root.iter():
        if node.tag.split("}")[-1] != "BililiveRecorderRecordInfo":
            continue
        log.streamer = (node.get("name") or "").strip()
        log.title = (node.get("title") or "").strip()
        log.room_id = (node.get("roomid") or "").strip()
        log.started_at = _parse_iso(node.get("start_time") or "")
        break


def _finalize(log: LiveLog) -> None:
    """推断录制时长；主播名缺失时用送礼最多的人兜底。"""
    times = [d.time for d in log.danmaku] + [g.time for g in log.gifts]
    log.duration = max(times) if times else 0.0
    if log.streamer:
        return
    from collections import Counter

    senders = Counter(d.user_name for d in log.danmaku if d.user_name)
    if senders:
        log.streamer = senders.most_common(1)[0][0]


def _parse_iso(text: str) -> float:
    """ISO 8601 时间 → unix 秒；失败返回 0。"""
    if not text:
        return 0.0
    try:
        from datetime import datetime

        # 兼容 7 位小数秒（.2003498）与 +08:00 时区
        cleaned = re.sub(r"(\.\d{6})\d+", r"\1", text)
        dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return 0.0


# ---------------- 文本工具 ----------------

_STOPWORDS = {
    "哈哈", "哈哈哈", "哈哈哈哈", "主播", "老师", "谢谢", "感谢", "好听",
    "来了", "来啦", "我来了", "前排", "第一", "沙发", "打卡", "签到",
    "晚安", "早安", "你好", "加油", "666", "888", "牛", "绝了", "可以",
    "什么", "怎么", "这个", "那个", "就是", "不是", "没有", "真的",
    "可爱", "喜欢", "支持", "棒", "赞", "笑死", "离谱",
}

# 表情码 [xxx] 与纯符号不参与热词统计
_EMOTE_RE = re.compile(r"\[[^\]\[]*\]")
_TERM_RE = re.compile(r"[\u4e00-\u9fffA-Za-z]{2,8}")
# 点歌句式：「想听《xxx》」「点一首 xxx」「来首 xxx」「求唱 xxx」
_SONG_RE = re.compile(
    r"(?:想听|点歌|点一首|点首|来一首|来首|唱一首|唱首|求唱|能不能唱|可以唱)"
    r"\s*(.+)$")
# 书名号/引号包住的歌名优先
_QUOTED_RE = re.compile(r"[《<「“\"']([^》>」”\"'\n]{1,30})[》>」”\"']")


def _extract_terms(text: str) -> list[str]:
    """从弹幕里抽候选专名（2~8 字连续中英文，剔除停用词与表情码）。"""
    cleaned = _EMOTE_RE.sub("", text or "")
    out = []
    for m in _TERM_RE.finditer(cleaned):
        w = m.group()
        if w in _STOPWORDS:
            continue
        if len(w) >= 2:
            out.append(w)
    return out


def _song_name_from_text(text: str) -> str:
    """从弹幕/SC 文本里提取点歌请求的歌名；无请求返回空串。

    优先取书名号/引号内的标题；否则取点歌句式之后的短语，
    并去掉「唱一首」「原key」等修饰与语气符号。
    """
    if not text or not any(h in text for h in _SONG_HINTS):
        return ""
    cleaned = _EMOTE_RE.sub("", text)
    m = _QUOTED_RE.search(cleaned)
    if m:
        name = _clean_song_name(m.group(1))
        if name:
            return name
    m = _SONG_RE.search(cleaned)
    if not m:
        return ""
    return _clean_song_name(m.group(1))


def _clean_song_name(raw: str) -> str:
    """去掉歌名周围的修饰词/标点，保留可检索的核心。

    注意顺序：先剥掉「一首/首/个」这类**量词**，再剥「唱」，
    否则「点一首七里香」会被误剥成空（实测踩过）。
    """
    name = (raw or "").strip()
    # 只要第一句（SC 常带长篇附言）
    name = re.split(r"[，,。！!？?~～\s]", name)[0].strip()
    for _ in range(3):        # 允许叠加修饰（如「主包唱一首」）
        before = name
        name = re.sub(r"^(?:那|这)?(?:个|首|支)\s*", "", name)
        # 常见称呼前缀（主播名请用 --host-names 配置，不要硬编码具体主播）
        name = re.sub(r"^(?:主包|主播|姐姐|哥哥|小姐姐|up主|up)\s*", "",
                      name, flags=re.I)
        name = re.sub(r"^(?:一首|一首歌|首|个|支)\s*", "", name)
        name = re.sub(r"^(?:原key|原\s*key|升调|降调)\s*", "", name, flags=re.I)
        name = re.sub(r"^(?:唱|来|点|听)\s*", "", name)
        if name == before:
            break
    # 去掉尾部修饰
    name = re.sub(r"(?:的原key|原key|升调|降调|的版本|版|那首|这首|好吗|行吗|"
                  r"可以不|可以吗|谢谢)$", "", name, flags=re.I)
    name = name.strip(" 　《》<>「」“”\"'·-—")
    return name if 2 <= len(name) <= 30 else ""


def _looks_like_gift(text: str) -> bool:
    return any(k in text for k in _GIFT_WORDS)


def _user_of(row: dict) -> str:
    for key in ("uname", "user_name", "username", "nickname", "name", "user"):
        if row.get(key):
            return str(row[key])
    user = row.get("user")
    if isinstance(user, dict):
        return str(user.get("name") or user.get("uname") or "")
    return ""


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _repair_truncated(text: str) -> str:
    """补齐被截断的 XML（录制中断常见）。

    保留 RecordInfo（主播名/标题/起始时间对后续用途有价值），
    并抽出所有完整的事件节点重新包裹，让解析器能继续工作。
    """
    info = re.search(r"<BililiveRecorderRecordInfo\b[^>]*/?>", text)
    rows = re.findall(
        r"<(?:d|gift|sc|guard)\b[^>]*(?:/>|>.*?</(?:d|gift|sc|guard)>)",
        text, flags=re.S)
    head = info.group(0) if info else ""
    return "<i>" + head + "".join(rows) + "</i>"


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def danmaku_density(log: LiveLog, duration: float, window: float = 30.0
                    ) -> list[float]:
    """按窗口统计弹幕密度（供「是否在互动/唱歌」判断）。"""
    if duration <= 0 or window <= 0:
        return []
    n = int(duration / window) + 1
    bins = [0.0] * n
    for d in log.danmaku:
        i = int(d.time / window)
        if 0 <= i < n:
            bins[i] += 1
    return bins


__all__ = [
    "Danmaku", "GiftEvent", "LiveLog", "parse_xml", "parse_jsonl", "load_log",
    "merge_logs", "danmaku_density",
]
