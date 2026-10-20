"""歌词库：本地导入 + 在线拉取 + 与识别字幕做相似度匹配校正。

设计动机
────────
识别模型对**歌声**的准确率显著低于说话（发音拉长、伴奏干扰、混响）。
但歌曲是有确定文本的：只要拿到歌词，就能把识别错误「纠正」回来。

因此这里提供一条独立于识别的链路：

    歌词库（导入文件 / 按歌名+作者在线拉取）
        ↓  与生成的字幕按时间 + 文本相似度匹配
    高置信匹配 → 用歌词原文替换识别文本（可选）

与 `song.py` 的分工
──────────────────
- `song.py`：识别阶段用音频/文本线索「发现」这是哪首歌
- `lyrics_library.py`（本模块）：用户显式维护的曲库 + 生成后的**文本校正**

两者共享 `song.norm_text` / `score_similarity` 的度量方式，避免两套标准。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .song import lrc_rows, norm_text, score_similarity

# 支持的歌词文本扩展名
LYRIC_EXTS = {".lrc", ".txt", ".srt", ".json"}

# 低于此相似度的匹配不使用（避免把无关歌词套进来）
DEFAULT_MATCH_THRESHOLD = 0.55
# 单行歌词与字幕行的最短可比长度，太短容易误匹配
MIN_LINE_CHARS = 3


@dataclass
class LyricLine:
    """一行歌词（可能带时间戳）。"""
    text: str
    time: float = 0.0          # 有 LRC 时间时为秒；纯文本为 0


@dataclass
class LyricDoc:
    """一首歌的歌词文档。"""
    title: str
    artist: str = ""
    lines: list[LyricLine] = field(default_factory=list)
    source: str = "import"     # import / netease / kugou / lrclib / manual
    synced: bool = False       # 是否带时间轴
    path: str = ""
    added_at: float = field(default_factory=time.time)

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)

    @property
    def line_count(self) -> int:
        return len(self.lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lines"] = [{"text": ln.text, "time": ln.time} for ln in self.lines]
        d["text"] = self.text
        d["line_count"] = self.line_count
        return d


# ---------------- 解析 ----------------

def parse_lyric_text(text: str, *, title: str = "", artist: str = "",
                     source: str = "import", path: str = "") -> LyricDoc:
    """解析歌词文本：自动识别 LRC（带时间）或纯文本。"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    rows = lrc_rows(text)
    lines: list[LyricLine] = []
    if rows:
        for t, content in rows:
            c = (content or "").strip()
            if c:
                lines.append(LyricLine(text=c, time=round(float(t), 3)))
        synced = True
    else:
        lines = [LyricLine(text=ln.strip()) for ln in text.split("\n")
                 if ln.strip()]
        synced = False
    return LyricDoc(title=title, artist=artist, lines=lines,
                    source=source, synced=synced, path=path)


def _guess_title_artist(name: str) -> tuple[str, str]:
    """从文件名猜歌名与歌手：`歌手 - 歌名.lrc` / `歌名 - 歌手.lrc`。"""
    stem = Path(name).stem
    for sep in (" - ", "-", "–", "—"):
        if sep in stem:
            a, b = stem.split(sep, 1)
            a, b = a.strip(), b.strip()
            if a and b:
                return b, a       # 常见约定：歌手在前
    return stem.strip(), ""


def load_lyric_file(path: str | Path) -> LyricDoc:
    """读取单个歌词文件（.lrc / .txt / .srt / .json）。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"歌词文件不存在：{p}")
    if p.suffix.lower() not in LYRIC_EXTS:
        raise ValueError(f"不支持的歌词格式：{p.suffix}（支持 "
                         f"{', '.join(sorted(LYRIC_EXTS))}）")
    raw = p.read_text(encoding="utf-8-sig", errors="replace")
    title, artist = _guess_title_artist(p.name)
    # JSON 形式允许显式给出 title/artist
    if p.suffix.lower() == ".json":
        try:
            data = json.loads(raw)
            title = str(data.get("title") or title)
            artist = str(data.get("artist") or artist)
        except Exception:
            pass
    return parse_lyric_text(raw, title=title, artist=artist,
                            source="import", path=str(p))


# ---------------- 曲库 ----------------

class LyricLibrary:
    """本地歌词库（落盘为单个 JSON，便于备份与迁移）。"""

    def __init__(self, root: str | Path = "lyrics") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "library.json"
        self._docs: dict[str, LyricDoc] = {}
        self._load()

    # ---- 持久化 ----

    def _load(self) -> None:
        if not self.index_path.is_file():
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return
        for item in data.get("docs") or []:
            try:
                doc = LyricDoc(
                    title=str(item.get("title") or ""),
                    artist=str(item.get("artist") or ""),
                    lines=[LyricLine(text=str(x.get("text", "")),
                                     time=float(x.get("time") or 0))
                           for x in (item.get("lines") or [])
                           if str(x.get("text", "")).strip()],
                    source=str(item.get("source") or "import"),
                    synced=bool(item.get("synced")),
                    path=str(item.get("path") or ""),
                    added_at=float(item.get("added_at") or time.time()),
                )
            except Exception:
                continue
            if doc.lines:
                self._docs[_key(doc.title, doc.artist)] = doc

    def _save(self) -> None:
        payload = {"version": 1,
                   "docs": [d.to_dict() for d in self._docs.values()]}
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.index_path)

    # ---- 增删查 ----

    @property
    def size(self) -> int:
        return len(self._docs)

    def list_docs(self) -> list[LyricDoc]:
        return sorted(self._docs.values(),
                      key=lambda d: (d.artist, d.title))

    def add(self, doc: LyricDoc, *, overwrite: bool = True) -> LyricDoc:
        if not doc.lines:
            raise ValueError("歌词内容为空，未导入")
        doc.title = (doc.title or "").strip() or "未命名"
        doc.artist = (doc.artist or "").strip()
        k = _key(doc.title, doc.artist)
        if k in self._docs and not overwrite:
            raise ValueError(f"歌词库中已存在《{doc.title}》")
        self._docs[k] = doc
        self._save()
        return doc

    def add_file(self, path: str | Path) -> LyricDoc:
        return self.add(load_lyric_file(path))

    def add_text(self, text: str, *, title: str, artist: str = "",
                 source: str = "manual") -> LyricDoc:
        doc = parse_lyric_text(text, title=title, artist=artist, source=source)
        return self.add(doc)

    def remove(self, title: str, artist: str = "") -> bool:
        k = _key(title, artist)
        if k in self._docs:
            del self._docs[k]
            self._save()
            return True
        # 宽松删除：只给歌名时删掉同名第一条
        for key, doc in list(self._docs.items()):
            if doc.title == title:
                del self._docs[key]
                self._save()
                return True
        return False

    def get(self, title: str, artist: str = "") -> Optional[LyricDoc]:
        k = _key(title, artist)
        if k in self._docs:
            return self._docs[k]
        for doc in self._docs.values():
            if doc.title == title:
                return doc
        return None

    def find_by_title(self, title: str) -> Optional[LyricDoc]:
        """按歌名模糊查（去掉标点/大小写差异）。"""
        want = norm_text(title)
        if not want:
            return None
        best, best_score = None, 0.0
        for doc in self._docs.values():
            s = score_similarity(norm_text(doc.title), want)
            if s > best_score:
                best, best_score = doc, s
        return best if best_score >= 0.7 else None

    # ---- 在线拉取 ----

    def fetch(self, title: str, artist: str = "",
              on_log=None) -> list[LyricDoc]:
        """从在线源拉取并**入库**，返回成功入库的文档。

        复用 `song.search_lyrics` 的多源 + 自动换源逻辑，避免重复实现。
        """
        from .song import search_lyrics

        rows = search_lyrics(title=title, artist=artist, on_log=on_log)
        added: list[LyricDoc] = []
        for r in rows[:3]:
            content = r.get("content") or ""
            if not content.strip():
                continue
            doc = parse_lyric_text(
                content,
                title=r.get("title") or title,
                artist=r.get("artist") or artist,
                source=r.get("source") or "online",
            )
            if not doc.lines:
                continue
            # 重名同歌手的不重复入库
            if _key(doc.title, doc.artist) in self._docs:
                continue
            self.add(doc, overwrite=False)
            added.append(doc)
        return added


def _key(title: str, artist: str) -> str:
    return f"{norm_text(title)}|{norm_text(artist)}"


# ---------------- 与字幕匹配校正 ----------------

@dataclass
class LineMatch:
    """一条字幕（窗口）与歌词行的匹配结果。"""
    line_index: int
    lyric_index: int
    score: float
    subtitle_text: str
    lyric_text: str
    start: float = 0.0
    end: float = 0.0
    # 该匹配覆盖到的最后一行字幕（多行合并时 > line_index）。
    # 替换时需把这几行合并成一行歌词文本。
    merged_to: int = -1


@dataclass
class MatchReport:
    doc: Optional[LyricDoc] = None
    score: float = 0.0                 # 整体相似度（0~1）
    matches: list[LineMatch] = field(default_factory=list)
    replaced: int = 0
    considered: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.doc.title if self.doc else "",
            "artist": self.doc.artist if self.doc else "",
            "score": round(self.score, 4),
            "replaced": self.replaced,
            "considered": self.considered,
            "reason": self.reason,
            "matches": [{"line_index": m.line_index,
                         "lyric_index": m.lyric_index,
                         "score": round(m.score, 4),
                         "subtitle": m.subtitle_text,
                         "lyric": m.lyric_text,
                         "start": round(m.start, 3),
                         "end": round(m.end, 3)}
                        for m in self.matches],
        }


def _subtitle_rows(lines: Iterable) -> list[tuple[int, str, float, float]]:
    """把字幕行整理成 (索引, 规范化文本, 起, 止)。"""
    out = []
    for i, ln in enumerate(lines):
        text = getattr(ln, "text", None)
        if text is None and isinstance(ln, dict):
            text = ln.get("text", "")
        text = str(text or "")
        nt = norm_text(text)
        if len(nt) < MIN_LINE_CHARS:
            continue
        start = float(getattr(ln, "start",
                              ln.get("start", 0) if isinstance(ln, dict) else 0))
        end = float(getattr(ln, "end",
                            ln.get("end", 0) if isinstance(ln, dict) else 0))
        out.append((i, nt, start, end))
    return out


def _windows(rows: list[tuple[int, str, float, float]], max_merge: int = 3
             ) -> list[tuple[int, int, str, float, float]]:
    """把连续字幕行合并成若干「窗口」，用于与单行歌词比较。

    为什么需要合并（真实数据实测）
    ──────────────────────────────
    ASR 的切行粒度与歌词**不同**：歌词一行「放开了拳头 反而更自由」
    常被切成字幕的两行「放开了拳头」「反而更自由」。按「一行对一行」
    比较时，每行只覆盖歌词的一半，相似度自然上不去 → 整首歌匹配失败。

    因此这里生成 1~`max_merge` 行的滑动窗口：
    单行能对上就用单行，对不上时尝试「2 行合起来」「3 行合起来」。
    这是解决真实数据匹配的关键一步。
    """
    out: list[tuple[int, int, str, float, float]] = []
    n = len(rows)
    for i in range(n):
        for k in range(1, max_merge + 1):
            if i + k > n:
                break
            chunk = rows[i:i + k]
            text = "".join(r[1] for r in chunk)
            out.append((chunk[0][0], chunk[-1][0], text,
                        chunk[0][2], chunk[-1][3]))
    return out


def match_document(sub_lines: Iterable, doc: LyricDoc, *,
                   threshold: float = DEFAULT_MATCH_THRESHOLD,
                   window: int = 8,
                   min_run: int = 2,
                   max_merge: int = 3) -> MatchReport:
    """用歌词去匹配**一段连续的相似字幕**（局部匹配，不看全片平均）。

    三层设计，逐层解决真实数据上的问题：

    1. **局部匹配**（不看全片平均）
       一首歌通常只占全片几分钟。若按全片行数平均，一首 5 行的歌混在
       200 行闲聊里得分会被稀释到 0.006 量级 → 整首歌被否决。

    2. **多行合并**（关键）
       ASR 的切行粒度与歌词不同：歌词一行常被切成字幕的两三行。
       只做「一行对一行」会让每行只覆盖歌词的一部分，相似度上不去。
       因此对每个位置尝试 1~`max_merge` 行的组合，取最佳匹配。

    3. **连续段评分**
       收集连续命中的片段，取最佳段评分——与全片长度无关，
       因此阈值可以稳定地设在 0.6~0.8。
    """
    rows = _subtitle_rows(sub_lines)
    if not rows or not doc.lines:
        return MatchReport(doc=doc, reason="没有可比对的内容")

    lyric = [(i, norm_text(ln.text)) for i, ln in enumerate(doc.lines)]
    lyric = [(i, t) for i, t in lyric if len(t) >= MIN_LINE_CHARS]
    if not lyric:
        return MatchReport(doc=doc, reason="歌词缺少有效文本")

    wins = _windows(rows, max_merge=max_merge)

    # 1) 候选索引：歌词行 → [(相似度, 字幕起始位置, 结束位置, 窗口文本)]
    #
    # 注意这里的方向：以**歌词行**为 key，而不是以字幕窗口为 key。
    # 早期按字幕窗口建索引、每个窗口只保留「最佳歌词行」，会让其它
    # 歌词行在后续对齐时找不到候选 —— 表现为整首歌 0 分（实测踩过）。
    #
    # 计算量控制：窗口数 × 歌词行数是 O(n·m)。真实数据 793 窗口 ×
    # 44 歌词行 ≈ 3.5 万次相似度，实测 1.7 秒，可以接受。
    #
    # 阈值放宽到 threshold*0.8 再入库：后续对齐会按「相似度 + 长度接近度」
    # 二次筛选，先收紧会让很多「字幕切得比歌词细」的候选根本进不来，
    # 表现为命中数只有两三处（实测）。
    loose = max(0.35, threshold * 0.8)
    by_lyric: dict[int, list[tuple[float, int, int, str]]] = {}
    for first_idx, last_idx, wtext, wstart, wend in wins:
        if len(wtext) < MIN_LINE_CHARS:
            continue
        for j, (lj, ltext) in enumerate(lyric):
            # 长度差过大时跳过：省算力，也避免长短句错配
            if abs(len(ltext) - len(wtext)) > max(10, len(wtext)):
                continue
            s = score_similarity(wtext, ltext)
            if s >= loose:
                by_lyric.setdefault(lj, []).append(
                    (s, first_idx, last_idx, wtext))

    # 2) 以**歌词为主动方**顺序对齐：对每句歌词，在字幕里找能覆盖它
    #    最多字符的连续窗口，然后推进到该窗口之后。
    #
    # 为什么以歌词为准：真实数据里字幕与歌词顺序一致，但一句歌词常被
    # 切成两行字幕（「在你的心里」/「我早已落榜」）。若以字幕为主动方
    # 贪心，会优先选得分更高的单行、把另一半留成碎片（实测命中数与
    # 得分都下降）。以歌词为准时，一句歌词一定被一个窗口完整吃掉。
    hits: list[Optional[LineMatch]] = [None] * len(rows)
    idx_to_pos = {r[0]: p for p, r in enumerate(rows)}
    # 2) 以**歌词为主动方**对齐：对每句歌词在字幕里找最佳窗口。
    #
    # 关键：允许窗口**互相重叠**。真实数据里字幕比歌词切得细，
    # 一句歌词常占 2~3 行字幕，而相邻两句歌词可能共享同一行
    # （「回忆如困兽」/「寂寞太久而渐渐温柔」在同一批行里）。
    # 早期用「已占用区间」禁止重叠，反而把有效匹配全挡住了
    # （实测命中数从 4 降到 2）。
    #
    # 重叠本身不会出错：apply_report 按 lyric_index 顺序、从后往前
    # 处理，合并时以「该句歌词的最佳窗口」为准，不会互相覆盖。
    # hits 用**原始行号**索引（不是 rows 里的位置）。
    #
    # 为什么必须用行号：`_subtitle_rows` 会跳过过短的行（如单字「了」），
    # 因此 rows 的位置 k 与真实行号并不相等。早期混用两者（hits 按位置、
    # line_index 按行号），替换时就会把歌词写到相邻的错行上——实测
    # 「让爱的人都沉默」被替换成下一句歌词的内容。
    hits_by_line: dict[int, LineMatch] = {}

    # 分两遍匹配。
    #
    # 第一遍：为每句歌词保留**多个**候选字幕位置（按得分排序）。
    # 第二遍：按歌词顺序做贪心分配，为每句挑一个尚未被占用的候选。
    #
    # 为什么必须保留多个候选（实测踩坑）：真实歌曲的副歌会重复 2~3 次，
    # 歌词文件里是**不同的行**但文字相同，它们应分别对应不同的字幕位置。
    # 早期每句只留 1 个候选，第二段副歌的最佳候选恰好是第一次唱的那些行
    # （已被占），于是整段副歌全被丢弃——情歌只命中 8 处，而实际
    # 有 33 处可替换。
    provisional: dict[int, list[LineMatch]] = {}
    for lj, _ltext_norm in lyric:
        opts = by_lyric.get(lj)
        if not opts:
            continue
        ref = len(_ltext_norm)
        cands: list[tuple[tuple[float, int], LineMatch]] = []
        for sc, first_idx, last_idx, wtext in opts:
            if sc < threshold:
                continue            # 入库时放宽了，使用时要按原阈值筛
            k = idx_to_pos.get(first_idx)
            if k is None:
                continue
            q = idx_to_pos.get(last_idx, k)
            # 排序键：相似度 → 窗口长度贴近歌词长度
            key = (round(sc, 2), -abs(len(wtext) - ref))
            cands.append((key, LineMatch(
                line_index=first_idx, lyric_index=lj, score=sc,
                subtitle_text=wtext, lyric_text=doc.lines[lj].text,
                start=rows[k][2], end=rows[q][3],
                merged_to=rows[q][0])))
        if not cands:
            continue
        cands.sort(key=lambda x: x[0], reverse=True)
        # 每个歌词行最多留 8 个候选：足够覆盖重复段落，又控制内存
        provisional[lj] = [m for _, m in cands[:8]]

    # 第二遍：按歌词顺序贪心分配
    #
    # 占用语义：**同一处字幕不会被两句不同歌词争抢**；但同一句歌词的
    # 重复出现（副歌）可以复用它自己的候选列表里的下一个位置。
    #
    # 顺序约束要能区分「副歌重复」与「误配」：
    #   - 副歌：歌词文字与前面某句相同（或起点明显更早），但字幕里
    #     确实有对应的一段 → 应当接受；
    #   - 误配：歌词文字不同却跳到很远的地方 → 应当拒绝。
    # 因此这里的规则是：**文字重复的歌词行不受顺序约束**（副歌必然
    # 会「回到」较早的位置），只对文字新的歌词施加顺序检查。
    order = sorted(provisional)
    line_owner: dict[int, int] = {}      # 字幕行 → 占用它的歌词行
    last_valid = -1
    seen: dict[str, int] = {}            # 歌词文字（规范化）→ 首次出现的行号
    for lj in order:
        text_key = norm_text(doc.lines[lj].text)
        is_repeat = text_key in seen
        picked: LineMatch | None = None
        for m in provisional[lj]:
            owner = line_owner.get(m.line_index)
            if owner is not None:
                continue                # 已被别的歌词占用
            # 顺序约束仅作用于**新歌词**；重复出现的副歌可自由定位
            if (not is_repeat and last_valid >= 0
                    and m.line_index < last_valid - 30):
                continue
            picked = m
            break
        if picked is None:
            seen.setdefault(text_key, lj)
            continue
        hits_by_line[picked.line_index] = picked
        line_owner[picked.line_index] = lj
        if not is_repeat:
            last_valid = max(last_valid, picked.line_index)
        seen.setdefault(text_key, lj)


    # 3) 汇总命中：**保留全部高置信匹配**，而不是只取一个连续段。
    #
    # 为什么不能只取最佳段（实测踩坑）：真实演唱的字幕是分散在整个
    # 时间轴上的（中间穿插聊天、游戏解说），并非一段连续区域。
    # 早期只返回最大连续段，会让一场直播的几十句歌词只剩 3~4 处可用，
    # 用户看到的就是「匹配效果不行」。
    #
    # 评分仍按「段质量」给，用于判断这首歌是否适用：
    # 取得分最高的连续段的平均相似度（段长给权重），保证阈值语义不变。
    all_hits = sorted(hits_by_line.values(), key=lambda m: m.line_index)
    if not all_hits:
        return MatchReport(doc=doc, score=0.0, considered=len(rows),
                           reason="没有达到阈值的匹配行")

    # 以「行号是否连续」切分连续段（用于评估这首歌是否真的在唱）
    runs: list[list[LineMatch]] = []
    cur: list[LineMatch] = []
    prev_line = -99
    for h in all_hits:
        if cur and h.line_index - prev_line > 6:
            runs.append(cur)
            cur = []
        cur.append(h)
        prev_line = h.line_index
    if cur:
        runs.append(cur)

    best_run = max(runs, key=lambda r: (sum(m.score for m in r) / len(r)
                                        * min(1.0, len(r) / 3.0)))
    avg = sum(m.score for m in best_run) / len(best_run)
    length_weight = min(1.0, len(best_run) / 3.0)
    overall = avg * (0.7 + 0.3 * length_weight)

    return MatchReport(
        doc=doc, score=round(overall, 4),
        # 返回**全部**命中（而非仅最佳段）：一场直播里的歌词散布在
        # 整个时间轴上，只取一段会丢掉大部分可替换的行。
        # 已按行号去重并保持顺序（同一行只保留得分最高的一句歌词）。
        matches=_dedupe_by_line(sorted(all_hits, key=lambda m: m.line_index)),
        considered=len(rows),
        reason=(f"在 {len(rows)} 行中命中 {len(all_hits)} 处歌词"
                f"（段均相似度 {avg:.2f}）"))


def _dedupe_by_line(hits: list[LineMatch]) -> list[LineMatch]:
    """同一字幕行只保留得分最高的一条匹配。

    场景：一句歌词被切成两行字幕时，「命运好幽默 / 让爱的人都沉默」
    这两句歌词都会匹配到同一批字幕行，导致替换时互相覆盖。
    这里按 (line_index, merged_to) 去重，保留得分高者。
    """
    best: dict[tuple[int, int], LineMatch] = {}
    for h in hits:
        key = (h.line_index, h.merged_to)
        prev = best.get(key)
        if prev is None or h.score > prev.score:
            best[key] = h
    # 再按起始行合并：同一行只留一条
    by_start: dict[int, LineMatch] = {}
    for h in best.values():
        prev = by_start.get(h.line_index)
        if prev is None or h.score > prev.score:
            by_start[h.line_index] = h
    return sorted(by_start.values(), key=lambda m: m.line_index)


def apply_report(lines: list, report: MatchReport, *,
                 min_score: float = 0.75,
                 merge: bool = True) -> tuple[list, int]:
    """按匹配结果替换字幕文本（只替换高置信行）。

    替换策略：只有当歌词行**明显更完整**时才替换——识别结果常被截断，
    直接无脑替换会丢掉识别到的额外内容。因此要求：
      - 相似度 ≥ `min_score`
      - 歌词行不短于识别文本（避免换成更短的版本）
      - **实际文本相似度也要达标**：防止「窗口含多行字幕」时把另一句
        歌词写到本行上（实测量测出「只能游荡离开 → 离开伤心的地点
        飘向远方」这类错配，旧文本与新歌词几乎无关）

    `merge=True` 时处理「多行字幕对应一行歌词」：把合并窗口覆盖的
    那几行**合成一行**（时间取首行起、末行止）。这是真实数据里最常见的
    情况（ASR 把一句歌词切成两三行），不合并就只能替换其中一行、
    留下另一行碎片。
    """
    if not report.matches:
        return lines, 0
    src = list(lines)
    n_total = len(src)

    # 先算出要合并的区间与被占用的行，再**一次性重建**列表，
    # 而不是边删边改。
    #
    # 为什么必须这样：多行合并会删除行，导致后面所有行号前移。
    # 即便从后往前处理，前面（行号更小）的匹配仍会因为「已删除的
    # 行数」而偏移，写出错位的歌词（实测：「让爱的人都沉默」被写成
    # 了下一句「一整个宇宙 换一颗红豆」）。
    plans: list[tuple[int, int, str, float]] = []   # (起, 止, 新文本, 结束时间)
    used: set[int] = set()
    # 已用过的歌词文本：避免**同一句歌词**被写到多处。
    # 实测：不限制时「在你的心里」和「我早已落宝」两行会被同一句
    # 「在你的心里我早已落榜」覆盖，等于丢掉了后一行。
    used_texts: set[str] = set()
    for m in sorted(report.matches, key=lambda x: x.line_index):
        if m.score < min_score:
            continue
        if m.line_index >= n_total or m.line_index in used:
            continue
        new_text = m.lyric_text.strip()
        if not new_text:
            continue
        nkey = norm_text(new_text)
        if nkey in used_texts:
            continue                    # 这句歌词已经用过了
        last = m.line_index
        if merge and m.merged_to > m.line_index:
            last = min(m.merged_to, n_total - 1)
        # 合并区间与已有计划冲突时收缩到不冲突的范围
        while last > m.line_index and any(j in used for j in range(m.line_index, last + 1)):
            last -= 1
        old_parts = []
        for j in range(m.line_index, last + 1):
            row = src[j]
            old_parts.append(str(row.get("text", "") if isinstance(row, dict)
                                 else getattr(row, "text", "")))
        old_text = "".join(old_parts)
        if len(norm_text(new_text)) < len(norm_text(old_text)):
            continue
        if norm_text(new_text) == norm_text(old_text):
            continue
        # 实际相似度校验：防止把无关歌词写到本行
        if score_similarity(norm_text(old_text),
                            norm_text(new_text)) < 0.45:
            continue
        end_val = float(src[last].get("end", 0) if isinstance(src[last], dict)
                        else getattr(src[last], "end", 0))
        plans.append((m.line_index, last, new_text, end_val))
        used.update(range(m.line_index, last + 1))
        used_texts.add(nkey)

    if not plans:
        return src, 0

    # 一次性重建
    out: list = []
    plan_at = {p[0]: p for p in plans}
    i = 0
    while i < n_total:
        p = plan_at.get(i)
        if p is None:
            out.append(src[i])
            i += 1
            continue
        start, last, new_text, end_val = p
        out.append(_replace_text(src[start], new_text, end_override=end_val))
        i = last + 1
    return out, len(plans)


def _replace_text(line, new_text: str, end_override: float | None = None):
    """替换一行字幕的文本，保持时间与词级信息结构。"""
    if isinstance(line, dict):
        d = dict(line)
        d["text"] = new_text
        s = float(d.get("start", 0))
        e = float(end_override if end_override is not None
                  else d.get("end", s + 1))
        d["words"] = [{"text": new_text, "start": s, "end": e}]
        return d
    # SubtitleLine：重建一个只含一个词的行
    from .models import SubtitleLine, SubtitleWord

    s = float(getattr(line, "start", 0.0))
    e = float(end_override if end_override is not None
              else getattr(line, "end", s + 1.0))
    return SubtitleLine(
        words=[SubtitleWord(text=new_text, start=s, end=e)],
        speaker=getattr(line, "speaker", ""),
        low_confidence=getattr(line, "low_confidence", False),
        low_confidence_reason=getattr(line, "low_confidence_reason", ""),
    )


def best_document(sub_lines: Iterable, library: LyricLibrary, *,
                  titles: Optional[list[str]] = None,
                  threshold: float = DEFAULT_MATCH_THRESHOLD,
                  on_log=None) -> Optional[MatchReport]:
    """在曲库中挑出与字幕最匹配的一首。

    `titles` 非空时只考虑这些歌（对应「手动录入本场演唱曲目」）。
    """
    docs = library.list_docs()
    if titles:
        wanted = {norm_text(t) for t in titles if str(t).strip()}
        docs = [d for d in docs if norm_text(d.title) in wanted
                or any(w in norm_text(d.title) or norm_text(d.title) in w
                       for w in wanted)]
        if on_log and not docs:
            on_log("指定的曲目在歌词库中没有对应歌词，跳过校正")
    if not docs:
        return None

    best: Optional[MatchReport] = None
    for doc in docs:
        rep = match_document(sub_lines, doc, threshold=threshold)
        if best is None or rep.score > best.score:
            best = rep
    if best and on_log:
        on_log(f"歌词匹配：《{best.doc.title}》 相似度 {best.score:.2f}"
               f"（{best.reason}）")
    return best


def lyric_boundary_hints(words, doc: LyricDoc,
                         ) -> Optional[list[bool]]:
    """把「歌词的分句位置」映射成与原子等长的断句线索。

    用途：断句阶段就参考歌词，让字幕在一行歌词结束处断开，而不是
    等生成完再改。比事后替换更自然——行的时间和长度都跟着歌词走。

    做法：把词的规范化文本与歌词整篇对齐（SequenceMatcher），
    找出每个原子在歌词中的字符位置；凡是落在「歌词行首」的位置，
    标记该原子之前应当断行。
    """
    from .segmenter import split_atoms

    atoms = split_atoms(list(words))
    if not atoms:
        return None
    full = norm_text("".join(a.text for a in atoms))
    if len(full) < MIN_LINE_CHARS:
        return None
    lyric_full = norm_text(doc.text)
    if not lyric_full:
        return None

    import difflib

    # 建立「歌词第 k 个字符」是否紧随行尾
    starts: set[int] = set()
    pos = 0
    for ln in doc.lines:
        n = norm_text(ln.text)
        if not n:
            continue
        starts.add(pos)
        pos += len(n)
    if not starts:
        return None

    # 原子 → 字符区间
    spans: list[tuple[int, int]] = []
    cur = 0
    for a in atoms:
        n = len(norm_text(a.text))
        spans.append((cur, cur + n))
        cur += n

    # 序列对齐：full（字幕）↔ lyric_full（歌词）
    sm = difflib.SequenceMatcher(None, full, lyric_full, autojunk=False)
    # 字幕位置 → 歌词位置
    map_to_lyric: dict[int, int] = {}
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            map_to_lyric[a + k] = b + k

    hints = [False] * len(atoms)
    for idx, (s, e) in enumerate(spans):
        if idx == 0:
            continue
        # 该原子在歌词中的起点位置
        ly = map_to_lyric.get(s)
        if ly is None:
            continue
        # ly 恰好是某歌词行的起点 → 说明「第 idx 个原子」应当是一行的开头，
        # 因此断点落在 idx-1 之后。
        if ly in starts:
            if idx - 1 >= 0:
                hints[idx - 1] = True
    if any(hints):
        # 去掉会产生单字碎片的首尾断点
        if len(hints) > 1 and hints[0]:
            hints[0] = False
        if hints and hints[-1]:
            hints[-1] = False
    return hints if any(hints) else None


def danmaku_boundary_hints(words, log, *, window: float = 12.0,
                           ) -> Optional[list[bool]]:
    """把「同期弹幕中出现的文本」映射为断句线索。

    思路与歌词类似，但证据更短：弹幕里常出现主播刚说的短句/口令，
    若某段字幕恰好与一条高赞/重复的弹幕一致，就在它结束处断行。
    """
    from .segmenter import split_atoms

    if log is None:
        return None
    atoms = split_atoms(list(words))
    if not atoms:
        return None
    full = norm_text("".join(a.text for a in atoms))
    if len(full) < MIN_LINE_CHARS:
        return None

    # 只在时间上重叠的弹幕里找（整段词流的时间范围）
    t0 = atoms[0].start - window
    t1 = atoms[-1].end + window
    cands = [norm_text(d.text) for d in log.danmaku
             if t0 <= d.time <= t1 and len(norm_text(d.text)) >= MIN_LINE_CHARS]
    if not cands:
        return None

    hints = [False] * len(atoms)
    # 原子 → 字符区间
    spans, cur = [], 0
    for a in atoms:
        n = len(norm_text(a.text))
        spans.append((cur, cur + n))
        cur += n

    for cand in cands:
        start = 0
        while True:
            idx = full.find(cand, start)
            if idx < 0:
                break
            end = idx + len(cand)
            # 找到「结束位置」对应的原子边界
            for k, (_s, e) in enumerate(spans):
                if e == end and k < len(atoms) - 1:
                    hints[k] = True
                    break
            start = end
    return hints if any(hints) else None


__all__ = [
    "LYRIC_EXTS", "DEFAULT_MATCH_THRESHOLD", "MIN_LINE_CHARS",
    "LyricLine", "LyricDoc", "LyricLibrary",
    "parse_lyric_text", "load_lyric_file",
    "LineMatch", "MatchReport", "match_document", "apply_report",
    "best_document", "lyric_boundary_hints", "danmaku_boundary_hints",
]
