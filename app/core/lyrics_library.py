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
    """一条字幕行与歌词行的匹配结果。"""
    line_index: int
    lyric_index: int
    score: float
    subtitle_text: str
    lyric_text: str
    start: float = 0.0
    end: float = 0.0


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


def match_document(sub_lines: Iterable, doc: LyricDoc, *,
                   threshold: float = DEFAULT_MATCH_THRESHOLD,
                   window: int = 8) -> MatchReport:
    """把字幕与一首歌词做「顺序感知」的匹配。

    做法：对每条字幕行，在歌词中**允许有限跳跃**地寻找最相似的一行
    （`window` 控制向前搜索的行数）。这比全局相似度更稳——歌词与字幕的
    行数通常不一致（识别会多切或少切），但**顺序**是一致的。

    返回的 `score` 是匹配行相似度的加权平均，乘以覆盖率修正，
    用于判断「这首歌词是否适用」。
    """
    rows = _subtitle_rows(sub_lines)
    if not rows or not doc.lines:
        return MatchReport(doc=doc, reason="没有可比对的内容")

    lyric = [(i, norm_text(ln.text)) for i, ln in enumerate(doc.lines)]
    lyric = [(i, t) for i, t in lyric if len(t) >= MIN_LINE_CHARS]
    if not lyric:
        return MatchReport(doc=doc, reason="歌词缺少有效文本")

    matches: list[LineMatch] = []
    cursor = 0                     # 当前歌词位置（保证顺序）
    hit_scores: list[float] = []

    for idx, ntext, start, end in rows:
        best = (0.0, -1)
        lo = max(0, cursor - 1)
        hi = min(len(lyric), cursor + window)
        for j in range(lo, hi):
            lj, ltext = lyric[j]
            # 长度差过大时直接跳过，省算力也减少误配
            if abs(len(ltext) - len(ntext)) > max(6, len(ntext)):
                continue
            s = score_similarity(ntext, ltext)
            if s > best[0]:
                best = (s, j)
        if best[1] >= 0 and best[0] >= threshold:
            j = best[1]
            lj, ltext = lyric[j]
            orig = doc.lines[lj].text
            matches.append(LineMatch(
                line_index=idx, lyric_index=lj, score=best[0],
                subtitle_text=str(ntext), lyric_text=orig,
                start=start, end=end))
            hit_scores.append(best[0])
            cursor = j + 1
        else:
            hit_scores.append(0.0)

    if not matches:
        return MatchReport(doc=doc, score=0.0,
                           considered=len(rows),
                           reason="没有达到阈值的匹配行")

    avg = sum(hit_scores) / len(hit_scores)
    coverage = len(matches) / max(1, len(rows))
    # 覆盖率太低说明可能只是零星巧合（如副歌重复）
    overall = avg * (0.5 + 0.5 * coverage)
    return MatchReport(doc=doc, score=round(overall, 4), matches=matches,
                       considered=len(rows),
                       reason=f"匹配 {len(matches)}/{len(rows)} 行")


def apply_report(lines: list, report: MatchReport, *,
                 min_score: float = 0.75) -> tuple[list, int]:
    """按匹配结果替换字幕文本（只替换高置信行）。

    替换策略：只有当歌词行**明显更完整**时才替换——识别结果常被截断，
    直接无脑替换会丢掉识别到的额外内容。因此要求：
      - 相似度 ≥ `min_score`
      - 歌词行不短于识别文本（避免换成更短的版本）
    """
    if not report.matches:
        return lines, 0
    out = list(lines)
    replaced = 0
    for m in report.matches:
        if m.score < min_score:
            continue
        if m.line_index >= len(out):
            continue
        ln = out[m.line_index]
        new_text = m.lyric_text.strip()
        old_text = str(getattr(ln, "text", "")
                       or (ln.get("text", "") if isinstance(ln, dict) else ""))
        if not new_text or len(norm_text(new_text)) < len(norm_text(old_text)):
            continue
        if norm_text(new_text) == norm_text(old_text):
            continue
        out[m.line_index] = _replace_text(ln, new_text)
        replaced += 1
    return out, replaced


def _replace_text(line, new_text: str):
    """替换一行字幕的文本，保持时间与词级信息结构。"""
    if isinstance(line, dict):
        d = dict(line)
        d["text"] = new_text
        words = d.get("words") or []
        if words:
            # 单行替换后词级时间失去意义，保留首尾时间给一个覆盖整行的词
            s = float(words[0].get("start", d.get("start", 0)))
            e = float(words[-1].get("end", d.get("end", s + 1)))
            d["words"] = [{"text": new_text, "start": s, "end": e}]
        return d
    # SubtitleLine：重建一个只含一个词的行
    from .models import SubtitleLine, SubtitleWord

    s = float(getattr(line, "start", 0.0))
    e = float(getattr(line, "end", s + 1.0))
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
