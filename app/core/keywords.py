"""用户自定义关键词字库。

用途
────
直播里有一批「模型必然念错」的专名：主播昵称、粉丝团名、口头禅、
游戏术语、产品型号……它们不在通用语料里，whisper 只能靠发音硬猜。

把这些词显式告诉模型，能显著降低专名错误。三个作用点：

1. **识别阶段**：作为 `initial_prompt` / `hotwords` 提示词，引导模型
   倾向输出这些写法（比任何后处理都根本——它让模型一开始就写对）。
2. **后处理阶段**：识别结果里出现「与某关键词高度相似的片段」时替换为
   关键词的正确写法（处理提示词没能救回来的情况）。
3. **质检阶段**：标记出「疑似应为某关键词但写法不同」的行，供人工复核。

存储
────
单个 JSON 文件（`keywords/keywords.json`），便于备份与手工编辑。
每条词可带 `alias`（常见误写），用于更精准的纠正。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .song import norm_text, score_similarity

# 与关键词相似度高于此值才认为「这是同一个词但写法有偏差」
ALIAS_THRESHOLD = 0.72
# 关键词最短长度（1 个字的词误伤风险高）
MIN_WORD_CHARS = 2


@dataclass
class Keyword:
    """一个自定义关键词。"""
    word: str
    alias: list[str] = field(default_factory=list)   # 常见误写
    note: str = ""                                    # 备注（供自己看）
    category: str = ""                                # 分类（昵称/术语…）
    enabled: bool = True
    added_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class KeywordStore:
    """关键词字库（落盘 JSON）。"""

    def __init__(self, root: str | Path = "keywords") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "keywords.json"
        self._items: dict[str, Keyword] = {}
        self._load()

    # ---- 持久化 ----

    def _load(self) -> None:
        if not self.index_path.is_file():
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return
        for it in data.get("keywords") or []:
            w = str(it.get("word") or "").strip()
            if not w:
                continue
            self._items[norm_text(w)] = Keyword(
                word=w,
                alias=[str(a).strip() for a in (it.get("alias") or [])
                       if str(a).strip()],
                note=str(it.get("note") or ""),
                category=str(it.get("category") or ""),
                enabled=bool(it.get("enabled", True)),
                added_at=float(it.get("added_at") or time.time()),
            )

    def _save(self) -> None:
        payload = {"version": 1,
                   "keywords": [k.to_dict() for k in self._items.values()]}
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.index_path)

    # ---- 增删查 ----

    @property
    def size(self) -> int:
        return len(self._items)

    def list_items(self) -> list[Keyword]:
        return sorted(self._items.values(), key=lambda k: (k.category, k.word))

    def add(self, word: str, *, alias: Optional[list[str]] = None,
            note: str = "", category: str = "") -> Keyword:
        w = (word or "").strip()
        if len(norm_text(w)) < MIN_WORD_CHARS:
            raise ValueError(f"关键词太短（至少 {MIN_WORD_CHARS} 个字）：{w!r}")
        k = Keyword(word=w,
                    alias=[a.strip() for a in (alias or []) if a.strip()],
                    note=note, category=category)
        self._items[norm_text(w)] = k
        self._save()
        return k

    def add_many(self, text: str, *, category: str = "",
                 note: str = "") -> int:
        """批量导入：每行一个词，支持 `词|误写1,误写2|备注` 格式。"""
        added = 0
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            word = parts[0]
            if not word:
                continue
            aliases: list[str] = []
            if len(parts) > 1 and parts[1]:
                aliases = [a.strip() for a in parts[1].split(",") if a.strip()]
            note2 = parts[2] if len(parts) > 2 else note
            try:
                self.add(word, alias=aliases, note=note2, category=category)
                added += 1
            except ValueError:
                continue
        return added

    def remove(self, word: str) -> bool:
        k = norm_text(word)
        if k in self._items:
            del self._items[k]
            self._save()
            return True
        return False

    def set_enabled(self, word: str, enabled: bool) -> bool:
        k = norm_text(word)
        if k in self._items:
            self._items[k].enabled = enabled
            self._save()
            return True
        return False

    def clear(self) -> int:
        n = len(self._items)
        self._items.clear()
        self._save()
        return n

    # ---- 供识别与校正使用 ----

    def prompt_words(self, limit: int = 60) -> list[str]:
        """用于 initial_prompt / hotwords 的词表（启用的词）。"""
        return [k.word for k in self.list_items() if k.enabled][:limit]

    def prompt_text(self, limit: int = 60) -> str:
        return " ".join(self.prompt_words(limit))

    def match_map(self) -> dict[str, str]:
        """{规范化写法: 正确写法}，含所有别名（供纠错查表）。"""
        out: dict[str, str] = {}
        for k in self._items.values():
            if not k.enabled:
                continue
            out[norm_text(k.word)] = k.word
            for a in k.alias:
                na = norm_text(a)
                if na:
                    out[na] = k.word
        return out


def correct_with_keywords(lines: Iterable, store: KeywordStore, *,
                          threshold: float = ALIAS_THRESHOLD
                          ) -> tuple[list, list[dict]]:
    """用关键词字库纠正字幕文本。

    两层处理：
    1. **精确别名替换**：字幕里出现已登记的误写（alias），直接换成正确写法；
    2. **模糊纠正**：整行与某关键词相似度 ≥ threshold 时替换（处理
       「模型把词写成了别的同音字」但用户没登记该误写的情况）。

    只改文本，不动时间轴。返回 (新行列表, 变更明细)。
    """
    src = list(lines)
    if not src or store.size == 0:
        return src, []

    exact = store.match_map()
    words = [(norm_text(k.word), k.word)
             for k in store.list_items() if k.enabled]
    changes: list[dict] = []
    out = list(src)

    for i, ln in enumerate(src):
        text = (ln.get("text", "") if isinstance(ln, dict)
                else getattr(ln, "text", ""))
        text = str(text or "")
        nt = norm_text(text)
        if len(nt) < MIN_WORD_CHARS:
            continue
        new_text = text

        # 1) 精确别名：子串替换（字幕里可能夹着这个词）
        for wrong, right in exact.items():
            if wrong == norm_text(right):
                continue
            if wrong and wrong in nt and right not in text:
                # 用规范化形式定位再映射回原文位置（简单起见按子串替换）
                if wrong in text:
                    new_text = text.replace(wrong, right)
                    nt = norm_text(new_text)

        # 2) 模糊纠正：整行近似某个关键词
        if new_text == text:
            best = (0.0, "")
            for nw, w in words:
                if abs(len(nw) - len(nt)) > 4:
                    continue
                s = score_similarity(nt, nw)
                if s > best[0]:
                    best = (s, w)
            if best[0] >= threshold and best[1]:
                new_text = best[1]

        if new_text != text:
            out[i] = _set_text(ln, new_text)
            changes.append({"line_index": i, "before": text,
                            "after": new_text,
                            "time": float(getattr(ln, "start", 0)
                                          if not isinstance(ln, dict)
                                          else ln.get("start", 0) or 0)})
    return out, changes


def _set_text(line, new_text: str):
    """替换一行文本，保持时间与结构。"""
    if isinstance(line, dict):
        d = dict(line)
        d["text"] = new_text
        words = d.get("words") or []
        if words:
            s = float(words[0].get("start", d.get("start", 0)))
            e = float(words[-1].get("end", d.get("end", s + 1)))
            d["words"] = [{"text": new_text, "start": s, "end": e}]
        return d
    from .models import SubtitleLine, SubtitleWord

    s = float(getattr(line, "start", 0.0))
    e = float(getattr(line, "end", s + 1.0))
    return SubtitleLine(
        words=[SubtitleWord(text=new_text, start=s, end=e)],
        speaker=getattr(line, "speaker", ""),
        low_confidence=getattr(line, "low_confidence", False),
        low_confidence_reason=getattr(line, "low_confidence_reason", ""))


__all__ = ["ALIAS_THRESHOLD", "MIN_WORD_CHARS", "Keyword", "KeywordStore",
           "correct_with_keywords"]
