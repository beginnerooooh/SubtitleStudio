"""字幕切行：词/字流 → 与人工校对习惯一致的字幕行。

人工字幕的分行规律（从 545 行人工校对样本统计得出）：
- 每行 1~17 字，中位数 9 字；一行一个完整语块，不切断词；
- 句末标点（。！？）必断；逗号/顿号处大概率断；长停顿处断；
- 行不重叠，短行（"嗯""哎"）独立成行。

本模块把「切行」建模为最短路径问题：每个候选边界带惩罚项，
用动态规划求全片总代价最小的切分，避免贪心造成的连锁错位：

    代价 = 长度偏差² + 语义边界奖励（标点 / 停顿）+ 硬约束（最少字数）

纯逻辑、零模型依赖，可直接单测；阈值集中在 SegmentTuning 中，
便于按语料标定（不再散落魔数）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from core.models import SubtitleLine, SubtitleWord

# 句末标点：必断
SENTENCE_END = tuple("。．！？!?；;…")
# 停顿类标点：大概率断（逗号、顿号、冒号）
CLAUSE_PUNCT = tuple("，,、：:")

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_TRAILING_PUNCT = set("。．！？!?；;…，,、：:、.·")


@dataclass
class SegmentTuning:
    """切行参数（默认值由人工校对样本标定，可整体替换）。"""

    target_chars: float = 9.0     # 目标行长（样本中位数）
    max_chars: int = 17           # 硬上限（样本最大值）
    min_chars: int = 1            # 硬下限（"嗯""哎"独立成行）
    max_duration: float = 8.5     # 硬上限（样本最长 8.20s + 余量）
    # 代价权重：全部相对「长度偏差」归一，避免某一项压倒其余项。
    # w_sentence / w_clause / w_pause 的单位是「多少个长度偏差单位」。
    w_sentence: float = 3.0       # 句末标点
    w_clause: float = 1.2         # 逗号/顿号
    w_pause: float = 0.8          # 长停顿（≥ pause_gap 秒）
    w_dev: float = 1.0            # 行长偏离目标
    w_duration: float = 0.5       # 时长偏离目标（3s）
    pause_gap: float = 0.28       # 视为停顿的相邻单元间隔（秒）
    # 行首/行尾孤立标点清理
    strip_punct: bool = True


def _is_punct(ch: str) -> bool:
    return ch in _TRAILING_PUNCT or (not ch.isalnum() and not _CJK_RE.match(ch))


def split_atoms(words: list[SubtitleWord]) -> list[SubtitleWord]:
    """词流 → 原子单元流。

    中文多字词按字符均分时间（人工字幕按字断行，不按模型的分词粒度）；
    拉丁词/数字整体保留。标点附着在其前面的原子上。
    每个原子恰好含一个「字」（拉丁词为一个词），因此外部标点线索可按
    序号与原子对齐（见 punctuate.boundary_flags）。
    """
    atoms: list[SubtitleWord] = []
    for word in words:
        text = word.text
        if not text:
            continue
        core = text.lstrip()
        lead = text[: len(text) - len(core)]
        body, trail = _split_trailing_punct(core)
        if not body:
            if atoms:
                atoms[-1].text += lead + trail
            continue
        if _cjk_run(body) and len(body) > 1:
            n = len(body)
            dur = max(0.0, word.end - word.start)
            for i, ch in enumerate(body):
                s = word.start + dur * i / n
                e = word.start + dur * (i + 1) / n
                atoms.append(SubtitleWord(
                    text=(lead + ch) if i == 0 else ch, start=s, end=e))
            atoms[-1].text += trail
            if len(trail) > 1:
                # 多标点（如 "？。"）：单独挂一个零时长原子，保持字符序
                for extra in trail[1:]:
                    atoms.append(SubtitleWord(
                        text=extra, start=word.end, end=word.end))
        elif _cjk_run(body) or len(body) == 1:
            atoms.append(SubtitleWord(text=lead + body + trail,
                                      start=word.start, end=word.end))
        else:
            # 拉丁词/数字串：保留整体（不逐字切分），标点附着
            atoms.append(SubtitleWord(text=lead + body + trail,
                                      start=word.start, end=word.end))
    return atoms


def _cjk_run(text: str) -> bool:
    """整段是否全为汉字（可逐字切分）。"""
    return bool(text) and all(_CJK_RE.match(c) for c in text)


def _split_trailing_punct(text: str) -> tuple[str, str]:
    """拆出结尾标点串（含空白），返回 (正文, 尾标点)。"""
    i = len(text)
    while i > 0 and (text[i - 1].isspace() or _is_punct(text[i - 1])):
        i -= 1
    return text[:i], text[i:]


def _join_text(atoms: list[SubtitleWord]) -> str:
    """拼接显示文本：汉字之间不加空格，拉丁词保留原有空格。"""
    out = ""
    for a in atoms:
        t = a.text
        if not t:
            continue
        if out and not out[-1].isspace() and t[:1].isspace():
            # 原子自带的空格已是有意分隔，保留
            out += t
        elif out and _needs_space(out, t):
            out += " " + t.lstrip()
        else:
            out += t
    return out.strip()


def _needs_space(prev: str, cur: str) -> bool:
    """拉丁词之间补空格；任一侧为汉字/标点时不补（中文不加空格）。"""
    a, b = prev[-1:], cur.lstrip()[:1]
    if not a or not b:
        return False
    return not (_CJK_RE.match(a) or _is_punct(a)) and not (
        _CJK_RE.match(b) or _is_punct(b))


def _boundary_bonus(prev: SubtitleWord, cur: SubtitleWord,
                    tuning: SegmentTuning) -> float:
    """在 prev/cur 之间断行的奖励（越大越好断）。"""
    text = prev.text.rstrip()
    gap = cur.start - prev.end
    if gap >= tuning.pause_gap:
        # 停顿越长越倾向断行（半秒以上封顶）
        return tuning.w_pause * min(1.0, gap / 0.5)
    if text.endswith(SENTENCE_END):
        return tuning.w_sentence
    if text.endswith(CLAUSE_PUNCT):
        return tuning.w_clause
    return 0.0


def segment_words(
    words: list[SubtitleWord],
    tuning: SegmentTuning | None = None,
    *,
    start: float | None = None,
    end: float | None = None,
    punctuation: list[bool] | None = None,
) -> list[SubtitleLine]:
    """词流 → 字幕行（DP 全局最优切分）。

    start/end：强制切分区间（来自 VAD 语音段或演唱块边界）。区间内不再
    使用模型给出的段边界作为硬切分点，由代价函数自然决定。
    punctuation：与「原子」等长的断句线索（长度或标点标记），若给出则
    与词内标点共同决定断行奖励——模型输出常无标点，需要外部补标点后
    再映射回来。传入时长度应为 split_atoms(words) 的长度。
    返回的 SubtitleLine 只含切好的词，low_confidence 等字段由调用方补充。
    """
    tuning = tuning or SegmentTuning()
    atoms = split_atoms(words)
    if not atoms:
        return []
    if start is not None and end is not None:
        atoms = [a for a in atoms if a.start >= start - 1e-6 and a.end <= end + 1e-6]
        if not atoms:
            return []
    hints = punctuation
    if hints is not None and len(hints) != len(atoms):
        # 长度不匹配（区间裁剪/多字词拆分）时放弃外部线索，避免错位
        hints = None

    n = len(atoms)
    prefix_chars = [0] * (n + 1)
    for i, a in enumerate(atoms):
        prefix_chars[i + 1] = prefix_chars[i] + len(_join_text([a]))

    INF = float("inf")
    best = [INF] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    tgt = max(1.0, tuning.target_chars)

    for i in range(n):
        if best[i] == INF:
            continue
        for j in range(i + tuning.min_chars, min(n, i + tuning.max_chars) + 1):
            chars = prefix_chars[j] - prefix_chars[i]
            dur = atoms[j - 1].end - atoms[i].start
            if dur > tuning.max_duration and j - i > 1:
                break
            cost = tuning.w_dev * ((chars - tgt) / tgt) ** 2
            cost += tuning.w_duration * max(0.0, (dur - 3.0) / 3.0) ** 2
            if j < n:
                # 标点/停顿奖励（负代价）；外部标点线索优先
                if hints is not None and hints[j - 1]:
                    cost -= tuning.w_clause
                else:
                    cost -= _boundary_bonus(atoms[j - 1], atoms[j], tuning)
            if best[i] + cost < best[j]:
                best[j] = best[i] + cost
                back[j] = i

    spans: list[tuple[int, int]] = []
    j = n
    while j > 0:
        i = back[j]
        spans.append((i, j))
        j = i
    spans.reverse()

    lines: list[SubtitleLine] = []
    for i, j in spans:
        chunk = atoms[i:j]
        text_atoms = list(chunk)
        if tuning.strip_punct:
            text_atoms = _drop_leading_punct(text_atoms)
        if not text_atoms:
            continue
        lines.append(SubtitleLine(words=text_atoms))
    return lines


def _drop_leading_punct(atoms: list[SubtitleWord]) -> list[SubtitleWord]:
    """行首孤立标点并回上一行末尾（人工字幕不会以逗号/句号开头）。"""
    out = list(atoms)
    if not out:
        return out
    first = out[0]
    stripped = first.text.lstrip()
    while stripped and _is_punct(stripped[0]) and not _CJK_RE.match(stripped[0]):
        stripped = stripped[1:]
    if stripped != first.text.lstrip():
        if stripped:
            out[0] = SubtitleWord(text=stripped, start=first.start, end=first.end)
        else:
            out.pop(0)
    return out


__all__ = [
    "SegmentTuning", "segment_words", "split_atoms", "SENTENCE_END",
    "CLAUSE_PUNCT",
]
