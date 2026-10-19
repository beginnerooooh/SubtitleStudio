"""用弹幕 XML 的旁路信息校正识别结果。

动机
────
直播场景里，弹幕/礼物/SC 常常**重复主播刚说过的话或歌名**：
- 主播念了一句口令 → 观众刷屏复述（时间高度接近）
- 主播唱某首歌 → 观众刷「XX 歌名」/点歌
- 礼物致谢：主播念出送礼者昵称

这些都是「时间对齐的外部文本源」。当识别结果里出现明显的专名错误
（昵称、歌名、术语），可以用同期弹幕里的正确写法替换。

设计要点
────────
1. **时间窗口加权**：同时刻附近的弹幕权重更高（主播与观众的反应几乎同步），
   隔太远的弹幕可能是无关闲聊。
2. **只替换「高频且高置信」的片段**：直接用弹幕整句替换字幕是危险的
   （弹幕多为短句、语气词），因此只在**字幕包含与某弹幕高度相似的片段**时
   做替换，且要求该写法在弹幕中反复出现。
3. **绝不新增/删除字幕行**：只做行内文本替换，保持时间轴不变。
"""
from __future__ import annotations

import difflib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .song import norm_text

# 弹幕文本短于此长度不参与校正（太短易误伤）
MIN_DANMAKU_CHARS = 3
# 时间窗口（秒）：字幕行与弹幕的允许时间差
DEFAULT_TIME_WINDOW = 12.0
# 同一写法至少出现这么多次才认为是「可靠写法」
MIN_CORROBORATION = 2
# 相似度区间：低于 low 视为不同内容；高于 high 视为已一致，都无需替换
SIM_LOW = 0.55
SIM_HIGH = 0.985


@dataclass
class Correction:
    """一条校正记录。"""
    line_index: int
    time: float
    before: str
    after: str
    score: float
    evidence: list[str] = field(default_factory=list)
    source: str = "danmaku"     # danmaku | sc | gift


@dataclass
class CorrectionReport:
    applied: int = 0
    considered: int = 0
    corrections: list[Correction] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "considered": self.considered,
            "reason": self.reason,
            "corrections": [
                {"line_index": c.line_index, "time": round(c.time, 3),
                 "before": c.before, "after": c.after,
                 "score": round(c.score, 4), "source": c.source,
                 "evidence": c.evidence[:3]}
                for c in self.corrections
            ],
        }


# ---------------- 证据收集 ----------------

def _evidence_pool(log, window: float = DEFAULT_TIME_WINDOW
                   ) -> dict[float, list[tuple[str, str]]]:
    """按时间归档「可作为证据的文本」：(写法, 来源种类)。

    收集弹幕正文、SC 附言；礼物只取名称（主播常念礼物名）。
    """
    pool: defaultdict[float, list[tuple[str, str]]] = defaultdict(list)
    for d in log.danmaku:
        t = (d.text or "").strip()
        if len(norm_text(t)) >= MIN_DANMAKU_CHARS:
            pool[round(d.time / window) * window].append((t, "danmaku"))
    for g in log.gifts:
        if g.text:                      # SC 附言
            t = g.text.strip()
            if len(norm_text(t)) >= MIN_DANMAKU_CHARS:
                pool[round(g.time / window) * window].append((t, "sc"))
        if g.gift:
            name = g.gift.strip()
            if len(norm_text(name)) >= 2:
                pool[round(g.time / window) * window].append((name, "gift"))
    return pool


def _nearby(pool: dict[float, list[tuple[str, str]]], t: float,
            window: float) -> list[tuple[str, str]]:
    """取时间上接近的证据（前后各一个桶）。"""
    out: list[tuple[str, str]] = []
    for bucket in pool:
        if abs(bucket - t) <= window * 1.5:
            out.extend(pool[bucket])
    return out


def _global_phrases(log) -> Counter:
    """全片高频短语（昵称 / 礼物名 / 弹幕里反复出现的短词）。

    用于「同一写法出现多次才可信」的判断。
    """
    counter: Counter[str] = Counter()
    for d in log.danmaku:
        t = norm_text(d.text)
        if MIN_DANMAKU_CHARS <= len(t) <= 12:
            counter[t] += 1
        if d.user_name:
            counter[norm_text(d.user_name)] += 2
    for g in log.gifts:
        if g.gift:
            counter[norm_text(g.gift)] += 1
        if g.user_name:
            counter[norm_text(g.user_name)] += 2
    return counter


# ---------------- 校正主流程 ----------------

def _line_text(line) -> str:
    if isinstance(line, dict):
        return str(line.get("text", "") or "")
    return str(getattr(line, "text", "") or "")


def _line_time(line) -> float:
    if isinstance(line, dict):
        return float(line.get("start", 0) or 0)
    return float(getattr(line, "start", 0) or 0)


def _set_line_text(line, text: str):
    if isinstance(line, dict):
        d = dict(line)
        d["text"] = text
        words = d.get("words") or []
        if words:
            s = float(words[0].get("start", d.get("start", 0)))
            e = float(words[-1].get("end", d.get("end", s + 1)))
            d["words"] = [{"text": text, "start": s, "end": e}]
        return d
    from .models import SubtitleLine, SubtitleWord

    s = float(getattr(line, "start", 0.0))
    e = float(getattr(line, "end", s + 1.0))
    return SubtitleLine(
        words=[SubtitleWord(text=text, start=s, end=e)],
        speaker=getattr(line, "speaker", ""),
        low_confidence=getattr(line, "low_confidence", False),
        low_confidence_reason=getattr(line, "low_confidence_reason", ""),
    )


def _best_evidence_for(text: str, candidates: list[tuple[str, str]],
                       phrases: Counter) -> Optional[tuple[str, float, str]]:
    """在一行字幕的候选证据里找出最可信的一种写法。

    返回 (推荐写法, 相似度, 来源) 或 None。

    只处理「局部差异」：整体相似度落在 [SIM_LOW, SIM_HIGH) 才算
    「同一句话但写法有偏差」。完全一致（≥SIM_HIGH）无需替换；
    差的太远可能是无关弹幕，忽略。
    """
    nt = norm_text(text)
    if len(nt) < MIN_DANMAKU_CHARS:
        return None
    best: Optional[tuple[str, float, str]] = None
    for cand, kind in candidates:
        nc = norm_text(cand)
        if len(nc) < MIN_DANMAKU_CHARS:
            continue
        # 弹幕比字幕长很多时，取字幕长度附近的子串比较（观众常加语气词）
        if len(nc) > len(nt) * 1.8:
            nc = _closest_substring(nc, nt)
        if abs(len(nc) - len(nt)) > max(4, len(nt) // 2):
            continue
        score = difflib.SequenceMatcher(None, nt, nc).ratio()
        if not (SIM_LOW <= score < SIM_HIGH):
            continue
        # 该写法需要被反复印证（或来自用户昵称这类高可信来源）
        trust = phrases.get(nc, 0) >= MIN_CORROBORATION or kind in ("sc", "gift")
        if not trust:
            continue
        if best is None or score > best[1]:
            best = (cand, score, kind)
    return best


def _closest_substring(long_text: str, target: str) -> str:
    """在长文本里截取与 target 等长的最相似片段。"""
    n = len(target)
    if len(long_text) <= n:
        return long_text
    best, best_score = long_text[:n], -1.0
    for i in range(0, len(long_text) - n + 1):
        piece = long_text[i:i + n]
        s = difflib.SequenceMatcher(None, piece, target).ratio()
        if s > best_score:
            best, best_score = piece, s
    return best


def correct_with_log(lines: Iterable, log, *,
                     window: float = DEFAULT_TIME_WINDOW,
                     threshold: float = SIM_LOW,
                     max_changes_ratio: float = 0.5) -> tuple[list, CorrectionReport]:
    """用弹幕/礼物/SC 证据校正字幕文本（只改行内文字，不动时间轴）。

    `max_changes_ratio` 是安全阀：若替换行数超过总行数的一半，说明
    匹配逻辑可能失控（例如同质化的刷屏弹幕），此时整体放弃并说明原因。
    """
    src = list(lines)
    if not src:
        return src, CorrectionReport(reason="没有字幕行")
    if log is None:
        return src, CorrectionReport(reason="没有弹幕数据")

    pool = _evidence_pool(log, window=window)
    if not pool:
        return src, CorrectionReport(reason="弹幕中缺少可用文本")
    phrases = _global_phrases(log)

    out = list(src)
    report = CorrectionReport(considered=len(src))
    for i, line in enumerate(src):
        text = _line_text(line)
        if len(norm_text(text)) < MIN_DANMAKU_CHARS:
            continue
        cands = _nearby(pool, _line_time(line), window)
        if not cands:
            continue
        hit = _best_evidence_for(text, cands, phrases)
        if not hit:
            continue
        cand, score, kind = hit
        new_text = cand.strip()
        if norm_text(new_text) == norm_text(text):
            continue
        out[i] = _set_line_text(line, new_text)
        report.corrections.append(Correction(
            line_index=i, time=_line_time(line), before=text,
            after=new_text, score=score, source=kind,
            evidence=[c for c, _ in cands[:3]]))

    if report.corrections:
        # 安全阀：仅在**样本足够大**时才用比例判断。
        # 小样本（如十来行字幕）下「替换 1 行」也会超过 50%，
        # 按比例否决会误伤正常校正（实测踩过）。
        too_many = (len(src) >= 20
                    and len(report.corrections) > len(src) * max_changes_ratio)
        if too_many:
            return src, CorrectionReport(
                considered=len(src),
                reason=(f"拟替换 {len(report.corrections)} 行，超过总数 "
                        f"{max_changes_ratio:.0%}，判定为不可靠并放弃"))
    report.applied = len(report.corrections)
    report.reason = (f"已校正 {report.applied} 行"
                     if report.applied else "没有可校正的行")
    return out, report


# ---------------- 与歌曲清单联动 ----------------

def song_titles_from_log(log, library_titles: Iterable[str] = ()) -> list[str]:
    """从弹幕里推断本场可能演唱的歌：点歌请求 + 与曲库歌名的匹配。

    与 `livexml.song_requests()` 的区别：这里额外把**曲库已有的歌名**
    与弹幕做匹配，即使观众没有写「点一首」也能发现。
    """
    from .livexml import song_requests

    found: list[str] = []
    for _t, _u, name in song_requests(log):
        if name and name not in found:
            found.append(name)
    texts = [(d.text or "").strip() for d in log.danmaku]
    for title in library_titles:
        nt = norm_text(title)
        if not nt:
            continue
        for txt in texts:
            if nt and nt in norm_text(txt) and title not in found:
                found.append(title)
                break
    return found
