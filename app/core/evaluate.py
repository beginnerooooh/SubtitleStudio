"""字幕评测：把生成结果与人工校对样本（ground truth）逐项比对。

设计目标（对应产品需求「能够输出结果（不一致数量、匹配率等），并标注仍需
人工确认的边界情况」）：

1. 文本层：先做繁→简、全半角、空白归一，再按时间重叠把两侧行配对，
   计算字符错误率（CER）与逐行文本一致率；
2. 时间层：配对行的起止时间绝对误差、落在容差内的比例；
3. 行层：文本与时间同时达标的「精确匹配行」占比；
4. 边界情况：识别不出差异原因时（时间偏移、文本近似、漏行/多行、
   重叠、时长异常）逐条标注，输出可人工复核清单。

纯逻辑、零重依赖（zhconv 可选），可直接被 UI / CLI / 测试调用。
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# ---------------- 解析 ----------------

_SRT_TS = re.compile(
    r"(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
_VTT_TS = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})"
)


@dataclass
class EvalLine:
    """一条字幕（评测用最简表示）。"""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def parse_srt(content: str) -> list[EvalLine]:
    """SRT/VTT 文本 → EvalLine 列表（容忍 BOM、CRLF、缺序号、VTT 头）。"""
    text = content.lstrip("\ufeff")
    lines: list[EvalLine] = []
    # 以「空行 + 时间轴行」为块边界；缺空行时靠时间轴行切分
    blocks = re.split(r"\r?\n\s*\r?\n", text)
    for block in blocks:
        rows = [r for r in block.splitlines() if r.strip()]
        if not rows:
            continue
        idx = next(
            (i for i, r in enumerate(rows) if _SRT_TS.search(r) or _VTT_TS.search(r)),
            None,
        )
        if idx is None:
            continue
        m = _SRT_TS.search(rows[idx]) or _VTT_TS.search(rows[idx])
        g = [int(x) for x in m.groups()]
        # 毫秒字段按「字符串长度」换算（",083" 是 83ms 而非 830ms）
        gs = m.groups()
        start = (g[0] * 3600 + g[1] * 60 + g[2]
                 + int(gs[3]) / (10 ** len(gs[3])))
        end = (g[4] * 3600 + g[5] * 60 + g[6]
               + int(gs[7]) / (10 ** len(gs[7])))
        body = "\n".join(rows[idx + 1:]).strip()
        lines.append(EvalLine(start=round(start, 3), end=round(end, 3), text=body))
    lines.sort(key=lambda x: (x.start, x.end))
    return lines


def load_srt(path: str | Path) -> list[EvalLine]:
    return parse_srt(Path(path).read_text(encoding="utf-8-sig", errors="replace"))


# ---------------- 归一化 ----------------

# 归一化只丢「显示层面无意义」的差异：标点、空白、全半角、繁简。
_PUNCT_RE = re.compile(
    r"[\s\u3000,，、。.！!？?；;：:…·—\-_~～\"'“”‘’（）()\[\]【】《》〈〉<>/\\|@#$%^&*+=`]"
)


def normalize_text(text: str, *, simplify: bool = True) -> str:
    """文本归一：去标点空白、全角→半角、繁→简（zhconv 可用时）。"""
    t = text
    if simplify:
        try:
            from core.text import t2s

            t = t2s(t)
        except Exception:
            pass
    t = _PUNCT_RE.sub("", t)
    out = []
    for ch in t:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:      # 全角 ASCII → 半角
            ch = chr(code - 0xFEE0)
        elif code == 0x3000:
            ch = " "
        out.append(ch.lower())
    return "".join(out).strip()


def _chars(text: str) -> str:
    """纯字符序列（归一后已是无标点，直接返回）。"""
    return text


def levenshtein(a: str, b: str) -> int:
    """编辑距离（两行滚动数组，O(min(n,m)) 空间）。"""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """字符错误率；参考为空时：假设也为空记 0，否则记 1。"""
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def _lcs_len(a: str, b: str) -> int:
    """最长公共子序列长度（用于覆盖率：参考文本有多少字被正确命中）。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


# ---------------- 配对 ----------------

@dataclass
class LinePair:
    """一条参考行与其最佳候选行。"""

    ref: EvalLine
    hyp: Optional[EvalLine]
    start_err: float = 0.0
    end_err: float = 0.0
    text_sim: float = 0.0       # 1 - CER
    exact_text: bool = False
    exact_time: bool = False


@dataclass
class EvalConfig:
    """评测阈值（时间容差按毫秒计，与字幕工具通行的 ±0.3s 对齐标准一致）。"""

    start_tol: float = 0.30
    end_tol: float = 0.30
    text_sim_tol: float = 0.999   # 归一后逐字一致才算文本匹配


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def pair_lines(
    ref: list[EvalLine],
    hyp: list[EvalLine],
    cfg: Optional[EvalConfig] = None,
) -> tuple[list[LinePair], list[EvalLine], list[EvalLine]]:
    """按时间重叠把参考行与候选行一一配对（贪心，按重叠时长降序）。

    返回 (配对列表, 未匹配的参考行(漏行), 未匹配的候选行(多行))。
    """
    cfg = cfg or EvalConfig()
    refn = [normalize_text(r.text) for r in ref]
    hypn = [normalize_text(h.text) for h in hyp]

    cand: list[tuple[float, int, int]] = []
    for i, r in enumerate(ref):
        for j, h in enumerate(hyp):
            ov = _overlap(r.start, r.end, h.start, h.end)
            if ov > 0:
                cand.append((ov, i, j))
    cand.sort(key=lambda x: -x[0])

    used_r: set[int] = set()
    used_h: set[int] = set()
    pairs_map: dict[int, int] = {}
    for _ov, i, j in cand:
        if i in used_r or j in used_h:
            continue
        used_r.add(i)
        used_h.add(j)
        pairs_map[i] = j

    # 未被时间重叠配上的行：按文本相似度补齐（时间轴整体漂移时的兜底）
    rest_r = [i for i in range(len(ref)) if i not in used_r]
    rest_h = [j for j in range(len(hyp)) if j not in used_h]
    if rest_r and rest_h:
        for i in rest_r:
            best_j, best = None, 0.0
            for j in rest_h:
                if j in used_h:
                    continue
                ratio = difflib.SequenceMatcher(None, refn[i], hypn[j]).ratio()
                if ratio > best:
                    best, best_j = ratio, j
            if best_j is not None and best >= 0.6:
                pairs_map[i] = best_j
                used_r.add(i)
                used_h.add(best_j)

    pairs: list[LinePair] = []
    for i, r in enumerate(ref):
        j = pairs_map.get(i)
        if j is None:
            pairs.append(LinePair(ref=r, hyp=None, text_sim=0.0))
            continue
        h = hyp[j]
        sim = 1.0 - cer(refn[i], hypn[j])
        exact_text = refn[i] == hypn[j]
        start_err = abs(r.start - h.start)
        end_err = abs(r.end - h.end)
        pairs.append(LinePair(
            ref=r, hyp=h,
            start_err=round(start_err, 3), end_err=round(end_err, 3),
            text_sim=round(sim, 4), exact_text=exact_text,
            exact_time=(start_err <= cfg.start_tol and end_err <= cfg.end_tol),
        ))
    missing = [ref[i] for i in range(len(ref)) if i not in pairs_map]
    extra = [hyp[j] for j in range(len(hyp)) if j not in used_h]
    return pairs, missing, extra


# ---------------- 汇总 ----------------

@dataclass
class EvalReport:
    """评测汇总（可直接序列化为 JSON 供 UI 展示）。"""

    ref_count: int
    hyp_count: int
    paired: int
    missing: int
    extra: int
    exact_text_lines: int
    exact_time_lines: int
    exact_both_lines: int
    text_match_rate: float          # 归一后逐字一致的行 / 参考行数
    time_match_rate: float
    exact_match_rate: float
    cer: float                      # 全文 CER（按全文拼接）
    char_recall: float              # 参考字符被正确命中的比例（LCS 覆盖）
    ma_start: float                 # 配对行起始时间平均绝对误差（秒）
    ma_end: float
    p90_start: float
    boundary_cases: list[dict] = field(default_factory=list)
    worst_lines: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ref_count": self.ref_count, "hyp_count": self.hyp_count,
            "paired": self.paired, "missing": self.missing, "extra": self.extra,
            "exact_text_lines": self.exact_text_lines,
            "exact_time_lines": self.exact_time_lines,
            "exact_both_lines": self.exact_both_lines,
            "text_match_rate": round(self.text_match_rate, 4),
            "time_match_rate": round(self.time_match_rate, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "cer": round(self.cer, 4),
            "char_recall": round(self.char_recall, 4),
            "ma_start_sec": round(self.ma_start, 4),
            "ma_end_sec": round(self.ma_end, 4),
            "p90_start_sec": round(self.p90_start, 4),
            "boundary_cases": self.boundary_cases,
            "worst_lines": self.worst_lines,
        }

    def summary_md(self, title: str = "评测报告") -> str:
        rows = [
            f"# {title}", "",
            "| 指标 | 数值 |", "|---|---:|",
            f"| 参考行数（人工校对） | {self.ref_count} |",
            f"| 生成行数 | {self.hyp_count} |",
            f"| 成功配对 | {self.paired} |",
            f"| 漏行（参考有、生成无） | {self.missing} |",
            f"| 多行（生成有、参考无） | {self.extra} |",
            f"| **文本逐字一致行** | {self.exact_text_lines}（{self.text_match_rate:.1%}） |",
            f"| **时间达标行** | {self.exact_time_lines}（{self.time_match_rate:.1%}） |",
            f"| **完全匹配行（文本+时间）** | {self.exact_both_lines}"
            f"（{self.exact_match_rate:.1%}） |",
            f"| 全文 CER | {self.cer:.2%} |",
            f"| 字符召回率（LCS） | {self.char_recall:.2%} |",
            f"| 起始时间平均误差 | {self.ma_start:.3f}s |",
            f"| 结束时间平均误差 | {self.ma_end:.3f}s |",
            f"| 起始时间 P90 误差 | {self.p90_start:.3f}s |",
            "",
        ]
        if self.boundary_cases:
            rows += ["## 仍需人工确认的边界情况", "",
                     "| # | 类型 | 参考时间 | 生成时间 | 参考文本 | 生成文本 | 说明 |",
                     "|---|---|---|---|---|---|---|"]
            for i, c in enumerate(self.boundary_cases, 1):
                rows.append(
                    f"| {i} | {c['kind']} | {c['ref_time']} | {c['hyp_time']} | "
                    f"{c['ref_text']} | {c['hyp_text']} | {c['note']} |")
            rows.append("")
        if self.worst_lines:
            rows += ["## 误差最大的行（文本）", "",
                     "| 参考 | 生成 | 相似度 | 时间误差 |", "|---|---|---:|---:|"]
            for c in self.worst_lines:
                rows.append(f"| {c['ref_text']} | {c['hyp_text']} | "
                            f"{c['sim']:.2f} | {c['time_err']:.2f}s |")
            rows.append("")
        return "\n".join(rows)


def _fmt_t(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:06.3f}"


def _classify(pair: LinePair, cfg: EvalConfig) -> Optional[dict]:
    """边界情况分类：时间偏移 / 文本近似 / 时间+文本均不符。"""
    r, h = pair.ref, pair.hyp
    if h is None:
        return {"kind": "漏行", "ref_time": f"{_fmt_t(r.start)}–{_fmt_t(r.end)}",
                "hyp_time": "-", "ref_text": r.text, "hyp_text": "",
                "note": "参考行未找到时间重叠的候选"}
    kinds = []
    note = []
    if not pair.exact_text:
        if pair.text_sim >= 0.6:
            kinds.append("文本近似")
            note.append(f"相似度 {pair.text_sim:.2f}，需人工确认是否同义或错字")
        else:
            kinds.append("文本不符")
            note.append(f"相似度 {pair.text_sim:.2f}，建议人工重听")
    if not pair.exact_time:
        if pair.start_err > cfg.start_tol or pair.end_err > cfg.end_tol:
            kinds.append("时间偏移")
            note.append(f"起 {pair.start_err:+.3f}s / 止 {pair.end_err:+.3f}s")
    if not kinds:
        return None
    return {
        "kind": " + ".join(kinds),
        "ref_time": f"{_fmt_t(r.start)}–{_fmt_t(r.end)}",
        "hyp_time": f"{_fmt_t(h.start)}–{_fmt_t(h.end)}",
        "ref_text": r.text, "hyp_text": h.text,
        "note": "；".join(note),
    }


def evaluate(
    ref: list[EvalLine],
    hyp: list[EvalLine],
    cfg: Optional[EvalConfig] = None,
    *,
    max_boundary_cases: int = 200,
    max_worst: int = 30,
) -> EvalReport:
    """完整评测：配对 → 指标 → 边界清单。"""
    cfg = cfg or EvalConfig()
    pairs, missing, extra = pair_lines(ref, hyp, cfg)

    ref_text = "".join(normalize_text(r.text) for r in ref)
    hyp_text = "".join(normalize_text(h.text) for h in hyp)
    total_cer = cer(ref_text, hyp_text)
    recall = _lcs_len(ref_text, hyp_text) / len(ref_text) if ref_text else 0.0

    exact_text = sum(1 for p in pairs if p.exact_text)
    exact_time = sum(1 for p in pairs if p.hyp is not None and p.exact_time)
    exact_both = sum(1 for p in pairs
                     if p.hyp is not None and p.exact_text and p.exact_time)
    paired = sum(1 for p in pairs if p.hyp is not None)
    starts = sorted(p.start_err for p in pairs if p.hyp is not None)
    ends = [p.end_err for p in pairs if p.hyp is not None]
    p90 = starts[int(0.9 * (len(starts) - 1))] if starts else 0.0

    boundary: list[dict] = []
    for p in pairs:
        c = _classify(p, cfg)
        if c is not None:
            boundary.append(c)
        if len(boundary) >= max_boundary_cases:
            break
    for e in extra[:max(0, max_boundary_cases - len(boundary))]:
        boundary.append({
            "kind": "多行", "ref_time": "-",
            "hyp_time": f"{_fmt_t(e.start)}–{_fmt_t(e.end)}",
            "ref_text": "", "hyp_text": e.text,
            "note": "生成行无对应参考（可能是静音/噪声幻听）",
        })

    worst = sorted(
        (p for p in pairs if p.hyp is not None and not p.exact_text),
        key=lambda p: p.text_sim,
    )[:max_worst]
    worst_rows = [{
        "ref_text": p.ref.text, "hyp_text": p.hyp.text,
        "sim": p.text_sim,
        "time_err": max(p.start_err, p.end_err),
    } for p in worst]

    n = max(1, len(ref))
    return EvalReport(
        ref_count=len(ref), hyp_count=len(hyp), paired=paired,
        missing=len(missing), extra=len(extra),
        exact_text_lines=exact_text, exact_time_lines=exact_time,
        exact_both_lines=exact_both,
        text_match_rate=exact_text / n,
        time_match_rate=exact_time / n,
        exact_match_rate=exact_both / n,
        cer=total_cer, char_recall=recall,
        ma_start=sum(starts) / len(starts) if starts else 0.0,
        ma_end=sum(ends) / len(ends) if ends else 0.0,
        p90_start=p90,
        boundary_cases=boundary,
        worst_lines=worst_rows,
    )


def evaluate_files(
    ref_path: str | Path,
    hyp_path: str | Path,
    out_dir: Optional[str | Path] = None,
    *,
    title: str = "评测报告",
    max_time: Optional[float] = None,
) -> EvalReport:
    """文件级评测；out_dir 不为空时写出 report.md / report.json。

    max_time：只比较该时间之前的参考行（评测片段文件时使用，避免把
    「参考里存在但候选文件根本没覆盖到的时间段」算成漏行）。
    """
    ref = load_srt(ref_path)
    if max_time is not None:
        ref = [r for r in ref if r.start < max_time]
    report = evaluate(ref, load_srt(hyp_path))
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
        (out / "report.md").write_text(
            report.summary_md(title), encoding="utf-8")
    return report
