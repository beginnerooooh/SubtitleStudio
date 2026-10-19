"""字幕编辑：切分 / 合并 / 增删 / 改字 / 时间平移（纯函数，供 UI 与 CLI 复用）。

设计要点：
- 所有操作都以 SubtitleLine 列表为输入输出，**不修改入参**（返回新列表），
  便于 UI 撤销（保留历史快照即可）；
- 切分点以「字符位置」表述（用户看到的是文本），内部按字符重建词时间；
- 时间编辑接受毫秒/秒两种输入，自动夹紧到相邻行边界内，避免出现重叠；
- 每步操作都返回 (新行列表, 提示语)，提示语直接显示给用户。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.models import SubtitleLine, SubtitleWord

MIN_LINE = 0.10       # 行最小时长（秒）
MIN_GAP = 0.01        # 相邻行最小间隙


# ---------------- 查询 / 序列化 ----------------

@dataclass
class EditRow:
    """编辑器表格的一行。"""

    index: int
    start: float
    end: float
    text: str
    speaker: str = ""


def to_rows(lines: list[SubtitleLine]) -> list[list]:
    """SubtitleLine[] → 表格行 [[序号, 开始, 结束, 说话人, 文本], ...]。"""
    return [
        [i + 1, fmt_ts(ln.start), fmt_ts(ln.end), ln.speaker, ln.text]
        for i, ln in enumerate(lines)
    ]


def fmt_ts(t: float) -> str:
    """秒 → MM:SS.mmm（UI 显示与输入通用）。"""
    t = max(0.0, t)
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:06.3f}"


def parse_ts(text: str) -> float:
    """'MM:SS.mmm' / 'HH:MM:SS,mmm' / 纯秒数 → 秒。非法抛 ValueError。"""
    s = (text or "").strip().replace(",", ".")
    if not s:
        raise ValueError("时间不能为空")
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError as exc:
        raise ValueError(f"时间格式无法解析：{text}") from exc
    raise ValueError(f"时间格式无法解析：{text}")


# ---------------- 结构编辑 ----------------

def split_line(lines: list[SubtitleLine], index: int, char_pos: int,
               ) -> tuple[list[SubtitleLine], str]:
    """在第 index 行的第 char_pos 个字符前切分（按时间比例分配）。"""
    if not (0 <= index < len(lines)):
        return lines, "行号超出范围"
    ln = lines[index]
    text = ln.text
    if not (1 <= char_pos < len(text)):
        return lines, f"切分位置需在 1~{len(text) - 1} 之间"
    total = max(1e-6, ln.end - ln.start)
    ratio = char_pos / max(1, len(text))
    cut = round(ln.start + total * ratio, 3)
    left = SubtitleLine(words=[SubtitleWord(text=text[:char_pos],
                                            start=ln.start, end=cut)],
                        speaker=ln.speaker)
    right = SubtitleLine(words=[SubtitleWord(text=text[char_pos:],
                                             start=cut, end=ln.end)],
                         speaker=ln.speaker)
    out = list(lines)
    out[index:index + 1] = [left, right]
    return out, f"已在第 {char_pos} 字处切分"


def merge_lines(lines: list[SubtitleLine], indices: list[int],
                ) -> tuple[list[SubtitleLine], str]:
    """合并选中的若干行（按出现顺序拼文本，时间取外沿）。"""
    idx = sorted({i for i in indices if 0 <= i < len(lines)})
    if len(idx) < 2:
        return lines, "请至少选择两行"
    if idx != list(range(idx[0], idx[-1] + 1)):
        return lines, "只能合并连续的行"
    chunk = [lines[i] for i in idx]
    text = "".join(c.text for c in chunk)
    merged = SubtitleLine(
        words=[SubtitleWord(text=text, start=chunk[0].start,
                            end=chunk[-1].end)],
        speaker=chunk[0].speaker)
    out = list(lines)
    out[idx[0]:idx[-1] + 1] = [merged]
    return out, f"已合并 {len(idx)} 行"


def delete_lines(lines: list[SubtitleLine], indices: list[int],
                 ) -> tuple[list[SubtitleLine], str]:
    """删除选中的行。"""
    idx = {i for i in indices if 0 <= i < len(lines)}
    if not idx:
        return lines, "请先选择要删除的行"
    out = [ln for i, ln in enumerate(lines) if i not in idx]
    return out, f"已删除 {len(idx)} 行"


def insert_line(lines: list[SubtitleLine], index: int, text: str,
                start: float, end: float,
                ) -> tuple[list[SubtitleLine], str]:
    """在 index 处插入一行（index = 行数表示追加到末尾）。"""
    if end <= start:
        return lines, "结束时间必须晚于开始时间"
    line = SubtitleLine(words=[SubtitleWord(text=text.strip(), start=start,
                                            end=end)])
    out = list(lines)
    out.insert(max(0, min(len(out), index)), line)
    return out, "已插入新行"


def update_line(lines: list[SubtitleLine], index: int, *,
                text: str | None = None, start: float | None = None,
                end: float | None = None,
                ) -> tuple[list[SubtitleLine], str]:
    """修改单行的文本/时间（时间自动夹紧，避免与邻行重叠）。"""
    if not (0 <= index < len(lines)):
        return lines, "行号超出范围"
    out = list(lines)
    ln = out[index]
    w = ln.words[0] if ln.words else SubtitleWord(text="", start=0.0, end=0.0)
    new_text = w.text if text is None else text
    new_start = w.start if start is None else start
    new_end = w.end if end is None else end
    if new_end <= new_start:
        new_end = new_start + MIN_LINE
    # 夹紧到相邻行之间
    if index > 0 and out[index - 1].words:
        new_start = max(new_start, out[index - 1].end + MIN_GAP)
    if index + 1 < len(out) and out[index + 1].words:
        new_end = min(new_end, out[index + 1].start - MIN_GAP)
    if new_end <= new_start:
        new_end = new_start + MIN_LINE
    out[index] = SubtitleLine(
        words=[SubtitleWord(text=new_text, start=round(new_start, 3),
                            end=round(new_end, 3))],
        speaker=ln.speaker, low_confidence=ln.low_confidence,
        low_confidence_reason=ln.low_confidence_reason)
    return out, "已更新"


def shift_lines(lines: list[SubtitleLine], delta: float, index: int | None = None
                ) -> tuple[list[SubtitleLine], str]:
    """整体（或单行）时间平移 delta 秒。"""
    out = []
    for i, ln in enumerate(lines):
        if index is not None and i != index:
            out.append(ln)
            continue
        words = [SubtitleWord(text=w.text,
                             start=round(max(0.0, w.start + delta), 3),
                             end=round(max(0.0, w.end + delta), 3))
                 for w in ln.words]
        out.append(SubtitleLine(words=words, speaker=ln.speaker,
                                low_confidence=ln.low_confidence,
                                low_confidence_reason=ln.low_confidence_reason))
    scope = "整片" if index is None else f"第 {index + 1} 行"
    return out, f"{scope}时间平移 {delta:+.3f}s"


def retime_by_ratio(lines: list[SubtitleLine], factor: float,
                    ) -> tuple[list[SubtitleLine], str]:
    """按倍率拉伸/压缩整条时间轴（对齐外部参考用）。"""
    if factor <= 0:
        return lines, "倍率必须大于 0"
    out = []
    for ln in lines:
        words = [SubtitleWord(text=w.text, start=round(w.start * factor, 3),
                              end=round(w.end * factor, 3)) for w in ln.words]
        out.append(SubtitleLine(words=words, speaker=ln.speaker,
                                low_confidence=ln.low_confidence))
    return out, f"时间轴×{factor:g}"


def split_long_lines(lines: list[SubtitleLine], max_chars: int = 18,
                     ) -> tuple[list[SubtitleLine], str]:
    """按字符数把过长行自动切分（导出前 QC）。"""
    out: list[SubtitleLine] = []
    n = 0
    for ln in lines:
        text = ln.text
        if len(text) <= max_chars:
            out.append(ln)
            continue
        parts = _split_text(text, max_chars)
        total = len(text)
        cur = ln.start
        span = max(0.2, ln.end - ln.start)
        for part in parts:
            dur = span * len(part) / total
            out.append(SubtitleLine(words=[SubtitleWord(
                text=part, start=round(cur, 3), end=round(cur + dur, 3))],
                speaker=ln.speaker))
            cur += dur
            n += 1
    return out, f"已切分 {n} 处超长行"


def _split_text(text: str, limit: int) -> list[str]:
    """在标点/空格处优先切分，退化时硬切。"""
    parts, buf = [], ""
    for ch in text:
        buf += ch
        if len(buf) >= limit * 0.6 and ch in "，。！？；、,.!?; ":
            parts.append(buf.strip())
            buf = ""
        elif len(buf) >= limit:
            parts.append(buf)
            buf = ""
    if buf.strip():
        parts.append(buf)
    return [p for p in parts if p]


def fix_overlaps(lines: list[SubtitleLine], min_gap: float = 0.02,
                 ) -> tuple[list[SubtitleLine], str]:
    """消除时间重叠（QC）：前行截到后行起点之前。"""
    n = 0
    for i in range(len(lines) - 1):
        a, b = lines[i], lines[i + 1]
        if a.words and b.words and a.words[-1].end > b.words[0].start - min_gap:
            a.words[-1].end = round(max(a.words[0].start + MIN_LINE,
                                        b.words[0].start - min_gap), 3)
            n += 1
    return lines, f"修正 {n} 处重叠"


def retime_edge(lines: list[SubtitleLine], index: int, *,
                start: float | None = None, end: float | None = None,
                min_dur: float = 0.2) -> tuple[list[SubtitleLine], str]:
    """拖动行边界：只改起点或终点（时间轴左右拖拽用）。

    与 update_line 的区别：这里**不做相邻行夹紧**——剪辑软件里拖动
    允许短暂越界，越界由「自动吸附/碰撞处理」另行决定；夹紧会让拖拽
    手感发滞（实测拖动时值被立刻改回去）。
    """
    if not (0 <= index < len(lines)):
        return lines, "行号超出范围"
    out = list(lines)
    ln = out[index]
    if not ln.words:
        return lines, "该行没有内容"
    s = ln.start if start is None else float(start)
    e = ln.end if end is None else float(end)
    s = max(0.0, s)
    if e - s < min_dur:
        if start is not None:
            s = max(0.0, e - min_dur)
        else:
            e = s + min_dur
    first, last = ln.words[0], ln.words[-1]
    span_old = max(1e-6, last.end - first.start)
    span_new = e - s
    scale = span_new / span_old if span_old > 0 else 1.0
    words = [SubtitleWord(text=w.text,
                          start=round(s + (w.start - first.start) * scale, 3),
                          end=round(s + (w.end - first.start) * scale, 3))
             for w in ln.words]
    out[index] = SubtitleLine(words=words, speaker=ln.speaker,
                              low_confidence=ln.low_confidence,
                              low_confidence_reason=ln.low_confidence_reason)
    which = "起点" if start is not None else "终点"
    return out, f"已调整第 {index + 1} 行{which}"


def copy_lines(lines: list[SubtitleLine], indices: list[int]
               ) -> tuple[list[dict], str]:
    """复制选中行（返回可序列化的剪贴板数据）。"""
    idx = sorted({i for i in indices if 0 <= i < len(lines)})
    if not idx:
        return [], "请先选择要复制的行"
    data = []
    for i in idx:
        ln = lines[i]
        data.append({
            "start": ln.start, "end": ln.end, "text": ln.text,
            "speaker": ln.speaker,
            "words": [{"text": w.text, "start": w.start, "end": w.end}
                      for w in ln.words],
        })
    return data, f"已复制 {len(idx)} 行"


def paste_lines(lines: list[SubtitleLine], clipboard: list[dict],
                at: float | None = None, index: int | None = None,
                ) -> tuple[list[SubtitleLine], str]:
    """粘贴剪贴板内容。

    at 非空时以该时间点为首行起点（保持行间相对间隔）——对应剪辑软件
    「在播放头处粘贴」；否则插到 index 位置（默认末尾）。
    """
    if not clipboard:
        return lines, "剪贴板为空"
    # 以剪贴板中最早的起点为基准，整体平移到目标时间点
    starts = []
    for c in clipboard:
        try:
            starts.append(float(c.get("start", 0.0)))
        except (TypeError, ValueError):
            continue
    if at is not None and not starts:
        return lines, "剪贴板内容无效"
    base = min(starts) if starts else 0.0
    shift = (float(at) - base) if at is not None else 0.0
    new_lines: list[SubtitleLine] = []
    for c in clipboard:
        try:
            s = float(c.get("start", 0.0)) + shift
            e = float(c.get("end", s + 1.0)) + shift
        except (TypeError, ValueError):
            continue
        if e <= s:
            e = s + 0.5
        words = [SubtitleWord(text=w.get("text", ""),
                              start=float(w.get("start", s)) + shift,
                              end=float(w.get("end", e)) + shift)
                 for w in c.get("words", []) if w.get("text")]
        if not words:
            words = [SubtitleWord(text=c.get("text", ""), start=s, end=e)]
        if not words[0].text:
            continue
        new_lines.append(SubtitleLine(words=words,
                                      speaker=c.get("speaker", "")))
    if not new_lines:
        return lines, "剪贴板内容无效"
    out = list(lines)
    pos = len(out) if index is None else max(0, min(len(out), int(index)))
    out[pos:pos] = new_lines
    # 粘贴后按时间重排，保证时间轴单调（粘贴到中间时段时必需）
    out.sort(key=lambda ln: ln.start)
    return out, f"已粘贴 {len(new_lines)} 行"


def duplicate_lines(lines: list[SubtitleLine], indices: list[int],
                    ) -> tuple[list[SubtitleLine], str]:
    """原地复制选中行（紧跟在原行之后）。

    副本时间**顺延到原行结束之后**（保持原时长），否则两条字幕会完全
    重叠、时间轴上分不清（实测直接拷贝时间会导致重叠）。
    """
    idx = sorted({i for i in indices if 0 <= i < len(lines)})
    if not idx:
        return lines, "请先选择要复制的行"
    out = list(lines)
    added = 0
    for offset, i in enumerate(idx):
        src = out[i + offset]
        dur = max(0.2, src.end - src.start)
        new_start = round(src.end, 3)
        new_end = round(new_start + dur, 3)
        out.insert(i + offset + 1, SubtitleLine(
            words=[SubtitleWord(text=src.text, start=new_start, end=new_end)],
            speaker=src.speaker, low_confidence=src.low_confidence))
        added += 1
    return out, f"已复制 {added} 行到原位置之后"


def stats(lines: list[SubtitleLine]) -> str:
    """编辑器状态摘要（行数/时长/异常提示）。"""
    if not lines:
        return "暂无字幕行"
    dur = [ln.end - ln.start for ln in lines]
    chars = [len(ln.text) for ln in lines]
    short = sum(1 for d in dur if d < 0.4)
    long_ = sum(1 for c in chars if c > 20)
    overlap = sum(1 for i in range(len(lines) - 1)
                  if lines[i].words and lines[i + 1].words
                  and lines[i].end > lines[i + 1].start)
    return (f"共 {len(lines)} 行 · 平均 {sum(dur)/len(dur):.2f}s / "
            f"{sum(chars)/len(chars):.1f} 字"
            + (f" · 过短 {short}" if short else "")
            + (f" · 超长 {long_}" if long_ else "")
            + (f" · 重叠 {overlap}" if overlap else ""))


__all__ = [
    "EditRow", "to_rows", "fmt_ts", "parse_ts", "split_line", "merge_lines",
    "delete_lines", "insert_line", "update_line", "shift_lines",
    "retime_by_ratio", "split_long_lines", "fix_overlaps", "stats",
    "retime_edge", "copy_lines", "paste_lines", "duplicate_lines",
]
