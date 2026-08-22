"""SRT / LRC / ASS 字幕生成（纯函数，无模型依赖）。

时间约定：内部一律用秒（float）；SRT 毫秒精度，LRC/ASS 厘秒精度。
四舍五入统一用「+0.5 取整」避免银行家舍入的意外。
"""
from __future__ import annotations

from core.models import SubtitleLine, SubtitleWord

# 句末标点：聚合时触发切分
_SENTENCE_END = tuple("。！？!?；;…")


def _ms(seconds: float) -> int:
    return max(0, int(seconds * 1000 + 0.5))


def _cs(seconds: float) -> int:
    return max(0, int(seconds * 100 + 0.5))


def format_srt_time(seconds: float) -> str:
    """秒 → SRT 时间 HH:MM:SS,mmm。"""
    total = _ms(seconds)
    h, rem = divmod(total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_lrc_time(seconds: float) -> str:
    """秒 → LRC 时间 mm:ss.xx（厘秒）。"""
    total = _cs(seconds)
    m, rem = divmod(total, 6000)
    s, cs = divmod(rem, 100)
    return f"{m:02d}:{s:02d}.{cs:02d}"


def format_ass_time(seconds: float) -> str:
    """秒 → ASS 时间 H:MM:SS.cc（厘秒）。"""
    total = _cs(seconds)
    h, rem = divmod(total, 360_000)
    m, rem = divmod(rem, 6_000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def aggregate_words(
    words: list[SubtitleWord],
    max_duration: float = 5.0,
    max_chars: int = 25,
) -> list[SubtitleLine]:
    """词流 → 字幕行：句末标点 / 最大时长 / 最大字符数任一触发切分。"""
    lines: list[SubtitleLine] = []
    current: list[SubtitleWord] = []

    def flush() -> None:
        nonlocal current
        if current:
            lines.append(SubtitleLine(words=current))
            current = []

    for word in words:
        if current:
            span = word.end - current[0].start
            chars = len("".join(x.text for x in current)) + len(word.text)
            if (
                span > max_duration
                or chars > max_chars
                or current[-1].text.endswith(_SENTENCE_END)
            ):
                flush()
        current.append(word)
    flush()
    return lines


def to_srt(lines: list[SubtitleLine]) -> str:
    """SubtitleLine[] → SRT 字符串。"""
    blocks: list[str] = []
    n = 0
    for ln in lines:
        if not ln.words:
            continue
        n += 1
        blocks.append(
            f"{n}\n{format_srt_time(ln.start)} --> {format_srt_time(ln.end)}\n{ln.text}"
        )
    return "\n\n".join(blocks) + "\n" if blocks else ""


def to_lrc(lines: list[SubtitleLine], title: str = "") -> str:
    """SubtitleLine[] → LRC 字符串（行级时间 + 元标签）；无有效行返回空串。"""
    body: list[str] = []
    for ln in lines:
        if not ln.words:
            continue
        body.append(f"[{format_lrc_time(ln.start)}]{ln.text}")
    if not body:
        return ""
    parts: list[str] = []
    if title:
        parts.append(f"[ti:{title}]")
    parts.append("[re:Subtitle Studio]")
    return "\n".join(parts + body) + "\n"


_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,90,&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,0,2,20,20,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def to_ass(
    lines: list[SubtitleLine],
    max_extension: float = 1.0,
    extension_gap: float = 0.05,
) -> str:
    """SubtitleLine[] → ASS（逐字 \\k 卡拉OK）字符串。

    末字延音：与下一行间隙 ≤ max_extension 时，行显示结束时间延伸到
    下一行开始前 extension_gap 秒，末字 \\k 覆盖到行尾。
    """
    events: list[str] = []
    for i, ln in enumerate(lines):
        if not ln.words:
            continue
        line_end = ln.end
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if nxt is not None and nxt.words:
            gap = nxt.start - line_end
            if 0 < gap <= max_extension:
                line_end = nxt.start - extension_gap
        text_parts: list[str] = []
        for j, word in enumerate(ln.words):
            k_end = line_end if j == len(ln.words) - 1 else word.end
            dur_cs = max(1, _cs(k_end) - _cs(word.start))
            text_parts.append(f"{{\\k{dur_cs}}}{word.text}")
        events.append(
            f"Dialogue: 0,{format_ass_time(ln.start)},{format_ass_time(line_end)},"
            f"Default,,0,0,0,,{''.join(text_parts)}"
        )
    if not events:
        return ""
    return _ASS_HEADER + "\n".join(events) + "\n"
