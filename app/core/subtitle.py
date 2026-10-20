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


# 低置信度行前缀：纯文本格式（SRT/LRC）的复核标记
_LOW_CONF_MARK = "\u26a0 "


def _line_text(ln: SubtitleLine, speaker_labels: bool,
               mark_low_confidence: bool = False) -> str:
    """行的显示文本。

    speaker_labels：加「[名字] 」前缀；
    mark_low_confidence：低置信度行加「⚠ 」前缀（供人工复核定位）。
    """
    prefix = ""
    if mark_low_confidence and ln.low_confidence:
        prefix += _LOW_CONF_MARK
    if speaker_labels and ln.speaker:
        prefix += f"[{ln.speaker}] "
    return prefix + ln.text


def to_srt(lines: list[SubtitleLine], speaker_labels: bool = False,
           mark_low_confidence: bool = False) -> str:
    """SubtitleLine[] → SRT 字符串。

    speaker_labels：行首标注 [说话人]；
    mark_low_confidence：低置信度行加 ⚠ 前缀。
    """
    blocks: list[str] = []
    n = 0
    for ln in lines:
        if not ln.words:
            continue
        n += 1
        blocks.append(
            f"{n}\n{format_srt_time(ln.start)} --> {format_srt_time(ln.end)}\n"
            f"{_line_text(ln, speaker_labels, mark_low_confidence)}"
        )
    return "\n\n".join(blocks) + "\n" if blocks else ""


def to_lrc(lines: list[SubtitleLine], title: str = "",
            speaker_labels: bool = False,
            mark_low_confidence: bool = False) -> str:
    """SubtitleLine[] → LRC 字符串（行级时间 + 元标签）；无有效行返回空串。"""
    body: list[str] = []
    for ln in lines:
        if not ln.words:
            continue
        body.append(
            f"[{format_lrc_time(ln.start)}]"
            f"{_line_text(ln, speaker_labels, mark_low_confidence)}")
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
Style: LowConf,Microsoft YaHei,90,&H0000A5FF,&H0000FFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,0,2,20,20,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_color(hex_color: str) -> str:
    """前端用 `#RRGGBB`，ASS 用 `&HAABBGGRR`（BGR 顺序，AA=00 不透明）。

    这是烧录样式与预览一致的关键一步：颜色通道顺序不同（RGB vs BGR），
    直接套用会让红色变蓝（实测踩过）。
    """
    s = str(hex_color or "").strip().lstrip("#")
    if len(s) != 6:
        return "&H00FFFFFF"
    try:
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return "&H00FFFFFF"
    return f"&H00{b:02X}{g:02X}{r:02X}"


# 前端 anchor → ASS Alignment（小键盘布局：1/4/7 左下/左中/左上，2/5/8 居中）
_ASS_ALIGN = {"bottom": 2, "middle": 5, "top": 8}


def to_ass(
    lines: list[SubtitleLine],
    max_extension: float = 1.0,
    extension_gap: float = 0.05,
    speaker_labels: bool = False,
    mark_low_confidence: bool = False,
    style: dict | None = None,
) -> str:
    """SubtitleLine[] → ASS（逐字 \\k 卡拉OK）字符串。

    末字延音：与下一行间隙 ≤ max_extension 时，行显示结束时间延伸到
    下一行开始前 extension_gap 秒，末字 \\k 覆盖到行尾。
    speaker_labels：说话人写入 ASS Name 字段（播放器演员名轨）；
    mark_low_confidence：低置信度行走 LowConf 样式（橙色主色，播放中
    一眼可辨），卡拉OK节奏不受影响。

    `style`：与前端预览一致的样式字典（size/color/outline/anchor/x/y/bg）。
    给出时会**覆盖**默认样式，使烧录出的视频与预览所见相同。
    行级覆盖（`SubtitleLine.style`）优先于这里的全局样式。
    """
    st = dict(style or {})
    base_style = _build_ass_style("Default", st, fallback_color="#FFFFFF",
                                 fallback_size=90, fallback_outline=3,
                                 fallback_align=2)
    low_style = _build_ass_style("LowConf", st, fallback_color="#FFA500",
                                 fallback_size=90, fallback_outline=3,
                                 fallback_align=2)
    header = _ASS_HEADER.replace(
        "Style: Default,Microsoft YaHei,90,&H00FFFFFF,&H0000FFFF,"
        "&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,0,2,20,20,60,1",
        base_style
    ).replace(
        "Style: LowConf,Microsoft YaHei,90,&H0000A5FF,&H0000FFFF,"
        "&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,0,2,20,20,60,1",
        low_style
    )

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
        actor = ln.speaker if speaker_labels and ln.speaker else ""
        # 行级样式优先：该行有覆盖时用覆盖值内联，实现「只给某几行单独设置」
        line_override = dict(getattr(ln, "style", {}) or {})
        inline = ""
        if line_override:
            inline = _ass_inline_override(line_override, st)
        style = ("LowConf" if mark_low_confidence and ln.low_confidence
                 else "Default")
        events.append(
            f"Dialogue: 0,{format_ass_time(ln.start)},{format_ass_time(line_end)},"
            f"{style},{actor},0,0,0,,{inline}{''.join(text_parts)}"
        )
    if not events:
        return ""
    return header + "\n".join(events) + "\n"


def _build_ass_style(name: str, style: dict, *, fallback_color: str,
                     fallback_size: int, fallback_outline: float,
                     fallback_align: int) -> str:
    """构造一行 ASS Style 定义（与预览样式对应）。"""
    size = int(style.get("size") or fallback_size)
    color = _ass_color(style.get("color") or fallback_color)
    outline = style.get("outline")
    outline = fallback_outline if outline is None else float(outline)
    align = _ASS_ALIGN.get(str(style.get("anchor") or ""), fallback_align)
    # bg=True 时用不透明背景框（BorderStyle=3），否则纯描边（=1）
    border_style = 3 if style.get("bg") else 1
    back = "&H80000000" if style.get("bg") else "&H64000000"
    # MarginV：底部留白 60px（与预览 bottom:7% 接近）
    margin_v = 60
    return (f"Style: {name},Microsoft YaHei,{size},{color},&H0000FFFF,"
            f"&H00000000,{back},0,0,0,0,100,100,0,0,{border_style},"
            f"{outline:g},0,{align},20,20,{margin_v},1")


def _ass_inline_override(line_style: dict, global_style: dict) -> str:
    """行级覆盖 → ASS 内联标签（\\fs 字号 / \\c&H..& 颜色 / \\an 对齐）。

    只输出与全局不同的项，保持标签串最短。
    """
    tags: list[str] = []
    if line_style.get("size") and line_style["size"] != global_style.get("size"):
        tags.append(f"\\fs{int(line_style['size'])}")
    if line_style.get("color") and line_style["color"] != global_style.get("color"):
        tags.append(f"\\c{_ass_color(line_style['color'])}")
    anchor = line_style.get("anchor")
    if anchor and anchor != global_style.get("anchor"):
        tags.append(f"\\an{_ASS_ALIGN.get(str(anchor), 2)}")
    # 自由定位：\pos(x,y)
    if anchor == "free" and line_style.get("x") is not None \
            and line_style.get("y") is not None:
        tags.append(f"\\pos({int(line_style['x'])},{int(line_style['y'])})")
    return "{" + "".join(tags) + "}" if tags else ""


def to_vtt(lines: list[SubtitleLine], speaker_labels: bool = False) -> str:
    """SubtitleLine[] → WebVTT（网页播放器可直接 <track> 加载）。"""
    blocks = ["WEBVTT", ""]
    for ln in lines:
        if not ln.words:
            continue
        start = format_srt_time(ln.start).replace(",", ".")
        end = format_srt_time(ln.end).replace(",", ".")
        text = _line_text(ln, speaker_labels)
        if speaker_labels and ln.speaker:
            text = f"<v {ln.speaker}>{ln.text}"
        blocks.append(f"{start} --> {end}\n{text}")
        blocks.append("")
    if len(blocks) <= 2:
        return ""
    return "\n".join(blocks).rstrip("\n") + "\n"


def to_json(lines: list[SubtitleLine], title: str = "", mode: str = "",
            duration: float = 0.0) -> str:
    """SubtitleLine[] → 词级 JSON（下游二次加工 / 编辑器交换格式）。"""
    import json

    payload = {
        "title": title,
        "mode": mode,
        "duration": round(duration, 3),
        "lines": [
            {
                "start": round(ln.start, 3),
                "end": round(ln.end, 3),
                "text": ln.text,
                "speaker": ln.speaker,
                "low_confidence": ln.low_confidence,
                "words": [
                    {"text": w.text, "start": round(w.start, 3),
                     "end": round(w.end, 3)}
                    for w in ln.words
                ],
            }
            for ln in lines if ln.words
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=1)


def from_json(content: str) -> list[SubtitleLine]:
    """JSON → SubtitleLine[]（编辑器读回）。"""
    import json

    data = json.loads(content)
    out: list[SubtitleLine] = []
    for row in data.get("lines", []):
        words = [SubtitleWord(text=w["text"], start=float(w["start"]),
                              end=float(w["end"]))
                 for w in row.get("words", []) if w.get("text")]
        if not words and row.get("text"):
            words = [SubtitleWord(text=row["text"], start=float(row["start"]),
                                  end=float(row["end"]))]
        if words:
            out.append(SubtitleLine(
                words=words, speaker=row.get("speaker", ""),
                low_confidence=bool(row.get("low_confidence"))))
    return out
