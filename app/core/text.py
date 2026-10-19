"""歌词文本预处理：标准化、分字分词、标点剥离与回填（纯逻辑，无重依赖）。

Token 契约：
- align  ：供声学模型对齐的剥离形式（无标点、全角拉丁/数字归一为半角）
- display：导出还原用的显示形式（词本体 + 尾随标点/空白；行首标点前缀到首 token）
拼接所有 display 可无损还原（已剥离时间戳后的）原行。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# LRC 内嵌时间戳：[00:12.34] [01:02] [00:12:34.5] 及 <00:12.34> 变体
_TIMESTAMP_RE = re.compile(r"[<\[]\d{1,2}:\d{2}(?:[:.]\d{1,3})?[>\]]")
# 元信息标记行：[ti:xxx] [ar:xxx] [offset:+500] 等
_METADATA_RE = re.compile(
    r"^\[(?:ti|ar|al|by|offset|re|ve|length|au):.*\]\s*$", re.IGNORECASE
)
# CJK 汉字（含扩展 A 区与兼容表意区）
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
# 词字符：ASCII 与全角拉丁字母/数字（撇号保留在词内）
_WORD_RE = re.compile(r"[0-9A-Za-z'\uff10-\uff19\uff21-\uff3a\uff41-\uff5a]+")


@dataclass
class Token:
    """一个对齐单元：align 用于对齐，display 用于导出还原。"""

    align: str
    display: str


def tokenize(text: str) -> list[Token]:
    """把一行文本切分为 token 序列；标点剥离进 display，不进入 align。"""
    tokens: list[Token] = []
    leading: list[str] = []  # 行首标点，最终前缀到第一个 token
    i = 0
    n = len(text)
    while i < n:
        if _CJK_RE.match(text[i]):
            tokens.append(Token(align=unicodedata.normalize("NFKC", text[i]), display=text[i]))
            i += 1
        elif (m := _WORD_RE.match(text, i)):
            tokens.append(Token(align=unicodedata.normalize("NFKC", m.group()), display=m.group()))
            i = m.end()
        else:
            # 标点/空白：尾随挂前一个 token 的 display；行首则暂存
            if tokens:
                tokens[-1].display += text[i]
            else:
                leading.append(text[i])
            i += 1
    if leading and tokens:
        tokens[0].display = "".join(leading) + tokens[0].display
    return tokens


def prepare_lyrics(raw: str) -> list[list[Token]]:
    """整段歌词 → 行 token 列表：剥离时间戳/元信息行/空行，逐行 tokenize。"""
    lines: list[list[Token]] = []
    for raw_line in raw.splitlines():
        line = _TIMESTAMP_RE.sub("", raw_line).strip()
        if not line or _METADATA_RE.match(line):
            continue
        tokens = tokenize(line)
        if tokens:
            lines.append(tokens)
    return lines


def cjk_ratio(text: str) -> float:
    """词字符中 CJK 占比（标点与空白不计入）；空文本返回 0.0。"""
    total = 0
    cjk = 0
    for token in tokenize(text):
        for ch in token.align:
            total += 1
            if _CJK_RE.match(ch):
                cjk += 1
    return cjk / total if total else 0.0


# ---------------- 繁→简统一 ----------------

# zhconv 缺失时保留原文（懒加载失败只发生一次）
_t2s_impl = None
_t2s_tried = False

# 转换结果缓存：字幕同词重复率极高（3.4h 直播 ≈ 2 万词但去重后仅数千），
# 命中缓存可省掉大部分 zhconv 调用开销。重新探测实现时清空，避免脏值。
_t2s_cache: dict[str, str] = {}
_T2S_CACHE_MAX = 32768


def clear_t2s_cache() -> None:
    """清空繁简转换缓存（实现切换/测试隔离用）。"""
    _t2s_cache.clear()


def zhconv_available() -> bool:
    """zhconv 是否可导入（繁简转换能力探测）。"""
    t2s("")
    return _t2s_impl is not None


def t2s(text: str) -> str:
    """繁体→简体（zhconv，懒加载 + 结果缓存）。

    仅映射繁简汉字，对其他语言文本无副作用；zhconv 未安装时
    原样返回（调用方用 zhconv_available 提示用户）。
    """
    global _t2s_impl, _t2s_tried
    if _t2s_impl is None and not _t2s_tried:
        _t2s_tried = True
        _t2s_cache.clear()  # 实现（重新）探测 → 旧缓存失效
        try:
            from zhconv import convert as _convert
            _t2s_impl = _convert
        except ImportError:
            _t2s_impl = None
    if _t2s_impl is None or not text:
        return text
    hit = _t2s_cache.get(text)
    if hit is not None:
        return hit
    out = _t2s_impl(text, "zh-cn")
    if len(_t2s_cache) < _T2S_CACHE_MAX:
        _t2s_cache[text] = out
    return out
