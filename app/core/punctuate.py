"""标点恢复：给无标点的识别文本补标点，**仅用于切句**，不进入最终字幕。

为什么需要：
- whisper / paraformer 的中文输出经常缺标点（实测 large-v3 在本项目样本上
  整段无标点），而人工字幕的分行高度依赖句读；
- 人工校对样本本身**不含标点**（见 Sample 统计：仅出现 '%'），
  因此标点只作为「切行线索」，导出前统一剥离。

实现：FunASR ct-punc（ModelScope，离线可用）批量处理；
不可用时退化为「停顿 + 长度」切分（调用方无需分支）。
"""
from __future__ import annotations

import re
import threading

_PUNCT = set("，。！？；、：,.!?;:")
_MODEL: dict[str, object] = {}
_LOCK = threading.Lock()


def _model(device: str):
    with _LOCK:
        if device in _MODEL:
            return _MODEL[device]
    try:
        from funasr import AutoModel

        m = AutoModel(model="ct-punc", device=device, disable_update=True)
    except Exception:
        m = None
    with _LOCK:
        _MODEL[device] = m
    return m


def restore(text: str, device: str = "cuda") -> str:
    """整段文本补标点；模型不可用/失败时原样返回。"""
    if not text.strip():
        return text
    m = _model(device)
    if m is None:
        return text
    try:
        result = m.generate(input=text)
        out = result[0]["text"] if result else ""
    except Exception:
        return text
    return out or text


def strip_punct(text: str) -> str:
    """移除标点与多余空白（人工字幕风格：只留文字）。"""
    return "".join(
        ch for ch in text
        if ch not in _PUNCT and not (ch.isspace() and False)
    ).strip()


def map_back(punctuated: str, source_chars: str) -> list[int]:
    """把带标点文本的字符位置映射到原始字符下标。

    返回长度为 len(punctuated) 的列表：标点位置记 -1，
    其余位置给出其在 source_chars 中的下标。
    """
    out: list[int] = []
    i = 0
    n = len(source_chars)
    for ch in punctuated:
        if i < n and ch == source_chars[i]:
            out.append(i)
            i += 1
        elif ch in _PUNCT or ch.isspace():
            out.append(-1)
        elif i < n:
            # 模型改字/漏字：按顺序消耗一个源字符，保持单调
            out.append(i)
            i += 1
        else:
            out.append(-1)
    return out


def boundary_flags(punctuated: str, source_chars: str) -> list[bool]:
    """返回「源字符之后是否应断行」的标记（长度 = len(source_chars)）。

    句末标点 → True；逗号类 → True（弱断句，仍由切行代价决定）；
    无标点 → False。
    """
    mapping = map_back(punctuated, source_chars)
    flags = [False] * len(source_chars)
    prev_src = -1
    for ch, src in zip(punctuated, mapping):
        if ch in _PUNCT and prev_src >= 0:
            flags[prev_src] = True
        if src >= 0:
            prev_src = src
    return flags


__all__ = ["restore", "strip_punct", "map_back", "boundary_flags"]
