"""统一中间表示：所有识别/对齐结果与导出格式共用的数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubtitleWord:
    """词/字级单元。text 为显示形式（词本体 + 尾随标点）。"""

    text: str
    start: float  # 秒
    end: float    # 秒


@dataclass
class SubtitleLine:
    """行级单元，由词聚合而来；start/end/text 均由 words 推导。"""

    words: list[SubtitleWord] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0

    @property
    def text(self) -> str:
        return "".join(word.text for word in self.words)
