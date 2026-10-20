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
    speaker: str = ""                 # 说话人名（多声纹模式后处理填充）
    low_confidence: bool = False      # 置信度标注：供人工复核
    low_confidence_reason: str = ""   # 标注原因（识别置信度低 / 说话人归属不确定）
    # 行级样式覆盖（空表示跟随全局样式）。键与前端 S.style 一致：
    # size / color / outline / anchor / x / y / bg
    # 用于「只给某几行单独设置样式」，以及「应用到所有字幕」把全局样式固化到行上。
    style: dict = field(default_factory=dict)

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0

    @property
    def text(self) -> str:
        return "".join(word.text for word in self.words)
