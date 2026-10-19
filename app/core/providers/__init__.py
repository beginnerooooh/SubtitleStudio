"""识别引擎抽象层（可插拔后端 + 外部 API 接入点）。

设计目的（对应评审「后端可插拔应有统一抽象基类」与需求 9「预留接口」）：

    ASRProvider (ABC)
    ├── WhisperProvider    本地 faster-whisper（CTranslate2，推荐）
    ├── FunASRProvider     本地 paraformer-zh（中文/标点/热词）
    ├── ExternalHTTPProvider  用户自建 HTTP 服务（POST 音频 → JSON 词级结果）
    └── register_provider()  第三方插件注册（SUBTITLE_STUDIO_PLUGINS）

统一契约：transcribe(wav_path, ...) -> list[AsrSegment]，
其中 AsrSegment 的 words 为「模型原始词单元」（含标点、时间、置信度），
行切分统一交给 core.segmenter，各后端不各自实现切行。
"""
from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional


class AsrError(RuntimeError):
    """识别失败（消息面向用户）。"""


@dataclass
class AsrWord:
    """模型输出的词单元（未切分）。"""

    text: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class AsrSegment:
    """模型输出的一段话（供过滤与低置信判定）。"""

    start: float
    end: float
    text: str
    words: list[AsrWord] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0
    language: str = ""


class ASRProvider(ABC):
    """识别后端统一接口。"""

    name: str = "base"

    @abstractmethod
    def transcribe(
        self,
        wav_path: str,
        *,
        language: str = "auto",
        total_duration: Optional[float] = None,
        on_progress: Optional[Callable[[float], None]] = None,
        cancel_event=None,
    ) -> list[AsrSegment]:
        """转写整段音频；时间戳必须是「全局时间轴」（相对 wav 起点）。"""

    def close(self) -> None:
        """释放资源（模型/显存）；默认无操作。"""


# ---------------- 注册表 ----------------

_REGISTRY: dict[str, type[ASRProvider]] = {}


def register_provider(cls: type[ASRProvider]) -> type[ASRProvider]:
    """注册后端（插件可用装饰器方式接入）。"""
    _REGISTRY[cls.name] = cls
    return cls


def available_providers() -> list[str]:
    """已注册后端名（含延迟导入的内置后端）。"""
    _ensure_builtin()
    return sorted(_REGISTRY)


def get_provider(name: str, **kwargs) -> ASRProvider:
    """按名取后端实例。"""
    _ensure_builtin()
    cls = _REGISTRY.get(name)
    if cls is None:
        raise AsrError(
            f"未知识别引擎：{name}（可用：{', '.join(sorted(_REGISTRY))}）")
    return cls(**kwargs)


def _ensure_builtin() -> None:
    if _REGISTRY:
        return
    from core.providers.whisper_provider import WhisperProvider
    from core.providers.funasr_provider import FunASRProvider
    from core.providers.http_provider import ExternalHTTPProvider

    for cls in (WhisperProvider, FunASRProvider, ExternalHTTPProvider):
        _REGISTRY[cls.name] = cls


def load_plugin_modules(modules: Iterable[str]) -> list[str]:
    """导入插件模块（模块内用 @register_provider 注册自己的后端）。"""
    import importlib

    loaded = []
    for name in modules:
        try:
            importlib.import_module(name)
            loaded.append(name)
        except Exception as exc:  # 插件失败不应影响主流程
            print(f"[provider] 插件加载失败 {name}: {exc}")
    return loaded


__all__ = [
    "ASRProvider", "AsrSegment", "AsrWord", "AsrError", "register_provider",
    "available_providers", "get_provider", "load_plugin_modules",
]
