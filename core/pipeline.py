"""流水线编排：抽取 5% → 人声分离 25%（可选）→ 识别/对齐 60% → 聚合导出 10%。

只做编排不碰算法；进度汇总为全局单值；取消 Event 贯穿各阶段；
已知模块异常映射为带阶段前缀的用户可读 PipelineError。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from core.aligner import AlignmentError, Aligner
from core.audio import AudioProcessError, extract_wav, probe_duration, validate_input
from core.env import detect_env, long_run_warning
from core.errors import TaskCancelled
from core.models import SubtitleLine
from core.separator import VocalSeparator
from core.subtitle import aggregate_words, to_ass, to_lrc, to_srt
from core.text import cjk_ratio, prepare_lyrics
from core.transcriber import Transcriber, TranscriptionError


class PipelineError(RuntimeError):
    """流水线失败（用户可读消息）。"""


@dataclass
class PipelineConfig:
    input_path: str
    lyrics_text: str = ""               # 空 = 盲识别；非空 = 强制对齐
    work_dir: str = "outputs"           # 输出根目录（<work_dir>/<文件名>/<时间戳>/）
    device: str = "auto"                # "auto" | "cuda" | "cpu"
    model_size: str = "small"           # 盲识别 whisper 模型
    compute_type: Optional[str] = None  # None = 跟随环境探测
    enable_separation: bool = False     # Demucs 人声分离开关
    language: str = "auto"              # 盲识别语言；对齐语言由歌词文本推断
    formats: tuple = ("srt", "lrc", "ass")
    title: str = ""                     # LRC [ti:] 标题（默认用文件名）


@dataclass
class PipelineResult:
    mode: str                       # "blind" | "align"
    duration: float
    lines: list[SubtitleLine]
    files: dict[str, str]           # {"srt": 路径, ...}
    warnings: list[str]
    out_dir: str


# 阶段进度权重（抽取 5% → 分离 25% → 识别/对齐 60% → 导出 10%）
_STAGE_BASE = {"extract": 0.00, "separate": 0.05, "recognize": 0.30, "export": 0.90}
_STAGE_SPAN = {"extract": 0.05, "separate": 0.25, "recognize": 0.60, "export": 0.10}


class Pipeline:
    """单任务流水线；GUI 在后台线程调用 run()，通过回调接收进度/日志。"""

    def __init__(
        self,
        config: PipelineConfig,
        on_progress: Optional[Callable[[float, str], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        self.config = config
        self.on_progress = on_progress
        self.on_log = on_log
        self.cancel_event = cancel_event or threading.Event()
        self._stage_label = ""

    def run(self) -> PipelineResult:
        try:
            return self._run()
        except (TaskCancelled, PipelineError):
            raise
        except (AudioProcessError, TranscriptionError, AlignmentError) as exc:
            raise PipelineError(f"{self._stage_label}失败：{exc}") from exc

    # ---------------- 内部流程 ----------------

    def _run(self) -> PipelineResult:
        cfg = self.config
        warnings: list[str] = []

        env = detect_env()
        if not env.ffmpeg_available:
            raise PipelineError("未检测到 FFmpeg，请先安装并将其加入 PATH 后重试")
        device = env.device if cfg.device == "auto" else cfg.device
        compute_type = cfg.compute_type or env.compute_type

        validate_input(cfg.input_path)
        duration = probe_duration(cfg.input_path)
        if (warn := long_run_warning(env, duration, cfg.enable_separation)) :
            warnings.append(warn)
            self._log(warn)

        out_dir = self._make_out_dir()

        # 阶段 1：抽取音频（5%）
        self._stage_label = "音频抽取"
        self._progress("extract", 0.0, "正在抽取音频…")
        wav_path = str(extract_wav(cfg.input_path, out_dir))
        self._progress("extract", 1.0, "音频抽取完成")

        # 阶段 2：人声分离（25%，可选）
        vocals_path = wav_path
        if cfg.enable_separation:
            self._stage_label = "人声分离"
            self._progress("separate", 0.0, "正在分离人声…")
            separator = VocalSeparator(
                device=device,
                on_progress=lambda r: self._progress("separate", r, "正在分离人声…"),
                on_warning=self._collect_warning(warnings),
                cancel_event=self.cancel_event,
            )
            vocals_path = separator.separate(wav_path, out_dir)
            self._progress("separate", 1.0, "人声分离完成")

        # 阶段 3：识别 / 对齐（60%）
        lyrics = prepare_lyrics(cfg.lyrics_text) if cfg.lyrics_text.strip() else []
        if cfg.lyrics_text.strip() and not lyrics:
            msg = "歌词文本无有效内容，已自动转为盲识别模式"
            warnings.append(msg)
            self._log(msg)
        if lyrics:
            mode, lines = "align", self._run_align(vocals_path, lyrics, device, warnings)
        else:
            mode, lines = "blind", self._run_blind(vocals_path, device, compute_type, duration)

        # 阶段 4：聚合导出（10%）
        self._stage_label = "字幕导出"
        self._progress("export", 0.0, "正在导出字幕…")
        files = self._export(out_dir, lines)
        self._progress("export", 1.0, "全部完成")
        return PipelineResult(
            mode=mode, duration=duration, lines=lines, files=files,
            warnings=warnings, out_dir=str(out_dir),
        )

    def _run_blind(self, vocals_path, device, compute_type, duration) -> list[SubtitleLine]:
        self._stage_label = "语音识别"
        self._progress("recognize", 0.0, "正在语音识别…")
        transcriber = Transcriber(
            model_size=self.config.model_size,
            device=device,
            compute_type=compute_type,
            language=self.config.language,
            on_progress=lambda r: self._progress("recognize", r, "正在语音识别…"),
            cancel_event=self.cancel_event,
        )
        segments = transcriber.transcribe(vocals_path, total_duration=duration)
        words = [w for line in segments for w in line.words]
        lines = aggregate_words(words)
        self._progress("recognize", 1.0, "语音识别完成")
        return lines

    def _run_align(self, vocals_path, lyrics, device, warnings) -> list[SubtitleLine]:
        language = "zh" if cjk_ratio(self.config.lyrics_text) >= 0.5 else "en"
        self._stage_label = "歌词对齐"
        self._progress("recognize", 0.0, "正在对齐歌词…")
        aligner = Aligner(
            language=language,
            device=device,
            on_progress=lambda r: self._progress("recognize", r, "正在对齐歌词…"),
            cancel_event=self.cancel_event,
        )
        result = aligner.align(vocals_path, lyrics)
        for idx in result.low_confidence:
            msg = f"第 {idx + 1} 行对齐置信度较低，建议人工复核"
            warnings.append(msg)
            self._log(msg)
        self._progress("recognize", 1.0, "歌词对齐完成")
        return result.lines

    def _export(self, out_dir: Path, lines: list[SubtitleLine]) -> dict[str, str]:
        cfg = self.config
        stem = Path(cfg.input_path).stem
        title = cfg.title or stem
        writers = {
            "srt": lambda ls: to_srt(ls),
            "lrc": lambda ls: to_lrc(ls, title=title),
            "ass": lambda ls: to_ass(ls),
        }
        files: dict[str, str] = {}
        fmts = [f for f in cfg.formats if f in writers]
        for i, fmt in enumerate(fmts):
            path = out_dir / f"{stem}.{fmt}"
            path.write_text(writers[fmt](lines), encoding="utf-8")
            files[fmt] = str(path)
            self._progress("export", (i + 1) / len(fmts), f"已导出 {fmt.upper()}")
        return files

    # ---------------- 辅助 ----------------

    def _make_out_dir(self) -> Path:
        stem = Path(self.config.input_path).stem
        out = Path(self.config.work_dir) / stem / time.strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _progress(self, stage: str, ratio: float, message: str) -> None:
        if self.on_progress:
            r = _STAGE_BASE[stage] + _STAGE_SPAN[stage] * min(1.0, max(0.0, ratio))
            self.on_progress(r, message)

    def _collect_warning(self, warnings: list[str]):
        def on_warning(message: str) -> None:
            warnings.append(message)
            self._log(message)

        return on_warning

    def _log(self, message: str) -> None:
        if self.on_log:
            self.on_log(message)
