"""流水线编排：抽取 5% → 人声分离 25%（可选）→ 识别/对齐 60% → 聚合导出 10%。

只做编排不碰算法；进度汇总为全局单值；取消 Event 贯穿各阶段；
已知模块异常映射为带阶段前缀的用户可读 PipelineError。

直播场景（盲识别模式可选开启）：识别阶段内部先做「语音分析」——
VAD 分段 → 声纹过滤（剔除非目标声音）→ 唱歌段识曲（生成歌单时间戳），
再对保留区域定向转写（clip_timestamps），最后导出字幕 + 歌单。
"""
from __future__ import annotations

import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from core.aligner import AlignmentError, Aligner
from core.audio import AudioProcessError, extract_wav, probe_duration, validate_input
from core.env import EnvInfo, detect_env, long_run_warning
from core.errors import TaskCancelled
from core.models import SubtitleLine
from core.separator import VocalSeparator
from core.song_recognizer import (
    SONG_MIN_SECONDS,
    SongRecognizer,
    format_timeline_csv,
    format_timeline_md,
    merge_blocks,
    merge_consecutive,
)
from core.subtitle import aggregate_words, to_ass, to_lrc, to_srt
from core.text import cjk_ratio, prepare_lyrics
from core.transcriber import Transcriber, TranscriptionError
from core.voiceprint import (
    VoiceprintEngine,
    VoiceprintError,
    load_profile,
    merge_regions,
    speech_intervals,
)


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
    # 直播场景
    enable_voiceprint: bool = False     # 主声线声纹过滤开关
    profile_name: str = ""              # 主播声纹 Profile 名
    voice_threshold: float = 0.55       # 声纹相似度阈值（0.3~0.8）
    profiles_dir: str = "profiles"      # 声纹库目录
    enable_song_detect: bool = False    # 唱歌检测 + 听歌识曲开关


@dataclass
class PipelineResult:
    mode: str                       # "blind" | "align"
    duration: float
    lines: list[SubtitleLine]
    files: dict[str, str]           # {"srt": 路径, ..., "songs_md": ..., "songs_csv": ...}
    warnings: list[str]
    out_dir: str
    songs: list = field(default_factory=list)  # SongEntry[]（识曲结果）


# 阶段进度权重（抽取 5% → 分离 25% → 识别/对齐 60% → 导出 10%）
_STAGE_BASE = {"extract": 0.00, "separate": 0.05, "recognize": 0.30, "export": 0.90}
_STAGE_SPAN = {"extract": 0.05, "separate": 0.25, "recognize": 0.60, "export": 0.10}


def _covered(segment: tuple[float, float],
             spans: list[tuple[float, float]], eps: float = 1e-6) -> bool:
    """段是否被任一区间完全覆盖（用于把已识别演唱块剔出转写区域）。"""
    s, e = segment
    return any(ps - eps <= s and e <= pe + eps for ps, pe in spans)


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
        except (AudioProcessError, TranscriptionError, AlignmentError,
                VoiceprintError) as exc:
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
        songs: list = []
        if lyrics:
            mode = "align"
            lines = self._run_align(vocals_path, lyrics, device, warnings)
            if cfg.enable_voiceprint or cfg.enable_song_detect:
                msg = "声纹过滤与听歌识曲仅支持盲识别模式，本次任务已跳过"
                warnings.append(msg)
                self._log(msg)
        else:
            mode = "blind"
            lines, songs = self._run_blind(
                vocals_path, device, compute_type, duration, env, warnings,
            )

        # 阶段 4：聚合导出（10%）
        self._stage_label = "字幕导出"
        self._progress("export", 0.0, "正在导出字幕…")
        files = self._export(
            out_dir, lines, songs,
            songs_enabled=(mode == "blind" and cfg.enable_song_detect),
        )
        self._progress("export", 1.0, "全部完成")
        return PipelineResult(
            mode=mode, duration=duration, lines=lines, files=files,
            warnings=warnings, out_dir=str(out_dir), songs=songs,
        )

    def _run_blind(self, vocals_path, device, compute_type, duration,
                   env: EnvInfo, warnings) -> tuple[list[SubtitleLine], list]:
        """盲识别：可选语音分析（声纹过滤 + 识曲）后定向转写。返回 (lines, songs)。"""
        cfg = self.config
        self._stage_label = "语音识别"
        songs: list = []
        regions: Optional[list[tuple[float, float]]] = None
        base = 0.0  # 语音分析占识别阶段前 35%，转写占后 65%
        if cfg.enable_voiceprint or cfg.enable_song_detect:
            regions, songs = self._analyze(vocals_path, device, env, duration, warnings)
            base = 0.35
            if regions is not None and not regions:
                self._progress("recognize", 1.0, "语音识别完成")
                return [], songs
        transcriber = Transcriber(
            model_size=cfg.model_size,
            device=device,
            compute_type=compute_type,
            language=cfg.language,
            on_progress=lambda r: self._progress(
                "recognize", base + (1.0 - base) * r, "正在语音识别…"),
            cancel_event=self.cancel_event,
        )
        segments = transcriber.transcribe(
            vocals_path, total_duration=duration, clip_timestamps=regions,
        )
        words = [w for line in segments for w in line.words]
        lines = aggregate_words(words)
        self._progress("recognize", 1.0, "语音识别完成")
        return lines, songs

    # ---------------- 直播场景：语音分析 ----------------

    def _analyze(self, vocals_path, device, env: EnvInfo, duration,
                 warnings) -> tuple[Optional[list[tuple[float, float]]], list]:
        """VAD 分段 → 声纹过滤 → 唱歌段识曲。

        返回 (转写区域, 歌单条目)：区域 None = 未分析（整段转写）；
        [] = 无有效语音。识别成功的演唱块从转写区域剔除。
        """
        cfg = self.config
        self._progress("recognize", 0.0, "正在检测语音段…")
        segments = speech_intervals(vocals_path)
        if not segments:
            warnings.append("未检测到语音活动，已跳过语音识别")
            self._progress("recognize", 0.35, "语音分析完成")
            return [], []

        kept_verdicts = None
        if cfg.enable_voiceprint:
            self._stage_label = "声纹分析"
            self._progress("recognize", 0.05, "正在声纹比对…")
            profile = load_profile(cfg.profile_name, cfg.profiles_dir)
            engine = VoiceprintEngine(device=device)
            verdicts = engine.classify_segments(
                vocals_path, segments, profile,
                on_progress=lambda r: self._progress(
                    "recognize", 0.05 + 0.15 * r, "正在声纹比对…"),
            )
            kept_verdicts = [v for v in verdicts
                             if v.best_sim >= cfg.voice_threshold]
            dropped = len(segments) - len(kept_verdicts)
            if dropped:
                self._log(f"声纹过滤：{dropped}/{len(segments)} 段非目标人声已剔除")
            if not kept_verdicts:
                warnings.append("所有人声段均低于声纹相似度阈值，未产生字幕")
                self._progress("recognize", 0.35, "语音分析完成")
                return [], []
            segments = [(v.start, v.end) for v in kept_verdicts]
            self._progress("recognize", 0.2, "声纹比对完成")
            self._stage_label = "语音识别"

        songs: list = []
        if cfg.enable_song_detect:
            songs, song_spans = self._detect_songs(
                vocals_path, kept_verdicts, segments, env,
            )
            if song_spans:
                segments = [s for s in segments if not _covered(s, song_spans)]

        regions = merge_regions(segments, duration=duration) if segments else []
        return regions, songs

    def _detect_songs(self, vocals_path, kept_verdicts, segments, env: EnvInfo):
        """演唱块识曲：候选（唱歌判定段 / 长语音段）→ 合并块 → shazam。

        返回 (歌单条目, 已识别演唱块区间)；未识别的块保留转写兜底。
        """
        cfg = self.config
        self._stage_label = "听歌识曲"
        self._progress("recognize", 0.2, "正在识别歌声…")
        has_sing_profile = kept_verdicts is not None and any(
            v.sing_sim is not None for v in kept_verdicts)
        if has_sing_profile:
            candidates = [(v.start, v.end) for v in kept_verdicts if v.is_singing]
        else:
            candidates = [s for s in segments if s[1] - s[0] > SONG_MIN_SECONDS]
        blocks = [b for b in merge_blocks(candidates) if b[1] - b[0] >= SONG_MIN_SECONDS]
        if not blocks:
            self._log("未检测到符合条件的演唱段")
            self._progress("recognize", 0.35, "听歌识曲完成")
            self._stage_label = "语音识别"
            return [], []

        recognizer = SongRecognizer(
            ffmpeg=env.ffmpeg_path or "ffmpeg",
            on_log=self._log,
            on_progress=lambda r: self._progress(
                "recognize", 0.2 + 0.15 * r, "正在识别歌声…"),
            cancel_event=self.cancel_event,
        )
        with tempfile.TemporaryDirectory(prefix="songs_") as work:
            entries = recognizer.recognize_blocks(cfg.input_path, blocks, work)
        songs = merge_consecutive(entries)
        if songs:
            self._log(f"共识别到 {len(songs)} 首歌曲")
        self._progress("recognize", 0.35, "听歌识曲完成")
        self._stage_label = "语音识别"
        return songs, [(e.start, e.end) for e in songs]

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

    def _export(self, out_dir: Path, lines: list[SubtitleLine], songs: list,
                songs_enabled: bool = True) -> dict[str, str]:
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
        total = len(fmts) + (2 if songs_enabled else 0)
        step = 0
        for fmt in fmts:
            path = out_dir / f"{stem}.{fmt}"
            path.write_text(writers[fmt](lines), encoding="utf-8")
            files[fmt] = str(path)
            step += 1
            self._progress("export", step / total if total else 1.0,
                           f"已导出 {fmt.upper()}")
        if songs_enabled:
            md_path = out_dir / "songs_timeline.md"
            md_path.write_text(format_timeline_md(songs), encoding="utf-8")
            files["songs_md"] = str(md_path)
            step += 1
            self._progress("export", step / total, "已导出歌单 Markdown")
            csv_path = out_dir / "songs_timeline.csv"
            csv_path.write_text(format_timeline_csv(songs), encoding="utf-8")
            files["songs_csv"] = str(csv_path)
            step += 1
            self._progress("export", step / total, "已导出歌单 CSV")
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
