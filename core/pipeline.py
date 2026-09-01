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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from core.aligner import AlignmentError, Aligner
from core.audio import AudioProcessError, extract_wav, probe_duration, validate_input
from core.env import EnvInfo, detect_env, long_run_warning
from core.errors import TaskCancelled
from core.lyrics_fetcher import build_lyric_lines, fetch_lyrics
from core.models import SubtitleLine
from core.separator import VocalSeparator
from core.speaker import SpeakerAnalysis, SpeakerAnalyzer
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
    enable_voiceprint: bool = False     # 主声线声纹过滤开关（= speaker_mode "single"）
    profile_name: str = ""              # 主播声纹 Profile 名
    voice_threshold: float = 0.55       # 声纹相似度阈值（0.3~0.8）
    profiles_dir: str = "profiles"      # 声纹库目录
    enable_song_detect: bool = False    # 唱歌检测 + 听歌识曲开关
    enable_lyrics_fetch: bool = False   # 识曲后自动拉取同步歌词（LRCLIB）
    # 多声纹说话人分离（v1.2）
    speaker_mode: str = "off"           # "off" | "single" | "multi"
    speaker_analysis: object = None     # UI 预分析的 SpeakerAnalysis（复用不重跑）
    selected_speakers: list = field(default_factory=list)  # 勾选保留的说话人名
    use_speaker_library: bool = True    # 默认处理：分析时结合声纹库匹配命名
    mark_low_confidence: bool = True    # 低置信度区域标注（预览 + review 清单）
    speaker_labels: bool = False        # 导出字幕标注说话人（SRT/LRC 前缀，ASS Name）


@dataclass
class PipelineResult:
    mode: str                       # "blind" | "align"
    duration: float
    lines: list[SubtitleLine]
    files: dict[str, str]           # {"srt": 路径, ..., "songs_md": ..., "songs_csv": ...}
    warnings: list[str]
    out_dir: str
    songs: list = field(default_factory=list)  # SongEntry[]（识曲结果）
    speakers: list = field(default_factory=list)  # SpeakerCluster[]（说话人分离结果）


# 阶段进度权重（抽取 5% → 分离 25% → 识别/对齐 60% → 导出 10%）
_STAGE_BASE = {"extract": 0.00, "separate": 0.05, "recognize": 0.30, "export": 0.90}
_STAGE_SPAN = {"extract": 0.05, "separate": 0.25, "recognize": 0.60, "export": 0.10}

_LYRICS_TAIL_TOLERANCE = 2.0   # 歌词截断容差：演唱块结束后仍允许显示的秒数
_LYRICS_CONCURRENCY = 4        # 歌词并行拉取线程数


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
        speakers: list = []
        if lyrics:
            mode = "align"
            lines = self._run_align(vocals_path, lyrics, device, warnings)
            if self._voice_analysis_enabled() or cfg.enable_song_detect:
                msg = "声纹过滤与听歌识曲仅支持盲识别模式，本次任务已跳过"
                warnings.append(msg)
                self._log(msg)
        else:
            mode = "blind"
            lines, songs, speakers = self._run_blind(
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
            speakers=speakers,
        )

    def _voice_analysis_enabled(self) -> bool:
        """是否需要语音分析（任一模式声纹或识曲）。"""
        cfg = self.config
        return (cfg.enable_voiceprint or cfg.speaker_mode != "off"
                or cfg.enable_song_detect)

    def _run_blind(self, vocals_path, device, compute_type, duration,
                   env: EnvInfo, warnings) -> tuple[list[SubtitleLine], list, list]:
        """盲识别：可选语音分析（声纹 + 识曲 + 歌词拉取）后定向转写。

        返回 (lines, songs, speakers)；歌词字幕行与转写行按时间轴合并。
        """
        cfg = self.config
        self._stage_label = "语音识别"
        songs: list = []
        speakers: list = []
        lyric_lines: list[SubtitleLine] = []
        regions: Optional[list[tuple[float, float]]] = None
        speaker_analysis: Optional[SpeakerAnalysis] = None
        keep_speaker_ids: set[int] = set()
        base = 0.0  # 语音分析占识别阶段前 35%，转写占后 65%
        if self._voice_analysis_enabled():
            regions, songs, lyric_lines, speaker_analysis, keep_speaker_ids = \
                self._analyze(vocals_path, device, env, duration, warnings)
            if speaker_analysis is not None:
                speakers = speaker_analysis.clusters
            base = 0.35
            if regions is not None and not regions:
                # 全部为演唱块（歌词行已生成）或无任何语音：跳过转写
                self._progress("recognize", 1.0, "语音识别完成")
                return lyric_lines, songs, speakers
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
        if speaker_analysis is not None and keep_speaker_ids:
            self._apply_speaker_annotations(lines, speaker_analysis,
                                            keep_speaker_ids)
        if lyric_lines:
            lines = sorted(lines + lyric_lines, key=lambda ln: ln.start)
        self._progress("recognize", 1.0, "语音识别完成")
        return lines, songs, speakers

    # ---------------- 直播场景：语音分析 ----------------

    def _analyze(self, vocals_path, device, env: EnvInfo, duration, warnings):
        """VAD 分段 → 声纹（主声线 / 多说话人分离）→ 唱歌段识曲。

        返回 (转写区域, 歌单条目, 歌词字幕行, 说话人分析, 勾选说话人 id 集)：
        区域 None = 未分析（整段转写）；[] = 无有效语音或全部被歌词覆盖。
        已配歌词的演唱块从转写区域剔除。
        """
        cfg = self.config
        self._progress("recognize", 0.0, "正在检测语音段…")
        segments = speech_intervals(vocals_path)
        if not segments:
            warnings.append("未检测到语音活动，已跳过语音识别")
            self._progress("recognize", 0.35, "语音分析完成")
            return [], [], [], None, set()

        kept_verdicts = None
        speaker_analysis: Optional[SpeakerAnalysis] = None
        keep_speaker_ids: set[int] = set()
        if cfg.speaker_mode == "multi":
            speaker_analysis, keep_speaker_ids = self._analyze_speakers(
                vocals_path, device, segments, warnings)
            if keep_speaker_ids:
                segments = [(a.start, a.end) for a in speaker_analysis.assignments
                            if a.speaker_id in keep_speaker_ids]
                dropped = sum(1 for a in speaker_analysis.assignments
                              if a.speaker_id not in keep_speaker_ids)
                if dropped:
                    self._log(
                        f"说话人过滤：{dropped}/{len(speaker_analysis.assignments)} "
                        "段未勾选声纹已剔除")
                if not segments:
                    warnings.append("勾选的说话人没有语音段，未产生字幕")
                    self._progress("recognize", 0.35, "语音分析完成")
                    return [], [], [], speaker_analysis, keep_speaker_ids
            else:
                if cfg.speaker_analysis is not None and speaker_analysis.clusters:
                    warnings.append("未勾选任何说话人声纹，已跳过语音识别")
                elif speaker_analysis.clusters:
                    warnings.append("说话人簇时长过短均被过滤，已跳过语音识别")
                else:
                    warnings.append("未发现有效说话人（语音过短或均为噪声），已跳过")
                self._progress("recognize", 0.35, "语音分析完成")
                return [], [], [], speaker_analysis, keep_speaker_ids
        elif cfg.enable_voiceprint or cfg.speaker_mode == "single":
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
                return [], [], [], None, set()
            segments = [(v.start, v.end) for v in kept_verdicts]
            self._progress("recognize", 0.2, "声纹比对完成")
            self._stage_label = "语音识别"

        songs: list = []
        lyric_lines: list[SubtitleLine] = []
        if cfg.enable_song_detect:
            songs, song_spans, lyric_lines = self._detect_songs(
                vocals_path, kept_verdicts, segments, env, warnings,
            )
            if song_spans:
                segments = [s for s in segments if not _covered(s, song_spans)]

        regions = merge_regions(segments, duration=duration) if segments else []
        return regions, songs, lyric_lines, speaker_analysis, keep_speaker_ids

    def _analyze_speakers(self, vocals_path, device, segments, warnings
                          ) -> tuple[SpeakerAnalysis, set[int]]:
        """多说话人分离；优先复用 UI 预分析结果（避免重复嵌入/聚类）。

        返回 (分析结果, 勾选簇 id 集)。UI 已预分析但勾选为空 → 空 id 集
        （用户明确全不选）；未预分析 → pipeline 内部分析并全选有效簇。
        """
        cfg = self.config
        self._stage_label = "说话人分离"
        analysis: Optional[SpeakerAnalysis] = cfg.speaker_analysis
        preanalyzed = analysis is not None
        if preanalyzed and cfg.enable_separation and not analysis.separated:
            # 任务开了人声分离但预分析在原始混音上做（如带 BGM 的直播回放）
            # → 声纹已被 BGM 污染，丢弃预分析结果，在纯人声上重做更准
            self._log("预分析未含人声分离，已在纯人声上重新识别说话人（BGM 场景）")
            analysis = None
        if analysis is None:
            self._progress("recognize", 0.05, "正在识别说话人…")
            analyzer = SpeakerAnalyzer(device=device)
            analysis = analyzer.analyze(
                vocals_path, segments,
                profiles_dir=cfg.profiles_dir,
                use_library=cfg.use_speaker_library,
                separated=cfg.enable_separation,
                on_progress=lambda r: self._progress(
                    "recognize", 0.05 + 0.15 * r, "正在识别说话人…"),
            )
            if analysis.clusters:
                names = "、".join(
                    f"{c.name}（{c.duration:.0f}s"
                    + ("，库匹配" if c.matched_library else "") + "）"
                    for c in analysis.clusters)
                self._log(f"识别到 {len(analysis.clusters)} 位说话人：{names}")
        self._progress("recognize", 0.2, "说话人分离完成")
        self._stage_label = "语音识别"

        if preanalyzed:
            # UI 预分析场景：按勾选筛选（空 = 用户明确全不选）；
            # BGM 重分析时按名字映射到新簇（库命中名稳定，编号按时长排序基本不变）
            selected = analysis.refs_for(cfg.selected_speakers)
            return analysis, {c.speaker_id for c in selected}
        # pipeline 内部分析场景：全选有效簇
        return analysis, {c.speaker_id for c in analysis.clusters}

    def _apply_speaker_annotations(self, lines: list[SubtitleLine],
                                   analysis: SpeakerAnalysis,
                                   keep_speaker_ids: set[int]) -> None:
        """转写行 → 说话人归属 + 归属不确定标注（原地修改）。

        行中点落在某 VAD 段内即归属其说话人；ambiguous 段的行标注
        low_confidence 供人工复核。转写区域含 ±pad 边界，行中点可能
        落在所有段之外 → 就近归属（距离最近的段）。
        VAD 段有序不相交 → 二分查找包含/最近段，O(log n) 每行。
        """
        import bisect

        id2name = {c.speaker_id: c.name for c in analysis.clusters
                   if c.speaker_id in keep_speaker_ids}
        assignments = sorted(analysis.assignments, key=lambda a: a.start)
        if not id2name or not assignments:
            return
        starts = [a.start for a in assignments]

        for ln in lines:
            if not ln.words:
                continue
            mid = (ln.start + ln.end) / 2
            i = bisect.bisect_right(starts, mid) - 1
            best = None
            for cand in (assignments[i] if i >= 0 else None,
                         assignments[i + 1] if i + 1 < len(assignments) else None):
                if cand is None:
                    continue
                if cand.start <= mid <= cand.end:
                    best = cand               # 落在段内，直接归属
                    break
                d = min(abs(mid - cand.start), abs(mid - cand.end))
                if best is None or d < min(abs(mid - best.start),
                                           abs(mid - best.end)):
                    best = cand
            if best is None:
                continue
            if best.speaker_id in id2name:
                ln.speaker = id2name[best.speaker_id]
            if best.ambiguous:
                ln.low_confidence = True
                if not ln.low_confidence_reason:
                    ln.low_confidence_reason = "说话人归属不确定"

    def _detect_songs(self, vocals_path, kept_verdicts, segments, env: EnvInfo,
                      warnings):
        """演唱块识曲（+ 可选歌词拉取）。

        返回 (歌单条目, 剔除转写的区间, 歌词字幕行)：
        - 未开启歌词拉取：识别成功的块照旧剔除转写（v1.0 行为）
        - 开启歌词拉取：仅剔除拿到同步歌词的块；未拉到的保留转写兜底
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
            return [], [], []

        # 识曲占 0.20→0.30，歌词拉取占 0.30→0.35
        recognizer = SongRecognizer(
            ffmpeg=env.ffmpeg_path or "ffmpeg",
            on_log=self._log,
            on_progress=lambda r: self._progress(
                "recognize", 0.2 + 0.10 * r, "正在识别歌声…"),
            cancel_event=self.cancel_event,
        )
        with tempfile.TemporaryDirectory(prefix="songs_") as work:
            entries = recognizer.recognize_blocks(cfg.input_path, blocks, work)
        songs = merge_consecutive(entries)
        if songs:
            self._log(f"共识别到 {len(songs)} 首歌曲")

        lyric_lines: list[SubtitleLine] = []
        song_spans: list[tuple[float, float]] = []
        if cfg.enable_lyrics_fetch and songs:
            self._fetch_song_lyrics(songs)
        offset_noted = False
        for song in songs:
            if song.lyrics_lrc:
                lines = build_lyric_lines(
                    song.lyrics_lrc, block_offset=song.start,
                    until=song.end + _LYRICS_TAIL_TOLERANCE,
                )
                if lines:
                    lyric_lines.extend(lines)
                    song_spans.append((song.start, song.end))
                    self._log(f"《{song.title}》歌词字幕 {len(lines)} 行已生成")
                    if not offset_noted:
                        warnings.append(
                            "歌词时间轴以演唱块起点对齐；若直播跳过歌曲前奏，"
                            "对应段落歌词可能整体偏移"
                        )
                        offset_noted = True
            elif cfg.enable_lyrics_fetch:
                msg = f"《{song.title}》未找到同步歌词，该段保留语音识别"
                warnings.append(msg)
                self._log(msg)
            else:
                song_spans.append((song.start, song.end))  # v1.0 行为：识别到即剔除
        self._progress("recognize", 0.35, "听歌识曲完成")
        self._stage_label = "语音识别"
        return songs, song_spans, lyric_lines

    def _fetch_song_lyrics(self, songs: list) -> None:
        """并行拉取各歌曲同步歌词（best-effort，填充 entry.lyrics_lrc）。"""
        self._stage_label = "歌词拉取"
        self._progress("recognize", 0.30, "正在拉取歌词…")
        total = len(songs)

        def fetch_one(entry):
            return fetch_lyrics(entry.title, entry.artist)

        done = 0
        with ThreadPoolExecutor(max_workers=_LYRICS_CONCURRENCY) as pool:
            for entry, track in zip(songs, pool.map(fetch_one, songs)):
                done += 1
                self._progress(
                    "recognize", 0.30 + 0.05 * done / total if total else 0.35,
                    "正在拉取歌词…")
                if track is not None and track.synced:
                    entry.lyrics_lrc = track.synced
                    self._log(f"《{entry.title}》已获取同步歌词（LRCLIB）")
                else:
                    self._log(f"《{entry.title}》未找到同步歌词")

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
            "srt": lambda ls: to_srt(ls, speaker_labels=cfg.speaker_labels),
            "lrc": lambda ls: to_lrc(ls, title=title,
                                     speaker_labels=cfg.speaker_labels),
            "ass": lambda ls: to_ass(ls, speaker_labels=cfg.speaker_labels),
        }
        files: dict[str, str] = {}
        fmts = [f for f in cfg.formats if f in writers]
        review_rows = [ln for ln in lines
                       if ln.low_confidence] if cfg.mark_low_confidence else []
        total = len(fmts) + (2 if songs_enabled else 0) + (1 if review_rows else 0)
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
        if review_rows:
            review_path = out_dir / "review_low_confidence.md"
            review_path.write_text(
                self._format_review_md(review_rows), encoding="utf-8")
            files["review_md"] = str(review_path)
            step += 1
            self._progress("export", step / total,
                           f"已导出复核清单（{len(review_rows)} 行待人工验证）")
        return files

    @staticmethod
    def _format_review_md(rows: list[SubtitleLine]) -> str:
        """低置信度行 → 人工复核清单（Markdown 表格）。"""
        head = ("# 低置信度复核清单\n\n"
                "以下字幕行由模型自动标注为「置信度低」，建议人工验证并修改：\n"
                "\n| 位置 | 原因 | 说话人 | 文本 |\n|---|---|---|---|\n")
        body = []
        for i, ln in enumerate(rows, 1):
            speaker = ln.speaker or "-"
            reason = ln.low_confidence_reason or "置信度低"
            body.append(
                f"| {i} · {_fmt_review_time(ln.start)}–{_fmt_review_time(ln.end)} "
                f"| {reason} | {speaker} | {ln.text} |")
        return head + "\n".join(body) + "\n"

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
