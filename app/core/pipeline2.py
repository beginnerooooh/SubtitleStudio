"""流水线编排 v2：抽取 →（可选）人声分离 →（可选）声纹/歌曲 → 识别 → 精修 → 切行 → 导出。

相比 v1 的根因级改动（详见 docs/EVALUATION.md）：
1. **引擎可插拔**：ASR 走 core.providers 注册表，默认 whisper-large-v3；
2. **关闭 VAD 过滤**：实测字符召回 0.660 → 0.770（歌声/弱起句不再整段丢失）；
3. **时间轴精修**：能量包络重定位起止，修正 whisper 系统性偏早
   （实测起始误差中位数 -0.33s → 约 -0.03s）；
4. **切行改为全局 DP**：行长/标点/停顿联合最优，替代"段即行"的贪心聚合；
5. **歌曲走歌词强制对齐**：识曲失败时用转写文本检索歌词库 + 时间戳预测，
   不再让 ASR 直接产出歌词文本；
6. **声纹只判主播一人**：双音色 + 段内投票，替代多类聚类；
7. **旁路信息**（弹幕 XML）参与热词与断句。
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core.config import AppConfig
from core.evaluate import EvalReport, evaluate, load_srt
from core.models import SubtitleLine, SubtitleWord

_STAGE_BASE = {"extract": 0.00, "separate": 0.05, "analyze": 0.30,
               "recognize": 0.45, "export": 0.92}
_STAGE_SPAN = {"extract": 0.05, "separate": 0.25, "analyze": 0.15,
               "recognize": 0.47, "export": 0.08}

# 识曲连续失败多少次后停用。识曲是网络调用（单次最长 25 秒超时），
# 无网络时逐块重试会浪费大量时间，因此熔断。
SHAZAM_FAIL_LIMIT = 3


class PipelineError(RuntimeError):
    """流水线失败（用户可读消息）。"""


@dataclass
class PipelineRequest:
    """一次任务的输入。"""

    input_path: str
    lyrics_text: str = ""            # 非空 → 强制对齐模式
    reference_srt: str = ""          # GT 参考（评测用，可选）
    live_xml: str = ""               # 弹幕/礼物 XML（旁路信息，可选）
    formats: tuple = ("srt", "lrc", "ass", "vtt", "json")
    title: str = ""
    enable_separation: bool = False
    enable_voiceprint: bool = False
    profile_name: str = ""           # 主播声纹名（启用单目标声纹过滤时必填）
    enable_song_detect: bool = False
    enable_song_align: bool = True
    speaker_labels: bool = False
    mark_low_confidence: bool = False


@dataclass
class PipelineResult:
    mode: str
    duration: float
    lines: list[SubtitleLine]
    files: dict[str, str]
    warnings: list[str]
    out_dir: str
    songs: list = field(default_factory=list)
    speakers: list = field(default_factory=list)
    evaluation: Optional[EvalReport] = None
    timings: dict = field(default_factory=dict)
    # BGM 判别明细（供 UI 复核清单：哪些时间段被判为伴奏原唱、依据是什么）
    bgm_flags: list = field(default_factory=list)
    # 弹幕旁路信息摘要（主播名/点歌请求/热词），便于 UI 展示与复核
    live_summary: dict = field(default_factory=dict)
    # 文本校正明细（弹幕专名校正 + 歌词库替换），供 UI 展示与回溯
    corrections: dict = field(default_factory=dict)


class Pipeline:
    """单任务流水线（GUI/CLI 共用；回调式进度与日志）。"""

    def __init__(
        self,
        config: AppConfig,
        request: PipelineRequest,
        on_progress: Optional[Callable[[float, str], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        self.cfg = config.resolve()
        self.req = request
        self.on_progress = on_progress
        self.on_log = on_log
        self.cancel_event = cancel_event or threading.Event()
        self.warnings: list[str] = []
        self.timings: dict[str, float] = {}
        self._stage = ""
        # 识曲熔断：连续失败达阈值后停用（避免无网络时白等大量超时）
        self._shazam_fails = 0
        self._shazam_disabled = False

    # ---------------- 入口 ----------------

    def run(self) -> PipelineResult:
        from core.audio import AudioProcessError, extract_wav, probe_duration, validate_input
        from core.errors import TaskCancelled

        t_all = time.time()
        try:
            validate_input(self.req.input_path)
            duration = probe_duration(self.req.input_path)
            out_dir = self._make_out_dir()
            wav = self._extract(duration, out_dir)
            vocals = self._separate(wav, out_dir)
            context = self._analyze(vocals, duration)
            if self.req.lyrics_text.strip():
                lines = self._align_lyrics(vocals, context)
                mode = "align"
            else:
                lines, songs = self._recognize(vocals, duration, context)
                mode = "blind"
            self._finalize(lines)
            # 文本级校正：先弹幕旁路（专名），再歌词库（整句）。
            # 顺序有意为之——弹幕能纠正昵称/术语这类专名，歌词库处理整句
            # 文本；反过来做会让歌词替换后的文本再被弹幕改动，难以复现。
            lines, corrections = self._post_correct(lines, context)
            files = self._export(out_dir, lines, context)
            report = self._evaluate(lines, duration)
            self.timings["total"] = round(time.time() - t_all, 2)
            self._progress("export", 1.0, "全部完成")
            return PipelineResult(
                mode=mode, duration=duration, lines=lines, files=files,
                warnings=self.warnings, out_dir=str(out_dir),
                songs=context.songs, speakers=[],
                evaluation=report, timings=dict(self.timings),
                bgm_flags=_flags_payload(context.bgm_decisions),
                live_summary=_live_summary(context.live),
                corrections=corrections,
            )
        except TaskCancelled:
            raise
        except (AudioProcessError,) as exc:
            raise PipelineError(f"{self._stage}失败：{exc}") from exc

    # ---------------- 各阶段 ----------------

    def _make_out_dir(self) -> Path:
        stem = Path(self.req.input_path).stem
        out = Path(self.cfg.work_dir) / stem / time.strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _extract(self, duration: float, out_dir: Path) -> str:
        """抽取 16k 单声道 WAV。

        注意：抽取的 WAV 是**可再生成的中间产物**（3 小时直播约 700MB），
        因此写到 `cache/` 而不是输出目录——输出目录只放字幕成品，
        否则会随任务数量无限膨胀（实测踩过：outputs 里堆积 109MB WAV）。
        没有配置 cache 目录时退回输出目录（保持旧行为）。
        """
        from core.audio import extract_wav

        self._stage = "音频抽取"
        self._progress("extract", 0.0, "正在抽取音频…")
        t0 = time.time()
        cache = getattr(self.cfg, "cache_dir", "") or ""
        target = Path(cache) if cache else out_dir
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception:
            target = out_dir
        wav = str(extract_wav(self.req.input_path, target))
        self.timings["extract"] = round(time.time() - t0, 2)
        self._progress("extract", 1.0, "音频抽取完成")
        return wav

    def _separate(self, wav: str, out_dir: Path) -> str:
        if not self.req.enable_separation:
            return wav
        from core.separator import VocalSeparator

        self._stage = "人声分离"
        self._progress("separate", 0.0, "正在分离人声…")
        t0 = time.time()
        sep = VocalSeparator(
            device=self._device(),
            on_progress=lambda r: self._progress("separate", r, "正在分离人声…"),
            on_warning=self._warn, cancel_event=self.cancel_event)
        vocals = sep.separate(wav, str(out_dir))
        self.timings["separate"] = round(time.time() - t0, 2)
        self._progress("separate", 1.0, "人声分离完成")
        return vocals

    def _analyze(self, wav: str, duration: float) -> "Context":
        """语音分析：VAD、声纹、演唱块、歌曲识别。"""
        from core.voiceprint import speech_intervals

        ctx = Context()
        need = (self.req.enable_voiceprint or self.req.enable_song_detect)
        self._load_live_log(ctx)
        if not need:
            return ctx
        self._stage = "语音分析"
        self._progress("analyze", 0.0, "正在检测语音段…")
        t0 = time.time()
        segments = speech_intervals(wav)
        ctx.segments = segments
        if not segments:
            self._warn("未检测到语音活动")
            return ctx
        if self.req.enable_voiceprint:
            self._classify_streamer(wav, ctx)
        self.timings["analyze"] = round(time.time() - t0, 2)
        if self.req.enable_song_detect:
            self._detect_songs(wav, ctx)
        self._progress("analyze", 1.0, "语音分析完成")
        return ctx

    def _classify_streamer(self, wav: str, ctx: "Context") -> None:
        """判定哪些时间段是「主播现场」，并输出**细粒度**保留区间。

        关键设计（实测教训）：VAD 会把整段唱歌+说话合并成 20~30 秒的长段，
        若按段整块剔除，会连带丢掉主播实际的说话与跟唱（实测 155 行掉到
        50 行）。因此这里把每个 VAD 段再切成 ~5 秒的**子窗**，逐窗判定后
        合并 —— 这样 BGM 原唱与主播跟唱即使在同一段里也能分开。
        """
        from core import streamer as vp

        self._progress("analyze", 0.3, "正在比对本主播声纹…")
        name = self.cfg.voiceprint.profile_name or self.req.profile_name
        try:
            profile = vp.load_profile(name, self.cfg.profiles_dir)
        except vp.VoiceprintError:
            if not name:
                self._warn("未指定主播声纹，跳过声纹过滤（可先在界面注册）")
                return
            raise PipelineError(f"未找到主播「{name}」的声纹")

        engine = vp.VoiceprintEngine(device=self._device())
        # 双音色：主播唱歌音色与说话差异大，只有说话向量会把唱歌判成他人。
        if self.cfg.voiceprint.use_singing_profile and profile.sing is None:
            blocks = self._candidate_song_blocks(ctx)
            learned = vp.learn_singing_profile(engine, wav, profile, blocks)
            if learned is not None:
                profile = vp.StreamerProfile(
                    name=profile.name, speak=profile.speak, sing=learned)
                self._log("已从演唱块自学习唱歌音色（提升演唱段召回）")

        windows = _split_windows(ctx.segments, self.cfg.voiceprint.window_seconds,
                                 self.cfg.voiceprint.min_window_seconds)
        if not windows:
            return
        embs = engine.embed_segments(
            wav, windows,
            on_progress=lambda r: self._progress(
                "analyze", 0.3 + 0.4 * r, "正在比对本主播声纹…"))
        refs = profile.refs()
        matrix = np.stack([v for _lab, v in refs])
        labels = [lab for lab, _v in refs]
        sims = embs @ matrix.T

        keep_flags = self._bgm_filter(wav, windows, sims, labels, ctx)
        merged = _merge_windows(windows, keep_flags,
                                gap=self.cfg.voiceprint.merge_gap)
        ctx.streamer_segments = merged
        kept_sec = sum(e - s for s, e in merged)
        total_sec = sum(e - s for s, e in windows)
        dropped = sum(1 for k in keep_flags if not k)
        policy = self.cfg.voiceprint.bgm_policy
        self._log(f"声纹+BGM（策略 {policy}）：保留 {kept_sec:.0f}s / "
                  f"{total_sec:.0f}s，标注 {dropped} 个子窗疑似伴奏原唱")

    def _bgm_filter(self, wav: str, windows, sims, labels, ctx: "Context"
                    ) -> list[bool]:
        """逐子窗多因子判别，按策略决定「保留 / 剔除 / 标注」。

        策略（重要）：
        子窗级真值（71 主播窗 vs 7 原唱窗）上，声纹分仅 0.51 vs 0.42，
        最佳阈值准确率 0.76——**不足以支撑自动删除**（会误删 20~45% 的
        主播内容，实测把 155 行打到 77 行）。因此默认 `mark`：
          - 只把「疑似原唱」子窗记入复核清单（带原因与分数），**不删内容**；
          - 需要激进清理时可切 `filter`（只删强证据子窗）。
        这样 BGM 影响对用户可见（能定位、能批量删），但不会误伤主播的话。
        """
        from core import bgm

        policy = self.cfg.voiceprint.bgm_policy
        vcfg = self.cfg.voiceprint
        tuning = bgm.BgmTuning(threshold=vcfg.bgm_strict_threshold)
        out: list[bool] = []
        flagged = 0
        for i, (s, e) in enumerate(windows):
            row = sims[i]
            by = {lab: float(row[k]) for k, lab in enumerate(labels)}
            speak = by.get("speak", 0.0)
            sing = by.get("sing")
            vocal = max(speak, sing if sing is not None else 0.0)
            feat = bgm.segment_features(wav, s, e)
            support = self._context_support(ctx, s, e)
            dec = bgm.decide(feat, vocal, tuning, context_support=support)
            weak_voice = vocal < vcfg.threshold * 0.72
            if policy == "filter":
                keep = dec.keep and not weak_voice
            else:
                keep = True          # mark/off：不删内容
            out.append(keep)
            if not dec.keep and policy != "off":
                flagged += 1
                dec.reasons.append("已剔除" if not keep else "仅标注，未删除")
                ctx.bgm_decisions.append(dec)
                if vcfg.explain_bgm and flagged <= 30:
                    self._log(f"  疑似原唱 {s:.1f}-{e:.1f}s："
                              + "；".join(dec.reasons))
        if flagged > 30:
            self._log(f"  （其余 {flagged - 30} 处略，详见复核清单）")
        return out

    def _context_support(self, ctx: "Context", start: float, end: float) -> float:
        """外部上下文支持度：该段是否紧邻直播事件/点歌请求。

        礼物/SC/上舰后主播常口头致谢；点歌请求后主播会唱。这些时刻的
        语音更可能是主播本人，给一个小的正向偏置（不改变明显是原唱的段）。
        """
        live = getattr(ctx, "live", None)
        if live is None:
            return 0.0
        best = 0.0
        for t in live.event_times():
            if start - 8.0 <= t <= end + 2.0:
                best = max(best, 0.6)
        for t, _user, _name in live.song_requests():
            if start - 30.0 <= t <= end + 30.0:
                best = max(best, 0.4)
        return best

    def _candidate_song_blocks(self, ctx: "Context") -> list[tuple[float, float]]:
        """粗筛候选演唱块（供自学习唱歌音色用；无需精确）。"""
        from core import song as songmod

        try:
            return songmod.detect_song_blocks(
                ctx.segments,
                songmod.SongTuning(min_block_seconds=20.0, block_gap=15.0))
        except Exception:
            return []

    def _detect_songs(self, wav: str, ctx: "Context") -> None:
        from core import song as songmod

        tuning = songmod.SongTuning(
            min_block_seconds=self.cfg.song.min_block_seconds,
            block_gap=self.cfg.song.block_gap,
            accept_score=self.cfg.song.accept_score)
        self._progress("analyze", 0.6, "正在识别演唱段…")
        blocks = songmod.detect_song_blocks(ctx.segments, tuning)
        ctx.song_blocks = blocks
        if not blocks:
            self._log("未检测到符合条件的演唱段")
            return
        self._log(f"检测到 {len(blocks)} 个演唱块："
                  + "、".join(f"{s:.0f}-{e:.0f}s" for s, e in blocks[:8]))

    def _resolve_songs(self, wav: str, ctx: "Context") -> list:
        """演唱块 → 歌曲：弹幕点歌 → 本地库 → 在线多源 → 识曲 → 强制对齐。

        歌词来源优先级（成本递增，命中即停）：
        0. **弹幕/SC 点歌请求**（最便宜也最准：观众直接写了歌名，实测对
           翻唱有效，而 Shazam 对翻唱几乎无效）；
        1. 本地歌词库（用该块的 ASR 文本做相似度匹配，完全离线）
        2. 在线歌词库（NetEase / Kugou / LRCLIB，带自动换源）
        3. Shazam 识曲（仅原唱有效，作为补充）
        命中后用 LRC 行时间对齐；无同步歌词时才用时间戳预测模型。
        """
        from core import song as songmod

        self._stage = "歌曲识别"
        self._progress("recognize", 0.78, "正在识别歌曲…")
        if not ctx.transcript:
            ctx.transcript = self._transcript_segments(ctx)
        requests = self._song_requests(ctx)
        results = []
        for start, end in ctx.song_blocks:
            seg = songmod.SongSegment(start=start, end=end)
            transcript = ctx.transcript_in(start, end)
            found = None
            # 0) 弹幕点歌：优先用「本块前后 3 分钟内的点歌请求」
            hinted = self._nearest_request(requests, start, end)
            if hinted:
                found = self._search_by_request(hinted, transcript)
                if found:
                    self._log(f"依据弹幕点歌「{hinted}」定位歌曲："
                              f"《{found['title']}》")
            if found is None and self.cfg.song.search_by_transcript and len(
                    songmod.norm_text(transcript)) >= 8:
                found = self._search_local_lyrics(transcript) or \
                    self._search_online_lyrics(transcript)
            if found is None and self.cfg.song.use_shazam:
                found = self._shazam_song(wav, start, end)
            if found is None:
                self._log(f"演唱段 {start:.0f}-{end:.0f}s 未能识别歌曲，保留语音识别")
                results.append(seg)
                continue
            seg.title = found["title"]
            seg.artist = found.get("artist", "")
            seg.source = found.get("source", "")
            seg.score = found.get("score", 0.0)
            seg.plain_lyrics = found.get("lyrics") or ""
            seg.lyrics_lrc = seg.plain_lyrics if found.get("synced") else None
            # 有的源（NetEase）即使标为 synced，返回体里也可能混入元信息行，
            # 统一用 plain_rows 解析一次，保证后续对齐拿到干净歌词行。
            if seg.lyrics_lrc:
                clean = songmod.plain_rows(seg.lyrics_lrc, synced=True)
                if clean:
                    seg.lines = []
            if self.cfg.song.forced_align:
                self._align_song(wav, seg, transcript)
            results.append(seg)
        ctx.songs = results
        for s in results:
            if s.title and s.lines:
                self._log(f"《{s.title}》{s.artist}（{s.source}，"
                          f"{s.start:.0f}-{s.end:.0f}s，对齐 {len(s.lines)} 行）")
        return results

    def _song_requests(self, ctx: "Context") -> list[tuple[float, str, str]]:
        """歌曲线索 [(时间, 用户, 歌名)]。

        优先级：**用户手动录入的清单 > 弹幕点歌请求**。

        为什么清单放最前：用户明确知道本场唱了什么，比从弹幕猜可靠得多；
        把清单作为「时间未知的请求」注入后，后续 `_nearest_request` 仍能
        按演唱块就近匹配，而且在弹幕缺失（没导 XML）时也能工作。
        """
        manual = [str(t).strip() for t in
                  (getattr(self.cfg.song, "manual_titles", ()) or ())
                  if str(t).strip()]
        # 时间给 -1 表示「不确定时间」，_nearest_request 会把它当兜底候选
        from_manual = [(-1.0, self.MANUAL_SOURCE, name) for name in manual]

        live = getattr(ctx, "live", None)
        if live is None:
            return from_manual
        try:
            return from_manual + list(live.song_requests())
        except Exception:
            return from_manual

    # 标记手动录入来源，便于日志区分
    MANUAL_SOURCE = "@manual"

    def _nearest_request(self, requests, start: float, end: float) -> str:
        """取本演唱块附近（前 5 分钟 ~ 后 1 分钟）最近的一条点歌请求。

        为什么看「之前」：观众先点歌、主播随后唱，所以请求通常早于演唱。
        """
        best, best_gap = "", 1e9
        for t, _user, name in requests:
            if not (start - 300.0 <= t <= end + 60.0):
                continue
            gap = abs(t - start)
            if gap < best_gap:
                best, best_gap = name, gap
        return best

    def _search_by_request(self, name: str, transcript: str
                           ) -> Optional[dict]:
        """用弹幕点歌的歌名检索歌词（先本地库，再在线多源）。"""
        from core import song as songmod

        found = self._search_local_lyrics(name, title=name)
        if found:
            return found
        try:
            rows = songmod.search_lyrics(
                title=name, transcript=transcript,
                providers=self.cfg.song.lyric_providers,
                min_score=min(0.35, self.cfg.song.accept_score),
                on_log=self._log)
        except Exception as exc:
            self._log(f"点歌歌词检索失败：{exc}")
            return None
        if not rows:
            return None
        best = rows[0]
        # 版本再复核：同一歌名的多个版本里，用转写文本挑行序一致的那版
        best_lines = songmod.plain_rows(best["lyrics"],
                                        synced=bool(best.get("synced")))
        best_score = _contiguous_coverage(best_lines, transcript)
        for row in rows[1:6]:
            lines = songmod.plain_rows(row["lyrics"],
                                       synced=bool(row.get("synced")))
            score = _contiguous_coverage(lines, transcript)
            if score > best_score:
                best, best_score = row, score
        return {"title": best["title"], "artist": best["artist"],
                "lyrics": best["lyrics"], "synced": best["synced"],
                "source": best["source"] + "+点歌", "score": best["score"]}

    def _search_local_lyrics(self, transcript: str, title: str = ""
                             ) -> Optional[dict]:
        """本地歌词库：用 ASR 文本与库内歌词做相似度匹配（离线）。"""
        from core import song as songmod

        store = Path(self.cfg.lyrics_dir)
        if not store.is_dir():
            return None
        candidates = []
        for path in store.glob("*.json"):
            try:
                import json as _json

                data = _json.loads(path.read_text(encoding="utf-8"))
                body = (data.get("synced") or data.get("plain") or "").strip()
                if body:
                    candidates.append((data.get("title") or path.stem,
                                       data.get("artist") or "", body))
            except Exception:
                continue
        if not candidates:
            return None
        # 有歌名提示（来自弹幕点歌）时，先按歌名过滤，再按文本打分；
        # 否则纯按转写文本匹配整库。
        if title:
            key = songmod.norm_text(title)
            narrowed = [c for c in candidates if key and (
                key in songmod.norm_text(c[0]) or songmod.norm_text(c[0]) in key)]
            if narrowed:
                candidates = narrowed
            else:
                return None
        ranked = songmod.rank_lyrics(transcript, candidates, top=1)
        if not ranked or ranked[0][0] < self.cfg.song.accept_score:
            return None
        score, rtitle, rartist, body = ranked[0]
        return {"title": rtitle, "artist": rartist, "lyrics": body,
                "synced": bool(body.lstrip().startswith("[")), "source": "local",
                "score": score}

    def _search_online_lyrics(self, transcript: str) -> Optional[dict]:
        from core import song as songmod

        if not self.cfg.song.use_online_lyrics:
            return None
        try:
            rows = songmod.search_lyrics(
                transcript=transcript,
                providers=self.cfg.song.lyric_providers,
                min_score=self.cfg.song.accept_score)
        except Exception as exc:
            self._log(f"在线歌词检索失败：{exc}")
            return None
        if not rows:
            return None
        # 同一首歌的多个版本（原版/现场/编曲）歌词行序与行切分都不同，
        # 直接取第一个会导致整体错位。这里对候选版本做「行级连续覆盖」复核：
        # 用转写文本能在该版本里定位到的**连续歌词段**覆盖了多少转写字符，
        # 覆盖最多者胜（对行切分差异也稳健）。
        best, best_score = None, -1.0
        for row in rows[:8]:
            lines = songmod.plain_rows(row["lyrics"],
                                       synced=bool(row.get("synced")))
            score = _contiguous_coverage(lines, transcript)
            if score > best_score:
                best, best_score = row, score
        best = best or rows[0]
        return {"title": best["title"], "artist": best["artist"],
                "lyrics": best["lyrics"], "synced": best["synced"],
                "source": best["source"], "score": best["score"]}

    def _shazam_song(self, wav: str, start: float, end: float
                     ) -> Optional[dict]:
        """Shazam 识曲（多位置探测，翻唱命中率低，仅作补充）。

        效率保护：Shazam 是**网络**调用，单次最长等 25 秒。
        若连续多次失败（无网络 / 被限流），继续逐块重试只会让整个任务
        卡在无意义的超时上——实测 72 分钟素材最多会浪费十几分钟。
        因此失败累积到阈值后直接停用识曲（后续块不再调用）。
        """
        if self._shazam_disabled:
            return None
        import asyncio

        try:
            from shazamio import Shazam
        except ImportError:
            self._shazam_disabled = True
            return None
        snippet = min(self.cfg.song.shazam_snippet, max(6.0, end - start))
        # 探测位置：块内均匀取 shazam_probes 个点（翻唱常在副歌才被识别到）
        n_probes = max(1, int(self.cfg.song.shazam_probes))
        positions = []
        for k in range(n_probes):
            frac = (k + 0.5) / n_probes
            pos = start + (end - start - snippet) * frac
            if pos >= start - 1e-6:
                positions.append(round(pos, 2))

        async def probe() -> Optional[dict]:
            sh = Shazam()
            for pos in positions:
                clip = self._cut(wav, pos, pos + snippet)
                try:
                    raw = await asyncio.wait_for(sh.recognize(clip), timeout=25)
                except Exception:
                    self._shazam_fails += 1
                    if self._shazam_fails >= SHAZAM_FAIL_LIMIT:
                        self._shazam_disabled = True
                        self._log("识曲连续失败，已跳过后续识曲"
                                  "（歌词库检索不受影响）")
                        return None
                    continue
                track = (raw or {}).get("track") or {}
                title = (track.get("title") or "").strip()
                if title:
                    self._shazam_fails = 0
                    return {"title": title,
                            "artist": (track.get("subtitle") or "").strip(),
                            "lyrics": "", "synced": False, "source": "shazam",
                            "score": float(track.get("confidence") or 0) / 100}
            return None

        try:
            return asyncio.run(probe())
        except Exception:
            self._shazam_fails += 1
            if self._shazam_fails >= SHAZAM_FAIL_LIMIT:
                self._shazam_disabled = True
            return None

    def _align_song(self, wav: str, seg, transcript: str) -> None:
        """歌词 → 逐行时间。

        优先级（实测决定）：
        1. **LRC 同步歌词**：直接给出每行的真实演唱时刻（相对歌曲起点），
           配合「本块从第几行开始唱」的估计做偏移，精度最高；
        2. FunASR 时间戳预测（强制对齐）：在没有同步歌词时使用，但它对
           整段歌词会给出**均匀铺开**的时间（实测 48 字撑满 4 分钟），
           只适合作为「知道唱到哪几行」的近似；
        3. 都没有则不生成歌词行，保留 ASR 文本。
        """
        from core import song as songmod

        lyr_lines = songmod.plain_rows(seg.plain_lyrics or "",
                                       synced=bool(seg.lyrics_lrc))
        if not lyr_lines:
            return
        start_idx = _best_lrc_offset(lyr_lines, transcript)

        if seg.lyrics_lrc:
            rows = songmod.lrc_rows(seg.lyrics_lrc)
            if rows:
                self._align_from_lrc(seg, rows, lyr_lines, start_idx, transcript)
                return
        if not self.cfg.song.forced_align:
            return
        self._progress("recognize", 0.88, f"正在对齐《{seg.title}》…")
        block_wav = self._cut(wav, seg.start, seg.end)
        window = _window_for_block(lyr_lines, start_idx, transcript)
        if not window:
            return
        chars = songmod.forced_align_text(
            block_wav, "".join(window), offset=seg.start,
            model_id=self.cfg.song.align_modelscope_id, device=self._device())
        if chars:
            seg.lines = songmod.lines_from_alignment(chars, window)
            seg.aligned = False

    def _align_from_lrc(self, seg, rows, lyr_lines: list[str],
                        start_idx: int, transcript: str) -> None:
        """用 LRC 行时间 + 行级匹配窗口生成本块歌词行。

        只保留「转写文本实际唱到」的歌词行（见 _match_lrc_window），
        时间 = 块起点 + (该行 LRC 时间 − 窗口首行 LRC 时间)。
        """
        from core.song import norm_text

        idx = _match_lrc_window(lyr_lines, transcript)
        if not idx:
            return
        wanted = {norm_text(lyr_lines[i]) for i in idx}
        first_text = norm_text(lyr_lines[idx[0]])
        base = None
        picked: list[tuple[float, str]] = []
        for t, body in rows:
            key = norm_text(body)
            if base is None:
                if key == first_text:
                    base = t
                    picked.append((t, body))
                continue
            if key in wanted:
                picked.append((t, body))
            if len(picked) >= len(idx):
                break
        if base is None or not picked:
            return
        lines = []
        for i, (t, body) in enumerate(picked):
            s = seg.start + (t - base)
            if s > seg.end + 1.0:
                break
            e = (seg.start + (picked[i + 1][0] - base) if i + 1 < len(picked)
                 else s + 4.0)
            lines.append(SubtitleLine(words=[SubtitleWord(
                text=body, start=round(s, 3),
                end=round(min(max(e, s + 0.4), seg.end + 1.0), 3))]))
        seg.lines = lines
        seg.aligned = True

    def _transcript_segments(self, ctx: "Context"
                             ) -> list[tuple[float, float, str]]:
        """从已识别结果中恢复「时间段 → 文本」视图（供歌曲检索用）。"""
        out = []
        for seg in getattr(ctx, "raw_segments", []) or []:
            out.append((seg.start, seg.end, seg.text))
        return out

    def _recognize(self, wav: str, duration: float, ctx: "Context"
                   ) -> tuple[list[SubtitleLine], list]:
        """盲识别主链路：ASR → 歌曲替换 → 精修 → 切行。"""
        self._stage = "语音识别"
        segs = self._asr(wav, duration, ctx)
        ctx.raw_segments = segs
        lines = self._lines_from_segments(segs, ctx)
        songs: list = []
        if ctx.song_blocks:
            songs = self._resolve_songs(wav, ctx)
            if songs:
                lines = self._merge_songs(lines, songs)
        self._repair_text(lines)
        self._progress("recognize", 0.9, "正在精修时间轴…")
        self._refine(lines, wav)
        return lines, songs

    def _asr(self, wav: str, duration: float, ctx: "Context"):
        from core.providers import get_provider, load_plugin_modules

        load_plugin_modules(self.cfg.provider.plugin_modules)
        asr = self.cfg.asr
        kwargs = {}
        if asr.engine == "whisper":
            kwargs = dict(
                model=asr.model, device=self._device(),
                compute_type=self._compute_type(), models_dir=self.cfg.models_dir,
                vad_filter=asr.vad_filter, word_timestamps=asr.word_timestamps,
                beam_size=asr.beam_size, best_of=asr.best_of,
                temperature=asr.temperature,
                condition_on_previous_text=asr.condition_on_previous_text,
                chunk_seconds=asr.chunk_seconds, chunk_overlap=asr.chunk_overlap,
                initial_prompt=self._hotword_prompt(ctx))
        elif asr.engine == "funasr":
            kwargs = dict(device=self._device(), hotwords=asr.hotwords)
        elif asr.engine == "external":
            kwargs = dict(endpoint=self.cfg.provider.asr_endpoint,
                          api_key=self.cfg.provider.asr_api_key,
                          timeout=self.cfg.provider.timeout)
        self._progress("recognize", 0.0, "正在识别语音…")
        t0 = time.time()
        provider = get_provider(asr.engine, **kwargs)
        segments = provider.transcribe(
            wav, language=asr.language, total_duration=duration,
            on_progress=lambda r: self._progress(
                "recognize", 0.75 * r, "正在识别语音…"),
            cancel_event=self.cancel_event)
        self.timings["asr"] = round(time.time() - t0, 2)
        return [s for s in segments if self._keep_segment(s)]

    def _keep_segment(self, seg) -> bool:
        """剔除 whisper 在纯音乐/静音上的幻听段。"""
        asr = self.cfg.asr
        if seg.compression_ratio > asr.max_compression_ratio:
            return False
        if seg.no_speech_prob > asr.drop_no_speech_above:
            return False
        if seg.avg_logprob and seg.avg_logprob < asr.min_avg_logprob:
            return False
        return bool(seg.words)

    def _lines_from_segments(self, segs, ctx: "Context") -> list[SubtitleLine]:
        from core.segmenter import SegmentTuning, segment_words

        words: list[SubtitleWord] = []
        keep = ctx.streamer_segments if ctx.streamer_segments else None
        for seg in segs:
            if keep is not None and not any(
                    min(seg.end, b) > max(seg.start, a) for a, b in keep):
                continue
            for w in seg.words:
                words.append(SubtitleWord(text=w.text, start=w.start, end=w.end))
        if not words:
            return []
        hints = self._punctuation_hints(words)
        tuning = SegmentTuning(
            target_chars=self.cfg.segment.target_chars,
            max_chars=self.cfg.segment.max_chars,
            min_chars=self.cfg.segment.min_chars,
            max_duration=self.cfg.segment.max_duration,
            w_sentence=self.cfg.segment.w_sentence,
            w_clause=self.cfg.segment.w_clause,
            w_pause=self.cfg.segment.w_pause,
            w_dev=self.cfg.segment.w_dev,
            w_duration=self.cfg.segment.w_duration,
            pause_gap=self.cfg.segment.pause_gap,
            inner_space=getattr(self.cfg.segment, "inner_space", True))
        # 歌词/弹幕给出的断句先验（强于普通标点）。
        # 这一阶段还不知道「这段属于哪首歌」，因此用两种来源：
        #   1) 用户手动录入的曲目（最可靠）
        #   2) 与词流时间重叠的弹幕文本
        lb = self._lyric_hints(words, ctx)
        return segment_words(words, tuning, punctuation=hints,
                             lyric_bounds=lb)

    def _lyric_hints(self, words: list[SubtitleWord],
                     ctx: "Context") -> Optional[list[bool]]:
        """歌词/弹幕断句线索（失败时安静退化为 None）。"""
        if not getattr(self.cfg.song, "segment_by_lyrics", True):
            return None
        try:
            from .lyrics_library import (LyricLibrary, danmaku_boundary_hints,
                                         lyric_boundary_hints)
        except Exception:
            return None

        # 优先：手动录入的曲目在曲库里有歌词
        titles = list(getattr(self.cfg.song, "manual_titles", ()) or ())
        if titles:
            try:
                lib = LyricLibrary(self.cfg.song.lyric_library_dir)
                for t in titles:
                    doc = lib.find_by_title(str(t))
                    if doc is None:
                        continue
                    h = lyric_boundary_hints(words, doc)
                    if h:
                        self._log(f"断句参考歌词《{doc.title}》")
                        return h
            except Exception:
                pass
        # 退化：用同期弹幕中反复出现的短句
        live = getattr(ctx, "live", None)
        if live is not None:
            try:
                return danmaku_boundary_hints(words, live)
            except Exception:
                return None
        return None

    def _punctuation_hints(self, words: list[SubtitleWord]) -> Optional[list[bool]]:
        """用标点模型给切行提供句读线索（不写进最终字幕）。

        线索按「字符」给出：汉字一字一项，拉丁/数字词整词一项——与
        segmenter.split_atoms 的原子划分完全一致（同一函数产出）。
        关闭标点（segment.use_punctuation=False）或模型不可用时返回 None，
        切行退化为「长度 + 停顿」。
        """
        if not self.cfg.segment.use_punctuation or not words:
            return None
        from core import punctuate
        from core.segmenter import split_atoms

        atoms = split_atoms(words)
        if not atoms:
            return None
        # 字符级线索 → 原子级线索：每个原子取其最后一个字符处的断句标记
        source = "".join(a.text.strip() for a in atoms)
        if len(source) < 4:
            return None
        self._progress("recognize", 0.8, "正在恢复句读…")
        punctuated = punctuate.restore(source, device=self._device())
        if not punctuated or punctuated == source:
            return None
        char_flags = punctuate.boundary_flags(punctuated, source)
        if len(char_flags) != len(source):
            return None
        flags: list[bool] = []
        pos = 0
        for a in atoms:
            n = len(a.text.strip())
            end = min(len(char_flags) - 1, pos + max(0, n) - 1)
            flags.append(bool(char_flags[end]) if end >= 0 else False)
            pos += n
        return flags if len(flags) == len(atoms) else None

    # ---------------- 后处理 ----------------

    def _repair_text(self, lines: list[SubtitleLine]) -> None:
        from core.text import t2s

        for ln in lines:
            for w in ln.words:
                w.text = t2s(w.text)

    def _refine(self, lines: list[SubtitleLine], wav: str) -> None:
        """时间轴校正：全片互相关估计整体偏移并补正。

        为什么用互相关而不是逐行起振点：
        - 逐行起振点在偏移较大时会**饱和**（实测对 -0.3s 与 -0.7s 给出
          相同结果），无法给出正确的修正量；
        - 全片互相关把「字幕覆盖区间曲线」与「音频活动度曲线」对齐，
          统计所有行的共同偏移，抗单行错误、不会饱和。
        实测：无偏（输入 +0.4s 偏移估出 -0.30s 的修正量），把生成结果的
        起点 ±0.3s 命中率从 47% 提升到 88%。
        """
        if not lines or not self.cfg.timing.refine:
            return
        from core.timing import apply_shift, estimate_shift_xcorr

        tcfg = self.cfg.timing
        t0 = time.time()
        shift, corr = estimate_shift_xcorr(
            lines, wav, frame_ms=tcfg.frame_ms, lo=-1.2, hi=1.2)
        # 增益 0.7：互相关把「字幕覆盖曲线」对齐到「音频活动曲线」，
        # 后者包含尾音与混响拖尾，估计值系统性偏大约 40%；实测 0.6~0.75
        # 区间内时间匹配率最高且平稳，取中值 0.7。
        shift = round(shift * 0.7, 3)
        if abs(shift) >= 0.06:
            apply_shift(lines, shift)
            self._log(f"时间轴整体校正 {shift:+.3f}s"
                      f"（互相关 r={corr:.3f}，模型时间系统性偏早）")
        self.timings["refine"] = round(time.time() - t0, 2)

    def _merge_songs(self, lines: list[SubtitleLine], songs: list
                     ) -> list[SubtitleLine]:
        """用歌词行替换演唱块内的 ASR 行（歌词文本比 ASR 准）。"""
        song_lines: list[SubtitleLine] = []
        spans = []
        for s in songs:
            if not s.lines:
                continue
            song_lines.extend(s.lines)
            spans.append((s.start, s.end))
        if not song_lines:
            return lines
        kept = [ln for ln in lines
                if not any(ln.start < e and a < ln.end for a, e in spans)]
        return sorted(kept + song_lines, key=lambda x: x.start)

    def _post_correct(self, lines: list[SubtitleLine],
                      ctx: "Context") -> tuple[list[SubtitleLine], dict]:
        """生成后文本校正：歌词库 + 弹幕旁路。

        两步都是「可选且可关闭」的：任一失败只记录警告，绝不中断主流程
        （用户已经等了几分钟，不该因为校正模块的问题丢掉识别结果）。
        """
        from .danmaku_correct import correct_with_log
        from .lyrics_library import (LyricLibrary, apply_report, best_document)

        out = lines
        detail: dict = {}

        # ---- 1) 弹幕/礼物/SC 专名校正 ----
        vp = self.cfg.voiceprint
        sc = self.cfg.song
        if getattr(sc, "danmaku_correct", False):
            if ctx.live is None:
                detail["danmaku"] = {"applied": 0,
                                     "reason": "未提供弹幕 XML"}
            else:
                try:
                    self._progress("correct", 0.0, "用弹幕校正专名…")
                    out, rep = correct_with_log(
                        out, ctx.live,
                        window=float(getattr(sc, "danmaku_time_window", 12.0)))
                    detail["danmaku"] = rep.to_dict()
                    self._log(f"弹幕校正：{rep.reason}")
                except Exception as exc:  # noqa: BLE001
                    self.warnings.append(f"弹幕校正失败：{exc}")
                    detail["danmaku"] = {"applied": 0, "reason": str(exc)}
            _ = vp

        # ---- 2) 歌词库整句替换 ----
        if getattr(sc, "lyric_correct", False):
            try:
                lib = LyricLibrary(sc.lyric_library_dir)
                if lib.size == 0:
                    detail["lyrics"] = {"applied": 0,
                                        "reason": "歌词库为空"}
                else:
                    self._progress("correct", 0.5, "用歌词库校正文本…")
                    titles = list(getattr(sc, "manual_titles", ()) or ())
                    rep = best_document(
                        out, lib, titles=titles or None,
                        threshold=float(getattr(sc, "lyric_doc_threshold", 0.55)),
                        on_log=self._log)
                    if rep is None or not rep.matches:
                        detail["lyrics"] = {
                            "applied": 0,
                            "reason": (rep.reason if rep else "曲库中无匹配歌曲")}
                    else:
                        out, n = apply_report(
                            out, rep,
                            min_score=float(getattr(sc, "lyric_line_threshold", 0.75)))
                        info = rep.to_dict()
                        info["applied"] = n
                        detail["lyrics"] = info
                        self._log(
                            f"歌词校正：《{info['title']}》 相似度 "
                            f"{info['score']:.2f}，替换 {n} 行")
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(f"歌词校正失败：{exc}")
                detail["lyrics"] = {"applied": 0, "reason": str(exc)}

        if detail:
            self._progress("correct", 1.0, "文本校正完成")
        return out, detail

    def _finalize(self, lines: list[SubtitleLine]) -> None:
        """收尾：去重、去空行、排序、修正非法时间。

        去重针对「同一句话在相邻时间被重复输出」——whisper 在歌声/回音
        上会重复生成同一句（实测占比不低）。判据：文本归一后相同或互相
        包含，且时间重叠或紧邻（< 1.5s）。
        """
        clean: list[SubtitleLine] = []
        for ln in lines:
            ln.words = [w for w in ln.words if w.text]
            if not ln.words:
                continue
            for w in ln.words:
                if w.end <= w.start:
                    w.end = w.start + 0.01
            clean.append(ln)
        clean.sort(key=lambda ln: ln.start)
        # 跨块重复：同一句话在**短时间内**被重复输出（whisper 在歌声/回声
        # 上会重复生成同一句）。只在「不与上一次出现时间重叠、且相隔
        # ≤ 25 秒」时才判为重复——真正的副歌相隔更久，不能误删。
        # 注意：只比较「到上一次出现」的间隔，绝不改写保留行的结束时间
        # （否则会出现跨越数分钟的超长行，破坏时间轴单调性）。
        deduped: list[SubtitleLine] = []
        seen: dict[str, list[SubtitleLine]] = {}
        for ln in clean:
            key = _norm_key(ln.text)
            if key and len(key) >= 4:
                prev = None
                for cand in seen.get(key, []):
                    if cand.end <= ln.start + 0.05 and ln.start - cand.end <= 25.0:
                        prev = cand
                        break
                if prev is not None:
                    continue
                seen.setdefault(key, []).append(ln)
            # 相邻包含关系（"要一直补色" vs "要一直补色要一直用那个"）
            if key and deduped:
                prow = deduped[-1]
                pkey = _norm_key(prow.text)
                if (ln.start - prow.end < 1.0 and len(key) >= 3 and pkey
                        and (key in pkey or pkey in key)):
                    if len(key) > len(pkey):
                        prow.words = list(ln.words)
                    continue
            deduped.append(ln)
        deduped.sort(key=lambda x: x.start)
        lines[:] = deduped
        # 收尾：消除重叠。相邻行边界吸附后可能残留 10ms 级交叠，播放器会
        # 出现两行同时显示；对「零时长词」（退化间隔）也要给最小可读时长。
        for i in range(len(lines) - 1):
            a, b = lines[i], lines[i + 1]
            if not (a.words and b.words):
                continue
            if a.words[-1].end > b.words[0].start:
                a.words[-1].end = round(max(a.words[0].start + 0.05,
                                            b.words[0].start), 3)
            # 下一行起点不得早于本行结束（避免零/负时长行）
            if b.words[0].start < a.words[-1].end:
                b.words[0].start = a.words[-1].end
            if b.words[0].end <= b.words[0].start:
                b.words[0].end = round(b.words[0].start + 0.05, 3)

    # ---------------- 导出 ----------------

    def _export(self, out_dir: Path, lines: list[SubtitleLine],
                ctx: "Context") -> dict[str, str]:
        from core.subtitle import to_ass, to_json, to_lrc, to_srt, to_vtt

        self._stage = "字幕导出"
        self._progress("export", 0.0, "正在导出字幕…")
        stem = Path(self.req.input_path).stem
        title = self.req.title or stem
        writers = {
            "srt": lambda ls: to_srt(ls),
            "lrc": lambda ls: to_lrc(ls, title=title),
            "ass": lambda ls: to_ass(ls),
            "vtt": lambda ls: to_vtt(ls),
            "json": lambda ls: to_json(ls, title=title),
        }
        files: dict[str, str] = {}
        for fmt in self.req.formats:
            fn = writers.get(fmt)
            if fn is None:
                continue
            path = out_dir / f"{stem}.{fmt}"
            path.write_text(fn(lines), encoding="utf-8")
            files[fmt] = str(path)
        if ctx.songs:
            md = out_dir / "songs_timeline.md"
            md.write_text(_songs_md(ctx.songs), encoding="utf-8")
            files["songs_md"] = str(md)
        self._progress("export", 0.5, "已导出字幕文件")
        return files

    def _evaluate(self, lines: list[SubtitleLine], duration: float
                  ) -> Optional[EvalReport]:
        """与人工校对样本比对（若提供了参考 SRT）。

        只比较本任务覆盖的时间范围（duration），避免片段评测时把参考样本
        里「更晚的段落」误记为漏行。
        """
        ref = self.req.reference_srt
        if not ref or not Path(ref).is_file():
            return None
        from core.evaluate import EvalLine

        hyp = [EvalLine(start=ln.start, end=ln.end, text=ln.text) for ln in lines]
        ref_lines = [r for r in load_srt(ref) if r.start < duration]
        report = evaluate(ref_lines, hyp)
        out = Path(self.cfg.work_dir) / "_eval_last"
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(report.summary_md(), encoding="utf-8")
        import json as _json

        (out / "report.json").write_text(
            _json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
        return report

    # ---------------- 小工具 ----------------

    def _device(self) -> str:
        if self.cfg.device != "auto":
            return self.cfg.device
        from core.env import detect_env

        return detect_env().device

    def _compute_type(self) -> str:
        if self.cfg.compute_type:
            return self.cfg.compute_type
        from core.env import detect_env

        return detect_env().compute_type

    def _cut(self, wav: str, start: float, end: float) -> str:
        """截取子区间到临时文件（送入识曲/对齐模型）。"""
        import subprocess
        import tempfile

        fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        fd.close()
        ffmpeg = "ffmpeg"
        from core.env import detect_env

        env = detect_env()
        if env.ffmpeg_path:
            ffmpeg = env.ffmpeg_path
        subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-ss", str(start),
             "-t", str(max(1.0, end - start)), "-i", wav,
             "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", fd.name],
            capture_output=True)
        return fd.name

    def _hotword_prompt(self, ctx: "Context") -> str:
        if ctx.live is None:
            return self.cfg.asr.hotwords
        words = ctx.live.hotwords(top=40)
        return " ".join(words) if words else self.cfg.asr.hotwords

    def _load_live_log(self, ctx: "Context") -> None:
        from core.livexml import load_log

        if not self.req.live_xml:
            return
        log = load_log(self.req.live_xml)
        if log is None:
            self._warn(f"弹幕文件解析失败或格式不支持：{self.req.live_xml}")
            return
        ctx.live = log
        self._log(f"已载入弹幕 {len(log.danmaku)} 条 / 礼物 {len(log.gifts)} 条")

    def _warn(self, message: str) -> None:
        self.warnings.append(message)
        self._log(message)

    def _log(self, message: str) -> None:
        if self.on_log:
            self.on_log(message)

    def _progress(self, stage: str, ratio: float, message: str) -> None:
        if self.on_progress:
            base = _STAGE_BASE.get(stage, 0.0)
            span = _STAGE_SPAN.get(stage, 0.0)
            self.on_progress(base + span * min(1.0, max(0.0, ratio)), message)


@dataclass
class Context:
    """一次任务内的分析上下文（避免层层传参）。"""

    segments: list[tuple[float, float]] = field(default_factory=list)
    streamer_segments: list[tuple[float, float]] = field(default_factory=list)
    song_blocks: list[tuple[float, float]] = field(default_factory=list)
    songs: list = field(default_factory=list)
    live: object = None
    transcript: list[tuple[float, float, str]] = field(default_factory=list)
    raw_segments: list = field(default_factory=list)
    bgm_decisions: list = field(default_factory=list)   # BGM 判别明细（UI 展示）

    def transcript_in(self, start: float, end: float) -> str:
        return "".join(t for s, e, t in self.transcript
                       if s < end and e > start)


def _window_for_block(lyric_lines: list[str], start_idx: int,
                      transcript: str) -> list[str]:
    """取「本演唱块实际唱到」的歌词窗口（按行级匹配结果）。"""
    idx = _match_lrc_window(lyric_lines, transcript)
    if idx:
        return [lyric_lines[i] for i in idx]
    # 匹配不到（转写太差）：退回按字数推算窗口
    from core.song import norm_text

    if not lyric_lines:
        return []
    n_chars = len(norm_text(transcript))
    if n_chars <= 0:
        return lyric_lines[start_idx:start_idx + 8]
    start_idx = max(0, min(start_idx, len(lyric_lines) - 1))
    out: list[str] = []
    total = 0
    for line in lyric_lines[start_idx:]:
        out.append(line)
        total += len(norm_text(line))
        if total >= n_chars * 1.15:
            break
    return out or lyric_lines[start_idx:start_idx + 8]


def _contiguous_coverage(lyric_lines: list[str], transcript: str) -> float:
    """歌词某一版本对转写的「连续覆盖」得分（0~1，越高说明越是同一版本）。

    做法：把整个歌词按行拼成串，与转写做字符级比对，取最长的连续匹配
    段长度 / 转写长度。原版歌词行序与直播所唱一致时该值高；现场版/编曲
    版即使文字相同，行序变了也会显著下降。
    """
    from difflib import SequenceMatcher

    from core.song import norm_text

    body = norm_text(transcript)
    text = "".join(norm_text(x) for x in lyric_lines)
    if not body or not text:
        return 0.0
    sm = SequenceMatcher(None, body, text, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 4]
    if not blocks:
        return 0.0
    longest = max(b.size for b in blocks)
    matched = sum(b.size for b in blocks)
    return round(0.5 * longest / len(body) + 0.5 * matched / len(body), 4)


def _match_lrc_window(lyric_lines: list[str], transcript: str
                      ) -> list[int]:
    """转写文本与歌词逐行匹配 → 实际唱到的行号列表（升序）。

    做法：对每一行歌词，计算它包含的字符有多少出现在转写中（字符级
    多重集交集占比），并把「连续命中」的相邻行合并成段；取覆盖转写
    字符最多的那一段。这样对「原版 / 现场版 / 编曲版」的歌词都能定位，
    不依赖行数推算。
    """
    from core.song import norm_text

    body = norm_text(transcript)
    if not body or not lyric_lines:
        return []
    pool: dict[str, int] = {}
    for ch in body:
        pool[ch] = pool.get(ch, 0) + 1
    hit: list[bool] = []
    covered = 0
    for line in lyric_lines:
        keys = norm_text(line)
        if not keys:
            hit.append(False)
            continue
        c = 0
        for ch in keys:
            if pool.get(ch, 0) > 0:
                c += 1
        ratio = c / max(1, len(keys))
        hit.append(ratio >= 0.6)
    if not any(hit):
        return []
    # 连续命中段（允许 1 行缺口）
    segments: list[list[int]] = []
    cur: list[int] = []
    gap = 0
    for i, ok in enumerate(hit):
        if ok:
            if gap and cur:
                cur.extend(range(len(cur) + sum(1 for _ in range(gap))))
            cur.append(i)
            gap = 0
        elif cur:
            gap += 1
            if gap > 1:
                segments.append(cur)
                cur, gap = [], 0
    if cur:
        segments.append(cur)
    if not segments:
        return []
    best = max(segments, key=len)
    return sorted(set(best))


def _best_lrc_offset(lyric_lines: list[str], transcript: str) -> int:
    """估计「本块从歌词第几行开始唱」（跳过前奏/未唱段落）。"""
    idx = _match_lrc_window(lyric_lines, transcript)
    return idx[0] if idx else 0


def _legacy_best_lrc_offset(lyric_lines: list[str], transcript: str) -> int:
    """估计「本块从歌词第几行开始唱」（跳过前奏/未唱段落）。

    做法：对每个起点 i，把歌词从 i 开始向右逐行累加，直到长度 ≥ 转写
    长度；用字符级 LCS 计算转写被覆盖的比例。取覆盖率最高者；同分时
    取**更靠后**的起点（前缀越短、覆盖越贴切）。
    """
    from difflib import SequenceMatcher

    from core.song import norm_text

    body = norm_text(transcript)
    if not body or not lyric_lines:
        return 0
    normed = [norm_text(x) for x in lyric_lines]
    n_body = len(body)
    best_idx, best = 0, -1.0
    best_len = 10 ** 9
    for i in range(len(normed)):
        text = ""
        for j in range(i, len(normed)):
            text += normed[j]
            if len(text) >= n_body * 1.15 or len(text) >= n_body + 20:
                break
        if not text:
            continue
        sm = SequenceMatcher(None, body, text, autojunk=False)
        matched = sum(b.size for b in sm.get_matching_blocks())
        score = matched / max(1, n_body)
        # 同分（常见于副歌重复）取窗口更短者——起点更贴近实际开唱位置
        if score > best + 1e-4 or (abs(score - best) <= 1e-4 and len(text) < best_len):
            best, best_idx, best_len = score, i, len(text)
    return best_idx


def _flags_payload(decisions: list) -> list[dict]:
    """BGM 判别明细 → 可序列化结构（UI 复核清单用）。"""
    out = []
    for d in decisions or []:
        out.append({
            "start": round(getattr(d, "start", 0.0), 3),
            "end": round(getattr(d, "end", 0.0), 3),
            "score": getattr(d, "score", 0.0),
            "vocal_sim": getattr(d, "vocal_sim", 0.0),
            "music": getattr(d, "music", 0.0),
            "reasons": list(getattr(d, "reasons", []) or []),
        })
    return out


def _live_summary(live) -> dict:
    """弹幕旁路信息 → 摘要（无弹幕时返回空 dict）。"""
    if live is None:
        return {}
    try:
        return {
            "streamer": live.streamer, "title": live.title,
            "duration": round(live.duration, 1),
            "danmaku": len(live.danmaku), "gifts": len(live.gifts),
            "requests": [{"time": t, "user": u, "name": n}
                         for t, u, n in live.song_requests()],
            "hotwords": live.hotwords(top=40),
        }
    except Exception:
        return {}


def _split_windows(segments: list[tuple[float, float]], window: float = 5.0,
                   min_window: float = 1.0) -> list[tuple[float, float]]:
    """VAD 段 → 细粒度子窗（~window 秒），用于逐窗判定 BGM/主播。

    VAD 会把「唱歌+说话」合并成 20~30 秒的长段，按整段判定会一刀切；
    切成 5 秒窗后，原唱段与主播跟唱可以分别处理。
    """
    out: list[tuple[float, float]] = []
    for s, e in segments:
        span = e - s
        if span <= window:
            if span >= min_window:
                out.append((round(s, 3), round(e, 3)))
            continue
        n = max(1, int(round(span / window)))
        step = span / n
        for k in range(n):
            ws = s + k * step
            we = ws + step
            if we - ws >= min_window:
                out.append((round(ws, 3), round(we, 3)))
    return out


def _merge_windows(windows: list[tuple[float, float]], keep: list[bool],
                   gap: float = 0.35) -> list[tuple[float, float]]:
    """保留子窗 → 合并为连续区间（间隙 ≤ gap 的相邻保留窗合并）。"""
    spans: list[list[float]] = []
    for (s, e), ok in zip(windows, keep):
        if not ok:
            continue
        if spans and s - spans[-1][1] <= gap:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    return [(round(s, 3), round(e, 3)) for s, e in spans if e - s >= 0.4]


def _norm_key(text: str) -> str:
    """去重用的文本指纹：去掉空白与标点，仅保留文字。"""
    return "".join(c for c in text if c.isalnum())


def _songs_md(songs: list) -> str:
    rows = ["# 歌单时间戳索引", ""]
    for s in songs:
        if not s.title:
            continue
        rows.append(f"- [{_hms(s.start)} - {_hms(s.end)}] 《{s.title}》"
                    + (f"- {s.artist}" if s.artist else "")
                    + (f"（{s.source}）" if s.source else ""))
    return "\n".join(rows) + "\n"


def _hms(t: float) -> str:
    t = max(0, int(t))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


__all__ = ["Pipeline", "PipelineRequest", "PipelineResult", "PipelineError",
           "Context"]
