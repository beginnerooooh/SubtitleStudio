"""Subtitle Studio — Gradio WebUI 入口。

三段式布局（输入 / 参数 / 执行与结果）；pipeline 在后台线程运行，
事件函数为 generator：每 0.5s 从队列拉取进度/日志/结果增量刷新 UI；
取消按钮只置位 threading.Event，由 pipeline 在阶段边界安全退出。

直播场景（可选）：主播声纹注册/过滤 + 唱歌检测听歌识曲 + 歌单时间戳导出。
多说话人场景（可选）：声纹分离（无监督聚类 + 声纹库命名）→ 试听验证 →
勾选说话人 → 定向转写；低置信度区域标注供人工复核。
"""
from __future__ import annotations

import queue
import shutil
import threading
import time
from pathlib import Path

import gradio as gr

from core.audio import AudioProcessError, extract_wav
from core.env import detect_env
from core.errors import TaskCancelled
from core.pipeline import Pipeline, PipelineConfig, PipelineError, PipelineResult
from core.separator import VocalSeparator
from core.speaker import SpeakerAnalyzer, SpeakerError
from core.song_recognizer import format_timeline_md
from core.voiceprint import (
    VoiceprintError,
    delete_profile,
    list_profiles,
    rename_profile,
    save_library_speaker,
    save_profile,
    speech_intervals,
)

EXAMPLE_LYRICS = """落叶的位置 谱出一首诗
时间在消逝 我们的故事开始
你说难忘记 是那年冬天的雪
我说会记得 每一次相遇的季节"""

_MEDIA_EXTS = [".mp4", ".mkv", ".mov", ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".webm"]
_LANGUAGE_CHOICES = {"自动": "auto", "中文": "zh", "英文": "en"}
_DEVICE_CHOICES = {"自动": "auto", "CUDA (GPU)": "cuda", "CPU": "cpu"}
_SPEAKER_MODE_CHOICES = {"关闭": "off", "主声线过滤": "single", "多说话人分离": "multi"}
_PROFILES_DIR = "profiles"

# 说话人试听片段目录（保留最近若干次分析，避免旧会话音频失效）
_ANALYSIS_ROOT = Path("outputs") / "_speaker_analysis"
_ANALYSIS_KEEP = 8

_task_lock = threading.Lock()
_cancel_event: threading.Event | None = None


# ---------------- 后台工作线程 ----------------

def _worker(cfg: PipelineConfig, q: "queue.Queue", cancel_event: threading.Event) -> None:
    """跑 pipeline；进度/日志/结果全部经队列转发给 UI generator。"""

    def on_progress(ratio: float, message: str) -> None:
        q.put({"kind": "progress", "ratio": ratio, "message": message})

    def on_log(message: str) -> None:
        q.put({"kind": "log", "message": message})

    try:
        result = Pipeline(cfg, on_progress=on_progress, on_log=on_log,
                          cancel_event=cancel_event).run()
        q.put({"kind": "done", "result": result})
    except TaskCancelled:
        q.put({"kind": "cancelled", "message": "任务已取消。"})
    except PipelineError as exc:
        q.put({"kind": "error", "message": f"任务失败：{exc}"})
    except Exception as exc:  # 兜底：未知异常也要反馈到界面
        q.put({"kind": "error", "message": f"未预期的错误：{exc!r}"})
    finally:
        q.put({"kind": "end"})


# ---------------- UI 辅助 ----------------

def _fmt_ts(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m):02d}:{s:05.2f}"


def _new_analysis_dir() -> Path:
    """新建说话人分析目录（时间戳命名）；只保留最近 _ANALYSIS_KEEP 个。

    试听片段需跨请求存活（gr.State 持有路径直到重新分析），故放
    outputs/ 下而非临时目录；旧目录超出保留数时清理。
    """
    _ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p for p in _ANALYSIS_ROOT.iterdir() if p.is_dir())
    for old in dirs[:max(0, len(dirs) - (_ANALYSIS_KEEP - 1))]:
        shutil.rmtree(old, ignore_errors=True)
    out = _ANALYSIS_ROOT / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _env_badge() -> str:
    info = detect_env()
    if info.device == "cuda":
        dev = f"CUDA ({info.vram_gb:.0f}GB)" if info.vram_gb else "CUDA"
    else:
        dev = "CPU"
    ff = "FFmpeg 正常" if info.ffmpeg_available else "FFmpeg 缺失（无法运行）"
    return f"运行环境：{dev} · 量化策略：{info.compute_type} · {ff}"


def _mode_hint(lyrics: str) -> str:
    if lyrics and lyrics.strip():
        return "已提供歌词：将使用 **强制对齐**（wav2vec2 逐字对齐，精度最高）"
    return "歌词为空：将使用 **盲识别**（faster-whisper，无需歌词）"


def _preview_rows(result: PipelineResult, limit: int = 500):
    rows = []
    for i, ln in enumerate(result.lines[:limit]):
        rows.append([i + 1, _fmt_ts(ln.start), _fmt_ts(ln.end), ln.speaker,
                     "是" if ln.low_confidence else "", ln.text])
    return rows


def _songs_markdown(result: PipelineResult | None) -> str:
    """歌单 Tab 内容：识别到的歌曲时间戳，或占位说明。"""
    if result is None or not result.songs:
        return "本次任务未开启听歌识曲，或未识别到歌曲。"
    return format_timeline_md(result.songs).strip()


def _profile_choices() -> list[str]:
    """已注册主播名（profiles/*.npy）。"""
    return list_profiles(_PROFILES_DIR)


# ---------------- 事件处理 ----------------

def start_task(file, lyrics, device, model_size, enable_separation,
               language, formats, title, speaker_mode_label, profile_name,
               voice_threshold, enable_song_detect, enable_lyrics_fetch,
               speaker_analysis_state, selected_speakers, use_speaker_library,
               mark_low_confidence, speaker_labels,
               progress=gr.Progress()):
    """启动按钮：generator，持续把队列消息刷到 UI。"""
    speaker_mode = _SPEAKER_MODE_CHOICES.get(speaker_mode_label, "off")
    if file is None:
        yield ("请先上传音视频文件。", "", [], [], "本次任务未开启听歌识曲，或未识别到歌曲。",
               gr.update(interactive=True), gr.update(interactive=False))
        return
    if speaker_mode == "single" and not (profile_name or "").strip():
        yield ("已启用主声线过滤但未选择主播 Profile，请先在「主播声纹」面板选择或注册主播。",
               "", [], [], "本次任务未开启听歌识曲，或未识别到歌曲。",
               gr.update(interactive=True), gr.update(interactive=False))
        return
    if not _task_lock.acquire(blocking=False):
        yield ("已有任务在进行中，请等待完成或先取消。", "", [], [],
               "本次任务未开启听歌识曲，或未识别到歌曲。",
               gr.update(interactive=False), gr.update(interactive=True))
        return

    global _cancel_event
    cfg = PipelineConfig(
        input_path=file,
        lyrics_text=lyrics or "",
        device=_DEVICE_CHOICES.get(device, "auto"),
        model_size=model_size,
        enable_separation=enable_separation,
        language=_LANGUAGE_CHOICES.get(language, "auto"),
        formats=tuple(formats) or ("srt", "lrc", "ass"),
        title=title or "",
        work_dir="outputs",
        enable_voiceprint=(speaker_mode == "single"),
        profile_name=(profile_name or "").strip(),
        voice_threshold=float(voice_threshold),
        profiles_dir=_PROFILES_DIR,
        enable_song_detect=enable_song_detect,
        enable_lyrics_fetch=enable_lyrics_fetch,
        speaker_mode=speaker_mode,
        speaker_analysis=speaker_analysis_state if speaker_mode == "multi" else None,
        selected_speakers=list(selected_speakers or []),
        use_speaker_library=bool(use_speaker_library),
        mark_low_confidence=bool(mark_low_confidence),
        speaker_labels=bool(speaker_labels),
    )
    _cancel_event = threading.Event()
    q: "queue.Queue" = queue.Queue()
    threading.Thread(
        target=_worker, args=(cfg, q, _cancel_event), daemon=False
    ).start()

    logs: list[str] = []
    ratio, message = 0.0, "启动中…"
    status = message
    preview: list = []
    files: list = []
    songs_md = _songs_markdown(None)
    result: PipelineResult | None = None

    while True:
        try:
            msg = q.get(timeout=0.5)
        except queue.Empty:
            msg = None

        if msg is not None:
            kind = msg["kind"]
            if kind == "progress":
                ratio, message = msg["ratio"], msg["message"]
                status = message
            elif kind == "log":
                logs.append(f"[{time.strftime('%H:%M:%S')}] {msg['message']}")
            elif kind == "done":
                result = msg["result"]
                status = "全部完成。"
                preview = _preview_rows(result)
                files = [result.files[f] for f in ("srt", "lrc", "ass", "songs_md",
                                                   "songs_csv", "review_md")
                         if f in result.files]
                songs_md = _songs_markdown(result)
                if len(result.lines) > 500:
                    logs.append(f"预览仅显示前 500 行（共 {len(result.lines)} 行），完整内容请下载文件。")
                for w in result.warnings:
                    logs.append(f"[警告] {w}")
            elif kind == "cancelled":
                status = msg["message"]
                logs.append(msg["message"])
            elif kind == "error":
                status = msg["message"]
                logs.append(msg["message"])
            elif kind == "end":
                _task_lock.release()
                if result is not None:
                    progress(1.0, desc="完成")
                yield (
                    status,
                    "\n".join(logs[-300:]),
                    preview,
                    files,
                    songs_md,
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                )
                return

        progress(min(0.999, ratio), desc=status)
        yield (
            status,
            "\n".join(logs[-300:]),
            preview,
            files,
            songs_md,
            gr.update(interactive=False),
            gr.update(interactive=True),
        )


def cancel_task():
    """取消按钮：只置位事件，由 pipeline 在阶段/分块边界安全退出。"""
    if _cancel_event is not None:
        _cancel_event.set()
    return gr.update(interactive=False)


# ---------------- 主播声纹管理 ----------------

def save_streamer_profile(name, speak_file, sing_file, device):
    """提取并保存主播声纹（双音色注册集）→ 刷新下拉框。

    返回 (状态说明, 下拉框 update)。
    """
    name = (name or "").strip()
    if not name:
        return "请输入主播名。", gr.update()
    if speak_file is None:
        return "请上传「日常说话样本」（10~30 秒清晰纯净干声）。", gr.update()
    dev = _DEVICE_CHOICES.get(device, "auto")
    if dev == "auto":
        dev = detect_env().device
    try:
        path = save_profile(
            name, speak_file, sing_file,
            profiles_dir=_PROFILES_DIR, device=dev,
        )
    except VoiceprintError as exc:
        return f"声纹保存失败：{exc}", gr.update()
    except Exception as exc:  # 兜底：模型下载失败等也要反馈到界面
        return f"未预期的错误：{exc!r}", gr.update()
    msg = f"已保存主播「{name}」的声纹（{path}）。"
    if sing_file is None:
        msg += " 未提供唱歌样本：副歌高音区可能被误过滤，建议补充注册。"
    return msg, gr.update(choices=_profile_choices(), value=name)


def refresh_profile_list():
    """刷新主播 Profile 下拉框。"""
    return gr.update(choices=_profile_choices())


def delete_streamer_profile(name, current_profile):
    """从声纹库删除指定 Profile；返回状态与两个下拉框的刷新。"""
    if not name:
        return "请先在下拉框选择要删除的声纹。", gr.update(), gr.update()
    try:
        path = delete_profile(name, profiles_dir=_PROFILES_DIR)
    except VoiceprintError as exc:
        return f"删除失败：{exc}", gr.update(), gr.update()
    except Exception as exc:  # 兜底：文件被占用等
        return f"未预期的错误：{exc!r}", gr.update(), gr.update()
    choices = _profile_choices()
    # 当前选中的 Profile 被删时清空选择，避免残留失效值
    remain = None if current_profile == name else current_profile
    return (
        f"已删除「{name}」的声纹（{path}）。",
        gr.update(choices=choices, value=remain),
        gr.update(choices=choices, value=None),
    )


def rename_streamer_profile(old, new, current_profile):
    """重命名声纹库中的 Profile；返回状态与两个下拉框的刷新。"""
    if not old:
        return "请先在下拉框选择要重命名的声纹。", gr.update(), gr.update()
    if not (new or "").strip():
        return "请输入新名称。", gr.update(), gr.update()
    try:
        path = rename_profile(old, new.strip(), profiles_dir=_PROFILES_DIR)
    except VoiceprintError as exc:
        return f"重命名失败：{exc}", gr.update(), gr.update()
    except Exception as exc:  # 兜底：磁盘写入失败等
        return f"未预期的错误：{exc!r}", gr.update(), gr.update()
    choices = _profile_choices()
    # 当前选中的正是被改名者 → 跟随新名字
    follow = new.strip() if current_profile == old else current_profile
    return (
        f"已将「{old}」重命名为「{new.strip()}」（{path}）。",
        gr.update(choices=choices, value=follow),
        gr.update(choices=choices, value=new.strip()),
    )


# ---------------- 多说话人分离 ----------------

def _speaker_panels(mode_label):
    """声纹模式 → 主声线 / 多说话人子面板可见性。"""
    mode = _SPEAKER_MODE_CHOICES.get(mode_label, "off")
    return (gr.update(visible=(mode == "single")),
            gr.update(visible=(mode == "multi")))


def _on_file_change():
    """更换文件后作废旧分析（试听片段/勾选与新文件无对应关系）。"""
    return (None, "文件已更换：请重新分析说话人声纹。",
            gr.update(choices=[], value=[]), gr.update(choices=[], value=None))


def analyze_speakers(file, device, use_library, separate, progress=gr.Progress()):
    """分析说话人声纹：抽取 →（可选）人声分离去 BGM → VAD → 聚类 → 声纹库匹配 → 试听片段。

    generator：分阶段刷新状态；结果存入 speaker_state 供卡片渲染与
    后续任务复用（pipeline 直接使用，不重复嵌入/聚类）。
    """
    if file is None:
        yield ("请先上传音视频文件，再分析说话人。", None,
               gr.update(choices=[], value=[]), gr.update(choices=[]))
        return
    dev = _DEVICE_CHOICES.get(device, "auto")
    if dev == "auto":
        dev = detect_env().device
    work = _new_analysis_dir()
    try:
        progress(0.02, desc="抽取音频…")
        wav_path = str(extract_wav(file, work))
        if separate:
            # 带 BGM 的直播回放：先剥离伴奏再提声纹，精度显著更高
            # （Demucs 失败时自动回退原始音频，任务不中断）
            progress(0.06, desc="分离人声（去除 BGM）…")
            wav_path = VocalSeparator(device=dev).separate(wav_path, work)
        progress(0.10, desc="检测语音段…")
        segments = speech_intervals(wav_path)
        if not segments:
            yield ("未检测到语音活动，无法分析说话人。", None,
                   gr.update(choices=[], value=[]), gr.update(choices=[]))
            return
        analyzer = SpeakerAnalyzer(device=dev)
        analysis = analyzer.analyze(
            wav_path, segments,
            profiles_dir=_PROFILES_DIR,
            use_library=use_library,
            exemplar_dir=work,
            separated=bool(separate),
            on_progress=lambda r: progress(0.10 + 0.85 * r, desc="识别说话人…"),
        )
    except (AudioProcessError, VoiceprintError, SpeakerError) as exc:
        yield (f"说话人分析失败：{exc}", None,
               gr.update(choices=[], value=[]), gr.update(choices=[]))
        return
    except Exception as exc:  # 兜底：模型下载失败等也要反馈到界面
        yield (f"未预期的错误：{exc!r}", None,
               gr.update(choices=[], value=[]), gr.update(choices=[]))
        return

    if not analysis.clusters:
        yield ("未发现有效说话人（语音过短或均为噪声）。", analysis,
               gr.update(choices=[], value=[]), gr.update(choices=[]))
        return
    names = [c.name for c in analysis.clusters]
    summary = "、".join(
        f"**{c.name}**（{c.duration:.0f}s / {len(c.segments)} 段"
        + (f"，命中声纹库 {c.library_sim:.2f}" if c.matched_library else "")
        + ")" for c in analysis.clusters)
    yield (
        f"识别到 {len(analysis.clusters)} 位说话人：{summary}。\n\n"
        "试听各片段验证身份 → 勾选参与转写的说话人（可选加入声纹库）→ 开始生成。"
        " 不分析直接生成时将自动识别全部说话人。",
        analysis,
        gr.update(choices=names, value=names),
        gr.update(choices=names, value=names[0]),
    )


def save_speaker_to_library(speaker_name, library_name, analysis):
    """把分析出的说话人声纹（簇中心）存入声纹库，供后续文件自动命名。"""
    if analysis is None or not getattr(analysis, "clusters", None):
        return "请先分析说话人声纹，再选择要入库的说话人。", gr.update()
    name = (library_name or "").strip()
    if not name:
        return "请输入声纹库名称（建议用真实姓名/昵称）。", gr.update()
    cluster = next((c for c in analysis.clusters if c.name == speaker_name), None)
    if cluster is None:
        return "未找到选中的说话人，请重新分析后再入库。", gr.update()
    try:
        path = save_library_speaker(name, cluster.embedding,
                                    profiles_dir=_PROFILES_DIR)
    except VoiceprintError as exc:
        return f"入库失败：{exc}", gr.update()
    except Exception as exc:  # 兜底：磁盘写入失败等
        return f"未预期的错误：{exc!r}", gr.update()
    return (f"已将「{speaker_name}」存入声纹库（名称：{name}，{path}）。"
            " 后续分析/任务将自动按此名称匹配该声纹。",
            gr.update(choices=_profile_choices(), value=name))


# ---------------- 界面 ----------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Subtitle Studio 字幕工坊") as demo:
        gr.Markdown("## Subtitle Studio 字幕工坊\n普通语音 / 歌声字幕生成：盲识别 + 歌词强制对齐，导出 SRT / LRC / ASS（逐字卡拉OK）")
        gr.Markdown(_env_badge())

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. 输入")
                file_in = gr.File(label="音视频文件", file_types=_MEDIA_EXTS)
                mode_hint = gr.Markdown(_mode_hint(""))
                lyrics_in = gr.Textbox(
                    label="歌词（可选：粘贴歌词文本走强制对齐；支持带 LRC 时间戳，将被剥离）",
                    placeholder="留空 = 盲识别；粘贴歌词 = 逐字强制对齐",
                    lines=10,
                )
                with gr.Row():
                    clear_btn = gr.Button("清空歌词")
                    example_btn = gr.Button("填入示例")
            with gr.Column():
                gr.Markdown("### 2. 参数")
                device_in = gr.Dropdown(
                    choices=list(_DEVICE_CHOICES), value="自动", label="运行设备"
                )
                model_in = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large-v3"],
                    value="small", label="盲识别模型（whisper）",
                )
                sep_in = gr.Checkbox(
                    value=False,
                    label="开启人声分离（Demucs，适合歌声 / 说话带 BGM 的直播回放）",
                )
                lang_in = gr.Dropdown(choices=list(_LANGUAGE_CHOICES), value="自动", label="识别语言（盲识别）")
                fmt_in = gr.Checkboxgroup(choices=["srt", "lrc", "ass"], value=["srt", "lrc", "ass"], label="导出格式")
                title_in = gr.Textbox(label="LRC 标题（可选）", placeholder="默认使用文件名")

                with gr.Accordion("直播 / 多说话人场景（可选，仅盲识别模式）", open=False):
                    gr.Markdown(
                        "过滤背景杂音/连麦人声，或对文件中各说话人做声纹分离；"
                        "检测到唱歌时自动联网识曲并生成歌单时间戳。"
                    )
                    speaker_mode_in = gr.Radio(
                        choices=list(_SPEAKER_MODE_CHOICES), value="关闭",
                        label="声纹模式",
                    )

                    with gr.Column(visible=False) as single_panel:
                        gr.Markdown("只保留目标主播的声音，剔除其他人声/杂音。")
                        with gr.Row():
                            profile_in = gr.Dropdown(
                                choices=_profile_choices(), label="主播声纹 Profile",
                                scale=3,
                            )
                            refresh_profile_btn = gr.Button("刷新列表", scale=1)
                        threshold_in = gr.Slider(
                            minimum=0.3, maximum=0.8, value=0.55, step=0.01,
                            label="声纹匹配严格度（余弦相似度阈值，越大越严格）",
                        )
                        with gr.Accordion("注册新主播声纹", open=False):
                            gr.Markdown(
                                "上传 10~30 秒清晰纯净干声：**说话样本必填**；"
                                "唱歌样本可选但强烈推荐（解决说话与唱歌音色差异大导致的误过滤）。"
                            )
                            new_name_in = gr.Textbox(label="主播名", placeholder="例如：小明")
                            speak_sample_in = gr.File(
                                label="参考音频 A：日常说话声线（必填）", file_types=_MEDIA_EXTS,
                            )
                            sing_sample_in = gr.File(
                                label="参考音频 B：唱歌/高音音色（可选）", file_types=_MEDIA_EXTS,
                            )
                            save_profile_btn = gr.Button("提取并保存声纹")
                            profile_status_out = gr.Markdown()

                        with gr.Accordion("管理声纹库", open=False):
                            gr.Markdown(
                                "删除或重命名已入库的声纹（含主播 Profile 与"
                                "「加入声纹库」的说话人）。"
                            )
                            manage_profile_in = gr.Dropdown(
                                choices=_profile_choices(),
                                label="选择声纹", scale=3,
                            )
                            with gr.Row():
                                manage_rename_in = gr.Textbox(
                                    label="新名称", placeholder="重命名时填写",
                                    scale=2,
                                )
                                rename_profile_btn = gr.Button("重命名", scale=1)
                                delete_profile_btn = gr.Button(
                                    "删除", variant="stop", scale=1,
                                )
                            manage_status_md = gr.Markdown()

                    with gr.Column(visible=False) as multi_panel:
                        gr.Markdown(
                            "自动分离文件中的各说话人（无需预注册）：先分析 → 试听验证 → "
                            "勾选参与转写的说话人；可直接把说话人加入声纹库，"
                            "后续文件自动识别其身份。"
                        )
                        use_library_in = gr.Checkbox(
                            value=True,
                            label="分析时结合声纹库（自动识别已入库说话人）",
                        )
                        analyze_sep_in = gr.Checkbox(
                            value=False,
                            label="分析前先分离人声（说话带 BGM 的直播回放强烈建议勾选，更准但更慢）",
                        )
                        analyze_btn = gr.Button("分析说话人声纹")
                        speaker_status_md = gr.Markdown("尚未分析：上传文件后点击「分析说话人声纹」。")
                        speaker_state = gr.State(None)

                        # 动态声纹卡片：每个说话人一张卡（时长/段数/库命中 + 试听按钮）
                        # equal_height：徽章行数不同时卡片仍等高；min_width=220 窄屏自动换行
                        @gr.render(inputs=speaker_state)
                        def render_speaker_cards(analysis):
                            clusters = getattr(analysis, "clusters", None) or []
                            if not clusters:
                                return
                            with gr.Row(equal_height=True):
                                for c in clusters:
                                    with gr.Column(variant="panel", scale=1, min_width=220):
                                        badge = (f"\n\n命中声纹库（相似度 {c.library_sim:.2f}）"
                                                 if c.matched_library else "")
                                        gr.Markdown(
                                            f"**{c.name}**{badge}\n\n"
                                            f"时长 {c.duration:.0f}s · {len(c.segments)} 段"
                                        )
                                        if c.exemplar_path:
                                            gr.Audio(
                                                value=c.exemplar_path, label="试听验证",
                                                format="wav", show_download_button=False,
                                            )

                        speaker_select_in = gr.Checkboxgroup(
                            choices=[], label="勾选参与转写的说话人（未勾选的将被剔除）",
                        )
                        with gr.Accordion("加入声纹库", open=False):
                            gr.Markdown(
                                "试听确认身份后，为说话人命名并存入声纹库；"
                                "之后所有分析/任务会自动匹配该名称。"
                            )
                            lib_speaker_in = gr.Dropdown(choices=[], label="说话人")
                            lib_name_in = gr.Textbox(
                                label="声纹库名称", placeholder="例如：小明",
                            )
                            lib_save_btn = gr.Button("加入声纹库")
                            lib_status_md = gr.Markdown()

                    song_detect_in = gr.Checkbox(
                        value=False, label="自动识别歌声并生成歌单时间戳（Shazam 在线识曲）",
                    )
                    lyrics_fetch_in = gr.Checkbox(
                        value=False,
                        label="识曲后自动拉取同步歌词生成字幕（LRCLIB，未拉到的段落回退语音识别）",
                    )
                    with gr.Accordion("扩展选项", open=False):
                        mark_low_confidence_in = gr.Checkbox(
                            value=True,
                            label="低置信度区域标注（预览标记 + 导出人工复核清单）",
                        )
                        speaker_labels_in = gr.Checkbox(
                            value=False,
                            label="导出字幕标注说话人（SRT/LRC 行首前缀，ASS 说话人字段）",
                        )

        gr.Markdown("### 3. 执行与结果")
        with gr.Row():
            start_btn = gr.Button("开始生成", variant="primary")
            cancel_btn = gr.Button("取消", interactive=False)

        status_out = gr.Markdown("等待任务…")
        log_out = gr.Textbox(label="运行日志", lines=12, max_lines=25, interactive=False, autoscroll=True)
        with gr.Tabs():
            with gr.TabItem("字幕预览"):
                preview_out = gr.Dataframe(
                    headers=["行号", "开始", "结束", "说话人", "待复核", "文本"],
                    label="预览（前 500 行；「待复核」= 低置信度区域，建议人工验证）",
                    interactive=False,
                )
            with gr.TabItem("歌单时间戳"):
                songs_out = gr.Code(
                    label="歌单（songs_timeline.md，可一键复制）",
                    language="markdown", interactive=False, lines=12,
                )
        files_out = gr.Files(label="下载")

        lyrics_in.change(_mode_hint, inputs=lyrics_in, outputs=mode_hint)
        clear_btn.click(lambda: "", outputs=lyrics_in).then(_mode_hint, inputs=lyrics_in, outputs=mode_hint)
        example_btn.click(lambda: EXAMPLE_LYRICS, outputs=lyrics_in).then(_mode_hint, inputs=lyrics_in, outputs=mode_hint)

        refresh_profile_btn.click(refresh_profile_list, outputs=profile_in)
        delete_profile_btn.click(
            delete_streamer_profile,
            inputs=[manage_profile_in, profile_in],
            outputs=[manage_status_md, profile_in, manage_profile_in],
        )
        rename_profile_btn.click(
            rename_streamer_profile,
            inputs=[manage_profile_in, manage_rename_in, profile_in],
            outputs=[manage_status_md, profile_in, manage_profile_in],
        )
        save_profile_btn.click(
            save_streamer_profile,
            inputs=[new_name_in, speak_sample_in, sing_sample_in, device_in],
            outputs=[profile_status_out, profile_in],
        )

        speaker_mode_in.change(
            _speaker_panels, inputs=speaker_mode_in,
            outputs=[single_panel, multi_panel],
        )
        file_in.change(
            _on_file_change,
            outputs=[speaker_state, speaker_status_md, speaker_select_in, lib_speaker_in],
        )
        analyze_btn.click(
            analyze_speakers,
            inputs=[file_in, device_in, use_library_in, analyze_sep_in],
            outputs=[speaker_status_md, speaker_state, speaker_select_in, lib_speaker_in],
        )
        lib_save_btn.click(
            save_speaker_to_library,
            inputs=[lib_speaker_in, lib_name_in, speaker_state],
            outputs=[lib_status_md, profile_in],
        )

        start_btn.click(
            start_task,
            inputs=[file_in, lyrics_in, device_in, model_in, sep_in, lang_in, fmt_in, title_in,
                    speaker_mode_in, profile_in, threshold_in, song_detect_in, lyrics_fetch_in,
                    speaker_state, speaker_select_in, use_library_in,
                    mark_low_confidence_in, speaker_labels_in],
            outputs=[status_out, log_out, preview_out, files_out, songs_out, start_btn, cancel_btn],
        )
        cancel_btn.click(cancel_task, outputs=cancel_btn)
    return demo


demo = build_ui()


# ---------------- 模型预载 ----------------

def _preload_model_once() -> None:
    """同步预载默认盲识别模型（small + 环境量化策略），命中流水线模型缓存。

    失败静默忽略：预载只是优化，首次任务时会再次加载并正常报错。
    """
    try:
        from core.transcriber import _get_model

        env = detect_env()
        _get_model("small", env.device, env.compute_type)
    except Exception:
        pass


def _preload_default_model() -> None:
    """启动时后台线程预载，不阻塞 UI。"""
    threading.Thread(target=_preload_model_once, daemon=True).start()


if __name__ == "__main__":
    _preload_default_model()
    demo.queue().launch()
