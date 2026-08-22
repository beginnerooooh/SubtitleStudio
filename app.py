"""Subtitle Studio — Gradio WebUI 入口。

三段式布局（输入 / 参数 / 执行与结果）；pipeline 在后台线程运行，
事件函数为 generator：每 0.5s 从队列拉取进度/日志/结果增量刷新 UI；
取消按钮只置位 threading.Event，由 pipeline 在阶段边界安全退出。

直播场景（可选）：主播声纹注册/过滤 + 唱歌检测听歌识曲 + 歌单时间戳导出。
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import gradio as gr

from core.env import detect_env
from core.errors import TaskCancelled
from core.pipeline import Pipeline, PipelineConfig, PipelineError, PipelineResult
from core.song_recognizer import format_timeline_md
from core.voiceprint import VoiceprintError, list_profiles, save_profile

EXAMPLE_LYRICS = """落叶的位置 谱出一首诗
时间在消逝 我们的故事开始
你说难忘记 是那年冬天的雪
我说会记得 每一次相遇的季节"""

_MEDIA_EXTS = [".mp4", ".mkv", ".mov", ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".webm"]
_LANGUAGE_CHOICES = {"自动": "auto", "中文": "zh", "英文": "en"}
_DEVICE_CHOICES = {"自动": "auto", "CUDA (GPU)": "cuda", "CPU": "cpu"}
_PROFILES_DIR = "profiles"

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
        rows.append([i + 1, _fmt_ts(ln.start), _fmt_ts(ln.end), ln.text])
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
               language, formats, title, enable_voiceprint, profile_name,
               voice_threshold, enable_song_detect, progress=gr.Progress()):
    """启动按钮：generator，持续把队列消息刷到 UI。"""
    if file is None:
        yield ("请先上传音视频文件。", "", [], [], "本次任务未开启听歌识曲，或未识别到歌曲。",
               gr.update(interactive=True), gr.update(interactive=False))
        return
    if enable_voiceprint and not (profile_name or "").strip():
        yield ("已启用声纹过滤但未选择主播 Profile，请先在「主播声纹」面板选择或注册主播。",
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
        enable_voiceprint=enable_voiceprint,
        profile_name=(profile_name or "").strip(),
        voice_threshold=float(voice_threshold),
        profiles_dir=_PROFILES_DIR,
        enable_song_detect=enable_song_detect,
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
                files = [result.files[f] for f in ("srt", "lrc", "ass", "songs_md", "songs_csv")
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
                sep_in = gr.Checkbox(value=False, label="开启人声分离（Demucs，适合歌声）")
                lang_in = gr.Dropdown(choices=list(_LANGUAGE_CHOICES), value="自动", label="识别语言（盲识别）")
                fmt_in = gr.Checkboxgroup(choices=["srt", "lrc", "ass"], value=["srt", "lrc", "ass"], label="导出格式")
                title_in = gr.Textbox(label="LRC 标题（可选）", placeholder="默认使用文件名")

                with gr.Accordion("直播场景：主声线声纹过滤与听歌识曲（可选）", open=False):
                    gr.Markdown(
                        "过滤背景杂音/连麦人声，只保留目标主播的声音；"
                        "检测到唱歌时自动联网识曲并生成歌单时间戳。**仅作用于盲识别模式**。"
                    )
                    voice_filter_in = gr.Checkbox(
                        value=False, label="启用主声线声纹过滤（仅识别目标主播的声音）",
                    )
                    song_detect_in = gr.Checkbox(
                        value=False, label="自动识别歌声并生成歌单时间戳（Shazam 在线识曲）",
                    )
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

        gr.Markdown("### 3. 执行与结果")
        with gr.Row():
            start_btn = gr.Button("开始生成", variant="primary")
            cancel_btn = gr.Button("取消", interactive=False)

        status_out = gr.Markdown("等待任务…")
        log_out = gr.Textbox(label="运行日志", lines=12, max_lines=25, interactive=False, autoscroll=True)
        with gr.Tabs():
            with gr.TabItem("字幕预览"):
                preview_out = gr.Dataframe(
                    headers=["行号", "开始", "结束", "文本"], label="预览（前 500 行）", interactive=False
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
        save_profile_btn.click(
            save_streamer_profile,
            inputs=[new_name_in, speak_sample_in, sing_sample_in, device_in],
            outputs=[profile_status_out, profile_in],
        )

        start_btn.click(
            start_task,
            inputs=[file_in, lyrics_in, device_in, model_in, sep_in, lang_in, fmt_in, title_in,
                    voice_filter_in, profile_in, threshold_in, song_detect_in],
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
