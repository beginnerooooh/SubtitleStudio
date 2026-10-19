"""Subtitle Studio v2 —— Gradio WebUI：生成、比对、编辑一体。

界面结构
────────
① 生成       上传音视频 → 参数 → 生成字幕（含声纹/歌曲/弹幕辅助）
② 校对编辑器  视频 + 字幕联动：点击行跳转、切分/合并/增删/改字/调时间
③ 评测对比    载入人工校对样本，输出不一致数量、匹配率、边界清单
④ 声纹与歌词  注册主播声纹、导入歌词库
⑤ 设置        引擎选择、外部 API 接入、阈值

设计取舍
────────
- 编辑器状态放在 gr.State（SubtitleLine 列表），所有操作都是纯函数
  （core.editor），便于撤销与单测；
- 视频与字幕联动用 gr.Video 的播放位置 + 表格选择实现：点击表格行 → 
  返回该行起点，前端用 JS 定位视频；切分点用「光标位置」输入；
- 评测走 core.evaluate，报告同时写入磁盘并内联展示。
"""
from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from pathlib import Path

import gradio as gr

from core import editor as ed
from core.config import AppConfig
from core.env import detect_env
from core.evaluate import evaluate, load_srt
from core.livexml import load_log
from core.pipeline2 import Pipeline, PipelineError, PipelineRequest
from core.providers import available_providers
from core.subtitle import (from_json, to_ass, to_json, to_lrc, to_srt,
                           to_vtt)
from core.streamer import (StreamerProfile, VoiceprintEngine, list_profiles,
                           load_profile, save_profile)

MEDIA_EXTS = [".mp4", ".mkv", ".mov", ".flv", ".mp3", ".wav", ".flac",
              ".m4a", ".aac", ".ogg", ".webm"]
EDITOR_HEADERS = ["#", "开始", "结束", "说话人", "文本"]

_VIDEO_JS = """
(sec) => {
  const v = document.querySelector('#main-video video');
  if (v) { v.currentTime = Math.max(0, sec); v.play(); }
  return sec;
}
"""

_task_lock = threading.Lock()
_cancel_event: threading.Event | None = None


# ---------------- 生成 ----------------

def _worker(cfg: AppConfig, req: PipelineRequest, q: "queue.Queue",
            cancel: threading.Event) -> None:
    def on_progress(ratio: float, msg: str) -> None:
        q.put({"kind": "progress", "ratio": ratio, "message": msg})

    def on_log(msg: str) -> None:
        q.put({"kind": "log", "message": msg})

    try:
        res = Pipeline(cfg, req, on_progress=on_progress, on_log=on_log,
                       cancel_event=cancel).run()
        q.put({"kind": "done", "result": res})
    except Exception as exc:  # noqa: BLE001 - UI 兜底
        q.put({"kind": "error", "message": str(exc)})
    finally:
        q.put({"kind": "end"})


def generate(file, engine, model, chunk, refine, punc, separate,
             voiceprint, profile_name, song_detect, live_xml, reference,
             formats, progress=gr.Progress()):
    """生成字幕（generator，持续刷新进度）。

    **防闪动设计**（对应反馈「日志栏 81.7% 进度条反复闪动」）：
    1. 只更新 `status_out`（一行 Markdown），不再每 0.4s 重写整个 400 行日志
       文本框——日志改为「有新内容才推送」，且用 `gr.skip()` 保持其余组件不动；
    2. 生成过程中不重绘预览表格 / 文件列表（它们只在完成时更新一次）；
    3. 进度用 gr.Progress(track_tqdm=False)，避免和 generator yield 争夺刷新。
    这样界面只在「状态文字变化」和「新日志」时局部刷新，不再整块重排。
    """
    if not file:
        yield ("请先上传音视频文件。", "", [], None, [], "", "", "", "", "",
               gr.update(), gr.update())
        return
    if not _task_lock.acquire(blocking=False):
        yield ("已有任务在进行中。", "", [], None, [], "", "", "", "", "",
               gr.update(interactive=False), gr.update(interactive=True))
        return
    global _cancel_event
    cfg = AppConfig.from_env(
        asr={"engine": engine, "model": model,
             "chunk_seconds": float(chunk), "chunk_overlap": max(4.0, float(chunk) * 0.08)},
        timing={"refine": bool(refine)},
        segment={"use_punctuation": bool(punc)},
        song={"enabled": bool(song_detect)},
        voiceprint={"enabled": bool(voiceprint), "profile_name": profile_name or ""},
    )
    req = PipelineRequest(
        input_path=file, enable_separation=bool(separate),
        enable_voiceprint=bool(voiceprint), profile_name=profile_name or "",
        enable_song_detect=bool(song_detect), live_xml=live_xml or "",
        reference_srt=reference or "",
        formats=tuple(formats) or ("srt",))
    _cancel_event = threading.Event()
    q: "queue.Queue" = queue.Queue()
    threading.Thread(target=_worker, args=(cfg, req, q, _cancel_event),
                     daemon=True).start()

    logs: list[str] = []
    status, ratio = "启动中…", 0.0
    rows: list[list] = []
    lines: list = []
    files: list[str] = []
    report_md, diff_md = "", ""
    songs_md = ""
    last_status = ""
    last_log_len = 0
    while True:
        try:
            msg = q.get(timeout=0.5)
        except queue.Empty:
            msg = None
        finished = False
        if msg is not None:
            if msg["kind"] == "progress":
                ratio, status = msg["ratio"], msg["message"]
            elif msg["kind"] == "log":
                logs.append(f"[{time.strftime('%H:%M:%S')}] {msg['message']}")
            elif msg["kind"] == "done":
                res = msg["result"]
                lines = res.lines
                rows = ed.to_rows(lines)
                files = [v for k, v in res.files.items()
                         if k in ("srt", "lrc", "ass", "vtt", "json", "songs_md")]
                status = (f"完成：{len(lines)} 行，用时 "
                          f"{res.timings.get('total', 0):.0f}s")
                songs_md = _songs_md(res.songs)
                if res.evaluation is not None:
                    report_md = res.evaluation.summary_md(
                        "本次生成 vs 人工校对样本")
                    diff_md = _diff_md(res.evaluation)
                for w in res.warnings:
                    logs.append(f"[警告] {w}")
            elif msg["kind"] == "error":
                status = f"失败：{msg['message']}"
                logs.append(status)
            elif msg["kind"] == "end":
                finished = True

        if finished:
            _task_lock.release()
            yield (status, "\n".join(logs[-400:]), lines,
                   files[0] if files else None, rows, ed.stats(lines),
                   report_md, diff_md, songs_md,
                   json.dumps(_lines_payload(lines), ensure_ascii=False),
                   gr.update(interactive=True), gr.update(interactive=False))
            return

        # 只有状态文字或日志真的变了才推送，其余组件用 gr.skip() 保持不变，
        # 避免每 0.5s 重绘整个界面导致的闪动。
        log_text = "\n".join(logs[-400:])
        if status != last_status or len(logs) != last_log_len:
            last_status, last_log_len = status, len(logs)
            yield (f"**{status}**  `{min(0.999, ratio):.0%}`",
                   log_text if log_text else gr.skip(),
                   gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                   gr.skip(), gr.skip(), gr.skip(),
                   gr.update(interactive=False), gr.update(interactive=True))


def cancel():
    if _cancel_event is not None:
        _cancel_event.set()
    return gr.update(interactive=False)


def _songs_md(songs) -> str:
    if not songs:
        return "未识别到歌曲（或未开启歌曲识别）。"
    rows = ["| # | 时间 | 歌曲 | 来源 | 对齐 |", "|---|---|---|---|---|"]
    for i, s in enumerate(songs, 1):
        if not s.title:
            continue
        rows.append(f"| {i} | {ed.fmt_ts(s.start)}–{ed.fmt_ts(s.end)} | "
                    f"《{s.title}》{s.artist} | {s.source} | "
                    f"{'逐行对齐' if s.aligned else '近似'} |")
    return "\n".join(rows)


def _diff_md(report) -> str:
    rows = ["| 类型 | 参考 | 生成 | 参考文本 | 生成文本 |", "|---|---|---|---|---|"]
    for c in report.boundary_cases[:60]:
        rows.append(f"| {c['kind']} | {c['ref_time']} | {c['hyp_time']} | "
                    f"{c['ref_text'][:20]} | {c['hyp_text'][:20]} |")
    return "\n".join(rows)


def _lines_payload(lines) -> list[dict]:
    return [{"start": ln.start, "end": ln.end, "text": ln.text,
             "speaker": ln.speaker, "words": [
                 {"text": w.text, "start": w.start, "end": w.end}
                 for w in ln.words]} for ln in lines]


def _load_payload(payload: str):
    from core.models import SubtitleLine, SubtitleWord

    data = json.loads(payload or "[]")
    out = []
    for row in data:
        words = [SubtitleWord(text=w["text"], start=w["start"], end=w["end"])
                 for w in row.get("words", []) if w.get("text")]
        if not words and row.get("text"):
            words = [SubtitleWord(text=row["text"], start=row["start"],
                                  end=row["end"])]
        if words:
            out.append(SubtitleLine(words=words, speaker=row.get("speaker", "")))
    return out


# ---------------- 编辑器 ----------------

def on_row_select(lines, payload, evt: gr.SelectData):
    """点击表格行 → 返回该行起点（用于定位视频）。"""
    idx = evt.index[0] if evt and evt.index else 0
    items = lines or _load_payload(payload)
    if not items or idx >= len(items):
        return gr.update(), f"第 {idx + 1} 行"
    ln = items[idx]
    return gr.update(value=ln.start), \
        f"第 {idx + 1} 行：{ed.fmt_ts(ln.start)}–{ed.fmt_ts(ln.end)}"


def do_split(lines, payload, row_no, char_pos):
    items = lines or _load_payload(payload)
    try:
        idx = int(row_no) - 1
        pos = int(char_pos)
    except (TypeError, ValueError):
        return items, json.dumps(_lines_payload(items), ensure_ascii=False), \
            ed.to_rows(items), "请输入行号与切分位置", ed.stats(items)
    items, msg = ed.split_line(items, idx, pos)
    return (items, json.dumps(_lines_payload(items), ensure_ascii=False),
            ed.to_rows(items), msg, ed.stats(items))


def do_merge(lines, payload, rows_text):
    items = lines or _load_payload(payload)
    try:
        idx = [int(x.strip()) - 1 for x in (rows_text or "").replace(
            "，", ",").split(",") if x.strip()]
    except ValueError:
        return items, json.dumps(_lines_payload(items), ensure_ascii=False), \
            ed.to_rows(items), "行号格式：如 3,4,5", ed.stats(items)
    items, msg = ed.merge_lines(items, idx)
    return (items, json.dumps(_lines_payload(items), ensure_ascii=False),
            ed.to_rows(items), msg, ed.stats(items))


def do_delete(lines, payload, rows_text):
    items = lines or _load_payload(payload)
    try:
        idx = [int(x.strip()) - 1 for x in (rows_text or "").replace(
            "，", ",").split(",") if x.strip()]
    except ValueError:
        return items, json.dumps(_lines_payload(items), ensure_ascii=False), \
            ed.to_rows(items), "行号格式：如 3,4,5", ed.stats(items)
    items, msg = ed.delete_lines(items, idx)
    return (items, json.dumps(_lines_payload(items), ensure_ascii=False),
            ed.to_rows(items), msg, ed.stats(items))


def do_update(lines, payload, row_no, new_text, start, end):
    items = lines or _load_payload(payload)
    try:
        idx = int(row_no) - 1
    except (TypeError, ValueError):
        return items, json.dumps(_lines_payload(items), ensure_ascii=False), \
            ed.to_rows(items), "请输入要修改的行号", ed.stats(items)
    kwargs = {}
    if new_text is not None and str(new_text).strip():
        kwargs["text"] = str(new_text)
    try:
        if start not in (None, ""):
            kwargs["start"] = ed.parse_ts(str(start))
        if end not in (None, ""):
            kwargs["end"] = ed.parse_ts(str(end))
    except ValueError as exc:
        return items, json.dumps(_lines_payload(items), ensure_ascii=False), \
            ed.to_rows(items), str(exc), ed.stats(items)
    items, msg = ed.update_line(items, idx, **kwargs)
    return (items, json.dumps(_lines_payload(items), ensure_ascii=False),
            ed.to_rows(items), msg, ed.stats(items))


def do_insert(lines, payload, row_no, text, start, end):
    items = lines or _load_payload(payload)
    try:
        idx = int(row_no) - 1 if row_no else len(items)
        s = ed.parse_ts(str(start)) if start else 0.0
        e = ed.parse_ts(str(end)) if end else s + 2.0
    except ValueError as exc:
        return items, json.dumps(_lines_payload(items), ensure_ascii=False), \
            ed.to_rows(items), str(exc), ed.stats(items)
    items, msg = ed.insert_line(items, idx, text or "", s, e)
    return (items, json.dumps(_lines_payload(items), ensure_ascii=False),
            ed.to_rows(items), msg, ed.stats(items))


def do_shift(lines, payload, delta, row_no):
    items = lines or _load_payload(payload)
    try:
        d = float(str(delta).replace("s", "").strip())
        idx = int(row_no) - 1 if str(row_no or "").strip() else None
    except ValueError:
        return items, json.dumps(_lines_payload(items), ensure_ascii=False), \
            ed.to_rows(items), "位移需为秒数（可负）", ed.stats(items)
    items, msg = ed.shift_lines(items, d, idx)
    return (items, json.dumps(_lines_payload(items), ensure_ascii=False),
            ed.to_rows(items), msg, ed.stats(items))


def do_qc(lines, payload):
    items = lines or _load_payload(payload)
    items, m1 = ed.fix_overlaps(items)
    items, m2 = ed.split_long_lines(items)
    return (items, json.dumps(_lines_payload(items), ensure_ascii=False),
            ed.to_rows(items), f"{m1}；{m2}", ed.stats(items))


def export_files(lines, payload, fmt, name):
    items = lines or _load_payload(payload)
    if not items:
        return [], "没有可导出的字幕行"
    out = Path("outputs") / "_exports"
    out.mkdir(parents=True, exist_ok=True)
    stem = (name or "subtitle").strip() or "subtitle"
    writers = {"srt": to_srt, "lrc": lambda ls: to_lrc(ls, title=stem),
               "ass": to_ass, "vtt": to_vtt,
               "json": lambda ls: to_json(ls, title=stem)}
    paths = []
    for f in (fmt or ["srt"]):
        fn = writers.get(f)
        if fn is None:
            continue
        p = out / f"{stem}.{f}"
        p.write_text(fn(items), encoding="utf-8")
        paths.append(str(p))
    return paths, f"已导出 {len(paths)} 个文件"


# ---------------- 评测 ----------------

def run_eval(ref_file, srt_file, report_title):
    if not ref_file or not srt_file:
        return "请同时提供人工校对样本与待评测字幕文件。", ""
    report = evaluate(load_srt(ref_file), load_srt(srt_file))
    out = Path("outputs") / "_eval_ui"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report.summary_md(report_title or "评测报告"),
                                   encoding="utf-8")
    (out / "report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8")
    return (report.summary_md(report_title or "评测报告"),
            _diff_md(report))


# ---------------- 声纹 ----------------

def register_profile(name, speak_file, sing_file):
    if not name or not speak_file:
        return "请填写主播名并上传说话样本。"
    dev = detect_env().device
    try:
        engine = VoiceprintEngine(device=dev)
        speak = engine.enroll(speak_file)
        sing = engine.enroll(sing_file) if sing_file else None
        path = save_profile(StreamerProfile(name=name, speak=speak, sing=sing))
    except Exception as exc:  # noqa: BLE001
        return f"注册失败：{exc}"
    return f"已保存声纹：{path}（唱歌样本：{'有' if sing else '无'}）"


def refresh_profiles():
    return gr.update(choices=list_profiles(), value=None)


def profile_status(name):
    if not name:
        return "未选择声纹"
    try:
        p = load_profile(name)
        return (f"{name}：说话向量 {'有' if p.speak is not None else '无'}，"
                f"唱歌向量 {'有' if p.sing is not None else '无'}")
    except Exception as exc:  # noqa: BLE001
        return f"读取失败：{exc}"


def danmaku_preview(path):
    log = load_log(path) if path else None
    if log is None:
        return "未载入（支持 B 站录制 .xml / .jsonl）"
    hot = "、".join(log.hotwords(top=20))
    return (f"弹幕 {len(log.danmaku)} 条 · 礼物 {len(log.gifts)} 条\n\n"
            f"高频词（将作为热词提示）：{hot}")


# ---------------- 界面 ----------------

def build_ui() -> gr.Blocks:
    env = detect_env()
    engines = [e for e in available_providers()]
    with gr.Blocks(title="Subtitle Studio 字幕工坊", css="""
        #main-video video { max-height: 420px; }
        .row-hint { font-size: 12px; color: #888; }
    """) as demo:
        gr.Markdown("# Subtitle Studio 字幕工坊\n"
                    "本地字幕生成 · 歌曲/声纹/弹幕辅助 · 视频联动校对")
        gr.Markdown(f"运行环境：`{env.device}` / `{env.compute_type}` · "
                    f"FFmpeg {'正常' if env.ffmpeg_available else '缺失'}")

        lines_state = gr.State([])
        payload_state = gr.State("[]")

        with gr.Tabs():
            # ---------- 生成 ----------
            with gr.TabItem("① 生成"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_in = gr.File(label="音视频文件", file_types=MEDIA_EXTS)
                        with gr.Row():
                            engine_in = gr.Dropdown(engines, value="whisper",
                                                    label="识别引擎")
                            model_in = gr.Dropdown(
                                ["large-v3", "medium", "small", "base"],
                                value="large-v3", label="模型")
                        with gr.Accordion("高级参数", open=False):
                            chunk_in = gr.Slider(30, 600, value=120, step=10,
                                                 label="分块时长（秒，越小越不容易漏段）")
                            refine_in = gr.Checkbox(True, label="时间轴自动校正")
                            punc_in = gr.Checkbox(True, label="用标点模型辅助断句")
                            separate_in = gr.Checkbox(False, label="人声分离（Demucs）")
                            with gr.Row():
                                voiceprint_in = gr.Checkbox(False, label="只保留主播声纹")
                                profile_in = gr.Dropdown(
                                    choices=list_profiles(), label="主播声纹")
                                refresh_btn = gr.Button("刷新", scale=0)
                            song_in = gr.Checkbox(True, label="识别歌曲并重建歌词")
                            live_in = gr.File(label="直播弹幕 XML（可选）",
                                              file_types=[".xml", ".jsonl"])
                            live_status = gr.Markdown("未载入弹幕文件")
                            ref_in = gr.File(label="人工校对样本（可选，生成后自动评测）",
                                             file_types=[".srt"])
                            fmt_in = gr.CheckboxGroup(
                                ["srt", "lrc", "ass", "vtt", "json"],
                                value=["srt"], label="导出格式")
                    with gr.Column(scale=2):
                        gen_btn = gr.Button("开始生成", variant="primary")
                        cancel_btn = gr.Button("取消", interactive=False)
                        status_out = gr.Markdown("就绪")
                        log_out = gr.Textbox(label="日志", lines=8, autoscroll=True)
                        with gr.Row():
                            gen_srt = gr.File(label="字幕文件")
                        with gr.Tabs():
                            with gr.TabItem("评测"): 
                                gen_report = gr.Markdown()
                            with gr.TabItem("不一致清单"):
                                gen_diff = gr.Markdown()
                            with gr.TabItem("歌曲"):
                                gen_songs = gr.Markdown()

            # ---------- 编辑器 ----------
            with gr.TabItem("② 校对编辑器"):
                with gr.Row():
                    with gr.Column(scale=3):
                        video = gr.Video(label="视频", elem_id="main-video")
                    with gr.Column(scale=2):
                        cur_pos = gr.Slider(0, 3600, value=0, step=0.1,
                                            label="定位（秒）",
                                            elem_id="pos-slider")
                        jump_btn = gr.Button("跳到该位置")
                        sel_hint = gr.Markdown("点击表格行可定位")
                        ed_stats = gr.Markdown("暂无字幕")
                table = gr.Dataframe(headers=EDITOR_HEADERS, interactive=False,
                                     label="字幕行（点击选中）")
                with gr.Row():
                    row_no = gr.Textbox(label="行号", scale=1)
                    new_text = gr.Textbox(label="新文本", scale=3)
                    new_start = gr.Textbox(label="开始", scale=1, placeholder="MM:SS.mmm")
                    new_end = gr.Textbox(label="结束", scale=1, placeholder="MM:SS.mmm")
                with gr.Row():
                    upd_btn = gr.Button("应用修改")
                    ins_btn = gr.Button("插入行")
                with gr.Row():
                    split_pos = gr.Textbox(label="切分位置（第几个字前）", scale=1)
                    split_btn = gr.Button("切分")
                    merge_rows = gr.Textbox(label="合并/删除行号（如 3,4,5）", scale=2)
                    merge_btn = gr.Button("合并")
                    del_btn = gr.Button("删除")
                with gr.Row():
                    shift_val = gr.Textbox(label="时间平移（秒，可负）", scale=1)
                    shift_btn = gr.Button("平移")
                    qc_btn = gr.Button("自动质检（重叠/超长）")
                ed_msg = gr.Markdown()
                with gr.Row():
                    exp_fmt = gr.CheckboxGroup(["srt", "lrc", "ass", "vtt", "json"],
                                               value=["srt"], label="导出格式")
                    exp_name = gr.Textbox(label="文件名", scale=2)
                    exp_btn = gr.Button("导出")
                exp_files = gr.Files(label="导出结果")

            # ---------- 评测 ----------
            with gr.TabItem("③ 评测对比"):
                gr.Markdown("以人工校对样本为基准，统计不一致数量、匹配率与"
                            "**仍需人工确认的边界情况**。")
                with gr.Row():
                    ref_eval = gr.File(label="人工校对样本", file_types=[".srt"])
                    hyp_eval = gr.File(label="待评测字幕", file_types=[".srt"])
                title_eval = gr.Textbox(label="报告标题", value="评测报告")
                eval_btn = gr.Button("开始评测", variant="primary")
                with gr.Tabs():
                    with gr.TabItem("汇总"):
                        eval_report = gr.Markdown()
                    with gr.TabItem("边界情况"):
                        eval_diff = gr.Markdown()

            # ---------- 声纹/歌词 ----------
            with gr.TabItem("④ 声纹与歌词"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 主播声纹（单目标识别）")
                        p_name = gr.Textbox(label="主播名")
                        p_speak = gr.Audio(label="说话样本（10~30s 干声）",
                                           type="filepath")
                        p_sing = gr.Audio(label="唱歌样本（可选）", type="filepath")
                        p_btn = gr.Button("注册声纹")
                        p_out = gr.Markdown()
                        p_pick = gr.Dropdown(choices=list_profiles(),
                                             label="已注册声纹")
                        p_show = gr.Button("查看")
                        p_info = gr.Markdown()
                    with gr.Column():
                        gr.Markdown("### 直播弹幕 XML")
                        xml_in = gr.File(label="录播 XML", file_types=[".xml", ".jsonl"])
                        xml_btn = gr.Button("解析预览")
                        xml_out = gr.Markdown()
                        gr.Markdown(
                            "弹幕数据用途：① 高频昵称/礼物名 → ASR 热词；\n"
                            "② 礼物事件时间点 → 断句锚点；\n"
                            "③ 观众点歌/复述歌词 → 歌曲识别线索。")

            # ---------- 设置 ----------
            with gr.TabItem("⑤ 设置"):
                gr.Markdown(
                    "### 外部识别接口\n"
                    "把任意自建服务接成识别后端：服务接收 `POST {audio_base64, "
                    "language, model}`，返回 `{\"segments\": [{start, end, text, "
                    "words: [{text, start, end}]}]}`。\n\n"
                    "也可用环境变量配置：`SUBTITLE_STUDIO_ASR_ENDPOINT` / "
                    "`SUBTITLE_STUDIO_ASR_API_KEY` / `SUBTITLE_STUDIO_PLUGINS`。")
                gr.Markdown(
                    "### 可插拔后端\n"
                    f"当前可用：`{', '.join(engines)}`。\n"
                    "第三方插件：在模块中用 `@register_provider` 注册 "
                    "`ASRProvider` 子类，并把模块路径写进 "
                    "`SUBTITLE_STUDIO_PLUGINS`（逗号分隔）。")

        # ---------- 事件绑定 ----------
        live_in.change(danmaku_preview, live_in, live_status)
        refresh_btn.click(refresh_profiles, outputs=profile_in)
        gen_btn.click(
            generate,
            inputs=[file_in, engine_in, model_in, chunk_in, refine_in, punc_in,
                    separate_in, voiceprint_in, profile_in, song_in, live_in,
                    ref_in, fmt_in],
            outputs=[status_out, log_out, lines_state, gen_srt, table, ed_stats,
                     gen_report, gen_diff, gen_songs, payload_state,
                     gen_btn, cancel_btn])
        cancel_btn.click(cancel, outputs=cancel_btn)

        file_in.change(lambda f: gr.update(value=f), file_in, video)
        table.select(on_row_select, [lines_state, payload_state],
                     [cur_pos, sel_hint])
        jump_btn.click(lambda s: s, cur_pos, cur_pos, js=_VIDEO_JS)

        split_btn.click(do_split, [lines_state, payload_state, row_no, split_pos],
                        [lines_state, payload_state, table, ed_msg, ed_stats])
        merge_btn.click(do_merge, [lines_state, payload_state, merge_rows],
                        [lines_state, payload_state, table, ed_msg, ed_stats])
        del_btn.click(do_delete, [lines_state, payload_state, merge_rows],
                      [lines_state, payload_state, table, ed_msg, ed_stats])
        upd_btn.click(do_update,
                      [lines_state, payload_state, row_no, new_text, new_start,
                       new_end],
                      [lines_state, payload_state, table, ed_msg, ed_stats])
        ins_btn.click(do_insert,
                      [lines_state, payload_state, row_no, new_text, new_start,
                       new_end],
                      [lines_state, payload_state, table, ed_msg, ed_stats])
        shift_btn.click(do_shift, [lines_state, payload_state, shift_val, row_no],
                        [lines_state, payload_state, table, ed_msg, ed_stats])
        qc_btn.click(do_qc, [lines_state, payload_state],
                     [lines_state, payload_state, table, ed_msg, ed_stats])
        exp_btn.click(export_files,
                      [lines_state, payload_state, exp_fmt, exp_name],
                      [exp_files, ed_msg])

        eval_btn.click(run_eval, [ref_eval, hyp_eval, title_eval],
                       [eval_report, eval_diff])
        p_btn.click(register_profile, [p_name, p_speak, p_sing], p_out)
        p_show.click(profile_status, p_pick, p_info)
        xml_btn.click(danmaku_preview, xml_in, xml_out)
    return demo


demo = build_ui()


def _preload_default_model() -> None:
    """启动时后台预载默认识别模型（省去首个任务的加载等待）。

    失败静默忽略：预载只是优化，首次任务会再次加载并正常报错。
    launcher.py 会调用本函数（历史接口名，保持兼容）。
    """
    import threading

    def _run() -> None:
        try:
            from core.providers.whisper_provider import _get_model

            env = detect_env()
            _get_model("large-v3", env.device, env.compute_type)
        except Exception:  # noqa: BLE001 - 预载失败不影响可用性
            pass

    threading.Thread(target=_run, daemon=True).start()


if __name__ == "__main__":
    _preload_default_model()
    demo.queue().launch(server_name="127.0.0.1", inbrowser=False)
