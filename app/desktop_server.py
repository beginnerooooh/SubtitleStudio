"""桌面端后端 API：为 Electron 前端提供本地 HTTP 服务。

为什么用标准库 http.server 而不是 FastAPI：
- 便携版要求「零额外依赖、单进程、可嵌入 Electron 子进程」，
  `http.server` + `ThreadingHTTPServer` 已足够，且不会有版本冲突；
- 复用 core/ 下全部逻辑（pipeline2 / editor / evaluate / store），
  前端只做 UI 与时间轴，不重复业务代码。

接口一览（全部返回 JSON，`/api/...`）
────────────────────────────────────
    GET    /api/health               健康检查（含环境信息）
    GET    /api/projects             项目/草稿列表
    POST   /api/projects             新建项目 {video?, name?, settings?}
    GET    /api/projects/<id>        读取项目（含字幕行）
    DELETE /api/projects/<id>        删除项目
    POST   /api/projects/<id>/rename 重命名
    POST   /api/projects/<id>/save   保存字幕行 {lines, push_history?}
    POST   /api/projects/<id>/autosave 自动保存（防抖）
    POST   /api/projects/<id>/undo   撤销
    POST   /api/projects/<id>/redo   重做
    GET    /api/projects/<id>/history 撤销/重做可用步数
    POST   /api/projects/<id>/export 导出 {fmt, name}
    POST   /api/edit/<op>            编辑操作（split/merge/delete/insert/
                                     update/shift/qc），无状态、传 lines
    POST   /api/generate             启动生成任务（异步，返回 job id）
    GET    /api/jobs/<id>            查询任务进度/日志/结果
    POST   /api/jobs/<id>/cancel     取消任务
    POST   /api/evaluate             评测 {ref_srt, hyp_srt} 或直接传 lines
    GET    /api/lyrics/providers     可用歌词源
    POST   /api/lyrics/search        歌词检索 {title, artist, transcript}
    POST   /api/lyrics/endpoint      配置自建歌词服务
    GET    /api/media?path=...       视频/音频流（支持 Range，供 <video> 播放）
    GET    /api/danmaku?path=...     解析弹幕 XML 摘要
    GET    /api/cache/stats          缓存占用
    POST   /api/cache/clean          清理缓存 {max_age_days, max_total_mb}
    GET    /api/profiles             声纹列表
    POST   /api/profiles             注册声纹 {name, speak_wav, sing_wav?}
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent      # app/ 的上一级 = 便携目录根
for _p in (str(ROOT), str(ROOT / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core import editor as ed                      # noqa: E402
from core import song as songmod                   # noqa: E402
from core import store as storemod                 # noqa: E402
from core.config import AppConfig                  # noqa: E402
from core.evaluate import EvalLine, evaluate, load_srt  # noqa: E402
from core.livexml import load_log                  # noqa: E402
from core.models import SubtitleLine, SubtitleWord  # noqa: E402
from core.pipeline2 import Pipeline, PipelineRequest  # noqa: E402
from core.providers import available_providers     # noqa: E402
from core.subtitle import to_ass, to_json, to_lrc, to_srt, to_vtt  # noqa: E402

STORE = storemod.ProjectStore(ROOT / "projects")
CACHE_DIR = ROOT / "cache"
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

# 允许前端访问的媒体扩展名（防目录穿越只做路径存在性校验）
MEDIA_EXTS = {".mp4", ".mkv", ".mov", ".flv", ".webm", ".mp3", ".wav",
              ".flac", ".m4a", ".aac", ".ogg", ".srt", ".ass", ".vtt",
              ".json", ".xml", ".lrc", ".ts", ".rmvb", ".avi", ".wmv"}
# Chromium 的 <video> 无法直接播放这些容器，需要后端用 ffmpeg 转码。
# （MP4/WebM/Ogg 及其音频对应格式可直接播放）
TRANSCODE_CONTAINERS = {".flv", ".mkv", ".ts", ".rmvb", ".avi", ".wmv"}

# 后台转码任务表（同一文件并发打开时复用同一个任务）
_TRANSCODE_JOBS: dict[str, dict] = {}
_TRANSCODE_LOCK = threading.Lock()
# 视频编码器参数（探测一次后复用；优先 NVENC）
_ENCODER_ARGS: list[str] | None = None
# 硬件解码参数（探测一次后复用；优先 cuda）
_HWACCEL_ARGS: list[str] | None = None

# 重编码格式先快速转出这么多秒，让用户「立刻看到画面」。
# 4GB / 72 分钟 HEVC 全片重编码约 7~10 分钟，不能阻塞首帧。
QUICK_PREVIEW_SECONDS = 90

# 预览代理的目标规格。
#
# 效率对比（实测 4GB HEVC 1080p60，转 60 秒素材）：
# | 规格 | 耗时 | 全片预估 | 产物 |
# |---|---:|---:|---:|
# | 1080p60（原规格） | 10.8s | 13.0 分钟 | 12.7 MB |
# | **540p30（采用）** | **6.1s** | **7.4 分钟** | **2.9 MB** |
# | 480p30 | 6.0s | 7.2 分钟 | 2.2 MB |
#
# 字幕校对只需看清画面与口型，540p 足够；产物小 4 倍也让拖动更流畅。
# 注意：这只影响「预览代理」，原始素材与导出字幕不受任何影响。
PREVIEW_VF = "scale=-2:540,fps=30"


# ---------------- 序列化辅助 ----------------

def lines_to_payload(lines: list[SubtitleLine]) -> list[dict]:
    return [{"start": round(ln.start, 3), "end": round(ln.end, 3),
             "text": ln.text, "speaker": ln.speaker,
             "low_confidence": ln.low_confidence,
             # 行级样式覆盖（空 dict 表示跟随全局样式）
             "style": dict(getattr(ln, "style", {}) or {}),
             "words": [{"text": w.text, "start": round(w.start, 3),
                        "end": round(w.end, 3)} for w in ln.words]}
            for ln in lines]


def payload_to_lines(rows: list[dict]) -> list[SubtitleLine]:
    out: list[SubtitleLine] = []
    for row in rows or []:
        words = [SubtitleWord(text=w.get("text", ""),
                              start=float(w.get("start", 0.0)),
                              end=float(w.get("end", 0.0)))
                 for w in row.get("words", []) if w.get("text")]
        if not words and row.get("text"):
            words = [SubtitleWord(text=row["text"],
                                  start=float(row.get("start", 0.0)),
                                  end=float(row.get("end", 0.0)))]
        if words:
            out.append(SubtitleLine(
                words=words, speaker=row.get("speaker", ""),
                low_confidence=bool(row.get("low_confidence")),
                style=dict(row.get("style") or {})))
    return out


def _wav_seconds(path: str) -> float:
    """读取 WAV 的时长（秒）；失败返回 0。"""
    try:
        import wave

        with wave.open(path, "rb") as w:
            sr = w.getframerate() or 1
            return float(w.getnframes()) / float(sr)
    except Exception:
        return 0.0


# ---------------- 后台生成任务 ----------------

def start_job(cfg_kwargs: dict, req_kwargs: dict) -> str:
    """启动一个生成任务，返回 job id（前端轮询 /api/jobs/<id>）。"""
    job_id = uuid.uuid4().hex[:12]
    job: dict[str, Any] = {
        "id": job_id, "status": "running", "ratio": 0.0,
        "message": "启动中…", "log": [], "result": None, "error": "",
        "started": time.time(), "cancel": threading.Event(),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    def on_progress(ratio: float, message: str) -> None:
        job["ratio"], job["message"] = ratio, message

    def on_log(message: str) -> None:
        job["log"].append(f"[{time.strftime('%H:%M:%S')}] {message}")
        if len(job["log"]) > 2000:
            del job["log"][:1000]

    def run() -> None:
        try:
            cfg = AppConfig.from_env(**cfg_kwargs)
            req = PipelineRequest(**req_kwargs)
            res = Pipeline(cfg, req, on_progress=on_progress, on_log=on_log,
                           cancel_event=job["cancel"]).run()
            payload = {
                "lines": lines_to_payload(res.lines),
                "files": res.files, "out_dir": res.out_dir,
                "warnings": res.warnings, "timings": res.timings,
                "songs": [{"start": s.start, "end": s.end, "title": s.title,
                           "artist": s.artist, "source": s.source,
                           "aligned": s.aligned,
                           "line_count": len(s.lines)} for s in res.songs],
                "bgm_flags": res.bgm_flags,
                "live": res.live_summary,
                "evaluation": (res.evaluation.to_dict()
                               if res.evaluation is not None else None),
            }
            job["result"] = payload
            job["status"] = "done"
            job["message"] = f"完成：{len(res.lines)} 行"
        except Exception as exc:  # noqa: BLE001 - 后台线程需兜住一切
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["log"].append("失败：" + job["error"])
            job["log"].append(traceback.format_exc()[-2000:])

    threading.Thread(target=run, daemon=True).start()
    return job_id


# ---------------- HTTP 处理 ----------------

def _build_filmstrip(src: Path, dst: Path, count: int, height: int,
                     start: float = 0.0, span: float = 0.0) -> bool:
    """用 ffmpeg 抽帧并拼接为一张 PNG（胶片条）。

    `start`/`span` 指定区间：时间轴平移或缩放后只抽当前可见区间，
    这样缩略图与时间刻度对得上（否则永远显示整片的第一印象）。
    `span<=0` 时取整片。

    三种做法实测对比（2GB / 50 分钟 1080p，抽 20 帧）：
    | 做法 | 耗时 |
    |---|---:|
    | 整片解码 + fps 滤镜 + tile | 87.6s |
    | `-skip_frame nokey` + fps + tile | 11.0s |
    | **逐点 `-ss` 抽单帧 + concat/tile** | **5.8s** |

    采用最快的一种：对每个采样点单独 `-ss` 跳读抽一帧（seek 直接定位到
    关键帧附近，无需解码中间内容），再用 concat + tile 拼成一张图。
    """
    from core.env import detect_env

    env = detect_env()
    ffmpeg = env.ffmpeg_path or "ffmpeg"
    n = max(4, min(120, int(count)))
    dur = 0.0
    try:
        from core.audio import probe_duration

        dur = probe_duration(src, ffprobe=env.ffprobe_path or "ffprobe")
    except Exception:
        dur = 0.0
    if dur <= 0:
        dur = 60.0

    # 限定到请求的区间，并夹在有效范围内
    base = max(0.0, float(start or 0.0))
    if span and span > 0:
        win = min(float(span), max(0.5, dur - base))
    else:
        win = max(0.5, dur - base)
    if base >= dur:
        base = max(0.0, dur - win)

    import shutil
    import subprocess
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="fs_", dir=str(dst.parent)))
    try:
        tiles = []
        for i in range(n):
            # 采样点落在 [base, base+win) 区间内：
            # 这样时间轴滑到哪，胶片条就显示那段画面。
            t = base + win * (i + 0.5) / n
            f = work / f"t{i:03d}.png"
            proc = subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-ss", f"{t:.3f}",
                 "-i", str(src), "-frames:v", "1",
                 "-vf", f"scale=-2:{height}", str(f)],
                capture_output=True, timeout=60)
            if proc.returncode == 0 and f.is_file() and f.stat().st_size > 0:
                tiles.append(f)
        if not tiles:
            return False
        lst = work / "list.txt"
        lst.write_text("".join(f"file '{t.name}'\n" for t in tiles),
                       encoding="utf-8")
        proc = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(lst), "-vf", f"tile={len(tiles)}x1",
             "-frames:v", "1", str(dst)],
            capture_output=True, timeout=120)
        return (proc.returncode == 0 and dst.is_file()
                and dst.stat().st_size > 0)
    except Exception:
        return False
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _count_keyframes(src: Path, ffprobe: str) -> int:
    """统计关键帧数量（备用工具；当前胶片条实现不需要它）。"""
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
           "-count_frames", "-skip_frame", "nokey",
           "-show_entries", "stream=nb_read_frames",
           "-of", "csv=p=0", str(src)]
    try:
        import subprocess

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            return int((proc.stdout or "0").strip().splitlines()[0] or 0)
    except Exception:
        pass
    return 0


_VIDEOINFO_CACHE: dict[str, dict] = {}

# Chromium 的 <video> 能解码的视频编码。
# 注意：**HEVC (H.265) 不在其中**——很多直播录制（尤其高画质档）用 HEVC，
# 直接 remux 进 MP4 后浏览器仍然黑屏（实测踩过）。遇到这些编码必须转码。
BROWSER_SAFE_VCODECS = {"h264", "vp8", "vp9", "av1"}
# 音频同理：AAC / Opus / Vorbis / MP3 可播；AC-3 / DTS 等需转码
BROWSER_SAFE_ACODECS = {"aac", "opus", "vorbis", "mp3"}


def _probe_video_info(src: Path) -> dict:
    """用 ffprobe 取帧率与时长；失败时给出可用默认值。

    加上 `-analyzeduration`/`-probesize` 限制扫描范围：默认 ffprobe 会
    为了「精确时长」扫全文件，4GB FLV 首次（系统缓存冷）实测要 25 秒。
    限制后仍能正确读到 `format.duration`（实测两者一致），但快得多。
    """
    from core.env import detect_env

    env = detect_env()
    ffprobe = env.ffprobe_path or "ffprobe"
    fps = 0.0
    dur = 0.0
    try:
        import subprocess

        proc = subprocess.run(
            [ffprobe, "-v", "error",
             "-analyzeduration", "2000000", "-probesize", "5000000",
             "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate,r_frame_rate",
             "-show_entries", "format=duration",
             "-of", "json", str(src)],
            capture_output=True, text=True, timeout=180)
        if proc.returncode == 0:
            data = json.loads(proc.stdout or "{}")
            streams = data.get("streams") or [{}]
            for key in ("avg_frame_rate", "r_frame_rate"):
                raw = (streams[0].get(key) or "").strip()
                if "/" in raw:
                    num, den = raw.split("/", 1)
                    try:
                        n, d = float(num), float(den)
                        if n > 0 and d > 0:
                            fps = n / d
                            break
                    except ValueError:
                        continue
            dur = float((data.get("format") or {}).get("duration") or 0)
    except Exception:
        pass
    estimated = False
    if fps <= 0 or fps > 240:
        fps, estimated = 25.0, True
    # 帧率取两位小数，避免 29.970029970... 这类长尾
    return {"ok": True, "fps": round(fps, 3),
            "fps_num": _fps_fraction(fps),
            "duration": round(dur, 3), "estimated": estimated}


def _fps_fraction(fps: float) -> dict:
    """把帧率表示成常见分数（24/25/30/29.97/50/60…），便于时间码换算。"""
    common = {23.976: "24000/1001", 24.0: "24/1", 25.0: "25/1",
              29.97: "30000/1001", 30.0: "30/1", 50.0: "50/1",
              59.94: "60000/1001", 60.0: "60/1", 120.0: "120/1"}
    for k, v in common.items():
        if abs(fps - k) < 0.02:
            return {"text": v, "value": k}
    return {"text": f"{fps:g}/1", "value": round(fps, 3)}


_CODEC_CACHE: dict[str, dict] = {}


def _cache_key(*parts) -> str:
    """由若干值生成**跨进程稳定**的缓存键。

    为什么不用 `hash()`（重要踩坑）：Python 对 str/bytes 的 `hash()` 按进程
    随机加盐（PYTHONHASHSEED），因此**每次重启算出的键都不同**——缓存永远
    命中不了，4GB 视频每次打开都重新转码一遍（实测三次独立进程给出三个
    不同键）。用 md5 对内容做摘要即可稳定。
    """
    import hashlib

    h = hashlib.md5()
    for x in parts:
        h.update(str(x).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:20]


def _file_key(p: Path, *extra) -> str:
    """文件内容指纹键：路径 + 大小 + mtime（+额外参数）。

    用大小与 mtime 而不是内容摘要——4GB 文件读全文算 md5 要几十秒，
    而「路径+大小+mtime」足以判断文件是否被替换过。
    """
    try:
        st = p.stat()
        return _cache_key(str(p.resolve()), st.st_size,
                          int(st.st_mtime), *extra)
    except OSError:
        return _cache_key(str(p), *extra)


def _probe_codecs(src: Path, ffprobe: str) -> dict:
    """探测首个视频/音频流的编码名（小写）。

    用途：判断能否「只换容器不重编码」。HEVC 装进 MP4 浏览器照样不能播，
    因此必须先看编码而不是只看扩展名。

    注意：对 4GB 的 FLV，ffprobe 有时会扫很久（实测同一文件冷启动 37 秒、
    热启动 0.7 秒，取决于系统文件缓存）。因此结果按「路径+大小+时间」缓存，
    避免每次打开视频都卡这么久。
    """
    try:
        st = src.stat()
        key = f"{src}|{st.st_size}|{st.st_mtime}"
    except OSError:
        key = str(src)
    cached = _CODEC_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    out = {"vcodec": "", "acodec": "", "width": 0, "height": 0}
    try:
        import subprocess

        # 关键：限制探测范围。默认 ffprobe 会为了「准确时长」扫描整个文件，
        # 4GB FLV 实测要 25 秒；加 analyzeduration/probesize 后只需 1.3 秒。
        # 我们只关心编码名，不需要精确时长。
        proc = subprocess.run(
            [ffprobe, "-v", "error",
             "-analyzeduration", "2000000",      # 2 秒
             "-probesize", "5000000",            # 5 MB
             "-show_entries", "stream=codec_type,codec_name,width,height",
             "-of", "json", str(src)],
            capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            return out
        data = json.loads(proc.stdout or "{}")
        for s in data.get("streams") or []:
            kind = s.get("codec_type")
            if kind == "video" and not out["vcodec"]:
                out["vcodec"] = str(s.get("codec_name") or "")
                out["width"] = int(s.get("width") or 0)
                out["height"] = int(s.get("height") or 0)
            elif kind == "audio" and not out["acodec"]:
                out["acodec"] = str(s.get("codec_name") or "")
        _CODEC_CACHE[key] = dict(out)
    except Exception:
        pass
    return out


def _run_transcode(ffmpeg: str, src: Path, out: Path,
                   attempts: list[list[str]]) -> tuple[bool, str]:
    """依次尝试若干 ffmpeg 参数组合，成功则把结果落到 `out`。

    返回 (是否成功, 最后一条错误摘要)。

    注意：临时文件必须保留 `.mp4` 扩展名——ffmpeg 靠扩展名推断封装格式，
    用 `.part` 会报 "Unable to choose an output format"（实测踩过）。
    """
    import subprocess

    tmp = out.with_suffix(".tmp.mp4")
    last_err = ""
    for args in attempts:
        cmd = [ffmpeg, "-y", "-v", "error", *_pick_hwaccel(ffmpeg), "-fflags", "+genpts",
               "-i", str(src), *args, "-movflags", "+faststart", str(tmp)]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=5400)
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            tmp.unlink(missing_ok=True)
            continue
        if proc.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            tmp.replace(out)
            return True, ""
        last_err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        tmp.unlink(missing_ok=True)
    return False, last_err


def _sweep_stale_transcode(cache: Path) -> None:
    """清理残留的转码临时文件。

    场景：进程被强杀（关窗、崩溃、系统重启）时后台 ffmpeg 会被中断，
    留下不完整但体积很大的 `.part.mp4`（实测 776 MB）。这些文件不能
    播放（moov 缺失）、白占磁盘，还会让状态查询误判。

    **必须跳过正在写入的文件**：若清理时后台转码仍在跑，删除会失败
    （Windows 下文件被占用，`PermissionError`）——这里主动检查任务表，
    只在没有对应任务时才删，避免无意义的报错。
    """
    try:
        if not cache.is_dir():
            return
        # 正在使用的临时文件路径集合
        active: set[str] = set()
        with _TRANSCODE_LOCK:
            for job in _TRANSCODE_JOBS.values():
                tmp = job.get("tmp")
                if tmp:
                    active.add(str(tmp))
        for pat in ("*.part.mp4", "*.tmp.mp4"):
            for f in cache.glob(pat):
                if str(f) in active:
                    continue                    # 正在写入，保留
                try:
                    f.unlink()
                except OSError:
                    pass
    except Exception:
        pass


def _transcode_running(key: str) -> bool:
    """该目标文件是否有后台转码任务在跑。"""
    with _TRANSCODE_LOCK:
        job = _TRANSCODE_JOBS.get(key)
    return bool(job and job.get("proc") and job["proc"].poll() is None)


def _pick_encoder(ffmpeg: str) -> list[str]:
    """挑选可用的视频编码参数，**优先硬件编码**。

    效率差异明显（实测 4GB HEVC 1080p60 → 540p30 H.264，全片 72 分钟）：
    硬件编码（NVENC）显著快于 CPU 编码，因此探测顺序是
    NVENC → QSV → AMF → CPU 兜底。

    启动时探测一次，结果缓存到 `_ENCODER_ARGS`，避免每次都跑探测。
    """
    global _ENCODER_ARGS
    if _ENCODER_ARGS is not None:
        return list(_ENCODER_ARGS)

    import subprocess

    candidates = [
        # NVENC：现代 NVIDIA 驱动都有；p4 是较快的预设
        ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
         "-cq", "26", "-b:v", "0"],
        # QSV（Intel 核显）
        ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "26"],
        # AMF（AMD）
        ["-c:v", "h264_amf", "-quality", "speed", "-rc", "cqp", "-qp_i", "26"],
        # CPU 兜底
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"],
    ]
    for args in candidates:
        try:
            # 用 1 秒的合成输入测试编码器是否可用（比编码真实文件快得多）
            proc = subprocess.run(
                [ffmpeg, "-v", "error", "-f", "lavfi",
                 "-i", "color=c=black:s=320x240:d=1:r=25",
                 *args, "-f", "null", "-"],
                capture_output=True, timeout=60)
        except Exception:
            continue
        if proc.returncode == 0:
            _ENCODER_ARGS = list(args)
            return list(args)
    _ENCODER_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    return list(_ENCODER_ARGS)


def _pick_hwaccel(ffmpeg: str) -> list[str]:
    """挑选可用的**硬件解码**参数。

    效率关键（实测 4GB HEVC 1080p60 → H.264）：
    - 纯 CPU 软解 HEVC 是瓶颈（编码器已经是 NVENC，但解码仍占满 CPU）
    - 加 `-hwaccel cuda` 后整体快约 1.5~2 倍

    探测方式：用合成输入试一次，失败就退回软解（不冒任何风险）。
    """
    global _HWACCEL_ARGS
    if _HWACCEL_ARGS is not None:
        return list(_HWACCEL_ARGS)

    import subprocess

    for args in (["-hwaccel", "cuda"],
                 ["-hwaccel", "d3d11va"],
                 ["-hwaccel", "auto"]):
        try:
            proc = subprocess.run(
                [ffmpeg, "-v", "error", *args, "-f", "lavfi",
                 "-i", "color=c=black:s=320x240:d=1:r=25",
                 "-c:v", "libx264", "-preset", "ultrafast",
                 "-f", "null", "-"],
                capture_output=True, timeout=60)
        except Exception:
            continue
        if proc.returncode == 0:
            _HWACCEL_ARGS = list(args)
            return list(args)
    _HWACCEL_ARGS = []
    return []


def _transcode_args(env, can_copy: bool) -> list[list[str]]:
    """按「能否只换容器」给出 ffmpeg 参数候选。

    HEVC / AC-3 等浏览器无法解码的编码必须重编码——仅换容器没有用
    （实测 4GB HEVC 直播录制 remux 成 MP4 后仍然黑屏）。

    重编码时：
    - 优先硬件编码器（`_pick_encoder`）
    - 缩到 540p30 作为**预览代理**（见 PREVIEW_VF 的效率对比）

    `env` 允许为 None（便于单测直接调用），此时用系统 PATH 上的 ffmpeg。
    """
    ffmpeg = getattr(env, "ffmpeg_path", None) or "ffmpeg"
    enc = _pick_encoder(ffmpeg)
    if can_copy:
        # remux 保持原画质（不缩放），只在编码不安全时才降规格
        return [["-c", "copy"],
                ["-vf", PREVIEW_VF, *enc, "-c:a", "aac", "-b:a", "96k"]]
    return [["-vf", PREVIEW_VF, *enc, "-c:a", "aac", "-b:a", "96k"]]


def _transcode_quick(ffmpeg: str, src: Path, dst: Path,
                     args: list[str], seconds: float) -> bool:
    """快速转出前 `seconds` 秒（用于让用户「立刻看到画面」）。"""
    import subprocess

    tmp = dst.with_suffix(".head.mp4")
    cmd = [ffmpeg, "-y", "-v", "error", *_pick_hwaccel(ffmpeg), "-fflags", "+genpts",
           "-t", str(seconds), "-i", str(src), *args,
           "-movflags", "+faststart", str(tmp)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=1200)
    except Exception:
        tmp.unlink(missing_ok=True)
        return False
    if proc.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
        tmp.replace(dst)
        return True
    tmp.unlink(missing_ok=True)
    return False


def _start_full_transcode(ffmpeg: str, src: Path, dst: Path,
                          args: list[str], head: Path | None = None) -> None:
    """后台把整个文件转完。

    为什么要后台：4GB / 72 分钟的 HEVC 直播录制，重编码约需 10~20 分钟，
    用户不可能等。因此先给出前段可播片段，后台继续补完。

    **关键设计（修复「处理前能播、处理后不能播」）**：
    完整版写到**另一个文件**（`dst`），前段写到 `head`，两者互不覆盖。
    早期实现是「完整版 replace 掉前段文件」，但 Windows 下当该文件正被
    HTTP 流读取时 `os.replace` 会失败（文件被占用），结果是：后台明明
    转完了，用户手里却仍只有 90 秒的前段——而且前段一旦被替换失败，
    连原来的片段也可能损坏。分开存放后，媒体接口优先用完整版、
    没有时才用前段，既不会失败也不会破坏正在播放的内容。
    """
    key = str(dst)
    with _TRANSCODE_LOCK:
        job = _TRANSCODE_JOBS.get(key)
        if job and job.get("proc") and job["proc"].poll() is None:
            return                      # 已有任务在跑
        import subprocess

        tmp = dst.with_suffix(".part.mp4")
        cmd = [ffmpeg, "-y", "-v", "error", *_pick_hwaccel(ffmpeg), "-fflags", "+genpts",
               "-i", str(src), *args, "-movflags", "+faststart", str(tmp)]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE)
        except Exception:
            return
        _TRANSCODE_JOBS[key] = {"proc": proc, "tmp": tmp, "dst": dst,
                                "head": head, "started": time.time()}

    def worker() -> None:
        try:
            proc.wait()
            job = _TRANSCODE_JOBS.get(key) or {}
            if proc.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
                # 只写自己的目标文件，不碰正在被播放的前段
                tmp.replace(dst)
                job["done"] = True
            else:
                job["error"] = (proc.stderr.read() or b"").decode(
                    "utf-8", "replace")[:300] if proc.stderr else ""
                tmp.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            # 绝不因后台失败影响已能播放的前段
            job = _TRANSCODE_JOBS.get(key) or {}
            job["error"] = f"{type(exc).__name__}: {exc}"
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
        finally:
            with _TRANSCODE_LOCK:
                _TRANSCODE_JOBS.pop(key, None)

    threading.Thread(target=worker, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    server_version = "SubtitleStudioDesktop/1.0"

    # ---------- 基础 ----------

    def log_message(self, fmt: str, *args) -> None:  # 静默访问日志
        if os.environ.get("SUBTITLE_STUDIO_HTTP_LOG"):
            super().log_message(fmt, *args)

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = 400) -> None:
        self._send_json({"ok": False, "error": message}, status)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {"value": data}
        except Exception:
            return {}

    # ---------- 路由 ----------

    def do_GET(self) -> None:  # noqa: N802 - 标准库命名
        try:
            self._route_get()
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(f"{type(exc).__name__}: {exc}", 500)

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._route_post()
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(f"{type(exc).__name__}: {exc}", 500)

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            self._route_delete()
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(f"{type(exc).__name__}: {exc}", 500)

    def _route_get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        if path == "/api/health":
            from core.env import detect_env

            env = detect_env()
            return self._send_json({
                "ok": True, "device": env.device,
                "compute_type": env.compute_type,
                "vram_gb": env.vram_gb, "ffmpeg": env.ffmpeg_available,
                "engines": available_providers(),
                "lyric_providers": songmod.available_lyric_providers(),
                "root": str(ROOT),
            })
        if path == "/api/projects":
            include = query.get("drafts", ["1"])[0] != "0"
            return self._send_json({
                "ok": True,
                "projects": [m.__dict__ for m in
                             STORE.list_projects(include_drafts=include)]})
        if path.startswith("/api/projects/"):
            parts = path.split("/")
            if len(parts) == 4:
                state = STORE.load(parts[3])
                return self._send_json({"ok": True, **state.__dict__,
                                        "meta": state.meta.__dict__})
            if len(parts) == 5 and parts[4] == "history":
                return self._send_json({"ok": True,
                                        **STORE.history_info(parts[3])})
        if path == "/api/lyrics/providers":
            return self._send_json({"ok": True,
                                    "providers": songmod.available_lyric_providers(),
                                    "order": list(songmod.DEFAULT_PROVIDER_ORDER)})
        if path == "/api/keywords":
            return self._keywords_list()
        if path == "/api/lyrics/library":
            return self._library_list()
        if path == "/api/lyrics/match":
            return self._lyrics_match(query)
        if path == "/api/profiles":
            from core.streamer import list_profiles

            return self._send_json({"ok": True,
                                    "profiles": list_profiles(ROOT / "profiles")})
        if path == "/api/uvr/status":
            return self._uvr_status()
        if path == "/api/cache/stats":
            stats = STORE.clean_cache(CACHE_DIR, dry_run=True)
            return self._send_json({"ok": True, **stats})
        if path == "/api/danmaku":
            target = (query.get("path") or [""])[0]
            log = load_log(target) if target else None
            if log is None:
                return self._send_json({"ok": False, "error": "解析失败"}, 404)
            return self._send_json({
                "ok": True, "streamer": log.streamer, "title": log.title,
                "duration": round(log.duration, 1),
                "danmaku": len(log.danmaku), "gifts": len(log.gifts),
                "hotwords": log.hotwords(top=40),
                "requests": log.song_requests(),
                "events": log.event_times()[:200]})
        if path == "/api/jobs":
            with JOBS_LOCK:
                return self._send_json({"ok": True, "jobs": [
                    {"id": j["id"], "status": j["status"],
                     "ratio": j["ratio"], "message": j["message"]}
                    for j in JOBS.values()]})
        if path.startswith("/api/jobs/"):
            jid = path.split("/")[3]
            with JOBS_LOCK:
                job = JOBS.get(jid)
            if job is None:
                return self._send_json({"ok": False, "error": "任务不存在"}, 404)
            return self._send_json({
                "ok": True, "id": job["id"], "status": job["status"],
                "ratio": job["ratio"], "message": job["message"],
                "log": job["log"][-400:], "result": job["result"],
                "error": job["error"]})
        if path == "/api/media":
            return self._serve_media((query.get("path") or [""])[0])
        if path == "/api/videoinfo":
            return self._video_info((query.get("path") or [""])[0])
        if path == "/api/transcode/status":
            return self._transcode_status((query.get("path") or [""])[0])
        if path == "/api/thumbs":
            return self._thumbs((query.get("path") or [""])[0],
                                float((query.get("count") or ["60"])[0]),
                                float((query.get("height") or ["44"])[0]),
                                float((query.get("start") or ["0"])[0]),
                                float((query.get("span") or ["0"])[0]))
        return self._send_error_json("未知接口：" + path, 404)

    def _route_post(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        body = self._read_json()

        if path == "/api/projects":
            state = STORE.create(video=body.get("video", ""),
                                 name=body.get("name", "未命名"),
                                 settings=body.get("settings") or {})
            return self._send_json({"ok": True, "id": state.meta.id,
                                    "meta": state.meta.__dict__})
        if path.startswith("/api/projects/"):
            parts = path.split("/")
            if len(parts) >= 5:
                pid, action = parts[3], parts[4]
                return self._project_action(pid, action, body)
        if path.startswith("/api/edit/"):
            op = path.split("/")[3]
            return self._edit_op(op, body)
        if path == "/api/generate":
            jid = start_job(body.get("config") or {},
                            body.get("request") or {})
            return self._send_json({"ok": True, "job": jid})
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            jid = path.split("/")[3]
            with JOBS_LOCK:
                job = JOBS.get(jid)
            if job is None:
                return self._send_json({"ok": False, "error": "任务不存在"}, 404)
            job["cancel"].set()
            return self._send_json({"ok": True})
        if path == "/api/evaluate":
            return self._evaluate(body)
        if path == "/api/lyrics/search":
            logs: list[str] = []
            rows = songmod.search_lyrics(
                title=body.get("title", ""), artist=body.get("artist", ""),
                transcript=body.get("transcript", ""),
                providers=body.get("providers") or songmod.DEFAULT_PROVIDER_ORDER,
                min_score=float(body.get("min_score", 0.45)),
                on_log=logs.append)
            return self._send_json({"ok": True, "results": rows, "log": logs})
        if path == "/api/lyrics/endpoint":
            name = songmod.set_custom_endpoint(
                body.get("endpoint", ""), body.get("api_key", ""))
            return self._send_json({"ok": True, "name": name})
        if path == "/api/cache/clean":
            stats = STORE.clean_cache(
                CACHE_DIR,
                max_age_days=float(body.get("max_age_days", 7)),
                max_total_mb=float(body.get("max_total_mb", 2048)),
                dry_run=bool(body.get("dry_run", False)))
            return self._send_json({"ok": True, **stats})
        if path == "/api/drafts/clean":
            stale = STORE.clean_old_drafts(
                ttl_days=float(body.get("ttl_days", 30)),
                dry_run=bool(body.get("dry_run", True)))
            return self._send_json({"ok": True,
                                    "drafts": [m.__dict__ for m in stale]})
        if path == "/api/lyrics/library/add":
            return self._library_add(body)
        if path == "/api/lyrics/library/fetch":
            return self._library_fetch(body)
        if path == "/api/lyrics/library/remove":
            return self._library_remove(body)

        if path == "/api/keywords/add":
            return self._keywords_add(body)
        if path == "/api/keywords/bulk":
            return self._keywords_bulk(body)
        if path == "/api/keywords/remove":
            return self._keywords_remove(body)
        if path == "/api/keywords/clear":
            return self._keywords_clear(body)

        if path == "/api/profiles/delete":
            from core.streamer import delete_profile

            name = (body.get("name") or "").strip()
            if not name:
                return self._send_json({"ok": False, "error": "需要 name"}, 400)
            removed = delete_profile(name, profiles_dir=str(ROOT / "profiles"))
            return self._send_json({"ok": True, "removed": removed})

        if path == "/api/uvr/status":
            return self._uvr_status()
        if path == "/api/uvr/enroll":
            return self._uvr_enroll(body)
        if path == "/api/uvr/status":
            return self._uvr_status()
        if path == "/api/profiles":
            from core.streamer import VoiceprintEngine, StreamerProfile, save_profile

            name = (body.get("name") or "").strip()
            speak = body.get("speak_wav") or ""
            if not name or not speak:
                return self._send_json(
                    {"ok": False, "error": "需要 name 与 speak_wav"}, 400)
            for label, p in (("说话样本", speak), ("唱歌样本", body.get("sing_wav") or "")):
                if p and not Path(p).is_file():
                    return self._send_json(
                        {"ok": False, "error": f"{label}文件不存在：{p}"}, 400)
            engine = VoiceprintEngine(device=body.get("device", "cuda"))
            # enroll() 的时间参数是**秒**，内部 _cap 会按文件时长裁剪。
            # 传 1e9 会被当成 5e8 秒算出越界区间并抛「无效音频区间」
            # （实测踩坑）。这里取 WAV 的实际时长。
            prof = StreamerProfile(
                name=name, speak=engine.enroll(speak, 0.0,
                                               _wav_seconds(speak)))
            if body.get("sing_wav"):
                sing = body["sing_wav"]
                prof.sing = engine.enroll(sing, 0.0, _wav_seconds(sing))
            path_saved = save_profile(prof, profiles_dir=str(ROOT / "profiles"))
            return self._send_json({"ok": True, "path": str(path_saved),
                                    "name": name})
        return self._send_error_json("未知接口：" + path, 404)

    # ---------- UVR5（人声分离 / 区间声纹注册） ----------

    def _uvr_status(self) -> None:
        """返回 UVR5 是否可用、有哪些模型。"""
        from core import uvr5

        st = uvr5.status()
        return self._send_json({"ok": True, **st.to_dict()})

    def _uvr_enroll(self, body: dict) -> None:
        """从时间轴选中的区间做声纹注册。

        流程（每一步都可能很慢，因此同步返回但前端带进度窗）：
          1. FFmpeg 按时间切出该区间；
          2. UVR5 的 MDX 模型分离出人声（去掉 BGM/伴奏）；
          3. 重采样到 16kHz（声纹引擎要求）；
          4. 提取声纹并保存为 profile。

        `use_uvr=false` 时跳过第 2 步（直接用原音频），用于素材本来就干净的情况。
        """
        from core import uvr5
        from core.env import detect_env
        from core.streamer import VoiceprintEngine, StreamerProfile, save_profile

        name = (body.get("name") or "").strip()
        src = (body.get("input") or "").strip()
        whole = bool(body.get("whole"))
        try:
            start = float(body.get("start") or 0.0)
            end = float(body.get("end") or 0.0)
        except (TypeError, ValueError):
            return self._send_json({"ok": False, "error": "时间参数无效"}, 400)
        if not name:
            return self._send_json({"ok": False, "error": "请填写声纹名称"}, 400)
        if not src or not Path(src).is_file():
            return self._send_json(
                {"ok": False, "error": f"视频/音频不存在：{src}"}, 400)
        # whole=True：整段导入（外部干声文件，跳过区间检查）
        if not whole:
            if end <= start:
                return self._send_json(
                    {"ok": False, "error": "区间无效：结束时间必须大于开始时间"}, 400)
            if end - start < 2.0:
                return self._send_json(
                    {"ok": False, "error": "区间太短（至少 2 秒）"}, 400)

        use_uvr = bool(body.get("use_uvr", True)) and not whole
        device = body.get("device") or "cpu"
        cache = CACHE_DIR / "voiceprint"
        cache.mkdir(parents=True, exist_ok=True)
        tag = _cache_key(src, start, end, use_uvr)
        vocal = cache / f"vp_{tag}.wav"

        try:
            if use_uvr and uvr5.status().available:
                uvr5.separate_segment_for_voiceprint(
                    src, str(vocal), start, end)
            else:
                # 不分离：直接转成 16kHz 单声道。
                # whole=True 表示整段导入（外部干声），不切时间段；
                # 否则按 start/end 切。
                ff = detect_env().ffmpeg_path or "ffmpeg"
                import subprocess

                cmd = [ff, "-y", "-v", "error"]
                if not whole:
                    cmd += ["-ss", f"{start:.3f}",
                            "-t", f"{end - start:.3f}"]
                cmd += ["-i", src,
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                        str(vocal)]
                subprocess.run(cmd, check=True, capture_output=True)
            engine = VoiceprintEngine(device=device)
            # enroll(wav, start, end) 的参数是**秒**（内部 _cap 会按文件
            # 时长裁剪）。这里传 0 到实际时长即可，别传 1e9 ——
            # _cap 会把它当成「秒」算出越界区间并抛「无效音频区间」。
            dur = _wav_seconds(str(vocal))
            vec = engine.enroll(str(vocal), 0.0, dur)
            prof = StreamerProfile(name=name, speak=vec)
            saved = save_profile(prof, profiles_dir=str(ROOT / "profiles"))
        except Exception as exc:  # noqa: BLE001
            return self._send_json(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)

        return self._send_json({
            "ok": True, "name": name, "path": str(saved),
            "wav": str(vocal),
            # whole 导入时 end-start 为 0，应回报实际音频时长
            "seconds": round(dur if whole else (end - start), 2),
            "uvr": use_uvr, "dim": int(len(vec)),
        })

    def _route_delete(self) -> None:
        parsed = urlparse(self.path)
        parts = parsed.path.rstrip("/").split("/")
        if len(parts) == 4 and parts[1] == "api" and parts[2] == "projects":
            STORE.delete(parts[3])
            return self._send_json({"ok": True})
        return self._send_error_json("未知接口：" + parsed.path, 404)

    # ---------- 项目操作 ----------

    def _project_action(self, pid: str, action: str, body: dict) -> None:
        if action == "save":
            state = STORE.load(pid)
            if "lines" in body:
                state.lines = body["lines"]
            if "settings" in body:
                state.settings = body["settings"]
            if "songs" in body:
                state.songs = body["songs"]
            if "bgm_flags" in body:
                state.bgm_flags = body["bgm_flags"]
            if "log" in body:
                state.log = body["log"][-400:]
            state.meta.line_count = len(state.lines)
            STORE.save(state, push_history=bool(body.get("push_history", True)),
                       note=body.get("note", "edit"))
            return self._send_json({"ok": True,
                                    **STORE.history_info(pid)})
        if action == "autosave":
            state = STORE.load(pid)
            if "lines" in body:
                state.lines = body["lines"]
            state.meta.line_count = len(state.lines)
            # force：生成完成后必须立即落盘，跳过防抖
            # （否则用户关掉应用再打开要重跑几小时任务）
            wrote = STORE.autosave(state, force=bool(body.get("force")))
            return self._send_json({"ok": True, "wrote": wrote,
                                    "line_count": state.meta.line_count})
        if action == "undo":
            state = STORE.undo(pid)
            if state is None:
                return self._send_json({"ok": False, "error": "没有可撤销的操作"})
            return self._send_json({"ok": True, "lines": state.lines,
                                    **STORE.history_info(pid)})
        if action == "redo":
            state = STORE.redo(pid)
            if state is None:
                return self._send_json({"ok": False, "error": "没有可重做的操作"})
            return self._send_json({"ok": True, "lines": state.lines,
                                    **STORE.history_info(pid)})
        if action == "rename":
            meta = STORE.rename(pid, body.get("name", ""))
            return self._send_json({"ok": True, "meta": meta.__dict__})
        if action == "export":
            return self._export(pid, body)
        if action == "burn":
            # 压制带字幕的视频：长任务，走异步 job
            return self._burn_video(pid, body)
        if action == "recover":
            lines = STORE.recover_autosave(pid)
            return self._send_json({"ok": True, "lines": lines})
        return self._send_error_json(f"未知操作：{action}", 404)

    def _burn_video(self, pid: str, body: dict) -> None:
        """把字幕压制进视频（硬字幕，样式与预览一致）。

        实现要点：
        - 先按当前样式生成 ASS（含行级覆盖），再用 ffmpeg 的 `subtitles`
          滤镜烧录。选 ASS 而不是 drawtext 的原因：ASS 支持逐字卡拉OK、
          描边、背景框、对齐与自由定位，与预览的样式模型一一对应。
        - **样式一致性**靠 `to_ass(style=...)`：前端传的 size/color/outline/
          anchor/bg 会被翻译成 ASS 的 Style 行（注意颜色是 BGR 顺序）。
        - 输出到用户指定目录；无 ffmpeg 时明确报错。
        - 走异步 job：压制长视频要几分钟，不能让 HTTP 请求挂着。
        """
        from core.env import detect_env
        from core.subtitle import to_ass

        state = STORE.load(pid)
        lines = payload_to_lines(state.lines)
        if not lines:
            return self._send_json({"ok": False, "error": "没有字幕可压制"}, 400)

        src = (body.get("input") or state.meta.video or "").strip()
        if not src:
            return self._send_json({"ok": False, "error": "项目没有关联视频"}, 400)
        src_path = Path(src)
        if not src_path.is_absolute():
            src_path = (ROOT / src_path).resolve()
        if not src_path.is_file():
            return self._send_json(
                {"ok": False, "error": f"视频不存在：{src_path}"}, 400)

        env = detect_env()
        if not env.ffmpeg_available:
            return self._send_json(
                {"ok": False, "error": "未找到 FFmpeg，无法压制视频"}, 503)

        style = dict(body.get("style") or {})
        out_dir = (body.get("out_dir") or "").strip()
        if out_dir:
            out = Path(out_dir)
            if not out.is_absolute():
                return self._send_json(
                    {"ok": False, "error": "out_dir 必须是绝对路径"}, 400)
        else:
            out = STORE.project_dir(pid) / "exports"
        try:
            out.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return self._send_json(
                {"ok": False, "error": f"无法创建目录：{exc}"}, 400)

        name = (body.get("name") or state.meta.name or "subtitle").strip()
        dst = out / f"{name}_sub.mp4"

        # 先把 ASS 写到项目目录（作为压制输入，也便于用户检查样式）
        ass_path = STORE.project_dir(pid) / "burn.ass"
        ass_path.write_text(
            to_ass(lines, style=style, mark_low_confidence=False),
            encoding="utf-8")

        job_id = uuid.uuid4().hex[:12]
        job: dict[str, Any] = {
            "id": job_id, "status": "running", "ratio": 0.0,
            "message": "准备压制…", "log": [], "result": None, "error": "",
            "started": time.time(), "cancel": threading.Event(),
        }
        with JOBS_LOCK:
            JOBS[job_id] = job

        def run() -> None:
            import subprocess
            try:
                # 路径含空格/冒号会让 subtitles 滤镜解析失败（实测报
                # 「Unable to parse "original_size" option value」）。
                # 转义规则很脆弱，因此改为：把 ASS 放到一个临时工作目录，
                # 用**纯 ASCII 文件名**（无空格无冒号），并让 ffmpeg 以该
                # 目录为 cwd —— 滤镜里只写相对文件名，彻底避开转义问题。
                work = Path(tempfile.mkdtemp(prefix="burn_"))
                safe_ass = work / "sub.ass"
                shutil.copyfile(ass_path, safe_ass)
                filt = "subtitles=sub.ass"
                enc = _pick_encoder(env.ffmpeg_path or "ffmpeg")
                cmd = [env.ffmpeg_path or "ffmpeg", "-y", "-v", "error",
                       *_pick_hwaccel(env.ffmpeg_path or "ffmpeg"),
                       "-i", str(src_path), "-vf", filt, *enc,
                       "-c:a", "copy", str(dst)]
                job["message"] = "正在压制字幕…"
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE, cwd=str(work))
                while proc.poll() is None:
                    if job["cancel"].is_set():
                        proc.kill()
                        job["status"] = "error"
                        job["error"] = "已取消"
                        return
                    # 无精确进度：用已耗时估算（长视频压制通常 1~3 倍速）
                    elapsed = time.time() - job["started"]
                    try:
                        dur = float(state.meta.duration or 0)
                    except Exception:
                        dur = 0.0
                    if dur > 0:
                        job["ratio"] = min(0.99, elapsed / max(dur, 1) * 0.5)
                    time.sleep(1.0)
                if proc.returncode == 0 and dst.is_file() and dst.stat().st_size > 0:
                    job["result"] = {"file": str(dst),
                                     "size_mb": round(dst.stat().st_size / 1048576, 1)}
                    job["status"] = "done"
                    job["ratio"] = 1.0
                    job["message"] = "压制完成"
                else:
                    err = (proc.stderr.read() or b"").decode("utf-8", "replace")
                    job["status"] = "error"
                    job["error"] = err[-400:] or "压制失败"
            except Exception as exc:  # noqa: BLE001
                job["status"] = "error"
                job["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    shutil.rmtree(work, ignore_errors=True)
                except Exception:
                    pass

        threading.Thread(target=run, daemon=True).start()
        return self._send_json({"ok": True, "job": job_id, "ass": str(ass_path),
                                "out": str(dst)})

    def _export(self, pid: str, body: dict) -> None:
        """导出字幕。

        `out_dir` 非空时导出到用户指定目录（桌面端用系统「另存为」对话框选择）；
        为空则导出到项目目录下的 exports/。
        """
        state = STORE.load(pid)
        lines = payload_to_lines(state.lines)
        if not lines:
            return self._send_json({"ok": False, "error": "没有字幕行"}, 400)
        fmt = body.get("fmt") or ["srt"]
        if isinstance(fmt, str):
            fmt = [fmt]
        name = (body.get("name") or state.meta.name or "subtitle").strip()
        out_dir = (body.get("out_dir") or "").strip()
        if out_dir:
            out = Path(out_dir)
            if not out.is_absolute():
                return self._send_json(
                    {"ok": False, "error": "out_dir 必须是绝对路径"}, 400)
        else:
            out = STORE.project_dir(pid) / "exports"
        try:
            out.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return self._send_json(
                {"ok": False, "error": f"无法创建目录：{exc}"}, 400)
        writers = {"srt": to_srt, "lrc": lambda ls: to_lrc(ls, title=name),
                   "ass": to_ass, "vtt": to_vtt,
                   "json": lambda ls: to_json(ls, title=name)}
        paths = []
        errors = []
        for f in fmt:
            fn = writers.get(f)
            if fn is None:
                continue
            p = out / f"{name}.{f}"
            try:
                p.write_text(fn(lines), encoding="utf-8")
                paths.append(str(p))
            except Exception as exc:
                errors.append(f"{f}: {exc}")
        if not paths:
            return self._send_json(
                {"ok": False, "error": "；".join(errors) or "没有有效格式"}, 400)
        return self._send_json({"ok": True, "files": paths,
                                "out_dir": str(out), "errors": errors})

    # ---------- 编辑操作（无状态） ----------

    def _edit_op(self, op: str, body: dict) -> None:
        lines = payload_to_lines(body.get("lines") or [])
        payload: dict[str, Any] = {}
        if op == "split":
            lines, msg = ed.split_line(lines, int(body.get("index", 0)),
                                       int(body.get("char_pos", 1)))
        elif op == "merge":
            lines, msg = ed.merge_lines(lines, body.get("indices") or [])
        elif op == "delete":
            lines, msg = ed.delete_lines(lines, body.get("indices") or [])
        elif op == "insert":
            lines, msg = ed.insert_line(
                lines, int(body.get("index", len(lines))),
                body.get("text", ""), float(body.get("start", 0.0)),
                float(body.get("end", 0.0)))
        elif op == "update":
            kwargs: dict[str, Any] = {}
            if body.get("text") is not None:
                kwargs["text"] = body["text"]
            for key in ("start", "end"):
                if body.get(key) not in (None, ""):
                    kwargs[key] = float(body[key])
            lines, msg = ed.update_line(lines, int(body.get("index", 0)),
                                        **kwargs)
        elif op == "shift":
            idx = body.get("index")
            lines, msg = ed.shift_lines(
                lines, float(body.get("delta", 0.0)),
                int(idx) if idx not in (None, "") else None)
        elif op == "retime":
            lines, msg = ed.retime_by_ratio(lines, float(body.get("factor", 1.0)))
        elif op == "retime_edge":
            kwargs: dict[str, Any] = {}
            for key in ("start", "end"):
                if body.get(key) not in (None, ""):
                    kwargs[key] = float(body[key])
            lines, msg = ed.retime_edge(lines, int(body.get("index", 0)), **kwargs)
        elif op == "copy":
            data, msg = ed.copy_lines(lines, body.get("indices") or [])
            payload["clipboard"] = data
        elif op == "paste":
            at = body.get("at")
            idx = body.get("index")
            lines, msg = ed.paste_lines(
                lines, body.get("clipboard") or [],
                at=float(at) if at not in (None, "") else None,
                index=int(idx) if idx not in (None, "") else None)
        elif op == "duplicate":
            lines, msg = ed.duplicate_lines(lines, body.get("indices") or [])
        elif op == "set_style":
            # 给选中行设置样式覆盖（style 为空 dict 表示清除覆盖）
            idx = body.get("indices") or []
            style = dict(body.get("style") or {})
            n = 0
            for i in idx:
                if 0 <= int(i) < len(lines):
                    lines[int(i)].style = dict(style)
                    n += 1
            msg = f"已为 {n} 行设置样式" if n else "请先选中要设置样式的行"
        elif op == "apply_style_all":
            # 把样式应用到**所有**字幕（含清除：style 为空时清空全部覆盖）
            style = dict(body.get("style") or {})
            for ln in lines:
                ln.style = dict(style)
            msg = (f"已把样式应用到全部 {len(lines)} 行" if style
                   else f"已清除全部 {len(lines)} 行的样式覆盖")
        elif op == "qc":
            lines, m1 = ed.fix_overlaps(lines)
            lines, m2 = ed.split_long_lines(
                lines, int(body.get("max_chars", 18)))
            msg = f"{m1}；{m2}"
        elif op == "split_long":
            lines, msg = ed.split_long_lines(
                lines, int(body.get("max_chars", 18)))
        elif op == "fix_overlaps":
            lines, msg = ed.fix_overlaps(lines)
        else:
            return self._send_error_json(f"未知编辑操作：{op}", 404)
        payload.update({"ok": True, "message": msg,
                        "lines": lines_to_payload(lines),
                        "stats": ed.stats(lines)})
        return self._send_json(payload)

    # ---------- 评测 ----------

    def _evaluate(self, body: dict) -> None:
        ref_path = body.get("ref_srt", "")
        if not ref_path or not Path(ref_path).is_file():
            return self._send_json({"ok": False, "error": "参考 SRT 不存在"}, 400)
        ref = load_srt(ref_path)
        if body.get("lines"):
            hyp = [EvalLine(start=float(r["start"]), end=float(r["end"]),
                            text=r.get("text", ""))
                   for r in body["lines"]]
        elif body.get("hyp_srt") and Path(body["hyp_srt"]).is_file():
            hyp = load_srt(body["hyp_srt"])
        else:
            return self._send_json(
                {"ok": False, "error": "需要 lines 或 hyp_srt"}, 400)
        report = evaluate(ref, hyp)
        out = ROOT / "outputs" / "_eval_last"
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(report.summary_md("评测报告"),
                                       encoding="utf-8")
        (out / "report.json").write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
        return self._send_json({"ok": True, "report": report.to_dict(),
                                "markdown": report.summary_md("评测报告")})

    # ---------- 视频轨缩略图（剪辑软件的胶片条） ----------

    def _thumbs(self, target: str, count: float, height: float,
                start: float = 0.0, span: float = 0.0) -> None:
        """按等间隔抽帧，拼成横向胶片条（一张 PNG，供时间轴视频轨显示）。

        `start`/`span` 指定时间区间：时间轴平移或缩放后，只取当前可见
        区间对应的画面，这样缩略图才会随视图滑动而更新。
        `span<=0` 时取整片（旧行为）。
        只抽一次、缓存到 cache/thumbs/，避免每次重绘都跑 ffmpeg。
        """
        if not target:
            return self._send_error_json("缺少 path 参数", 400)
        p = Path(unquote(target))
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if not p.is_file():
            return self._send_error_json(f"文件不存在：{p}", 404)
        n = max(4, min(120, int(count or 60)))
        h = max(24, min(120, int(height or 44)))
        cache = CACHE_DIR / "thumbs"
        cache.mkdir(parents=True, exist_ok=True)
        # 起止时间按 0.5 秒取整做键：轻微拖动不会反复重新抽帧
        s0 = max(0.0, round(float(start or 0.0) * 2) / 2)
        sp = max(0.0, round(float(span or 0.0) * 2) / 2)
        key = f"{_file_key(p, 'thumbs', n, h, s0, sp)}.png"
        out = cache / key
        if not out.is_file():
            if not _build_filmstrip(p, out, n, h, start=s0, span=sp):
                return self._send_error_json("无法生成缩略图", 404)
        data = out.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    # ---------- 歌词库写操作 ----------

    def _library_add(self, body: dict) -> None:
        """导入歌词：可给文件路径、或直接给文本。"""
        lib = self._library()
        path = (body.get("path") or "").strip()
        text = body.get("text") or ""
        title = (body.get("title") or "").strip()
        artist = (body.get("artist") or "").strip()
        try:
            if path:
                p = Path(path)
                if not p.is_absolute():
                    p = (ROOT / p).resolve()
                if not p.is_file():
                    return self._send_json(
                        {"ok": False, "error": f"文件不存在：{p}"}, 400)
                doc = lib.add_file(p)
            elif text.strip():
                if not title:
                    return self._send_json(
                        {"ok": False, "error": "粘贴歌词时需要提供歌名"}, 400)
                doc = lib.add_text(text, title=title, artist=artist)
            else:
                return self._send_json(
                    {"ok": False, "error": "需要 path 或 text"}, 400)
        except Exception as exc:  # noqa: BLE001
            return self._send_json({"ok": False, "error": str(exc)}, 400)
        return self._send_json({"ok": True, "title": doc.title,
                                "artist": doc.artist,
                                "line_count": doc.line_count,
                                "size": lib.size})

    def _library_fetch(self, body: dict) -> None:
        """按歌名（+可选歌手）在线拉取歌词并入库。"""
        lib = self._library()
        title = (body.get("title") or "").strip()
        artist = (body.get("artist") or "").strip()
        if not title:
            return self._send_json({"ok": False, "error": "需要 title"}, 400)
        logs: list[str] = []
        try:
            added = lib.fetch(title, artist, on_log=logs.append)
        except Exception as exc:  # noqa: BLE001
            return self._send_json({"ok": False, "error": str(exc),
                                    "log": logs}, 500)
        return self._send_json({
            "ok": True, "added": [{"title": d.title, "artist": d.artist,
                                   "line_count": d.line_count,
                                   "synced": d.synced} for d in added],
            "size": lib.size, "log": logs,
            "error": "" if added else "没有找到可用歌词",
        })

    def _library_remove(self, body: dict) -> None:
        lib = self._library()
        title = (body.get("title") or "").strip()
        artist = (body.get("artist") or "").strip()
        if not title:
            return self._send_json({"ok": False, "error": "需要 title"}, 400)
        ok = lib.remove(title, artist)
        return self._send_json({"ok": ok, "size": lib.size})

    # ---------- 关键词字库 ----------

    def _keyword_store(self):
        from core.keywords import KeywordStore

        return KeywordStore(ROOT / "keywords")

    def _keywords_list(self) -> None:
        st = self._keyword_store()
        return self._send_json({
            "ok": True, "size": st.size,
            "items": [k.to_dict() for k in st.list_items()],
            "prompt": st.prompt_text(),
        })

    def _keywords_add(self, body: dict) -> None:
        st = self._keyword_store()
        word = (body.get("word") or "").strip()
        alias = body.get("alias") or []
        if isinstance(alias, str):
            alias = [a.strip() for a in alias.split(",") if a.strip()]
        try:
            k = st.add(word, alias=alias,
                       note=(body.get("note") or "").strip(),
                       category=(body.get("category") or "").strip())
        except Exception as exc:  # noqa: BLE001
            return self._send_json({"ok": False, "error": str(exc)}, 400)
        return self._send_json({"ok": True, "word": k.word, "size": st.size})

    def _keywords_bulk(self, body: dict) -> None:
        st = self._keyword_store()
        added = st.add_many(body.get("text") or "",
                            category=(body.get("category") or "").strip())
        return self._send_json({"ok": True, "added": added, "size": st.size})

    def _keywords_remove(self, body: dict) -> None:
        st = self._keyword_store()
        ok = st.remove((body.get("word") or "").strip())
        return self._send_json({"ok": ok, "size": st.size})

    def _keywords_clear(self, body: dict) -> None:
        st = self._keyword_store()
        n = st.clear()
        return self._send_json({"ok": True, "removed": n, "size": st.size})

    # ---------- 歌词库 ----------

    def _library(self):
        from core.lyrics_library import LyricLibrary

        return LyricLibrary(ROOT / "lyrics")

    def _library_list(self) -> None:
        lib = self._library()
        return self._send_json({
            "ok": True, "size": lib.size,
            "docs": [{"title": d.title, "artist": d.artist,
                      "line_count": d.line_count, "synced": d.synced,
                      "source": d.source, "path": d.path}
                     for d in lib.list_docs()],
        })

    def _lyrics_match(self, query: dict) -> None:
        """预览「字幕与某首歌词」的匹配结果（不改动项目数据）。"""
        from core.lyrics_library import match_document

        pid = (query.get("project") or [""])[0]
        title = (query.get("title") or [""])[0]
        artist = (query.get("artist") or [""])[0]
        lib = self._library()
        doc = lib.get(title, artist) or lib.find_by_title(title) if title else None
        if doc is None:
            return self._send_json({"ok": False, "error": "歌词库中找不到该曲目"}, 404)
        if not pid:
            return self._send_json({"ok": False, "error": "需要 project 参数"}, 400)
        state = STORE.load(pid)
        lines = payload_to_lines(state.lines)
        rep = match_document(lines, doc)
        return self._send_json({"ok": True, **rep.to_dict()})

    # ---------- 转码进度（供前端显示「正在转码」弹窗） ----------

    def _transcode_status(self, target: str) -> None:
        """查询某文件是否需要转码、以及后台转码进度。

        前端在打开不支持的容器时先查这里：
        - `needed=false`：可直接播放
        - `ready=true`：已有可用产物（可能是前段），可先播
        - `progress`：0~1，转码进度（按时长估算）
        """
        from core.env import detect_env

        env = detect_env()
        if not target:
            return self._send_error_json("缺少 path 参数", 400)
        p = Path(unquote(target))
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if not p.is_file():
            return self._send_error_json(f"文件不存在：{p}", 404)

        needed = p.suffix.lower() in TRANSCODE_CONTAINERS
        if not needed:
            return self._send_json({"ok": True, "needed": False,
                                    "ready": True, "progress": 1.0})

        info = _probe_codecs(p, env.ffprobe_path or "ffprobe")
        vcodec = (info.get("vcodec") or "").lower()
        acodec = (info.get("acodec") or "").lower()
        can_copy = (vcodec in BROWSER_SAFE_VCODECS
                    and acodec in BROWSER_SAFE_ACODECS)
        base = _file_key(p)
        head = CACHE_DIR / "transcode" / f"{base}.head.mp4"
        full = CACHE_DIR / "transcode" / f"{base}.mp4"
        # ready 表示「已有可播放的东西」——前段或完整版都算。
        # 前端据此放行播放，不必等全片转完。
        has_head = head.is_file() and head.stat().st_size > 0
        has_full = full.is_file() and full.stat().st_size > 0
        ready = has_head or has_full
        job = _TRANSCODE_JOBS.get(str(full))
        progress = 1.0 if has_full else 0.0
        if job:
            # 粗估进度：临时文件大小相对源文件（HEVC→H.264 约 0.5~1.0 倍）
            try:
                tmp_size = job["tmp"].stat().st_size if job["tmp"].is_file() else 0
                src_size = p.stat().st_size or 1
                progress = max(0.0, min(0.99, tmp_size / max(src_size * 0.6, 1)))
            except OSError:
                progress = 0.0
        return self._send_json({
            "ok": True, "needed": True, "ready": ready,
            "hasHead": has_head, "hasFull": has_full,
            "copyable": can_copy, "vcodec": vcodec, "acodec": acodec,
            "progress": round(progress, 3),
            "transcoding": bool(job),
        })

    # ---------- 视频信息（帧率，供时间码「时:分:秒:帧」使用） ----------

    def _video_info(self, target: str) -> None:
        """返回视频的帧率与时长。

        「时:分:秒:帧」时间码需要准确帧率：优先用 ffprobe 的 avg_frame_rate
        （如 30000/1001 = 29.97），拿不到时退回 25fps 并标记 estimated。
        """
        if not target:
            return self._send_error_json("缺少 path 参数", 400)
        p = Path(unquote(target))
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if not p.is_file():
            return self._send_error_json(f"文件不存在：{p}", 404)
        cache_key = f"{p}|{p.stat().st_mtime}"
        cached = _VIDEOINFO_CACHE.get(cache_key)
        if cached is not None:
            return self._send_json(cached)
        info = _probe_video_info(p)
        _VIDEOINFO_CACHE[cache_key] = info
        return self._send_json(info)

    # ---------- 媒体流（支持 Range，供 <video> 拖动） ----------

    def _serve_media(self, target: str) -> None:
        """媒体流。

        Chromium 只能播放它自带的容器集合（MP4/WebM/Ogg…），**FLV、MKV、
        RMVB、TS 等一律不能直接播放**——表现为 `<video>` 黑屏或 error。
        因此这里对不支持的容器走「ffmpeg 实时转码为 fMP4」的代理路径：
        仍然以流的方式输出，`<video>` 可以边下边播、可 seek（靠 Range 分片）。
        """
        if not target:
            return self._send_error_json("缺少 path 参数", 400)
        p = Path(unquote(target))
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if not p.is_file():
            return self._send_error_json(f"文件不存在：{p}", 404)
        if p.suffix.lower() not in MEDIA_EXTS:
            return self._send_error_json(f"不支持的类型：{p.suffix}", 403)
        if p.suffix.lower() in TRANSCODE_CONTAINERS:
            return self._serve_transcoded(p)
        return self._send_file(p)

    def _send_file(self, p: Path) -> None:
        """按 Range 发送文件（供 <video> 播放与拖动）。"""
        size = p.stat().st_size
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                end = min(end, size - 1)
        length = max(0, end - start + 1)
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(p, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    # ---------- 容器转码代理（FLV / MKV / TS 等） ----------

    def _serve_transcoded(self, src: Path) -> None:
        """把 Chromium 不支持的容器转成可播放的 MP4，再走普通 Range 流。

        为什么不在线流式转码（踩坑记录）
        ────────────────────────────
        流式输出 fMP4（`frag_keyframe+empty_moov`）虽然能边转边播，但
        **fMP4 分片没有总时长**——浏览器只能报告「已缓冲的时长」，
        `<video>.duration` 会是 2.52s / 10.45s 这样的错误值（实测），
        导致进度条与字幕时间轴完全错位，对字幕工具是致命的。

        因此改为「先转成带 faststart 的完整 MP4 并缓存」：
        - remux（`-c copy`）通常只需 0.1 秒/20 秒素材，几乎无感；
        - 输出文件有正确 moov + 总时长，走普通 Range 流，可自由 seek；
        - 缓存按「路径 + 修改时间 + 大小」命名，二次打开秒开。
        """
        from core.env import detect_env

        env = detect_env()
        ffmpeg = env.ffmpeg_path or "ffmpeg"
        if not env.ffmpeg_available:
            return self._send_error_json(
                f"该格式（{src.suffix}）浏览器无法直接播放，且未找到 FFmpeg"
                " 用于转码。请安装 FFmpeg 后重试。", 503)

        cache = CACHE_DIR / "transcode"
        cache.mkdir(parents=True, exist_ok=True)
        base = _file_key(src)
        # 前段（可立即播放）与完整版分开存放，互不覆盖。
        # 详见 _start_full_transcode 的注释：Windows 下覆盖正在播放的文件会失败。
        head = cache / f"{base}.head.mp4"
        full = cache / f"{base}.mp4"

        # 完整版已就绪 → 直接用（最佳情况：有全片时长）
        if full.is_file() and full.stat().st_size > 0:
            return self._send_file(full)
        # 已有前段 → 先播前段，同时确保后台在继续转完整版
        if head.is_file() and head.stat().st_size > 0:
            if not _transcode_running(str(full)):
                env2 = detect_env()
                info = _probe_codecs(src, env2.ffprobe_path or "ffprobe")
                can_copy = ((info.get("vcodec") or "").lower()
                            in BROWSER_SAFE_VCODECS
                            and (info.get("acodec") or "").lower()
                            in BROWSER_SAFE_ACODECS)
                args = _transcode_args(env2, can_copy)
                _start_full_transcode(env2.ffmpeg_path or "ffmpeg", src,
                                      full, args[0], head=head)
            return self._send_file(head)

        info = _probe_codecs(src, env.ffprobe_path or "ffprobe")
        vcodec = (info.get("vcodec") or "").lower()
        acodec = (info.get("acodec") or "").lower()
        # 只有编码本身浏览器能解码时才可以无损 remux。
        # 仅换容器没用：HEVC 装进 MP4 依然黑屏（实测 4GB FLV 场景）。
        can_copy = (vcodec in BROWSER_SAFE_VCODECS
                    and acodec in BROWSER_SAFE_ACODECS)
        attempts = _transcode_args(env, can_copy)
        try:
            self.log_message(
                f"transcode {src.name}: v={vcodec or '?'} a={acodec or '?'} "
                f"copy={can_copy}")
        except Exception:
            pass

        if can_copy:
            # remux 极快（通常 <5s），直接同步完成
            ok, last_err = _run_transcode(ffmpeg, src, full, attempts)
            if not ok:
                return self._send_error_json(
                    "转码失败：无法解码该文件" +
                    (f"（{last_err[:120]}）" if last_err else ""), 500)
            return self._send_file(full)

        # 重编码可能耗时很久（4GB/72min 约 10 分钟），因此：
        #   1) 先快速转出前段（head）让用户立刻看到画面
        #   2) 后台把完整版写到 full（不碰 head）
        head_ok = _transcode_quick(ffmpeg, src, head, attempts[0],
                                   QUICK_PREVIEW_SECONDS)
        if head_ok:
            _start_full_transcode(ffmpeg, src, full, attempts[0], head=head)
            return self._send_file(head)
        # 前段都转不出来 → 同步全量转码（慢，但至少能用）
        ok, last_err = _run_transcode(ffmpeg, src, full, attempts)
        if not ok:
            return self._send_error_json(
                "转码失败：无法解码该文件" +
                (f"（{last_err[:120]}）" if last_err else ""), 500)
        return self._send_file(full)

    def _stream_ffmpeg(self, cmd: list[str], src: Path) -> bool:
        """执行 ffmpeg 并把输出边读边写。首个字节到达后才发响应头，
        这样失败时可以干净地返回 JSON 错误（已发头就无法改状态码）。"""
        import subprocess

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, bufsize=0)
        except Exception:
            return False
        try:
            first = proc.stdout.read(1) if proc.stdout else b""
            if not first:
                proc.wait(timeout=20)
                return False
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Accept-Ranges", "none")
            self.end_headers()
            self.wfile.write(first)
            while True:
                chunk = proc.stdout.read(1 << 16)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # 客户端 seek 或关闭播放器时主动断开，属于正常情况
            return True
        except Exception:
            return False
        finally:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass


def serve(host: str = "127.0.0.1", port: int = 8756,
          quiet: bool = False) -> ThreadingHTTPServer:
    """启动 API 服务（返回 server 对象，调用方决定是否 serve_forever）。"""
    # 清理上次被强杀留下的转码临时文件（见 _sweep_stale_transcode）
    _sweep_stale_transcode(CACHE_DIR / "transcode")
    httpd = ThreadingHTTPServer((host, port), Handler)
    if not quiet:
        print(f"[api] Subtitle Studio 后端已启动：http://{host}:{port}", flush=True)
    return httpd


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Subtitle Studio 桌面端后端")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8756)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    httpd = serve(args.host, args.port, quiet=args.quiet)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
