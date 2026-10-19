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
import sys
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

# 重编码格式先快速转出这么多秒，让用户「立刻看到画面」。
# 4GB / 72 分钟 HEVC 全片重编码约 20 分钟，不能阻塞首帧。
QUICK_PREVIEW_SECONDS = 90


# ---------------- 序列化辅助 ----------------

def lines_to_payload(lines: list[SubtitleLine]) -> list[dict]:
    return [{"start": round(ln.start, 3), "end": round(ln.end, 3),
             "text": ln.text, "speaker": ln.speaker,
             "low_confidence": ln.low_confidence,
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
                low_confidence=bool(row.get("low_confidence"))))
    return out


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

def _build_filmstrip(src: Path, dst: Path, count: int, height: int) -> bool:
    """用 ffmpeg 抽帧并拼接为一张 PNG（胶片条）。

    三种做法实测对比（2GB / 50 分钟 1080p，抽 20 帧）：
    | 做法 | 耗时 |
    |---|---:|
    | 整片解码 + fps 滤镜 + tile | 87.6s |
    | `-skip_frame nokey` + fps + tile | 11.0s |
    | **逐点 `-ss` 抽单帧 + concat/tile** | **5.8s** |

    采用最快的一种：对每个采样点单独 `-ss` 跳读抽一帧（seek 直接定位到
    关键帧附近，无需解码中间内容），再用 concat + tile 拼成一张图。
    采样点均匀分布在整片，得到剪辑软件那样的胶片条。
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

    import shutil
    import subprocess
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="fs_", dir=str(dst.parent)))
    try:
        tiles = []
        for i in range(n):
            t = dur * (i + 0.5) / n
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
    """用 ffprobe 取帧率与时长；失败时给出可用默认值。"""
    from core.env import detect_env

    env = detect_env()
    ffprobe = env.ffprobe_path or "ffprobe"
    fps = 0.0
    dur = 0.0
    try:
        import subprocess

        proc = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate,r_frame_rate",
             "-show_entries", "format=duration",
             "-of", "json", str(src)],
            capture_output=True, text=True, timeout=60)
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
        cmd = [ffmpeg, "-y", "-v", "error", "-fflags", "+genpts",
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


def _transcode_args(env, can_copy: bool) -> list[list[str]]:
    """按「能否只换容器」给出 ffmpeg 参数候选。

    HEVC / AC-3 等浏览器无法解码的编码必须重编码——仅换容器没有用
    （实测 4GB HEVC 直播录制 remux 成 MP4 后仍然黑屏）。
    """
    if can_copy:
        return [["-c", "copy"],
                ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-c:a", "aac", "-b:a", "128k"]]
    return [["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-c:a", "aac", "-b:a", "128k"]]


def _transcode_quick(ffmpeg: str, src: Path, dst: Path,
                     args: list[str], seconds: float) -> bool:
    """快速转出前 `seconds` 秒（用于让用户「立刻看到画面」）。"""
    import subprocess

    tmp = dst.with_suffix(".head.mp4")
    cmd = [ffmpeg, "-y", "-v", "error", "-fflags", "+genpts",
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
                          args: list[str]) -> None:
    """后台把整个文件转完（覆盖前段产物）。

    为什么要后台：4GB / 72 分钟的 HEVC 直播录制，重编码约需 20 分钟，
    用户不可能等。因此先返回前段可播片段，后台继续补完；转完后
    相同 URL 就能访问完整时长。进度通过 `_TRANSCODE_JOBS` 暴露。
    """
    key = str(dst)
    with _TRANSCODE_LOCK:
        job = _TRANSCODE_JOBS.get(key)
        if job and job.get("proc") and job["proc"].poll() is None:
            return                      # 已有任务在跑
        import subprocess

        tmp = dst.with_suffix(".full.mp4")
        cmd = [ffmpeg, "-y", "-v", "error", "-fflags", "+genpts",
               "-i", str(src), *args, "-movflags", "+faststart", str(tmp)]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE)
        except Exception:
            return
        _TRANSCODE_JOBS[key] = {"proc": proc, "tmp": tmp, "dst": dst,
                                "started": time.time()}

    def worker() -> None:
        try:
            proc.wait()
            job = _TRANSCODE_JOBS.get(key) or {}
            if proc.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
                tmp.replace(dst)        # 完整版覆盖前段产物
                job["done"] = True
            else:
                job["error"] = (proc.stderr.read() or b"").decode(
                    "utf-8", "replace")[:300] if proc.stderr else ""
                tmp.unlink(missing_ok=True)
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
        if path == "/api/lyrics/library":
            return self._library_list()
        if path == "/api/lyrics/match":
            return self._lyrics_match(query)
        if path == "/api/profiles":
            from core.streamer import list_profiles

            return self._send_json({"ok": True,
                                    "profiles": list_profiles(ROOT / "profiles")})
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
                                float((query.get("height") or ["44"])[0]))
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

        if path == "/api/profiles/delete":
            from core.streamer import delete_profile

            name = (body.get("name") or "").strip()
            if not name:
                return self._send_json({"ok": False, "error": "需要 name"}, 400)
            removed = delete_profile(name, profiles_dir=str(ROOT / "profiles"))
            return self._send_json({"ok": True, "removed": removed})

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
            prof = StreamerProfile(name=name, speak=engine.enroll(speak, 0.0, 1e9))
            if body.get("sing_wav"):
                prof.sing = engine.enroll(body["sing_wav"], 0.0, 1e9)
            path_saved = save_profile(prof, profiles_dir=str(ROOT / "profiles"))
            return self._send_json({"ok": True, "path": str(path_saved)})
        return self._send_error_json("未知接口：" + path, 404)

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
            wrote = STORE.autosave(state)
            return self._send_json({"ok": True, "wrote": wrote})
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
        if action == "recover":
            lines = STORE.recover_autosave(pid)
            return self._send_json({"ok": True, "lines": lines})
        return self._send_error_json(f"未知操作：{action}", 404)

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

    def _thumbs(self, target: str, count: float, height: float) -> None:
        """按等间隔抽帧，拼成横向胶片条（一张 PNG，供时间轴视频轨显示）。

        只抽一次、缓存到 cache/thumbs/，避免每次重绘都跑 ffmpeg。
        抽帧失败（无 ffmpeg / 非视频）时返回 404，前端会退化为纯色轨道。
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
        key = f"{abs(hash((str(p), p.stat().st_mtime, n, h)))}.png"
        out = cache / key
        if not out.is_file():
            if not _build_filmstrip(p, out, n, h):
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
        try:
            st = p.stat()
            key = f"{abs(hash((str(p), st.st_mtime, st.st_size)))}.mp4"
        except OSError:
            return self._send_error_json("无法读取源文件", 500)
        out = CACHE_DIR / "transcode" / key
        ready = out.is_file() and out.stat().st_size > 0
        job = _TRANSCODE_JOBS.get(str(out))
        progress = 1.0 if (ready and not job) else 0.0
        if job:
            # 按时长估算：用已写出的临时文件大小 / 预估总大小过于不准，
            # 这里改为「已完成秒数」不现实（需要解析 moov），
            # 因此用「临时文件大小相对源文件比例」做粗略进度。
            try:
                tmp_size = job["tmp"].stat().st_size if job["tmp"].is_file() else 0
                # 转码后体积通常小于源（HEVC→H.264 crf23 约 0.6~1.2 倍）
                src_size = p.stat().st_size or 1
                progress = max(0.0, min(0.99, tmp_size / max(src_size * 0.8, 1)))
            except OSError:
                progress = 0.0
        return self._send_json({
            "ok": True, "needed": True, "ready": ready,
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
        try:
            st = src.stat()
            key = f"{abs(hash((str(src), st.st_mtime, st.st_size)))}.mp4"
        except OSError:
            return self._send_error_json("无法读取源文件", 500)
        out = cache / key
        full = out.with_suffix(".complete")

        if not out.is_file() or out.stat().st_size == 0:
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
                ok, last_err = _run_transcode(ffmpeg, src, out, attempts)
                if not ok:
                    return self._send_error_json(
                        "转码失败：无法解码该文件" +
                        (f"（{last_err[:120]}）" if last_err else ""), 500)
            else:
                # 重编码可能耗时很久（4GB/72min 约 20 分钟），
                # 因此先快速转出前段让用户立刻看到画面，再后台补完。
                head_ok = _transcode_quick(ffmpeg, src, out, attempts[0],
                                           QUICK_PREVIEW_SECONDS)
                if not head_ok:
                    ok, last_err = _run_transcode(ffmpeg, src, out, attempts)
                    if not ok:
                        return self._send_error_json(
                            "转码失败：无法解码该文件" +
                            (f"（{last_err[:120]}）" if last_err else ""), 500)
                else:
                    _start_full_transcode(ffmpeg, src, out, attempts[0])

        # 交给通用的文件流逻辑（支持 Range / seek）
        return self._send_file(out)

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
