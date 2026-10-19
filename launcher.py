"""Subtitle Studio 便携版启动器。

职责（全部在便携目录内完成，零系统环境污染）：
- 环境变量预热：bin/ 注入 PATH（内置 FFmpeg）、HF_HOME / TORCH_HOME /
  MODELSCOPE_CACHE 指向 <安装目录>/models/（避免写入 C:\\Users\\...\\.cache）
- 静默启动 Gradio WebUI（隐藏控制台黑窗，--debug 保留）
- 端口健康检查轮询，就绪后自动拉起系统默认浏览器
- 日志重定向到 logs/app.log（10MB 轮转 x3）
- 优雅退出：stop.bat 写入 stop.flag → 关闭 WebUI 服务器线程 → 进程干净退出
  （GPU 显存随进程退出由操作系统回收）

便携目录布局（launcher.py 所在目录即根）：
    launcher.py  run.bat  stop.bat  version.txt
    runtime/     # Embedded Python + site-packages
    app/         # 项目源码（app.py + core/）
    bin/         # ffmpeg.exe / ffprobe.exe
    models/      # hf/ torch/ modelscope/ 缓存根
    profiles/  outputs/  logs/
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.request import urlopen

SERVER_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
PORT_SCAN_ATTEMPTS = 20        # 端口被占时向后尝试的次数
STOP_FLAG = "stop.flag"
POLL_INTERVAL = 0.5            # 主循环 / 健康检查轮询间隔（秒）
SERVER_TIMEOUT = 180.0         # WebUI 启动健康检查超时（模型多时较慢）
URL_TIMEOUT = 2.0              # 单次 HTTP 探测超时
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUPS = 3

# 需要在根目录下确保存在的子目录（模型缓存/用户数据全部落在安装目录内）
ROOT_DIRS = (
    "models/hf", "models/torch", "models/modelscope",
    "profiles", "outputs", "logs", "bin",
)


# ---------------- 路径与环境 ----------------

def app_root() -> Path:
    """便携目录根（launcher.py 所在目录）。"""
    return Path(__file__).resolve().parent


def prepare_env(root: Path) -> dict[str, str]:
    """构造启动环境变量（返回 dict，调用方负责 os.environ.update）。

    - bin/ 前插 PATH：core.env 用 shutil.which 找 FFmpeg，无需改核心代码
    - HF_HOME / TORCH_HOME / MODELSCOPE_CACHE 本地化：模型与缓存写入
      安装目录 models/，卸载/便携迁移不残留 C 盘
    """
    env = {
        "PATH": str(root / "bin") + os.pathsep + os.environ.get("PATH", ""),
        "HF_HOME": str(root / "models" / "hf"),
        "TORCH_HOME": str(root / "models" / "torch"),
        "MODELSCOPE_CACHE": str(root / "models" / "modelscope"),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "GRADIO_ANALYTICS_ENABLED": "False",   # 禁用遥测，纯离线可用
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    return env


def ensure_dirs(root: Path) -> list[Path]:
    """创建根目录下所需的子目录，返回实际新建的目录。"""
    created = []
    for rel in ROOT_DIRS:
        d = root / rel
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(d)
    return created


# ---------------- 端口与服务探测 ----------------

def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def find_free_port(start: int = DEFAULT_PORT, host: str = SERVER_HOST) -> int:
    """从 start 起向后找可绑定端口；全部被占则返回 start（交由上层报错）。"""
    for port in range(start, start + PORT_SCAN_ATTEMPTS):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    return start


def server_url(port: int, host: str = SERVER_HOST) -> str:
    return f"http://{host}:{port}/"


def wait_for_server(port: int, timeout: float = SERVER_TIMEOUT,
                    host: str = SERVER_HOST, interval: float = POLL_INTERVAL) -> bool:
    """轮询 WebUI 首页直到 HTTP 就绪或超时。"""
    url = server_url(port, host)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=URL_TIMEOUT) as resp:
                if resp.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


# ---------------- 停止信号 ----------------

def stop_flag_path(root: Path) -> Path:
    return root / STOP_FLAG


def stop_requested(root: Path) -> bool:
    return stop_flag_path(root).exists()


# ---------------- 控制台与日志 ----------------

def hide_console() -> None:
    """隐藏自身控制台窗口（双击 run.bat 的黑窗）；非 Windows 无操作。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:  # 环境异常时保留窗口，不影响功能
        pass


def setup_logging(root: Path, debug: bool) -> Path:
    """logs/app.log 轮转日志；debug 模式同时输出到控制台。"""
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "app.log"
    handlers: list[logging.Handler] = [
        RotatingFileHandler(path, maxBytes=LOG_MAX_BYTES,
                            backupCount=LOG_BACKUPS, encoding="utf-8"),
    ]
    if debug:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return path


# ---------------- 主流程 ----------------

def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="Subtitle Studio", description="Subtitle Studio 便携版启动器")
    parser.add_argument("--debug", action="store_true",
                        help="保留控制台窗口并输出调试日志")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"WebUI 端口（默认 {DEFAULT_PORT}，被占用时自动向后寻找）")
    args = parser.parse_args(argv)

    root = app_root()
    if not args.debug:
        hide_console()
    log_path = setup_logging(root, args.debug)
    log = logging.getLogger("launcher")

    os.environ.update(prepare_env(root))
    ensure_dirs(root)

    sys.path.insert(0, str(root / "app"))
    log.info("Subtitle Studio 启动中（根目录 %s，日志 %s）", root, log_path)

    # 清理上次异常退出残留的停止标记
    if stop_flag_path(root).exists():
        stop_flag_path(root).unlink()

    try:
        import app as app_module
    except Exception:
        log.exception("应用加载失败：请查看日志排查依赖完整性")
        return 1

    port = find_free_port(args.port)
    if port != args.port:
        log.info("默认端口 %d 被占用，改用 %d", args.port, port)
    url = server_url(port)

    try:
        app_module.demo.queue().launch(
            server_name=SERVER_HOST,
            server_port=port,
            inbrowser=False,             # 浏览器由本启动器统一拉起
            prevent_thread_lock=True,    # launch 不阻塞，主线程继续做健康检查
            show_error=True,
            quiet=not args.debug,
        )
    except Exception:
        log.exception("WebUI 启动失败")
        return 1

    log.info("等待 WebUI 就绪（%s）…", url)
    if wait_for_server(port):
        log.info("WebUI 就绪，正在打开浏览器：%s", url)
        if not webbrowser.open(url):
            log.warning("无法自动打开浏览器，请手动访问 %s", url)
    else:
        log.warning("WebUI 未在 %.0f 秒内就绪，请稍后手动访问 %s",
                    SERVER_TIMEOUT, url)

    # 后台预热默认盲识别模型（命中流水线进程内缓存；失败静默）
    threading.Thread(target=_safe_preload, args=(app_module,),
                     daemon=True).start()

    log.info("服务运行中。停止方式：双击 stop.bat（或删除 %s）", STOP_FLAG)
    try:
        while True:
            time.sleep(POLL_INTERVAL)
            if stop_requested(root):
                log.info("收到停止请求（stop.flag），正在退出…")
                break
    except KeyboardInterrupt:
        log.info("收到 Ctrl+C，正在退出…")

    try:
        app_module.demo.close()  # 关闭 HTTP server 线程，释放端口
    except Exception:
        log.exception("关闭 WebUI 时出错（已忽略）")
    if stop_flag_path(root).exists():
        stop_flag_path(root).unlink()
    log.info("Subtitle Studio 已退出。")
    # 强制退出兜底：防止第三方库残留非 daemon 线程悬挂进程
    os._exit(0)


def _safe_preload(app_module) -> None:
    try:
        app_module._preload_default_model()
    except Exception:  # 预载只是优化，失败不影响首次任务正常加载
        logging.getLogger("launcher").debug("默认模型预载失败（忽略）", exc_info=True)


if __name__ == "__main__":
    raise SystemExit(run())
