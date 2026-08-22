"""Subtitle Studio 便携版一键构建脚本（在【开发机 Windows】上运行）。

产物：dist/SubtitleStudio/ —— 零环境依赖的完整便携目录
    launcher.py / run.bat / run-debug.bat / stop.bat / version.txt
    runtime/   Embedded Python 3.12 + 全部依赖（site-packages）
    app/       项目源码（app.py + core/，已过滤缓存/测试/文档）
    bin/       ffmpeg.exe / ffprobe.exe（静态构建）
    models/    模型缓存根（HF_HOME / TORCH_HOME / MODELSCOPE_CACHE）
    profiles/  outputs/  logs/   （空目录，运行时生成数据）

用法（Windows 开发机 PowerShell / CMD）：
    python packaging\\build_portable.py                        # CPU 完整构建
    python packaging\\build_portable.py --torch-index cu124    # GPU 构建
    python packaging\\build_portable.py --ffmpeg-dir D:\\tools\\ffmpeg-bin
    python packaging\\build_portable.py --with-models --preset full
    python packaging\\build_portable.py --installer            # 构建后自动编译 Setup.exe

镜像（中国大陆网络）：
    --python-mirror https://mirrors.huaweicloud.com/python
    --pip-mirror https://pypi.tuna.tsinghua.edu.cn/simple
    环境变量 HF_ENDPOINT=https://hf-mirror.com（模型下载）
"""
from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGING = REPO / "packaging"
DEFAULT_DIST = REPO / "dist" / "SubtitleStudio"
DEFAULT_PYTHON_VERSION = "3.12.8"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

FFMPEG_ZIP_SOURCES = {
    # BtbN 静态完整构建（含 ffprobe，推荐）
    "btbn": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
    # gyan.dev essentials（含 ffprobe）
    "gyan": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
}

# 源码复制时排除的目录名（相对 app/ 源树）
EXCLUDED_DIR_NAMES = {
    "__pycache__", ".git", ".github", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".idea", ".vscode", "node_modules",
    "tests", "docs", "dist", "outputs", "profiles",
}
EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo", ".log"}
EXCLUDED_FILE_NAMES = {".gitignore", ".gitattributes", ".DS_Store", "Thumbs.db", ".env"}


class BuildError(RuntimeError):
    """构建失败（用户可读消息）。"""


# ---------------- 通用工具 ----------------

def log(message: str) -> None:
    print(f"[build] {message}", flush=True)


def download_file(url: str, dst: Path, timeout: float = 60.0) -> Path:
    """流式下载（支持代理环境变量），.part 原子改名。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            copied = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                copied += len(chunk)
                if total:
                    pct = copied * 100 // total
                    print(f"\r[build] {dst.name}: {copied / 1e6:.1f}/{total / 1e6:.1f} MB ({pct}%)",
                          end="", flush=True)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise BuildError(f"下载失败 {url}：{exc}") from exc
    print()
    tmp.replace(dst)
    return dst


def run_command(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """执行子进程；失败抛 BuildError（附输出尾部）。"""
    log("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:]
        raise BuildError(f"命令失败（exit={proc.returncode}）：\n{tail}")
    return proc


# ---------------- 源码复制（过滤缓存/测试/文档） ----------------

def _excluded(name: str, is_dir: bool) -> bool:
    if is_dir:
        return name in EXCLUDED_DIR_NAMES
    return name in EXCLUDED_FILE_NAMES or Path(name).suffix in EXCLUDED_FILE_SUFFIXES


def copytree_filtered(src: Path, dst: Path) -> list[Path]:
    """递归复制并过滤；返回写入的文件列表。"""
    if not src.is_dir():
        raise BuildError(f"源目录不存在：{src}")
    written: list[Path] = []
    for item in sorted(src.iterdir()):
        if _excluded(item.name, item.is_dir()):
            continue
        target = dst / item.name
        if item.is_dir():
            written.extend(copytree_filtered(item, target))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            written.append(target)
    return written


def copy_project_sources(repo: Path, app_dir: Path) -> int:
    """白名单复制：app.py、core/、requirements.txt → dist/SubtitleStudio/app/。"""
    app_dir.mkdir(parents=True, exist_ok=True)
    files = copytree_filtered(repo / "core", app_dir / "core")
    for name in ("app.py", "requirements.txt"):
        src = repo / name
        if not src.is_file():
            raise BuildError(f"缺少项目文件：{src}")
        shutil.copy2(src, app_dir / name)
        files.append(app_dir / name)
    return len(files)


# ---------------- Embedded Python ----------------

def embed_download_url(version: str, mirror: str | None) -> str:
    base = mirror or "https://www.python.org/ftp/python"
    return f"{base}/{version}/python-{version}-embed-amd64.zip"


def pth_filename(version: str) -> str:
    """Embedded 发行版的路径配置文件名，如 python312._pth。"""
    return f"python{version.replace('.', '')[:3]}._pth"


def pth_content(version: str) -> str:
    """._pth 内容：启用 site（支持 site-packages 与 .pth 机制）并注册 app/ 源码目录。"""
    tag = version.replace(".", "")[:3]
    lines = [
        f"python{tag}.zip",
        ".",
        "Lib\\site-packages",
        "..\\app",          # 便携根下的源码目录
        "",
        "# Uncomment to run site.main() automatically",
        "import site",
        "",
    ]
    return "\n".join(lines)


def setup_embedded_python(runtime: Path, version: str, mirror: str | None,
                          cache_dir: Path) -> None:
    """下载/解压 Embedded Python，并启用 site-packages。"""
    if (runtime / "python.exe").is_file():
        log(f"runtime/ 已存在 Python（跳过下载）")
    else:
        url = embed_download_url(version, mirror)
        log(f"下载 Embedded Python {version}：{url}")
        zip_path = cache_dir / f"python-{version}-embed-amd64.zip"
        if not zip_path.is_file():
            download_file(url, zip_path)
        runtime.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(runtime)
        log("解压完成")

    pth = runtime / pth_filename(version)
    if not pth.is_file():
        raise BuildError(f"Embedded Python 中未找到 {pth.name}（版本号是否正确？）")
    pth.write_text(pth_content(version), encoding="ascii")
    log(f"已写入 {pth.name}（启用 site-packages 与 app/ 搜索路径）")


def install_pip(runtime: Path, cache_dir: Path) -> None:
    """Embedded Python 无 pip：get-pip.py 引导安装。"""
    if (runtime / "Lib" / "site-packages" / "pip").is_dir():
        log("pip 已存在（跳过安装）")
        return
    get_pip = cache_dir / "get-pip.py"
    if not get_pip.is_file():
        download_file(GET_PIP_URL, get_pip)
    run_command([runtime / "python.exe", get_pip, "--no-warn-script-location"])


def install_dependencies(runtime: Path, requirements: Path,
                          torch_index: str | None, pip_mirror: str | None) -> None:
    """安装 requirements；GPU 模式先从 torch 官方源装 torch/torchaudio。"""
    py = runtime / "python.exe"
    base = [py, "-m", "pip", "install", "--no-cache-dir", "--no-warn-script-location"]
    if pip_mirror:
        base += ["-i", pip_mirror]
    if torch_index:
        # cuXXX wheel 版本号带本地段（如 2.5.1+cu124），先装即可满足 requirements 约束，
        # 第二步 pip 不会重复下载 PyPI 的 CPU 版
        log(f"安装 GPU 版 PyTorch（{torch_index}）…")
        run_command(base + ["torch", "torchaudio", "--index-url", torch_index])
    log("安装项目依赖（requirements.txt）…")
    run_command(base + ["-r", requirements])


def slim_runtime(runtime: Path) -> None:
    """删除构建工具链与缓存：pip/setuptools/wheel/Scripts/__pycache__。"""
    removed = []
    sp = runtime / "Lib" / "site-packages"
    for pkg in ("pip", "setuptools", "wheel", "pkg_resources", "_distutils_hack"):
        d = sp / pkg
        if d.is_dir():
            shutil.rmtree(d)
            removed.append(pkg)
    scripts = runtime / "Scripts"
    if scripts.is_dir():
        shutil.rmtree(scripts)
        removed.append("Scripts")
    for cache in runtime.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    log(f"瘦身完成，移除：{', '.join(removed) or '（无）'}")


# ---------------- FFmpeg ----------------

def extract_ffmpeg_binaries(zip_path: Path, dst_bin: Path) -> list[Path]:
    """从 FFmpeg 发行 zip 中提取 bin/ffmpeg.exe 与 bin/ffprobe.exe。"""
    dst_bin.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="ffmpeg_") as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            for exe in ("ffmpeg.exe", "ffprobe.exe"):
                matches = [n for n in names
                           if n.replace("\\", "/").endswith(f"/bin/{exe}")
                           or n.replace("\\", "/") == exe]
                if not matches:
                    raise BuildError(f"FFmpeg 压缩包中未找到 {exe}")
                zf.extract(matches[0], tmp)
                src = Path(tmp) / matches[0]
                dst = dst_bin / exe
                shutil.copy2(src, dst)
                extracted.append(dst)
    return extracted


def fetch_ffmpeg(bin_dir: Path, source: str, cache_dir: Path) -> list[Path]:
    """获取 ffmpeg.exe / ffprobe.exe：本地目录直接复制，否则下载官方静态构建。"""
    if source in ("skip", "none"):
        log("跳过 FFmpeg（用户要求）")
        return []
    if source.startswith("dir:"):
        local = Path(source[4:])
        found = []
        for exe in ("ffmpeg.exe", "ffprobe.exe"):
            p = local / exe
            if not p.is_file():
                raise BuildError(f"{p} 不存在（--ffmpeg-dir 需同时包含两个 exe）")
            bin_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, bin_dir / exe)
            found.append(bin_dir / exe)
        log(f"已从本地目录复制 FFmpeg：{local}")
        return found
    if source not in FFMPEG_ZIP_SOURCES:
        raise BuildError(f"未知 FFmpeg 源：{source}（可选 {'/'.join(FFMPEG_ZIP_SOURCES)}/dir:<路径>/skip）")
    url = FFMPEG_ZIP_SOURCES[source]
    zip_path = cache_dir / f"ffmpeg-{source}.zip"
    if not zip_path.is_file():
        log(f"下载 FFmpeg：{url}")
        download_file(url, zip_path, timeout=120.0)
    out = extract_ffmpeg_binaries(zip_path, bin_dir)
    log(f"FFmpeg 就绪：{', '.join(p.name for p in out)}")
    return out


# ---------------- 模型预下载 ----------------

def prefetch_models(dist: Path, preset: str) -> None:
    """调用 models/download_models.py 用便携 runtime 预热模型缓存。"""
    script = dist / "models" / "download_models.py"
    if not script.is_file():
        raise BuildError(f"缺少模型下载脚本：{script}")
    env = {
        **__import__("os").environ,
        "HF_HOME": str(dist / "models" / "hf"),
        "TORCH_HOME": str(dist / "models" / "torch"),
        "MODELSCOPE_CACHE": str(dist / "models" / "modelscope"),
    }
    run_command([dist / "runtime" / "python.exe", script, "--preset", preset], env=env)


# ---------------- Inno Setup ----------------

def compile_installer(version: str, dist_root: Path) -> Path:
    """调用 Inno Setup 编译 Setup.exe（ISCC 不在 PATH 时给出指引）。"""
    iscc = shutil.which("iscc") or shutil.which("ISCC")
    if iscc is None:
        for candidate in (
            r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
            r"C:\Program Files\Inno Setup 6\ISCC.exe",
        ):
            if Path(candidate).is_file():
                iscc = candidate
                break
    if iscc is None:
        raise BuildError(
            "未找到 Inno Setup 编译器（ISCC.exe）。请安装 Inno Setup 6.x 后重试，"
            "或手动执行：iscc /DMyAppVersion=" + version + " packaging\\installer.iss"
        )
    run_command([iscc, f"/DMyAppVersion={version}", str(PACKAGING / "installer.iss")])
    setup = dist_root.parent / f"SubtitleStudio_Setup_v{version}.exe"
    if not setup.is_file():
        raise BuildError(f"安装包未生成：{setup}")
    log(f"安装包已生成：{setup}")
    return setup


# ---------------- 组装 ----------------

def build_version_manifest(dist: Path, version: str) -> Path:
    path = dist / "version.txt"
    path.write_text(
        f"Subtitle Studio v{version}\n"
        f"built_at={_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"python={sys.version.split()[0]}\n",
        encoding="utf-8",
    )
    return path


def assemble(args: argparse.Namespace) -> Path:
    dist = Path(args.dist).resolve()
    cache_dir = Path(args.cache).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.clean and dist.exists():
        log(f"清理旧构建：{dist}")
        shutil.rmtree(dist)
    dist.mkdir(parents=True, exist_ok=True)
    for d in ("models", "profiles", "outputs", "logs", "bin"):
        (dist / d).mkdir(parents=True, exist_ok=True)

    # 1. 源码（过滤 __pycache__ / tests / docs / .git / 日志）
    n = copy_project_sources(REPO, dist / "app")
    log(f"源码复制完成：{n} 个文件")

    # 2. 启动器与批处理
    for name in ("launcher.py", "run.bat", "run-debug.bat", "stop.bat"):
        shutil.copy2(PACKAGING / name, dist / name)
    if (PACKAGING / "app.ico").is_file():
        shutil.copy2(PACKAGING / "app.ico", dist / "app.ico")

    # 3. Embedded Python + 依赖
    if not args.skip_deps:
        runtime = dist / "runtime"
        setup_embedded_python(runtime, args.python_version, args.python_mirror, cache_dir)
        install_pip(runtime, cache_dir)
        install_dependencies(runtime, dist / "app" / "requirements.txt",
                             args.torch_index, args.pip_mirror)
        if args.slim:
            slim_runtime(runtime)
    else:
        log("跳过依赖安装（--skip-deps，假定 runtime/ 已就绪）")

    # 4. FFmpeg 静态二进制
    fetch_ffmpeg(dist / "bin", args.ffmpeg, cache_dir)

    # 5. 模型预下载（可选：制作纯离线完整版）
    if args.with_models:
        prefetch_models(dist, args.preset)

    # 6. 版本清单
    build_version_manifest(dist, args.version)
    log(f"便携目录构建完成：{dist}")
    return dist


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build_portable",
        description="构建 Subtitle Studio 零依赖便携目录（需在 Windows 开发机运行）")
    p.add_argument("--dist", default=str(DEFAULT_DIST), help="便携目录输出路径")
    p.add_argument("--cache", default=str(REPO / "dist" / "_cache"),
                   help="下载缓存目录（zip/get-pip 等中间产物）")
    p.add_argument("--version", default=_read_version(), help="版本号（默认读 packaging/version.txt）")
    p.add_argument("--clean", action="store_true", help="构建前清空 dist 目录")
    p.add_argument("--python-version", default=DEFAULT_PYTHON_VERSION,
                   help=f"Embedded Python 版本（默认 {DEFAULT_PYTHON_VERSION}）")
    p.add_argument("--python-mirror", default=None,
                   help="Python 下载镜像（默认官方；国内可用华为云镜像）")
    p.add_argument("--pip-mirror", default=None, help="PyPI 镜像（如清华源）")
    p.add_argument("--torch-index", default=None, metavar="CU124",
                   help="GPU 版 PyTorch 的 wheel 源后缀（如 cu124 = "
                        "https://download.pytorch.org/whl/cu124；缺省 CPU 版）")
    p.add_argument("--ffmpeg", default="btbn",
                   help=f"FFmpeg 来源：{'/'.join(FFMPEG_ZIP_SOURCES)} | dir:<目录> | skip")
    p.add_argument("--skip-deps", action="store_true",
                   help="跳过 Python runtime 与依赖安装（增量组装时使用）")
    p.add_argument("--slim", action="store_true",
                   help="删除 pip/setuptools 等构建工具链，减小体积（删后无法再 pip install）")
    p.add_argument("--with-models", action="store_true",
                   help="构建后预下载模型（制作纯离线完整版）")
    p.add_argument("--preset", default="basic", choices=["basic", "full"],
                   help="预下载模型规格（配合 --with-models）")
    p.add_argument("--installer", action="store_true",
                   help="构建完成后调用 Inno Setup 编译 Setup.exe")
    return p.parse_args(argv)


def _read_version() -> str:
    vf = PACKAGING / "version.txt"
    if vf.is_file():
        return vf.read_text(encoding="utf-8").strip() or "0.0.0"
    return "0.0.0"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dist = assemble(args)
        if args.installer:
            compile_installer(args.version, dist)
    except BuildError as exc:
        print(f"\n[build] 构建失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
