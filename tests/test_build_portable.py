"""packaging/build_portable.py 单测：源码过滤、._pth 生成、FFmpeg 提取、瘦身、下载。"""
import http.server
import threading
import zipfile

import pytest

from build_portable import (
    BuildError,
    copytree_filtered,
    download_file,
    embed_download_url,
    extract_ffmpeg_binaries,
    fetch_ffmpeg,
    pth_content,
    pth_filename,
    slim_runtime,
)


class TestCopyFiltered:
    def _make_tree(self, base):
        (base / "core").mkdir(parents=True)
        (base / "core" / "__init__.py").write_text("", encoding="utf-8")
        (base / "core" / "engine.py").write_text("x = 1", encoding="utf-8")
        (base / "core" / "__pycache__").mkdir()
        (base / "core" / "__pycache__" / "engine.cpython-312.pyc").write_bytes(b"\x00")
        (base / "core" / "engine.py.log").write_text("log", encoding="utf-8")
        (base / ".git").mkdir()
        (base / ".git" / "HEAD").write_text("ref", encoding="utf-8")
        (base / "tests").mkdir()
        (base / "tests" / "test_x.py").write_text("", encoding="utf-8")
        (base / ".gitignore").write_text("", encoding="utf-8")
        (base / ".env").write_text("SECRET=1", encoding="utf-8")

    def test_filters_caches_git_tests_logs(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        self._make_tree(src)
        dst = tmp_path / "dst"
        written = copytree_filtered(src, dst)
        names = {p.relative_to(dst).as_posix() for p in written}
        assert names == {"core/__init__.py", "core/engine.py"}
        assert not (dst / "core" / "__pycache__").exists()
        assert not (dst / ".git").exists()
        assert not (dst / "tests").exists()
        assert not (dst / ".gitignore").exists()
        assert not (dst / ".env").exists()
        assert not (dst / "core" / "engine.py.log").exists()

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(BuildError, match="源目录不存在"):
            copytree_filtered(tmp_path / "nope", tmp_path / "dst")


class TestEmbeddedPythonConfig:
    def test_pth_filename(self):
        assert pth_filename("3.12.8") == "python312._pth"
        assert pth_filename("3.11.4") == "python311._pth"

    def test_pth_content_enables_site_and_app_path(self):
        content = pth_content("3.12.8")
        lines = [ln.strip() for ln in content.splitlines()]
        assert "python312.zip" in lines
        assert "." in lines
        assert "Lib\\site-packages" in lines
        assert "..\\app" in lines          # 便携根下的源码目录
        assert "import site" in lines       # 启用 site-packages 机制

    def test_embed_download_url(self):
        url = embed_download_url("3.12.8", None)
        assert url == "https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip"
        mirror = embed_download_url("3.12.8", "https://mirrors.example.com/python")
        assert mirror.startswith("https://mirrors.example.com/python/3.12.8/")


class TestFfmpegExtraction:
    def _make_zip(self, path):
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("readme.txt", "docs")
            zf.writestr("ffmpeg-2024-win64-gpl/bin/ffmpeg.exe", b"FAKE_FFMPEG")
            zf.writestr("ffmpeg-2024-win64-gpl/bin/ffprobe.exe", b"FAKE_FFPROBE")

    def test_extracts_both_binaries_from_nested_zip(self, tmp_path):
        zip_path = tmp_path / "ffmpeg.zip"
        self._make_zip(zip_path)
        out = extract_ffmpeg_binaries(zip_path, tmp_path / "bin")
        assert [p.name for p in out] == ["ffmpeg.exe", "ffprobe.exe"]
        assert (tmp_path / "bin" / "ffmpeg.exe").read_bytes() == b"FAKE_FFMPEG"
        assert (tmp_path / "bin" / "ffprobe.exe").read_bytes() == b"FAKE_FFPROBE"

    def test_missing_binary_raises(self, tmp_path):
        zip_path = tmp_path / "bad.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("only/bin/ffmpeg.exe", b"X")
        with pytest.raises(BuildError, match="未找到 ffprobe.exe"):
            extract_ffmpeg_binaries(zip_path, tmp_path / "bin")

    def test_fetch_from_local_dir(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        (local / "ffmpeg.exe").write_bytes(b"F")
        (local / "ffprobe.exe").write_bytes(b"P")
        out = fetch_ffmpeg(tmp_path / "bin", f"dir:{local}", tmp_path / "cache")
        assert (tmp_path / "bin" / "ffmpeg.exe").read_bytes() == b"F"
        assert len(out) == 2

    def test_fetch_skip(self, tmp_path):
        assert fetch_ffmpeg(tmp_path / "bin", "skip", tmp_path / "cache") == []

    def test_fetch_unknown_source(self, tmp_path):
        with pytest.raises(BuildError, match="未知 FFmpeg 源"):
            fetch_ffmpeg(tmp_path / "bin", "nowhere", tmp_path / "cache")


class TestSlimRuntime:
    def test_removes_toolchain_keeps_packages(self, tmp_path):
        rt = tmp_path / "runtime"
        (rt / "Scripts").mkdir(parents=True)
        (rt / "Scripts" / "pip.exe").write_bytes(b"")
        sp = rt / "Lib" / "site-packages"
        (sp / "pip").mkdir(parents=True)
        (sp / "setuptools").mkdir()
        (sp / "torch" / "__pycache__").mkdir(parents=True)
        (sp / "torch" / "__init__.py").write_text("", encoding="utf-8")
        (sp / "torch" / "__pycache__" / "init.pyc").write_bytes(b"")

        slim_runtime(rt)

        assert not (rt / "Scripts").exists()
        assert not (sp / "pip").exists()
        assert not (sp / "setuptools").exists()
        assert not (sp / "torch" / "__pycache__").exists()
        assert (sp / "torch" / "__init__.py").is_file()  # 运行库不受影响


class TestDownloadFile:
    @pytest.fixture
    def file_server(self, tmp_path):
        (tmp_path / "blob.bin").write_bytes(b"0123456789" * 100)
        handler = http.server.SimpleHTTPRequestHandler

        def quiet(*args):
            pass

        handler.log_message = quiet
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        original_dir = __import__("os").getcwd()
        __import__("os").chdir(tmp_path)
        yield srv.server_address[1]
        __import__("os").chdir(original_dir)
        srv.shutdown()
        srv.server_close()

    def test_downloads_content(self, file_server, tmp_path):
        dst = tmp_path / "out" / "blob.bin"
        url = f"http://127.0.0.1:{file_server}/blob.bin"
        result = download_file(url, dst)
        assert result == dst
        assert dst.read_bytes() == b"0123456789" * 100
        assert not dst.with_name("blob.bin.part").exists()  # 临时文件已清理

    def test_failed_download_raises_and_cleans(self, file_server, tmp_path):
        dst = tmp_path / "out" / "missing.bin"
        url = f"http://127.0.0.1:{file_server}/missing.bin"
        with pytest.raises(BuildError, match="下载失败"):
            download_file(url, dst)
        assert not dst.exists()
        assert not dst.with_name("missing.bin.part").exists()
