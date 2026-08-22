"""packaging/launcher.py 单测：环境预热、端口探测、健康检查、停止信号（纯逻辑）。"""
import http.server
import socket
import threading

import pytest

from launcher import (
    DEFAULT_PORT,
    app_root,
    ensure_dirs,
    find_free_port,
    hide_console,
    port_in_use,
    prepare_env,
    server_url,
    stop_flag_path,
    stop_requested,
    wait_for_server,
)


class TestPrepareEnv:
    def test_bin_dir_prepends_path(self, tmp_path):
        env = prepare_env(tmp_path)
        assert env["PATH"].startswith(str(tmp_path / "bin"))
        assert str(tmp_path / "bin") in env["PATH"].split(":")[0] or \
            str(tmp_path / "bin") in env["PATH"]

    def test_cache_dirs_localized_under_models(self, tmp_path):
        env = prepare_env(tmp_path)
        assert env["HF_HOME"] == str(tmp_path / "models" / "hf")
        assert env["TORCH_HOME"] == str(tmp_path / "models" / "torch")
        assert env["MODELSCOPE_CACHE"] == str(tmp_path / "models" / "modelscope")
        # 全部落在安装目录内，不写 C 盘用户缓存
        for key in ("HF_HOME", "TORCH_HOME", "MODELSCOPE_CACHE"):
            assert str(tmp_path) in env[key]

    def test_telemetry_disabled_and_utf8(self, tmp_path):
        env = prepare_env(tmp_path)
        assert env["GRADIO_ANALYTICS_ENABLED"] == "False"
        assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"
        assert env["PYTHONIOENCODING"] == "utf-8"
        assert env["PYTHONUTF8"] == "1"

    def test_does_not_mutate_process_env(self, tmp_path):
        before = dict(__import__("os").environ)
        prepare_env(tmp_path)
        assert dict(__import__("os").environ) == before


class TestPorts:
    def test_find_free_port_returns_free_port(self):
        port = find_free_port()
        # 返回的端口应可被立即绑定（除极小竞态外）
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))

    def test_find_free_port_skips_occupied(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            occupied = s.getsockname()[1]
            s.listen(1)
            assert find_free_port(start=occupied) != occupied

    def test_port_in_use(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.listen(1)
            assert port_in_use("127.0.0.1", port)
        assert not port_in_use("127.0.0.1", port)  # 关闭后释放

    def test_server_url(self):
        assert server_url(7860) == "http://127.0.0.1:7860/"


class TestWaitForServer:
    @pytest.fixture
    def http_server(self, tmp_path):
        handler = http.server.SimpleHTTPRequestHandler

        def quiet_log(*args):
            pass

        handler.log_message = quiet_log
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        original_dir = __import__("os").getcwd()
        __import__("os").chdir(tmp_path)  # GET / 返回目录列表（200）
        yield srv.server_address[1]
        __import__("os").chdir(original_dir)
        srv.shutdown()
        srv.server_close()

    def test_returns_true_when_serving(self, http_server):
        assert wait_for_server(http_server, timeout=5.0) is True

    def test_returns_false_on_timeout(self):
        # 找一个必然无人监听的端口：绑住再释放通常安全，直接扫描高位
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            dead_port = s.getsockname()[1]
        assert wait_for_server(dead_port, timeout=0.4, interval=0.1) is False


class TestStopSignal:
    def test_stop_flag_path(self, tmp_path):
        assert stop_flag_path(tmp_path) == tmp_path / "stop.flag"

    def test_not_requested_by_default(self, tmp_path):
        assert stop_requested(tmp_path) is False

    def test_requested_when_flag_exists(self, tmp_path):
        stop_flag_path(tmp_path).write_text("", encoding="utf-8")
        assert stop_requested(tmp_path) is True


class TestDirsAndConsole:
    def test_ensure_dirs_creates_missing(self, tmp_path):
        created = ensure_dirs(tmp_path)
        for rel in ("models/hf", "models/torch", "models/modelscope",
                    "profiles", "outputs", "logs", "bin"):
            assert (tmp_path / rel).is_dir()
        assert all(c.is_dir() for c in created)

    def test_ensure_dirs_idempotent(self, tmp_path):
        ensure_dirs(tmp_path)
        assert ensure_dirs(tmp_path) == []

    def test_hide_console_noop_on_non_windows(self):
        hide_console()  # Linux 沙箱：不应抛异常

    def test_app_root_contains_launcher(self):
        assert (app_root() / "launcher.py").is_file()
