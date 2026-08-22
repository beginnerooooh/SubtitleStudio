"""core/env.py 单测：CUDA/FFmpeg 探测与量化策略（torch 以假模块注入）。"""
import sys

import pytest

import core.env as env_mod


class _FakeCudaProperties:
    def __init__(self, total_memory: int):
        self.total_memory = total_memory


class _FakeCuda:
    def __init__(self, available: bool, vram_bytes: int | None):
        self._available = available
        self._vram_bytes = vram_bytes

    def is_available(self) -> bool:
        return self._available

    def get_device_properties(self, idx: int) -> _FakeCudaProperties:
        if self._vram_bytes is None:
            raise RuntimeError("no device")
        return _FakeCudaProperties(self._vram_bytes)


class _FakeTorch:
    def __init__(self, cuda: _FakeCuda):
        self.cuda = cuda


@pytest.fixture(autouse=True)
def _reset_cache():
    env_mod.reset_cache()
    yield
    env_mod.reset_cache()


@pytest.fixture
def no_torch(monkeypatch):
    """让 `import torch` 抛 ImportError。"""
    monkeypatch.setitem(sys.modules, "torch", None)


@pytest.fixture
def install_fake_torch(monkeypatch):
    def _install(available: bool, vram_gb: float | None) -> None:
        vram_bytes = None if vram_gb is None else int(vram_gb * 1024**3)
        monkeypatch.setitem(sys.modules, "torch", _FakeTorch(_FakeCuda(available, vram_bytes)))

    return _install


class TestQuantizationPolicy:
    def test_cpu_uses_int8(self, no_torch):
        info = env_mod.detect_env(force=True)
        assert info.device == "cpu"
        assert info.compute_type == "int8"
        assert info.vram_gb is None

    def test_gpu_high_vram_uses_float16(self, install_fake_torch):
        install_fake_torch(True, 12.0)
        info = env_mod.detect_env(force=True)
        assert info.device == "cuda"
        assert info.compute_type == "float16"
        assert info.vram_gb == pytest.approx(12.0)

    def test_gpu_low_vram_uses_int8_float16(self, install_fake_torch):
        install_fake_torch(True, 4.0)
        info = env_mod.detect_env(force=True)
        assert info.compute_type == "int8_float16"

    def test_gpu_unknown_vram_defaults_to_int8_float16(self, install_fake_torch):
        install_fake_torch(True, None)
        info = env_mod.detect_env(force=True)
        assert info.compute_type == "int8_float16"

    def test_boundary_8gb_uses_float16(self, install_fake_torch):
        install_fake_torch(True, 8.0)
        info = env_mod.detect_env(force=True)
        assert info.compute_type == "float16"


class TestFfmpegDetection:
    def test_found_via_which(self, monkeypatch):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        info = env_mod.detect_env(force=True)
        assert info.ffmpeg_path == "/usr/bin/ffmpeg"
        assert info.ffprobe_path == "/usr/bin/ffprobe"
        assert info.ffmpeg_available is True

    def test_not_found(self, monkeypatch):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: None)
        info = env_mod.detect_env(force=True)
        assert info.ffmpeg_path is None
        assert info.ffprobe_path is None
        assert info.ffmpeg_available is False


class TestCaching:
    def test_detect_env_caches_result(self, monkeypatch):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/x")
        first = env_mod.detect_env()
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: None)
        assert env_mod.detect_env() is first  # 命中缓存，未重新探测

    def test_force_bypasses_cache(self, monkeypatch):
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: "/usr/bin/x")
        env_mod.detect_env(force=True)
        monkeypatch.setattr(env_mod.shutil, "which", lambda name: None)
        assert env_mod.detect_env(force=True).ffmpeg_path is None


class TestLongRunWarning:
    def test_warns_cpu_long_audio_with_separation(self, no_torch):
        info = env_mod.detect_env(force=True)
        msg = env_mod.long_run_warning(info, duration_sec=31 * 60, enable_separation=True)
        assert msg is not None
        assert "CPU" in msg

    def test_no_warning_for_short_audio(self, no_torch):
        info = env_mod.detect_env(force=True)
        assert env_mod.long_run_warning(info, 29 * 60, True) is None

    def test_no_warning_without_separation(self, no_torch):
        info = env_mod.detect_env(force=True)
        assert env_mod.long_run_warning(info, 31 * 60, False) is None

    def test_no_warning_on_gpu(self, install_fake_torch):
        install_fake_torch(True, 12.0)
        info = env_mod.detect_env(force=True)
        assert env_mod.long_run_warning(info, 31 * 60, True) is None
