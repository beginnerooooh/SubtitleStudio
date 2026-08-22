"""core/audio.py 单测：输入校验、ffprobe 时长、ffmpeg 转码（mock + 真实冒烟）。"""
import shutil
import subprocess
from pathlib import Path

import pytest

from core.audio import AudioProcessError, extract_wav, probe_duration, validate_input


def make_file(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x00" * 16)
    return p


class TestValidateInput:
    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(AudioProcessError, match="不存在"):
            validate_input(tmp_path / "nope.mp4")

    def test_rejects_unsupported_extension(self, tmp_path):
        with pytest.raises(AudioProcessError, match="不支持"):
            validate_input(make_file(tmp_path, "evil.txt"))

    @pytest.mark.parametrize(
        "name",
        ["a.mp4", "b.mkv", "c.mov", "d.webm", "e.mp3", "f.wav", "g.flac", "h.m4a", "i.aac", "j.ogg"],
    )
    def test_accepts_supported_extensions(self, tmp_path, name):
        assert validate_input(make_file(tmp_path, name)) is None

    def test_extension_case_insensitive(self, tmp_path):
        assert validate_input(make_file(tmp_path, "video.MP4")) is None


class TestProbeDuration:
    def test_parses_duration(self, tmp_path, monkeypatch):
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="12.5\n", stderr="")

        monkeypatch.setattr("core.audio.subprocess.run", fake_run)
        assert probe_duration(tmp_path / "a.mp4") == 12.5
        assert "-show_entries" in seen["cmd"]

    def test_error_on_bad_exit_carries_stderr(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "core.audio.subprocess.run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom"),
        )
        with pytest.raises(AudioProcessError, match="boom"):
            probe_duration(tmp_path / "a.mp4")


class TestExtractWav:
    def test_command_args_and_output_path(self, tmp_path, monkeypatch):
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"RIFF-fake")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("core.audio.subprocess.run", fake_run)
        src = make_file(tmp_path, "in.mp4")
        out = extract_wav(src, tmp_path / "out")
        assert out == tmp_path / "out" / "in.wav"
        cmd = seen["cmd"]
        assert cmd[cmd.index("-ar") + 1] == "16000"
        assert cmd[cmd.index("-ac") + 1] == "1"
        assert "s16" in cmd
        assert str(src) in cmd

    def test_nonzero_exit_raises_with_stderr_tail(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "core.audio.subprocess.run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="E" * 800),
        )
        with pytest.raises(AudioProcessError) as excinfo:
            extract_wav(make_file(tmp_path, "in.mp4"), tmp_path / "out")
        msg = str(excinfo.value)
        assert "E" * 500 in msg  # 保留尾部 500 字符
        assert "E" * 600 not in msg  # 不含更长的头部部分

    def test_timeout_raises(self, tmp_path, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 60)

        monkeypatch.setattr("core.audio.subprocess.run", fake_run)
        with pytest.raises(AudioProcessError, match="超时"):
            extract_wav(make_file(tmp_path, "in.mp4"), tmp_path / "out")

    def test_missing_output_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "core.audio.subprocess.run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        )
        with pytest.raises(AudioProcessError, match="未产出"):
            extract_wav(make_file(tmp_path, "in.mp4"), tmp_path / "out")

    def test_rejects_bad_input_before_running(self, tmp_path, monkeypatch):
        def fake_run(cmd, **kwargs):  # 不应被调用
            raise AssertionError("subprocess.run should not be called")

        monkeypatch.setattr("core.audio.subprocess.run", fake_run)
        with pytest.raises(AudioProcessError):
            extract_wav(make_file(tmp_path, "bad.txt"), tmp_path / "out")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="需要系统安装 ffmpeg")
class TestRealFfmpegSmoke:
    def test_transcode_to_16k_mono_16bit(self, tmp_path):
        src = tmp_path / "in.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-ac", "2", "-ar", "44100", str(src),
            ],
            check=True, capture_output=True,
        )
        out = extract_wav(src, tmp_path / "out")
        assert out.is_file()
        header = out.read_bytes()[:44]
        assert header[22:24] == (1).to_bytes(2, "little")       # 单声道
        assert header[24:28] == (16000).to_bytes(4, "little")   # 16kHz
        assert header[34:36] == (16).to_bytes(2, "little")      # 16bit
        assert probe_duration(src) == pytest.approx(1.0, abs=0.05)
