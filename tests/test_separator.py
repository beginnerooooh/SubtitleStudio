"""core/separator.py 单测：成功路径、失败/缺依赖回退、取消传播、资源释放（demucs 用假模块注入）。"""
import sys
import threading
import types
from pathlib import Path

import pytest
import torch

import core.separator as sep_mod
from core.errors import TaskCancelled
from core.separator import VocalSeparator


@pytest.fixture
def fake_demucs(monkeypatch):
    """注入假 demucs.api.Separator；fail_error 控制分离时抛什么异常。"""
    state = {"fail_error": None, "constructed": []}

    class FakeSeparator:
        """按 demucs 4.x api.Separator 真实接口 mock（dict 回调 / (原波形, 声部字典)）。"""

        samplerate = 16000  # 真实模型为 44100；测试用 16k 免 resample

        def __init__(self, model="htdemucs", device="cpu", progress=None,
                     callback=None, **kwargs):
            self.callback = callback
            # demucs 4.x 无 two_stems 等参数；意外 kwarg 记录下来供断言
            state["constructed"].append(
                {"model": model, "device": device, "unexpected": sorted(kwargs)})

        def separate_audio_file(self, path):
            if state["fail_error"] is not None:
                raise state["fail_error"]
            if self.callback is not None:
                # demucs 4.x 回调：进度字典（帧数为模型采样率下计数）
                self.callback({"state": "end", "segment_offset": 441000,
                               "audio_length": 882000})
                self.callback({"state": "end", "segment_offset": 882000,
                               "audio_length": 882000})
            # 返回 (原始波形, {声部名: 张量})
            return (torch.zeros(1, 16000),
                    {"vocals": torch.zeros(1, 16000),
                     "no_vocals": torch.zeros(1, 16000)})

    demucs = types.ModuleType("demucs")
    api = types.ModuleType("demucs.api")
    api.Separator = FakeSeparator
    demucs.api = api
    monkeypatch.setitem(sys.modules, "demucs", demucs)
    monkeypatch.setitem(sys.modules, "demucs.api", api)
    return state


@pytest.fixture
def release_spies(monkeypatch):
    """替换资源释放辅助函数，统计调用次数。"""
    calls = {"gc": 0, "cuda": 0}
    monkeypatch.setattr(sep_mod, "_gc_collect", lambda: calls.__setitem__("gc", calls["gc"] + 1))
    monkeypatch.setattr(sep_mod, "_empty_cuda_cache", lambda: calls.__setitem__("cuda", calls["cuda"] + 1))
    return calls


class TestSeparateSuccess:
    def test_writes_vocals_wav_and_reports_progress(self, fake_demucs, tmp_path):
        ratios, warnings = [], []
        s = VocalSeparator(device="cuda", on_progress=ratios.append, on_warning=warnings.append)
        out = s.separate(str(tmp_path / "in.wav"), str(tmp_path))
        assert out.endswith("vocals.wav")
        assert Path(out).exists()
        assert warnings == []
        assert ratios == [pytest.approx(0.5), pytest.approx(1.0)]

    def test_passes_model_and_device_no_legacy_kwargs(self, fake_demucs, tmp_path):
        s = VocalSeparator(model_name="htdemucs", device="cuda")
        s.separate(str(tmp_path / "in.wav"), str(tmp_path))
        # demucs 4.x Separator 无 two_stems 等参数；出现即说明误用了旧接口
        assert fake_demucs["constructed"] == [
            {"model": "htdemucs", "device": "cuda", "unexpected": []}
        ]


class TestFallback:
    def test_oom_falls_back_to_original_audio(self, fake_demucs, tmp_path):
        fake_demucs["fail_error"] = RuntimeError("CUDA out of memory")
        warnings = []
        s = VocalSeparator(device="cuda", on_warning=warnings.append)
        src = str(tmp_path / "in.wav")
        assert s.separate(src, str(tmp_path)) == src
        assert len(warnings) == 1
        assert "回退" in warnings[0]

    def test_missing_dependency_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "demucs", None)  # 强制 import demucs 抛 ImportError
        warnings = []
        s = VocalSeparator(on_warning=warnings.append)
        src = str(tmp_path / "in.wav")
        assert s.separate(src, str(tmp_path)) == src
        assert len(warnings) == 1

    def test_non_oom_error_also_falls_back(self, fake_demucs, tmp_path):
        fake_demucs["fail_error"] = ValueError("corrupt audio")
        s = VocalSeparator()
        src = str(tmp_path / "in.wav")
        assert s.separate(src, str(tmp_path)) == src


class TestCancel:
    def test_cancel_before_start(self, fake_demucs, tmp_path):
        event = threading.Event()
        event.set()
        s = VocalSeparator(cancel_event=event)
        with pytest.raises(TaskCancelled):
            s.separate(str(tmp_path / "in.wav"), str(tmp_path))

    def test_cancel_mid_run_not_swallowed_by_fallback(self, fake_demucs, tmp_path):
        event = threading.Event()

        def on_progress(ratio):
            if ratio >= 0.5:
                event.set()

        s = VocalSeparator(on_progress=on_progress, cancel_event=event)
        with pytest.raises(TaskCancelled):
            s.separate(str(tmp_path / "in.wav"), str(tmp_path))


class TestRelease:
    def test_release_on_success(self, fake_demucs, release_spies, tmp_path):
        s = VocalSeparator()
        s.separate(str(tmp_path / "in.wav"), str(tmp_path))
        assert release_spies["gc"] >= 1
        assert release_spies["cuda"] >= 1
        assert s._separator is None

    def test_release_on_failure(self, fake_demucs, release_spies, tmp_path):
        fake_demucs["fail_error"] = RuntimeError("CUDA out of memory")
        s = VocalSeparator()
        s.separate(str(tmp_path / "in.wav"), str(tmp_path))
        assert release_spies["gc"] >= 1
        assert release_spies["cuda"] >= 1

    def test_release_on_cancel(self, fake_demucs, release_spies, tmp_path):
        event = threading.Event()
        event.set()
        s = VocalSeparator(cancel_event=event)
        with pytest.raises(TaskCancelled):
            s.separate(str(tmp_path / "in.wav"), str(tmp_path))
        assert release_spies["gc"] >= 1
