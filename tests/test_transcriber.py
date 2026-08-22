"""core/transcriber.py 单测：降档链、进度回调、取消、模型缓存（faster_whisper 用假模块注入）。"""
import sys
import threading
import types

import pytest

import core.transcriber as tr_mod
from core.errors import TaskCancelled
from core.transcriber import Transcriber, TranscriptionError


class _FakeWord:
    def __init__(self, word: str, start: float, end: float):
        self.word = word
        self.start = start
        self.end = end


class _FakeSegment:
    def __init__(self, words, end: float):
        self.words = words
        self.end = end


@pytest.fixture(autouse=True)
def _reset_model_cache():
    tr_mod.reset_model_cache()
    yield
    tr_mod.reset_model_cache()


@pytest.fixture
def fake_fw(monkeypatch):
    """注入假 faster_whisper；通过 oom_configs 控制哪些 (device, compute) 触发 OOM。"""
    state = {"constructed": [], "oom_configs": set(), "error_configs": {}}

    class FakeModel:
        def __init__(self, model_size, device="cpu", compute_type="int8", **kwargs):
            self.model_size = model_size
            self.device = device
            self.compute_type = compute_type
            state["constructed"].append((model_size, device, compute_type))

        def transcribe(self, path, **kwargs):
            key = (self.device, self.compute_type)
            if key in state["error_configs"]:
                raise state["error_configs"][key]
            if key in state["oom_configs"]:
                raise RuntimeError("CUDA out of memory")
            segments = [
                _FakeSegment([_FakeWord(" 你", 0.0, 0.4), _FakeWord("好。", 0.4, 0.9)], 0.9),
                _FakeSegment([_FakeWord("世界", 1.2, 2.0)], 2.0),
            ]
            return iter(segments), object()

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return state


class TestTranscribe:
    def test_words_mapped_and_aggregated(self, fake_fw):
        t = Transcriber(model_size="small", device="cpu", compute_type="int8")
        lines = t.transcribe("a.wav")
        assert [ln.text for ln in lines] == ["你好。", "世界"]
        assert lines[0].start == 0.0
        assert lines[0].end == 0.9

    def test_language_auto_passes_none(self, fake_fw, monkeypatch):
        seen = {}
        orig_get = tr_mod._get_model

        def spy_get(size, device, compute_type):
            model = orig_get(size, device, compute_type)
            real_transcribe = model.transcribe

            def spy(path, **kwargs):
                seen.update(kwargs)
                return real_transcribe(path, **kwargs)

            model.transcribe = spy
            return model

        monkeypatch.setattr(tr_mod, "_get_model", spy_get)
        t = Transcriber(model_size="small", device="cpu", compute_type="int8", language="auto")
        t.transcribe("a.wav")
        assert seen["language"] is None
        assert seen["vad_filter"] is True
        assert seen["word_timestamps"] is True
        assert seen["beam_size"] == 5

    def test_progress_reported_with_total_duration(self, fake_fw):
        ratios = []
        t = Transcriber(
            model_size="small", device="cpu", compute_type="int8",
            on_progress=ratios.append,
        )
        t.transcribe("a.wav", total_duration=2.0)
        assert ratios == [pytest.approx(0.45), pytest.approx(1.0)]

    def test_no_progress_without_total_duration(self, fake_fw):
        ratios = []
        t = Transcriber(
            model_size="small", device="cpu", compute_type="int8",
            on_progress=ratios.append,
        )
        t.transcribe("a.wav")
        assert ratios == []

    def test_cancel_raises_task_cancelled(self, fake_fw):
        event = threading.Event()
        event.set()  # 一开始就取消
        t = Transcriber(model_size="small", device="cpu", compute_type="int8", cancel_event=event)
        with pytest.raises(TaskCancelled):
            t.transcribe("a.wav")

    def test_cancel_mid_stream(self, fake_fw):
        event = threading.Event()

        def on_progress(ratio):
            if ratio > 0.4:
                event.set()

        t = Transcriber(
            model_size="small", device="cpu", compute_type="int8",
            on_progress=on_progress, cancel_event=event,
        )
        with pytest.raises(TaskCancelled):
            t.transcribe("a.wav", total_duration=2.0)


class TestOomFallbackChain:
    def test_falls_back_to_next_compute_type(self, fake_fw):
        fake_fw["oom_configs"].add(("cuda", "float16"))
        t = Transcriber(model_size="small", device="cuda", compute_type="float16")
        lines = t.transcribe("a.wav")
        assert [ln.text for ln in lines] == ["你好。", "世界"]
        assert fake_fw["constructed"] == [
            ("small", "cuda", "float16"),
            ("small", "cuda", "int8_float16"),
        ]

    def test_falls_back_to_cpu_when_all_cuda_oom(self, fake_fw):
        fake_fw["oom_configs"].add(("cuda", "int8"))
        t = Transcriber(model_size="small", device="cuda", compute_type="int8")
        lines = t.transcribe("a.wav")
        assert len(lines) == 2
        assert ("small", "cpu", "int8") in fake_fw["constructed"]

    def test_all_levels_oom_raises(self, fake_fw):
        for dev, comp in [("cuda", "float16"), ("cuda", "int8_float16"), ("cuda", "int8"), ("cpu", "int8")]:
            fake_fw["oom_configs"].add((dev, comp))
        t = Transcriber(model_size="small", device="cuda", compute_type="float16")
        with pytest.raises(TranscriptionError, match="显存不足"):
            t.transcribe("a.wav")

    def test_non_oom_error_propagates(self, fake_fw):
        fake_fw["error_configs"][("cpu", "int8")] = ValueError("boom")
        t = Transcriber(model_size="small", device="cpu", compute_type="int8")
        with pytest.raises(ValueError, match="boom"):
            t.transcribe("a.wav")


class TestModelCache:
    def test_model_cached_between_calls(self, fake_fw):
        t = Transcriber(model_size="small", device="cpu", compute_type="int8")
        t.transcribe("a.wav")
        t.transcribe("b.wav")
        assert len(fake_fw["constructed"]) == 1

    def test_oom_evicts_failed_model_from_cache(self, fake_fw):
        fake_fw["oom_configs"].add(("cuda", "float16"))
        t = Transcriber(model_size="small", device="cuda", compute_type="float16")
        t.transcribe("a.wav")
        # 第二次调用：float16 已被逐出缓存，会重新构造（并再次 OOM 降档）
        t.transcribe("b.wav")
        assert fake_fw["constructed"].count(("small", "cuda", "float16")) == 2
