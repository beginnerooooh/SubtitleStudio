"""core/pipeline.py 单测：编排状态机、进度汇总、取消、Demucs 回退、异常映射（全阶段 mock）。"""
import threading

import pytest

import core.pipeline as pl
from core.aligner import AlignmentError, AlignResult
from core.env import EnvInfo
from core.errors import TaskCancelled
from core.models import SubtitleLine, SubtitleWord
from core.pipeline import Pipeline, PipelineConfig, PipelineError
from core.text import prepare_lyrics
from core.transcriber import TranscriptionError


def _env(ffmpeg="/usr/bin/ffmpeg", device="cpu"):
    return EnvInfo(
        device=device, compute_type="int8", vram_gb=None,
        ffmpeg_path=ffmpeg, ffprobe_path="/usr/bin/ffprobe",
    )


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    """注入全部阶段假件；state 控制行为并记录调用。"""
    state = {
        "transcriber_kwargs": None,
        "transcriber_path": None,
        "transcriber_total": None,
        "aligner_kwargs": None,
        "aligner_args": None,
        "separator_kwargs": None,
        "separator_path": None,
        "sep_fail": False,
        "transcribe_error": None,
        "transcribe_cancel": False,
        "align_error": None,
        "env": _env(),
        "duration": 100.0,
    }
    wav = str(tmp_path / "in.wav")
    vocals = str(tmp_path / "vocals.wav")

    class FakeTranscriber:
        def __init__(self, model_size="small", device="cpu", compute_type="int8",
                     language="auto", on_progress=None, cancel_event=None):
            state["transcriber_kwargs"] = dict(
                model_size=model_size, device=device, compute_type=compute_type, language=language,
            )
            self.on_progress = on_progress

        def transcribe(self, path, total_duration=None):
            state["transcriber_path"] = path
            state["transcriber_total"] = total_duration
            if state["transcribe_error"]:
                raise state["transcribe_error"]
            if state["transcribe_cancel"]:
                if self.on_progress:
                    self.on_progress(0.5)
                raise TaskCancelled("用户取消")
            if self.on_progress:
                self.on_progress(0.5)
                self.on_progress(1.0)
            # 两个“段”各一词 → 无标点/未超限 → 应聚合为一行
            return [
                SubtitleLine(words=[SubtitleWord("你好", 0.0, 0.5)]),
                SubtitleLine(words=[SubtitleWord("世界", 0.6, 1.0)]),
            ]

    class FakeAligner:
        def __init__(self, language="zh", device="cpu", on_progress=None,
                     cancel_event=None, confidence_threshold=0.5):
            state["aligner_kwargs"] = dict(language=language, device=device)

        def align(self, vocals_path, lyrics):
            state["aligner_args"] = (vocals_path, lyrics)
            if state["align_error"]:
                raise state["align_error"]
            return AlignResult(
                lines=[SubtitleLine(words=[SubtitleWord("歌词行", 0.0, 2.0)])],
                low_confidence=[0] if state.get("low_conf") else [],
            )

    class FakeSeparator:
        def __init__(self, model_name="htdemucs", device="cpu",
                     on_progress=None, on_warning=None, cancel_event=None):
            state["separator_kwargs"] = dict(model_name=model_name, device=device)
            self.on_progress = on_progress
            self.on_warning = on_warning

        def separate(self, wav_path, output_dir):
            state["separator_path"] = wav_path
            if self.on_progress:
                self.on_progress(0.5)
                self.on_progress(1.0)
            if state["sep_fail"]:
                if self.on_warning:
                    self.on_warning("人声分离失败（OOM），已回退使用原始音频继续任务")
                return wav_path
            return vocals

    monkeypatch.setattr(pl, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(pl, "Aligner", FakeAligner)
    monkeypatch.setattr(pl, "VocalSeparator", FakeSeparator)
    monkeypatch.setattr(pl, "detect_env", lambda force=False: state["env"])
    monkeypatch.setattr(pl, "probe_duration", lambda path, **kw: state["duration"])
    monkeypatch.setattr(pl, "validate_input", lambda path: None)
    monkeypatch.setattr(pl, "extract_wav", lambda src, dst_dir, **kw: wav)
    state["vocals"] = vocals
    state["wav"] = wav
    return state


def _make_input(tmp_path):
    p = tmp_path / "song.mp4"
    p.write_bytes(b"\x00" * 16)
    return str(p)


class TestBlindMode:
    def test_full_flow_aggregates_and_exports(self, fakes, tmp_path):
        ratios, messages, logs = [], [], []
        cfg = PipelineConfig(input_path=_make_input(tmp_path), work_dir=str(tmp_path / "out"))
        p = Pipeline(cfg, on_progress=lambda r, m: (ratios.append(r), messages.append(m)),
                     on_log=logs.append)
        result = p.run()
        assert result.mode == "blind"
        # 盲识别参数来自环境探测
        assert fakes["transcriber_kwargs"] == dict(
            model_size="small", device="cpu", compute_type="int8", language="auto",
        )
        assert fakes["transcriber_path"] == fakes["wav"]
        assert fakes["transcriber_total"] == 100.0
        # 两段聚合为一行（无句末标点、未超时长/字数）
        assert [ln.text for ln in result.lines] == ["你好世界"]
        # 三格式导出且内容正确
        assert set(result.files) == {"srt", "lrc", "ass"}
        for fmt, path in result.files.items():
            assert path.endswith(f"song.{fmt}")
            with open(path, encoding="utf-8") as f:
                content = f.read()
            # ASS 的 \k 标签插在字词间，不保证连续子串
            assert "你好" in content and "世界" in content
        # 进度单调且到 1.0
        assert ratios[-1] == pytest.approx(1.0)
        assert ratios == sorted(ratios)

    def test_formats_subset(self, fakes, tmp_path):
        cfg = PipelineConfig(input_path=_make_input(tmp_path), work_dir=str(tmp_path / "out"),
                             formats=("srt",))
        result = Pipeline(cfg).run()
        assert set(result.files) == {"srt"}

    def test_transcriber_error_mapped(self, fakes, tmp_path):
        fakes["transcribe_error"] = TranscriptionError("模型加载失败")
        cfg = PipelineConfig(input_path=_make_input(tmp_path), work_dir=str(tmp_path / "out"))
        with pytest.raises(PipelineError, match="语音识别失败.*模型加载失败") as excinfo:
            Pipeline(cfg).run()
        assert isinstance(excinfo.value.__cause__, TranscriptionError)

    def test_cancel_propagates_unwrapped(self, fakes, tmp_path):
        fakes["transcribe_cancel"] = True
        cfg = PipelineConfig(input_path=_make_input(tmp_path), work_dir=str(tmp_path / "out"))
        with pytest.raises(TaskCancelled):
            Pipeline(cfg).run()


class TestAlignMode:
    def test_lyrics_use_aligner_no_aggregation(self, fakes, tmp_path):
        cfg = PipelineConfig(
            input_path=_make_input(tmp_path), work_dir=str(tmp_path / "out"),
            lyrics_text="你好\n世界",
        )
        result = Pipeline(cfg).run()
        assert result.mode == "align"
        assert fakes["aligner_kwargs"] == dict(language="zh", device="cpu")
        path, lyrics = fakes["aligner_args"]
        assert path == fakes["wav"]
        assert [len(line) for line in lyrics] == [2, 2]  # prepare_lyrics 行结构
        assert isinstance(lyrics[0][0], type(prepare_lyrics("你")[0][0]))
        # 对齐行不参与聚合
        assert [ln.text for ln in result.lines] == ["歌词行"]

    def test_english_lyrics_choose_en(self, fakes, tmp_path):
        cfg = PipelineConfig(
            input_path=_make_input(tmp_path), work_dir=str(tmp_path / "out"),
            lyrics_text="hello world\nfoo bar",
        )
        Pipeline(cfg).run()
        assert fakes["aligner_kwargs"]["language"] == "en"

    def test_low_confidence_warning(self, fakes, tmp_path):
        fakes["low_conf"] = True
        logs = []
        cfg = PipelineConfig(
            input_path=_make_input(tmp_path), work_dir=str(tmp_path / "out"),
            lyrics_text="你好\n世界",
        )
        result = Pipeline(cfg, on_log=logs.append).run()
        assert any("第 1 行" in w and "人工复核" in w for w in result.warnings)
        assert any("人工复核" in m for m in logs)

    def test_aligner_error_mapped(self, fakes, tmp_path):
        fakes["align_error"] = AlignmentError("歌词与音频不匹配")
        cfg = PipelineConfig(
            input_path=_make_input(tmp_path), work_dir=str(tmp_path / "out"),
            lyrics_text="你好",
        )
        with pytest.raises(PipelineError, match="歌词对齐失败.*不匹配"):
            Pipeline(cfg).run()

    def test_metadata_only_lyrics_fall_back_to_blind(self, fakes, tmp_path):
        logs = []
        cfg = PipelineConfig(
            input_path=_make_input(tmp_path), work_dir=str(tmp_path / "out"),
            lyrics_text="[ti:test]",
        )
        result = Pipeline(cfg, on_log=logs.append).run()
        assert result.mode == "blind"
        assert fakes["transcriber_path"] is not None
        assert any("盲识别" in m for m in logs)


class TestSeparation:
    def test_separation_enabled_uses_vocals(self, fakes, tmp_path):
        ratios = []
        cfg = PipelineConfig(
            input_path=_make_input(tmp_path), work_dir=str(tmp_path / "out"),
            enable_separation=True,
        )
        result = Pipeline(cfg, on_progress=lambda r, m: ratios.append(r)).run()
        assert fakes["separator_kwargs"] == dict(model_name="htdemucs", device="cpu")
        assert fakes["separator_path"] == fakes["wav"]
        # 识别收到的是人声轨
        assert fakes["transcriber_path"] == fakes["vocals"]
        # 阶段权重：分离 0.05+0.25*0.5=0.175；识别 0.30+0.60*0.5=0.60
        assert 0.175 in [pytest.approx(r) for r in ratios]
        assert 0.60 in [pytest.approx(r) for r in ratios]
        assert result.mode == "blind"

    def test_separation_fallback_continues(self, fakes, tmp_path):
        fakes["sep_fail"] = True
        logs = []
        cfg = PipelineConfig(
            input_path=_make_input(tmp_path), work_dir=str(tmp_path / "out"),
            enable_separation=True,
        )
        result = Pipeline(cfg, on_log=logs.append).run()
        # 回退原始音频
        assert fakes["transcriber_path"] == fakes["wav"]
        assert any("回退" in m for m in logs)
        assert any("回退" in w for w in result.warnings)

    def test_long_run_warning_emitted(self, fakes, tmp_path):
        fakes["duration"] = 2000.0  # > 30min
        logs = []
        cfg = PipelineConfig(
            input_path=_make_input(tmp_path), work_dir=str(tmp_path / "out"),
            enable_separation=True,
        )
        result = Pipeline(cfg, on_log=logs.append).run()
        assert any("极长" in w for w in result.warnings)


class TestEnvGuard:
    def test_ffmpeg_missing_raises_readable_error(self, fakes, tmp_path):
        fakes["env"] = _env(ffmpeg=None)
        cfg = PipelineConfig(input_path=_make_input(tmp_path), work_dir=str(tmp_path / "out"))
        with pytest.raises(PipelineError, match="FFmpeg"):
            Pipeline(cfg).run()

    def test_device_override(self, fakes, tmp_path):
        cfg = PipelineConfig(
            input_path=_make_input(tmp_path), work_dir=str(tmp_path / "out"),
            device="cuda", compute_type="float16",
        )
        Pipeline(cfg).run()
        assert fakes["transcriber_kwargs"]["device"] == "cuda"
        assert fakes["transcriber_kwargs"]["compute_type"] == "float16"
