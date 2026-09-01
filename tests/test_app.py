"""app.py 单测：UI 可构建 + 后台 worker 的队列消息协议（pipeline 注入假件）。"""
import queue
import threading

import gradio as gr
import pytest

import app
from core.errors import TaskCancelled
from core.models import SubtitleLine, SubtitleWord
from core.pipeline import PipelineConfig, PipelineError
from core.song_recognizer import SongEntry
from core.speaker import SpeakerAnalysis, SpeakerCluster
from core.voiceprint import VoiceprintError


@pytest.fixture(scope="module")
def blocks():
    return app.build_ui()


class TestUI:
    def test_builds_blocks(self, blocks):
        assert isinstance(blocks, gr.Blocks)

    def test_mode_hint_switches_by_lyrics(self):
        assert "盲识别" in app._mode_hint("")
        assert "盲识别" in app._mode_hint("   ")
        assert "强制对齐" in app._mode_hint("[00:01.00] 你好")

    def test_env_badge_mentions_ffmpeg(self):
        badge = app._env_badge()
        assert "FFmpeg" in badge

    def test_fmt_ts(self):
        assert app._fmt_ts(0.0) == "00:00.00"
        assert app._fmt_ts(83.456) == "01:23.46"

    def test_preview_rows_limit(self):
        from core.pipeline import PipelineResult

        lines = [SubtitleLine(words=[SubtitleWord("字", i, i + 1)]) for i in range(600)]
        result = PipelineResult(mode="blind", duration=600.0, lines=lines,
                                files={}, warnings=[], out_dir="")
        rows = app._preview_rows(result, limit=500)
        assert len(rows) == 500
        assert rows[0] == [1, "00:00.00", "00:01.00", "", "", "字"]

    def test_preview_rows_speaker_and_review_flags(self):
        from core.pipeline import PipelineResult

        low = SubtitleLine(words=[SubtitleWord("低", 0.0, 1.0)],
                           low_confidence=True, low_confidence_reason="识别置信度低")
        ok = SubtitleLine(words=[SubtitleWord("好", 1.0, 2.0)])
        low.speaker = "小明"
        result = PipelineResult(mode="blind", duration=2.0, lines=[low, ok],
                                files={}, warnings=[], out_dir="")
        rows = app._preview_rows(result)
        assert rows[0][3] == "小明" and rows[0][4] == "是"
        assert rows[1][3] == "" and rows[1][4] == ""


class TestSongsPanel:
    def _result(self, songs):
        from core.pipeline import PipelineResult

        return PipelineResult(mode="blind", duration=600.0, lines=[],
                              files={}, warnings=[], out_dir="", songs=songs)

    def test_songs_markdown_formats_entries(self):
        songs = [SongEntry(start=932.0, end=1185.0, title="晴天", artist="周杰伦",
                           confidence=98.0)]
        md = app._songs_markdown(self._result(songs))
        assert "《晴天》- 周杰伦" in md
        assert "00:15:32 - 00:19:45" in md
        assert "置信度: 98%" in md

    def test_songs_markdown_placeholder_when_empty(self):
        assert "未识别到歌曲" in app._songs_markdown(self._result([]))

    def test_songs_markdown_placeholder_when_none(self):
        assert "未识别到歌曲" in app._songs_markdown(None)


class TestProfileManagement:
    def test_save_requires_name(self):
        msg, update = app.save_streamer_profile("", "/tmp/a.wav", None, "自动")
        assert "主播名" in msg

    def test_save_requires_speak_sample(self):
        msg, update = app.save_streamer_profile("小明", None, None, "自动")
        assert "说话样本" in msg

    def test_save_success_refreshes_dropdown(self, monkeypatch, tmp_path):
        saved = {}

        def fake_save(name, speak, sing, profiles_dir="profiles", device="cpu"):
            saved.update(name=name, speak=speak, sing=sing,
                         profiles_dir=profiles_dir, device=device)
            return tmp_path / f"{name}.npy"

        monkeypatch.setattr(app, "save_profile", fake_save)
        monkeypatch.setattr(app, "list_profiles",
                            lambda d: ["小明", "小红"])
        msg, update = app.save_streamer_profile("小明", "/tmp/a.wav", "/tmp/b.wav", "CPU")
        assert "已保存" in msg and "小明" in msg
        assert saved == dict(name="小明", speak="/tmp/a.wav", sing="/tmp/b.wav",
                             profiles_dir=app._PROFILES_DIR, device="cpu")
        assert update["choices"] == ["小明", "小红"] and update["value"] == "小明"

    def test_save_without_sing_sample_hints(self, monkeypatch, tmp_path):
        monkeypatch.setattr(app, "save_profile",
                            lambda n, s, g, profiles_dir="profiles", device="cpu": tmp_path / "x.npy")
        monkeypatch.setattr(app, "list_profiles", lambda d: ["小明"])
        msg, _ = app.save_streamer_profile("小明", "/tmp/a.wav", None, "CPU")
        assert "唱歌样本" in msg

    def test_save_voiceprint_error_readable(self, monkeypatch):
        def boom(*a, **kw):
            raise VoiceprintError("声纹样本几乎为静音")

        monkeypatch.setattr(app, "save_profile", boom)
        msg, _ = app.save_streamer_profile("小明", "/tmp/a.wav", None, "CPU")
        assert "声纹保存失败" in msg and "静音" in msg

    def test_save_unexpected_error_caught(self, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("模型下载失败")

        monkeypatch.setattr(app, "save_profile", boom)
        msg, _ = app.save_streamer_profile("小明", "/tmp/a.wav", None, "CPU")
        assert "未预期的错误" in msg

    def test_auto_device_resolved_from_env(self, monkeypatch):
        calls = {}

        class FakeEnv:
            device = "cuda"

        monkeypatch.setattr(app, "detect_env", lambda: FakeEnv())

        def fake_save(name, speak, sing, profiles_dir="profiles", device="cpu"):
            calls["device"] = device
            return "x.npy"

        monkeypatch.setattr(app, "save_profile", fake_save)
        monkeypatch.setattr(app, "list_profiles", lambda d: [])
        app.save_streamer_profile("小明", "/tmp/a.wav", None, "自动")
        assert calls["device"] == "cuda"

    def test_refresh_updates_choices(self, monkeypatch):
        monkeypatch.setattr(app, "list_profiles", lambda d: ["小明"])
        update = app.refresh_profile_list()
        assert update["choices"] == ["小明"]


class TestStartTaskValidation:
    def _run(self, **kw):
        args = dict(
            file="x.mp4", lyrics="", device="自动", model_size="small",
            enable_separation=False, language="自动", formats=["srt"], title="",
            speaker_mode_label="关闭", profile_name="", voice_threshold=0.55,
            enable_song_detect=False, enable_lyrics_fetch=False,
            speaker_analysis_state=None, selected_speakers=[],
            use_speaker_library=True, mark_low_confidence=True,
            speaker_labels=False,
            progress=lambda *a, **k: None,
        )
        args.update(kw)
        return list(app.start_task(**args))

    def test_missing_file_rejected(self):
        outputs = self._run(file=None)
        assert len(outputs) == 1 and "请先上传" in outputs[0][0]

    def test_single_mode_without_profile_rejected(self):
        outputs = self._run(speaker_mode_label="主声线过滤", profile_name="  ")
        assert len(outputs) == 1
        assert "未选择主播" in outputs[0][0]
        # 任务未启动：未占用任务锁
        assert app._task_lock.acquire(blocking=False)
        app._task_lock.release()

    def test_multi_mode_without_analysis_starts(self, monkeypatch):
        """未预分析的多说话人模式：pipeline 内部自动分析，任务正常启动。"""
        seen = {}

        class FakePipeline:
            def __init__(self, cfg, on_progress=None, on_log=None, cancel_event=None):
                seen["cfg"] = cfg
                seen["cancel"] = cancel_event

            def run(self):
                from core.pipeline import PipelineResult
                return PipelineResult(mode="blind", duration=1.0, lines=[],
                                      files={}, warnings=[], out_dir="")

        monkeypatch.setattr(app, "Pipeline", FakePipeline)
        outputs = self._run(speaker_mode_label="多说话人分离",
                            selected_speakers=["说话人 1"])
        assert outputs[-1][0] == "全部完成。"
        cfg = seen["cfg"]
        assert cfg.speaker_mode == "multi"
        assert cfg.speaker_analysis is None           # 无预分析 → None
        assert cfg.selected_speakers == ["说话人 1"]
        assert cfg.use_speaker_library is True
        assert cfg.mark_low_confidence is True
        assert cfg.speaker_labels is False
        assert cfg.enable_voiceprint is False
        # 任务锁已释放
        assert app._task_lock.acquire(blocking=False)
        app._task_lock.release()

    def test_multi_mode_passes_pre_analysis(self, monkeypatch):
        analysis = SpeakerAnalysis()
        seen = {}

        class FakePipeline:
            def __init__(self, cfg, on_progress=None, on_log=None, cancel_event=None):
                seen["cfg"] = cfg

            def run(self):
                from core.pipeline import PipelineResult
                return PipelineResult(mode="blind", duration=1.0, lines=[],
                                      files={}, warnings=[], out_dir="")

        monkeypatch.setattr(app, "Pipeline", FakePipeline)
        self._run(speaker_mode_label="多说话人分离",
                  speaker_analysis_state=analysis,
                  selected_speakers=["甲"], use_speaker_library=False,
                  mark_low_confidence=False, speaker_labels=True)
        cfg = seen["cfg"]
        assert cfg.speaker_analysis is analysis
        assert cfg.selected_speakers == ["甲"]
        assert cfg.use_speaker_library is False
        assert cfg.mark_low_confidence is False
        assert cfg.speaker_labels is True

    def test_non_multi_mode_drops_pre_analysis(self, monkeypatch):
        analysis = SpeakerAnalysis()
        seen = {}

        class FakePipeline:
            def __init__(self, cfg, on_progress=None, on_log=None, cancel_event=None):
                seen["cfg"] = cfg

            def run(self):
                from core.pipeline import PipelineResult
                return PipelineResult(mode="blind", duration=1.0, lines=[],
                                      files={}, warnings=[], out_dir="")

        monkeypatch.setattr(app, "Pipeline", FakePipeline)
        self._run(speaker_mode_label="关闭", speaker_analysis_state=analysis)
        assert seen["cfg"].speaker_analysis is None


class TestSpeakerPanels:
    """多说话人分离面板：分析按钮 / 入库按钮 / 模式切换 / 文件更换。"""

    @staticmethod
    def _cluster(name, sid=1, matched=False, sim=0.0):
        import numpy as np

        return SpeakerCluster(speaker_id=sid, name=name,
                              embedding=np.ones(4, dtype=np.float32),
                              matched_library=matched, library_sim=sim,
                              segments=[(0.0, 10.0)], duration=10.0,
                              exemplar_span=(0.0, 10.0), exemplar_path="x.wav")

    def test_mode_panels_visibility(self):
        single, multi = app._speaker_panels("关闭")
        assert single["visible"] is False and multi["visible"] is False
        single, multi = app._speaker_panels("主声线过滤")
        assert single["visible"] is True and multi["visible"] is False
        single, multi = app._speaker_panels("多说话人分离")
        assert single["visible"] is False and multi["visible"] is True

    def test_file_change_resets_analysis(self):
        state, status, select, lib = app._on_file_change()
        assert state is None
        assert "重新分析" in status
        assert select["choices"] == [] and lib["choices"] == []

    def test_analyze_requires_file(self):
        outputs = list(app.analyze_speakers(None, "自动", True, False,
                                            progress=lambda *a, **k: None))
        assert len(outputs) == 1 and "请先上传" in outputs[0][0]

    def test_analyze_no_speech(self, monkeypatch, tmp_path):
        monkeypatch.setattr(app, "extract_wav",
                            lambda src, dst, **kw: tmp_path / "a.wav")
        monkeypatch.setattr(app, "speech_intervals", lambda path: [])
        outputs = list(app.analyze_speakers("a.mp4", "CPU", True, False,
                                            progress=lambda *a, **k: None))
        assert len(outputs) == 1
        assert "未检测到语音活动" in outputs[0][0]

    def _fake_analyzer(self, monkeypatch, analysis):
        calls = {}

        class FakeAnalyzer:
            def __init__(self, device="cpu"):
                calls["device"] = device

            def analyze(self, wav_path, segments, profiles_dir="profiles",
                        use_library=True, exemplar_dir=None, separated=False,
                        on_progress=None):
                calls.update(segments=list(segments), profiles_dir=profiles_dir,
                             use_library=use_library, exemplar_dir=exemplar_dir,
                             separated=separated, wav_path=wav_path)
                if on_progress:
                    on_progress(1.0)
                return analysis

        monkeypatch.setattr(app, "SpeakerAnalyzer", FakeAnalyzer)
        return calls

    def test_analyze_success_populates_choices(self, monkeypatch, tmp_path):
        analysis = SpeakerAnalysis(clusters=[
            self._cluster("小明", 1, matched=True, sim=0.9),
            self._cluster("说话人 2", 2),
        ])
        monkeypatch.setattr(app, "extract_wav",
                            lambda src, dst, **kw: tmp_path / "a.wav")
        monkeypatch.setattr(app, "speech_intervals",
                            lambda path: [(0, 5), (10, 20)])
        calls = self._fake_analyzer(monkeypatch, analysis)
        outputs = list(app.analyze_speakers("a.mp4", "CPU", False, False,
                                            progress=lambda *a, **k: None))
        assert len(outputs) == 1
        status, state, select, lib = outputs[0]
        assert state is analysis
        assert "识别到 2 位说话人" in status and "小明" in status
        assert "命中声纹库 0.90" in status
        # 默认全选；入库下拉指向第一位
        assert select["choices"] == ["小明", "说话人 2"]
        assert select["value"] == ["小明", "说话人 2"]
        assert lib["choices"] == ["小明", "说话人 2"] and lib["value"] == "小明"
        # 分析参数透传（含声纹库开关、试听片段目录）
        assert calls["segments"] == [(0, 5), (10, 20)]
        assert calls["use_library"] is False
        assert calls["profiles_dir"] == app._PROFILES_DIR
        assert calls["exemplar_dir"] is not None

    def test_analyze_no_valid_speakers(self, monkeypatch, tmp_path):
        monkeypatch.setattr(app, "extract_wav",
                            lambda src, dst, **kw: tmp_path / "a.wav")
        monkeypatch.setattr(app, "speech_intervals", lambda path: [(0, 5)])
        self._fake_analyzer(monkeypatch, SpeakerAnalysis())
        outputs = list(app.analyze_speakers("a.mp4", "CPU", True, False,
                                            progress=lambda *a, **k: None))
        status, state, select, _ = outputs[0]
        assert "未发现有效说话人" in status
        assert state is not None and state.clusters == []
        assert select["choices"] == []

    def test_analyze_error_readable(self, monkeypatch, tmp_path):
        from core.audio import AudioProcessError

        def boom(src, dst, **kw):
            raise AudioProcessError("ffprobe 探测失败")

        monkeypatch.setattr(app, "extract_wav", boom)
        outputs = list(app.analyze_speakers("a.mp4", "CPU", True, False,
                                            progress=lambda *a, **k: None))
        assert "分析失败" in outputs[0][0] and "ffprobe" in outputs[0][0]

    def test_analyze_auto_device_resolved(self, monkeypatch, tmp_path):
        class FakeEnv:
            device = "cuda"

        monkeypatch.setattr(app, "detect_env", lambda: FakeEnv())
        monkeypatch.setattr(app, "extract_wav",
                            lambda src, dst, **kw: tmp_path / "a.wav")
        monkeypatch.setattr(app, "speech_intervals", lambda path: [(0, 5)])
        calls = self._fake_analyzer(monkeypatch, SpeakerAnalysis())
        list(app.analyze_speakers("a.mp4", "自动", True, False,
                                  progress=lambda *a, **k: None))
        assert calls["device"] == "cuda"

    def test_analyze_with_separation(self, monkeypatch, tmp_path):
        """勾选分离：Demucs 先行，VAD/声纹/试听全部在纯人声上进行。"""
        vocals = tmp_path / "vocals.wav"
        sep_calls = {}

        class FakeSeparator:
            def __init__(self, device="cpu"):
                sep_calls["device"] = device

            def separate(self, wav_path, output_dir):
                sep_calls.update(src=wav_path, out=output_dir)
                return str(vocals)

        monkeypatch.setattr(app, "VocalSeparator", FakeSeparator)
        monkeypatch.setattr(app, "extract_wav",
                            lambda src, dst, **kw: tmp_path / "a.wav")
        monkeypatch.setattr(app, "speech_intervals", lambda path: [(0, 5)])
        calls = self._fake_analyzer(monkeypatch, SpeakerAnalysis())
        list(app.analyze_speakers("a.mp4", "CPU", True, True,
                                  progress=lambda *a, **k: None))
        assert sep_calls["src"] == str(tmp_path / "a.wav")   # 先抽取再分离
        assert calls["wav_path"] == str(vocals)             # 分析用分离后人声
        assert calls["separated"] is True                    # 来源标记写入结果

    def test_analyze_separator_fallback_continues(self, monkeypatch, tmp_path):
        """Demucs 失败回退原始音频（separate 返回原路径）→ 分析不中断。"""
        monkeypatch.setattr(app, "VocalSeparator",
                            lambda device="cpu": type("S", (), {
                                "separate": staticmethod(
                                    lambda w, o: w)})())
        monkeypatch.setattr(app, "extract_wav",
                            lambda src, dst, **kw: tmp_path / "a.wav")
        monkeypatch.setattr(app, "speech_intervals", lambda path: [(0, 5)])
        calls = self._fake_analyzer(monkeypatch, SpeakerAnalysis())
        outputs = list(app.analyze_speakers("a.mp4", "CPU", True, True,
                                            progress=lambda *a, **k: None))
        assert calls["wav_path"] == str(tmp_path / "a.wav")
        assert "未发现有效说话人" in outputs[0][0]

    def test_analyze_cleans_old_dirs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(app, "_ANALYSIS_ROOT", tmp_path)
        for i in range(app._ANALYSIS_KEEP):
            (tmp_path / f"20260101-00000{i}").mkdir()
        monkeypatch.setattr(app, "extract_wav",
                            lambda src, dst, **kw: tmp_path / "a.wav")
        monkeypatch.setattr(app, "speech_intervals", lambda path: [(0, 5)])
        self._fake_analyzer(monkeypatch, SpeakerAnalysis())
        list(app.analyze_speakers("a.mp4", "CPU", True, False,
                                  progress=lambda *a, **k: None))
        remaining = [p.name for p in tmp_path.iterdir() if p.is_dir()]
        assert len(remaining) == app._ANALYSIS_KEEP   # 旧 8 个 + 新 1 个 − 清理 1 个

    def test_save_requires_analysis(self):
        msg, update = app.save_speaker_to_library("甲", "小明", None)
        assert "请先分析" in msg

    def test_save_requires_name(self):
        analysis = SpeakerAnalysis(clusters=[self._cluster("甲")])
        msg, _ = app.save_speaker_to_library("甲", "  ", analysis)
        assert "声纹库名称" in msg

    def test_save_unknown_speaker(self):
        analysis = SpeakerAnalysis(clusters=[self._cluster("甲")])
        msg, _ = app.save_speaker_to_library("乙", "小明", analysis)
        assert "未找到" in msg

    def test_save_success_calls_save_library_speaker(self, monkeypatch):
        import numpy as np

        cluster = self._cluster("甲")
        analysis = SpeakerAnalysis(clusters=[cluster])
        saved = {}

        def fake_save(name, embedding, profiles_dir="profiles"):
            saved.update(name=name, embedding=embedding,
                         profiles_dir=profiles_dir)
            return "profiles/小明.npy"

        monkeypatch.setattr(app, "save_library_speaker", fake_save)
        monkeypatch.setattr(app, "list_profiles", lambda d: ["小明"])
        msg, update = app.save_speaker_to_library("甲", "小明", analysis)
        assert "已将「甲」存入声纹库" in msg and "小明" in msg
        assert saved["name"] == "小明"
        assert np.array_equal(saved["embedding"], cluster.embedding)
        assert saved["profiles_dir"] == app._PROFILES_DIR
        assert update["choices"] == ["小明"] and update["value"] == "小明"

    def test_save_voiceprint_error_readable(self, monkeypatch):
        analysis = SpeakerAnalysis(clusters=[self._cluster("甲")])

        def boom(*a, **kw):
            raise VoiceprintError("主播名不能为空")

        monkeypatch.setattr(app, "save_library_speaker", boom)
        msg, _ = app.save_speaker_to_library("甲", "小明", analysis)
        assert "入库失败" in msg and "主播名不能为空" in msg

    def test_save_unexpected_error_caught(self, monkeypatch):
        analysis = SpeakerAnalysis(clusters=[self._cluster("甲")])

        def boom(*a, **kw):
            raise RuntimeError("磁盘满")

        monkeypatch.setattr(app, "save_library_speaker", boom)
        msg, _ = app.save_speaker_to_library("甲", "小明", analysis)
        assert "未预期的错误" in msg


class TestPreload:
    def test_preload_loads_default_model_with_env_config(self, monkeypatch):
        calls = []

        class FakeEnv:
            device = "cpu"
            compute_type = "int8"

        monkeypatch.setattr(app, "detect_env", lambda: FakeEnv())

        import core.transcriber as tr_mod

        monkeypatch.setattr(tr_mod, "_get_model", lambda size, dev, ct: calls.append((size, dev, ct)))
        app._preload_model_once()
        assert calls == [("small", "cpu", "int8")]

    def test_preload_swallows_failure(self, monkeypatch):
        class FakeEnv:
            device = "cuda"
            compute_type = "float16"

        monkeypatch.setattr(app, "detect_env", lambda: FakeEnv())

        import core.transcriber as tr_mod

        def boom(size, dev, ct):
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(tr_mod, "_get_model", boom)
        app._preload_model_once()  # 不应抛出


class TestWorker:
    def test_success_messages(self, monkeypatch, tmp_path):
        lines = [SubtitleLine(words=[SubtitleWord("你好", 0.0, 1.0)])]
        from core.pipeline import PipelineResult

        result = PipelineResult(mode="blind", duration=1.0, lines=lines,
                                files={"srt": "a.srt"}, warnings=[], out_dir=str(tmp_path))

        class FakePipeline:
            def __init__(self, cfg, on_progress=None, on_log=None, cancel_event=None):
                self.on_progress = on_progress
                self.on_log = on_log

            def run(self):
                self.on_progress(0.5, "识别中")
                self.on_log("日志消息")
                return result

        monkeypatch.setattr(app, "Pipeline", FakePipeline)
        q = queue.Queue()
        app._worker(PipelineConfig(input_path="x"), q, threading.Event())
        kinds = [m["kind"] for m in drain(q)]
        assert kinds == ["progress", "log", "done", "end"]
        assert q_empty(q)

    def test_cancelled_message(self, monkeypatch):
        class FakePipeline:
            def __init__(self, cfg, on_progress=None, on_log=None, cancel_event=None):
                pass

            def run(self):
                raise TaskCancelled("取消")

        monkeypatch.setattr(app, "Pipeline", FakePipeline)
        q = queue.Queue()
        app._worker(PipelineConfig(input_path="x"), q, threading.Event())
        kinds = [m["kind"] for m in drain(q)]
        assert kinds == ["cancelled", "end"]
        assert q_empty(q)

    def test_error_message(self, monkeypatch):
        class FakePipeline:
            def __init__(self, cfg, on_progress=None, on_log=None, cancel_event=None):
                pass

            def run(self):
                raise PipelineError("音频抽取失败：boom")

        monkeypatch.setattr(app, "Pipeline", FakePipeline)
        q = queue.Queue()
        app._worker(PipelineConfig(input_path="x"), q, threading.Event())
        msgs = drain(q)
        assert [m["kind"] for m in msgs] == ["error", "end"]
        assert "boom" in msgs[0]["message"]

    def test_unexpected_error_caught(self, monkeypatch):
        class FakePipeline:
            def __init__(self, cfg, on_progress=None, on_log=None, cancel_event=None):
                pass

            def run(self):
                raise ValueError("炸了")

        monkeypatch.setattr(app, "Pipeline", FakePipeline)
        q = queue.Queue()
        app._worker(PipelineConfig(input_path="x"), q, threading.Event())
        msgs = drain(q)
        assert [m["kind"] for m in msgs] == ["error", "end"]
        assert "炸了" in msgs[0]["message"]


def drain(q: queue.Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def q_empty(q: queue.Queue) -> bool:
    return q.empty()
