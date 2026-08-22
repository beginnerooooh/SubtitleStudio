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
        assert rows[0] == [1, "00:00.00", "00:01.00", "字"]


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
            enable_voiceprint=False, profile_name="", voice_threshold=0.55,
            enable_song_detect=False, progress=lambda *a, **k: None,
        )
        args.update(kw)
        return list(app.start_task(**args))

    def test_missing_file_rejected(self):
        outputs = self._run(file=None)
        assert len(outputs) == 1 and "请先上传" in outputs[0][0]

    def test_voiceprint_without_profile_rejected(self):
        outputs = self._run(enable_voiceprint=True, profile_name="  ")
        assert len(outputs) == 1
        assert "未选择主播" in outputs[0][0]
        # 任务未启动：未占用任务锁
        assert app._task_lock.acquire(blocking=False)
        app._task_lock.release()


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
