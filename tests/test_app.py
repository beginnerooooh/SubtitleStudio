"""app.py 单测：UI 可构建 + 后台 worker 的队列消息协议（pipeline 注入假件）。"""
import queue
import threading

import gradio as gr
import pytest

import app
from core.errors import TaskCancelled
from core.models import SubtitleLine, SubtitleWord
from core.pipeline import PipelineConfig, PipelineError


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
