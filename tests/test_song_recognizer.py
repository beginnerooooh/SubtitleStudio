"""core/song_recognizer.py 单测：块合并、时间戳格式化、识曲控制流（mock shazamio/FFmpeg）。"""
import asyncio
import sys
import threading
import types

import pytest

import core.song_recognizer as sr
from core.errors import TaskCancelled
from core.song_recognizer import (
    SongEntry,
    SongRecognizer,
    fmt_hms,
    format_timeline_csv,
    format_timeline_md,
    merge_blocks,
    merge_consecutive,
)


class TestFmtHms:
    def test_pad_and_carry(self):
        assert fmt_hms(0) == "00:00:00"
        assert fmt_hms(59.9) == "00:00:59"
        assert fmt_hms(932.0) == "00:15:32"
        assert fmt_hms(3661.5) == "01:01:01"

    def test_hours_beyond_24(self):
        assert fmt_hms(90061.0) == "25:01:01"


class TestMergeBlocks:
    def test_close_segments_merged(self):
        assert merge_blocks([(0, 30), (38, 60)]) == [(0, 60)]

    def test_far_segments_kept(self):
        assert merge_blocks([(0, 30), (41, 60)]) == [(0, 30), (41, 60)]

    def test_unsorted_input(self):
        assert merge_blocks([(41, 60), (0, 30)]) == [(0, 30), (41, 60)]

    def test_empty(self):
        assert merge_blocks([]) == []

    def test_gap_boundary_inclusive(self):
        assert merge_blocks([(0, 30), (40, 60)], gap=10) == [(0, 60)]


class TestMergeConsecutive:
    def _entry(self, start, end, title="晴天", artist="周杰伦", conf=90):
        return SongEntry(start, end, title, artist, conf)

    def test_same_song_adjacent_merged(self):
        out = merge_consecutive([
            self._entry(100, 200),
            self._entry(205, 300, conf=95),  # gap 5s ≤ 30s
        ])
        assert len(out) == 1
        assert (out[0].start, out[0].end) == (100, 300)
        assert out[0].confidence == 95  # 取最大置信度

    def test_different_songs_not_merged(self):
        out = merge_consecutive([
            self._entry(100, 200),
            self._entry(205, 300, title="起风了", artist="买辣椒也用券"),
        ])
        assert len(out) == 2

    def test_same_song_far_gap_not_merged(self):
        out = merge_consecutive([
            self._entry(100, 200),
            self._entry(240, 300),  # gap 40s > 30s（重复唱同一首不合并）
        ])
        assert len(out) == 2

    def test_case_insensitive_title_match(self):
        out = merge_consecutive([
            SongEntry(0, 10, "Sunny Day", "A", 80),
            SongEntry(15, 25, "sunny day", "a", 85),
        ])
        assert len(out) == 1

    def test_three_in_a_row(self):
        out = merge_consecutive([
            self._entry(0, 10),
            self._entry(12, 20),
            self._entry(22, 30),
        ])
        assert len(out) == 1
        assert (out[0].start, out[0].end) == (0, 30)


class TestTimelineMd:
    def test_format_with_confidence(self):
        md = format_timeline_md([SongEntry(932.0, 1185.0, "晴天", "周杰伦", 98)])
        assert md == (
            "# 直播歌单时间戳索引\n\n"
            "- [00:15:32 - 00:19:45] 《晴天》- 周杰伦 (置信度: 98%)\n"
        )

    def test_format_without_confidence(self):
        md = format_timeline_md([SongEntry(0, 60, "晴天", "周杰伦", None)])
        assert "- [00:00:00 - 00:01:00] 《晴天》- 周杰伦\n" in md
        assert "置信度" not in md

    def test_format_without_artist(self):
        md = format_timeline_md([SongEntry(0, 60, "晴天", "", None)])
        assert "- [00:00:00 - 00:01:00] 《晴天》\n" in md

    def test_empty(self):
        md = format_timeline_md([])
        assert md == "# 直播歌单时间戳索引\n\n未检测到歌曲。\n"

    def test_multiple_entries_in_order(self):
        md = format_timeline_md([
            SongEntry(2710, 2900, "起风了", "买辣椒也用券", 95),
            SongEntry(932, 1185, "晴天", "周杰伦", 98),
        ])
        lines = [ln for ln in md.splitlines() if ln.startswith("- ")]
        assert "00:15:32" in lines[0]
        assert "00:45:10" in lines[1]


class TestTimelineCsv:
    def test_rows_and_header(self):
        csv = format_timeline_csv([
            SongEntry(932.0, 1185.0, "晴天", "周杰伦", 98),
            SongEntry(0, 60, "起,风了", "A B", None),
        ])
        lines = csv.splitlines()
        assert lines[0] == "start,end,title,artist,confidence"
        # 按起始时间排序：0s 条目在前
        assert lines[1] == '00:00:00,00:01:00,"起,风了",A B,'
        assert lines[2] == "00:15:32,00:19:45,晴天,周杰伦,98"

    def test_empty_header_only(self):
        assert format_timeline_csv([]) == "start,end,title,artist,confidence\n"


class TestCutSnippet:
    def test_ffmpeg_command_shape(self, tmp_path, monkeypatch):
        calls = []

        class FakeProc:
            returncode = 0
            stderr = ""

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return FakeProc()

        monkeypatch.setattr(sr.subprocess, "run", fake_run)
        out = tmp_path / "snip.wav"
        sr._cut_snippet("in.mp4", 63.5, 12.0, out, ffmpeg="/usr/bin/ffmpeg")
        assert calls == [[
            "/usr/bin/ffmpeg", "-y",
            "-ss", "63.5", "-t", "12.0", "-i", "in.mp4",
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            str(out),
        ]]

    def test_failure_raises_with_stderr(self, tmp_path, monkeypatch):
        class FakeProc:
            returncode = 1
            stderr = "boom"

        monkeypatch.setattr(sr.subprocess, "run", lambda cmd, **kw: FakeProc())
        with pytest.raises(RuntimeError, match="boom"):
            sr._cut_snippet("in.mp4", 0, 12.0, tmp_path / "s.wav")


class TestShazam:
    def _fake_shazamio(self, monkeypatch, result=None, exc=None):
        class FakeShazam:
            async def recognize(self, path):
                if exc:
                    raise exc
                return result
        module = types.ModuleType("shazamio")
        module.Shazam = FakeShazam
        monkeypatch.setitem(sys.modules, "shazamio", module)

    def test_parses_track(self, monkeypatch):
        self._fake_shazamio(monkeypatch, result={
            "track": {"title": "晴天", "subtitle": "周杰伦", "confidence": 98},
        })
        assert sr._shazam("x.wav", timeout=5) == {
            "title": "晴天", "artist": "周杰伦", "confidence": 98,
        }

    def test_no_track_returns_none(self, monkeypatch):
        self._fake_shazamio(monkeypatch, result={"matches": []})
        assert sr._shazam("x.wav", timeout=5) is None

    def test_exception_returns_none(self, monkeypatch):
        self._fake_shazamio(monkeypatch, exc=RuntimeError("network down"))
        assert sr._shazam("x.wav", timeout=5) is None

    def test_timeout_returns_none(self, monkeypatch):
        class SlowShazam:
            async def recognize(self, path):
                await asyncio.sleep(0.5)
                return {"track": {"title": "x", "subtitle": "y"}}
        module = types.ModuleType("shazamio")
        module.Shazam = SlowShazam
        monkeypatch.setitem(sys.modules, "shazamio", module)
        assert sr._shazam("x.wav", timeout=0.05) is None

    def test_missing_confidence_tolerated(self, monkeypatch):
        self._fake_shazamio(monkeypatch, result={
            "track": {"title": "晴天", "subtitle": "周杰伦"},
        })
        meta = sr._shazam("x.wav", timeout=5)
        assert meta["confidence"] is None


class TestRecognizeBlocks:
    @pytest.fixture
    def patched(self, tmp_path, monkeypatch):
        state = {"snips": [], "results": {}, "progress": []}
        monkeypatch.setattr(sr, "_cut_snippet",
                            lambda src, start, secs, dst, ffmpeg="ffmpeg":
                            (state["snips"].append((start, secs)) or dst))
        monkeypatch.setattr(sr, "_shazam",
                            lambda path, timeout:
                            state["results"].get(str(path), None))
        return state

    def test_success_and_failure_mix(self, patched, tmp_path):
        patched["results"] = {
            str(tmp_path / "song_snippet_0.wav"): {"title": "晴天", "artist": "周杰伦", "confidence": 98},
            # block 1 无识别结果
        }
        rec = SongRecognizer()
        entries = rec.recognize_blocks("live.mp4", [(100, 200), (400, 500)], tmp_path)
        assert patched["snips"] == [(100, 12.0), (400, 12.0)]
        assert len(entries) == 1
        assert (entries[0].start, entries[0].end) == (100, 200)
        assert entries[0].title == "晴天"

    def test_progress_per_block(self, patched, tmp_path):
        rec = SongRecognizer(on_progress=patched["progress"].append)
        rec.recognize_blocks("live.mp4", [(0, 60), (100, 160), (200, 260)], tmp_path)
        assert patched["progress"] == pytest.approx([1/3, 2/3, 1.0])

    def test_cancel_between_blocks(self, patched, tmp_path):
        cancel = threading.Event()
        cancel.set()
        rec = SongRecognizer(cancel_event=cancel)
        with pytest.raises(TaskCancelled):
            rec.recognize_blocks("live.mp4", [(0, 60), (100, 160)], tmp_path)

    def test_logs_each_outcome(self, patched, tmp_path):
        logs = []
        patched["results"] = {
            str(tmp_path / "song_snippet_0.wav"): {"title": "晴天", "artist": "周杰伦", "confidence": None},
        }
        rec = SongRecognizer(on_log=logs.append)
        rec.recognize_blocks("live.mp4", [(0, 60), (100, 160)], tmp_path)
        assert any("晴天" in m for m in logs)
        assert any("未能识别" in m for m in logs)

    def test_cut_failure_skips_block(self, patched, tmp_path, monkeypatch):
        def boom(src, start, secs, dst, ffmpeg="ffmpeg"):
            raise RuntimeError("ffmpeg gone")
        monkeypatch.setattr(sr, "_cut_snippet", boom)
        logs = []
        rec = SongRecognizer(on_log=logs.append)
        entries = rec.recognize_blocks("live.mp4", [(0, 60)], tmp_path)
        assert entries == []
        assert any("截取失败" in m for m in logs)
