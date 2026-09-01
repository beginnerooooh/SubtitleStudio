"""core/lyrics_fetcher.py 单测：LRC 解析、行构造、LRCLIB 查询（mock HTTP）。"""
import json
import urllib.error

import pytest

import core.lyrics_fetcher as lf
from core.lyrics_fetcher import (
    LyricTrack,
    build_lyric_lines,
    distribute_words,
    fetch_lyrics,
    parse_lrc,
    reset_cache,
)


@pytest.fixture(autouse=True)
def clean_cache():
    reset_cache()
    yield
    reset_cache()


def _mock_http(monkeypatch, payloads):
    """按 URL 子串分发假响应；payloads: {子串: dict|list|Exception}；记录请求 URL。"""
    calls = []

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        calls.append(url)
        for key, payload in payloads.items():
            if key in url:
                if isinstance(payload, Exception):
                    raise payload
                return FakeResp(payload)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(lf.urllib.request, "urlopen", fake_urlopen)
    return calls


class TestParseLrc:
    def test_basic_timestamps(self):
        text = "[00:01.00]第一行\n[00:03.50]第二行"
        assert parse_lrc(text) == [(1.0, "第一行"), (3.5, "第二行")]

    def test_multiple_stamps_expand(self):
        text = "[00:12.00][01:15.00]副歌"
        assert parse_lrc(text) == [(12.0, "副歌"), (75.0, "副歌")]

    def test_offset_shifts_earlier(self):
        text = "[offset:+500]\n[00:02.00]行"  # +500ms → 提前 0.5s
        assert parse_lrc(text) == [(1.5, "行")]

    def test_negative_offset(self):
        text = "[offset:-250]\n[00:01.00]行"
        assert parse_lrc(text) == [(1.25, "行")]

    def test_metadata_and_blank_skipped(self):
        text = "[ti:晴天]\n[ar:周杰伦]\n[00:05.00]\n\n[00:01.00]歌词"
        assert parse_lrc(text) == [(1.0, "歌词")]

    def test_fraction_formats(self):
        assert parse_lrc("[00:01:50]a") == [(1.5, "a")]      # 冒号毫秒
        assert parse_lrc("[00:01.5]b") == [(1.5, "b")]       # 两位小数
        assert parse_lrc("[1:01.250]c") == [(61.25, "c")]    # 一位分钟

    def test_output_sorted(self):
        assert parse_lrc("[00:20.00]后\n[00:10.00]前") == [
            (10.0, "前"), (20.0, "后")]

    def test_empty(self):
        assert parse_lrc("") == []


class TestDistributeWords:
    def test_cjk_even_by_char_weight(self):
        words = distribute_words("你好世界", 0.0, 4.0)
        assert [w.text for w in words] == ["你", "好", "世", "界"]
        assert words[0].start == pytest.approx(0.0)
        assert words[-1].end == pytest.approx(4.0)
        for a, b in zip(words, words[1:]):
            assert a.end == pytest.approx(b.start)

    def test_monotonic_within_line(self):
        words = distribute_words("hello world 你", 10.0, 13.0)
        times = [t for w in words for t in (w.start, w.end)]
        assert times == sorted(times)

    def test_zero_span_gets_minimum_duration(self):
        words = distribute_words("你好", 5.0, 5.0)
        assert words[-1].end > words[0].start


class TestBuildLyricLines:
    LRC = "[00:01.00]第一行\n[00:03.00]第二行\n[00:05.00]第三行"

    def test_offset_applied(self):
        lines = build_lyric_lines(self.LRC, block_offset=100.0)
        assert [ln.start for ln in lines] == pytest.approx([101.0, 103.0, 105.0])
        # 行结束 = 下一行开始
        assert lines[0].end == pytest.approx(103.0)
        assert [ln.text for ln in lines] == ["第一行", "第二行", "第三行"]

    def test_until_truncates(self):
        lines = build_lyric_lines(self.LRC, block_offset=0.0, until=3.5)
        assert [ln.text for ln in lines] == ["第一行", "第二行"]
        assert lines[-1].end == pytest.approx(3.5)  # 末行被夹紧到截断点

    def test_last_line_default_span(self):
        lines = build_lyric_lines("[00:01.00]唯一", block_offset=10.0)
        assert lines[0].end == pytest.approx(11.0 + lf.DEFAULT_LINE_SPAN)

    def test_negative_time_clamped_to_block_start(self):
        # offset 使行时间早于 0 → 夹紧到块起点
        lines = build_lyric_lines(
            "[offset:+2000]\n[00:01.00]行", block_offset=10.0)
        assert lines[0].start == pytest.approx(10.0)

    def test_invalid_timestamp_line_skipped(self):
        assert build_lyric_lines("[-1.0]行", block_offset=0.0) == []


class TestFetchLyrics:
    def _payload(self, synced="[00:01.00]行", name="晴天"):
        return {
            "trackName": name, "artistName": "周杰伦",
            "syncedLyrics": synced, "plainLyrics": "行",
        }

    def test_exact_get_hit(self, monkeypatch):
        calls = _mock_http(monkeypatch, {"/api/get?": self._payload()})
        track = fetch_lyrics("晴天", "周杰伦")
        assert track is not None and track.synced == "[00:01.00]行"
        assert any("/api/get?" in u for u in calls)
        assert not any("/api/search?" in u for u in calls)  # 精确命中不再搜索

    def test_get_404_falls_back_to_search(self, monkeypatch):
        calls = _mock_http(monkeypatch, {"/api/search?": [self._payload()]})
        track = fetch_lyrics("晴天", "周杰伦")
        assert track is not None and track.synced
        assert any("/api/search?" in u for u in calls)

    def test_search_prefers_exact_name_with_synced(self, monkeypatch):
        rows = [
            {"trackName": "晴天 (Live)", "artistName": "周杰伦",
             "syncedLyrics": "[00:01.00]live", "plainLyrics": None},
            {"trackName": "晴天", "artistName": "周杰伦",
             "syncedLyrics": "[00:02.00]studio", "plainLyrics": None},
        ]
        _mock_http(monkeypatch, {"/api/search?": rows})
        track = fetch_lyrics("晴天", "周杰伦")
        assert track.synced == "[00:02.00]studio"  # 跳过 Live 版取精确匹配

    def test_search_fallback_any_synced(self, monkeypatch):
        rows = [{"trackName": "晴天 (Live)", "artistName": "周杰伦",
                 "syncedLyrics": "[00:01.00]live", "plainLyrics": None}]
        _mock_http(monkeypatch, {"/api/search?": rows})
        assert fetch_lyrics("晴天", "周杰伦").synced == "[00:01.00]live"

    def test_no_results_returns_none(self, monkeypatch):
        _mock_http(monkeypatch, {"/api/search?": []})
        assert fetch_lyrics("不存在的歌", "x") is None

    def test_network_error_returns_none(self, monkeypatch):
        _mock_http(monkeypatch, {})  # 所有请求 404
        assert fetch_lyrics("晴天", "周杰伦") is None

    def test_plain_only_accepted(self, monkeypatch):
        payload = {"trackName": "晴天", "artistName": "周杰伦",
                   "syncedLyrics": None, "plainLyrics": "纯文本"}
        _mock_http(monkeypatch, {"/api/get?": payload})
        track = fetch_lyrics("晴天", "周杰伦")
        assert track is not None and track.synced is None and track.plain == "纯文本"

    def test_cache_avoids_second_request(self, monkeypatch):
        calls = _mock_http(monkeypatch, {"/api/get?": self._payload()})
        assert fetch_lyrics("晴天", "周杰伦") is not None
        n = len(calls)
        assert fetch_lyrics("晴天", "周杰伦") is not None
        assert len(calls) == n  # 第二次命中进程内缓存

    def test_cache_stores_negative_result(self, monkeypatch):
        calls = _mock_http(monkeypatch, {})
        assert fetch_lyrics("查无此歌", "x") is None
        n = len(calls)
        assert fetch_lyrics("查无此歌", "x") is None
        assert len(calls) == n
