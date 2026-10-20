"""新模块单测：切行 / 评测 / 编辑 / 弹幕 / 配置 / 外部接口解析。

这些测试不依赖 GPU 与模型，可在 CI 中快速跑完：
    pytest tests/test_core_v2.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from core import editor as ed                        # noqa: E402
from core.config import AppConfig                     # noqa: E402
from core.evaluate import EvalLine, evaluate, normalize_text, parse_srt  # noqa: E402
from core.livexml import parse_xml                    # noqa: E402
from core.models import SubtitleLine, SubtitleWord    # noqa: E402
from core.segmenter import SegmentTuning, segment_words, split_atoms  # noqa: E402


# ---------------- 切行 ----------------

def _words(chars: str, step: float = 0.3) -> list[SubtitleWord]:
    return [SubtitleWord(text=c, start=round(i * step, 3),
                         end=round((i + 1) * step, 3))
            for i, c in enumerate(chars)]


def test_split_atoms_splits_cjk_words():
    words = [SubtitleWord(text="晚上好", start=0.0, end=0.9)]
    atoms = split_atoms(words)
    assert [a.text for a in atoms] == ["晚", "上", "好"]
    assert atoms[0].start == 0.0 and atoms[-1].end == 0.9


def test_split_atoms_keeps_latin_words():
    words = [SubtitleWord(text="Butterfly", start=0.0, end=0.6),
             SubtitleWord(text="都忘了", start=0.6, end=1.2)]
    atoms = split_atoms(words)
    assert atoms[0].text == "Butterfly"
    assert [a.text for a in atoms[1:]] == ["都", "忘", "了"]


def test_segment_respects_punctuation_hint():
    """标点线索应把句子在标点处断开（hints[i] = 第 i 个原子之后断行）。"""
    chars = "晚上好晚上好大家好"
    words = _words(chars)
    hints = [False] * len(chars)
    hints[4] = True          # 第 5 个字之后断
    tuning = SegmentTuning(min_chars=2, max_chars=20, target_chars=9,
                           w_clause=30.0)
    lines = segment_words(words, tuning, punctuation=hints)
    texts = [ln.text for ln in lines]
    assert texts[0] == chars[:5]
    assert "".join(texts) == chars
    assert len(lines) >= 2


def test_segment_preserves_all_characters():
    chars = "要多努力才走得到远方如果梦想不曾绝望"
    words = _words(chars)
    lines = segment_words(words, SegmentTuning())
    assert "".join(ln.text for ln in lines) == chars


def test_segment_no_overlap_and_ordered():
    words = _words("abcdefghij" * 5)
    lines = segment_words(words, SegmentTuning(max_chars=8))
    for a, b in zip(lines, lines[1:]):
        assert a.end <= b.start + 1e-6


def test_segment_max_chars_respected():
    words = _words("字" * 60)
    lines = segment_words(words, SegmentTuning(max_chars=10))
    assert all(len(ln.text) <= 10 for ln in lines)


# ---------------- 评测 ----------------

def test_normalize_text_removes_punct_and_spaces():
    assert normalize_text("晚上好，晚上好！") == "晚上好晚上好"
    assert normalize_text("hello world") == "helloworld"


def test_parse_srt_handles_crlf_and_bom():
    body = ("\ufeff1\r\n00:00:07,083 --> 00:00:08,683\r\n晚上好\r\n\r\n"
            "2\r\n00:00:15,200 --> 00:00:16,933\r\n晚上好呀\r\n")
    lines = parse_srt(body)
    assert len(lines) == 2
    assert lines[0].start == pytest.approx(7.083)
    assert lines[1].text == "晚上好呀"


def test_evaluate_perfect_match():
    ref = [EvalLine(0.0, 1.0, "你好世界"), EvalLine(2.0, 3.0, "再见")]
    hyp = [EvalLine(0.0, 1.0, "你好世界"), EvalLine(2.0, 3.0, "再见")]
    report = evaluate(ref, hyp)
    assert report.exact_match_rate == 1.0
    assert report.cer == 0.0
    assert report.missing == 0 and report.extra == 0


def test_evaluate_counts_mismatches_and_marks_boundary_cases():
    ref = [EvalLine(0.0, 1.0, "你好世界"), EvalLine(2.0, 3.0, "再见")]
    hyp = [EvalLine(0.0, 1.0, "你好时间"), EvalLine(2.6, 3.4, "再见")]
    report = evaluate(ref, hyp)
    assert report.ref_count == 2
    assert report.exact_text_lines == 1
    assert report.missing == 0
    assert any(c["kind"].startswith("文本") for c in report.boundary_cases)
    assert any("时间偏移" in c["kind"] for c in report.boundary_cases)


def test_evaluate_detects_missing_and_extra_lines():
    ref = [EvalLine(0.0, 1.0, "甲"), EvalLine(5.0, 6.0, "乙")]
    hyp = [EvalLine(0.0, 1.0, "甲"), EvalLine(20.0, 21.0, "丙")]
    report = evaluate(ref, hyp)
    assert report.missing == 1
    assert report.extra == 1


def test_report_serialises_to_json():
    report = evaluate([EvalLine(0.0, 1.0, "abc")], [EvalLine(0.0, 1.0, "abc")])
    data = report.to_dict()
    assert data["exact_match_rate"] == 1.0
    assert "boundary_cases" in data


# ---------------- 编辑器 ----------------

def _line(text: str, start: float, end: float) -> SubtitleLine:
    return SubtitleLine(words=[SubtitleWord(text=text, start=start, end=end)])


def test_editor_split_line():
    lines = [_line("晚上好晚上好", 7.0, 9.0)]
    out, msg = ed.split_line(lines, 0, 3)
    assert len(out) == 2
    assert out[0].text == "晚上好" and out[1].text == "晚上好"
    assert out[0].start == 7.0 and out[1].end == 9.0
    assert "切分" in msg


def test_editor_merge_lines():
    lines = [_line("甲", 0.0, 1.0), _line("乙", 1.0, 2.0), _line("丙", 2.0, 3.0)]
    out, _msg = ed.merge_lines(lines, [0, 1])
    assert len(out) == 2
    assert out[0].text == "甲乙"
    assert out[0].start == 0.0 and out[0].end == 2.0


def test_editor_merge_requires_contiguous():
    lines = [_line("甲", 0.0, 1.0), _line("乙", 1.0, 2.0), _line("丙", 2.0, 3.0)]
    out, msg = ed.merge_lines(lines, [0, 2])
    assert len(out) == 3
    assert "连续" in msg


def test_editor_delete_and_insert():
    lines = [_line("甲", 0.0, 1.0), _line("乙", 1.0, 2.0)]
    out, _ = ed.delete_lines(lines, [0])
    assert [ln.text for ln in out] == ["乙"]
    out2, _ = ed.insert_line(out, 0, "新", 0.0, 0.9)
    assert [ln.text for ln in out2] == ["新", "乙"]


def test_editor_update_clamps_to_neighbours():
    lines = [_line("甲", 0.0, 1.0), _line("乙", 1.0, 2.0)]
    out, _ = ed.update_line(lines, 0, end=5.0)
    assert out[0].end <= out[1].start


def test_editor_shift_and_overlap_fix():
    lines = [_line("甲", 0.0, 1.0), _line("乙", 1.5, 2.5)]
    out, _ = ed.shift_lines(lines, 0.8, index=0)
    assert out[0].start == pytest.approx(0.8)
    fixed, msg = ed.fix_overlaps(out)
    assert fixed[0].end <= fixed[1].start
    assert "重叠" in msg


def test_editor_parse_ts_formats():
    assert ed.parse_ts("01:02.500") == pytest.approx(62.5)
    assert ed.parse_ts("00:01:02,500") == pytest.approx(62.5)
    assert ed.parse_ts("12.5") == pytest.approx(12.5)
    with pytest.raises(ValueError):
        ed.parse_ts("abc")


def test_editor_does_not_mutate_input():
    lines = [_line("甲", 0.0, 1.0)]
    before = [(ln.start, ln.end, ln.text) for ln in lines]
    ed.split_line(lines, 0, 1)
    after = [(ln.start, ln.end, ln.text) for ln in lines]
    assert before == after


# ---------------- 剪辑式操作（拖动/复制/粘贴） ----------------

def test_retime_edge_changes_only_one_side():
    lines = [_line("甲", 1.0, 3.0)]
    out, msg = ed.retime_edge(lines, 0, end=5.0)
    assert out[0].start == pytest.approx(1.0)
    assert out[0].end == pytest.approx(5.0)
    out2, _ = ed.retime_edge(lines, 0, start=0.5)
    assert out2[0].start == pytest.approx(0.5)
    assert out2[0].end == pytest.approx(3.0)
    assert "终点" in msg


def test_retime_edge_enforces_min_duration():
    lines = [_line("甲", 1.0, 3.0)]
    out, _ = ed.retime_edge(lines, 0, end=1.05, min_dur=0.5)
    assert out[0].end - out[0].start >= 0.5 - 1e-9


def test_retime_edge_allows_crossing_neighbours():
    """拖拽不做相邻夹紧（剪辑软件手感）；重叠由后续质检处理。"""
    lines = [_line("甲", 0.0, 2.0), _line("乙", 2.0, 4.0)]
    out, _ = ed.retime_edge(lines, 0, end=3.5)
    assert out[0].end == pytest.approx(3.5)


def test_copy_and_paste_preserves_relative_gaps():
    lines = [_line("甲", 1.0, 3.0), _line("乙", 4.0, 6.0)]
    data, msg = ed.copy_lines(lines, [0, 1])
    assert len(data) == 2 and "复制" in msg
    out, msg2 = ed.paste_lines(lines, data, at=20.0)
    starts = sorted(ln.start for ln in out)
    # 粘贴后保持 1s 的原始间隔（4.0 - 3.0）
    assert starts == pytest.approx([1.0, 4.0, 20.0, 23.0])
    assert "粘贴" in msg2


def test_paste_keeps_timeline_monotonic():
    lines = [_line("甲", 1.0, 3.0)]
    data, _ = ed.copy_lines(lines, [0])
    out, _ = ed.paste_lines(lines, data, at=50.0)
    assert all(out[i].start <= out[i + 1].start for i in range(len(out) - 1))


def test_paste_empty_clipboard_is_safe():
    lines = [_line("甲", 1.0, 3.0)]
    out, msg = ed.paste_lines(lines, [], at=5.0)
    assert len(out) == 1 and "空" in msg


def test_duplicate_offsets_time_to_avoid_overlap():
    lines = [_line("甲", 1.0, 3.0)]
    out, msg = ed.duplicate_lines(lines, [0])
    assert len(out) == 2
    assert out[0].end <= out[1].start + 1e-9   # 不重叠
    assert out[1].end - out[1].start == pytest.approx(2.0)  # 保持时长
    assert "复制" in msg


# ---------------- 项目存储：撤销/草稿/缓存 ----------------

def test_store_undo_redo_roundtrip(tmp_path):
    from core.store import ProjectStore

    st = ProjectStore(tmp_path / "projects")
    state = st.create(video="x.mp4", name="测试")
    state.lines = [{"start": 0, "end": 1, "text": "A", "words": []}]
    st.save(state, push_history=True, note="first")
    state.lines = [{"start": 0, "end": 1, "text": "B", "words": []}]
    st.save(state, push_history=True, note="second")

    back = st.undo(state.meta.id)
    assert back is not None and back.lines[0]["text"] == "A"
    again = st.redo(state.meta.id)
    assert again is not None and again.lines[0]["text"] == "B"
    info = st.history_info(state.meta.id)
    assert info["undo"] >= 1


def test_store_draft_flag_and_rename(tmp_path):
    from core.store import ProjectStore

    st = ProjectStore(tmp_path / "projects")
    draft = st.create(name="未命名")
    assert draft.meta.draft is True
    formal = st.create(name="正式项目")
    assert formal.meta.draft is False
    st.rename(draft.meta.id, "改好名字了")
    assert all(m.draft is False for m in st.list_projects()
               if m.id == draft.meta.id)


def test_store_autosave_and_recover(tmp_path):
    from core.store import ProjectStore

    st = ProjectStore(tmp_path / "projects")
    state = st.create(name="自动保存测试")
    state.lines = [{"start": 0, "end": 2, "text": "自动保存内容", "words": []}]
    assert st.autosave(state) is True
    rec = st.recover_autosave(state.meta.id)
    assert rec and rec[0]["text"] == "自动保存内容"


def test_store_clean_cache_only_touches_target_dir(tmp_path):
    """缓存清理只动 cache 目录，绝不碰 projects/profiles/lyrics。"""
    from core.store import ProjectStore

    st = ProjectStore(tmp_path / "projects")
    cache = tmp_path / "cache"
    cache.mkdir()
    stale = cache / "old.wav"
    stale.write_bytes(b"x" * 2048)
    keep = tmp_path / "profiles"
    keep.mkdir()
    voice = keep / "主播.npy"
    voice.write_bytes(b"y" * 1024)
    import os
    import time as _t

    old = _t.time() - 30 * 86400
    os.utime(stale, (old, old))
    stats = st.clean_cache(cache, max_age_days=7)
    assert stats["files"] == 1
    assert not stale.exists()
    assert voice.exists(), "声纹文件被误删"


def test_store_delete_removes_project(tmp_path):
    from core.store import ProjectStore

    st = ProjectStore(tmp_path / "projects")
    state = st.create(name="待删除")
    st.delete(state.meta.id)
    assert st.list_projects() == []
    with pytest.raises(FileNotFoundError):
        st.load(state.meta.id)


# ---------------- 外部歌词接口 ----------------

def test_custom_lyric_provider_parses_plain_lrc():
    from core.song import CustomLyricProvider

    # 无网络时返回空（安全降级）
    p = CustomLyricProvider("http://127.0.0.1:9/never", timeout=0.5)
    assert p.search(title="x") == []


def test_set_custom_endpoint_registers_provider():
    from core.song import PROVIDERS, set_custom_endpoint

    name = set_custom_endpoint("http://127.0.0.1:1/lyrics", "k")
    assert name == "custom" and "custom" in PROVIDERS


# ---------------- 桌面端源码约定（防止回归） ----------------

def _desktop_file(name: str) -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "desktop" / "renderer"
    return (root / name).read_text(encoding="utf-8")


def test_overlay_only_falls_back_to_selection_when_paused():
    """回归：选中字幕后播放到无字幕处，不应继续显示该字幕。

    实现约定：updateOverlay 的选中回退必须同时满足
    `_previewSelection` 与 `v.paused` 两个条件。
    """
    js = _desktop_file("app.js")
    assert "_previewSelection" in js
    # 回退条件必须是「预览标志 + 暂停」双条件
    seg = js[js.index("function updateOverlay"):]
    seg = seg[: seg.index("\nfunction ", 1)] if "\nfunction " in seg[1:] else seg
    assert "_previewSelection" in seg and "v.paused" in seg, \
        "updateOverlay 的选中回退必须受 _previewSelection 与 paused 双重约束"


def test_video_uses_absolute_url():
    """回归：渲染层是 file:// 页面，媒体 URL 必须带 http 前缀。"""
    js = _desktop_file("app.js")
    assert "mediaUrl" in js
    assert "`${API_BASE()}/api/media" in js, "媒体地址必须用绝对 URL"


def test_preview_container_is_column_flex():
    """回归：.preview 必须是纵向 flex，否则画面框会被播放条挤成 0 宽。"""
    css = _desktop_file("style.css")
    seg = css[css.index(".preview {"):]
    seg = seg[: seg.index("}") + 1]
    assert "flex-direction: column" in seg
    # 预览要有最小高度，避免被时间轴/日志拉伸压没
    assert "min-height" in seg


def test_splitters_do_not_shrink_preview():
    """回归：拖时间轴时应从日志抽屉扣空间，而不是压预览。"""
    js = _desktop_file("app.js")
    seg = js[js.index("$('split-timeline')"): ]
    seg = seg[: seg.index("$('split-drawer')")]
    assert "drawer" in seg, "拉伸时间轴必须优先压缩日志抽屉"


def test_zoom_control_exists_and_updates_label():
    js = _desktop_file("app.js")
    html = _desktop_file("index.html")
    assert 'id="zoom"' in html and 'id="zoom-label"' in html
    assert "zoom-label" in js and "setZoom" in js


def test_clip_shows_text():
    """回归：字幕轨片段必须渲染对应文字。"""
    js = _desktop_file("app.js")
    assert "clip-text" in js, "片段上必须渲染文字节点"
    css = _desktop_file("style.css")
    # 极窄片段要去掉 padding，否则宽度会被撑大、缩放失效
    assert ".clip.tiny" in css
    assert "min-width: 0" in css


def test_collapse_state_is_remembered():
    """回归：左面板折叠状态要持久化并能恢复。"""
    js = _desktop_file("app.js")
    assert "restoreDetails" in js and "bindDetailsMemory" in js
    assert "toggle" in js, "折叠变化应触发保存"
    # saveLayout 必须记录 details
    seg = js[js.index("function saveLayout"):]
    seg = seg[: seg.index("\n}")]
    assert "details" in seg


def test_zoom_percentage_baseline_is_min_span():
    """回归：比例以最短区间（5 秒）为 100% 基准。"""
    js = _desktop_file("app.js")
    seg = js[js.index("function updateZoomUI"):]
    seg = seg[: seg.index("\n}")]
    assert "SPAN_MIN / span * 100" in seg, \
        "比例必须用 SPAN_MIN / span 计算（最短区间 = 100%）"


def test_span_limits_are_5s_and_10min():
    js = _desktop_file("app.js")
    assert "SPAN_MIN = 5" in js, "最短区间应为 5 秒"
    assert "SPAN_MAX = 10 * 60" in js, "最长区间应为 10 分钟"


def test_timecode_uses_hhmmssff():
    """回归：时间统一用「时:分:秒:帧」，且能往返解析。"""
    js = _desktop_file("app.js")
    assert "function fmtTC" in js and "function parseTC" in js
    seg = js[js.index("function fmtTC"):]
    seg = seg[: seg.index("\n}")]
    # 必须包含时/分/秒/帧四段
    for part in ("hh", "mm", "ss"):
        assert part in seg
    assert "FPS" in js, "帧率必须来自视频信息"


def test_overlay_handle_removed():
    """回归：自由模式的辅助点已删除，不应再出现。"""
    html = _desktop_file("index.html")
    js = _desktop_file("app.js")
    assert 'id="ov-handle"' not in html, "辅助点节点应已删除"
    assert "ov-handle" not in js, "辅助点逻辑应已删除"
    css = _desktop_file("style.css")
    assert ".ov-handle" not in css, "辅助点样式应已删除"


def test_marquee_selection_exists():
    """回归：支持按住左键框选批量选择。"""
    html = _desktop_file("index.html")
    js = _desktop_file("app.js")
    assert 'id="marquee"' in html
    assert "beginMarquee" in js and "onMarquee" in js and "endMarquee" in js


def test_clip_edges_have_distinct_colors():
    """回归：左右拖拽线用不同颜色。"""
    css = _desktop_file("style.css")
    assert "--edge-l:" in css and "--edge-r:" in css
    # 两者取值必须不同
    import re

    l = re.search(r"--edge-l:\s*([^;]+);", css).group(1).strip()
    r = re.search(r"--edge-r:\s*([^;]+);", css).group(1).strip()
    assert l != r, "左右拖拽线颜色必须不同"


def test_selected_clip_keeps_primary_color():
    """回归：选中片段保持主色（蓝），不整体变成强调色。"""
    css = _desktop_file("style.css")
    seg = css[css.index(".clip.selected"):]
    seg = seg[: seg.index("}") + 1]
    assert "background: var(--clip)" in seg, \
        "选中态必须保持主色，只用描边/阴影区分"


# ---------------- 歌词库 ----------------

def test_lyric_library_parse_lrc_and_plain():
    from core.lyrics_library import parse_lyric_text

    lrc = "[00:12.00]要多努力才走得到远方\n[00:18.50]如果梦想不曾坠落悬崖"
    doc = parse_lyric_text(lrc, title="最初的梦想", artist="范玮琪")
    assert doc.synced is True and doc.line_count == 2
    assert doc.lines[0].time == pytest.approx(12.0)
    assert doc.lines[1].text.startswith("如果梦想")

    plain = "第一行歌词\n第二行歌词\n"
    d2 = parse_lyric_text(plain, title="无时间")
    assert d2.synced is False and d2.line_count == 2


def test_lyric_library_add_remove_and_persist(tmp_path):
    from core.lyrics_library import LyricLibrary

    lib = LyricLibrary(tmp_path / "lyrics")
    assert lib.size == 0
    lib.add_text("甲行\n乙行", title="测试曲", artist="某人")
    assert lib.size == 1
    # 重新打开应能读回（落盘 JSON）
    lib2 = LyricLibrary(tmp_path / "lyrics")
    assert lib2.size == 1
    assert lib2.get("测试曲").line_count == 2
    assert lib2.remove("测试曲") is True
    assert LyricLibrary(tmp_path / "lyrics").size == 0


def test_lyric_file_title_artist_inference(tmp_path):
    from core.lyrics_library import load_lyric_file

    p = tmp_path / "周杰伦 - 七里香.lrc"
    p.write_text("[00:01.00]窗外的麻雀\n", encoding="utf-8")
    doc = load_lyric_file(p)
    assert doc.title == "七里香" and doc.artist == "周杰伦"


def test_lyric_match_corrects_asr_errors():
    """核心：识别文本与歌词有局部差异时，应被歌词纠正。"""
    from core.lyrics_library import apply_report, match_document, parse_lyric_text

    doc = parse_lyric_text(
        "要多努力才走得到远方\n如果梦想不曾坠落悬崖千钧一发",
        title="最初的梦想")
    subs = [_line("要多努力才走得到远方", 0.0, 3.0),
            _line("如果梦想不曾坠落悬崖千军一发", 3.0, 7.0)]
    rep = match_document(subs, doc, threshold=0.5)
    assert rep.score > 0.7
    out, n = apply_report(subs, rep, min_score=0.7)
    assert n == 1
    assert out[1].text == "如果梦想不曾坠落悬崖千钧一发"


def test_lyric_match_rejects_unrelated_document():
    from core.lyrics_library import match_document, parse_lyric_text

    doc = parse_lyric_text("完全不同的另一首歌的歌词内容在这里",
                           title="无关歌曲")
    subs = [_line("今天我们要聊一个技术话题", 0.0, 3.0)]
    rep = match_document(subs, doc, threshold=0.55)
    assert rep.matches == [] and rep.score == 0.0


def test_lyric_boundary_hints_align_to_lines():
    """歌词分句应映射为断句线索，让字幕在歌词行末断开。"""
    from core.lyrics_library import lyric_boundary_hints, parse_lyric_text
    from core.models import SubtitleWord

    doc = parse_lyric_text("要多努力才走得到远方\n如果梦想不曾坠落",
                           title="x")
    text = "要多努力才走得到远方如果梦想不曾坠落"
    words = [SubtitleWord(text=ch, start=i * 0.25, end=(i + 1) * 0.25)
             for i, ch in enumerate(text)]
    hints = lyric_boundary_hints(words, doc)
    assert hints is not None
    # 断点应落在第 1 行歌词末尾（「方」= 索引 9）之后
    assert hints[9] is True
    assert sum(1 for h in hints if h) == 1


# ---------------- 弹幕旁路校正 ----------------

def test_danmaku_correction_fixes_proper_noun():
    """核心：弹幕反复出现的正确写法应纠正同音误识别。"""
    from core.danmaku_correct import correct_with_log
    from core.livexml import Danmaku, LiveLog

    log = LiveLog()
    for k in range(5):
        log.danmaku.append(Danmaku(time=10.0 + k * 0.5,
                                   text="主播叫脆脆鲨呀", user_id="u1"))
    subs = [_line("主播叫翠翠沙呀", 10.2, 13.0)]
    out, rep = correct_with_log(subs, log)
    assert rep.applied == 1
    assert out[0].text == "主播叫脆脆鲨呀"


def test_danmaku_correction_ignores_far_chatter():
    """时间上远离的弹幕不应影响字幕。"""
    from core.danmaku_correct import correct_with_log
    from core.livexml import Danmaku, LiveLog

    log = LiveLog()
    for k in range(5):
        log.danmaku.append(Danmaku(time=900.0 + k, text="主播叫脆脆鲨呀"))
    subs = [_line("主播叫翠翠沙呀", 10.0, 13.0)]
    out, rep = correct_with_log(subs, log, window=5.0)
    assert rep.applied == 0
    assert out[0].text == "主播叫翠翠沙呀"


def test_danmaku_correction_never_changes_timeline():
    from core.danmaku_correct import correct_with_log
    from core.livexml import Danmaku, LiveLog

    log = LiveLog()
    for k in range(4):
        log.danmaku.append(Danmaku(time=10.0, text="主播叫脆脆鲨呀"))
    subs = [_line("主播叫翠翠沙呀", 10.0, 13.0)]
    out, _ = correct_with_log(subs, log)
    assert out[0].start == pytest.approx(10.0)
    assert out[0].end == pytest.approx(13.0)


def test_danmaku_correction_without_log_is_safe():
    from core.danmaku_correct import correct_with_log

    subs = [_line("任意文本内容", 0.0, 2.0)]
    out, rep = correct_with_log(subs, None)
    assert rep.applied == 0 and out[0].text == "任意文本内容"


# ---------------- 配置项存在性 ----------------

def test_new_song_and_segment_config_fields():
    from core.config import SegmentConfig, SongConfig

    sc = SongConfig()
    for name in ("manual_titles", "restrict_bgm_to_manual", "lyric_correct",
                 "lyric_library_dir", "segment_by_lyrics",
                 "lyric_doc_threshold", "lyric_line_threshold",
                 "danmaku_correct", "danmaku_time_window"):
        assert hasattr(sc, name), f"SongConfig 缺少 {name}"
    sg = SegmentConfig()
    assert hasattr(sg, "inner_space")


def test_flv_and_mkv_require_transcode():
    """回归：浏览器不支持的容器必须走转码，而不是直接流式返回。"""
    import desktop_server as ds

    for ext in (".flv", ".mkv", ".ts", ".rmvb", ".avi"):
        assert ext in ds.TRANSCODE_CONTAINERS, f"{ext} 应走转码"
    # MP4/WebM 可直接播放，不应转码
    for ext in (".mp4", ".webm"):
        assert ext not in ds.TRANSCODE_CONTAINERS


def test_hevc_is_not_browser_safe():
    """回归：HEVC 不在浏览器可解码集合里。

    这是 4GB 直播 FLV 黑屏的根因——只换容器（-c copy）把 HEVC 原样
    搬进 MP4，Chromium 依然无法解码。
    """
    import desktop_server as ds

    assert "hevc" not in ds.BROWSER_SAFE_VCODECS
    assert "h264" in ds.BROWSER_SAFE_VCODECS
    # 音频侧：AC-3 也不可播
    assert "ac3" not in ds.BROWSER_SAFE_ACODECS
    assert "aac" in ds.BROWSER_SAFE_ACODECS


def test_transcode_args_skip_copy_for_unsafe_codec():
    """回归：编码不安全时不能只 remux，且重编码必须带缩放滤镜。"""
    import desktop_server as ds

    safe = ds._transcode_args(None, can_copy=True)
    assert safe[0] == ["-c", "copy"], "可 remux 时应优先 copy"
    unsafe = ds._transcode_args(None, can_copy=False)
    assert all("copy" not in a for a in unsafe), \
        "编码不安全时必须重编码，不能 copy"
    # 重编码路径必须缩到预览规格（540p30），否则 1080p60 要慢一倍
    assert any("-vf" in a for a in unsafe), "重编码必须带 -vf 缩放"
    assert any(ds.PREVIEW_VF in a for a in unsafe), "应使用 PREVIEW_VF 规格"


# ---------------- 阻塞进度层（防重复点击） ----------------

def test_busy_overlay_exists_and_blocks_input():
    html = _desktop_file("index.html")
    css = _desktop_file("style.css")
    js = _desktop_file("app.js")
    assert 'id="busy"' in html, "缺少阻塞层节点"
    assert 'id="busy-text"' in html and 'id="busy-fill"' in html
    # 遮罩必须覆盖全屏且在最上层
    seg = css[css.index(".busy {"):]
    seg = seg[: seg.index("}") + 1]
    assert "position: fixed" in seg and "inset: 0" in seg
    assert "z-index" in seg
    # JS 侧必须有显示/隐藏/进度/防重入
    for fn in ("busyShow", "busyHide", "busyProgress", "withBusy"):
        assert f"function {fn}" in js, f"缺少 {fn}"


def test_api_json_wraps_slow_operations():
    """回归：写操作应自动套阻塞层（防重复点击）。"""
    js = _desktop_file("app.js")
    assert "function apiLabel" in js
    # 关键慢操作都要有可读标签
    for kw in ("正在识别字幕", "正在导出", "正在拉取歌词", "正在提取声纹"):
        assert kw in js, f"缺少操作标签：{kw}"
    # 自动保存不应打扰用户
    assert "autosave" in js


def test_video_prepare_checks_transcode_status():
    """回归：打开视频前先查转码状态并显示进度。"""
    js = _desktop_file("app.js")
    assert "preparePlayable" in js
    assert "/api/transcode/status" in js
    assert "HEVC" in js or "hevc" in js.lower()


def test_codec_probe_is_cached_and_bounded():
    """回归：编码探测要缓存且限制扫描范围（4GB 文件否则要 25 秒）。"""
    import inspect

    import desktop_server as ds

    src = inspect.getsource(ds._probe_codecs)
    assert "_CODEC_CACHE" in src, "编码探测必须缓存"
    assert "-analyzeduration" in src and "-probesize" in src, \
        "必须限制 ffprobe 扫描范围，否则大文件极慢"


# ---------------- 效率相关回归 ----------------

def test_cache_key_is_process_stable():
    """回归：缓存键必须跨进程稳定。

    这是「每次打开都重新转码半小时」的根因：Python 的 `hash()` 对
    字符串按进程随机加盐，三次独立进程会给出三个不同的键。
    """
    import inspect

    import desktop_server as ds

    src = inspect.getsource(ds._cache_key)
    assert "hashlib" in src, "必须用 hashlib 生成稳定键"
    # 同一输入在同一进程内必须稳定
    assert ds._cache_key("a", 1, 2.0) == ds._cache_key("a", 1, 2.0)
    # 不同输入必须不同
    assert ds._cache_key("a") != ds._cache_key("b")


def test_file_key_uses_size_and_mtime():
    import desktop_server as ds
    from pathlib import Path

    p = Path(__file__)
    k1 = ds._file_key(p)
    assert isinstance(k1, str) and len(k1) >= 16
    # 额外参数应改变键（缩略图/转码互不覆盖）
    assert ds._file_key(p, "thumbs") != ds._file_key(p, "transcode")


def test_asr_defaults_are_speed_tuned():
    """回归：默认值必须是实测验证过的提速组合。"""
    from core.config import AsrConfig

    cfg = AsrConfig()
    # beam=1 实测又快又准（见 config 注释中的对照表）
    assert cfg.beam_size == 1, "beam_size 应为 1（贪心，实测更准且快 1.8 倍）"
    assert cfg.best_of == 1, "best_of 应为 1"
    # chunk=120 避免 whisper 在长块上静默跳段（1200s 时 CER 0.721）
    assert cfg.chunk_seconds == 120.0, "chunk_seconds 应为 120 秒"
    assert cfg.vad_filter is False, "VAD 必须默认关闭（否则吞掉歌声）"


def test_preview_proxy_is_downscaled():
    """回归：预览代理必须降规格，否则 1080p60 转码慢一倍。"""
    import desktop_server as ds

    assert "540" in ds.PREVIEW_VF, "预览代理应缩到 540p"
    assert "fps=30" in ds.PREVIEW_VF, "预览代理应降到 30fps"


def test_encoder_detection_prefers_hardware():
    """回归：编码器探测应优先硬件（NVENC/QSV/AMF），最后才 CPU。"""
    import inspect

    import desktop_server as ds

    src = inspect.getsource(ds._pick_encoder)
    for hw in ("h264_nvenc", "h264_qsv", "h264_amf"):
        assert hw in src, f"应尝试 {hw}"
    assert "libx264" in src, "应有 CPU 兜底"
    # NVENC 必须排在 libx264 之前（硬件优先）
    assert src.index("h264_nvenc") < src.index("libx264")

def test_hwaccel_detection_exists():
    import inspect

    import desktop_server as ds

    src = inspect.getsource(ds._pick_hwaccel)
    assert "cuda" in src
    # 必须有无硬件时的空列表兜底
    assert "_HWACCEL_ARGS = []" in src


def test_transcode_outputs_are_separate_files():
    """回归：前段与完整版必须是不同文件。

    Windows 下覆盖正在被 HTTP 流读取的文件会失败（os.replace 报
    文件被占用），导致「后台转完了但用户仍只有前段」。
    """
    import inspect

    import desktop_server as ds

    # _serve_transcoded 是 Handler 的方法，从类上取源码
    src = inspect.getsource(ds.Handler._serve_transcoded)
    assert ".head.mp4" in src, "前段应使用独立的 .head.mp4"
    assert "full" in src, "应有完整版路径"
    # 不应再用「完整版覆盖前段」的写法
    assert "out.with_suffix" not in src, \
        "前段与完整版必须是两个文件，不能互相覆盖"


def test_stale_transcode_sweep_skips_active_jobs():
    """回归：清理残留临时文件时必须跳过正在写入的。"""
    import inspect

    import desktop_server as ds

    src = inspect.getsource(ds._sweep_stale_transcode)
    assert "_TRANSCODE_JOBS" in src, "必须检查活跃任务"
    assert "active" in src, "必须跳过正在写入的文件"


def test_timeline_uses_source_duration():
    """回归：时间轴时长必须用源文件时长，而非 <video>.duration。

    转码格式只播前段代理（90 秒）时，若按 video.duration 渲染，
    生成的字幕会与时间轴完全错位。
    """
    js = _desktop_file("app.js")
    assert "sourceDuration" in js, "必须记录源文件时长"
    seg = js[js.index("function timelineDuration"):]
    seg = seg[: seg.index("\n}")]
    assert "S.sourceDuration" in seg, \
        "timelineDuration 必须优先使用 S.sourceDuration"


def test_non_blocking_toast_replaces_alert():
    """回归：不应再使用阻塞式 alert/confirm。

    在 Electron 中 alert() 会同步阻塞渲染进程，弹窗未关时所有定时器、
    轮询、动画全部冻结。
    """
    js = _desktop_file("app.js")
    assert "function toast" in js and "function askConfirm" in js
    # 业务代码里不应残留 alert( 调用（注释与函数定义除外）
    code = "\n".join(ln for ln in js.splitlines()
                     if not ln.strip().startswith("*")
                     and not ln.strip().startswith("//"))
    assert "alert(" not in code, "仍有阻塞式 alert 调用"
    assert "confirm(" not in code.replace("askConfirm(", ""), \
        "仍有阻塞式 confirm 调用"


# ---------------- 歌词库：局部匹配（用户核心反馈） ----------------

def test_lyric_match_is_local_not_global_average():
    """回归：歌词匹配必须按「局部连续段」评分，不能按全片平均。

    这是「歌词库匹配效果不行」的根因：一首 5 行的歌混在 200 行闲聊里，
    若按全片平均，得分被稀释到 0.006 量级 → 整首歌被否决。
    """
    from core.lyrics_library import match_document, parse_lyric_text

    doc = parse_lyric_text(
        "倘若这世间\n一切都在无情的崩裂\n我会用手中的线为你缝原",
        title="测试曲")
    # 200 行闲聊 + 3 行歌词（模拟真实直播：歌只占几分钟）
    rows = ["今天天气不错我们来聊聊天吧"] * 197
    rows += ["倘若这世界", "一切都在无尽的崩裂", "我会用手中的线为你奉约"]
    subs = [_line(t, i * 2.0, i * 2.0 + 1.8) for i, t in enumerate(rows)]

    rep = match_document(subs, doc, threshold=0.55)
    assert rep.score > 0.7, (
        f"局部匹配应给出高分，实际 {rep.score}（若按全片平均会被稀释）")
    assert len(rep.matches) == 3
    # 命中的必须是最后三行（即歌词所在位置）
    assert [m.line_index for m in rep.matches] == [197, 198, 199]


def test_lyric_replaces_asr_errors_with_original_text():
    """回归：高相似度的识别偏差应被歌词原文替换。"""
    from core.lyrics_library import apply_report, match_document, parse_lyric_text

    subs = [_line("倘若这世界", 0.0, 2.0),
            _line("一切都在无尽的崩裂", 2.0, 4.0),
            _line("我会用手中的线为你奉约", 4.0, 6.0)]
    doc = parse_lyric_text(
        "倘若这世间\n一切都在无情的崩裂\n我会用手中的线为你缝原",
        title="x")
    rep = match_document(subs, doc)
    out, n = apply_report(subs, rep, min_score=0.75)
    assert n == 3
    assert out[0].text == "倘若这世间"
    assert out[1].text == "一切都在无情的崩裂"
    assert out[2].text == "我会用手中的线为你缝原"


def test_lyric_rejects_unrelated_document_locally():
    """回归：无关歌词在局部评分下仍应被拒绝（避免误替换）。"""
    from core.lyrics_library import match_document, parse_lyric_text

    subs = [_line("倘若这世界", 0.0, 2.0), _line("一切都在无尽的崩裂", 2.0, 4.0)]
    doc = parse_lyric_text("完全不同的另一首歌的歌词内容在这里", title="y")
    rep = match_document(subs, doc)
    assert rep.score < 0.55, f"无关歌词不应得高分，实际 {rep.score}"


# ---------------- 关键词字库 ----------------

def test_keyword_store_add_and_alias(tmp_path):
    from core.keywords import KeywordStore

    st = KeywordStore(tmp_path / "keywords")
    st.add("脆脆鲨", alias=["翠翠沙", "脆脆沙"], category="昵称")
    st.add("舰长", category="术语")
    assert st.size == 2
    assert "脆脆鲨" in st.prompt_text()
    # 别名映射应指向正确写法
    assert st.match_map().get("翠翠沙") == "脆脆鲨"


def test_keyword_store_bulk_import_and_comments(tmp_path):
    from core.keywords import KeywordStore

    st = KeywordStore(tmp_path / "keywords")
    n = st.add_many("# 注释行\n脆脆鲨|翠翠沙\n\n舰长\n九宫格")
    assert n == 3, f"应导入 3 个，实际 {n}"
    assert st.size == 3


def test_keyword_store_rejects_too_short(tmp_path):
    from core.keywords import KeywordStore

    st = KeywordStore(tmp_path / "keywords")
    with pytest.raises(ValueError):
        st.add("的")


def test_keyword_correction_fixes_alias_in_sentence(tmp_path):
    """回归：别名出现在句子中间时也应被替换。"""
    from core.keywords import KeywordStore, correct_with_keywords

    st = KeywordStore(tmp_path / "kw")
    st.add("脆脆鲨", alias=["翠翠沙"])
    subs = [_line("今天翠翠沙来啦", 0.0, 2.0), _line("完全无关的句子", 2.0, 4.0)]
    out, changes = correct_with_keywords(subs, st)
    assert len(changes) == 1
    assert out[0].text == "今天脆脆鲨来啦"
    assert out[1].text == "完全无关的句子"


def test_keyword_correction_keeps_timeline(tmp_path):
    from core.keywords import KeywordStore, correct_with_keywords

    st = KeywordStore(tmp_path / "kw")
    st.add("脆脆鲨", alias=["翠翠沙"])
    subs = [_line("翠翠沙来了", 1.5, 3.5)]
    out, _ = correct_with_keywords(subs, st)
    assert out[0].start == pytest.approx(1.5)
    assert out[0].end == pytest.approx(3.5)


# ---------------- 行级样式 / 应用到所有字幕 ----------------

def test_subtitle_line_supports_style_override():
    from core.models import SubtitleLine, SubtitleWord

    ln = SubtitleLine(words=[SubtitleWord(text="甲", start=0, end=1)])
    assert ln.style == {}, "默认无行级覆盖"
    ln.style = {"color": "#FF0000", "size": 90}
    assert ln.style["size"] == 90


def test_ass_color_is_bgr_order():
    """回归：ASS 颜色是 BGR 顺序，直接套用 RGB 会让红蓝互换。"""
    from core.subtitle import _ass_color

    assert _ass_color("#FFFFFF") == "&H00FFFFFF"
    assert _ass_color("#FF0000") == "&H000000FF", "红色应落在 B 通道位置"
    assert _ass_color("#0000FF") == "&H00FF0000"


def test_ass_style_reflects_preview_settings():
    """回归：烧录样式必须与预览一致（字号/颜色/描边/对齐）。"""
    from core.subtitle import to_ass

    lines = [_line("测试字幕", 0.0, 2.0)]
    ass = to_ass(lines, style={"size": 72, "color": "#FFD700",
                              "outline": 5, "anchor": "middle", "bg": True})
    style_line = next(ln for ln in ass.splitlines()
                      if ln.startswith("Style: Default"))
    assert ",72," in style_line, "字号应写入样式"
    fields = style_line.split(",")
    assert fields[3] == "&H0000D7FF", "颜色应为 BGR 转换结果"
    # bg=True → BorderStyle=3（不透明背景框）；Alignment=5（居中）
    assert fields[15] == "3" and fields[18] == "5"


def test_ass_inline_override_for_single_line():
    """回归：行级样式应通过内联标签覆盖全局样式。"""
    from core.subtitle import to_ass

    ln = _line("特殊行", 0.0, 2.0)
    ln.style = {"color": "#FF0000", "size": 110}
    ass = to_ass([ln], style={"size": 72, "color": "#FFFFFF"})
    dialogue = next(x for x in ass.splitlines() if x.startswith("Dialogue:"))
    assert "\\fs110" in dialogue, "行级字号应内联"
    assert "\\c&H000000FF" in dialogue, "行级颜色应内联"


# ---------------- 数据脱敏 ----------------

def test_desensitize_script_detects_common_leaks(tmp_path):
    """回归：脱敏脚本必须能识别常见敏感信息。"""
    import subprocess

    root = Path(__file__).resolve().parent.parent
    script = root / "tools" / "desensitize.py"
    assert script.is_file(), "缺少脱敏脚本"
    src = script.read_text(encoding="utf-8")
    for kw in ("Downloads", "gh[pousr]_", "Sample", "sensitive.txt"):
        assert kw in src, f"脱敏规则缺少 {kw}"
    proc = subprocess.run([sys.executable, str(script)],
                          capture_output=True, timeout=180,
                          # 脚本输出含中文，Windows 默认 GBK 会解码失败
                          encoding="utf-8", errors="replace")
    assert proc.returncode in (0, 1), "脚本应正常结束"


def test_pre_push_hook_runs_desensitize():
    """回归：pre-push 钩子应调用脱敏脚本（推送前强制检查）。"""
    hook = Path(__file__).resolve().parent.parent / ".git" / "hooks" / "pre-push"
    if not hook.is_file():
        pytest.skip("无 pre-push 钩子（非 git 环境）")
    text = hook.read_text(encoding="utf-8", errors="replace")
    assert "desensitize.py" in text, "钩子必须运行脱敏检查"


# ---------------- 弹幕 XML ----------------

XML = """<?xml version="1.0" encoding="UTF-8"?>
<i>
  <BililiveRecorderRecordInfo roomid="12345" name="测试主播" title="测试标题"
      start_time="2026-10-07T20:09:36.2003498+08:00" />
  <d p="12.34,1,25,16777215,1696000000000,0,abc123,0" user="用户甲">主播晚上好</d>
  <d p="15.00,1,25,16777215,1696000003000,0,def456,0" user="用户乙">点一首七里香</d>
  <gift ts="39.943" user="用户丙" uid="ghi789" giftname="粉丝团灯牌" giftcount="2" />
  <sc ts="211.5" user="用户丁" uid="jkl012" price="50" time="120">想听《发如雪》</sc>
  <guard ts="112.018" user="用户戊" uid="mno345" level="3" count="1" />
</i>
"""


def test_parse_xml_basic(tmp_path):
    """真实录播姬格式：昵称在 user 属性、礼物/SC/舰长是独立节点。"""
    p = tmp_path / "danmaku.xml"
    p.write_text(XML, encoding="utf-8")
    log = parse_xml(p)
    assert len(log.danmaku) == 2
    assert log.danmaku[0].time == pytest.approx(12.34)
    assert log.danmaku[0].text == "主播晚上好"
    assert log.danmaku[0].user_id == "abc123"
    # 昵称取自 user 属性（旧实现误取 p 的第 8 段，恒为 0）
    assert log.danmaku[0].user_name == "用户甲"
    # 礼物/SC/舰长分别解析
    kinds = sorted(g.kind for g in log.gifts)
    assert kinds == ["gift", "guard", "sc"]
    gift = next(g for g in log.gifts if g.kind == "gift")
    assert gift.gift == "粉丝团灯牌" and gift.count == 2
    sc = next(g for g in log.gifts if g.kind == "sc")
    assert sc.price == 50 and "发如雪" in sc.text
    guard = next(g for g in log.gifts if g.kind == "guard")
    assert guard.gift == "舰长"
    # 录制信息
    assert log.streamer == "测试主播" and log.title == "测试标题"
    assert log.room_id == "12345" and log.started_at > 0


def test_song_requests_from_danmaku_and_sc(tmp_path):
    """点歌请求：弹幕与 SC 都能提取出歌名（这是最可靠的歌曲来源）。"""
    p = tmp_path / "danmaku.xml"
    p.write_text(XML, encoding="utf-8")
    log = parse_xml(p)
    reqs = log.song_requests()
    names = [n for _t, _u, n in reqs]
    assert "七里香" in names
    assert "发如雪" in names


def test_hotwords_include_user_names(tmp_path):
    p = tmp_path / "danmaku.xml"
    p.write_text(XML, encoding="utf-8")
    log = parse_xml(p)
    words = log.hotwords(top=20)
    assert "用户甲" in words or "用户乙" in words


def test_parse_xml_repairs_truncated_file(tmp_path):
    """录制中断产生的截断 XML 也要能解析（保留弹幕与录制信息）。"""
    p = tmp_path / "broken.xml"
    p.write_text(XML.rsplit("</i>", 1)[0], encoding="utf-8")
    log = parse_xml(p)
    assert len(log.danmaku) == 2
    assert len(log.gifts) == 3
    # 截断修复时不应丢掉 RecordInfo（主播名/标题）
    assert log.streamer == "测试主播"


def test_hotwords_exclude_stopwords():
    from core.livexml import Danmaku, LiveLog

    log = LiveLog(danmaku=[
        Danmaku(1.0, "哈哈哈哈", user_name="观众A"),
        Danmaku(2.0, "主播唱得好", user_name="铁粉老王"),
        Danmaku(3.0, "铁粉老王 送你 盲盒 x2"),
    ])
    words = log.hotwords(top=10)
    assert "铁粉老王" in words
    assert "哈哈哈哈" not in words


# ---------------- 配置 ----------------

def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("SUBTITLE_STUDIO_ASR_MODEL", "medium")
    monkeypatch.setenv("SUBTITLE_STUDIO_VAD", "1")
    cfg = AppConfig.from_env()
    assert cfg.asr.model == "medium"
    assert cfg.asr.vad_filter is True


def test_config_nested_override():
    cfg = AppConfig.from_env(segment={"target_chars": 12.0})
    assert cfg.segment.target_chars == 12.0
    assert cfg.segment.max_chars == 17      # 其他字段保持默认


def test_config_resolve_makes_dirs_absolute():
    cfg = AppConfig.from_env().resolve()
    assert Path(cfg.work_dir).is_absolute()
    assert Path(cfg.lyrics_dir).is_absolute()


def test_all_config_field_references_exist():
    """静态自检：pipeline/ui 里引用的 cfg.<section>.<field> 必须真实存在。

    这类拼写错误在运行时才炸（曾导致歌曲识别分支 AttributeError），
    用一条静态测试把它挡在提交前。
    """
    import re
    from dataclasses import fields as dc_fields

    from core.config import (AsrConfig, ProviderConfig, SegmentConfig,
                             SongConfig, TimingConfig, VoiceprintConfig)

    sections = {
        "segment": SegmentConfig, "asr": AsrConfig, "timing": TimingConfig,
        "song": SongConfig, "voiceprint": VoiceprintConfig,
        "provider": ProviderConfig,
    }
    app_dir = ROOT / "app"
    bad = []
    pattern = re.compile(r"\.(segment|asr|timing|song|voiceprint|provider)\.(\w+)")
    for src in list(app_dir.glob("*.py")) + list(app_dir.glob("core/*.py")):
        text = src.read_text(encoding="utf-8", errors="replace")
        for section, attr in pattern.findall(text):
            valid = {f.name for f in dc_fields(sections[section])}
            if attr not in valid:
                bad.append(f"{src.name}: {section}.{attr}")
    assert not bad, "未知配置字段：" + ", ".join(bad)


# ---------------- 外部接口 ----------------

def test_external_response_parsing():
    from core.providers.http_provider import parse_external_response

    data = {"segments": [
        {"start": 1.0, "end": 2.0, "text": "你好",
         "words": [{"text": "你", "start": 1.0, "end": 1.5},
                   {"text": "好", "start": 1.5, "end": 2.0}]},
        {"start": 3.0, "end": 4.0, "text": "再见"},   # 无词级
    ]}
    segs = parse_external_response(data)
    assert len(segs) == 2
    assert segs[0].words[0].text == "你"
    assert len(segs[1].words) == 2          # 自动按字均分


def test_external_response_rejects_unknown_shape():
    from core.providers.http_provider import parse_external_response
    from core.providers import AsrError

    with pytest.raises(AsrError):
        parse_external_response({"unexpected": 1})


# ---------------- 歌词/歌曲工具 ----------------

def test_lrc_rows_parses_timestamps():
    from core.song import lrc_rows

    rows = lrc_rows("[00:12.34]第一行\n[00:15.00]第二行\n[ti:标题]\n")
    assert len(rows) == 2
    assert rows[0][0] == pytest.approx(12.34)
    assert rows[1][1] == "第二行"


def test_plain_rows_drops_credits():
    from core.song import plain_rows

    text = "[00:00.00] 词 : 张三\n[00:01.00] 曲 : 李四\n[00:21.95]垂柳斜向茉莉雨\n"
    assert plain_rows(text, synced=True) == ["垂柳斜向茉莉雨"]


def test_layout_score_prefers_matching_order():
    from core.song import layout_score

    transcript = "甲乙丙丁戊己庚辛"
    same = "甲乙丙丁\n戊己庚辛\n"
    shuffled = "戊己庚辛\n甲乙丙丁\n"
    assert layout_score(transcript, same) >= layout_score(transcript, shuffled)


# ---------------- 歌词多源链 ----------------

def test_lyric_providers_registry():
    from core.song import (DEFAULT_PROVIDER_ORDER, PROVIDERS,
                           available_lyric_providers)

    names = available_lyric_providers()
    # 内置国内音源必须可用（需求：测试阶段涵盖国内音源）
    assert "netease" in names
    assert "kugou" in names
    for name in DEFAULT_PROVIDER_ORDER:
        assert name in PROVIDERS


def test_custom_lyric_endpoint_registration():
    from core.song import (PROVIDERS, CustomLyricProvider,
                           set_custom_endpoint)

    set_custom_endpoint("http://127.0.0.1:1/lyrics", "key")
    assert isinstance(PROVIDERS["custom"], CustomLyricProvider)
    # 无有效端点时检索应安全返回空（不抛异常）
    assert PROVIDERS["custom"].search(title="x") == []


def test_search_lyrics_skips_unregistered_provider():
    from core.song import search_lyrics

    logs = []
    rows = search_lyrics(title="不存在", providers=["not_a_provider"],
                         on_log=logs.append)
    assert rows == []
    assert any("未注册" in m for m in logs)


def test_artist_mismatch_detection():
    from core.song import _artist_mismatch

    assert _artist_mismatch("周杰伦", "Xai小爱") is True
    assert _artist_mismatch("周杰伦", "周杰伦") is False
    assert _artist_mismatch("周杰伦", "周杰伦、费玉清") is False
    # 空值不判为不匹配（避免误降权）
    assert _artist_mismatch("", "某人") is False
    assert _artist_mismatch("某人", "") is False


# ---------------- BGM 多因子判别 ----------------

def test_music_presence_rises_with_bass_and_flatness():
    from core.bgm import SegmentFeatures, music_presence

    clean = SegmentFeatures(start=0, end=3, bass_ratio=0.05,
                            high_ratio=0.03, flatness=0.03,
                            side_ratio=0.05)
    with_music = SegmentFeatures(start=0, end=3, bass_ratio=0.35,
                                 high_ratio=0.20, flatness=0.35,
                                 side_ratio=0.35)
    assert music_presence(clean) < music_presence(with_music)
    assert 0.0 <= music_presence(clean) <= 1.0
    assert 0.0 <= music_presence(with_music) <= 1.0


def test_bgm_decide_keeps_streamer_rejects_backing_vocal():
    """声纹高的现场段保留；声纹低 + 伴奏明显的段剔除。"""
    from core.bgm import BgmTuning, SegmentFeatures, decide

    cfg = BgmTuning()
    live = SegmentFeatures(start=0, end=3, bass_ratio=0.08, high_ratio=0.04,
                           flatness=0.05, side_ratio=0.08, loud_cv=0.6)
    bgm = SegmentFeatures(start=0, end=3, bass_ratio=0.32, high_ratio=0.18,
                          flatness=0.32, side_ratio=0.30, loud_cv=0.4)
    assert decide(live, 0.78, cfg).keep is True
    assert decide(bgm, 0.28, cfg).keep is False


def test_bgm_decide_marks_reasons():
    from core.bgm import BgmTuning, SegmentFeatures, decide

    feat = SegmentFeatures(start=0, end=3, bass_ratio=0.32, flatness=0.35,
                           side_ratio=0.4)
    dec = decide(feat, 0.25, BgmTuning())
    assert dec.reasons, "被剔除的段必须给出可复核的原因"
    assert dec.vocal_sim == pytest.approx(0.25)


def test_split_windows_covers_segments():
    """子窗切分必须完整覆盖原段（不能漏掉音频内容）。"""
    from core.pipeline2 import _split_windows

    segs = [(0.0, 12.0), (20.0, 22.5)]
    wins = _split_windows(segs, window=5.0, min_window=1.0)
    for s, e in segs:
        covered = sum(min(e, we) - max(s, ws) for ws, we in wins
                      if we > s and ws < e)
        assert covered == pytest.approx(e - s, abs=0.01)


def test_merge_windows_bridges_small_gaps():
    from core.pipeline2 import _merge_windows

    wins = [(0.0, 5.0), (5.2, 10.0), (12.0, 15.0)]
    merged = _merge_windows(wins, [True, True, True], gap=0.35)
    assert merged == [(0.0, 10.0), (12.0, 15.0)]


def test_merge_windows_drops_kept_false():
    from core.pipeline2 import _merge_windows

    wins = [(0.0, 5.0), (5.0, 10.0)]
    assert _merge_windows(wins, [True, False], gap=0.35) == [(0.0, 5.0)]
    assert _merge_windows(wins, [False, False], gap=0.35) == []


# ---------------- 声纹（单目标） ----------------

def test_streamer_profile_refs():
    import numpy as np

    from core.streamer import StreamerProfile

    v = np.ones(4, dtype=np.float32) / 2
    p = StreamerProfile(name="主播", speak=v)
    assert len(p.refs()) == 1
    p2 = StreamerProfile(name="主播", speak=v, sing=v)
    assert [lab for lab, _ in p2.refs()] == ["speak", "sing"]
    assert p.empty is False
    assert StreamerProfile(name="x").empty is True


def test_streamer_config_thresholds():
    from core.streamer import StreamerConfig

    cfg = StreamerConfig()
    # 唱歌阈值必须低于说话阈值（同一人唱歌音色漂移更大）
    assert cfg.sing_threshold < cfg.speak_threshold


def test_classify_marks_non_streamer_when_below_threshold(monkeypatch):
    """相似度低于阈值的段应被判为「非主播」。"""
    import numpy as np

    import core.streamer as st

    # 用替身绕开真实音频读取与模型加载
    monkeypatch.setattr(st, "_load_span",
                        lambda _p, s, e: np.zeros(int((e - s) * 16000),
                                                  dtype=np.float32))
    monkeypatch.setattr(st.VoiceprintEngine, "embed_wavs",
                        lambda self, wavs: np.tile(
                            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                            (len(wavs), 1)))
    prof = st.StreamerProfile(name="主播", speak=np.array(
        [0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    res = st.classify(st.VoiceprintEngine("cpu"), "unused.wav", [(0.0, 3.0)],
                      prof, st.StreamerConfig(vote_ratio=0.5))
    assert isinstance(res[0], st.SegmentMatch)
    assert res[0].is_streamer is False


def test_classify_accepts_matching_streamer(monkeypatch):
    """相似度高于阈值的段应被判为「主播」。"""
    import numpy as np

    import core.streamer as st

    monkeypatch.setattr(st, "_load_span",
                        lambda _p, s, e: np.zeros(int((e - s) * 16000),
                                                  dtype=np.float32))
    monkeypatch.setattr(st.VoiceprintEngine, "embed_wavs",
                        lambda self, wavs: np.tile(
                            np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                            (len(wavs), 1)))
    prof = st.StreamerProfile(name="主播", speak=np.array(
        [0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    res = st.classify(st.VoiceprintEngine("cpu"), "unused.wav", [(0.0, 3.0)],
                      prof, st.StreamerConfig(vote_ratio=0.5, adaptive=False))
    assert res[0].is_streamer is True


# ---------------- 歌曲强制对齐工具 ----------------

def test_lines_from_alignment_splits_by_lyric_length():
    from core.song import lines_from_alignment

    chars = [(i * 0.1, (i + 1) * 0.1, c) for i, c in enumerate("abcd")]
    lines = lines_from_alignment(chars, ["ab", "cd"])
    assert [ln.text for ln in lines] == ["ab", "cd"]
    assert lines[0].start == pytest.approx(0.0)


def test_lines_from_alignment_truncates_when_short():
    from core.song import lines_from_alignment

    chars = [(0.0, 0.1, "a")]
    lines = lines_from_alignment(chars, ["ab", "cd"])
    assert len(lines) == 1


def test_score_similarity_identical_is_one():
    from core.song import score_similarity

    assert score_similarity("雨下整夜", "雨下整夜") == 1.0
    assert score_similarity("雨下整夜", "完全不同") < 0.3


# ---------------- 流水线收尾（时间轴合法性） ----------------

def test_finalize_removes_zero_duration_and_overlap():
    """退化输入（零时长词、临界重叠）经收尾后必须时长>0 且单调不重叠。"""
    from core.config import AppConfig
    from core.pipeline2 import Pipeline, PipelineRequest

    lines = [
        SubtitleLine(words=[SubtitleWord("甲", 1.0, 1.05)]),
        SubtitleLine(words=[SubtitleWord("乙", 1.05, 1.05)]),   # 零时长
        SubtitleLine(words=[SubtitleWord("丙", 1.05, 1.10)]),
        SubtitleLine(words=[SubtitleWord("丁", 1.08, 2.0)]),    # 重叠
    ]
    pipe = Pipeline(AppConfig.from_env(), PipelineRequest(input_path="x"))
    pipe._finalize(lines)
    assert all(ln.end > ln.start for ln in lines)
    for a, b in zip(lines, lines[1:]):
        assert a.end <= b.start + 1e-9


def test_finalize_dedupes_repeated_sentence():
    """短间隔重复的同一句应被去重（whisper 在歌声/回声上的常见现象）。"""
    from core.config import AppConfig
    from core.pipeline2 import Pipeline, PipelineRequest

    lines = [
        SubtitleLine(words=[SubtitleWord("晚上好晚上好", 1.0, 2.0)]),
        SubtitleLine(words=[SubtitleWord("晚上好晚上好", 2.5, 3.5)]),  # 重复
        SubtitleLine(words=[SubtitleWord("晚上好晚上好", 300.0, 301.0)]),  # 远处副歌保留
    ]
    pipe = Pipeline(AppConfig.from_env(), PipelineRequest(input_path="x"))
    pipe._finalize(lines)
    assert len(lines) == 2
    assert lines[-1].start == 300.0
