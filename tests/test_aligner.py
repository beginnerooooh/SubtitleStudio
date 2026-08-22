"""core/aligner.py 单测：窗口规划/span 提取/分块/行分配/去重纯函数 + 假模型接缝的全流程。"""
import threading

import numpy as np
import pytest
import soundfile as sf
import torch

import core.aligner as al_mod
from core.aligner import (
    AlignmentError,
    Aligner,
    AlignResult,
    TokenSpan,
    _assign_lines_to_chunks,
    _build_chunks,
    _choose_candidate,
    _expand_targets,
    _plan_windows,
    _spans_from_alignments,
)
from core.errors import TaskCancelled
from core.models import SubtitleLine, SubtitleWord
from core.text import Token, prepare_lyrics

ZH_VOCAB = {c: i + 1 for i, c in enumerate("你好世界再见")}
EN_VOCAB = {c: i + 1 for i, c in enumerate("helowrd")}


def _fake_get_models_zh(language, device):
    return (None, None, dict(ZH_VOCAB), 0, None, len(ZH_VOCAB) + 1)


def _fake_forward(waveform, model, processor, device):
    """T = len(waveform)//320 帧，内容无意义（forced_align 也被替换）。"""
    return torch.full((len(waveform) // 320, 32), -5.0)


def _fake_forced_align(frame_per_char=10, hold=6, score=0.9, low_scores=None):
    def fake(log_probs, char_ids, blank_id):
        spans = []
        for i in range(len(char_ids)):
            s = low_scores.get(i, score) if low_scores else score
            spans.append(TokenSpan(i * frame_per_char, i * frame_per_char + hold, s))
        return spans

    return fake


@pytest.fixture
def wav_factory(tmp_path):
    def make(seconds: float, sr: int = 16000) -> str:
        path = tmp_path / f"t{seconds}_{sr}.wav"
        sf.write(path, np.zeros(int(seconds * sr)), sr)
        return str(path)

    return make


@pytest.fixture(autouse=True)
def _reset_model_cache():
    al_mod.reset_model_cache()
    yield
    al_mod.reset_model_cache()


def _patch_seams(monkeypatch, get_models=None, vad=None):
    monkeypatch.setattr(al_mod, "_get_models", get_models or _fake_get_models_zh)
    monkeypatch.setattr(al_mod, "_forward", _fake_forward)
    if vad is not None:
        monkeypatch.setattr(al_mod, "_vad_intervals", lambda path: vad)


class TestPlanWindows:
    def test_short_audio_single_window_no_trim(self):
        assert _plan_windows(2.0) == [(0.0, 2.0, 0.0, 2.0)]

    def test_multi_window_trims_one_second_each_side(self):
        assert _plan_windows(33.0) == [
            (0.0, 32.0, 0.0, 31.0),
            (30.0, 33.0, 31.0, 33.0),
        ]

    def test_kept_ranges_tile_timeline(self):
        for duration in (31.0, 61.0, 62.0, 63.0, 65.0, 100.0, 121.7):
            kept = [(ks, ke) for _, _, ks, ke in _plan_windows(duration)]
            assert kept[0][0] == 0.0
            assert kept[-1][1] == duration
            for (_, prev_ke), (cur_ks, _) in zip(kept, kept[1:]):
                assert prev_ke == cur_ks  # 无缝拼接

    def test_empty_keep_window_skipped(self):
        # 31s：第二个窗口保留区间为空 → 只剩首窗口（首窗口即末窗口不裁剪）
        assert _plan_windows(31.0) == [(0.0, 31.0, 0.0, 31.0)]


class TestSpansFromAlignments:
    def test_basic_runs_with_blanks(self):
        spans = _spans_from_alignments([0, 1, 1, 0, 2, 0, 2, 2, 3], blank_id=0)
        assert [(s.start, s.end) for s in spans] == [(1, 3), (4, 5), (6, 8), (8, 9)]

    def test_adjacent_different_tokens_split(self):
        spans = _spans_from_alignments([1, 2], blank_id=0)
        assert [(s.start, s.end) for s in spans] == [(0, 1), (1, 2)]

    def test_repeated_without_blank_is_one_token(self):
        spans = _spans_from_alignments([1, 1, 1], blank_id=0)
        assert [(s.start, s.end) for s in spans] == [(0, 3)]

    def test_trailing_run_closed(self):
        spans = _spans_from_alignments([0, 1, 1], blank_id=0)
        assert [(s.start, s.end) for s in spans] == [(1, 3)]

    def test_all_blank_empty(self):
        assert _spans_from_alignments([0, 0, 0], blank_id=0) == []

    def test_scores_attached_in_order(self):
        spans = _spans_from_alignments([1, 0, 2], blank_id=0, scores=[0.9, 0.4])
        assert [s.score for s in spans] == [0.9, 0.4]


class TestBuildChunks:
    def test_merges_until_target(self):
        vad = [(0.0, 4.0), (5.0, 7.0), (8.0, 20.0)]
        assert _build_chunks(vad, target=10.0) == [(0.0, 7.0), (8.0, 20.0)]

    def test_splits_when_exceeding_target(self):
        vad = [(0.0, 4.0), (6.0, 12.0)]
        assert _build_chunks(vad, target=5.0) == [(0.0, 4.0), (6.0, 12.0)]

    def test_single_long_interval_is_own_chunk(self):
        assert _build_chunks([(0.0, 700.0)], target=600.0) == [(0.0, 700.0)]

    def test_empty_vad(self):
        assert _build_chunks([], target=600.0) == []


class TestAssignLines:
    def test_proportional_with_overlap(self):
        lines = [[Token(c, c)] for c in "ABCDEFGHIJ"]  # 10 行各 1 字
        vad = [(0.0, 4.0), (6.0, 12.0)]
        chunks = [(0.0, 4.0), (6.0, 12.0)]
        # 语音时长：块0=4s，块1=6s，总 10s；行中心（语音时间）0.5,1.5,...,9.5
        # → 基础分配：块0 行0-3，块1 行4-9；±2 重叠 → (0,6) 与 (2,10)
        ranges = _assign_lines_to_chunks(lines, vad, chunks, overlap=2)
        assert ranges == [(0, 6), (2, 10)]

    def test_all_lines_covered_monotonic(self):
        lines = [[Token(c, c)] for c in "ABCDEFGH"]
        vad = [(0.0, 3.0), (4.0, 6.0), (7.0, 10.0)]
        chunks = _build_chunks(vad, target=4.0)
        ranges = _assign_lines_to_chunks(lines, vad, chunks, overlap=2)
        covered = set()
        for lo, hi in ranges:
            covered.update(range(lo, hi))
        assert covered == set(range(8))
        # 基础分配行号随块单调不减
        assert [lo for lo, _ in ranges] == sorted(lo for lo, _ in ranges)

    def test_empty_chunk_gets_empty_range(self):
        lines = [[Token(c, c)] for c in "AB"]
        vad = [(0.0, 10.0), (20.0, 21.0)]  # 块1 语音仅 1s
        chunks = _build_chunks(vad, target=5.0)  # [(0,10),(20,21)]
        ranges = _assign_lines_to_chunks(lines, vad, chunks, overlap=2)
        # 2 行按语音比例（10:1）都落入块0；块1 基础为空 → (0,0)
        assert ranges[0] == (0, 2)
        assert ranges[1] == (0, 0)


class TestChooseCandidate:
    def test_prefers_farther_from_chunk_edge(self):
        near = SubtitleLine(words=[SubtitleWord("A", 6.0, 6.2)])  # 块 (6,12) 边缘
        far = SubtitleLine(words=[SubtitleWord("A", 0.0, 3.0)])   # 块 (0,4) 中央
        best = _choose_candidate([(near, 6.0, 12.0), (far, 0.0, 4.0)])
        assert best is far

    def test_single_candidate_returned(self):
        line = SubtitleLine(words=[SubtitleWord("A", 0.0, 1.0)])
        assert _choose_candidate([(line, 0.0, 4.0)]) is line


class TestExpandTargets:
    def test_chinese_chars_no_separator(self):
        lines = [[Token("你", "你"), Token("好", "好")]]
        refs, ids = _expand_targets(lines, {"你": 1, "好": 2}, sep_id=None, unk_id=9)
        assert ids == [1, 2]
        assert refs == [(0, 0), (0, 1)]

    def test_english_words_separated(self):
        lines = [[Token("hello", "hello"), Token("world", "world")]]
        refs, ids = _expand_targets(lines, EN_VOCAB, sep_id=8, unk_id=9)
        # hello(5字) + sep + world(5字)；o 的 id 是 4
        assert ids == [1, 2, 3, 3, 4, 8, 5, 4, 6, 3, 7]
        assert refs == [(0, 0)] * 5 + [None] + [(0, 1)] * 5

    def test_unknown_char_falls_back_to_unk(self):
        lines = [[Token("×", "×")]]
        _, ids = _expand_targets(lines, {"你": 1}, sep_id=None, unk_id=9)
        assert ids == [9]

    def test_uppercase_vocab_fallback(self):
        """wav2vec2 英文模型词表仅含大写：小写输入应回退命中大写 id。"""
        vocab = {"<pad>": 0, "|": 4, "H": 11, "I": 12}
        lines = [[Token("hi", "hi")]]
        _, ids = _expand_targets(lines, vocab, sep_id=4, unk_id=3)
        assert ids == [11, 12]

    def test_uppercase_fallback_never_replaces_exact_hit(self):
        """词表同时含大小写时精确命中优先，不做大小写改写。"""
        vocab = {"a": 1, "A": 2}
        lines = [[Token("aA", "aA")]]
        _, ids = _expand_targets(lines, vocab, sep_id=None, unk_id=9)
        assert ids == [1, 2]


class TestAlignGlobal:
    def test_short_audio_global_alignment(self, monkeypatch, wav_factory):
        _patch_seams(monkeypatch)
        monkeypatch.setattr(al_mod, "_forced_align", _fake_forced_align())
        a = Aligner()
        lyrics = prepare_lyrics("你好，世界。\n再见！")
        result = a.align(wav_factory(2.0), lyrics)
        assert isinstance(result, AlignResult)
        assert [ln.text for ln in result.lines] == ["你好，世界。", "再见！"]
        assert result.lines[0].start == pytest.approx(0.0)
        assert result.lines[0].end == pytest.approx(0.72)
        assert result.lines[1].start == pytest.approx(0.8)
        assert result.lines[1].end == pytest.approx(1.12)
        # 词级：好=字2 → 帧(10,16) → 0.20–0.32
        assert result.lines[0].words[1].start == pytest.approx(0.20)
        assert result.lines[0].words[1].end == pytest.approx(0.32)
        assert result.low_confidence == []

    def test_english_word_level_merge(self, monkeypatch, wav_factory):
        monkeypatch.setattr(
            al_mod, "_get_models",
            lambda lang, dev: (None, None, dict(EN_VOCAB), 0, 8, 9),
        )
        monkeypatch.setattr(al_mod, "_forward", _fake_forward)
        monkeypatch.setattr(al_mod, "_forced_align", _fake_forced_align(frame_per_char=2, hold=1))
        a = Aligner(language="en")
        result = a.align(wav_factory(1.0), prepare_lyrics("hello\nworld"))
        # hello：字0-4 → 帧(0,9) → 0.00–0.18；world：字6-10 → 帧(12,21) → 0.24–0.42
        assert [ln.text for ln in result.lines] == ["hello", "world"]
        assert result.lines[0].start == pytest.approx(0.0)
        assert result.lines[0].end == pytest.approx(0.18)
        assert result.lines[1].start == pytest.approx(0.24)
        assert result.lines[1].end == pytest.approx(0.42)

    def test_low_confidence_lines_reported(self, monkeypatch, wav_factory):
        _patch_seams(monkeypatch)
        monkeypatch.setattr(
            al_mod, "_forced_align",
            _fake_forced_align(low_scores={0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}),
        )
        a = Aligner()
        result = a.align(wav_factory(2.0), prepare_lyrics("你好世界\n再见"))
        assert result.low_confidence == [0]

    def test_progress_ends_at_one(self, monkeypatch, wav_factory):
        _patch_seams(monkeypatch)
        monkeypatch.setattr(al_mod, "_forced_align", _fake_forced_align())
        ratios = []
        a = Aligner(on_progress=ratios.append)
        a.align(wav_factory(2.0), prepare_lyrics("你好"))
        assert ratios[-1] == pytest.approx(1.0)
        assert ratios == sorted(ratios)

    def test_cancel_raises(self, monkeypatch, wav_factory):
        _patch_seams(monkeypatch)
        event = threading.Event()
        event.set()
        a = Aligner(cancel_event=event)
        with pytest.raises(TaskCancelled):
            a.align(wav_factory(2.0), prepare_lyrics("你好"))

    def test_empty_lyrics_raises(self, monkeypatch, wav_factory):
        _patch_seams(monkeypatch)
        with pytest.raises(AlignmentError, match="歌词"):
            Aligner().align(wav_factory(2.0), [])

    def test_wrong_sample_rate_raises(self, monkeypatch, wav_factory):
        _patch_seams(monkeypatch)
        with pytest.raises(AlignmentError, match="16000"):
            Aligner().align(wav_factory(2.0, sr=44100), prepare_lyrics("你好"))


class TestAlignChunked:
    def test_long_audio_vad_chunking_and_dedup(self, monkeypatch, wav_factory):
        _patch_seams(monkeypatch, vad=[(0.0, 4.0), (6.0, 12.0)])
        monkeypatch.setattr(al_mod, "_forced_align", _fake_forced_align(frame_per_char=20, hold=10))
        monkeypatch.setattr(al_mod, "_GLOBAL_ALIGN_MAX", 10.0)
        monkeypatch.setattr(al_mod, "_CHUNK_TARGET", 5.0)
        lyrics = prepare_lyrics("\n".join("一二三四五六七八九十"))
        ratios = []
        a = Aligner(on_progress=ratios.append)
        result = a.align(wav_factory(12.0), lyrics)
        # 10 行全部对齐、时间单调
        assert [ln.text for ln in result.lines] == list("一二三四五六七八九十")
        starts = [ln.start for ln in result.lines]
        assert starts == sorted(starts)
        # 行2 在两块中重复：块0（局部字2→帧40-50→0.8-1.0）距边缘更远，胜出
        assert result.lines[2].start == pytest.approx(0.8)
        assert result.lines[2].end == pytest.approx(1.0)
        # 行9 只在块1（局部字7→帧140-150→6.0+2.8=8.8）
        assert result.lines[9].start == pytest.approx(8.8)
        assert result.lines[9].end == pytest.approx(9.0)
        assert ratios[-1] == pytest.approx(1.0)

    def test_no_vad_falls_back_to_single_chunk(self, monkeypatch, wav_factory):
        _patch_seams(monkeypatch, vad=[])
        monkeypatch.setattr(al_mod, "_forced_align", _fake_forced_align())
        monkeypatch.setattr(al_mod, "_GLOBAL_ALIGN_MAX", 10.0)
        a = Aligner()
        result = a.align(wav_factory(12.0), prepare_lyrics("你好"))
        assert [ln.text for ln in result.lines] == ["你好"]
