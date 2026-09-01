"""core/voiceprint.py 单测：Profile 存取、相似度判定、区域合并（模型接缝 mock）。"""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import core.voiceprint as vp
from core.voiceprint import (
    SegmentVerdict,
    VoiceProfile,
    VoiceprintEngine,
    VoiceprintError,
    list_profiles,
    load_profile,
    merge_regions,
    save_profile,
)

SR = 16000


def _write_pattern_wav(path, patterns_per_second, sr=SR):
    """写一个「每秒一个 4 样本周期 pattern」的 16k mono WAV。

    任意整秒起点的前 4 个样本 == 该秒的 pattern → 可驱动假 _embed。
    """
    blocks = [
        np.tile(np.asarray(pat, dtype=np.float32), sr // 4)
        for pat in patterns_per_second
    ]
    sf.write(str(path), np.concatenate(blocks), sr, subtype="PCM_16")
    return str(path)


@pytest.fixture
def fake_embed(monkeypatch):
    """_embed 接缝：embedding = L2 归一化(前 4 样本)。"""
    def _fake(self, wavs):
        arr = np.stack([np.asarray(w[:4], dtype=np.float32) for w in wavs])
        return vp._normalize(arr)
    monkeypatch.setattr(VoiceprintEngine, "_embed", _fake)


@pytest.fixture
def passthrough_extract(monkeypatch):
    """save_profile 的样本转 wav 直接透传（测试样本本身就是 16k wav）。"""
    monkeypatch.setattr(vp, "extract_wav", lambda src, dst_dir, **kw: Path(src))


def _profile(name="测试主播", speak=(1, 0, 0, 0), sing=(0, 1, 0, 0)):
    return VoiceProfile(
        name=name,
        speak=vp._normalize(np.asarray(speak, dtype=np.float32)) if speak else None,
        sing=vp._normalize(np.asarray(sing, dtype=np.float32)) if sing else None,
    )


class TestProfileIO:
    def test_save_load_roundtrip_dual(self, tmp_path, fake_embed, passthrough_extract):
        speak = _write_pattern_wav(tmp_path / "s.wav", [[1, 0, 0, 0]])
        sing = _write_pattern_wav(tmp_path / "g.wav", [[0, 1, 0, 0]])
        out = save_profile("主播A", speak, sing, profiles_dir=tmp_path)
        assert out == tmp_path / "主播A.npy"

        p = load_profile("主播A", profiles_dir=tmp_path)
        assert p.name == "主播A"
        assert np.allclose(p.speak, [1, 0, 0, 0])
        assert np.allclose(p.sing, [0, 1, 0, 0])

    def test_save_without_sing(self, tmp_path, fake_embed, passthrough_extract):
        speak = _write_pattern_wav(tmp_path / "s.wav", [[1, 0, 0, 0]])
        save_profile("B", speak, None, profiles_dir=tmp_path)
        p = load_profile("B", profiles_dir=tmp_path)
        assert p.speak is not None
        assert p.sing is None

    def test_overwrite_existing_profile(self, tmp_path, fake_embed, passthrough_extract):
        speak = _write_pattern_wav(tmp_path / "s.wav", [[1, 0, 0, 0]])
        save_profile("C", speak, None, profiles_dir=tmp_path)
        speak2 = _write_pattern_wav(tmp_path / "s2.wav", [[0, 0, 1, 0]])
        save_profile("C", speak2, None, profiles_dir=tmp_path)
        p = load_profile("C", profiles_dir=tmp_path)
        assert np.allclose(p.speak, [0, 0, 1, 0])

    def test_missing_speak_raises(self, tmp_path, fake_embed, passthrough_extract):
        with pytest.raises(VoiceprintError, match="说话样本"):
            save_profile("D", None, None, profiles_dir=tmp_path)

    def test_silent_sample_rejected(self, tmp_path, fake_embed, passthrough_extract):
        silent = _write_pattern_wav(tmp_path / "sil.wav", [[0, 0, 0, 0]])
        with pytest.raises(VoiceprintError, match="静音"):
            save_profile("E", silent, None, profiles_dir=tmp_path)

    def test_load_missing_raises_readable(self, tmp_path):
        with pytest.raises(VoiceprintError, match="未找到主播.*不存在"):
            load_profile("不存在", profiles_dir=tmp_path)

    def test_corrupt_profile_raises(self, tmp_path):
        (tmp_path / "坏.npy").write_bytes(b"not a numpy file")
        with pytest.raises(VoiceprintError, match="损坏"):
            load_profile("坏", profiles_dir=tmp_path)

    def test_list_profiles_sorted(self, tmp_path, fake_embed, passthrough_extract):
        speak = _write_pattern_wav(tmp_path / "s.wav", [[1, 0, 0, 0]])
        for name in ("乙", "甲", "cc"):
            save_profile(name, speak, None, profiles_dir=tmp_path)
        assert list_profiles(profiles_dir=tmp_path) == ["cc", "乙", "甲"]  # sorted: 拼音前 ascii

    def test_list_profiles_missing_dir(self, tmp_path):
        assert list_profiles(profiles_dir=tmp_path / "nope") == []


class TestNameValidation:
    @pytest.mark.parametrize("bad", ["", "   ", "/etc", "a\\b", "a:b", "a*b", 'a"b',
                                     ".hidden", "x" * 65])
    def test_invalid_names_rejected(self, bad, tmp_path, fake_embed, passthrough_extract):
        speak = _write_pattern_wav(tmp_path / "s.wav", [[1, 0, 0, 0]])
        with pytest.raises(VoiceprintError):
            save_profile(bad, speak, None, profiles_dir=tmp_path)

    def test_name_stripped(self, tmp_path, fake_embed, passthrough_extract):
        speak = _write_pattern_wav(tmp_path / "s.wav", [[1, 0, 0, 0]])
        save_profile("  小明  ", speak, None, profiles_dir=tmp_path)
        assert list_profiles(profiles_dir=tmp_path) == ["小明"]


class TestSegmentVerdict:
    def test_best_sim_takes_max(self):
        v = SegmentVerdict(0, 1, speak_sim=0.4, sing_sim=0.7)
        assert v.best_sim == pytest.approx(0.7)
        assert v.is_singing is True

    def test_speak_higher_not_singing(self):
        v = SegmentVerdict(0, 1, speak_sim=0.9, sing_sim=0.2)
        assert v.best_sim == pytest.approx(0.9)
        assert v.is_singing is False

    def test_no_sing_profile(self):
        v = SegmentVerdict(0, 1, speak_sim=0.9, sing_sim=None)
        assert v.best_sim == pytest.approx(0.9)
        assert v.is_singing is False


class TestClassifySegments:
    def test_dual_profile_classification(self, tmp_path, fake_embed):
        wav = _write_pattern_wav(tmp_path / "a.wav", [
            [1, 0, 0, 0],      # seg0: 说话声线
            [0, 1, 0, 0],      # seg1: 唱歌声线
            [0.6, 0.8, 0, 0],  # seg2: 偏唱歌
            [0, 0, 1, 0],      # seg3: 都不像
        ])
        engine = VoiceprintEngine()
        verdicts = engine.classify_segments(wav, [(0, 1), (1, 2), (2, 3), (3, 4)], _profile())
        assert len(verdicts) == 4
        assert verdicts[0].speak_sim == pytest.approx(1.0)
        assert verdicts[0].sing_sim == pytest.approx(0.0)
        assert verdicts[0].is_singing is False
        assert verdicts[1].sing_sim == pytest.approx(1.0)
        assert verdicts[1].is_singing is True
        assert verdicts[2].speak_sim == pytest.approx(0.6, abs=1e-3)
        assert verdicts[2].sing_sim == pytest.approx(0.8, abs=1e-3)
        assert verdicts[2].is_singing is True
        assert verdicts[3].best_sim == pytest.approx(0.0)

    def test_speak_only_profile(self, tmp_path, fake_embed):
        wav = _write_pattern_wav(tmp_path / "a.wav", [[1, 0, 0, 0]])
        engine = VoiceprintEngine()
        verdicts = engine.classify_segments(
            wav, [(0, 1)], _profile(sing=None))
        assert verdicts[0].sing_sim is None
        assert verdicts[0].best_sim == pytest.approx(1.0)

    def test_empty_profile_raises(self, tmp_path, fake_embed):
        wav = _write_pattern_wav(tmp_path / "a.wav", [[1, 0, 0, 0]])
        with pytest.raises(VoiceprintError, match="为空"):
            VoiceprintEngine().classify_segments(
                wav, [(0, 1)], VoiceProfile(name="x", speak=None, sing=None))

    def test_progress_reported_per_batch(self, tmp_path, fake_embed, monkeypatch):
        monkeypatch.setattr(vp, "_BATCH_SECONDS", 1.0)  # 每段一批
        wav = _write_pattern_wav(tmp_path / "a.wav", [[1, 0, 0, 0]] * 3)
        ratios = []
        VoiceprintEngine().classify_segments(
            wav, [(0, 1), (1, 2), (2, 3)], _profile(),
            on_progress=ratios.append)
        assert ratios == pytest.approx([1 / 3, 2 / 3, 1.0])

    def test_long_segment_center_cropped(self, tmp_path, fake_embed):
        # 100s 段居中截取 (35, 65)：该窗口铺唱歌 pattern → 判定唱歌
        patterns = [[1, 0, 0, 0]] * 100
        for i in range(35, 65):
            patterns[i] = [0, 1, 0, 0]
        wav = _write_pattern_wav(tmp_path / "a.wav", patterns)
        verdicts = VoiceprintEngine().classify_segments(wav, [(0, 100)], _profile())
        assert verdicts[0].is_singing is True


class TestMergeRegions:
    def test_close_segments_merged_with_padding(self):
        regions = merge_regions([(0, 1), (1.2, 2)])
        assert regions == [(0.0, 2.25)]

    def test_far_segments_stay_separate(self):
        regions = merge_regions([(0, 1), (5, 6)])
        assert regions == [(0.0, 1.25), (4.75, 6.25)]

    def test_gap_over_threshold_not_merged(self):
        regions = merge_regions([(0, 1), (2.2, 3)], gap=1.0)
        assert len(regions) == 2

    def test_tiny_region_dropped(self):
        assert merge_regions([(10, 10.3)]) == []

    def test_padding_drops_to_tiny(self):
        # 0.2s + 双侧 padding 后 0.45s < 0.5s → 丢弃
        assert merge_regions([(0, 0.2)]) == []

    def test_clamped_to_duration(self):
        regions = merge_regions([(9.9, 10.5)], duration=10.0)
        assert regions == [(9.65, 10.0)]

    def test_unsorted_input_handled(self):
        regions = merge_regions([(5, 6), (0, 1)])
        assert regions == [(0.0, 1.25), (4.75, 6.25)]

    def test_empty(self):
        assert merge_regions([]) == []


class TestCapSpan:
    def test_short_span_unchanged(self):
        assert vp._cap_span(0, 5) == (0, 5)

    def test_long_span_center_cropped(self):
        assert vp._cap_span(10, 100) == (40.0, 70.0)


class TestSpeechIntervals:
    def test_converts_sample_stamps_to_seconds(self, tmp_path, monkeypatch):
        wav = _write_pattern_wav(tmp_path / "a.wav", [[1, 0, 0, 0]] * 2)

        def fake_get_speech(audio, vad_options=None, **kwargs):
            return [{"start": SR, "end": 2 * SR}]

        monkeypatch.setattr("faster_whisper.vad.get_speech_timestamps", fake_get_speech)
        assert vp.speech_intervals(wav) == [(1.0, 2.0)]


class TestLoadSpan:
    def test_non_16k_rejected(self, tmp_path):
        path = tmp_path / "hi.wav"
        sf.write(str(path), np.zeros(44100, dtype=np.float32), 44100)
        with pytest.raises(VoiceprintError, match="16000"):
            vp._load_span(str(path), 0, 0.5)


class TestEmbedVariableLength:
    """_embed 批内变长 pad 行为（mock _get_ecapa 接缝，不碰真实模型）。"""

    def test_variable_batch_padded_to_max_with_lens(self, monkeypatch):
        import torch

        captured = {}

        class FakeModel:
            def encode_batch(self, tensor, wav_lens=None):
                captured["shape"] = tuple(tensor.shape)
                captured["lens"] = wav_lens.tolist() if wav_lens is not None else None
                # speechbrain 风格返回 (N, 1, D)；ones 保证 L2 归一化后有定义
                n = tensor.shape[0]
                return torch.ones(n, 1, 192)

        monkeypatch.setattr(vp, "_get_ecapa", lambda device: FakeModel())
        engine = VoiceprintEngine()
        wavs = [np.ones(16000, dtype=np.float32),
                np.full(8000, 0.5, dtype=np.float32)]
        out = engine._embed(wavs)
        assert captured["shape"] == (2, 16000)          # pad 到最长
        assert captured["lens"] == [1.0, 0.5]           # 相对长度
        assert out.shape == (2, 192)                     # (N, D)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)

    def test_equal_length_batch_unchanged(self, monkeypatch):
        import torch

        captured = {}

        class FakeModel:
            def encode_batch(self, tensor, wav_lens=None):
                captured["shape"] = tuple(tensor.shape)
                captured["lens"] = wav_lens.tolist() if wav_lens is not None else None
                n = tensor.shape[0]
                return torch.zeros(n, 1, 192)

        monkeypatch.setattr(vp, "_get_ecapa", lambda device: FakeModel())
        engine = VoiceprintEngine()
        wavs = [np.ones(16000, dtype=np.float32),
                np.zeros(16000, dtype=np.float32)]
        engine._embed(wavs)
        assert captured["shape"] == (2, 16000)
        assert captured["lens"] == [1.0, 1.0]


class TestModelSeam:
    def test_embed_raises_readable_when_speechbrain_missing(self, monkeypatch):
        def boom(device):
            raise VoiceprintError("未安装 speechbrain")
        monkeypatch.setattr(vp, "_load_ecapa", boom)
        with pytest.raises(VoiceprintError, match="speechbrain"):
            VoiceprintEngine()._embed([np.zeros(16000, dtype=np.float32)])
