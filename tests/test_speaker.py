"""core/speaker.py 单测：聚类、库匹配、段级置信度、试听片段（embed_fn 接缝 mock）。"""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core.speaker import (
    AMBIGUOUS_MARGIN,
    CLUSTER_THRESHOLD,
    MIN_SPEAKER_SECONDS,
    SpeakerAnalyzer,
    SpeakerAnalysis,
    SpeakerAssignment,
    SpeakerCluster,
    SpeakerError,
    _center_of,
    _match_library,
    _pick_exemplar,
    cluster_embeddings,
)
from core.voiceprint import VoiceProfile, _normalize, load_profile, save_library_speaker

SR = 16000


def _write_pattern_wav(path, patterns_per_second, sr=SR):
    """每秒一个 4 样本 pattern 的 16k mono WAV（同 test_voiceprint 手法）。"""
    blocks = [
        np.tile(np.asarray(pat, dtype=np.float32), sr // 4)
        for pat in patterns_per_second
    ]
    sf.write(str(path), np.concatenate(blocks), sr, subtype="PCM_16")
    return str(path)


def _embed_patterns(wavs):
    """embed_fn 接缝：embedding = L2 归一化(前 4 样本)。"""
    arr = np.stack([np.asarray(w[:4], dtype=np.float32) for w in wavs])
    return _normalize(arr)


A = [1.0, 0.0, 0.0, 0.0]
B = [0.0, 1.0, 0.0, 0.0]
C = [0.0, 0.0, 1.0, 0.0]
A_B = [0.7071, 0.7071, 0.0, 0.0]     # A、B 正中间 → 歧义段


class TestClusterEmbeddings:
    def test_empty(self):
        assert cluster_embeddings(np.zeros((0, 4))) == []

    def test_single(self):
        assert cluster_embeddings(np.ones((1, 4))) == [0]

    def test_two_separated_groups(self):
        emb = _normalize(np.array([A] * 5 + [B] * 5, dtype=np.float32))
        labels = cluster_embeddings(emb)
        assert len(set(labels)) == 2
        assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1
        assert labels[0] != labels[5]

    def test_all_similar_single_cluster(self):
        emb = _normalize(np.array([A, [1, 0.01, 0, 0], [1, 0.02, 0, 0]], dtype=np.float32))
        assert len(set(cluster_embeddings(emb))) == 1

    def test_disjoint_below_threshold_not_merged(self):
        # A 与 C 相似度 0 < 阈值 → 各自成簇
        emb = _normalize(np.array([A, C], dtype=np.float32))
        assert len(set(cluster_embeddings(emb))) == 2

    def test_threshold_changes_result(self):
        emb = _normalize(np.array([A, A_B, B], dtype=np.float32))
        # 宽松阈值：A_B 并入 A 后，avg({A,A_B},B)=(0+0.707)/2≈0.35 仍 > 0.3 → 全并
        assert len(set(cluster_embeddings(emb, threshold=0.3))) == 1
        # 中间阈值：A_B 并入 A（0.707），但 B 与 {A,A_B} 均值 0.35 < 0.5 → 两簇
        assert len(set(cluster_embeddings(emb, threshold=0.5))) == 2
        # 严格阈值（> 0.707）：互不合并
        assert len(set(cluster_embeddings(emb, threshold=0.8))) == 3

    def test_many_segments_fast_and_correct(self):
        """数百段双说话人：结果正确（性能用例，并行互最佳合并）。"""
        rng = np.random.default_rng(42)
        spk1 = _normalize(np.array([1.0, 0.1, 0.0, 0.05]))
        spk2 = _normalize(np.array([0.0, 0.05, 1.0, 0.1]))
        pts = []
        for _ in range(150):
            pts.append(spk1 + rng.normal(0, 0.02, 4))
            pts.append(spk2 + rng.normal(0, 0.02, 4))
        emb = _normalize(np.array(pts, dtype=np.float32))
        labels = cluster_embeddings(emb)
        assert len(set(labels)) == 2
        assert all(labels[i] == labels[0] for i in range(0, 300, 2))
        assert all(labels[i] == labels[1] for i in range(1, 300, 2))

    def test_large_input_performance(self):
        """2000 段（原逐对合并 O(N³) 不可用；并行轮次应秒级内）。"""
        import time

        rng = np.random.default_rng(7)
        emb = _normalize(rng.normal(0, 1, (2000, 16)).astype(np.float32))
        t0 = time.monotonic()
        labels = cluster_embeddings(emb)
        elapsed = time.monotonic() - t0
        assert len(labels) == 2000
        assert elapsed < 10.0, f"聚类耗时 {elapsed:.1f}s（2000 段应远小于 10s）"


class TestHelpers:
    def test_center_of_normalized_mean(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        center = _center_of(emb, [0, 1])
        assert np.allclose(center, [2**-0.5, 2**-0.5], atol=1e-6)

    def test_pick_exemplar_longest(self):
        assert _pick_exemplar([(0, 1), (5, 9), (2, 3)]) == (5, 9)

    def test_pick_exemplar_center_cropped(self):
        s, e = _pick_exemplar([(0, 100)])
        assert e - s == pytest.approx(15.0)
        assert (s + e) / 2 == pytest.approx(50.0)

    def test_match_library_above_threshold(self):
        profiles = [VoiceProfile(name="小明", speak=_normalize(np.array(A)), sing=None)]
        name, sim = _match_library(_normalize(np.array(A)), profiles)
        assert name == "小明" and sim == pytest.approx(1.0)

    def test_match_library_below_threshold(self):
        profiles = [VoiceProfile(name="小明", speak=_normalize(np.array(A)), sing=None)]
        name, sim = _match_library(_normalize(np.array(B)), profiles)
        assert name == "" and sim == pytest.approx(0.0)

    def test_match_library_takes_max_of_dual(self):
        profiles = [VoiceProfile(name="小明",
                                 speak=_normalize(np.array(A)),
                                 sing=_normalize(np.array(B)))]
        name, sim = _match_library(_normalize(np.array(B)), profiles)
        assert name == "小明" and sim == pytest.approx(1.0)


class TestSpeakerAnalyzer:
    def _analyze(self, tmp_path, patterns, seg_len=1.0, **kw):
        """patterns[i] 为第 i 秒的说话人 pattern；返回 (analysis, wav_path)。"""
        wav = _write_pattern_wav(tmp_path / "a.wav", patterns)
        segments = [(i * seg_len, (i + 1) * seg_len) for i in range(len(patterns))]
        analyzer = SpeakerAnalyzer(embed_fn=_embed_patterns)
        analysis = analyzer.analyze(wav, segments, profiles_dir=tmp_path, **kw)
        return analysis, wav

    def test_two_speakers_discovered(self, tmp_path):
        patterns = [A] * 10 + [B] * 10
        analysis, _ = self._analyze(tmp_path, patterns)
        assert len(analysis.clusters) == 2
        # 时长并列时按编号：说话人 1 / 说话人 2
        names = {c.name for c in analysis.clusters}
        assert names == {"说话人 1", "说话人 2"}
        # 每簇 10 段、累计 10s
        for c in analysis.clusters:
            assert len(c.segments) == 10 and c.duration == pytest.approx(10.0)
        # 段归属与 pattern 一致：前 10 段同一说话人，后 10 段另一说话人
        ids = [a.speaker_id for a in analysis.assignments]
        assert len(set(ids[:10])) == 1 and len(set(ids[10:])) == 1
        assert ids[0] != ids[10]
        # 非歧义段置信度高
        assert all(a.confidence > 0.99 for a in analysis.assignments)
        assert not any(a.ambiguous for a in analysis.assignments)

    def test_short_noise_cluster_dropped(self, tmp_path):
        # C 仅出现 2s（< MIN_SPEAKER_SECONDS）→ 视为噪声丢弃，段归属 0
        patterns = [A] * 10 + [B] * 10 + [C] * 2
        analysis, _ = self._analyze(tmp_path, patterns)
        assert len(analysis.clusters) == 2
        dropped = [a for a in analysis.assignments if a.speaker_id == 0]
        assert len(dropped) == 2
        assert all(a.confidence == 0.0 for a in dropped)

    def test_speaker_ids_ordered_by_duration(self, tmp_path):
        patterns = [A] * 12 + [B] * 9
        analysis, _ = self._analyze(tmp_path, patterns)
        by_id = {c.speaker_id: c for c in analysis.clusters}
        assert by_id[1].duration == pytest.approx(12.0)
        assert by_id[2].duration == pytest.approx(9.0)

    def test_ambiguous_segment_flagged(self, tmp_path):
        # A_B 与两个簇中心距离相等（差 < AMBIGUOUS_MARGIN）→ 歧义标注
        patterns = [A] * 8 + [B] * 8 + [A_B]
        analysis, _ = self._analyze(tmp_path, patterns)
        last = analysis.assignments[-1]
        assert last.ambiguous is True
        assert 0 < last.confidence < 1.0
        others = analysis.assignments[:-1]
        assert not any(a.ambiguous for a in others)

    def test_library_match_names_cluster(self, tmp_path):
        save_library_speaker("小明", _normalize(np.array(A)), profiles_dir=tmp_path)
        patterns = [A] * 10 + [B] * 10
        analysis, _ = self._analyze(tmp_path, patterns, use_library=True)
        matched = [c for c in analysis.clusters if c.matched_library]
        assert len(matched) == 1
        assert matched[0].name == "小明"
        assert matched[0].library_sim == pytest.approx(1.0)

    def test_use_library_disabled_leaves_unnamed(self, tmp_path):
        save_library_speaker("小明", _normalize(np.array(A)), profiles_dir=tmp_path)
        patterns = [A] * 10
        analysis, _ = self._analyze(tmp_path, patterns, use_library=False)
        assert analysis.clusters[0].matched_library is False
        assert analysis.clusters[0].name == "说话人 1"

    def test_corrupt_library_file_skipped(self, tmp_path):
        (tmp_path / "坏.npy").write_bytes(b"not numpy")
        patterns = [A] * 10
        analysis, _ = self._analyze(tmp_path, patterns, use_library=True)
        assert len(analysis.clusters) == 1  # 损坏库不阻断分析

    def test_exemplar_written(self, tmp_path):
        patterns = [A] * 10
        exemplar_dir = tmp_path / "ex"
        analysis, _ = self._analyze(tmp_path, patterns, exemplar_dir=exemplar_dir)
        c = analysis.clusters[0]
        assert c.exemplar_path and Path(c.exemplar_path).is_file()
        wav, sr = sf.read(c.exemplar_path)
        assert sr == 16000
        assert len(wav) == pytest.approx(SR, abs=SR // 10)   # 最长段 = 1s
        # 内容与源音频对应区间一致（pattern A）
        assert np.allclose(wav[:4], A, atol=1e-3)

    def test_no_exemplar_dir_skips_cutting(self, tmp_path):
        patterns = [A] * 10
        analysis, _ = self._analyze(tmp_path, patterns, exemplar_dir=None)
        assert analysis.clusters[0].exemplar_path == ""

    def test_progress_reported(self, tmp_path):
        ratios = []
        patterns = [A] * 3 + [B] * 3
        wav = _write_pattern_wav(tmp_path / "a.wav", patterns)
        segments = [(i, i + 1) for i in range(6)]
        SpeakerAnalyzer(embed_fn=_embed_patterns).analyze(
            wav, segments, profiles_dir=tmp_path,
            on_progress=ratios.append)
        assert ratios[-1] == 1.0
        assert ratios == sorted(ratios)

    def test_empty_segments_returns_empty(self, tmp_path):
        analysis = SpeakerAnalyzer(embed_fn=_embed_patterns).analyze(
            "whatever.wav", [], profiles_dir=tmp_path)
        assert analysis.clusters == [] and analysis.assignments == []

    def test_non_16k_wav_rejected(self, tmp_path):
        path = tmp_path / "hi.wav"
        sf.write(str(path), np.zeros(44100, dtype=np.float32), 44100)
        with pytest.raises(SpeakerError, match="16kHz"):
            SpeakerAnalyzer(embed_fn=_embed_patterns).analyze(
                str(path), [(0, 1)], profiles_dir=tmp_path)


class TestSpeakerAnalysis:
    def test_cluster_by_id(self):
        clusters = [SpeakerCluster(speaker_id=1, name="甲",
                                   embedding=np.zeros(4), matched_library=False,
                                   library_sim=0.0)]
        analysis = SpeakerAnalysis(clusters=clusters)
        assert analysis.cluster_by_id(1) is clusters[0]
        assert analysis.cluster_by_id(2) is None

    def test_refs_for_filters_by_name_in_order(self):
        clusters = [
            SpeakerCluster(speaker_id=i + 1, name=n, embedding=np.zeros(4),
                           matched_library=False, library_sim=0.0)
            for i, n in enumerate(["甲", "乙", "丙"])
        ]
        analysis = SpeakerAnalysis(clusters=clusters)
        refs = analysis.refs_for(["丙", "甲"])
        assert [c.name for c in refs] == ["甲", "丙"]     # 保持 clusters 顺序
        assert analysis.refs_for(None) == []

    def test_defaults(self):
        a = SpeakerAnalysis()
        assert a.clusters == [] and a.assignments == []
        assert a.separated is False      # 默认在原始混音上分析
        s = SpeakerAssignment(start=0.0, end=1.0, speaker_id=1,
                              confidence=0.9, ambiguous=False)
        assert s.speaker_id == 1

    def test_separated_provenance_recorded(self, tmp_path):
        """separated=True（Demucs 去 BGM 后分析）→ 写入结果供 pipeline 复用判断。"""
        patterns = [A] * 10
        wav = _write_pattern_wav(tmp_path / "a.wav", patterns)
        segments = [(i, i + 1) for i in range(10)]
        analysis = SpeakerAnalyzer(embed_fn=_embed_patterns).analyze(
            wav, segments, profiles_dir=tmp_path, separated=True)
        assert analysis.separated is True


class TestSaveLibrarySpeaker:
    def test_roundtrip_loadable_as_profile(self, tmp_path):
        emb = _normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        path = save_library_speaker("小明", emb, profiles_dir=tmp_path)
        assert path == tmp_path / "小明.npy"
        p = load_profile("小明", profiles_dir=tmp_path)
        assert p.name == "小明"
        assert np.allclose(p.speak, emb)
        assert p.sing is None

    def test_invalid_embedding_rejected(self, tmp_path):
        with pytest.raises(Exception, match="声纹向量无效"):
            save_library_speaker("坏", np.zeros((2, 2)), profiles_dir=tmp_path)

    def test_invalid_name_rejected(self, tmp_path):
        with pytest.raises(Exception):
            save_library_speaker("a/b", np.ones(4), profiles_dir=tmp_path)


class TestConstants:
    """调参常量合理性：约束之间不互相矛盾。"""

    def test_thresholds_sane(self):
        assert 0.4 < CLUSTER_THRESHOLD < 0.8
        assert MIN_SPEAKER_SECONDS > 0
        assert 0 < AMBIGUOUS_MARGIN < 0.2
