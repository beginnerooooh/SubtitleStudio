"""多声纹说话人分离：VAD 分段 → ECAPA 批量嵌入 → 凝聚聚类 → 声纹库匹配。

无监督发现文件中的各说话人（不依赖预注册），并结合声纹库命名：
    1. VAD 切出语音段，逐段提取 192 维声纹 embedding（批量前向）
    2. 段间余弦相似度做 average-linkage 凝聚聚类（阈值自动定簇数）
    3. 簇中心与声纹库（profiles/*.npy，说话+唱歌双音色取 max）比对命名
    4. 段级归属置信度：与簇中心的相似度；与次近簇差距过小 → ambiguous

用途（直播回放等多说话人场景）：
    - UI 展示声纹卡片（时长/段数/库匹配/代表片段试听）→ 用户勾选保留
    - 勾选后仅转写对应区域；归属不确定的段参与转写但标注供人工复核

纯 numpy 实现聚类（段数通常数百，O(n²) 相似度矩阵毫秒级），无需 scipy。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core.voiceprint import (
    _BATCH_SECONDS,
    _cap_span,
    _load_span,
    VoiceprintEngine,
    list_profiles,
    load_profile,
)

# ---------------- 调参常量 ----------------
CLUSTER_THRESHOLD = 0.55        # 凝聚聚类合并阈值（余弦相似度）
MIN_SPEAKER_SECONDS = 8.0       # 簇累计时长下限（过短视为噪声丢弃）
LIBRARY_MATCH_THRESHOLD = 0.55  # 簇中心与库 Profile 的命名阈值
AMBIGUOUS_MARGIN = 0.08         # top1-top2 相似度差 < 此值 → 归属不确定
EXEMPLAR_MAX_SECONDS = 15.0     # 代表片段上限（试听用，超出居中截取）
_EXEMPLAR_SR = 16000


class SpeakerError(RuntimeError):
    """说话人分离错误，message 面向用户可读。"""


@dataclass
class SpeakerAssignment:
    """单个 VAD 段的说话人归属。speaker_id=0 表示未归属（噪声簇已丢弃）。"""
    start: float
    end: float
    speaker_id: int          # 1-based；0 = 未归属
    confidence: float        # 与簇中心余弦相似度
    ambiguous: bool          # 与次近簇差距 < AMBIGUOUS_MARGIN


@dataclass
class SpeakerCluster:
    """聚类得到的一个说话人（可能匹配到声纹库中的已知主播）。"""
    speaker_id: int                       # 1-based
    name: str                             # 库名或「说话人 N」
    embedding: np.ndarray                 # 簇中心 (D,) L2 归一化
    matched_library: bool                 # 是否命中声纹库
    library_sim: float                    # 命中相似度（未命中为 0）
    segments: list[tuple[float, float]] = field(default_factory=list)
    duration: float = 0.0                 # segments 累计时长
    exemplar_span: tuple[float, float] = (0.0, 0.0)   # 代表片段（最长段）
    exemplar_path: str = ""               # 切出的试听 wav 路径


@dataclass
class SpeakerAnalysis:
    """一次完整分析的结果（可序列化进 gr.State 跨请求复用）。"""
    clusters: list[SpeakerCluster] = field(default_factory=list)
    assignments: list[SpeakerAssignment] = field(default_factory=list)
    separated: bool = False    # 是否在 Demucs 分离后的纯人声上分析（BGM 场景）

    def cluster_by_id(self, speaker_id: int) -> Optional[SpeakerCluster]:
        for c in self.clusters:
            if c.speaker_id == speaker_id:
                return c
        return None

    def refs_for(self, selected_names: list[str]) -> list[SpeakerCluster]:
        """按勾选名筛选簇（保持 clusters 顺序）。"""
        wanted = set(selected_names or [])
        return [c for c in self.clusters if c.name in wanted]


# ---------------- 聚类（纯函数，numpy 向量化） ----------------

def cluster_embeddings(
    embeddings: np.ndarray,
    threshold: float = CLUSTER_THRESHOLD,
) -> list[int]:
    """average-linkage 凝聚聚类（Lance-Williams 增量 + 互最佳对并行合并）。

    embeddings: (N, D) L2 归一化。返回长度 N 的簇标签（0-based）。
    两簇平均相似度 > threshold 时合并；单元素簇也保留（由时长过滤兜底）。

    性能：每轮同时合并所有「互为最佳伙伴」的簇对（与逐对合并的
    average-linkage 结果一致——全局最佳对必互为最佳，且合并不影响
    其余互最佳对的相对顺序），簇数每轮近似减半 → 总复杂度 ~O(N²)，
    数千段语音毫秒~秒级完成（逐对合并为 O(N³)，长音频不可用）。
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]

    sims = embeddings @ embeddings.T              # (N, N) 段间余弦相似度
    np.fill_diagonal(sims, 0.0)
    # sim_sum[a, b] = 簇 a 全体成员与簇 b 全体成员的相似度总和（平均链接分子）
    sim_sum = sims.astype(np.float64, copy=True)
    sizes = np.ones(n, dtype=np.float64)
    alive = np.ones(n, dtype=bool)
    members: list[list[int]] = [[i] for i in range(n)]

    while True:
        idx = np.flatnonzero(alive)
        k = len(idx)
        if k <= 1:
            break
        avg = sim_sum[np.ix_(idx, idx)] / np.outer(sizes[idx], sizes[idx])
        np.fill_diagonal(avg, -2.0)               # 对角线不参与选对
        best = np.argmax(avg, axis=1)             # 各簇当前的最佳合并伙伴
        best_val = avg[np.arange(k), best]
        # 本轮合并：互为最佳且平均相似度 > threshold 的对（并行、互不相交）
        done = np.zeros(k, dtype=bool)
        merged_any = False
        for i in range(k):
            j = int(best[i])
            if done[i] or j == i or best_val[i] <= threshold:
                continue
            if int(best[j]) != i or done[j]:
                continue                          # 非互最佳 → 等下一轮
            a, b = int(idx[i]), int(idx[j])       # 合并 b → a
            sim_sum[a, :] += sim_sum[b, :]
            sim_sum[:, a] = sim_sum[a, :]
            sim_sum[a, a] = 0.0
            sizes[a] += sizes[b]
            alive[b] = False
            members[a].extend(members[b])
            done[i] = done[j] = True
            merged_any = True
        if not merged_any:
            break

    labels = [0] * n
    for new_id, slot in enumerate(np.flatnonzero(alive)):
        for member in members[int(slot)]:
            labels[member] = new_id
    return labels


def _center_of(embeddings: np.ndarray, indices: list[int]) -> np.ndarray:
    """簇成员均值 → L2 归一化中心。"""
    mean = embeddings[indices].mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 0 else mean


def _pick_exemplar(segments: list[tuple[float, float]]) -> tuple[float, float]:
    """代表片段 = 最长段（超长居中截取 EXEMPLAR_MAX_SECONDS）。"""
    start, end = max(segments, key=lambda s: s[1] - s[0])
    span = end - start
    if span <= EXEMPLAR_MAX_SECONDS:
        return start, end
    mid = (start + end) / 2
    return mid - EXEMPLAR_MAX_SECONDS / 2, mid + EXEMPLAR_MAX_SECONDS / 2


def _cut_exemplar(wav_path: str, span: tuple[float, float], out_path: Path) -> None:
    """从整段 wav 切出试听片段（soundfile 定点读，不经过 ffmpeg）。"""
    import soundfile as sf

    info = sf.info(wav_path)
    s = max(0, int(round(span[0] * info.samplerate)))
    e = min(info.frames, int(round(span[1] * info.samplerate)))
    if e <= s:
        raise SpeakerError(f"试听片段区间无效：{span}")
    wav, _ = sf.read(wav_path, start=s, stop=e, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, np.ascontiguousarray(wav, dtype="float32"),
             _EXEMPLAR_SR, subtype="PCM_16")


def _match_library(
    center: np.ndarray,
    profiles: list,
    threshold: float = LIBRARY_MATCH_THRESHOLD,
) -> tuple[str, float]:
    """簇中心 vs 声纹库全部 Profile（说话+唱歌取 max）→ (名称, 最高相似度)。"""
    best_name, best_sim = "", 0.0
    for profile in profiles:
        for ref in (profile.speak, profile.sing):
            if ref is None:
                continue
            sim = float(center @ ref)
            if sim > best_sim:
                best_name, best_sim = profile.name, sim
    if best_sim >= threshold:
        return best_name, best_sim
    return "", best_sim


# ---------------- 分析器 ----------------

class SpeakerAnalyzer:
    """说话人分离分析器。embed_fn 为测试接缝（生产用 VoiceprintEngine）。"""

    def __init__(self, device: str = "cpu",
                 embed_fn: Optional[Callable[[list], np.ndarray]] = None):
        self.device = device
        self._engine: Optional[VoiceprintEngine] = None
        self._embed_fn_override = embed_fn

    def _embed(self, wavs: list) -> np.ndarray:
        if self._embed_fn_override is not None:
            return self._embed_fn_override(wavs)
        if self._engine is None:
            self._engine = VoiceprintEngine(device=self.device)
        return self._engine._embed(wavs)

    def analyze(
        self,
        wav_path: str,
        segments: list[tuple[float, float]],
        profiles_dir: str | Path = "profiles",
        use_library: bool = True,
        cluster_threshold: float = CLUSTER_THRESHOLD,
        exemplar_dir: str | Path | None = None,
        separated: bool = False,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> SpeakerAnalysis:
        """完整分析：嵌入 → 聚类 → 库匹配 → 段级归属 + 试听片段。

        wav_path 须为 16k 单声道 wav（与声纹模块同一约定）。
        exemplar_dir 不为空时切出各簇代表片段（UI 试听用）。
        separated: wav_path 是否为人声分离后的纯人声（写入结果供复用判断）。
        """
        import soundfile as sf

        if not segments:
            return SpeakerAnalysis(separated=separated)
        info = sf.info(wav_path)
        if info.samplerate != 16000:
            raise SpeakerError(
                f"声纹输入需为 16kHz WAV（当前 {info.samplerate}Hz），"
                "请先经过音频抽取流程")

        def report(r: float) -> None:
            if on_progress:
                on_progress(min(1.0, max(0.0, r)))

        # 1) 批量嵌入（复用声纹引擎的 60s 累计分批策略，控制内存峰值）
        embeddings: list[np.ndarray] = []
        batch: list[np.ndarray] = []
        acc = 0.0
        total = len(segments)
        done = 0
        for s, e in segments:
            cs, ce = _cap_span(s, e)
            batch.append(_load_span(wav_path, cs, ce))
            acc += ce - cs
            if acc >= _BATCH_SECONDS:
                embeddings.extend(self._embed(batch))
                batch, acc = [], 0.0
            done += 1
            report(0.7 * done / total)
        if batch:
            embeddings.extend(self._embed(batch))
        matrix = np.stack(embeddings)             # (N, D)
        report(0.72)

        # 2) 聚类 + 时长过滤（过短簇视为噪声丢弃 → 段归属 0）
        labels = cluster_embeddings(matrix, threshold=cluster_threshold)
        report(0.8)
        by_label: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            by_label.setdefault(lab, []).append(idx)

        # 命名时按累计时长降序编号（说话人 1 = 出现最久的）
        def label_duration(lab: int) -> float:
            return sum(segments[i][1] - segments[i][0] for i in by_label[lab])

        ordered = sorted(by_label, key=lambda l: -label_duration(l))
        keep = {lab: rank + 1 for rank, lab in enumerate(ordered)
                if label_duration(lab) >= MIN_SPEAKER_SECONDS}
        dropped_labels = {lab for lab in ordered if lab not in keep}

        # 3) 声纹库加载与匹配（失败不影响分析，仅退化为未命名）
        profiles: list = []
        if use_library:
            for name in list_profiles(profiles_dir):
                try:
                    profiles.append(load_profile(name, profiles_dir))
                except Exception:
                    continue  # 损坏的库文件跳过

        clusters: list[SpeakerCluster] = []
        for lab, sid in sorted(keep.items(), key=lambda kv: kv[1]):
            indices = by_label[lab]
            segs = [segments[i] for i in indices]
            center = _center_of(matrix, indices)
            lib_name, lib_sim = _match_library(center, profiles)
            matched = bool(lib_name)
            clusters.append(SpeakerCluster(
                speaker_id=sid,
                name=lib_name if matched else f"说话人 {sid}",
                embedding=center,
                matched_library=matched,
                library_sim=lib_sim,
                segments=segs,
                duration=label_duration(lab),
                exemplar_span=_pick_exemplar(segs),
            ))
        report(0.9)

        # 4) 段级归属 + 置信度/歧义检测
        centers = np.stack([c.embedding for c in clusters]) if clusters else None
        assignments: list[SpeakerAssignment] = []
        for i, (s, e) in enumerate(segments):
            if labels[i] in dropped_labels or centers is None:
                assignments.append(SpeakerAssignment(
                    start=s, end=e, speaker_id=0, confidence=0.0, ambiguous=False))
                continue
            sims = centers @ matrix[i]            # (K,)
            order = np.argsort(-sims)
            top1 = int(order[0])
            conf = float(sims[top1])
            ambiguous = False
            if len(order) > 1:
                ambiguous = float(sims[top1] - sims[order[1]]) < AMBIGUOUS_MARGIN
            assignments.append(SpeakerAssignment(
                start=s, end=e, speaker_id=top1 + 1,
                confidence=conf, ambiguous=ambiguous))

        # 5) 代表片段切取（IO 放最后，失败不影响结构化结果）
        if exemplar_dir is not None:
            out_dir = Path(exemplar_dir)
            for c in clusters:
                try:
                    p = out_dir / f"speaker_{c.speaker_id}_exemplar.wav"
                    _cut_exemplar(wav_path, c.exemplar_span, p)
                    c.exemplar_path = str(p)
                except Exception:
                    continue
        report(1.0)
        return SpeakerAnalysis(clusters=clusters, assignments=assignments,
                               separated=separated)
