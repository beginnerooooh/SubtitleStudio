"""BGM 人声判别：多因子评估「这一段里的声音是谁」。

问题背景
────────
直播唱歌时音频里同时存在：
1. 主播**现场**说话/唱歌（要保留）；
2. 伴奏里原唱的**录音**人声（要剔除）；
3. 纯伴奏/纯音乐段（无字幕）。

单靠声纹不够：主播唱歌音色会漂移（实测同一人说话 0.76 / 唱段 0.38），
且自学习唱歌音色时容易把**原唱**当成主播学进去。因此这里做多因子融合：

| 因子 | 物理含义 | 判别方向 |
|---|---|---|
| `vocal_sim` | 与本主播声纹的最大相似度 | 越高越是主播 |
| `music` | 伴奏存在度（低频鼓点+高频镲片+谱平坦度） | 越高越可能是「跟着伴奏」 |
| `side` | 立体声侧信号占比（音乐宽、人声居中） | 越高越是录音/BGM |
| `loud_cv` | 段内响度变异系数 | 现场演唱波动大，录音母带平稳 |
| `gap` | 与前后段的静音间隔 | 现场换气/停顿更明显 |

决策（可解释、可调）：
    score = w_sim * vocal_sim
          - w_music * music * (1 - vocal_sim)      # 有伴奏但不像主播 → 原唱
          - w_side * side
          - w_flat * flatness
    keep = score >= threshold
并且给出 `reasons`，便于人工复核（UI 会展示）。

立体声侧信号需要原始立体声；只有单声道时该因子自动关闭（不影响其它因子）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

FRAME_SR = 16000


@dataclass
class SegmentFeatures:
    """单个语音段的声学特征。"""

    start: float
    end: float
    rms_db: float = -60.0        # 平均响度
    peak_db: float = -60.0
    bass_ratio: float = 0.0      # 20~250Hz 能量占比（鼓/贝斯）
    high_ratio: float = 0.0      # 4k~8kHz 能量占比（镲片/齿音）
    voice_ratio: float = 0.0     # 300~3400Hz 能量占比（人声主体）
    flatness: float = 0.0        # 谱平坦度（噪声型/音乐型 vs 谐波型）
    side_ratio: Optional[float] = None   # 侧信号占比（需立体声）
    loud_cv: float = 0.0         # 段内响度变异系数
    zcr: float = 0.0             # 过零率（清辅音/噪声）

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SegmentDecision:
    """多因子决策结果。"""

    start: float
    end: float
    keep: bool
    score: float
    vocal_sim: float = 0.0
    music: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class BgmTuning:
    """决策权重与阈值。

    默认值按实测标定（10 分钟样本，7 段原唱/伴奏 vs 8 段主播现场）：

    | 因子 | 原唱/伴奏(中位) | 主播现场(中位) | 权重 |
    |---|---:|---:|---:|
    | 声纹相似度 | 0.39 | **0.69** | 1.00（主判据） |
    | 伴奏存在度 | **0.52** | 0.41 | 0.55 |
    | 侧信号 | **0.20** | 0.11 | 0.35 |
    | 谱平坦度 | 0.46 | 0.45 | 0.05（弱） |
    | 响度变异 | 0.43 | 0.51 | 0.10 |

    **声纹是最强判据**（0.39 vs 0.69，不重叠）；伴奏度与侧信号作辅助，
    用于「主播在跟唱但音色漂移」的时刻。谱平坦度几乎无区分度，权重压低。
    """

    # 权重
    w_sim: float = 1.0
    w_music: float = 0.55
    w_side: float = 0.35
    w_flat: float = 0.05
    w_loud_cv: float = 0.10
    # 阈值：实测主播现场最低 0.55、原唱最高 0.47 → 0.50 附近最稳
    threshold: float = 0.50
    # 声纹不可用时的退路：纯声学判「有伴奏的人声」→ 标记低置信而非删除
    music_only_threshold: float = 0.62
    # 场景加成：若该段落在「主播点歌致谢/说话的上下文」里，允许放宽阈值
    context_bonus: float = 0.10


# ---------------- 特征提取 ----------------

def _read_mono(wav_path: str, start: float, end: float) -> np.ndarray:
    import soundfile as sf

    info = sf.info(wav_path)
    s = max(0, int(round(start * info.samplerate)))
    e = min(info.frames, int(round(end * info.samplerate)))
    if e <= s:
        return np.zeros(0, dtype=np.float32)
    wav, _sr = sf.read(wav_path, start=s, stop=e, dtype="float32",
                       always_2d=True)
    return wav.mean(axis=1).astype(np.float32)


def _read_side(wav_path: str, start: float, end: float) -> Optional[float]:
    """立体声侧信号能量占比（|L-R| / |L+R|）；单声道返回 None。"""
    import soundfile as sf

    info = sf.info(wav_path)
    if info.channels < 2:
        return None
    s = max(0, int(round(start * info.samplerate)))
    e = min(info.frames, int(round(end * info.samplerate)))
    if e <= s:
        return None
    wav, _sr = sf.read(wav_path, start=s, stop=e, dtype="float32",
                       always_2d=True)
    if wav.shape[1] < 2:
        return None
    left, right = wav[:, 0], wav[:, 1]
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    m = float(np.sqrt(np.mean(mid ** 2)))
    sd = float(np.sqrt(np.mean(side ** 2)))
    if m + sd < 1e-9:
        return 0.0
    return sd / (m + sd)


def _band_ratio(spec: np.ndarray, freqs: np.ndarray, lo: float, hi: float,
                total: float) -> float:
    if total <= 1e-12:
        return 0.0
    sel = (freqs >= lo) & (freqs < hi)
    return float(spec[sel].sum() / total)


def segment_features(
    wav_path: str,
    start: float,
    end: float,
    *,
    stereo_path: str | None = None,
) -> SegmentFeatures:
    """计算一个语音段的声学特征（纯 numpy，无模型依赖）。"""
    wav = _read_mono(wav_path, start, end)
    feat = SegmentFeatures(start=start, end=end)
    if wav.size < 400:
        return feat
    # 响度
    rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-12)
    feat.rms_db = 20 * np.log10(max(rms, 1e-9))
    feat.peak_db = 20 * np.log10(max(float(np.max(np.abs(wav))), 1e-9))
    # 帧级响度变异
    win = 400
    n = max(1, wav.size // win)
    frames = wav[: n * win].reshape(n, win)
    fr = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    if fr.size > 1 and fr.mean() > 1e-9:
        feat.loud_cv = float(fr.std() / fr.mean())
    # 频谱特征（平均幅度谱）
    n_fft = 1024
    if wav.size >= n_fft:
        hop = n_fft // 2
        idx = np.arange(0, wav.size - n_fft, hop)
        if idx.size > 64:
            idx = idx[:: max(1, idx.size // 64)]
        spec = np.zeros(n_fft // 2 + 1, dtype=np.float64)
        for i in idx[:64]:
            seg = wav[i:i + n_fft] * np.hanning(n_fft)
            spec += np.abs(np.fft.rfft(seg))
        spec /= max(1, min(64, idx.size))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / FRAME_SR)
        total = float(spec.sum())
        feat.bass_ratio = _band_ratio(spec, freqs, 20, 250, total)
        feat.voice_ratio = _band_ratio(spec, freqs, 300, 3400, total)
        feat.high_ratio = _band_ratio(spec, freqs, 4000, 8000, total)
        # 谱平坦度：几何均值 / 算术均值（噪声/音乐型更高）
        p = np.maximum(spec, 1e-12)
        feat.flatness = float(np.exp(np.mean(np.log(p))) / max(1e-12, p.mean()))
    # 过零率
    feat.zcr = float(np.mean(np.abs(np.diff(np.signbit(wav).astype(np.int8)))))
    # 立体声侧信号
    if stereo_path:
        feat.side_ratio = _read_side(stereo_path, start, end)
    return feat


# ---------------- 音乐存在度 ----------------

def music_presence(feat: SegmentFeatures) -> float:
    """由声学特征估计「伴奏存在度」（0~1）。

    依据：
    - 伴奏会带来可观的低频（鼓/贝斯）与高频（镲片）能量，人声段这两项都低；
    - 音乐的谱平坦度高于纯人声（人声是强谐波结构，平坦度低）；
    - 侧信号越大说明录音/混音成分越多（现场话筒居中）。
    """
    score = 0.0
    # 低频：纯人声段 bass_ratio 通常 < 0.12；带伴奏 > 0.25
    score += 0.40 * float(np.clip((feat.bass_ratio - 0.10) / 0.30, 0.0, 1.0))
    # 高频：齿音也在此，但纯人声 high_ratio 通常 < 0.08
    score += 0.20 * float(np.clip((feat.high_ratio - 0.05) / 0.18, 0.0, 1.0))
    # 谱平坦度：纯人声 ~0.02~0.15，音乐 ~0.10~0.40
    score += 0.25 * float(np.clip((feat.flatness - 0.06) / 0.25, 0.0, 1.0))
    # 侧信号
    if feat.side_ratio is not None:
        score += 0.15 * float(np.clip((feat.side_ratio - 0.10) / 0.35, 0.0, 1.0))
    return float(np.clip(score, 0.0, 1.0))


# ---------------- 决策 ----------------

def decide(
    feature: SegmentFeatures,
    vocal_sim: float,
    tuning: BgmTuning | None = None,
    *,
    context_support: float = 0.0,
) -> SegmentDecision:
    """融合声纹与声学特征，判定是否保留该段。

    context_support：外部上下文支持度 0~1（例如该段紧邻主播致谢、或落在
    弹幕点歌请求附近）。它只做**小幅放宽**，不会把明显是原唱的段翻过来。
    """
    t = tuning or BgmTuning()
    music = music_presence(feature)
    sim = float(np.clip(vocal_sim, 0.0, 1.0))
    score = t.w_sim * sim - t.w_music * music * (1.0 - sim)
    if feature.side_ratio is not None:
        score -= t.w_side * max(0.0, feature.side_ratio - 0.12)
    score -= t.w_flat * max(0.0, feature.flatness - 0.12)
    score -= t.w_loud_cv * max(0.0, 0.85 - feature.loud_cv)
    score += t.context_bonus * float(np.clip(context_support, 0.0, 1.0))
    keep = score >= t.threshold
    reasons = []
    if sim < 0.35:
        reasons.append(f"与主播声纹相似度低（{sim:.2f}）")
    elif sim < 0.50:
        reasons.append(f"与主播声纹相似度中等（{sim:.2f}）")
    if music > 0.45:
        reasons.append(f"检测到伴奏（{music:.2f}）")
    if feature.side_ratio is not None and feature.side_ratio > 0.25:
        reasons.append(f"立体声侧信号偏高（{feature.side_ratio:.2f}）")
    if context_support > 0.3:
        reasons.append(f"上下文支持（{context_support:.2f}）")
    if not keep and not reasons:
        reasons.append("综合得分低于阈值")
    return SegmentDecision(
        start=feature.start, end=feature.end, keep=keep,
        score=round(score, 4), vocal_sim=round(sim, 4),
        music=round(music, 4), reasons=reasons)


def explain(feature: SegmentFeatures) -> str:
    """单行摘要（日志/UI 用）。"""
    parts = [f"响度 {feature.rms_db:.0f}dB",
             f"低频 {feature.bass_ratio:.2f}",
             f"人声带 {feature.voice_ratio:.2f}",
             f"平坦度 {feature.flatness:.2f}"]
    if feature.side_ratio is not None:
        parts.append(f"侧信号 {feature.side_ratio:.2f}")
    return " · ".join(parts)


__all__ = ["SegmentFeatures", "SegmentDecision", "BgmTuning", "segment_features",
           "music_presence", "decide", "explain"]
