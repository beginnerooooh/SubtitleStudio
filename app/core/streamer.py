"""单目标声纹识别：只判定「这一段是不是当前主播」。

为什么只做单目标（而不是通用多说话人聚类）：
- 业务诉求是「精确识别出当前主播的声音」，多人场景下只需要一个正类；
- 单目标可以用**双阈值 + 双音色**（说话/唱歌）把误判压下去，
  而多类聚类在直播（连麦、观众语音、BGM 原唱）里阈值极难通用。

核心机制：
1. **双音色 Profile**：说话样本与唱歌样本分别取 embedding；
   唱歌音色相对说话整体漂移（实测同一人说话 0.76 vs 唱段 0.38），
   因此 Sim = max(Sim_speak, Sim_sing)，且唱歌阈值单独标定。
2. **自学习唱歌声纹**：任务里若识别到「演唱块」且其说话相似度低但
   在演唱块内部高度自洽，则把该块的中心并入唱歌 Profile
   （只用高分簇，避免把原唱/他人声音学进来）。
3. **自适应阈值**：用本文件里「最像主播」的若干段作为参考分布，
   把阈值定位在参考分布的第 p 分位与拒绝样本之间，减少跨录音漂移。
4. **段内多数投票**：长段切成 3s 窗，按窗投票决定整段归属，
   避免单窗被噪声带偏（门限噪声、咳嗽、笑）。

模型：ECAPA-TDNN（speechbrain/spkrec-ecapa-voxceleb，192 维）。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

SAMPLE_RATE = 16000
MAX_EMBED_SECONDS = 30.0      # ECAPA 训练分布偏好短语句
BATCH_SECONDS = 60.0          # 批量前向的累计音频上限
SILENT_PEAK = 1e-4

_ECAPA: dict[str, object] = {}
_LOCK = threading.Lock()


class VoiceprintError(RuntimeError):
    """声纹相关错误（消息面向用户）。"""


@dataclass
class StreamerProfile:
    """主播声纹（双音色）。向量均已 L2 归一化。"""

    name: str
    speak: Optional[np.ndarray] = None
    sing: Optional[np.ndarray] = None

    def refs(self) -> list[tuple[str, np.ndarray]]:
        out = []
        if self.speak is not None:
            out.append(("speak", self.speak))
        if self.sing is not None:
            out.append(("sing", self.sing))
        return out

    @property
    def empty(self) -> bool:
        return self.speak is None and self.sing is None


@dataclass
class SegmentMatch:
    """单段的匹配结果。"""

    start: float
    end: float
    speak_sim: Optional[float] = None
    sing_sim: Optional[float] = None
    is_streamer: bool = True
    singing: bool = False

    @property
    def best(self) -> float:
        vals = [v for v in (self.speak_sim, self.sing_sim) if v is not None]
        return max(vals) if vals else 0.0


@dataclass
class StreamerConfig:
    """识别参数（阈值可用标定脚本按实际录音调整）。"""

    speak_threshold: float = 0.45
    sing_threshold: float = 0.32
    window_seconds: float = 3.0      # 段内投票窗长
    min_window_seconds: float = 0.8
    vote_ratio: float = 0.5          # 超过该比例的窗判为主播 → 整段保留
    adaptive: bool = True            # 用本文件分布自适应阈值
    adaptive_percentile: float = 25.0
    max_adapt_below: float = 0.62    # 自适应阈值不超过该上限


# ---------------- embedding ----------------

def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norm == 0, 1.0, norm)


def _cap(start: float, end: float, limit: float = MAX_EMBED_SECONDS
         ) -> tuple[float, float]:
    """超长段居中截取（保留人声最集中的部分）。"""
    if end - start <= limit:
        return start, end
    mid = (start + end) / 2
    return mid - limit / 2, mid + limit / 2


def _load_span(wav_path: str, start: float, end: float) -> np.ndarray:
    import soundfile as sf

    info = sf.info(wav_path)
    if info.samplerate != SAMPLE_RATE:
        raise VoiceprintError(
            f"声纹输入需为 {SAMPLE_RATE}Hz WAV（当前 {info.samplerate}Hz）")
    s = max(0, int(round(start * info.samplerate)))
    e = min(info.frames, int(round(end * info.samplerate)))
    if e <= s:
        raise VoiceprintError(f"无效音频区间：[{start}, {end})")
    wav, _ = sf.read(wav_path, start=s, stop=e, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.ascontiguousarray(wav, dtype=np.float32)


def _get_ecapa(device: str):
    with _LOCK:
        if device in _ECAPA:
            return _ECAPA[device]
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        try:
            from speechbrain.pretrained import EncoderClassifier
        except ImportError as exc:
            raise VoiceprintError(
                "未安装 speechbrain（声纹提取依赖），请执行 pip install speechbrain"
            ) from exc
    try:
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device})
    except Exception as exc:
        raise VoiceprintError(f"声纹模型加载失败：{exc}") from exc
    with _LOCK:
        _ECAPA[device] = model
    return model


class VoiceprintEngine:
    """ECAPA 批量嵌入（模型懒加载；_embed 为测试接缝）。"""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def embed_wavs(self, wavs: list[np.ndarray]) -> np.ndarray:
        """(N, T) 波形 → (N, 192) L2 归一化向量。"""
        import torch

        if not wavs:
            return np.zeros((0, 192), dtype=np.float32)
        model = _get_ecapa(self.device)
        lengths = torch.tensor([len(w) for w in wavs], dtype=torch.float32)
        max_len = int(lengths.max())
        batch = np.zeros((len(wavs), max_len), dtype=np.float32)
        for i, w in enumerate(wavs):
            batch[i, : len(w)] = w
        tensor = torch.tensor(batch, device=self.device)
        with torch.no_grad():
            emb = model.encode_batch(tensor, lengths / max_len)
        arr = emb.detach().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[:, 0, :]
        return _normalize(arr.astype(np.float32))

    # 兼容旧接口
    _embed = embed_wavs

    def embed_span(self, wav_path: str, start: float, end: float) -> np.ndarray:
        cs, ce = _cap(start, end)
        return self.embed_wavs([_load_span(wav_path, cs, ce)])[0]

    def embed_segments(self, wav_path: str, segments: list[tuple[float, float]],
                       on_progress: Optional[Callable[[float], None]] = None,
                       ) -> np.ndarray:
        """批量嵌入（按累计 60s 分批前向，控制显存峰值）。"""
        out: list[np.ndarray] = []
        batch: list[np.ndarray] = []
        acc = 0.0
        total = max(1, len(segments))
        for i, (s, e) in enumerate(segments):
            cs, ce = _cap(s, e)
            batch.append(_load_span(wav_path, cs, ce))
            acc += ce - cs
            if acc >= BATCH_SECONDS:
                out.extend(self.embed_wavs(batch))
                batch, acc = [], 0.0
                if on_progress:
                    on_progress(len(out) / total)
        if batch:
            out.extend(self.embed_wavs(batch))
        if on_progress:
            on_progress(1.0)
        return np.stack(out) if out else np.zeros((0, 192), dtype=np.float32)

    def enroll(self, wav_path: str, start: float, end: float) -> np.ndarray:
        """注册样本：整段（或区间）→ 单一 embedding。"""
        wav = _load_span(wav_path, *_cap(start, end))
        if float(np.max(np.abs(wav))) < SILENT_PEAK:
            raise VoiceprintError("声纹样本几乎为静音，请更换清晰的人声样本")
        return self.embed_wavs([wav])[0]


# ---------------- 识别 ----------------

def _threshold_for(cfg: StreamerConfig, singing: bool) -> float:
    return cfg.sing_threshold if singing else cfg.speak_threshold


def classify(
    engine: VoiceprintEngine,
    wav_path: str,
    segments: list[tuple[float, float]],
    profile: StreamerProfile,
    cfg: StreamerConfig | None = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> list[SegmentMatch]:
    """逐段判定是否主播（段内 3s 窗投票）。"""
    cfg = cfg or StreamerConfig()
    if profile.empty:
        raise VoiceprintError("主播声纹为空，请先注册说话样本")
    if not segments:
        return []

    # 展开为窗，逐窗嵌入
    windows: list[tuple[int, float, float]] = []
    for idx, (s, e) in enumerate(segments):
        span = e - s
        if span <= cfg.window_seconds:
            windows.append((idx, s, e))
            continue
        n = max(1, int(span // cfg.window_seconds))
        step = span / n
        for k in range(n):
            ws = s + k * step
            we = ws + step
            if we - ws >= cfg.min_window_seconds:
                windows.append((idx, ws, we))

    embs = engine.embed_segments(
        wav_path, [(s, e) for _i, s, e in windows], on_progress=on_progress)
    refs = profile.refs()
    labels = [lab for lab, _v in refs]
    matrix = np.stack([v for _lab, v in refs])
    sims = embs @ matrix.T
    score_speak = {i: [] for i in range(len(segments))}
    score_sing = {i: [] for i in range(len(segments))}
    for (idx, _s, _e), row in zip(windows, sims):
        by = {lab: float(row[k]) for k, lab in enumerate(labels)}
        if "speak" in by:
            score_speak[idx].append(by["speak"])
        if "sing" in by:
            score_sing[idx].append(by["sing"])

    # 自适应阈值：用全片最像主播的若干窗做参考（只在有足够窗时启用）
    if cfg.adaptive and len(windows) >= 12:
        best = np.array([max(v) for v in (
            score_speak[i] + score_sing[i] for i in range(len(segments))) if v])
        if best.size >= 12:
            floor = float(np.percentile(best, cfg.adaptive_percentile))
            cfg = _adapt(cfg, floor)

    out: list[SegmentMatch] = []
    wi = 0
    window_votes: dict[int, list[bool]] = {i: [] for i in range(len(segments))}
    for i, _ws, _we in windows:
        row = sims[wi]
        wi += 1
        by = {lab: float(row[k]) for k, lab in enumerate(labels)}
        vote = False
        if "speak" in by and by["speak"] >= cfg.speak_threshold:
            vote = True
        if "sing" in by and by["sing"] >= cfg.sing_threshold:
            vote = True
        window_votes[i].append(vote)

    for idx, (s, e) in enumerate(segments):
        sp = score_speak[idx] or None
        sg = score_sing[idx] or None
        speak_sim = float(np.mean(sp)) if sp else None
        sing_sim = float(np.mean(sg)) if sg else None
        votes = window_votes[idx]
        ratio = (sum(votes) / len(votes)) if votes else 1.0
        singing = bool(sg and speak_sim is not None and sing_sim > speak_sim)
        out.append(SegmentMatch(
            start=s, end=e, speak_sim=speak_sim, sing_sim=sing_sim,
            is_streamer=ratio >= cfg.vote_ratio, singing=singing))
    return out


def _adapt(cfg: StreamerConfig, floor: float) -> StreamerConfig:
    """按本文件分布微调阈值（不下探到过低水平，避免把他人学成主播）。"""
    from dataclasses import replace

    speak = min(cfg.max_adapt_below, max(cfg.speak_threshold, floor))
    sing = min(cfg.max_adapt_below, max(cfg.sing_threshold, floor - 0.08))
    return replace(cfg, speak_threshold=round(speak, 3),
                   sing_threshold=round(sing, 3))


def auto_enroll(
    engine: VoiceprintEngine,
    wav_path: str,
    segments: list[tuple[float, float]],
    *,
    name: str = "主播",
    top_ratio: float = 0.35,
    min_seconds: float = 20.0,
) -> StreamerProfile:
    """无样本自注册：取「最自洽的最大簇」作为主播声纹。

    直播里主播的说话时间通常远超其他人；用凝聚聚类找最大簇，
    再用簇内最靠近中心的若干段平均，得到稳健的说话 Profile。
    唱歌 Profile 留空（由自学习环节补齐）。
    """
    if not segments:
        raise VoiceprintError("没有可用的语音段，无法自注册声纹")
    spans = [s for s in segments if s[1] - s[0] >= 1.0]
    if sum(e - s for s, e in spans) < min_seconds:
        raise VoiceprintError(
            f"语音总时长不足 {min_seconds:.0f}s，无法可靠自注册声纹")
    embs = engine.embed_segments(wav_path, spans)
    if len(embs) < 3:
        center = _normalize(embs.mean(axis=0, keepdims=True))[0]
        return StreamerProfile(name=name, speak=center)

    # 余弦相似度矩阵 → 找「与最多段高度相似」的段作为种子
    sims = embs @ embs.T
    np.fill_diagonal(sims, -1.0)
    k = max(3, int(len(spans) * top_ratio))
    k = min(k, len(spans) - 1)
    # 每个段的「前 k 邻居平均相似度」——主播出现最多，簇最紧
    cohesion = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
    seed = int(np.argmax(cohesion))
    keep = np.argsort(-sims[seed])[: max(3, int(len(spans) * 0.5))]
    keep = np.append(keep, seed)
    center = _normalize(embs[keep].mean(axis=0, keepdims=True))[0]
    return StreamerProfile(name=name, speak=center)


def learn_singing_profile(
    engine: VoiceprintEngine,
    wav_path: str,
    profile: StreamerProfile,
    blocks: list[tuple[float, float]],
    *,
    max_similarity: float = 0.62,
    min_internal: float = 0.70,
) -> Optional[np.ndarray]:
    """从高置信演唱块自学习「唱歌音色」。

    条件（同时满足才并入，避免学到原唱/他人）：
    1. 块内各窗之间高度自洽（同类声音，说明是同一个人一直唱）；
    2. 该块与说话 Profile 的相似度不算太高（说明是音色漂移的唱段，
       而不是普通说话）也不太低（太低说明不是主播）。
    """
    if profile.speak is None or not blocks:
        return None
    candidates = []
    for s, e in blocks:
        if e - s < 10.0:
            continue
        n = max(1, int((e - s) // 6.0))
        step = (e - s) / n
        spans = [(s + step * i, s + step * (i + 1)) for i in range(n)]
        embs = engine.embed_segments(wav_path, spans)
        if len(embs) < 2:
            continue
        center = _normalize(embs.mean(axis=0, keepdims=True))[0]
        internal = float(np.mean(embs @ center))
        to_speak = float(center @ profile.speak)
        if internal >= min_internal and 0.15 <= to_speak <= max_similarity:
            candidates.append((internal, center))
    if not candidates:
        return None
    best = max(candidates, key=lambda x: x[0])[1]
    return _normalize(best.astype(np.float32))


# ---------------- 持久化 ----------------

def save_profile(profile: StreamerProfile, profiles_dir: str | Path = "profiles") -> Path:
    """保存主播声纹（<名称>.npy，dict 容器）。"""
    name = _validate_name(profile.name)
    out = Path(profiles_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.npy"
    np.save(path, {
        "version": 2,
        "name": name,
        "speak": profile.speak,
        "sing": profile.sing,
    })
    return path


def load_profile(name: str, profiles_dir: str | Path = "profiles") -> StreamerProfile:
    name = _validate_name(name)
    path = Path(profiles_dir) / f"{name}.npy"
    if not path.is_file():
        raise VoiceprintError(f"未找到主播「{name}」的声纹（{path}），请先注册")
    try:
        data = np.load(path, allow_pickle=True).item()
    except Exception as exc:
        raise VoiceprintError(f"声纹文件损坏：{path}") from exc
    speak = data.get("speak")
    sing = data.get("sing")
    return StreamerProfile(
        name=name,
        speak=_normalize(speak) if speak is not None else None,
        sing=_normalize(sing) if sing is not None else None,
    )


def list_profiles(profiles_dir: str | Path = "profiles") -> list[str]:
    d = Path(profiles_dir)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.npy"))


def delete_profile(name: str, profiles_dir: str | Path = "profiles") -> Path:
    name = _validate_name(name)
    path = Path(profiles_dir) / f"{name}.npy"
    if not path.is_file():
        raise VoiceprintError(f"未找到「{name}」的声纹（{path}）")
    path.unlink()
    return path


def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise VoiceprintError("主播名不能为空")
    if len(name) > 64:
        raise VoiceprintError("主播名过长（≤64 字符）")
    if any(c in name for c in '/\\:*?"<>|'):
        raise VoiceprintError("主播名含非法字符")
    if name.startswith("."):
        raise VoiceprintError("主播名不能以点开头")
    return name


__all__ = [
    "VoiceprintEngine", "StreamerProfile", "SegmentMatch", "StreamerConfig",
    "VoiceprintError", "classify", "auto_enroll", "learn_singing_profile",
    "save_profile", "load_profile", "list_profiles", "delete_profile",
]
