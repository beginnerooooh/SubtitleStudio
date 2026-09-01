"""主声线声纹：ECAPA-TDNN 提取、双音色 Profile 管理、分段比对与区域合并。

模型：speechbrain/spkrec-ecapa-voxceleb（192 维 embedding，懒加载 + 缓存）。
唱歌音色相对说话存在整体漂移 → 双 Profile（说话 A 必填 + 唱歌 B 可选）：
    Sim = max(Sim_speak, Sim_sing)
低于阈值（默认 0.55，UI 可调）的语音段判为背景音/其他人声，转写前剔除。

Profile 持久化：profiles/<主播名>.npy（dict 容器，np.save 对象数组）。
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from core.audio import extract_wav

_SAMPLE_RATE = 16000
_MAX_EMBED_SECONDS = 30.0   # 送入 ECAPA 的单段音频上限（居中截取，模型训练分布为短语句）
_BATCH_SECONDS = 60.0       # 批量前向的累计音频上限（控制内存/显存峰值）
_SILENT_PEAK = 1e-4         # 注册样本近静音判定阈值
_BAD_NAME_CHARS = '/\\:*?"<>|'
_NAME_MAX = 64

# 进程内模型缓存（key = device）
_ecapa_cache: dict[str, object] = {}


class VoiceprintError(RuntimeError):
    """声纹相关错误，message 面向用户可读。"""


@dataclass
class VoiceProfile:
    name: str
    speak: Optional[np.ndarray]   # (192,) L2 归一化
    sing: Optional[np.ndarray]    # (192,) L2 归一化


@dataclass
class SegmentVerdict:
    """单个 VAD 语音段的声纹判定结果。"""
    start: float
    end: float
    speak_sim: Optional[float]
    sing_sim: Optional[float]

    @property
    def best_sim(self) -> float:
        """双音色取最大：唱歌段由唱歌声纹兜底，避免高音区误过滤。"""
        return max(s for s in (self.speak_sim, self.sing_sim) if s is not None)

    @property
    def is_singing(self) -> bool:
        """唱歌状态判定：仅当注册了唱歌声纹且相似度更高。"""
        return (
            self.sing_sim is not None
            and (self.speak_sim is None or self.sing_sim > self.speak_sim)
        )


# ---------------- 纯函数 ----------------

def _normalize(x: np.ndarray) -> np.ndarray:
    """L2 归一化（余弦相似度 = 归一化向量内积）。"""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return x / norm


def _cap_span(start: float, end: float,
              max_seconds: float = _MAX_EMBED_SECONDS) -> tuple[float, float]:
    """超长段居中截取 max_seconds。"""
    if end - start <= max_seconds:
        return start, end
    mid = (start + end) / 2
    half = max_seconds / 2
    return mid - half, mid + half


def merge_regions(
    segments: list[tuple[float, float]],
    duration: Optional[float] = None,
    gap: float = 1.0,
    pad: float = 0.25,
    min_len: float = 0.5,
) -> list[tuple[float, float]]:
    """保留段 → whisper clip_timestamps 区域：排序合并（间隔 ≤ gap）、
    边界 ±pad、按 duration 夹紧、丢弃 < min_len 的碎片。"""
    if not segments:
        return []
    spans = sorted(segments)
    merged: list[list[float]] = [list(spans[0])]
    for s, e in spans[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    regions: list[tuple[float, float]] = []
    for s, e in merged:
        if e - s < min_len:  # 碎片判定在 padding 前做（padding 不应救活碎片）
            continue
        s, e = max(0.0, s - pad), e + pad
        if duration is not None:
            e = min(duration, e)
        regions.append((round(s, 3), round(e, 3)))
    return regions


# ---------------- 模型接缝 ----------------

def _load_ecapa(device: str):
    """加载 ECAPA-TDNN（首次调用下载模型，~80MB）。"""
    try:
        from speechbrain.inference.speaker import EncoderClassifier  # speechbrain >= 1.0
    except ImportError:
        try:
            from speechbrain.pretrained import EncoderClassifier  # speechbrain < 1.0
        except ImportError as exc:
            raise VoiceprintError(
                "未安装 speechbrain（声纹提取依赖），请执行 pip install speechbrain"
            ) from exc
    try:
        return EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
    except Exception as exc:
        raise VoiceprintError(f"声纹模型加载失败：{exc}") from exc


def _get_ecapa(device: str):
    if device not in _ecapa_cache:
        _ecapa_cache[device] = _load_ecapa(device)
    return _ecapa_cache[device]


def speech_intervals(wav_path: str) -> list[tuple[float, float]]:
    """Silero VAD（faster-whisper 内置）切出语音区间（秒）。"""
    import torch
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    wav, sr = _load_full(wav_path)
    stamps = get_speech_timestamps(
        torch.from_numpy(wav), vad_options=VadOptions(), sampling_rate=sr
    )
    return [(s["start"] / sr, s["end"] / sr) for s in stamps]


# ---------------- 音频读取 ----------------

def _load_full(wav_path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def _load_span(wav_path: str, start: float, end: float) -> np.ndarray:
    """读取 [start, end) 秒音频（float32 mono）。"""
    info = sf.info(wav_path)
    if info.samplerate != _SAMPLE_RATE:
        raise VoiceprintError(
            f"声纹输入需为 {_SAMPLE_RATE}Hz WAV（当前 {info.samplerate}Hz），"
            "请先经过音频抽取流程"
        )
    s = max(0, int(round(start * info.samplerate)))
    e = min(info.frames, int(round(end * info.samplerate)))
    if e <= s:
        raise VoiceprintError(f"无效音频区间：[{start}, {end})")
    wav, _sr = sf.read(wav_path, start=s, stop=e, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.ascontiguousarray(wav, dtype=np.float32)


# ---------------- 引擎 ----------------

class VoiceprintEngine:
    """声纹提取与分段比对（模型懒加载，_embed 为测试接缝）。"""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def extract(self, wav_path: str, start: Optional[float] = None,
                end: Optional[float] = None) -> np.ndarray:
        """整文件（或指定区间）→ L2 归一化 embedding。注册样本入口。"""
        info = sf.info(wav_path)
        total = info.frames / info.samplerate
        s = 0.0 if start is None else max(0.0, start)
        e = total if end is None else min(end, total)
        cs, ce = _cap_span(s, e)
        wav = _load_span(wav_path, cs, ce)
        if float(np.max(np.abs(wav))) < _SILENT_PEAK:
            raise VoiceprintError("声纹样本几乎为静音，请更换清晰的人声样本")
        return self._embed([wav])[0]

    def _embed(self, wavs: list[np.ndarray]) -> np.ndarray:
        """(N, T) 波形批 → (N, 192) L2 归一化 embedding。

        批内变长时右 pad 到最长，并以 wav_lens 相对长度告知 ECAPA
        有效语音长度（pad 部分被注意力 mask 排除，不影响嵌入）。
        """
        import torch

        model = _get_ecapa(self.device)
        lengths = torch.tensor([len(w) for w in wavs], dtype=torch.float32)
        max_len = int(lengths.max())
        batch = np.zeros((len(wavs), max_len), dtype=np.float32)
        for i, w in enumerate(wavs):
            batch[i, : len(w)] = w
        tensor = torch.tensor(batch, device=self.device)
        wav_lens = lengths / max_len
        emb = model.encode_batch(tensor, wav_lens)
        arr = emb.detach().cpu().numpy()
        if arr.ndim == 3:  # speechbrain 返回 (N, 1, D)
            arr = arr[:, 0, :]
        return _normalize(arr.astype(np.float32))

    def classify_segments(
        self,
        wav_path: str,
        segments: list[tuple[float, float]],
        profile: VoiceProfile,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> list[SegmentVerdict]:
        """逐段计算与双音色 Profile 的余弦相似度。

        段居中截取 ≤30s 后按累计 ≤60s 分批前向；返回与 segments 等长的判定。
        """
        if profile.speak is None and profile.sing is None:
            raise VoiceprintError(f"主播「{profile.name}」的声纹 Profile 为空")

        refs, labels = [], []
        if profile.speak is not None:
            refs.append(profile.speak)
            labels.append("speak")
        if profile.sing is not None:
            refs.append(profile.sing)
            labels.append("sing")
        matrix = np.stack(refs)  # (K, D)

        verdicts: list[SegmentVerdict] = []
        batch: list[tuple[float, float, np.ndarray]] = []
        acc_seconds = 0.0
        total = len(segments)

        def flush() -> None:
            nonlocal batch, acc_seconds
            if not batch:
                return
            embs = self._embed([w for _, _, w in batch])  # (N, D)
            sims = embs @ matrix.T                         # (N, K)
            for (s, e, _w), row in zip(batch, sims):
                by = {lab: float(row[i]) for i, lab in enumerate(labels)}
                verdicts.append(SegmentVerdict(
                    start=s, end=e,
                    speak_sim=by.get("speak"), sing_sim=by.get("sing"),
                ))
            if on_progress:
                on_progress(min(1.0, len(verdicts) / total if total else 1.0))
            batch = []
            acc_seconds = 0.0

        for s, e in segments:
            cs, ce = _cap_span(s, e)
            batch.append((s, e, _load_span(wav_path, cs, ce)))
            acc_seconds += ce - cs
            if acc_seconds >= _BATCH_SECONDS:
                flush()
        flush()
        return verdicts


# ---------------- Profile 持久化 ----------------

def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise VoiceprintError("主播名不能为空")
    if len(name) > _NAME_MAX:
        raise VoiceprintError(f"主播名过长（≤{_NAME_MAX} 字符）")
    if name.startswith("."):
        raise VoiceprintError("主播名不能以点开头")
    bad = [c for c in name if c in _BAD_NAME_CHARS]
    if bad:
        raise VoiceprintError(f"主播名含非法字符：{''.join(bad)}")
    return name


def save_profile(
    name: str,
    speak_path: str | Path | None,
    sing_path: str | Path | None = None,
    profiles_dir: str | Path = "profiles",
    device: str = "cpu",
) -> Path:
    """注册主播声纹：样本转 16k WAV → 提取双音色 embedding → 持久化。"""
    name = _validate_name(name)
    if not speak_path:
        raise VoiceprintError("缺少说话样本（参考音频 A，10~30s 清晰干声）")
    out_dir = Path(profiles_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = VoiceprintEngine(device=device)
    with tempfile.TemporaryDirectory(prefix="vp_enroll_") as td:
        speak_wav = extract_wav(speak_path, td)
        speak_emb = engine.extract(str(speak_wav))
        sing_emb = None
        if sing_path:
            sing_wav = extract_wav(sing_path, td)
            sing_emb = engine.extract(str(sing_wav))

    path = out_dir / f"{name}.npy"
    np.save(path, {"name": name, "speak": speak_emb, "sing": sing_emb})
    return path


def load_profile(name: str, profiles_dir: str | Path = "profiles") -> VoiceProfile:
    """加载已保存的主播声纹。"""
    name = _validate_name(name)
    path = Path(profiles_dir) / f"{name}.npy"
    if not path.is_file():
        raise VoiceprintError(
            f"未找到主播「{name}」的声纹 Profile（{path}），请先在界面创建"
        )
    try:
        data = np.load(path, allow_pickle=True).item()
        return VoiceProfile(
            name=name,
            speak=_normalize(data["speak"]) if data.get("speak") is not None else None,
            sing=_normalize(data["sing"]) if data.get("sing") is not None else None,
        )
    except VoiceprintError:
        raise
    except Exception as exc:
        raise VoiceprintError(f"声纹 Profile 损坏：{path}") from exc


def save_library_speaker(
    name: str,
    embedding: np.ndarray,
    profiles_dir: str | Path = "profiles",
) -> Path:
    """把已发现的说话人声纹（簇中心 embedding）直接存入声纹库。

    用于说话人分离流程：用户试听验证后命名入库，后续文件即可自动匹配。
    """
    name = _validate_name(name)
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.ndim != 1 or emb.size == 0:
        raise VoiceprintError("声纹向量无效（需一维非空数组）")
    emb = _normalize(emb)
    out_dir = Path(profiles_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npy"
    np.save(path, {"name": name, "speak": emb, "sing": None})
    return path


def list_profiles(profiles_dir: str | Path = "profiles") -> list[str]:
    """已注册主播名列表（文件名去掉 .npy）。"""
    d = Path(profiles_dir)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.npy"))
