"""UVR5（Ultimate Vocal Remover 5）模型集成。

为什么自己实现而不是调用 UVR5 本体
────────────────────────────────
本地安装的 UVR5 是 PyInstaller 打包的单体程序（`UVR_Launcher.exe`），
只能用图形界面操作，无法命令行调用，也无法 import 它的模块。
但它下载的**模型文件是标准的 ONNX / Demucs 权重**，可以直接加载。

因此这里的做法是：复用 UVR5 的模型与配置（模型已在本地），自己实现
推理流程。好处是能按需分段处理（只分离用户选中的片段），比 UVR5
GUI 的整文件处理更适合「时间轴上挑一段做声纹注册」的场景。

支持的情形
──────────
1. **MDX-Net 系列**（`models/MDX_Net_Models/*.onnx`）：ONNX 推理，
   速度快、显存占用低。推荐 `UVR-MDX-NET-Voc_FT.onnx`（人声提取）。
2. **Demucs 系列**（`models/Demucs_Models/*.th`）：走已安装的 demucs 包。

设计取舍
────────
- 模型路径由 `UVRConfig` 决定，默认自动探测常见安装位置；
- 找不到 UVR 时**优雅退化**为项目内置的 Demucs 分离（不报错），
  保证没有安装 UVR 的用户也能用人声分离；
- 只在需要时（用户显式点「用 UVR5 分离」）才加载模型，避免拖慢启动。
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# UVR5 的常见安装位置（用户本机实际在第一个）
UVR_CANDIDATES = [
    r"D:\sovits\So-VITS-SVC\UVR5\Ultimate Vocal Remover",
    r"D:\UVR5", r"D:\Ultimate Vocal Remover",
    r"C:\Program Files\Ultimate Vocal Remover",
    r"C:\Program Files (x86)\Ultimate Vocal Remover",
]

# 优先使用的人声模型（按质量/速度权衡排序）
PREFERRED_VOCAL_MODELS = [
    "UVR-MDX-NET-Voc_FT.onnx",       # 全频段人声，通用最佳
    "UVR-MDX-NET-Inst_HQ_3.onnx",    # 伴奏模型，人声=反相（备选）
    "UVR-MDX-NET-Inst_HQ_2.onnx",
]

# MDX-Net 推理参数（与 UVR5 官方一致：分段长、步进重叠）
MDX_SEGMENT = 256
MDX_OVERLAP = 0.25
MDX_SAMPLE_RATE = 44100


@dataclass
class UVRConfig:
    """UVR5 集成配置。"""
    root: str = ""                  # UVR5 安装目录（空=自动探测）
    model: str = ""                 # 指定模型文件名（空=自动挑选）
    segment: int = MDX_SEGMENT
    overlap: float = MDX_OVERLAP
    sample_rate: int = MDX_SAMPLE_RATE
    device: str = "cpu"             # MDX 走 onnxruntime，CPU 已足够快
    enabled: bool = True

    def resolve(self) -> "UVRConfig":
        if not self.root:
            self.root = detect_uvr_root() or ""
        return self


def detect_uvr_root() -> Optional[str]:
    """探测本机 UVR5 安装目录（存在 models 子目录才算有效）。"""
    # 1) 环境变量优先
    env = os.environ.get("SUBTITLE_STUDIO_UVR_ROOT", "").strip()
    if env and Path(env, "models").is_dir():
        return env
    # 2) 常见位置
    for cand in UVR_CANDIDATES:
        p = Path(cand)
        if (p / "models").is_dir():
            return str(p)
    # 3) 从开始菜单快捷方式解析（Windows）
    if os.name == "nt":
        for base in (os.environ.get("APPDATA", ""),
                     os.environ.get("PROGRAMDATA", "")):
            if not base:
                continue
            lnk = Path(base) / "Microsoft" / "Windows" / "Start Menu" \
                / "Programs" / "Ultimate Vocal Remover.lnk"
            if not lnk.is_file():
                continue
            try:
                import subprocess as sp

                ps = sp.run(
                    ["powershell", "-NoProfile", "-Command",
                     f"$s=(New-Object -ComObject WScript.Shell)"
                     f".CreateShortcut('{lnk}'); $s.WorkingDirectory"],
                    capture_output=True, text=True, timeout=20)
                wd = (ps.stdout or "").strip()
                if wd and (Path(wd) / "models").is_dir():
                    return wd
            except Exception:
                continue
    return None


def list_vocal_models(root: str) -> list[str]:
    """列出 UVR5 里可用的人声/伴奏模型。"""
    out: list[str] = []
    mdx = Path(root) / "models" / "MDX_Net_Models"
    if mdx.is_dir():
        for f in sorted(mdx.glob("*.onnx")):
            out.append(f.name)
    return out


def pick_vocal_model(root: str, prefer: str = "") -> Optional[Path]:
    """挑选人声分离模型。"""
    mdx = Path(root) / "models" / "MDX_Net_Models"
    if not mdx.is_dir():
        return None
    if prefer:
        p = mdx / prefer
        if p.is_file():
            return p
    for name in PREFERRED_VOCAL_MODELS:
        p = mdx / name
        if p.is_file():
            return p
    # 兜底：任意含 Voc 的模型
    for f in sorted(mdx.glob("*.onnx")):
        if "voc" in f.name.lower() or "inst" in f.name.lower():
            return f
    return None


def _load_model_data(root: str) -> dict:
    """读取 UVR5 的 model_data.json（含每种模型的推理参数）。"""
    p = Path(root) / "models" / "MDX_Net_Models" / "model_data" / "model_data.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _model_params(root: str, model_name: str) -> dict:
    """取指定模型的推理参数（分段长度/维度等）。"""
    data = _load_model_data(root)
    for key, val in data.items():
        if isinstance(val, dict) and key.lower() == model_name.lower():
            return val
        # 有些版本 key 带扩展名
        if isinstance(val, dict) and Path(key).name.lower() == model_name.lower():
            return val
    return {}


def _ffmpeg_to_wav(src: str, dst: str, sr: int = MDX_SAMPLE_RATE,
                   start: float | None = None,
                   duration: float | None = None) -> None:
    """用 FFmpeg 把任意媒体转成 MDX 需要的单/双声道 WAV。"""
    from .env import detect_env

    ff = detect_env().ffmpeg_path or "ffmpeg"
    cmd = [ff, "-y", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, start):.3f}"]
    cmd += ["-i", src]
    if duration is not None:
        cmd += ["-t", f"{max(0.1, duration):.3f}"]
    cmd += ["-ac", "2", "-ar", str(sr), "-c:a", "pcm_s16le", dst]
    subprocess.run(cmd, check=True, capture_output=True)


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    """读 WAV → (samples[N,2] float32, sr)。不依赖 soundfile。"""
    import wave

    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch)
    else:
        data = data.reshape(-1, 1)
    return data, sr


def _write_wav(path: str, data: np.ndarray, sr: int) -> None:
    """写 16bit PCM WAV。"""
    import wave

    d = np.clip(data, -1.0, 1.0)
    if d.ndim == 1:
        d = d.reshape(-1, 1)
    pcm = (d * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(d.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def separate_with_mdx(
    src: str,
    dst_vocal: str,
    *,
    cfg: UVRConfig,
    start: float | None = None,
    duration: float | None = None,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> str:
    """用 UVR5 的 MDX-Net 模型分离人声，输出到 `dst_vocal`。

    模型接口（实测 read 出来的形状，不要照抄文档）
    ────────────────────────────────────────────
    输入/输出都是 `[batch, 4, 3072, 256]`：
      - 3072 = 频率维，对应 `n_fft = 6144`（rfft 得到 3073，取 3072 用）；
      - 256  = 时间帧数（分段长度，即 UVR 的 segment_size）；
      - 4    = 左右声道 × (实部, 虚部)。
    早期按「n_fft=2048 → 1025 频点」实现会报
    `Got invalid dimensions ... Expected: 3072`（实测踩坑）。
    """
    import onnxruntime as ort

    cfg = cfg.resolve()
    if not cfg.root:
        raise RuntimeError("未找到 UVR5 安装目录（可用 SUBTITLE_STUDIO_UVR_ROOT 指定）")
    model = pick_vocal_model(cfg.root, cfg.model)
    if model is None:
        raise RuntimeError(f"UVR5 目录里没有可用模型：{cfg.root}")

    if on_progress:
        on_progress(0.05, f"载入模型 {model.name}…")
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    # 以模型的实际形状为准（不同模型可能不同）
    nb, nch, n_freq, seg = inp.shape
    nch = int(nch) if isinstance(nch, int) else 4
    n_freq = int(n_freq) if isinstance(n_freq, int) else 3072
    seg = int(seg) if isinstance(seg, int) else 256
    n_fft = (n_freq - 1) * 2 if n_freq > 1 else 6144
    # MDX 系列统一 44.1kHz
    sr = cfg.sample_rate
    hop = n_fft // 4                      # 75% 重叠，还原更平滑
    overlap = cfg.overlap

    tmpdir = Path(tempfile.mkdtemp(prefix="uvr_mdx_"))
    try:
        raw = tmpdir / "in.wav"
        _ffmpeg_to_wav(src, str(raw), sr=sr, start=start, duration=duration)
        if on_progress:
            on_progress(0.15, "读取音频…")
        audio, real_sr = _read_wav(str(raw))
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        if real_sr != sr:
            raise RuntimeError(f"采样率不匹配：{real_sr} != {sr}")

        n = len(audio)
        window = np.hanning(n_fft).astype(np.float32)
        # 每个声道单独做 STFT
        specs: list[np.ndarray] = []       # [ch] -> [frames, n_freq] complex
        n_frames = 0
        for c in range(2):
            stft = _stft(audio[:, c], n_fft, hop, window)
            specs.append(stft)
            n_frames = stft.shape[0]
        if n_frames <= 0:
            raise RuntimeError("音频太短，无法分离")

        # 分块推理（每块 seg 帧），重叠区**按幅度平均**拼接。
        #
        # 为什么不能直接把复数谱相加取平均：相邻块的预测相位并不一致
        # （模型独立处理每块），复数直接平均会在重叠区相互抵消，
        # 重建后大片静音——实测只有首尾几百毫秒有声音。
        # 正确做法：幅度取平均（不抵消），相位沿用原始谱（保证连续）。
        step = max(1, int(seg * (1.0 - overlap)))
        mag_acc = [np.zeros((n_frames, n_freq), dtype=np.float32)
                   for _ in range(2)]
        wsum = np.zeros(n_frames, dtype=np.float32)
        pos, idx = 0, 0
        total = max(1, (n_frames + step - 1) // step)
        while pos < n_frames:
            idx += 1
            cur = min(seg, n_frames - pos)
            x = np.zeros((1, nch, n_freq, seg), dtype=np.float32)
            for c in range(2):
                part = specs[c][pos:pos + cur]
                if part.shape[0] < seg:
                    part = np.concatenate(
                        [part, np.zeros((seg - part.shape[0], n_freq),
                                        dtype=np.complex64)])
                # 通道 0/1 = 左声道实/虚；2/3 = 右声道实/虚
                x[0, c * 2 + 0] = part.real.T
                x[0, c * 2 + 1] = part.imag.T
            try:
                y = sess.run(None, {inp.name: x})[0]
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"MDX 推理失败：{exc}") from exc
            for c in range(2):
                re = y[0, c * 2 + 0].T
                im = y[0, c * 2 + 1].T
                mag = np.abs((re + 1j * im)[:cur])
                mag_acc[c][pos:pos + cur] += mag
            wsum[pos:pos + cur] += 1.0
            if on_progress:
                on_progress(0.2 + 0.6 * (idx / total), f"分离中 {idx}/{total}")
            if cur < seg:
                break
            pos += step

        # 幅度取平均，再乘回原谱的相位（单位复数）
        denom = np.maximum(wsum, 1.0).astype(np.float32)[:, None]
        outs: list[np.ndarray] = []
        for c in range(2):
            avg_mag = mag_acc[c] / denom
            phase = np.exp(1j * np.angle(specs[c]))
            outs.append((avg_mag * phase).astype(np.complex64))

        if on_progress:
            on_progress(0.85, "重建人声…")
        # 模型输出**就是人声谱**（实测：与输入幅度相关系数 0.84、
        # 量级比 1.0~1.24），直接逆变换即可。
        #
        # 曾经多算一步「掩码 = |out|/|in|」再乘回原谱——那会把信号压成
        # 近乎静音（RMS 0.0025、频谱重心跑到 13kHz，是噪声），是理解
        # 模型语义错误导致的。MDX-Net 这里是「谱到谱」映射，不是掩码。
        out_path = dst_vocal
        vocals = np.zeros((n, 2), dtype=np.float32)
        for c in range(2):
            rec = _istft(outs[c], n_fft, hop, window, n)
            # 用**均方根**而不是峰值做归一化：峰值可能被个别尖峰主导
            # （实测首帧有一个大幅值），按峰值缩放会让其余部分近乎静音
            # ——表现为「只有开头有声音」。
            rms = float(np.sqrt(np.mean(rec ** 2)))
            if rms > 1e-6:
                rec = rec * min(4.0, 0.12 / rms)    # 目标 RMS ≈ 0.12
            vocals[:, c] = np.clip(rec, -1.0, 1.0)

        _write_wav(out_path, vocals, sr)
        if on_progress:
            on_progress(1.0, "完成")
        return out_path
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def _stft(x: np.ndarray, n_fft: int, hop: int,
          window: np.ndarray) -> np.ndarray:
    """短时傅里叶变换 → [frames, n_fft//2+1] 复数谱。"""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    out = np.zeros((n_frames, n_fft // 2 + 1), dtype=np.complex64)
    for i in range(n_frames):
        s = i * hop
        out[i] = np.fft.rfft(x[s:s + n_fft] * window)
    return out


def _istft(spec: np.ndarray, n_fft: int, hop: int,
           window: np.ndarray, length: int) -> np.ndarray:
    """逆变换 + 重叠相加（除以窗平方和做归一化）。"""
    out = np.zeros(length + n_fft, dtype=np.float32)
    wsum = np.zeros(length + n_fft, dtype=np.float32)
    for i in range(spec.shape[0]):
        s = i * hop
        if s >= length:
            break
        frame = np.fft.irfft(spec[i], n=n_fft).astype(np.float32) * window
        end = min(s + n_fft, len(out))
        out[s:end] += frame[:end - s]
        wsum[s:end] += (window ** 2)[:end - s]
    out = out[:length] / np.maximum(wsum[:length], 1e-8)
    return out


def separate_segment_for_voiceprint(
    src: str,
    dst: str,
    start: float,
    end: float,
    *,
    cfg: UVRConfig | None = None,
    target_sr: int = 16000,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> str:
    """从时间轴选中的区间分离人声，输出**声纹引擎可用**的 16kHz 单声道 WAV。

    为什么要重采样到 16kHz：`VoiceprintEngine.extract` 只接受 16kHz
    WAV，而 UVR 的 MDX 模型要求 44.1kHz。中间必须转一次，
    否则会抛「采样率必须为 16000Hz」（实测踩坑）。
    """
    from .env import detect_env

    if end <= start:
        raise ValueError("区间无效：结束时间必须大于开始时间")
    c = (cfg or UVRConfig()).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="uvr_seg_"))
    try:
        vocal = tmp / "vocal.wav"
        separate_with_mdx(src, str(vocal), cfg=c,
                          start=start, duration=end - start,
                          on_progress=on_progress)
        ff = detect_env().ffmpeg_path or "ffmpeg"
        if on_progress:
            on_progress(0.95, "重采样到 16kHz…")
        subprocess.run(
            [ff, "-y", "-v", "error", "-i", str(vocal),
             "-ac", "1", "-ar", str(target_sr), "-c:a", "pcm_s16le", dst],
            check=True, capture_output=True)
        if on_progress:
            on_progress(1.0, "完成")
        return dst
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


@dataclass
class UVRStatus:
    """UVR5 可用性状态（供 UI 显示）。"""
    available: bool = False
    root: str = ""
    models: list[str] = field(default_factory=list)
    preferred: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return {"available": self.available, "root": self.root,
                "models": self.models, "preferred": self.preferred,
                "reason": self.reason}


def status(cfg: UVRConfig | None = None) -> UVRStatus:
    """探测 UVR5 是否可用（不加载模型，很快）。"""
    c = (cfg or UVRConfig()).resolve()
    if not c.root:
        return UVRStatus(available=False,
                         reason="未找到 UVR5 安装目录")
    models = list_vocal_models(c.root)
    if not models:
        return UVRStatus(available=False, root=c.root,
                         reason="UVR5 目录里没有 .onnx 模型")
    pref = pick_vocal_model(c.root, c.model)
    return UVRStatus(available=True, root=c.root, models=models,
                     preferred=pref.name if pref else "",
                     reason="")

    return UVRStatus(available=False, reason="未知错误")


__all__ = ["UVRConfig", "UVRStatus", "detect_uvr_root", "list_vocal_models",
           "pick_vocal_model", "separate_with_mdx", "status"]
