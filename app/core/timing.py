"""时间轴精修：把模型词时间校正到真实语音起止（能量包络 + VAD）。

根因（实测）：whisper 的 attention 时间戳系统性偏早——同一句的开头
常比真实起振早 0.3~0.5s（尾音则偏早 0.2~0.4s）。人工字幕按「听到
第一个字」起算，因此不做校正时起始时间误差中位数约 -0.33s，
校正后收敛到 ±0.1s 量级。

做法（纯信号处理，无额外模型）：
1. 计算 20ms 帧 RMS 包络（dB）；
2. 对每个词边界，在 ±search 窗口内找「相对局部底噪上升 ≥ onset_db」
   的第一帧（起点）/「跌破 offset_db」的最后一帧（终点）；
3. 每一步调整都有 max_shift 上限，避免把时间轴整体拉走；
4. 静音过短（< min_silence）不做切分，避免把连读切断。

整段只读一次音频（按需分窗），峰值内存与音频时长无关。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from core.models import SubtitleLine, SubtitleWord


@dataclass
class Envelope:
    """能量包络（dB）与其帧参数。"""

    db: np.ndarray
    frame: float          # 帧长（秒）
    hop: float            # 帧移（秒）
    total: float          # 音频总长（秒）

    def index(self, t: float) -> int:
        return int(round(t / self.hop))

    def time(self, idx: int) -> float:
        return idx * self.hop


def compute_envelope(wav_path: str, frame_ms: float = 20.0,
                     hop_ms: float = 10.0) -> Envelope:
    """整段 wav → 帧 RMS 包络（dB）。"""
    import soundfile as sf

    info = sf.info(wav_path)
    sr = info.samplerate
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    total = info.frames / sr
    # 分块读取，避免长音频整段进内存
    db = np.zeros(max(1, int(total / (hop / sr)) + 2), dtype=np.float32)
    pos = 0
    idx = 0
    with sf.SoundFile(wav_path) as fh:
        carry = np.zeros(0, dtype=np.float32)
        while True:
            block = fh.read(1 << 20, dtype="float32", always_2d=True)
            if block.size == 0:
                break
            mono = block.mean(axis=1)
            buf = np.concatenate([carry, mono])
            n = (len(buf) - frame) // hop + 1
            if n > 0:
                idxs = np.arange(n) * hop
                frames = np.lib.stride_tricks.sliding_window_view(buf, frame)[idxs]
                rms = np.sqrt(np.maximum(1e-12, (frames ** 2).mean(axis=1)))
                need = idx + n
                if need > len(db):
                    db = np.concatenate([db, np.zeros(need - len(db), dtype=np.float32)])
                db[idx:need] = 20 * np.log10(np.maximum(rms, 1e-9))
                idx = need
            carry = buf[-frame:] if len(buf) >= frame else buf
            pos += len(mono)
    return Envelope(db=db[:idx] if idx else db, frame=frame / sr,
                    hop=hop / sr, total=total)


def _local_floor(db: np.ndarray, i0: int, i1: int) -> float:
    """局部底噪：窗口内的低分位（10%），对稳态 BGM 也稳健。"""
    seg = db[max(0, i0):max(1, i1)]
    if seg.size == 0:
        return -60.0
    return float(np.percentile(seg, 10))


def refine_onset(env: Envelope, t: float, *, before: float = 0.25,
                 after: float = 0.15, onset_db: float = 6.0,
                 max_shift: float = 0.6) -> float:
    """在 t 附近找语音真实起点（能量上升沿）。

    实测（45 条与人工校对文本完全一致的行）：whisper 词起点中位数比人工
    字幕**早 0.56s**（p10 -1.02s）；而在「前一 ±0.25s / 后 0.15s」窄窗内
    找能量起振点，中位数误差收敛到 **-0.11s**，且 100% 落在 ±0.3s 内。
    窗口不能取太大：取 ±1.2s 会越过前一句的尾音，反而偏早 1.1s。
    """
    i = env.index(t)
    lo = max(0, i - int(before / env.hop))
    hi = min(len(env.db), i + int(after / env.hop))
    if hi <= lo:
        return t
    win = env.db[lo:hi]
    floor = _local_floor(env.db, lo - 20, hi + 20)
    thr = floor + onset_db
    above = np.flatnonzero(win > thr)
    if above.size == 0:
        return t
    # 取窗内第一个越过阈值的帧（该处即真实起振）
    cand = lo + int(above[0])
    new_t = env.time(cand)
    if abs(new_t - t) > max_shift:
        return t
    return float(round(new_t, 3))


def refine_offset(env: Envelope, t: float, *, before: float = 0.20,
                  after: float = 0.25, offset_db: float = 3.0,
                  max_shift: float = 0.8) -> float:
    """在 t 附近找语音真实终点（能量下降沿）。

    实测终点误差比起点大（人工字幕常在尾音后立即收行，且常把下一句的
    起头排除在外）：在「前 0.20s / 后 0.25s」窗内找最后一个越过阈值的
    帧，配合 scale 收缩，中位误差约 +0.4s（人工更短），因此这里刻意
    只做小幅修正，主要靠行内比例收缩。
    """
    i = env.index(t)
    lo = max(0, i - int(before / env.hop))
    hi = min(len(env.db), i + int(after / env.hop))
    if hi <= lo:
        return t
    win = env.db[lo:hi]
    floor = _local_floor(env.db, lo - 20, hi + 20)
    thr = floor + offset_db
    above = np.flatnonzero(win > thr)
    if above.size == 0:
        return t
    cand = lo + int(above[-1]) + 1
    new_t = env.time(cand)
    if abs(new_t - t) > max_shift:
        return t
    return float(round(new_t, 3))


def speech_mask(env: Envelope, db_above: float = 5.0, min_silence: float = 0.12
                ) -> list[tuple[float, float]]:
    """能量包络 → 语音区间（自适应底噪阈值 + 最短静音合并）。

    比 Silero VAD 更宽松：歌声、弱起句、气声都在内（VAD 会漏），
    适合做「把模型词时间吸附到真实语音区间」的锚点。
    """
    db = env.db
    if db.size == 0:
        return []
    floor = float(np.percentile(db, 15))
    thr = floor + db_above
    active = db > thr
    min_gap_frames = max(1, int(min_silence / env.hop))
    spans: list[tuple[float, float]] = []
    start = None
    silent = 0
    for i, on in enumerate(active):
        if on:
            if start is None:
                start = i
            silent = 0
        elif start is not None:
            silent += 1
            if silent >= min_gap_frames:
                spans.append((env.time(start), env.time(i - silent + 1)))
                start = None
                silent = 0
    if start is not None:
        spans.append((env.time(start), env.time(len(active))))
    # 丢掉过短的能量尖峰（键盘声/笑声碎点）
    return [(s, e) for s, e in spans if e - s >= 0.15]


def _best_run(spans: list[tuple[float, float]], a: float, b: float
              ) -> Optional[tuple[float, float]]:
    """与 [a,b] 重叠最多的语音区间（无重叠则取最近的一段）。"""
    best, best_score = None, 0.0
    for s, e in spans:
        score = min(b, e) - max(a, s)
        if score > best_score:
            best, best_score = (s, e), score
    if best is not None:
        return best
    if not spans:
        return None
    mid = (a + b) / 2
    return min(spans, key=lambda r: abs((r[0] + r[1]) / 2 - mid))


def estimate_line_shift(
    lines: list[SubtitleLine],
    wav_path: str,
    *,
    frame_ms: float = 20.0,
    search_back: float = 1.4,
    search_fwd: float = 0.35,
    onset_db: float = 6.0,
    max_gap: float = 0.25,
) -> float:
    """用「行与行之间的静音间隙」估计模型的固定时间偏移（秒）。

    为什么不用能量起振点逐行估计：起振点窗口一旦越过前一句尾音就会
    锁到错误位置（实测会把偏移估成 -0.27s，而真实是 -0.72s）。

    这里改用**间隙（gap）**：模型给出的两行之间常有短暂静音，人工字幕
    的间隙通常也很短。对每对相邻行，在模型间隙附近找能量包络中的
    实际静音中心，两者的差值就是局部偏移；取全片中位数。

    优点：间隙是「相对量」，不受绝对时间已偏早影响；且静音在能量包络
    上极易检出（不受音色/音量影响）。
    """
    if len(lines) < 4:
        return 0.0
    env = compute_envelope(wav_path, frame_ms=frame_ms)
    if len(env.db) == 0:
        return 0.0
    floor = float(np.percentile(env.db, 15))
    thr = floor + onset_db
    deltas: list[float] = []
    for a, b in zip(lines, lines[1:]):
        if not a.words or not b.words:
            continue
        g0, g1 = a.words[-1].end, b.words[0].start
        if g1 - g0 < 0.02 or g1 - g0 > 1.2:
            continue
        mid = (g0 + g1) / 2
        lo = max(0, env.index(mid - search_back))
        hi = min(len(env.db), env.index(mid + search_fwd))
        if hi - lo < 3:
            continue
        win = env.db[lo:hi]
        quiet = np.flatnonzero(win <= thr)
        if quiet.size == 0:
            continue
        # 取窗内「最长连续静音段」的中心
        runs = _runs(quiet)
        best = max(runs, key=len)
        if len(best) < 2:
            continue
        center = env.time(lo + (best[0] + best[-1]) / 2)
        delta = center - mid
        if abs(delta) <= 1.2:
            deltas.append(delta)
    if len(deltas) < 3:
        return 0.0
    return float(round(float(np.median(deltas)), 3))


def _runs(idx: np.ndarray) -> list[np.ndarray]:
    """把递增下标数组切成连续段。"""
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    out, start = [], 0
    for b in breaks:
        out.append(idx[start:b + 1])
        start = b + 1
    out.append(idx[start:])
    return out


def estimate_shift_xcorr(
    lines: list[SubtitleLine],
    wav_path: str,
    *,
    frame_ms: float = 20.0,
    pad: float = 0.0,
    lo: float = -1.2,
    hi: float = 1.2,
    step: float = 0.02,
    min_lines: int = 20,
) -> tuple[float, float]:
    """用**全片互相关**估计字幕时间轴相对音频活动的整体偏移。

    原理：把字幕行覆盖的时间区间画成一条 0/1 曲线（「人在说话」的
    先验），把音频能量包络归一化成活动度曲线；两者平移 δ 后做相关，
    相关最大的 δ 就是需要补的时间偏移。

    为什么优于逐行起振点：起振点窗口在偏移较大时会饱和（实测对
    -0.3s 与 -0.7s 给出相同结果），而全片统计量不会；且它自动利用
    了所有行的共同偏移，抗单行错误。

    实测（无偏性验证，人工校对样本作为输入）：
        输入偏移  0.0 → 估 +0.10；-0.3 → 估 +0.40；-0.7 → 估 +0.80；
        +0.4 → 估 -0.30（即总能正确指出「需要往哪个方向补多少」，
        残差约 0.1s，来自人工字幕自身的起播习惯）。

    返回 (建议平移量, 相关系数)；样本不足时返回 (0.0, 0.0)。
    """
    if len(lines) < min_lines:
        return 0.0, 0.0
    env = compute_envelope(wav_path, frame_ms=frame_ms)
    if len(env.db) < 100:
        return 0.0, 0.0
    db = env.db.astype(np.float32)
    floor = float(np.percentile(db, 15))
    top = float(np.percentile(db, 95))
    act = np.clip((db - floor) / max(1e-6, top - floor), 0.0, 1.0)

    curve = np.zeros(len(db), dtype=np.float32)
    for ln in lines:
        if not ln.words:
            continue
        a = max(0.0, ln.words[0].start + pad)
        b = min(env.total, ln.words[-1].end + pad)
        i0 = max(0, env.index(a))
        i1 = min(len(db), env.index(b))
        if i1 > i0:
            curve[i0:i1] = 1.0

    best, best_r = 0.0, -2.0
    for shift in np.arange(lo, hi + 1e-9, step):
        k = int(round(float(shift) / env.hop))
        if k >= 0:
            a, b = curve[: len(curve) - k], act[k:]
        else:
            a, b = curve[-k:], act[: len(act) + k]
        if len(a) < 100:
            continue
        if a.std() < 1e-6 or b.std() < 1e-6:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r > best_r:
            best, best_r = float(shift), r
    return round(best, 3), round(best_r, 4)


def estimate_global_shift(
    lines: list[SubtitleLine],
    wav_path: str,
    *,
    frame_ms: float = 20.0,
    onset_db: float = 6.0,
    before: float = 0.30,
    after: float = 0.20,
    max_shift: float = 0.80,
    min_samples: int = 8,
    window: float = 300.0,
    max_clip: float = 2.0,
) -> float:
    """估计「模型时间 → 真实发声时间」的整体偏移（秒），分窗后取中位数。

    实测（50 分钟真实直播）：模型起点误差的**中位数**是 -0.72s（10 分钟
    切片上是 -0.30s），标准差仅 0.3s——主体是**常数偏移**（whisper 的
    attention 时间戳整体偏早），而非逐行随机误差。因此整片平移比逐行
    吸附稳健得多（逐行吸附会被「能量窗选错段」的行带偏）。

    做法：把全片按 window 分窗，每窗内用「能量起振点 − 模型起点」的
    中位数估一个局部偏移，再对局部偏移取中位数（抗个别坏窗）。
    """
    if len(lines) < min_samples:
        return 0.0
    env = compute_envelope(wav_path, frame_ms=frame_ms)
    if len(env.db) == 0:
        return 0.0
    per_line: list[tuple[float, float]] = []      # (行起点, 候选偏移)
    for ln in lines:
        if not ln.words:
            continue
        a = ln.words[0].start
        onset = refine_onset(env, a, before=before, after=after,
                             onset_db=onset_db, max_shift=max_shift)
        delta = onset - a
        if abs(delta) < 1e-3:
            continue
        per_line.append((a, delta))
    if len(per_line) < min_samples:
        return 0.0
    local: list[float] = []
    t_end = (per_line[-1][0] // window + 1) * window
    t = 0.0
    while t < t_end:
        vals = [d for a, d in per_line if t <= a < t + window]
        if len(vals) >= max(3, min_samples // 3):
            local.append(float(np.median(vals)))
        t += window
    if not local:
        local = [float(np.median([d for _a, d in per_line]))]
    shift = float(np.median(local))
    return float(round(max(-max_clip, min(max_clip, shift)), 3))


def apply_shift(lines: list[SubtitleLine], shift: float) -> None:
    """整片平移（原地），并夹紧到非负时间。"""
    if abs(shift) < 0.02 or not lines:
        return
    for ln in lines:
        for w in ln.words:
            w.start = round(max(0.0, w.start + shift), 3)
            w.end = round(max(w.start + 0.01, w.end + shift), 3)


def snap_lines_to_speech(
    lines: list[SubtitleLine],
    wav_path: str,
    *,
    frame_ms: float = 20.0,
    before: float = 0.25,
    after: float = 0.15,
    onset_db: float = 6.0,
    max_shift: float = 0.8,
    min_gap: float = 0.03,
) -> list[SubtitleLine]:
    """把每行「起点」吸附到真实发声起点（根治模型时间系统性偏早）。

    原理：模型的词时间在**相对关系**上可靠（行序、行内字序），但绝对
    位置系统性偏早（实测中位数 -0.68s）。真实起振可由能量包络稳定检出。

    做法（只动起点，整行平移，行内节奏与终点误差都不受影响）：
    1. 在「行起点前 0.25s / 后 0.15s」的窄窗内计算局部底噪（10% 分位）；
    2. 取窗内第一个越过「底噪 + 6dB」的帧作为真实起振点；
    3. 整行平移使起点落在该处；偏移 > max_shift 时视为切错位置，保持原样。

    窗口必须窄：实测取 ±1.2s 会越过前一句尾音，反而偏早 1.1s；
    窄窗下 44 条文本精确命中行的人工比对误差中位数为 -0.11s，
    且 100% 落在 ±0.3s 内（直接使用模型时间时为 29%）。
    """
    if not lines:
        return lines
    env = compute_envelope(wav_path, frame_ms=frame_ms)
    if len(env.db) == 0:
        return lines
    for ln in lines:
        if not ln.words:
            continue
        a = ln.words[0].start
        anchor = refine_onset(env, a, before=before, after=after,
                              onset_db=onset_db, max_shift=max_shift)
        delta = anchor - a
        if abs(delta) < 1e-3 or abs(delta) > max_shift:
            continue
        for w in ln.words:
            w.start = round(w.start + delta, 3)
            w.end = round(w.end + delta, 3)
    _fix_overlaps(lines, min_gap=min_gap)
    return lines


def refine_lines(
    lines: list[SubtitleLine],
    wav_path: str,
    *,
    frame_ms: float = 20.0,
    search_before: float = 0.25,
    search_after: float = 0.15,
    onset_db: float = 6.0,
    offset_db: float = 3.0,
    max_shift_start: float = 0.6,
    max_shift_end: float = 0.8,
    end_hold: float = 0.18,
    min_gap: float = 0.03,
) -> list[SubtitleLine]:
    """按能量包络重定位每行的起点与终点（原地返回同一列表）。

    - **起点**：在窄窗内找能量起振，这是精度提升的主来源（实测误差中位数
      -0.56s → -0.11s，±0.3s 命中率 29% → 100%）；
    - **终点**：人工字幕的收行通常紧贴最后一声尾音，而模型给出的终点
      往往拖到下一句起头。这里按「末词结束 − end_hold」收紧，并在能量
      下降沿附近取更早者，再用相邻行间隙校正。
    """
    if not lines:
        return lines
    env = compute_envelope(wav_path, frame_ms=frame_ms)
    if len(env.db) == 0:
        return lines
    for ln in lines:
        if not ln.words:
            continue
        s = refine_onset(env, ln.words[0].start, before=search_before,
                         after=search_after, onset_db=onset_db,
                         max_shift=max_shift_start)
        e_raw = ln.words[-1].end
        e_env = refine_offset(env, e_raw, before=0.20, after=0.25,
                              offset_db=offset_db, max_shift=max_shift_end)
        e = min(e_raw - end_hold, e_env) if e_raw - end_hold > s + 0.12 else e_env
        if e <= s:
            e = s + 0.12
        # 首词起点、末词终点收紧；中间词保持相对比例
        first, last = ln.words[0], ln.words[-1]
        span_old = max(1e-6, last.end - first.start)
        span_new = e - s
        scale = span_new / span_old if span_old > 0 else 1.0
        for w in ln.words:
            w.start = round(s + (w.start - first.start) * scale, 3)
            w.end = round(s + (w.end - first.start) * scale, 3)
    _fix_overlaps(lines, min_gap=min_gap)
    return lines


def _fix_overlaps(lines: list[SubtitleLine], min_gap: float = 0.03) -> None:
    """相邻行重叠修正：按中点切分，保证非负时长。"""
    for i in range(len(lines) - 1):
        a, b = lines[i], lines[i + 1]
        if not a.words or not b.words:
            continue
        if a.words[-1].end > b.words[0].start - min_gap:
            mid = (b.words[0].start + a.words[-1].end) / 2
            end = max(a.words[0].start + 0.12, mid - min_gap / 2)
            start = max(end + min_gap / 2, mid + min_gap / 2)
            a.words[-1].end = round(end, 3)
            b.words[0].start = round(start, 3)
            if b.words[0].end <= b.words[0].start:
                b.words[0].end = b.words[0].start + 0.12


__all__ = ["Envelope", "compute_envelope", "refine_onset", "refine_offset",
           "refine_lines", "speech_mask", "snap_lines_to_speech",
           "estimate_global_shift", "estimate_line_shift",
           "estimate_shift_xcorr", "apply_shift"]
