"""集中配置：算法阈值、模型选择、外部接口开关。

设计目标（针对评审报告的 R1/R2/G21）：
- 所有此前散落在各模块的魔数集中在此，可按语料标定；
- 支持环境变量覆盖（SUBTITLE_STUDIO_*），便于批量实验与部署定制；
- 默认值由人工校对样本（见 docs/EVALUATION.md）与实测基准标定，
  修改默认值前请先跑 `python -m tools.eval_cli` 回归。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


@dataclass
class SegmentConfig:
    """切行参数（默认值来自 545 行人工校对样本的统计与网格标定）。"""

    target_chars: float = 11.0
    max_chars: int = 17
    min_chars: int = 1
    max_duration: float = 8.5
    w_sentence: float = 3.0
    w_clause: float = 1.2
    w_pause: float = 0.8
    w_dev: float = 1.0
    w_duration: float = 0.5
    pause_gap: float = 0.28
    use_punctuation: bool = True   # 用标点模型提供断句线索（不进入字幕文本）
    # 强行在一行内部断开时用空格分隔（不引入逗号等标点）
    inner_space: bool = True


@dataclass
class AsrConfig:
    """语音识别参数。

    实测结论（Sample 前 10 分钟，CER 以人工校对文本为参考）：
    - large-v3(int8_float16) 明显优于 small：字符召回 0.660 vs 0.598；
    - 关闭 VAD 过滤比开启多召回 11 个百分点（0.770 vs 0.660）——
      faster-whisper 的 Silero VAD 会把歌声/弱起句判成非语音整段丢弃；
    - 因此默认 vad_filter=False，改用识别后处理剔除幻听。
    """

    engine: str = "whisper"            # whisper | funasr | <外部 provider 名>
    model: str = "large-v3"            # CT2 模型名或本地目录
    vad_filter: bool = False           # 默认关闭（见上）
    word_timestamps: bool = True
    # beam_size=1（贪心）——实测**又快又准**，反直觉但有依据。
    #
    # 在 10 分钟人工校对样本上对比（beam 5 vs 1）：
    # | beam | EXACT | TEXT  | TIME  | CER   | 召回  | ASR 耗时 |
    # |-----:|------:|------:|------:|------:|------:|--------:|
    # |    5 | 0.156 | 0.213 | 0.281 | 0.317 | 0.788 |   445s  |
    # |    1 | 0.189 | 0.257 | 0.289 | 0.301 | 0.792 |   247s  |
    #
    # beam=1 在**全部指标上更好**，且快 1.8 倍。原因：长音频直播里
    # beam search 更容易陷入重复循环与幻听（beam 越大，越倾向「凑出
    # 通顺的句子」而不是忠实转写），而这些错误会连带影响切行。
    beam_size: int = 1
    best_of: int = 1
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    # 长音频分块：**单块 120 秒**，块间重叠 10 秒保证跨块语句完整。
    #
    # 为什么是 120 秒而不是更长（重要实测结论）：
    # whisper 在长块上会**静默跳过内容**——它把整块解码后按内部 30 秒
    # 窗口推进，遇到音乐/长静音会直接跳过若干窗口而不报错。实测同一
    # 素材用 1200 秒分块时 CER 0.721，改用 120 秒分块后 CER 0.301
    # （且行数从 785 降到接近参考的 545，说明多出来的都是幻听）。
    #
    # 因此默认 120 秒；块越多单块越快，总耗时反而更短。
    chunk_seconds: float = 120.0
    chunk_overlap: float = 10.0
    # 幻听/静音过滤（whisper 在纯音乐上会生成文本）
    max_compression_ratio: float = 2.4
    drop_no_speech_above: float = 0.90
    min_avg_logprob: float = -1.2
    # 热词提示（whisper initial_prompt / funasr hotword）
    hotwords: str = ""
    language: str = "auto"


@dataclass
class TimingConfig:
    """时间轴精细化（针对 whisper 词时间系统性偏早的根因修正）。"""

    refine: bool = True
    # 能量包络参数
    frame_ms: float = 20.0
    onset_db: float = 6.0          # 相对局部底噪的起振阈值
    offset_db: float = 4.0
    search_before: float = 0.45    # 在词起点前搜索真实起振的范围
    search_after: float = 0.35
    min_silence: float = 0.12      # 短于此的静音不切分
    # 行首/行尾对齐的最大调整量（防止把时间轴整体拉坏）
    max_shift_start: float = 0.6
    max_shift_end: float = 0.6


@dataclass
class SongConfig:
    """歌曲处理：演唱检测 → 识曲/歌词检索 → 强制对齐。"""

    enabled: bool = False
    # 演唱块检测
    min_block_seconds: float = 25.0
    block_gap: float = 12.0
    # 识曲（Shazam，联网）
    use_shazam: bool = True
    shazam_snippet: float = 12.0
    shazam_probes: int = 3          # 每块探测位置数（开头/中段/后段）
    # 歌词检索：本地库优先，其次在线（LRCLIB / NetEase）
    use_online_lyrics: bool = True
    lyric_providers: tuple = ("local", "lrclib", "netease")
    # 强制对齐（FunASR 时间戳预测，离线可用）
    forced_align: bool = True
    align_modelscope_id: str = "iic/speech_timestamp_prediction-v1-16k-offline"
    # 识曲/检索失败时，用 ASR 文本在本地/在线歌词库中检索
    search_by_transcript: bool = True
    accept_score: float = 0.55

    # ---- 手动录入的本场演唱曲目 ----
    # 由用户显式给出「这场直播唱了哪些歌」，用于：
    #   1) 直接驱动歌词库检索（不必靠识曲猜）
    #   2) 缩小 BGM 判定的搜索范围，降低误判
    # 空白分隔的字符串或列表均可。
    manual_titles: tuple = ()
    # 给出清单后，是否只在该清单范围内判定 BGM（更保守，误判更少）
    restrict_bgm_to_manual: bool = False

    # ---- 歌词库校正（生成完成后执行）----
    # 用曲库歌词与识别文本做相似度匹配，高置信行优先采用歌词原文
    lyric_correct: bool = False
    lyric_library_dir: str = "lyrics"
    # 断句阶段就参考歌词（比事后替换更自然：行的时间与长度跟着歌词走）
    segment_by_lyrics: bool = True
    # 整体相似度低于此值认为「不是同一首歌」，不做校正
    lyric_doc_threshold: float = 0.55
    # 单行替换门槛：歌词行需明显更可靠才替换
    lyric_line_threshold: float = 0.75

    # ---- 弹幕旁路校正（生成完成后执行）----
    # 用弹幕/礼物/SC 的反复写法纠正专名（昵称、歌名、术语）
    danmaku_correct: bool = False
    danmaku_time_window: float = 12.0
    danmaku_min_corroboration: int = 2


@dataclass
class VoiceprintConfig:
    """声纹：只精确识别「当前主播」一个声纹。"""

    enabled: bool = False
    profile_name: str = ""
    threshold: float = 0.45         # 余弦相似度阈值（ECAPA 同人约 0.5~0.85）
    # 双音色：说话/唱歌分别建模，取 max（唱歌音色与说话差异大）
    use_singing_profile: bool = True
    auto_enroll_singing: bool = True   # 从高置信演唱段自学习唱歌声纹
    min_segment_seconds: float = 0.6
    # BGM 人声抑制：声纹 + 伴奏度/侧信号/响度变异的融合阈值
    bgm_filter: bool = True
    bgm_threshold: float = 0.35
    explain_bgm: bool = True        # 日志输出每段剔除原因（便于人工复核）
    # 处理策略：
    #   "mark"   —— 只标注「可能为BGM原唱」进复核清单，不删除（默认，零回归风险）
    #   "filter" —— 高置信度时直接剔除（阈值更严）
    #   "off"    —— 完全关闭
    bgm_policy: str = "mark"
    bgm_strict_threshold: float = 0.12   # filter 模式下才删除的强证据阈值
    # 细粒度判定：VAD 段先切成 ~5s 子窗再逐窗判定（避免整段 20~30s 一刀切）
    window_seconds: float = 5.0
    min_window_seconds: float = 1.0
    merge_gap: float = 0.35         # 相邻保留子窗的合并间隙


@dataclass
class ProviderConfig:
    """外部接口：允许用户接入自建 ASR / 歌词 / 声纹服务。

    环境变量：
        SUBTITLE_STUDIO_ASR_ENDPOINT   自定义识别服务 URL（POST JSON）
        SUBTITLE_STUDIO_ASR_API_KEY    Bearer Token
        SUBTITLE_STUDIO_LYRIC_ENDPOINT 自定义歌词服务
        SUBTITLE_STUDIO_PLUGINS        额外插件模块（逗号分隔的导入路径）
    """

    asr_endpoint: str = ""
    asr_api_key: str = ""
    lyric_endpoint: str = ""
    plugin_modules: tuple = ()
    timeout: float = 60.0


@dataclass
class AppConfig:
    """全局配置（pipeline / UI / CLI 共用）。"""

    work_dir: str = "outputs"
    profiles_dir: str = "profiles"
    lyrics_dir: str = "lyrics"
    models_dir: str = "models"
    # 可再生成的中间产物（抽取音频、歌曲切片、缩略图）放这里，便于整体清理
    cache_dir: str = "cache"
    device: str = "auto"
    compute_type: Optional[str] = None
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    song: SongConfig = field(default_factory=SongConfig)
    voiceprint: VoiceprintConfig = field(default_factory=VoiceprintConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)

    @classmethod
    def from_env(cls, **overrides) -> "AppConfig":
        """读取环境变量覆盖（不改变默认值，便于实验对比）。"""
        cfg = cls(
            work_dir=_env_str("SUBTITLE_STUDIO_WORK_DIR", "outputs"),
            profiles_dir=_env_str("SUBTITLE_STUDIO_PROFILES_DIR", "profiles"),
            lyrics_dir=_env_str("SUBTITLE_STUDIO_LYRICS_DIR", "lyrics"),
            models_dir=_env_str("SUBTITLE_STUDIO_MODELS_DIR", "models"),
            device=_env_str("SUBTITLE_STUDIO_DEVICE", "auto"),
            asr=AsrConfig(
                engine=_env_str("SUBTITLE_STUDIO_ASR_ENGINE", "whisper"),
                model=_env_str("SUBTITLE_STUDIO_ASR_MODEL", "large-v3"),
                vad_filter=_env_str("SUBTITLE_STUDIO_VAD", "0") == "1",
                chunk_seconds=_env_float("SUBTITLE_STUDIO_CHUNK_SECONDS", 120.0),
            ),
            timing=TimingConfig(
                refine=_env_str("SUBTITLE_STUDIO_REFINE", "1") == "1",
            ),
            provider=ProviderConfig(
                asr_endpoint=_env_str("SUBTITLE_STUDIO_ASR_ENDPOINT", ""),
                asr_api_key=_env_str("SUBTITLE_STUDIO_ASR_API_KEY", ""),
                lyric_endpoint=_env_str("SUBTITLE_STUDIO_LYRIC_ENDPOINT", ""),
                plugin_modules=tuple(
                    m.strip() for m in
                    _env_str("SUBTITLE_STUDIO_PLUGINS", "").split(",") if m.strip()),
            ),
        )
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise AttributeError(f"未知配置项：{key}")
            current = getattr(cfg, key)
            if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
                setattr(cfg, key, replace(current, **value))
            else:
                setattr(cfg, key, value)
        return cfg

    def resolve(self) -> "AppConfig":
        """把相对目录解析为绝对路径（与启动目录无关，见评审 M5）。"""
        root = Path(__file__).resolve().parent.parent.parent
        cfg = replace(self)
        for name in ("work_dir", "profiles_dir", "lyrics_dir", "models_dir",
                     "cache_dir"):
            p = Path(getattr(self, name))
            setattr(cfg, name, str(p if p.is_absolute() else root / p))
        return cfg


__all__ = [
    "AppConfig", "AsrConfig", "SegmentConfig", "SongConfig", "TimingConfig",
    "VoiceprintConfig", "ProviderConfig",
]
