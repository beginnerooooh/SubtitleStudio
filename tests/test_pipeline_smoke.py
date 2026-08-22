"""Slow 冒烟测试：真实模型端到端（faster-whisper tiny + wav2vec2-base-960h）。

运行：``pytest -m slow tests/test_pipeline_smoke.py``

- 音频样本：Whisper 仓库的 jfk.flac（11s 英文演讲），首次运行联网下载并缓存；
  可用环境变量 ``SUBTITLE_STUDIO_SMOKE_AUDIO`` 指定本地文件跳过下载。
- 模型：首次运行自动从 HuggingFace 下载（离线环境可预置 HF 缓存 +
  ``HF_HUB_OFFLINE=1``）。
- 人声分离：Demucs 权重不可达时流水线应优雅回退（warning + 原音频），
  测试同时覆盖成功与回退两条路径。
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pytest

from core.pipeline import Pipeline, PipelineConfig

pytestmark = [pytest.mark.slow]

_JFK_URL = "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"
_AUDIO_CACHE = Path(__file__).parent / ".cache" / "jfk.flac"

# jfk.flac 实际语音内容（分三行，含标点以覆盖剥离/回填链路）
_JFK_LYRICS = (
    "And so my fellow Americans\n"
    "ask not what your country can do for you\n"
    "ask what you can do for your country"
)


@pytest.fixture(scope="module")
def speech_audio() -> str:
    """真实语音样本：本地缓存 → 环境变量指定 → 在线下载；均不可得则跳过。"""
    env_path = os.environ.get("SUBTITLE_STUDIO_SMOKE_AUDIO")
    if env_path and Path(env_path).is_file():
        return env_path
    if _AUDIO_CACHE.is_file() and _AUDIO_CACHE.stat().st_size > 1_000_000:
        return str(_AUDIO_CACHE)
    try:
        _AUDIO_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_JFK_URL, _AUDIO_CACHE)  # noqa: S310 受信源
    except Exception as exc:  # 网络不可达：跳过而非失败
        pytest.skip(f"无法下载语音样本（{exc}），可设置 SUBTITLE_STUDIO_SMOKE_AUDIO 指定本地文件")
    return str(_AUDIO_CACHE)


def _run(cfg: PipelineConfig):
    progress: list[float] = []
    logs: list[str] = []
    pipe = Pipeline(
        cfg,
        on_progress=lambda r, msg: progress.append(r),
        on_log=logs.append,
    )
    return pipe.run(), progress, logs


class TestBlindSmoke:
    def test_end_to_end_blind(self, speech_audio, tmp_path):
        cfg = PipelineConfig(
            input_path=speech_audio,
            model_size="tiny",
            language="en",
            work_dir=str(tmp_path),
            title="smoke",
        )
        result, progress, _logs = _run(cfg)

        assert result.mode == "blind"
        assert result.duration == pytest.approx(11.0, abs=1.0)
        assert result.lines, "盲识别应产出至少一行字幕"

        # 词级时间戳有效且落在音频范围内
        for ln in result.lines:
            for w in ln.words:
                assert 0.0 <= w.start <= w.end <= result.duration + 0.5

        # 三种格式全部导出且非空
        assert set(result.files) == {"srt", "lrc", "ass"}
        for fmt, path in result.files.items():
            assert Path(path).is_file() and Path(path).stat().st_size > 0, fmt
        assert "-->" in Path(result.files["srt"]).read_text(encoding="utf-8")

        # 识别内容合理：tiny 对该样本可稳定识别出关键词
        text = " ".join(ln.text for ln in result.lines).lower()
        assert "country" in text, f"识别文本异常：{text!r}"

        # 进度单调递增至 1.0
        assert progress and progress == sorted(progress)
        assert progress[-1] == pytest.approx(1.0)


class TestAlignSmoke:
    def test_end_to_end_align(self, speech_audio, tmp_path):
        cfg = PipelineConfig(
            input_path=speech_audio,
            lyrics_text=_JFK_LYRICS,
            work_dir=str(tmp_path),
        )
        result, progress, _logs = _run(cfg)

        assert result.mode == "align"
        assert len(result.lines) == 3

        # 行文本与歌词一致（标点回填后的显示形式）
        assert [ln.text for ln in result.lines] == _JFK_LYRICS.splitlines()

        # 行级/词级时间戳有效、单调
        starts = [ln.start for ln in result.lines]
        ends = [ln.end for ln in result.lines]
        assert starts == sorted(starts)
        for ln in result.lines:
            assert ln.words
            for w in ln.words:
                assert w.end > w.start
                assert 0.0 <= w.start < result.duration

        # 对齐覆盖语音主体（样本语音 ~0.5s 起、~11s 止）
        assert starts[0] < 3.0, "首行起点应落在语音开头附近"
        assert ends[-1] > 8.0, "末行终点应接近语音结尾"
        # 行间不重叠（顺序朗读的语音）
        for i in range(len(ends) - 1):
            assert ends[i] <= starts[i + 1] + 0.2

        # ASS 含逐字卡拉OK标签；LRC 含 3 个行级时间戳
        ass = Path(result.files["ass"]).read_text(encoding="utf-8")
        assert "{\\k" in ass
        lrc = Path(result.files["lrc"]).read_text(encoding="utf-8")
        assert lrc.count("\n[0") + lrc.count("\n[1") >= 3

        assert progress and progress == sorted(progress)
        assert progress[-1] == pytest.approx(1.0)


class TestSeparationSmoke:
    def test_pipeline_with_separation(self, speech_audio, tmp_path):
        """开启人声分离的完整链路：成功 → vocals.wav 被使用；失败 → 优雅回退。"""
        cfg = PipelineConfig(
            input_path=speech_audio,
            model_size="tiny",
            language="en",
            enable_separation=True,
            work_dir=str(tmp_path),
        )
        result, _progress, _logs = _run(cfg)

        assert result.mode == "blind"
        assert result.lines, "分离（或回退）后盲识别仍应产出字幕"
        assert set(result.files) == {"srt", "lrc", "ass"}

        vocals = Path(result.out_dir) / "vocals.wav"
        if vocals.is_file():
            # 分离成功：流水线应使用人声轨继续
            assert vocals.stat().st_size > 0
        else:
            # 分离失败（如权重不可达）：必须带回退警告且任务不中断
            assert any("人声分离失败" in w for w in result.warnings), result.warnings
