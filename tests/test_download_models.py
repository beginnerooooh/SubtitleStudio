"""models/download_models.py 单测：预设与显式参数合并（不触发真实下载）。"""
import pytest

import download_models as dm


class TestResolvePlan:
    def test_default_preset_is_basic(self):
        plan = dm.resolve_plan()
        assert plan == {
            "hf": [
                "Systran/faster-whisper-small",
                "speechbrain/spkrec-ecapa-voxceleb",
                "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
            ],
            "demucs": False,
        }

    def test_full_preset(self):
        plan = dm.resolve_plan(preset="full")
        assert plan["hf"][0] == "Systran/faster-whisper-base"
        assert "Systran/faster-whisper-small" in plan["hf"]
        assert "jonatasgrosman/wav2vec2-large-xlsr-53-english" in plan["hf"]
        assert plan["demucs"] is True

    def test_explicit_whisper_overrides_preset(self):
        plan = dm.resolve_plan(preset="full", whisper=["tiny"])
        assert plan["hf"] == [
            "Systran/faster-whisper-tiny",
            "speechbrain/spkrec-ecapa-voxceleb",
            "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
            "jonatasgrosman/wav2vec2-large-xlsr-53-english",
        ]

    def test_voiceprint_off(self):
        plan = dm.resolve_plan(voiceprint=False)
        assert not any("spkrec" in r for r in plan["hf"])

    def test_demucs_flag_overrides(self):
        assert dm.resolve_plan(demucs=True)["demucs"] is True
        assert dm.resolve_plan(preset="full", demucs=False)["demucs"] is False

    def test_deduplicates_repos(self):
        plan = dm.resolve_plan(whisper=["small"], aligner=["zh"])
        # small + zh + 声纹，无重复条目
        assert len(plan["hf"]) == len(set(plan["hf"])) == 3

    def test_unknown_whisper_size_exits(self):
        with pytest.raises(SystemExit, match="未知的 whisper 规格"):
            dm.resolve_plan(whisper=["giant"])

    def test_unknown_aligner_lang_exits(self):
        with pytest.raises(SystemExit, match="未知的对齐语言"):
            dm.resolve_plan(aligner=["jp"])


class TestCli:
    def test_list_mode_exits_zero(self, capsys):
        assert dm.main(["--preset", "basic", "--list"]) == 0
        out = capsys.readouterr().out
        assert "Systran/faster-whisper-small" in out
        assert "demucs htdemucs" in out

    def test_flags_map_to_plan(self, capsys):
        assert dm.main(["--demucs", "0", "--list"]) == 0
        assert "否" in capsys.readouterr().out
