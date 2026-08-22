"""core/models.py + core/subtitle.py 单测：数据模型、聚合规则与三格式导出。"""
import pytest

from core.models import SubtitleLine, SubtitleWord
from core.subtitle import (
    aggregate_words,
    format_ass_time,
    format_lrc_time,
    format_srt_time,
    to_ass,
    to_lrc,
    to_srt,
)


def w(text: str, start: float, end: float) -> SubtitleWord:
    return SubtitleWord(text=text, start=start, end=end)


def line(*words: SubtitleWord) -> SubtitleLine:
    return SubtitleLine(words=list(words))


class TestModels:
    def test_line_start_end_from_words(self):
        ln = line(w("你", 1.0, 1.2), w("好", 1.5, 2.0))
        assert ln.start == 1.0
        assert ln.end == 2.0

    def test_line_text_is_concat(self):
        ln = line(w("你", 0.0, 0.5), w("好，", 0.5, 1.0))
        assert ln.text == "你好，"

    def test_empty_line_safe_defaults(self):
        ln = SubtitleLine(words=[])
        assert ln.start == 0.0
        assert ln.end == 0.0
        assert ln.text == ""


class TestTimeFormats:
    def test_srt_format(self):
        assert format_srt_time(3661.234) == "01:01:01,234"

    def test_srt_rounds_half_up(self):
        assert format_srt_time(2.56789) == "00:00:02,568"

    def test_srt_zero(self):
        assert format_srt_time(0.0) == "00:00:00,000"

    def test_lrc_format(self):
        assert format_lrc_time(61.5) == "01:01.50"

    def test_lrc_centisecond_carry(self):
        assert format_lrc_time(10.999) == "00:11.00"

    def test_ass_format(self):
        assert format_ass_time(3661.234) == "1:01:01.23"

    def test_ass_under_one_hour(self):
        assert format_ass_time(1.95) == "0:00:01.95"


class TestAggregateWords:
    def test_empty_input(self):
        assert aggregate_words([]) == []

    def test_split_after_sentence_punct(self):
        words = [w("你好。", 0.0, 0.6), w("世界。", 0.8, 1.4)]
        lines = aggregate_words(words)
        assert len(lines) == 2
        assert lines[0].text == "你好。"
        assert lines[1].text == "世界。"

    def test_split_when_duration_exceeds_max(self):
        words = [w("一", 0.0, 0.5), w("二", 3.0, 3.5), w("三", 6.0, 6.5)]
        # 追加「三」后行跨度 0→6.5s 超过 5s → 在「三」前切分
        lines = aggregate_words(words, max_duration=5.0)
        assert [ln.text for ln in lines] == ["一二", "三"]

    def test_split_when_chars_exceed_max(self):
        words = [w("字", i * 0.1, i * 0.1 + 0.05) for i in range(26)]
        lines = aggregate_words(words, max_chars=10)
        assert [len(ln.text) for ln in lines] == [10, 10, 6]

    def test_no_split_within_limits(self):
        words = [w("你", 0.0, 0.2), w("好", 0.2, 0.4)]
        lines = aggregate_words(words)
        assert len(lines) == 1
        assert lines[0].text == "你好"

    def test_oversized_single_word_stays_alone(self):
        lines = aggregate_words([w("一个特别特别长的句子超过上限", 0.0, 1.0)], max_chars=5)
        assert len(lines) == 1


class TestToSrt:
    def test_basic_block(self):
        srt = to_srt([line(w("你好", 1.0, 2.5))])
        assert srt == "1\n00:00:01,000 --> 00:00:02,500\n你好\n"

    def test_multiple_blocks_numbered(self):
        srt = to_srt([line(w("一", 0.0, 1.0)), line(w("二", 2.0, 3.0))])
        assert "1\n00:00:00,000 --> 00:00:01,000\n一\n\n" in srt
        assert "2\n00:00:02,000 --> 00:00:03,000\n二\n" in srt

    def test_time_components(self):
        srt = to_srt([line(w("好", 3661.234, 7322.05))])
        assert "01:01:01,234 --> 02:02:02,050" in srt

    def test_empty_lines_input(self):
        assert to_srt([]) == ""

    def test_skips_wordless_lines_without_number_gap(self):
        srt = to_srt([SubtitleLine(words=[]), line(w("好", 0.0, 1.0))])
        assert srt.startswith("1\n")


class TestToLrc:
    def test_basic_line(self):
        lrc = to_lrc([line(w("你好", 61.5, 63.0))])
        assert "[01:01.50]你好\n" in lrc

    def test_metadata_tags(self):
        lrc = to_lrc([line(w("好", 0.0, 1.0))], title="歌曲名")
        assert "[ti:歌曲名]" in lrc

    def test_centisecond_rounding(self):
        lrc = to_lrc([line(w("好", 10.994, 11.5))])
        assert "[00:10.99]" in lrc

    def test_empty_lines_input(self):
        assert to_lrc([]) == ""


class TestToAss:
    def test_headers_present(self):
        ass = to_ass([line(w("好", 0.0, 1.0))])
        assert "[Script Info]" in ass
        assert "PlayResX: 1920" in ass
        assert "[V4+ Styles]" in ass
        assert "[Events]" in ass

    def test_dialogue_time_format(self):
        ass = to_ass([line(w("好", 1.0, 2.0))])
        assert "Dialogue: 0,0:00:01.00,0:00:02.00,Default," in ass

    def test_karaoke_k_tags_in_centiseconds(self):
        ln = line(w("我", 1.0, 1.2), w("爱", 1.5, 1.8), w("你", 1.8, 2.0))
        ass = to_ass([ln])
        assert "{\\k20}我{\\k30}爱{\\k20}你" in ass

    def test_display_text_with_punct(self):
        ass = to_ass([line(w("好，", 0.0, 0.5))])
        assert "{\\k50}好，" in ass

    def test_last_word_extends_to_near_next_line(self):
        l1 = line(w("我", 1.0, 1.5))
        l2 = line(w("好", 2.0, 2.5))
        ass = to_ass([l1, l2])
        # 行间隙 0.5s ≤ 1s → 末字延音至 2.0-0.05=1.95 → \k=95
        assert "{\\k95}我" in ass
        assert "0:00:01.95" in ass

    def test_no_extension_across_long_gap(self):
        l1 = line(w("我", 1.0, 1.5))
        l2 = line(w("好", 10.0, 10.5))
        ass = to_ass([l1, l2])
        assert "{\\k50}我" in ass

    def test_empty_lines_input(self):
        assert to_ass([]) == ""
