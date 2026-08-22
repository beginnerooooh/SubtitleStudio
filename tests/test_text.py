"""core/text.py 纯逻辑单测：标准化、分字分词、标点剥离与回填。"""
import pytest

from core.text import cjk_ratio, prepare_lyrics, tokenize


class TestTokenize:
    def test_chinese_split_per_char(self):
        tokens = tokenize("你好世界")
        assert [t.align for t in tokens] == ["你", "好", "世", "界"]

    def test_english_split_per_word(self):
        tokens = tokenize("hello world")
        assert [t.align for t in tokens] == ["hello", "world"]

    def test_mixed_chinese_english(self):
        tokens = tokenize("这是beautiful的歌")
        assert [t.align for t in tokens] == ["这", "是", "beautiful", "的", "歌"]

    def test_digit_run_is_one_token(self):
        tokens = tokenize("2024年")
        assert [t.align for t in tokens] == ["2024", "年"]

    def test_apostrophe_kept_inside_word(self):
        tokens = tokenize("don't stop")
        assert [t.align for t in tokens] == ["don't", "stop"]

    def test_fullwidth_latin_normalized_in_align_form(self):
        tokens = tokenize("ｈｅｌｌｏ　ｗｏｒｌｄ")
        assert [t.align for t in tokens] == ["hello", "world"]

    def test_fullwidth_digits_normalized_in_align_form(self):
        tokens = tokenize("２０２４年")
        assert [t.align for t in tokens] == ["2024", "年"]

    def test_punct_only_line_yields_no_tokens(self):
        assert tokenize("……——！！！") == []

    def test_empty_string_yields_no_tokens(self):
        assert tokenize("") == []

    def test_align_form_strips_trailing_punct(self):
        tokens = tokenize("好，")
        assert tokens[0].align == "好"
        assert tokens[0].display == "好，"

    def test_display_roundtrip_chinese_punct(self):
        line = "你好，世界！"
        tokens = tokenize(line)
        assert "".join(t.display for t in tokens) == line

    def test_display_roundtrip_mixed_with_spaces(self):
        line = "这是 beautiful 的歌，OK？"
        tokens = tokenize(line)
        assert "".join(t.display for t in tokens) == line

    def test_leading_punct_attaches_to_first_token(self):
        tokens = tokenize("「你好」")
        assert tokens[0].display == "「你"
        assert tokens[1].display == "好」"


class TestPrepareLyrics:
    def test_strips_lrc_timestamps(self):
        raw = "[00:12.34]你好\n[01:00]世界\n[02:03.456]再见"
        lines = prepare_lyrics(raw)
        assert [[t.align for t in ln] for ln in lines] == [
            ["你", "好"],
            ["世", "界"],
            ["再", "见"],
        ]

    def test_strips_metadata_tag_lines(self):
        raw = "[ti:测试歌曲]\n[ar:歌手]\n[al:专辑]\n[by:某人]\n[offset:+500]\n\n第一句"
        lines = prepare_lyrics(raw)
        assert [[t.align for t in ln] for ln in lines] == [["第", "一", "句"]]

    def test_drops_blank_and_whitespace_lines(self):
        raw = "第一行\n\n\n第二行\n  \n"
        lines = prepare_lyrics(raw)
        assert [[t.align for t in ln] for ln in lines] == [["第", "一", "行"], ["第", "二", "行"]]

    def test_inline_timestamp_also_stripped(self):
        raw = "你好[00:30.00]世界"
        lines = prepare_lyrics(raw)
        assert [[t.align for t in ln] for ln in lines] == [["你", "好", "世", "界"]]

    def test_display_roundtrip_after_timestamp_strip(self):
        raw = "[00:01.00]你好，世界！"
        lines = prepare_lyrics(raw)
        assert ["".join(t.display for t in ln) for ln in lines] == ["你好，世界！"]

    def test_empty_input(self):
        assert prepare_lyrics("") == []


class TestCjkRatio:
    def test_pure_chinese_is_one(self):
        assert cjk_ratio("你好世界") == 1.0

    def test_pure_english_is_zero(self):
        assert cjk_ratio("hello world") == 0.0

    def test_mixed_ratio_ignores_punctuation(self):
        # 词字符：这、是、beautiful(9字母) → 2/11
        assert cjk_ratio("这是beautiful，") == pytest.approx(2 / 11)

    def test_empty_text_is_zero(self):
        assert cjk_ratio("") == 0.0
