from summarizer.token_budget import (
    DEFAULT_CHARS_PER_TOKEN,
    RESERVED_TOKENS,
    calculate_char_budget,
    get_chars_per_token,
    supported_languages,
)


class TestGetCharsPerToken:
    def test_english(self):
        assert get_chars_per_token("en") == 4.0

    def test_german(self):
        assert get_chars_per_token("de") == 3.0

    def test_chinese(self):
        assert get_chars_per_token("zh") == 1.5

    def test_japanese(self):
        assert get_chars_per_token("ja") == 1.5

    def test_russian(self):
        assert get_chars_per_token("ru") == 2.0

    def test_unknown_language_returns_default(self):
        assert get_chars_per_token("xx") == DEFAULT_CHARS_PER_TOKEN

    def test_case_insensitive(self):
        assert get_chars_per_token("EN") == 4.0
        assert get_chars_per_token("De") == 3.0

    def test_strips_whitespace(self):
        assert get_chars_per_token(" en ") == 4.0


class TestCalculateCharBudget:
    def test_english_4k_model(self):
        budget = calculate_char_budget(max_tokens=4096, language="en")
        # (4096 - 1500) * 4.0 = 10384
        assert budget == 10384

    def test_german_4k_model(self):
        budget = calculate_char_budget(max_tokens=4096, language="de")
        # (4096 - 1500) * 3.0 = 7788
        assert budget == 7788

    def test_chinese_4k_model(self):
        budget = calculate_char_budget(max_tokens=4096, language="zh")
        # (4096 - 1500) * 1.5 = 3894
        assert budget == 3894

    def test_large_context_model(self):
        budget = calculate_char_budget(max_tokens=32768, language="en")
        # (32768 - 1500) * 4.0 = 125072
        assert budget == 125072

    def test_german_large_model(self):
        budget = calculate_char_budget(max_tokens=32768, language="de")
        # (32768 - 1500) * 3.0 = 93804
        assert budget == 93804

    def test_minimum_budget_floor(self):
        # Even with tiny token limit, should not go below 500 * ratio
        budget = calculate_char_budget(max_tokens=100, language="en")
        assert budget == 500 * 4.0  # min(100 - 1500, 500) * 4.0

    def test_custom_reserved_tokens(self):
        budget = calculate_char_budget(max_tokens=4096, language="en", reserved_tokens=1000)
        # (4096 - 1000) * 4.0 = 12384
        assert budget == 12384

    def test_unknown_language_uses_default(self):
        budget = calculate_char_budget(max_tokens=4096, language="xx")
        # (4096 - 1500) * 3.0 = 7788
        assert budget == int((4096 - RESERVED_TOKENS) * DEFAULT_CHARS_PER_TOKEN)


class TestSupportedLanguages:
    def test_returns_sorted_list(self):
        langs = supported_languages()
        assert langs == sorted(langs)

    def test_includes_common_languages(self):
        langs = supported_languages()
        for code in ["en", "de", "fr", "es", "zh", "ja", "ko", "ru"]:
            assert code in langs

    def test_all_entries_are_strings(self):
        for lang in supported_languages():
            assert isinstance(lang, str)
            assert len(lang) == 2
