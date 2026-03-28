from __future__ import annotations

# Approximate characters per token by language.
# Based on BPE tokenizer behavior (GPT/Llama family):
# - Latin-script languages: UTF-8 is compact, common words are single tokens
# - CJK: most characters are 3 bytes in UTF-8, often 1 token per character
# - Cyrillic/Arabic/Hebrew: 2 bytes per char, fewer merged tokens
#
# These are conservative estimates — actual ratios vary by model tokenizer,
# vocabulary size, and text content. Erring low is safer (produces smaller
# chunks that always fit).
CHARS_PER_TOKEN: dict[str, float] = {
    # Latin-script languages
    "en": 4.0,  # English
    "de": 3.0,  # German — compound words split into more tokens
    "fr": 3.5,  # French
    "es": 3.5,  # Spanish
    "pt": 3.5,  # Portuguese
    "it": 3.5,  # Italian
    "nl": 3.2,  # Dutch — compound words similar to German
    "pl": 2.5,  # Polish — diacritics, longer words
    "sv": 3.2,  # Swedish
    "da": 3.2,  # Danish
    "no": 3.2,  # Norwegian
    "fi": 2.5,  # Finnish — very long compound words
    "cs": 2.5,  # Czech
    "ro": 3.2,  # Romanian
    "hu": 2.5,  # Hungarian — agglutinative
    "tr": 2.5,  # Turkish — agglutinative
    # Cyrillic
    "ru": 2.0,  # Russian
    "uk": 2.0,  # Ukrainian
    "bg": 2.0,  # Bulgarian
    # CJK
    "zh": 1.5,  # Chinese
    "ja": 1.5,  # Japanese
    "ko": 1.5,  # Korean
    # Other scripts
    "ar": 2.0,  # Arabic
    "he": 2.0,  # Hebrew
    "hi": 1.8,  # Hindi (Devanagari)
    "th": 1.5,  # Thai
    "vi": 3.0,  # Vietnamese (Latin script + diacritics)
}

# Fallback for unknown languages — conservative estimate
DEFAULT_CHARS_PER_TOKEN = 3.0

# Tokens reserved for system prompt + LLM response output.
# System prompt: ~300 tokens (extraction prompt is the longest)
# LLM response: ~1024 tokens (num_predict setting)
# Safety margin: ~200 tokens
RESERVED_TOKENS = 1500


def get_chars_per_token(language: str) -> float:
    """Get the chars-per-token ratio for a language code."""
    return CHARS_PER_TOKEN.get(language.lower().strip(), DEFAULT_CHARS_PER_TOKEN)


def calculate_char_budget(
    max_tokens: int,
    language: str = "en",
    reserved_tokens: int = RESERVED_TOKENS,
) -> int:
    """Calculate the character budget for LLM prompt context.

    Args:
        max_tokens: The model's context window size in tokens.
        language: ISO 639-1 language code of the session content.
        reserved_tokens: Tokens reserved for system prompt + response.

    Returns:
        Maximum characters to include in the prompt context.
    """
    available_tokens = max(max_tokens - reserved_tokens, 500)
    ratio = get_chars_per_token(language)
    return int(available_tokens * ratio)


def supported_languages() -> list[str]:
    """Return list of language codes with known chars-per-token ratios."""
    return sorted(CHARS_PER_TOKEN.keys())
