from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import build_regular_dataset  # noqa: E402


def test_tokenize_words_strips_tea_italics_markers() -> None:
    text = "Similarly, unique profiles of $i$Arabidopsis lyrata$/i$ ssp."
    words = build_regular_dataset._tokenize_words(text)

    assert "Arabidopsis" in words
    assert "lyrata" in words

    # no token should include the markup markers
    assert all("$i$" not in w for w in words)
    assert all("$/i$" not in w for w in words)


def test_tokenize_words_drops_standalone_markup_tokens() -> None:
    text = "a $i$ Arabidopsis $/i$ b"
    words = build_regular_dataset._tokenize_words(text)

    assert words == ["a", "Arabidopsis", "b"]
