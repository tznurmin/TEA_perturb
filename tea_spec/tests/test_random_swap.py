from __future__ import annotations

import sys
from pathlib import Path

# ensure we can import the src modules (scripts are flat modules in tea_spec/src)
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import pytest  # noqa: E402
from utils import make_random_word_swapper, wordfreq_is_ready  # noqa: E402


def _tokenize(s: str):
    toks = []
    for t in s.strip().split():
        toks.append(t.strip(".,;:!?()[]{}\"'"))
    return toks


def _diff_count(a: str, b: str) -> int:
    ta, tb = _tokenize(a), _tokenize(b)
    if len(ta) != len(tb):
        return 999
    return sum(x != y for x, y in zip(ta, tb))


def test_random_swap_ok_preserves_protected_phrase_and_changes_one_token():
    if not wordfreq_is_ready():
        pytest.skip("wordfreq not available")
    protected = "Staphylococcus aureus"
    text = "the big colony of Staphylococcus aureus was observed in culture."

    swap = make_random_word_swapper(
        [protected], seed=0, max_attempts=100, wordlist_size=50_000
    )
    res = swap(text)

    assert res.ok
    assert res.text_out is not None
    assert res.text_out != text
    assert protected in res.text_out
    assert _diff_count(text, res.text_out) == 1


def test_random_swap_is_deterministic_per_text():
    if not wordfreq_is_ready():
        pytest.skip("wordfreq not available")
    protected = "Staphylococcus aureus"
    text = "the big colony of Staphylococcus aureus was observed in culture."

    swap1 = make_random_word_swapper([protected], seed=123, max_attempts=100)
    swap2 = make_random_word_swapper([protected], seed=123, max_attempts=100)

    r1 = swap1(text)
    r2 = swap2(text)

    assert r1.ok and r2.ok
    assert r1.text_out == r2.text_out
    assert r1.changed_token == r2.changed_token
    assert r1.replacement == r2.replacement


def test_random_swap_fail_closed_when_no_candidate_tokens_exist():
    protected = "Staphylococcus aureus"
    text = "Staphylococcus aureus"

    swap = make_random_word_swapper([protected], seed=0, max_attempts=10)
    res = swap(text)

    assert not res.ok
    assert res.text_out is None
    assert res.reason in {
        "wordfreq_unavailable",
        "no_candidate",
        "max_attempts_exhausted",
    }
