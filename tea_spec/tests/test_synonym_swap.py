# Copyright 2026 tznurmin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import pytest  # noqa: E402
from utils import make_synonym_swapper, wordnet_is_ready  # noqa: E402


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


def test_synonym_swap_ok_preserves_protected_phrase_and_changes_one_token():
    if not wordnet_is_ready():
        pytest.skip("WordNet corpus not available")
    protected = "Staphylococcus aureus"
    text = "the big colony of Staphylococcus aureus was observed in culture."

    swap = make_synonym_swapper([protected], seed=0, max_attempts=100)
    res = swap(text)

    assert res.ok
    assert res.text_out is not None
    assert res.text_out != text
    assert protected in res.text_out
    assert _diff_count(text, res.text_out) == 1


def test_synonym_swap_is_deterministic_per_text():
    if not wordnet_is_ready():
        pytest.skip("WordNet corpus not available")
    protected = "Staphylococcus aureus"
    text = "the big colony of Staphylococcus aureus was observed in culture."

    swap1 = make_synonym_swapper([protected], seed=123, max_attempts=100)
    swap2 = make_synonym_swapper([protected], seed=123, max_attempts=100)

    r1 = swap1(text)
    r2 = swap2(text)

    assert r1.ok and r2.ok
    assert r1.text_out == r2.text_out
    assert r1.changed_token == r2.changed_token
    assert r1.replacement == r2.replacement


def test_synonym_swap_fail_closed_when_no_candidate_tokens_exist():
    protected = "Staphylococcus aureus"
    text = "Staphylococcus aureus"

    swap = make_synonym_swapper([protected], seed=0, max_attempts=10)
    res = swap(text)

    assert not res.ok
    assert res.text_out is None
    assert res.reason in {
        "wordnet_unavailable",
        "no_candidate",
        "max_attempts_exhausted",
    }
