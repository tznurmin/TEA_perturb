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

# ensure we can import the src modules (scripts are flat modules in tea_spec/src)
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))


def test_strict_guard_rejects_species_like_sentence_initial_phrases() -> None:
    from generate_templates_from_curated import exactly_one_anchor

    anchor = "Staphylococcus aureus"
    full_set = {anchor}
    abbr_set = {"S. aureus"}

    for sent in (
        "The growth of Staphylococcus aureus was measured in culture.",
        "Most isolates of Staphylococcus aureus were susceptible.",
    ):
        assert not exactly_one_anchor(
            sent,
            anchor_full=anchor,
            full_set=full_set,
            abbr_set=abbr_set,
            min_tokens=3,
            max_tokens=200,
        )


def test_known_genus_guard_accepts_sentence_initial_common_phrases() -> None:
    from generate_templates_from_curated import exactly_one_anchor

    anchor = "Staphylococcus aureus"
    full_set = {anchor}
    abbr_set = {"S. aureus"}

    for sent in (
        "The growth of Staphylococcus aureus was measured in culture.",
        "Most isolates of Staphylococcus aureus were susceptible.",
    ):
        assert exactly_one_anchor(
            sent,
            anchor_full=anchor,
            full_set=full_set,
            abbr_set=abbr_set,
            min_tokens=3,
            max_tokens=200,
            guard_mode="known-genus",
        )


def test_exactly_one_anchor_rejects_known_extra_species_pairs() -> None:
    from generate_templates_from_curated import exactly_one_anchor

    anchor = "Staphylococcus aureus"
    sent = "Bacillus subtilis and Staphylococcus aureus were cultured together."

    full_set = {anchor, "Bacillus subtilis"}
    abbr_set = {"S. aureus", "B. subtilis"}

    assert not exactly_one_anchor(
        sent,
        anchor_full=anchor,
        full_set=full_set,
        abbr_set=abbr_set,
        min_tokens=3,
        max_tokens=200,
    )


def test_exactly_one_anchor_rejects_unknown_species_with_known_genus() -> None:
    from generate_templates_from_curated import exactly_one_anchor

    anchor = "Staphylococcus aureus"
    sent = "We cultured Staphylococcus aureus with Bacillus subtilis."

    full_set = {anchor, "Bacillus cereus"}
    abbr_set = {"S. aureus", "B. cereus"}

    assert not exactly_one_anchor(
        sent,
        anchor_full=anchor,
        full_set=full_set,
        abbr_set=abbr_set,
        min_tokens=3,
        max_tokens=200,
        guard_mode="known-genus",
    )
