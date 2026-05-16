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

import extract_species_in_corpus  # noqa: E402


def test_extract_species_from_words_bigram_whitelist(tmp_path: Path):
    species_txt = tmp_path / "all_species.txt"
    species_txt.write_text(
        "Arabidopsis thaliana\nStaphylococcus aureus\n",
        encoding="utf-8",
    )

    allowed = extract_species_in_corpus._load_species_pairs(species_txt)
    words = [
        "A",
        "study",
        "of",
        "Arabidopsis",
        "thaliana",
        "and",
        "Staphylococcus",
        "aureus",
        ".",
    ]
    found = list(extract_species_in_corpus._extract_binomials_from_words(words, allowed))
    assert "Arabidopsis thaliana" in found
    assert "Staphylococcus aureus" in found


def test_extract_species_ignores_non_binomials(tmp_path: Path):
    species_txt = tmp_path / "all_species.txt"
    species_txt.write_text("Arabidopsis thaliana\n", encoding="utf-8")

    allowed = extract_species_in_corpus._load_species_pairs(species_txt)
    words = ["Arabidopsis", "sp", "was", "measured"]
    found = list(extract_species_in_corpus._extract_binomials_from_words(words, allowed))
    assert found == []
