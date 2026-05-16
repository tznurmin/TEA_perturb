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

import pytest

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import build_species_list  # noqa: E402


def _write_mini_speclist(path: Path) -> None:
    # Minimal fixture that preserves the structural markers used by the parser:
    # - table header line that matches TABLE_START_RE
    # - entry header lines that match ENTRY_RE
    # - the virtual section marker that matches VIRTUAL_SECTION_RE
    content = """\
Some header text that should be ignored.

(1) Real organism codes

Code    Taxon    N=Official (scientific) name
__________________________________________
ABABO E 3053407: N=Abaeis boisduvaliana
                 C=Boisduval's yellow butterfly
                 S=Eurema boisduvaliana
EUREM E 999111: N=Eurema boisduvaliana
AAV2S V  648242: N=Adeno-associated virus 2 (isolate Srivastava/1982)
                 C=AAV-2
BRACI E   52824: N=Brassica carinata
                 C=Ethiopian mustard
                 S=Abyssinian cabbage
ARTIF O  999999: N=Artificial construct (some synthetic thing)

(2) "Virtual" codes that regroup organisms at a certain taxonomic level
VIRT1 E  123456: N=This should not be parsed
"""
    path.write_text(content, encoding="utf-8", newline="\n")


def test_extract_abe_includes_scientific_names_and_synonyms(tmp_path: Path) -> None:
    speclist = tmp_path / "speclist.txt"
    _write_mini_speclist(speclist)

    selected, stats = build_species_list.extract_names_from_speclist(
        str(speclist),
        kingdoms="ABE",
        include_synonyms=True,
        synonyms_official_only=True,
        strip_parentheses=True,
        binomial_only=True,
    )

    # the result includes sets for all requested kingdoms
    assert set(selected.keys()) == {"A", "B", "E"}

    # official scientific name is extracted
    assert "Abaeis boisduvaliana" in selected.get("E", set())

    # synonym is extracted because it is also an official N= name elsewhere
    assert "Eurema boisduvaliana" in selected.get("E", set())

    # common-name synonyms must be excluded even though they look like binomials
    assert "Abyssinian cabbage" not in selected.get("E", set())

    # common name is not extracted
    assert all("Boisduval" not in s for s in selected.get("E", set()))

    # virus entry is excluded by kingdom filtering
    for k, names in selected.items():
        assert all("virus" not in n.lower() for n in names)

    # virtual section must stop parsing
    assert "This should not be parsed" not in selected.get("E", set())

    # stats must be populated
    assert stats.total_entries >= 1


def test_extract_viruses_only_requires_non_binomial_mode(tmp_path: Path) -> None:
    speclist = tmp_path / "speclist.txt"
    _write_mini_speclist(speclist)

    selected, _stats = build_species_list.extract_names_from_speclist(
        str(speclist),
        kingdoms="V",
        include_synonyms=False,
        strip_parentheses=True,
        binomial_only=False,
    )

    assert "Adeno-associated virus 2" in selected.get("V", set())

    assert "Abaeis boisduvaliana" not in selected.get("V", set())


def test_extract_others_only(tmp_path: Path) -> None:
    speclist = tmp_path / "speclist.txt"
    _write_mini_speclist(speclist)

    selected, _stats = build_species_list.extract_names_from_speclist(
        str(speclist),
        kingdoms="O",
        include_synonyms=False,
        strip_parentheses=True,
        binomial_only=False,
    )

    assert any("Artificial construct" in s for s in selected.get("O", set()))
