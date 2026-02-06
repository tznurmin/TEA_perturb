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
