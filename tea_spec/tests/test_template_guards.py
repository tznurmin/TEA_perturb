from __future__ import annotations

import sys
from pathlib import Path

# ensure we can import the src modules (scripts are flat modules in tea_spec/src)
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))


def test_exactly_one_anchor_rejects_unknown_extra_species_pairs() -> None:
    # even if the whitelist misses a species, regex fallback must reject it

    from generate_templates_from_curated import exactly_one_anchor

    anchor = "Staphylococcus aureus"
    sent = "We cultured Staphylococcus aureus with Bacillus subtilis under aerobic conditions."

    # deliberately omit Bacillus subtilis from the sets
    full_set = {anchor}
    abbr_set = {"S. aureus"}

    assert not exactly_one_anchor(
        sent,
        anchor_full=anchor,
        full_set=full_set,
        abbr_set=abbr_set,
        min_tokens=3,
        max_tokens=200,
    )
