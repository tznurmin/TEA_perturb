from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple


def _load_species_pairs(species_file: Path) -> set[Tuple[str, str]]:
    """Load strict binomial names from a whitelist and return (Genus, epithet) pairs.

    This helper is intentionally simple and fast
    - only accepts 2-token lines
    - keeps case-sensitive genus, case-sensitive epithet as seen in the file

    Downstream extraction lowercases epithets for matching robustness.
    """

    pairs: set[Tuple[str, str]] = set()
    for line in species_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) != 2:
            continue
        genus, epithet = parts
        if len(genus) < 2 or len(epithet) < 2:
            continue
        # Reject placeholders like "sp" or "sp." which are not real species
        ep = epithet.rstrip(".")
        if ep.lower() in {"sp", "spp"}:
            continue
        pairs.add((genus, ep))
    return pairs


def _iter_regular_word_lists(regular_path: Path) -> Iterable[List[str]]:
    obj = json.loads(regular_path.read_text(encoding="utf-8"))

    for key, rec in obj.items():
        if key == "all_labels":
            continue
        for bucket in ("relevant", "irrelevant"):
            block = rec.get(bucket) or {}
            raw = block.get("raw") or []
            for row in raw:
                words = row.get("words")
                if isinstance(words, list) and words:
                    yield [str(w) for w in words]


def _extract_binomials_from_words(
    words: List[str], allowed_pairs: set[Tuple[str, str]]
) -> Iterator[str]:
    """
    Extract binomial names from a token list using a whitelist.
     - scan adjacent bigrams: (Genus, epithet)
     - match genus case-sensitively, epithet case-insensitively (after stripping trailing punctuation)

    This is designed for TEA token streams where punctuation is often separated into its own token,
    but it also tolerates cases like 'aureus,' by stripping trailing commas/periods.
    """

    # Precompute a fast lookup from genus to allowed epithets (lowercased)
    genus_map: dict[str, set[str]] = {}
    for g, e in allowed_pairs:
        genus_map.setdefault(g, set()).add(e.lower())

    for i in range(len(words) - 1):
        g = words[i]
        if g not in genus_map:
            continue

        e_raw = words[i + 1]
        e = e_raw.strip(".,;:!?()[]{}\"'`).").rstrip(".")
        if not e:
            continue

        e_l = e.lower()
        if e_l in {"sp", "spp"}:
            continue

        if e_l in genus_map[g]:
            yield f"{g} {e}"


def extract_species_counts(
    regular_json: str,
    species_list: str,
) -> Counter[str]:
    regular_path = Path(regular_json)
    species_path = Path(species_list)

    allowed_pairs = _load_species_pairs(species_path)
    counts: Counter[str] = Counter()

    for words in _iter_regular_word_lists(regular_path):
        for name in _extract_binomials_from_words(words, allowed_pairs):
            counts[name] += 1

    return counts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract (in-corpus) binomial species names from work/regular_*.json using a whitelist."
    )
    p.add_argument(
        "--regular",
        required=True,
        help="Path to work/regular_*.json",
    )
    p.add_argument(
        "--all-species",
        required=True,
        help="Path to work/species/all_species.txt (binomial whitelist)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output list file (one binomial per line)",
    )
    p.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum occurrence count to keep (default: 1)",
    )
    p.add_argument(
        "--sort",
        choices=["count", "alpha"],
        default="count",
        help="Sort output by count (desc) or alphabetically (default: count)",
    )
    return p


def main() -> None:
    a = build_parser().parse_args()

    counts = extract_species_counts(a.regular, a.all_species)

    items = [(name, c) for name, c in counts.items() if c >= a.min_count]
    if a.sort == "count":
        items.sort(key=lambda t: (-t[1], t[0]))
    else:
        items.sort(key=lambda t: t[0])

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_lines = [name for name, _ in items]
    out_path.write_text(
        "\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8"
    )

    print(f"[INFO] regular={a.regular}")
    print(f"[INFO] whitelist={a.all_species}")
    print(
        f"[INFO] unique_species={len(out_lines)}  total_mentions={sum(c for _, c in items)}"
    )
    print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
