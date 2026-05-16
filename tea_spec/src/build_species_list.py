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

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# UniProt speclist kingdom codes
#   A = archaea
#   B = bacteria
#   E = eukaryota
#   V = viruses and phages
#   O = others (artificial sequences, etc.)
KNOWN_KINGDOMS = "ABEVO"

# Entry header example:
# ABABO E 3053407: N=Abaeis boisduvaliana
ENTRY_RE = re.compile(r"^(\S+)\s+([A-Z])\s+(\d+):\s*(.*)$")

# Field lines (the value is everything after N= or S=)
FIELD_N_RE = re.compile(r"\bN=(.+)$")
FIELD_S_RE = re.compile(r"\bS=(.+)$")

TABLE_START_RE = re.compile(r"^Code\s+Taxon\s+N=")
VIRTUAL_SECTION_RE = re.compile(r"^\(2\)\s+\"Virtual\"\s+codes", re.IGNORECASE)

# A strict binomial: Genus species
#
# IMPORTANT:
# A purely syntactic "Genus species" regex is not enough to ensure scientific
# names. UniProt `S=` synonym lines can include English common-name synonyms
# (e.g. "Abyssinian cabbage"), which match the binomial shape
#
BINOMIAL_RE = re.compile(r"^[A-Z][a-z]+\s+[a-z][a-z-]+$")

# Epithets that are placeholders and should never be considered a species
BANNED_EPITHETS = {"sp", "sp.", "spp", "spp."}


@dataclass
class BuildStats:
    total_entries: int
    total_names: int
    per_kingdom_entries: Dict[str, int]
    per_kingdom_names: Dict[str, int]


def normalize_kingdoms(s: str) -> str:
    """Normalise a kingdom selection string

    - Default is ABE
    - `all` means ABEVO
    - Removes duplicates while preserving order
    """

    s = (s or "").strip()
    if not s:
        s = "ABE"
    if s.lower() == "all":
        s = KNOWN_KINGDOMS

    out: List[str] = []
    for ch in s:
        if ch in KNOWN_KINGDOMS and ch not in out:
            out.append(ch)
    return "".join(out)


def clean_name(name: str, *, strip_parentheses: bool = True) -> str:
    s = name.strip()
    if strip_parentheses:
        # Speclist often appends isolates/strains in parentheses
        if " (" in s:
            s = s.split(" (", 1)[0]
    s = re.sub(r"\s+", " ", s)
    s = s.strip().strip(".;")
    return s


def is_binomial(name: str) -> bool:
    if not BINOMIAL_RE.fullmatch(name):
        return False
    parts = name.split()
    if len(parts) != 2:
        return False
    genus, epithet = parts
    if epithet in BANNED_EPITHETS:
        return False
    if len(epithet) < 3:
        return False
    return True


def _iter_real_section_lines(lines: Iterable[str]) -> Iterable[str]:
    """Yield lines in the 'Real organism codes' table of UniProt speclist."""

    in_table = False
    for line in lines:
        if not in_table:
            if TABLE_START_RE.match(line):
                in_table = True
            continue

        # Skip the table separator line
        if line.startswith("_____"):
            continue

        # Stop before the Virtual regrouping codes section
        if VIRTUAL_SECTION_RE.match(line):
            break

        yield line.rstrip("\n")


def extract_names_from_speclist(
    speclist_path: str,
    *,
    kingdoms: str = "ABE",
    include_synonyms: bool = False,
    synonyms_official_only: bool = True,
    strip_parentheses: bool = True,
    binomial_only: bool = True,
) -> Tuple[Dict[str, Set[str]], BuildStats]:
    """Extract scientific names from UniProt speclist.txt

    Returns:
      - per_kingdom: dict of kingdom -> set of extracted names
      - stats: counters for auditing

    The extractor:
      - selects only the 'Real organism codes' section
      - filters entries by kingdom letter
      - extracts N= and S= fields (ignores C=)
      - optionally restricts to strict binomial names
    """

    kingdoms_norm = normalize_kingdoms(kingdoms)

    per_kingdom: Dict[str, Set[str]] = {k: set() for k in KNOWN_KINGDOMS}
    per_kingdom_entries: Dict[str, int] = {k: 0 for k in KNOWN_KINGDOMS}

    with open(speclist_path, "r", encoding="utf-8") as f:
        table_lines = list(_iter_real_section_lines(f))

    @dataclass
    class _Entry:
        kingdom: str
        n_names: List[str]
        s_names: List[str]

    entries: List[_Entry] = []

    current_kingdom: Optional[str] = None
    current_lines: List[str] = []

    def flush_entry() -> None:
        nonlocal current_kingdom, current_lines
        if current_kingdom is None:
            current_lines = []
            return

        per_kingdom_entries[current_kingdom] += 1

        n_names: List[str] = []
        s_names: List[str] = []
        for ln in current_lines:
            m_n = FIELD_N_RE.search(ln)
            if m_n:
                n_names.append(m_n.group(1))
            m_s = FIELD_S_RE.search(ln)
            if m_s:
                s_names.append(m_s.group(1))

        entries.append(
            _Entry(kingdom=current_kingdom, n_names=n_names, s_names=s_names)
        )

        current_kingdom = None
        current_lines = []

    for line in table_lines:
        m = ENTRY_RE.match(line)
        if m:
            # New entry starts
            flush_entry()
            _code, k, _taxid, _rest = m.groups()
            if k not in KNOWN_KINGDOMS:
                current_kingdom = None
                current_lines = []
                continue
            current_kingdom = k
            current_lines = [line]
        else:
            # Continuation line; keep as-is
            if current_kingdom is not None:
                current_lines.append(line)

    flush_entry()

    # Build a global official-name set for synonym filtering
    official_names: Set[str] = set()
    for e in entries:
        for raw in e.n_names:
            nm = clean_name(raw, strip_parentheses=strip_parentheses)
            if not nm:
                continue
            if binomial_only and not is_binomial(nm):
                continue
            official_names.add(nm)

    # Populate per-kingdom sets from the parsed entries
    for e in entries:
        k = e.kingdom
        for raw in e.n_names:
            nm = clean_name(raw, strip_parentheses=strip_parentheses)
            if not nm:
                continue
            if binomial_only and not is_binomial(nm):
                continue
            per_kingdom[k].add(nm)

        if include_synonyms:
            for raw in e.s_names:
                nm = clean_name(raw, strip_parentheses=strip_parentheses)
                if not nm:
                    continue
                if binomial_only and not is_binomial(nm):
                    continue
                if synonyms_official_only and nm not in official_names:
                    # UniProt `S=` includes English common-name synonyms
                    # Only keep synonyms that correspond to some official name
                    continue
                per_kingdom[k].add(nm)

    # Filter by selected kingdoms
    selected = {k: per_kingdom[k] for k in kingdoms_norm}

    per_kingdom_names: Dict[str, int] = {k: len(per_kingdom[k]) for k in KNOWN_KINGDOMS}

    stats = BuildStats(
        total_entries=sum(per_kingdom_entries.values()),
        total_names=sum(len(v) for v in selected.values()),
        per_kingdom_entries=per_kingdom_entries,
        per_kingdom_names=per_kingdom_names,
    )

    return selected, stats


def write_name_list(path: str, names: Sequence[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(names) + "\n", encoding="utf-8", newline="\n")


def build_cli(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    default_speclist = repo_root / "tea_spec" / "data" / "speclist.txt"
    default_out = repo_root / "work" / "species" / "all_species.txt"

    ap = argparse.ArgumentParser(
        description="Extract a species list from UniProt speclist.txt"
    )
    ap.add_argument(
        "--speclist",
        default=str(default_speclist),
        help="Path to UniProt speclist.txt (default: tea_spec/data/speclist.txt)",
    )
    ap.add_argument(
        "--out",
        default=str(default_out),
        help="Output path for the combined list (default: work/species/all_species.txt)",
    )
    ap.add_argument(
        "--kingdoms",
        default="ABE",
        help="Which kingdoms to include (default: ABE). Use e.g. ABEVO or V or all.",
    )
    ap.add_argument(
        "--write-kingdom-lists",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write per-kingdom lists into --kingdom-dir.",
    )
    ap.add_argument(
        "--kingdom-dir",
        default=None,
        help="Directory for per-kingdom lists (default: sibling of --out).",
    )
    ap.add_argument(
        "--include-synonyms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include S= synonyms (default: false). WARNING: S= lines may contain common-name synonyms.",
    )
    ap.add_argument(
        "--synonyms-official-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --include-synonyms is set, keep only synonyms that are also an official N= name somewhere (default: true)",
    )
    ap.add_argument(
        "--strip-parentheses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strip parenthetical qualifiers like (isolate ...) (default: true)",
    )
    ap.add_argument(
        "--binomial-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only strict binomials 'Genus species' (default: true)",
    )

    args = ap.parse_args(argv)

    selected, stats = extract_names_from_speclist(
        args.speclist,
        kingdoms=args.kingdoms,
        include_synonyms=args.include_synonyms,
        synonyms_official_only=args.synonyms_official_only,
        strip_parentheses=args.strip_parentheses,
        binomial_only=args.binomial_only,
    )

    # Merge selected kingdoms into one combined list
    combined: Set[str] = set()
    for _k, names in selected.items():
        combined.update(names)

    combined_sorted = sorted(combined)
    write_name_list(args.out, combined_sorted)

    print(
        f"[SAVE] {args.out}  names={len(combined_sorted)}  kingdoms={normalize_kingdoms(args.kingdoms)}"
    )
    print(
        f"[INFO] entries_total={stats.total_entries}  names_total={stats.total_names}"
    )
    for k in KNOWN_KINGDOMS:
        print(
            f"[INFO] kingdom={k} entries={stats.per_kingdom_entries.get(k, 0)} names={stats.per_kingdom_names.get(k, 0)}"
        )

    if args.write_kingdom_lists:
        out_dir = (
            Path(args.kingdom_dir)
            if args.kingdom_dir
            else Path(args.out).resolve().parent
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Re-run extraction with all kingdoms, then write each kingdom list
        all_selected, _all_stats = extract_names_from_speclist(
            args.speclist,
            kingdoms="all",
            include_synonyms=args.include_synonyms,
            synonyms_official_only=args.synonyms_official_only,
            strip_parentheses=args.strip_parentheses,
            binomial_only=args.binomial_only,
        )
        for k in KNOWN_KINGDOMS:
            names_k = sorted(all_selected.get(k, set()))
            out_path = out_dir / f"all_species_{k}.txt"
            write_name_list(str(out_path), names_k)
            print(f"[SAVE] {out_path}  names={len(names_k)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(build_cli())
