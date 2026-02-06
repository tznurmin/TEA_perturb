"""
Generate template sentences from curated TEA-style datasets.

This generator consumes a curated TEA dataset where each example stores
tokenised source text as a list of words. It extracts single-sentence
templates that:

  1) Contain exactly one recognisable species mention in the original text based on the provided species list
  2) After replacing that mention with the anchor phrase, contain exactly one anchor mention (full or abbreviated) and no other recognised species mentions
  3) Fall within a token-length band (whitespace tokens)

Outputs:
  --out-grouped : {"relevant": [...], "irrelevant": [...]} (strings)
  --out-flat    : [{"id": "s0", "text": "...", "context": "relevant"}, ...]

The flat output is compatible with tea_spec/src/embed_cache.py (--dataset)

Determinism:
  - by default, the script iterates the curated dataset in file order and selects the first matching sentence per (crc, context)
  - optional shuffling is available
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

from utils import abbreviate_binomial

_CIT_PAT = re.compile(r"\[\s*(?:\d+(?:\s*[-\u2013]\s*\d+)?(?:\s*,\s*\d+)*)\s*\]")
_FIGTAB_PAREN = re.compile(r"\((?:\s*(?:supplementary|figure|fig|table)[^)]*)\)", re.I)
_FLOATING_NUM = re.compile(r"\)\s*\d+\s*\.?\s*")
_MULTISPACE = re.compile(r"\s+")


def clean_text_block(raw: str) -> str:
    t = html.unescape(raw)
    # strip TEA inline markers
    t = t.replace("$i$", "").replace("$/i$", "")
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = _CIT_PAT.sub("", t)
    t = _FIGTAB_PAREN.sub("", t)
    t = _FLOATING_NUM.sub(")", t)
    t = _MULTISPACE.sub(" ", t).strip()
    return t


def _punkt_tokenizer() -> PunktSentenceTokenizer:
    pp = PunktParameters()
    pp.abbrev_types = {
        "fig",
        "figs",
        "dr",
        "mr",
        "ms",
        "no",
        "et",
        "al",
        "e.g",
        "i.e",
        "s",
        "e",
        "p",
        "m",
        "vs",
    }
    return PunktSentenceTokenizer(pp)


def split_clean_sentences(raw_block: str) -> List[str]:
    txt = clean_text_block(raw_block)
    tok = _punkt_tokenizer()
    sents = [s.strip() for s in tok.tokenize(txt) if s.strip()]
    return sents


_FULL_OR_ABBR_PAT = re.compile(
    r"\b(?:[A-Z][a-z]{2,}\s+[a-z]{2,}|[A-Z]\.\s*[a-z]{2,})\b"
)


@dataclasses.dataclass(frozen=True)
class SpeciesMention:
    raw: str
    canonical: str
    span: Tuple[int, int]


def _normalize_spaces(s: str) -> str:
    return _MULTISPACE.sub(" ", s.strip())


def _canonicalize_species(s: str) -> str:
    """Canonicalize a matched species-like string.

    - Full: "Genus species" -> as-is (with single spaces)
    - Abbr: "G. species" or "G.species" -> "G. species"
    """
    s = _normalize_spaces(s)
    if re.fullmatch(r"[A-Z]\.\s*[a-z]{2,}", s):
        g = s[0]
        rest = s.split(".", 1)[1].strip()
        return f"{g}. {rest}"
    return s


def load_species_list(path: Path) -> List[str]:
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        out.append(s)
    return out


def build_species_sets(species_full: Sequence[str]) -> Tuple[set[str], set[str]]:
    full = set(_normalize_spaces(x) for x in species_full if x.strip())
    abbr = set(abbreviate_binomial(x) for x in full)
    return full, abbr


def find_species_mentions(
    text: str, full_set: set[str], abbr_set: set[str]
) -> List[SpeciesMention]:
    """
    Return all recognisable species mentions in order of appearance.

    Recognition rule:
      - full binomial must be in full_set
      - abbreviated binomial must be in abbr_set

    Returns mentions with raw string (as in text), canonical string, and span.
    """
    out: List[SpeciesMention] = []
    seen: set[Tuple[int, int]] = set()

    for m in _FULL_OR_ABBR_PAT.finditer(text):
        raw = m.group(0)
        canon = _canonicalize_species(raw)
        if canon in full_set or canon in abbr_set:
            sp = (m.start(), m.end())
            if sp in seen:
                continue
            seen.add(sp)
            out.append(SpeciesMention(raw=raw, canonical=canon, span=sp))

    return out


def replace_span(text: str, span: Tuple[int, int], replacement: str) -> str:
    i, j = span
    return text[:i] + replacement + text[j:]


def has_other_species(
    sent: str, anchor_full: str, full_set: set[str], abbr_set: set[str]
) -> bool:
    """
    Return True if the sentence contains any non-anchor species mention.

    Two-stage detection:

    1) Whitelist based detection using the provided species sets.
       This is the preferred path (low false positives).

    2) Regex fallback: if the species whitelist is incomplete, reject any extra
       binomial-looking mention (full or abbreviated) that is not the anchor.
    """
    anchor_abbr = abbreviate_binomial(anchor_full)

    # stage 1: known species mentions (whitelist-based)
    mentions = find_species_mentions(sent, full_set=full_set, abbr_set=abbr_set)
    for m in mentions:
        if m.canonical not in (anchor_full, anchor_abbr):
            return True

    # stage 2: regex fallback for unknown-but-binomial-looking mentions
    for m in _FULL_OR_ABBR_PAT.finditer(sent):
        canon = _canonicalize_species(m.group(0))
        if canon in (anchor_full, anchor_abbr):
            continue
        return True

    return False


def exactly_one_anchor(
    sent: str,
    anchor_full: str,
    full_set: set[str],
    abbr_set: set[str],
    min_tokens: int,
    max_tokens: int,
) -> bool:
    anchor_abbr = abbreviate_binomial(anchor_full)
    n_anchor = sent.count(anchor_full) + sent.count(anchor_abbr)
    if n_anchor != 1:
        return False

    if has_other_species(
        sent, anchor_full=anchor_full, full_set=full_set, abbr_set=abbr_set
    ):
        return False

    n_tokens = len(sent.split())
    if not (min_tokens <= n_tokens <= max_tokens):
        return False

    return True


def extract_templates_from_curated(
    curated: Dict,
    full_set: set[str],
    abbr_set: set[str],
    anchor_full: str,
    min_tokens: int = 8,
    max_tokens: int = 60,
    sentence_split: bool = True,
    shuffle_examples: bool = False,
    rng: Optional[random.Random] = None,
) -> Dict[str, List[str]]:
    # extract one template sentence per (crc, context) where possible

    if rng is None:
        rng = random.Random(0)

    relevant: List[str] = []
    irrelevant: List[str] = []

    for crc, block in curated.items():
        if crc == "all_labels":
            continue
        if not isinstance(block, dict):
            continue

        for ctx, out_list in (("irrelevant", irrelevant), ("relevant", relevant)):
            got = False
            examples = block.get(ctx, {}).get("raw", [])
            if not isinstance(examples, list) or not examples:
                continue

            if shuffle_examples:
                examples = list(examples)
                rng.shuffle(examples)

            for e in examples:
                if got:
                    break
                words = e.get("words") if isinstance(e, dict) else None
                if not isinstance(words, list) or not words:
                    continue
                raw_text = " ".join(str(w) for w in words)

                mentions = find_species_mentions(
                    raw_text, full_set=full_set, abbr_set=abbr_set
                )
                if len(mentions) != 1:
                    continue

                swapped = replace_span(raw_text, mentions[0].span, anchor_full)

                candidates = (
                    split_clean_sentences(swapped)
                    if sentence_split
                    else [clean_text_block(swapped)]
                )
                for sent in candidates:
                    if exactly_one_anchor(
                        sent,
                        anchor_full=anchor_full,
                        full_set=full_set,
                        abbr_set=abbr_set,
                        min_tokens=min_tokens,
                        max_tokens=max_tokens,
                    ):
                        out_list.append(sent)
                        got = True
                        break

    return {"relevant": relevant, "irrelevant": irrelevant}


def make_flat(records: Dict[str, List[str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    idx = 0
    for ctx in ("relevant", "irrelevant"):
        for s in records.get(ctx, []):
            out.append({"id": f"s{idx}", "text": s, "context": ctx})
            idx += 1
    return out


def _load_curated_json(path: Path) -> Dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("curated dataset must be a JSON object/dict")
    return obj


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Extract anchor-normalized template sentences from a curated TEA-style dataset."
    )
    repo_root = Path(__file__).resolve().parents[2]
    default_workdir = repo_root / "work"

    p.add_argument(
        "--workdir",
        type=Path,
        default=default_workdir,
        help="Work directory root (default: ./work at repo root)",
    )
    p.add_argument(
        "--curated",
        default=None,
        type=Path,
        help="Path to curated dataset JSON (regular_42.json style). If omitted, uses <workdir>/regular_42.json",
    )
    p.add_argument(
        "--species",
        default=None,
        type=Path,
        help="Path to all_species.txt (one binomial per line). If omitted, uses <workdir>/species/all_species.txt",
    )
    p.add_argument(
        "--anchor",
        default="Staphylococcus aureus",
        help="Placeholder species phrase used as the anchor (default: Staphylococcus aureus)",
    )
    p.add_argument(
        "--out-flat",
        default=None,
        type=Path,
        help="Output JSON list of {id,text,context}. If omitted, uses <workdir>/templates/cluster_all_examples_flat_42.json",
    )
    p.add_argument(
        "--out-grouped",
        default=None,
        type=Path,
        help="Optional output JSON {relevant:[...], irrelevant:[...]} (strings). If omitted, not written.",
    )
    p.add_argument("--min-tokens", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=60)
    p.add_argument(
        "--no-sentence-split",
        action="store_true",
        help="Do not split into sentences; use whole blocks",
    )
    p.add_argument(
        "--shuffle-examples",
        action="store_true",
        help="Shuffle per-crc examples before selection (makes output depend on --seed)",
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for --shuffle-examples")

    args = p.parse_args(argv)

    workdir = Path(args.workdir)
    curated_path = args.curated or (workdir / "regular_42.json")
    species_path = args.species or (workdir / "species" / "all_species.txt")
    out_flat_path = args.out_flat or (
        workdir / "templates" / "cluster_all_examples_flat_42.json"
    )
    out_grouped_path = args.out_grouped

    curated = _load_curated_json(curated_path)
    species = load_species_list(species_path)
    full_set, abbr_set = build_species_sets(species)

    rng = random.Random(args.seed)

    grouped = extract_templates_from_curated(
        curated=curated,
        full_set=full_set,
        abbr_set=abbr_set,
        anchor_full=args.anchor,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        sentence_split=not args.no_sentence_split,
        shuffle_examples=args.shuffle_examples,
        rng=rng,
    )

    flat = make_flat(grouped)

    out_flat_path.parent.mkdir(parents=True, exist_ok=True)
    out_flat_path.write_text(
        json.dumps(flat, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if out_grouped_path is not None:
        out_grouped_path.parent.mkdir(parents=True, exist_ok=True)
        out_grouped_path.write_text(
            json.dumps(grouped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(
        f"[TEMPLATES] relevant={len(grouped['relevant'])} irrelevant={len(grouped['irrelevant'])} total={len(flat)}"
    )
    print(f"[WROTE] {out_flat_path}")
    if out_grouped_path is not None:
        print(f"[WROTE] {out_grouped_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
