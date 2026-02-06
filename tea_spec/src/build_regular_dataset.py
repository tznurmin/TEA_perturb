"""
Compile a TEA-style `regular_<seed>.json` from TEA_curated_data.

The upstream curated data in TEA_curated_data stores word-level tag locations.
Those tags are not sentence-scoped, so this script maps each tag span back to
a sentence slice inside the corresponding source article text.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_SENT_END = (".", "?", "!")


def _strip_tea_markup_token(tok: str) -> str:
    # Strip TEA inline markers like $i$ and $/i$
    return tok.replace("$i$", "").replace("$/i$", "")


def _tokenize_words(text: str) -> List[str]:
    # Word indexing in TEA_curated_data is described as word level
    raw = re.findall(r"\S+", text)
    out: List[str] = []
    for tok in raw:
        t = _strip_tea_markup_token(tok)
        if not t:
            continue
        out.append(t)
    return out


def _ends_sentence(tok: str) -> bool:
    return tok.endswith(_SENT_END) or tok in _SENT_END


def _sentence_span(
    words: Sequence[str], span_start: int, span_len: int
) -> Tuple[int, int]:
    # Return (lo, hi) indices (hi exclusive) for the sentence containing the span
    n = len(words)
    if n == 0:
        return (0, 0)

    i = max(0, min(span_start, n - 1))
    j = max(i + 1, min(span_start + max(1, span_len), n))

    lo = i
    while lo > 0 and not _ends_sentence(words[lo - 1]):
        lo -= 1

    hi = j
    while hi < n and not _ends_sentence(words[hi - 1]):
        hi += 1

    return (lo, hi)


def _parse_loc(loc: str) -> Optional[Tuple[int, int]]:
    # Expect "42+2" as format
    m = re.fullmatch(r"(\d+)\+(\d+)", str(loc).strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def _is_negative_tag(tag: str) -> bool:
    # Both datasets contain a negatives category to mark irrelevant examples
    t = tag.strip().lower()
    return t == "negatives" or t.startswith("negatives/")


@dataclass(frozen=True)
class TagSpan:
    tag: str
    start: int
    length: int


def _load_curated_json(path: Path) -> Dict[str, Dict[str, List[str]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object at top-level in {path}")
    # {article_hash: {tag_name: ["start+len", ...]}}
    out: Dict[str, Dict[str, List[str]]] = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        tag_map: Dict[str, List[str]] = {}
        for tag, locs in v.items():
            if isinstance(locs, list):
                tag_map[str(tag)] = [str(x) for x in locs]
        out[str(k)] = tag_map
    return out


def _iter_spans(tag_map: Dict[str, List[str]]) -> Iterable[TagSpan]:
    for tag, locs in tag_map.items():
        for loc in locs:
            parsed = _parse_loc(loc)
            if parsed is None:
                continue
            start, length = parsed
            yield TagSpan(tag=tag, start=start, length=length)


def build_regular(
    curated_root: Path,
    datasets: Sequence[str],
    out_path: Path,
    dedup: bool = True,
) -> None:
    """Build compiled regular JSON and write it to out_path."""

    curated_root = curated_root.resolve()

    # Load all curated spans across datasets
    by_hash: Dict[str, List[Tuple[str, TagSpan]]] = defaultdict(list)
    for ds in datasets:
        p = curated_root / "curation_data" / ds / f"{ds}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing curated file: {p}")
        cur = _load_curated_json(p)
        for h, tag_map in cur.items():
            for span in _iter_spans(tag_map):
                by_hash[h].append((ds, span))

    compiled: Dict[str, dict] = {}
    missing_txt = 0
    emitted = 0
    emitted_rel = 0
    emitted_irrel = 0

    seen_keys: set[str] = set()

    for h in sorted(by_hash.keys()):
        txt_path = curated_root / "source_articles" / h / f"{h}.txt"
        if not txt_path.exists():
            missing_txt += 1
            continue

        text = txt_path.read_text(encoding="utf-8", errors="replace")
        words = _tokenize_words(text)

        rel_raw: List[dict] = []
        irrel_raw: List[dict] = []

        for ds, span in sorted(
            by_hash[h], key=lambda x: (x[0], x[1].tag, x[1].start, x[1].length)
        ):
            lo, hi = _sentence_span(words, span.start, span.length)
            sent_words = words[lo:hi]
            if not sent_words:
                continue

            ex = {
                "hash": h,
                "dataset": ds,
                "tag": span.tag,
                "loc": f"{span.start}+{span.length}",
                "words": sent_words,
            }

            if dedup:
                # Stable per-sentence key: dataset + tag + sentence text
                k = f"{ds}\0{span.tag}\0{' '.join(sent_words)}"
                if k in seen_keys:
                    continue
                seen_keys.add(k)

            if _is_negative_tag(span.tag):
                irrel_raw.append(ex)
                emitted_irrel += 1
            else:
                rel_raw.append(ex)
                emitted_rel += 1
            emitted += 1

        if rel_raw or irrel_raw:
            compiled[h] = {
                "relevant": {"raw": rel_raw},
                "irrelevant": {"raw": irrel_raw},
            }

    # Add an informational key (ignored by downstream tools)
    compiled["all_labels"] = []
    compiled["_meta"] = {
        "curated_root": str(curated_root),
        "datasets": list(datasets),
        "dedup": bool(dedup),
        "missing_txt": int(missing_txt),
        "emitted_total": int(emitted),
        "emitted_relevant": int(emitted_rel),
        "emitted_irrelevant": int(emitted_irrel),
        "articles_total": int(len(by_hash)),
        "articles_emitted": int(
            sum(1 for k in compiled.keys() if k and not k.startswith("_"))
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(compiled, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build TEA-style regular_*.json from TEA_curated_data"
    )
    ap.add_argument(
        "--curated-root",
        default="work/TEA_curated_data",
        help="Path to TEA_curated_data directory (default: work/TEA_curated_data)",
    )
    ap.add_argument(
        "--datasets",
        default="pathogens,strains",
        help="Comma-separated list of dataset folders under curation_data/ (default: pathogens,strains)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used only for output naming / audit (default: 42)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: work/regular_<seed>.json",
    )
    ap.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable sentence-level deduplication",
    )
    args = ap.parse_args(argv)

    curated_root = Path(args.curated_root)
    datasets = [s.strip() for s in str(args.datasets).split(",") if s.strip()]
    out_path = (
        Path(args.out) if args.out else Path("work") / f"regular_{args.seed}.json"
    )

    build_regular(
        curated_root=curated_root,
        datasets=datasets,
        out_path=out_path,
        dedup=not args.no_dedup,
    )
    print(f"[SAVE] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
