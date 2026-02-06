from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional


def _iter_manifest_paths(path: Path) -> List[Path]:
    # return manifest files from a directory, or the file itself
    if path.is_file():
        return [path]

    if not path.exists():
        return []

    syn = sorted(path.glob("*.synswap.jsonl"))
    rnd = sorted(path.glob("*.randswap.jsonl"))
    return syn + rnd


def _matches(value: Optional[str], pattern: Optional[str]) -> bool:
    if not pattern:
        return True
    if value is None:
        return False
    return pattern.lower() in value.lower()


def _format_row(j: dict) -> str:
    species = j.get("species")
    variant = j.get("variant")
    idx = j.get("template_index")
    ok = j.get("ok")
    reason = j.get("reason")
    attempts = j.get("attempts")
    changed = j.get("changed_token")
    repl = j.get("replacement")
    protected = j.get("protected")
    text_in = j.get("text_in")
    text_out = j.get("text_out")

    lines: List[str] = []
    lines.append(
        f"species={species}  variant={variant}  template_index={idx}  ok={ok}  attempts={attempts}  reason={reason}"
    )
    if changed is not None or repl is not None:
        lines.append(f"swap={changed!r} -> {repl!r}")
    if protected:
        lines.append(f"protected={protected!r}")
    if text_in is not None:
        lines.append("IN:  " + str(text_in))
    if text_out is not None:
        lines.append("OUT: " + str(text_out))
    return "\n".join(lines)


def inspect_manifests(
    paths: Iterable[Path],
    max_rows: int,
    species_filter: Optional[str],
    variant_filter: Optional[str],
    ok_only: bool,
) -> None:
    """Print a small sample of swap attempts."""
    seen = 0
    for p in paths:
        if not p.exists():
            continue

        print(f"\n===== {p} =====\n")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    j = json.loads(line)
                except Exception:
                    continue

                if ok_only and not j.get("ok"):
                    continue

                if not _matches(str(j.get("species")), species_filter):
                    continue

                if not _matches(str(j.get("variant")), variant_filter):
                    continue

                print(_format_row(j))
                print("-" * 80)

                seen += 1
                if seen >= max_rows:
                    return


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    default_cache = repo_root / "work" / "cache"

    p = argparse.ArgumentParser(
        description=(
            "Inspect swap manifest files (*.synswap.jsonl, *.randswap.jsonl). "
            "These are written by embed_cache.py when --syn-write-manifest and/or --rand-write-manifest are enabled."
        )
    )
    p.add_argument(
        "--path",
        type=str,
        default=str(default_cache),
        help="Manifest file or a directory to search (default: work/cache)",
    )
    p.add_argument("--n", type=int, default=20, help="Maximum rows to print")
    p.add_argument(
        "--species", type=str, default=None, help="Substring filter for species"
    )
    p.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Substring filter for variant (syn_full, syn_abbr, rand_full, rand_abbr)",
    )
    p.add_argument("--ok-only", action="store_true", help="Only print successful swaps")
    return p


def main() -> None:
    a = build_parser().parse_args()
    path = Path(a.path)

    paths = _iter_manifest_paths(path)
    if not paths:
        raise SystemExit(f"No manifests found at: {path}")

    inspect_manifests(
        paths=paths,
        max_rows=int(a.n),
        species_filter=a.species,
        variant_filter=a.variant,
        ok_only=bool(a.ok_only),
    )


if __name__ == "__main__":
    main()
