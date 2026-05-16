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
import csv
import re
from pathlib import Path
from typing import Dict, Tuple

from io_utils import slugify_model_id


def _model_to_dirname(model_id: str) -> str:
    return slugify_model_id(model_id)


def _resolve_summaries_root(workdir: Path, model_id: str | None) -> Path:
    """
    Resolve the summaries root.

    Supports both layouts:
      - <workdir>/summaries_from_cache/seed<seed>/<method>/minimal_summary.csv
      - <workdir>/summaries_from_cache/models/<model_dir>/seed<seed>/<method>/minimal_summary.csv

    If --model is not provided and the legacy layout is not present, the function auto-selects a single model directory if one exists.

    """

    base = workdir / "summaries_from_cache"
    if model_id and model_id.strip():
        return base / "models" / _model_to_dirname(model_id)

    # Prefer legacy layout if present.
    if any(re.fullmatch(r"seed\d+", p.name) for p in base.glob("seed*") if p.is_dir()):
        return base

    models = base / "models"
    if models.exists():
        dirs = sorted([p for p in models.iterdir() if p.is_dir()])
        if len(dirs) == 1:
            return dirs[0]
        if len(dirs) > 1:
            names = ", ".join(p.name for p in dirs)
            raise SystemExit(
                "Multiple model result folders found under work/summaries_from_cache/models. "
                "Pass --model to select one. Available: " + names
            )

    return base


def _cond(baseline: str, abbrev: bool) -> str:
    if baseline == "synonym":
        return "species_abbrev_delta_vs_syn" if abbrev else "species_full_delta_vs_syn"
    if baseline == "random":
        return (
            "species_abbrev_delta_vs_rand" if abbrev else "species_full_delta_vs_rand"
        )
    raise ValueError(f"Unknown baseline: {baseline}")


def _read_one(
    csv_path: Path, *, species: str, metric: str, baseline: str, abbrev: bool
) -> Tuple[float, float, float] | None:
    if not csv_path.exists():
        return None
    want_cond = _cond(baseline, abbrev)
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("stat") or "") != metric:
                continue
            if (row.get("condition") or "") != want_cond:
                continue
            if (row.get("anchor") or "") != species:
                continue
            try:
                mean = float(row["mean"])
                lo = float(row["ci_lo"])
                hi = float(row["ci_hi"])
            except Exception:
                return None
            return mean, lo, hi
    return None


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="work")
    ap.add_argument(
        "--model",
        default="",
        help=(
            "Model id used for the run (selects the matching model-slug directory under work/summaries_from_cache/models). "
            "If omitted, the script uses the legacy layout if present, or auto-selects a single model dir."
        ),
    )
    ap.add_argument("--species", required=True)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--methods", default="none,abtt,whiten")
    ap.add_argument("--baselines", default="synonym,random")
    ap.add_argument("--metric", default="one_minus_cos")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    species = args.species
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    metric = args.metric

    summaries_root = _resolve_summaries_root(workdir, args.model)

    print(f"SPECIES: {species}")
    print(f"WORKDIR: {workdir}")
    print(f"SEEDS: {','.join(map(str, seeds))}")
    print(f"METHODS: {','.join(methods)}")
    print(f"BASELINES: {','.join(baselines)}")
    print(f"SUMMARIES_ROOT: {summaries_root}")
    print()

    base = summaries_root

    for baseline in baselines:
        print(f"BASELINE: {baseline}")
        for method in methods:
            print(f"  METHOD: {method}")
            for seed in seeds:
                csv_path = base / f"seed{seed}" / method / "minimal_summary.csv"
                full = _read_one(
                    csv_path,
                    species=species,
                    metric=metric,
                    baseline=baseline,
                    abbrev=False,
                )
                abbr = _read_one(
                    csv_path,
                    species=species,
                    metric=metric,
                    baseline=baseline,
                    abbrev=True,
                )
                if full is None and abbr is None:
                    print(f"    seed{seed}: missing")
                    continue

                parts: Dict[str, Tuple[float, float, float]] = {}
                if full is not None:
                    parts["FULL"] = full
                if abbr is not None:
                    parts["ABBR"] = abbr

                out = []
                for label in ("FULL", "ABBR"):
                    if label not in parts:
                        continue
                    mean, lo, hi = parts[label]
                    sign = "POS" if lo > 0 else ("NEG" if hi < 0 else "MIX")
                    out.append(
                        f"{label} mean={_fmt(mean)} CI=[{_fmt(lo)},{_fmt(hi)}] {sign}"
                    )

                print(f"    seed{seed}: " + " | ".join(out))
            print()
        print()


if __name__ == "__main__":
    main()
