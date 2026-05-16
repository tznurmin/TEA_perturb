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
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from io_utils import slugify_model_id


# Helper script to inspect deltas
#
# Prints pooled medians over taxa:
#  - baseline shifts (synonym_full, random_full)
#  - implied species substitution shifts, recovered as baseline_shift + delta
#
# -> useful when the baseline perturbations have very different magnitudes across models, which can change how delta should be interpreted


@dataclass(frozen=True)
class Row:
    anchor: str
    mean: float


def _model_to_dirname(model_id: str) -> str:
    return slugify_model_id(model_id)


def _resolve_summaries_root(workdir: Path, model_id: str | None) -> Path:
    base = workdir / "summaries_from_cache"
    models_dir = base / "models"

    # Legacy layout.
    if model_id is None or model_id.strip() == "":
        has_seed_dirs = any(
            p.is_dir() and re.fullmatch(r"seed\d+", p.name) for p in base.iterdir()
        ) if base.exists() else False
        if has_seed_dirs:
            return base

        if models_dir.exists():
            model_dirs = sorted([p for p in models_dir.iterdir() if p.is_dir()])
            if len(model_dirs) == 1:
                return model_dirs[0]
            if len(model_dirs) > 1:
                opts = "\n".join(f"- {p.name}" for p in model_dirs)
                raise SystemExit(
                    "Multiple model result roots detected. Pass --model. Available: \n" + opts
                )

        return base

    return models_dir / _model_to_dirname(model_id)


def _parse_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs: Sequence[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return float("nan")
    m = n // 2
    return ys[m] if n % 2 == 1 else 0.5 * (ys[m - 1] + ys[m])


def _fmt(x: float, nd: int = 6) -> str:
    if not math.isfinite(x):
        return "nan"
    if abs(x) >= 1:
        s = f"{x:.{nd}f}"
        return s.rstrip("0").rstrip(".")
    return f"{x:.{nd}g}"


def _read_condition(path: Path, metric: str, condition: str) -> Dict[str, Row]:
    out: Dict[str, Row] = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("stat") != metric:
                continue
            if row.get("condition") != condition:
                continue
            anchor = (row.get("anchor") or "").strip()
            if not anchor:
                continue
            mean = _parse_float(row.get("mean", "nan"))
            if not math.isfinite(mean):
                continue
            out[anchor] = Row(anchor=anchor, mean=mean)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="work")
    ap.add_argument(
        "--model",
        default="",
        help="Optional model id. If set, reads from the matching model-slug directory under work/summaries_from_cache/models.",
    )
    ap.add_argument("--seeds", default="")
    ap.add_argument("--methods", default="none,abtt,whiten")
    ap.add_argument("--metric", default="one_minus_cos")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    metric = args.metric
    model = args.model.strip() or None

    summaries_root = _resolve_summaries_root(workdir, model)

    print(f"WORKDIR: {workdir}")
    print(f"SEEDS: {','.join(map(str, seeds))}")
    print(f"METHODS: {','.join(methods)}")
    print(f"METRIC: {metric}")
    if model:
        print(f"MODEL: {model}")
    print(f"SUMMARIES_ROOT: {summaries_root}")
    print()

    for method in methods:
        per_seed_base_syn = []
        per_seed_base_rand = []
        per_seed_delta_syn = []
        per_seed_delta_rand = []

        ok = True
        for seed in seeds:
            csv_path = summaries_root / f"seed{seed}" / method / "minimal_summary.csv"
            if not csv_path.exists():
                ok = False
                continue

            per_seed_base_syn.append(_read_condition(csv_path, metric, "synonym_full"))
            per_seed_base_rand.append(_read_condition(csv_path, metric, "random_full"))
            per_seed_delta_syn.append(
                _read_condition(csv_path, metric, "species_full_delta_vs_syn")
            )
            per_seed_delta_rand.append(
                _read_condition(csv_path, metric, "species_full_delta_vs_rand")
            )

        print(method)
        if not ok or not per_seed_base_syn:
            print("  missing")
            print()
            continue

        # Anchors present in all required maps.
        keys = None
        for maps in (
            per_seed_base_syn,
            per_seed_base_rand,
            per_seed_delta_syn,
            per_seed_delta_rand,
        ):
            for m in maps:
                ks = set(m.keys())
                keys = ks if keys is None else keys & ks
        keys = sorted(keys or [])
        n = len(keys)

        if n == 0:
            print("  n=0")
            print()
            continue

        pooled_base_syn = []
        pooled_base_rand = []
        pooled_species_syn = []
        pooled_species_rand = []

        for k in keys:
            bs = _mean([m[k].mean for m in per_seed_base_syn])
            br = _mean([m[k].mean for m in per_seed_base_rand])
            ds = _mean([m[k].mean for m in per_seed_delta_syn])
            dr = _mean([m[k].mean for m in per_seed_delta_rand])

            pooled_base_syn.append(bs)
            pooled_base_rand.append(br)
            pooled_species_syn.append(bs + ds)
            pooled_species_rand.append(br + dr)

        med_base_syn = _median(pooled_base_syn)
        med_base_rand = _median(pooled_base_rand)
        med_species_syn = _median(pooled_species_syn)
        med_species_rand = _median(pooled_species_rand)
        med_gap = _median([abs(a - b) for a, b in zip(pooled_species_syn, pooled_species_rand)])

        print(f"  n={n}")
        print(f"  median baseline shift synonym_full: {_fmt(med_base_syn, 6)}")
        print(f"  median baseline shift random_full : {_fmt(med_base_rand, 6)}")
        print(f"  median species shift (syn baseline): {_fmt(med_species_syn, 6)}")
        print(f"  median species shift (rnd baseline): {_fmt(med_species_rand, 6)}")
        print(f"  median abs gap (syn vs rnd)        : {_fmt(med_gap, 6)}")
        print()


if __name__ == "__main__":
    main()
