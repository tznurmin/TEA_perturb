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
import itertools
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from io_utils import slugify_model_id

# delta is shift(species substitution) minus shift(control baseline)
# FULL and ABBR are full and abbreviated binomial replacements
# fold is (baseline + delta) / baseline
# ABBR fold uses the FULL baseline, because ABBR deltas are computed against the left species' FULL baseline
# SNR is a signal-to-noise ratio across species


@dataclass(frozen=True)
class Row:
    anchor: str
    mean: float
    ci_lo: float
    ci_hi: float


def _parse_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _fmt(x: float, nd: int = 6) -> str:
    if not math.isfinite(x):
        return "nan"
    if abs(x) >= 1:
        s = f"{x:.{nd}f}"
        return s.rstrip("0").rstrip(".")
    return f"{x:.{nd}g}"


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs: Sequence[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return float("nan")
    m = n // 2
    return ys[m] if n % 2 == 1 else 0.5 * (ys[m - 1] + ys[m])


def _std(xs: Sequence[float]) -> float:
    n = len(xs)
    if n <= 1:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return float("nan")

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return float("nan")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")

    res = spearmanr(x, y)
    rho = getattr(res, "correlation", None)
    if rho is None:
        rho = getattr(res, "statistic", None)
    if rho is None:
        rho = res[0]
    return float(rho)


def _detect_seeds_under(summaries_root: Path) -> List[int]:
    out: List[int] = []
    if not summaries_root.exists():
        return out
    for p in summaries_root.iterdir():
        m = re.fullmatch(r"seed(\d+)", p.name)
        if m and p.is_dir():
            out.append(int(m.group(1)))
    return sorted(out)


def _delta_condition(baseline: str, full: bool) -> str:
    if baseline == "synonym":
        return "species_full_delta_vs_syn" if full else "species_abbrev_delta_vs_syn"
    if baseline == "random":
        return "species_full_delta_vs_rand" if full else "species_abbrev_delta_vs_rand"
    raise ValueError(f"unknown baseline: {baseline}")


def _baseline_condition(baseline: str, full: bool) -> str:
    if baseline == "synonym":
        return "synonym_full" if full else "synonym_abbrev"
    if baseline == "random":
        return "random_full" if full else "random_abbrev"
    raise ValueError(f"unknown baseline: {baseline}")


def _read_minimal_summary(path: Path, metric: str, condition: str) -> Dict[str, Row]:
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
            ci_lo = _parse_float(row.get("ci_lo", "nan"))
            ci_hi = _parse_float(row.get("ci_hi", "nan"))
            if not (
                math.isfinite(mean) and math.isfinite(ci_lo) and math.isfinite(ci_hi)
            ):
                continue
            out[anchor] = Row(anchor=anchor, mean=mean, ci_lo=ci_lo, ci_hi=ci_hi)
    return out


def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def _fold(delta: float, baseline: float) -> float:
    if not (math.isfinite(delta) and math.isfinite(baseline)) or baseline <= 0:
        return float("nan")
    return 1.0 + (delta / baseline)


def _score_candidate_root(
    root: Path, seeds: Sequence[int], methods: Sequence[str]
) -> Tuple[int, float]:
    count = 0
    newest = 0.0
    for seed in seeds:
        for method in methods:
            p = root / f"seed{seed}" / method / "minimal_summary.csv"
            if p.exists():
                count += 1
                try:
                    newest = max(newest, p.stat().st_mtime)
                except Exception:
                    pass
    return count, newest


def _auto_pick_summaries_root(
    workdir: Path, seeds: Sequence[int], methods: Sequence[str]
) -> Path:
    base = workdir / "summaries_from_cache"
    if not base.exists():
        return base

    candidates: Dict[Path, int] = {}

    # Any minimal_summary.csv implies a root at parents[2]:
    # <root>/seed42/<method>/minimal_summary.csv
    for f in base.rglob("minimal_summary.csv"):
        try:
            root = f.parents[2]
        except Exception:
            continue
        candidates[root] = candidates.get(root, 0) + 1

    if not candidates:
        return base

    if seeds:
        scored = [(root, *_score_candidate_root(root, seeds, methods)) for root in candidates]
        scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
        best_root, best_count, _ = scored[0]
        return best_root if best_count > 0 else base

    # No seeds specified: pick the root with most files, then newest.
    scored2 = []
    for root, nfiles in candidates.items():
        newest = 0.0
        try:
            for f in root.rglob("minimal_summary.csv"):
                newest = max(newest, f.stat().st_mtime)
        except Exception:
            pass
        scored2.append((root, nfiles, newest))
    scored2.sort(key=lambda t: (t[1], t[2]), reverse=True)
    return scored2[0][0]


def _model_to_dirname(model_id: str) -> str:
    return slugify_model_id(model_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--methods", default="none,abtt,whiten")
    ap.add_argument("--baselines", default="synonym,random")
    ap.add_argument("--metric", default="one_minus_cos")
    ap.add_argument("--no-folds", action="store_true", help="Do not print fold columns.")
    ap.add_argument(
        "--summaries-root",
        default="",
        help="Override summaries root. Expected layout: <root>/seed42/<method>/minimal_summary.csv",
    )
    ap.add_argument(
        "--model",
        default="",
        help="Model id used in the run; maps through the same filesystem-safe slug used by the pipeline.",
    )
    args = ap.parse_args()

    workdir = Path(args.workdir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    metric = args.metric
    show_folds = not args.no_folds

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else []

    if args.summaries_root.strip():
        summaries_root = Path(args.summaries_root)
    elif args.model.strip():
        summaries_root = (
            workdir
            / "summaries_from_cache"
            / "models"
            / _model_to_dirname(args.model)
        )
    else:
        summaries_root = _auto_pick_summaries_root(workdir, seeds, methods)

    if not seeds:
        seeds = _detect_seeds_under(summaries_root)

    per_delta: Dict[Tuple[int, str, str, bool], Dict[str, Row]] = {}
    per_base: Dict[Tuple[int, str, str, bool], Dict[str, Row]] = {}

    for seed in seeds:
        for method in methods:
            csv_path = summaries_root / f"seed{seed}" / method / "minimal_summary.csv"
            if not csv_path.exists():
                continue
            for baseline in baselines:
                for full in (True, False):
                    per_delta[(seed, method, baseline, full)] = _read_minimal_summary(
                        csv_path, metric, _delta_condition(baseline, full)
                    )
                    per_base[(seed, method, baseline, full)] = _read_minimal_summary(
                        csv_path, metric, _baseline_condition(baseline, full)
                    )

    print(f"WORKDIR: {workdir}")
    print(f"SEEDS: {','.join(map(str, seeds))}")
    print(f"METHODS: {','.join(methods)}")
    print(f"BASELINES: {','.join(baselines)}")
    print(f"METRIC: {metric}")
    print(f"SUMMARIES_ROOT: {summaries_root}")
    print("delta is species-substitution shift minus baseline-perturbation shift.")
    if show_folds:
        print("fold is (baseline + delta) / baseline; ABBR fold uses the FULL baseline.")
    print()

    for baseline in baselines:
        rows_out: List[List[str]] = []

        for method in methods:
            d_full_maps = [
                per_delta.get((seed, method, baseline, True), {}) for seed in seeds
            ]
            d_abbr_maps = [
                per_delta.get((seed, method, baseline, False), {}) for seed in seeds
            ]
            b_full_maps = [
                per_base.get((seed, method, baseline, True), {}) for seed in seeds
            ]

            if (
                any(not m for m in d_full_maps)
                or any(not m for m in d_abbr_maps)
                or any(not m for m in b_full_maps)
            ):
                miss = [method, "missing", "", "", "", "", "", "", "", "", ""]
                if show_folds:
                    miss += ["", "", "", ""]
                rows_out.append(miss)
                continue

            keys_full = set.intersection(*(set(m.keys()) for m in d_full_maps))
            keys_abbr = set.intersection(*(set(m.keys()) for m in d_abbr_maps))
            keys_base = set.intersection(*(set(m.keys()) for m in b_full_maps))
            keys = sorted(keys_full & keys_abbr & keys_base)
            n = len(keys)

            if n == 0:
                miss = [method, "0", "", "", "", "", "", "", "", "", ""]
                if show_folds:
                    miss += ["", "", "", ""]
                rows_out.append(miss)
                continue

            pooled_full: Dict[str, float] = {}
            pooled_abbr: Dict[str, float] = {}
            sd_full: Dict[str, float] = {}
            sd_abbr: Dict[str, float] = {}

            pooled_full_fold: Dict[str, float] = {}
            pooled_abbr_fold: Dict[str, float] = {}

            for k in keys:
                v_full = [
                    per_delta[(seed, method, baseline, True)][k].mean for seed in seeds
                ]
                v_abbr = [
                    per_delta[(seed, method, baseline, False)][k].mean for seed in seeds
                ]
                pooled_full[k] = _mean(v_full)
                pooled_abbr[k] = _mean(v_abbr)
                sd_full[k] = _std(v_full)
                sd_abbr[k] = _std(v_abbr)

                if show_folds:
                    ffs = []
                    afs = []
                    for seed in seeds:
                        d_f = per_delta[(seed, method, baseline, True)][k].mean
                        d_a = per_delta[(seed, method, baseline, False)][k].mean
                        b_f = per_base[(seed, method, baseline, True)][k].mean
                        ffs.append(_fold(d_f, b_f))
                        afs.append(_fold(d_a, b_f))
                    pooled_full_fold[k] = _mean([x for x in ffs if math.isfinite(x)])
                    pooled_abbr_fold[k] = _mean([x for x in afs if math.isfinite(x)])

            pooled_full_vals = [pooled_full[k] for k in keys]
            pooled_abbr_vals = [pooled_abbr[k] for k in keys]

            mean_full = _mean(pooled_full_vals)
            med_full = _median(pooled_full_vals)
            mean_abbr = _mean(pooled_abbr_vals)
            med_abbr = _median(pooled_abbr_vals)

            penalties = []
            full_gt_abbr = 0
            for k in keys:
                f = pooled_full[k]
                a = pooled_abbr[k]
                if f > 0:
                    penalties.append(max(0.0, 1.0 - (a / f)))
                if f > a:
                    full_gt_abbr += 1
            pen_med = _median(penalties) if penalties else float("nan")
            pct_full_gt_abbr = 100.0 * full_gt_abbr / n

            sp_full_vals = []
            sp_abbr_vals = []
            for s1, s2 in itertools.combinations(seeds, 2):
                m1f = per_delta[(s1, method, baseline, True)]
                m2f = per_delta[(s2, method, baseline, True)]
                m1a = per_delta[(s1, method, baseline, False)]
                m2a = per_delta[(s2, method, baseline, False)]
                v1f = [m1f[k].mean for k in keys]
                v2f = [m2f[k].mean for k in keys]
                v1a = [m1a[k].mean for k in keys]
                v2a = [m2a[k].mean for k in keys]
                sp_full_vals.append(_spearman(v1f, v2f))
                sp_abbr_vals.append(_spearman(v1a, v2a))
            sp_full = _mean([v for v in sp_full_vals if math.isfinite(v)])
            sp_abbr = _mean([v for v in sp_abbr_vals if math.isfinite(v)])

            noise_full = _mean([sd_full[k] for k in keys if math.isfinite(sd_full[k])])
            noise_abbr = _mean([sd_abbr[k] for k in keys if math.isfinite(sd_abbr[k])])
            signal_sd_full = _std(pooled_full_vals)
            signal_sd_abbr = _std(pooled_abbr_vals)
            snr_full = (
                signal_sd_full / noise_full
                if noise_full and math.isfinite(noise_full) and noise_full > 0
                else float("nan")
            )
            snr_abbr = (
                signal_sd_abbr / noise_abbr
                if noise_abbr and math.isfinite(noise_abbr) and noise_abbr > 0
                else float("nan")
            )

            out_row = [
                method,
                str(n),
                _fmt(mean_full, 6),
                _fmt(med_full, 6),
                _fmt(mean_abbr, 6),
                _fmt(med_abbr, 6),
                _fmt(pen_med, 4),
                _fmt(pct_full_gt_abbr, 4),
                _fmt(sp_full, 4),
                _fmt(sp_abbr, 4),
                _fmt(snr_full, 4) + "/" + _fmt(snr_abbr, 4),
            ]

            if show_folds:
                full_folds = [
                    pooled_full_fold[k]
                    for k in keys
                    if math.isfinite(pooled_full_fold.get(k, float("nan")))
                ]
                abbr_folds = [
                    pooled_abbr_fold[k]
                    for k in keys
                    if math.isfinite(pooled_abbr_fold.get(k, float("nan")))
                ]
                out_row += [
                    _fmt(_mean(full_folds), 6),
                    _fmt(_median(full_folds), 6),
                    _fmt(_mean(abbr_folds), 6),
                    _fmt(_median(abbr_folds), 6),
                ]

            rows_out.append(out_row)

        print(f"BASELINE: {baseline}")

        headers = [
            "method",
            "n",
            "mean_delta_full",
            "median_delta_full",
            "mean_delta_abbr",
            "median_delta_abbr",
            "abbr_pen_med",
            "%FULL>ABBR",
            "Spearman_full",
            "Spearman_abbr",
            "SNR_full/abbr",
        ]
        if show_folds:
            headers += [
                "mean_fold_full",
                "median_fold_full",
                "mean_fold_abbr",
                "median_fold_abbr",
            ]

        _print_table(headers=headers, rows=rows_out)
        print()


if __name__ == "__main__":
    main()
