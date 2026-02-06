from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr


def _read_summary_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _collect_species_jsons(out_dir: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for fn in os.listdir(out_dir):
        if fn.startswith("summary_") and fn.endswith(".json"):
            with open(os.path.join(out_dir, fn), "r", encoding="utf-8") as f:
                j = json.load(f)
            s = j.get("_meta", {}).get("anchor")
            if s:
                out[s] = j
    return out


def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _finite(arr: List[Optional[float]]) -> np.ndarray:
    return np.array([x for x in arr if x is not None and math.isfinite(x)], dtype=float)


def _spearman(x: List[Optional[float]], y: List[Optional[float]]) -> float:
    pairs = [
        (xi, yi)
        for xi, yi in zip(x, y)
        if xi is not None and yi is not None and math.isfinite(xi) and math.isfinite(yi)
    ]
    if len(pairs) < 2:
        return float("nan")

    xs, ys = zip(*pairs)
    a = np.asarray(xs, dtype=float)
    b = np.asarray(ys, dtype=float)

    if a.size < 2:
        return float("nan")
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")

    res = spearmanr(a, b)
    rho = getattr(res, "correlation", None)
    if rho is None:
        rho = getattr(res, "statistic", None)
    if rho is None:
        rho = res[0]
    return float(rho)


class RunData:
    """Holds per-run, per-species stats for a metric

    The per-species delta columns can be defined against different baselines:
      - baseline=synonym: delta vs synonym swap (WordNet)
      - baseline=random:  delta vs random word swap (wordfreq)
    """

    def __init__(self, name: str, path: str, metric: str, baseline: str = "synonym"):
        self.name = name
        self.path = path
        self.metric = metric
        self.baseline = baseline

        if baseline not in {"synonym", "random"}:
            raise ValueError(f"Unknown baseline: {baseline}")

        # condition label mapping
        if baseline == "synonym":
            self.cond_baseline_full = "synonym_full"
            self.cond_baseline_abbr = "synonym_abbrev"
            self.cond_delta_full = "species_full_delta_vs_syn"
            self.cond_delta_abbr = "species_abbrev_delta_vs_syn"
        else:
            self.cond_baseline_full = "random_full"
            self.cond_baseline_abbr = "random_abbrev"
            self.cond_delta_full = "species_full_delta_vs_rand"
            self.cond_delta_abbr = "species_abbrev_delta_vs_rand"

        self.rows = _read_summary_csv(os.path.join(path, "minimal_summary.csv"))
        self.species_jsons = _collect_species_jsons(path)

        # species -> values
        self.full_delta: Dict[str, Optional[float]] = {}
        self.abbr_delta: Dict[str, Optional[float]] = {}
        self.full_ci_lo: Dict[str, Optional[float]] = {}
        self.full_ci_hi: Dict[str, Optional[float]] = {}
        self.abbr_ci_lo: Dict[str, Optional[float]] = {}
        self.abbr_ci_hi: Dict[str, Optional[float]] = {}

        self.base_full: Dict[str, Optional[float]] = {}
        self.base_abbr: Dict[str, Optional[float]] = {}

        # raw folds (default ABBREV uses FULL baseline; we also expose an abbr-baseline variant)
        self.full_raw_fold: Dict[str, Optional[float]] = {}
        self.abbr_raw_fold: Dict[str, Optional[float]] = {}  # default = full-baseline
        self.abbr_raw_fold_fullbase: Dict[str, Optional[float]] = {}
        self.abbr_raw_fold_abbrbase: Dict[str, Optional[float]] = {}

        self._parse()

    def _parse(self):
        # optional global baseline (very old runs)
        global_baseline = None
        for r in self.rows:
            if r.get("condition") == "synonym" and r.get("stat") == self.metric:
                global_baseline = _to_float(r.get("mean"))
                break

        # index rows by species/condition/stat
        bys: Dict[str, Dict[Tuple[str, str], dict]] = {}
        for r in self.rows:
            s = r["anchor"]
            bys.setdefault(s, {})
            bys[s][(r["condition"], r["stat"])] = r

        for s, d in bys.items():
            # deltas + CIs
            fd = d.get((self.cond_delta_full, self.metric))
            ad = d.get((self.cond_delta_abbr, self.metric))
            self.full_delta[s] = _to_float(fd["mean"]) if fd else None
            self.abbr_delta[s] = _to_float(ad["mean"]) if ad else None
            self.full_ci_lo[s] = _to_float(fd["ci_lo"]) if fd else None
            self.full_ci_hi[s] = _to_float(fd["ci_hi"]) if fd else None
            self.abbr_ci_lo[s] = _to_float(ad["ci_lo"]) if ad else None
            self.abbr_ci_hi[s] = _to_float(ad["ci_hi"]) if ad else None

            # per-species baselines (preferred) or global baseline
            bf = d.get((self.cond_baseline_full, self.metric))
            ba = d.get((self.cond_baseline_abbr, self.metric))
            self.base_full[s] = _to_float(bf["mean"]) if bf else global_baseline
            self.base_abbr[s] = _to_float(ba["mean"]) if ba else global_baseline

            # raw fold = (baseline + delta) / baseline
            def _fold(delta, base):
                if delta is None or base is None or base <= 0:
                    return None
                return 1.0 + (delta / base)

            # FULL fold uses FULL baseline
            self.full_raw_fold[s] = _fold(self.full_delta[s], self.base_full[s])

            # IMPORTANT: ABBREV delta in this pipeline is computed vs the LEFT species' FULL baseline
            # So the *correct* raw fold for ABBREV uses the FULL baseline
            self.abbr_raw_fold_fullbase[s] = _fold(
                self.abbr_delta[s], self.base_full[s]
            )
            # Also compute an informational variant using ABBREV baseline (not used by default)
            self.abbr_raw_fold_abbrbase[s] = _fold(
                self.abbr_delta[s], self.base_abbr[s]
            )

            # default
            self.abbr_raw_fold[s] = self.abbr_raw_fold_fullbase[s]

    def aggregate(self) -> Dict[str, float]:
        fd = _finite(list(self.full_delta.values()))
        ad = _finite(list(self.abbr_delta.values()))
        ff = _finite(list(self.full_raw_fold.values()))
        af = _finite(list(self.abbr_raw_fold.values()))
        out = {}
        if fd.size:
            out.update(
                full_delta_mean=float(fd.mean()),
                full_delta_median=float(np.median(fd)),
                n_full=int(fd.size),
            )
        if ad.size:
            out.update(
                abbr_delta_mean=float(ad.mean()),
                abbr_delta_median=float(np.median(ad)),
                n_abbr=int(ad.size),
            )
        if ff.size:
            out.update(
                full_fold_mean=float(ff.mean()), full_fold_median=float(np.median(ff))
            )
        if af.size:
            out.update(
                abbr_fold_mean=float(af.mean()), abbr_fold_median=float(np.median(af))
            )

        # share where FULL delta > ABBREV delta
        flags = []
        for s in self.full_delta.keys():
            fdv, adv = self.full_delta.get(s), self.abbr_delta.get(s)
            if (
                fdv is not None
                and adv is not None
                and math.isfinite(fdv)
                and math.isfinite(adv)
            ):
                flags.append(float(fdv > adv))
        if flags:
            out["pct_full_gt_abbrev"] = 100.0 * (sum(flags) / len(flags))
        return out


def compare_runs(runs: List[RunData]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            species = sorted(set(a.full_delta) & set(b.full_delta))
            a_full = [a.full_delta[s] for s in species]
            b_full = [b.full_delta[s] for s in species]
            a_abbr = [a.abbr_delta[s] for s in species]
            b_abbr = [b.abbr_delta[s] for s in species]
            out[f"{a.name}~{b.name}"] = {
                "spearman_full_delta": _spearman(a_full, b_full),
                "spearman_abbrev_delta": _spearman(a_abbr, b_abbr),
            }
    return out


def robust_species(
    runs: List[RunData], fold_thresh: float = 2.0, penalty_min: float = 0.1
) -> List[dict]:
    """Species that are strong and consistent across ALL runs.

    Criteria:
      - FULL raw fold >= fold_thresh in every run
      - ABBREV raw fold <= FULL raw fold * (1 - penalty_min) in every run
      - CI lower bound for FULL delta > 0 in every run (if available)

    Baseline is determined by each RunData instance (synonym vs random).
    """

    species = sorted(set().union(*[set(r.full_delta.keys()) for r in runs]))
    robust: List[dict] = []
    for s in species:
        ok = True
        row: dict = {"species": s}
        for r in runs:
            ff, af = r.full_raw_fold.get(s), r.abbr_raw_fold.get(s)
            if (ff is None) or (ff < fold_thresh):
                ok = False
                break
            if (af is None) or (af > ff * (1 - penalty_min)):
                ok = False
                break
            lo = r.full_ci_lo.get(s)
            if (lo is not None) and (lo <= 0):
                ok = False
                break
            row[f"{r.name}_full_fold"] = ff
            row[f"{r.name}_abbr_fold"] = af
        if ok:
            robust.append(row)
    return robust


def sensitivity_table(runs: List[RunData]) -> List[dict]:
    # Species whose FULL raw fold changes the most across runs (max-min)

    species = sorted(set().union(*[set(r.full_raw_fold.keys()) for r in runs]))
    rows: List[dict] = []
    for s in species:
        vals = []
        for r in runs:
            v = r.full_raw_fold.get(s)
            if v is not None and math.isfinite(v):
                vals.append(v)
        if len(vals) >= 2:
            rows.append({"species": s, "spread": max(vals) - min(vals)})
    rows.sort(key=lambda x: x["spread"], reverse=True)
    return rows


def write_per_species_csv(out_path: str, species: List[str], runs: List[RunData]):
    cols = ["species"]
    for r in runs:
        p = r.name
        cols += [
            f"{p}_baseline_full",
            f"{p}_full_delta",
            f"{p}_full_raw_fold",
            f"{p}_baseline_abbrev",
            f"{p}_abbrev_delta",
            f"{p}_abbrev_raw_fold_fullbase",  # correct baseline for your delta
            f"{p}_abbrev_raw_fold_abbrbase",  # informational
        ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in species:
            row = [s]
            for r in runs:
                row += [
                    r.base_full.get(s),
                    r.full_delta.get(s),
                    r.full_raw_fold.get(s),
                    r.base_abbr.get(s),
                    r.abbr_delta.get(s),
                    r.abbr_raw_fold_fullbase.get(s),
                    r.abbr_raw_fold_abbrbase.get(s),
                ]
            w.writerow(row)


def write_global_csv(out_path: str, runs: List[RunData]):
    keys = [
        "full_delta_mean",
        "full_delta_median",
        "n_full",
        "abbr_delta_mean",
        "abbr_delta_median",
        "n_abbr",
        "full_fold_mean",
        "full_fold_median",
        "abbr_fold_mean",
        "abbr_fold_median",
        "pct_full_gt_abbrev",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run"] + keys)
        for r in runs:
            agg = r.aggregate()
            w.writerow([r.name] + [agg.get(k) for k in keys])


def _sign_test_counts(
    x: List[Optional[float]], y: List[Optional[float]]
) -> Tuple[int, int, int]:
    # Return (n_pos, n_neg, n_tie) for paired x>y, x<y, x==y

    n_pos = n_neg = n_tie = 0
    for a, b in zip(x, y):
        if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
            continue
        if a > b:
            n_pos += 1
        elif a < b:
            n_neg += 1
        else:
            n_tie += 1
    return n_pos, n_neg, n_tie


def _median_penalty(
    full_fold: List[Optional[float]], abbr_fold: List[Optional[float]]
) -> Optional[float]:
    diffs = []
    for f, a in zip(full_fold, abbr_fold):
        if (
            f is not None
            and a is not None
            and f > 0
            and math.isfinite(f)
            and math.isfinite(a)
        ):
            diffs.append((f - a) / f)
    return float(np.median(diffs)) if diffs else None


def write_md_report(
    out_path: str,
    metric: str,
    runs: List[RunData],
    pairwise: Dict[str, Dict[str, float]],
    robust: List[dict],
    sens: List[dict],
    sens_topn: int = 25,
    baseline: str = "synonym",
):
    lines: List[str] = []
    lines.append(f"# Cross-run comparison ({metric})")
    lines.append(f"Baseline: **{baseline}**")
    lines.append("")

    for r in runs:
        a = r.aggregate()
        lines += [
            f"## Run: **{r.name}**",
            f"- FULL delta: mean **{a.get('full_delta_mean', float('nan')):.6f}**, median **{a.get('full_delta_median', float('nan')):.6f}**, n={int(a.get('n_full', 0))}",
            f"- ABBREV delta: mean **{a.get('abbr_delta_mean', float('nan')):.6f}**, median **{a.get('abbr_delta_median', float('nan')):.6f}**, n={int(a.get('n_abbr', 0))}",
            f"- FULL raw fold: mean **{a.get('full_fold_mean', float('nan')):.3f}x**, median **{a.get('full_fold_median', float('nan')):.3f}x**",
            f"- ABBREV raw fold: mean **{a.get('abbr_fold_mean', float('nan')):.3f}x**, median **{a.get('abbr_fold_median', float('nan')):.3f}x**",
            f"- % species with FULL delta > ABBREV delta: **{a.get('pct_full_gt_abbrev', float('nan')):.1f}%**",
            "",
        ]

    lines.append("## Paired FULL vs ABBREV (per species)")
    for r in runs:
        sp = sorted(set(r.full_raw_fold) & set(r.abbr_raw_fold))
        fvals = [r.full_raw_fold[s] for s in sp]
        avals = [r.abbr_raw_fold[s] for s in sp]
        n_pos, n_neg, n_tie = _sign_test_counts(fvals, avals)
        pen = _median_penalty(fvals, avals)
        lines.append(
            f"- **{r.name}**: FULL>ABBREV={n_pos}, FULL<ABBREV={n_neg}, ties={n_tie}"
            + (f"; median ABBREV penalty **{pen*100:.1f}%**" if pen is not None else "")
        )
    lines.append("")

    if pairwise:
        lines.append("## Rank correlations (Spearman) of per-species delta")
        for k, v in pairwise.items():
            lines.append(
                f"- {k}: FULL **{v['spearman_full_delta']:.3f}**, ABBREV **{v['spearman_abbrev_delta']:.3f}**"
            )
        lines.append("")

    if robust:
        lines.append(f"## Robust species across runs (n={len(robust)})")
        lines.append(
            "Criteria: FULL raw fold >= threshold in *all* runs, ABBREV <= FULL by min penalty, and FULL delta CI_lo > 0."
        )
        for r in robust[:50]:
            lines.append(
                f"- {r['species']}: "
                + ", ".join(f"{k}={v:.2f}x" for k, v in r.items() if k != "species")
            )
        if len(robust) > 50:
            lines.append(f"... and {len(robust)-50} more.")
        lines.append("")

    if sens:
        lines.append(
            f"## Most sensitive species to post-processing (top {min(sens_topn, len(sens))})"
        )
        for row in sens[:sens_topn]:
            lines.append(
                f"- {row['species']}: delta fold spread across runs = {row['spread']:.2f}x"
            )
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_per_species_console(
    species: List[str], runs: List[RunData], condition: str = "full"
):
    assert condition in ("full", "abbrev")
    hdr = f"{'Species':40s} " + "  ".join([f"{r.name:^36s}" for r in runs])
    print(hdr)
    if condition == "full":
        subhdr = " " * 41 + "  ".join(
            [f"{'baseline':>10s} {'delta':>10s} {'foldx':>10s}" for _ in runs]
        )
    else:
        # ABBREV fold uses FULL baseline by default
        subhdr = " " * 41 + "  ".join(
            [f"{'baseline(FULL)':>14s} {'delta':>10s} {'foldx':>10s}" for _ in runs]
        )
    print(subhdr)
    for s in species:
        cells = []
        for r in runs:
            if condition == "full":
                b = r.base_full.get(s)
                d = r.full_delta.get(s)
                f = r.full_raw_fold.get(s)
            else:
                b = r.base_full.get(
                    s
                )  # IMPORTANT: show FULL baseline to match fold definition
                d = r.abbr_delta.get(s)
                f = r.abbr_raw_fold.get(s)  # default = full-baseline version
            cells.append(
                f"{(b if b is not None else float('nan')):>10.4f} "
                f"{(d if d is not None else float('nan')):>10.4f} "
                f"{(f if f is not None else float('nan')):>10.2f}"
            )
        print(f"{s:40.40s} " + "  ".join(cells))


def build_parser():
    p = argparse.ArgumentParser(
        description="Compare multiple runs (none/abtt/whiten) per species and globally."
    )
    p.add_argument(
        "--workdir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "work"),
        help="Work directory root (default: ./work at repo root)",
    )
    p.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help=(
            "List of name=path entries. If omitted, uses none/abtt/whiten under <workdir>/summaries_from_cache"
        ),
    )
    p.add_argument(
        "--metric",
        choices=["one_minus_cos", "unit_L2"],
        default="one_minus_cos",
        help="Metric for baseline + delta comparisons.",
    )
    p.add_argument(
        "--baseline",
        choices=["synonym", "random"],
        default="synonym",
        help="Which control baseline to treat delta columns against.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output directory for combined CSVs and report (default: <workdir>/summaries_from_cache/compare_all*)",
    )
    p.add_argument(
        "--print-all",
        action="store_true",
        help="Print per-species rows for ALL species to stdout",
    )
    p.add_argument(
        "--topn",
        type=int,
        default=25,
        help="If not printing all, show top-N by FULL delta of the first run",
    )
    p.add_argument(
        "--condition",
        choices=["full", "abbrev"],
        default="full",
        help="Which condition to print",
    )

    # robust & sensitivity knobs
    p.add_argument(
        "--robust-fold",
        type=float,
        default=2.0,
        help="Raw fold threshold for robustness (across all runs)",
    )
    p.add_argument(
        "--penalty-min",
        type=float,
        default=0.10,
        help="Min ABBREV penalty fraction vs FULL (e.g., 0.10=10%)",
    )
    p.add_argument(
        "--sensitivity-topn",
        type=int,
        default=25,
        help="How many sensitive species to list in the report",
    )
    return p


def main():
    args = build_parser().parse_args()
    workdir = Path(args.workdir)

    if args.runs is None:
        base = workdir / "summaries_from_cache"
        args.runs = [
            f"none={base / 'none'}",
            f"abtt={base / 'abtt'}",
            f"white={base / 'whiten'}",
        ]

    if args.out is None:
        base_name = (
            "compare_all" if args.metric == "one_minus_cos" else "compare_all_L2"
        )
        suffix = "" if args.baseline == "synonym" else f"_{args.baseline}"
        args.out = str(workdir / "summaries_from_cache" / f"{base_name}{suffix}")

    run_specs: List[Tuple[str, str]] = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"Bad --runs entry: {spec} (use name=path)")
        name, path = spec.split("=", 1)
        if not os.path.isdir(path):
            raise SystemExit(f"Not a directory: {path}")
        run_specs.append((name, path))

    os.makedirs(args.out, exist_ok=True)

    runs: List[RunData] = [
        RunData(name, path, metric=args.metric, baseline=args.baseline)
        for (name, path) in run_specs
    ]
    species = sorted(set().union(*[set(r.full_delta.keys()) for r in runs]))

    # per-species CSV (all rows)
    per_species_csv = os.path.join(
        args.out, f"per_species_comparison_{args.metric}.csv"
    )
    write_per_species_csv(per_species_csv, species, runs)

    # global CSV
    global_csv = os.path.join(args.out, f"global_summary_{args.metric}.csv")
    write_global_csv(global_csv, runs)

    # pairwise rank correlations
    pairwise = compare_runs(runs)

    # robust & sensitivity tables
    robust = robust_species(
        runs, fold_thresh=args.robust_fold, penalty_min=args.penalty_min
    )
    sens = sensitivity_table(runs)

    # markdown report
    md = os.path.join(args.out, f"compare_report_{args.metric}.md")
    write_md_report(
        md,
        args.metric,
        runs,
        pairwise,
        robust,
        sens,
        sens_topn=args.sensitivity_topn,
        baseline=args.baseline,
    )

    # console print (all or topN by FULL delta of first run)
    if args.print_all:
        to_show = species
    else:
        first = runs[0]
        pairs = [(s, first.full_delta.get(s)) for s in species]
        pairs = [(s, v) for (s, v) in pairs if v is not None and math.isfinite(v)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        to_show = [s for (s, _) in pairs[: args.topn]]
    print_per_species_console(to_show, runs, condition=args.condition)

    print(f"\n[WROTE] {per_species_csv}")
    print(f"[WROTE] {global_csv}")
    print(f"[WROTE] {md}")

    if robust:
        robust_csv = os.path.join(args.out, f"robust_species_{args.metric}.csv")
        with open(robust_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            cols = ["species"] + [k for k in robust[0].keys() if k != "species"]
            w.writerow(cols)
            for r in robust:
                w.writerow([r.get(c) for c in ["species"] + cols[1:]])
        print(f"[WROTE] {robust_csv}")


if __name__ == "__main__":
    main()
