# Report seed stability (signal vs noise)

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


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


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _std(xs: Iterable[float]) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    return float(np.std(np.asarray(xs, dtype=float), ddof=1))


@dataclass
class SeedMethodDeltas:
    seed: int
    method: str
    baseline: str
    metric: str
    full_delta: Dict[str, Optional[float]]
    abbr_delta: Dict[str, Optional[float]]


def _baseline_conditions(baseline: str) -> Tuple[str, str]:
    if baseline == "synonym":
        return "species_full_delta_vs_syn", "species_abbrev_delta_vs_syn"
    if baseline == "random":
        return "species_full_delta_vs_rand", "species_abbrev_delta_vs_rand"
    raise ValueError(f"Unknown baseline: {baseline}")


def load_deltas(
    summary_csv: Path, baseline: str, metric: str
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    # parse summarize_from_cache minimal_summary.csv into per-species delta dicts

    want_full_cond, want_abbr_cond = _baseline_conditions(baseline)

    rows = _read_csv(summary_csv)
    by_key: Dict[Tuple[str, str, str], dict] = {}
    for r in rows:
        anchor = r.get("anchor")
        condition = r.get("condition")
        stat = r.get("stat")
        if anchor and condition and stat:
            by_key[(anchor, condition, stat)] = r

    full: Dict[str, Optional[float]] = {}
    abbr: Dict[str, Optional[float]] = {}

    anchors = sorted({r.get("anchor") for r in rows if r.get("anchor")})
    for a in anchors:
        r_full = by_key.get((a, want_full_cond, metric))
        r_abbr = by_key.get((a, want_abbr_cond, metric))
        full[a] = _to_float(r_full.get("mean")) if r_full else None
        abbr[a] = _to_float(r_abbr.get("mean")) if r_abbr else None

    return full, abbr


def collect_seed_method_deltas(
    workdir: Path,
    seeds: List[int],
    methods: List[str],
    baseline: str,
    metric: str,
) -> List[SeedMethodDeltas]:
    out: List[SeedMethodDeltas] = []
    for seed in seeds:
        for method in methods:
            p = (
                workdir
                / "summaries_from_cache"
                / f"seed{seed}"
                / method
                / "minimal_summary.csv"
            )
            if not p.exists():
                raise FileNotFoundError(str(p))
            full, abbr = load_deltas(p, baseline=baseline, metric=metric)
            out.append(
                SeedMethodDeltas(
                    seed=seed,
                    method=method,
                    baseline=baseline,
                    metric=metric,
                    full_delta=full,
                    abbr_delta=abbr,
                )
            )
    return out


@dataclass
class StabilityRow:
    baseline: str
    method: str
    metric: str
    n_species: int
    full_delta_mean_mean: float
    full_delta_mean_std: float
    full_delta_mean_cv: float
    abbr_delta_mean_mean: float
    abbr_delta_mean_std: float
    abbr_delta_mean_cv: float
    full_spearman_mean: float
    abbr_spearman_mean: float
    full_noise_mean: float
    abbr_noise_mean: float
    full_signal_sd: float
    abbr_signal_sd: float
    full_snr: float
    abbr_snr: float


def _pairwise_seed_pairs(seeds: List[int]) -> List[Tuple[int, int]]:
    pairs = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            pairs.append((seeds[i], seeds[j]))
    return pairs


def compute_stability(
    deltas: List[SeedMethodDeltas],
    seeds: List[int],
    methods: List[str],
    baseline: str,
    metric: str,
) -> List[StabilityRow]:
    by_sm: Dict[Tuple[int, str], SeedMethodDeltas] = {
        (d.seed, d.method): d for d in deltas
    }
    pairs = _pairwise_seed_pairs(seeds)

    rows: List[StabilityRow] = []

    for method in methods:
        # common species across all seeds for this method
        species_sets = []
        for seed in seeds:
            d = by_sm[(seed, method)]
            s_ok = {
                s for s, v in d.full_delta.items() if v is not None and math.isfinite(v)
            }
            species_sets.append(s_ok)
        common = sorted(set.intersection(*species_sets)) if species_sets else []
        if not common:
            continue

        # per-seed global means
        full_means = []
        abbr_means = []
        for seed in seeds:
            d = by_sm[(seed, method)]
            full_vals = [d.full_delta[s] for s in common]
            abbr_vals = [d.abbr_delta.get(s) for s in common]
            full_vals_f = [
                float(x) for x in full_vals if x is not None and math.isfinite(float(x))
            ]
            abbr_vals_f = [
                float(x) for x in abbr_vals if x is not None and math.isfinite(float(x))
            ]
            full_means.append(_mean(full_vals_f))
            abbr_means.append(_mean(abbr_vals_f))

        # pairwise correlations
        full_corrs = []
        abbr_corrs = []
        for a, b in pairs:
            da = by_sm[(a, method)]
            db = by_sm[(b, method)]
            xa = [da.full_delta[s] for s in common]
            xb = [db.full_delta[s] for s in common]
            ya = [da.abbr_delta.get(s) for s in common]
            yb = [db.abbr_delta.get(s) for s in common]
            full_corrs.append(_spearman(xa, xb))
            abbr_corrs.append(_spearman(ya, yb))

        # per-species across-seed noise
        full_noise = []
        abbr_noise = []
        full_means_by_species = []
        abbr_means_by_species = []
        for s in common:
            vs_full = [float(by_sm[(seed, method)].full_delta[s]) for seed in seeds]
            vs_abbr = [
                float(by_sm[(seed, method)].abbr_delta[s])
                for seed in seeds
                if by_sm[(seed, method)].abbr_delta.get(s) is not None
            ]

            full_noise.append(_std(vs_full))
            full_means_by_species.append(_mean(vs_full))

            if vs_abbr:
                abbr_noise.append(_std(vs_abbr))
                abbr_means_by_species.append(_mean(vs_abbr))

        # signal = across-species spread of the across-seed mean per species
        full_signal_sd = _std(full_means_by_species)
        abbr_signal_sd = _std(abbr_means_by_species)

        full_noise_mean = _mean(full_noise)
        abbr_noise_mean = _mean(abbr_noise)

        # signal-to-noise ratio (higher = less seed noise)
        full_snr = (
            (full_signal_sd / full_noise_mean)
            if full_noise_mean and full_noise_mean > 0
            else float("inf")
        )
        abbr_snr = (
            (abbr_signal_sd / abbr_noise_mean)
            if abbr_noise_mean and abbr_noise_mean > 0
            else float("inf")
        )

        # global mean mean/std/cv
        full_mean_mean = _mean(full_means)
        full_mean_std = _std(full_means)
        full_mean_cv = (
            (full_mean_std / abs(full_mean_mean))
            if full_mean_mean and abs(full_mean_mean) > 0
            else 0.0
        )

        abbr_mean_mean = _mean(abbr_means)
        abbr_mean_std = _std(abbr_means)
        abbr_mean_cv = (
            (abbr_mean_std / abs(abbr_mean_mean))
            if abbr_mean_mean and abs(abbr_mean_mean) > 0
            else 0.0
        )

        rows.append(
            StabilityRow(
                baseline=baseline,
                method=method,
                metric=metric,
                n_species=len(common),
                full_delta_mean_mean=full_mean_mean,
                full_delta_mean_std=full_mean_std,
                full_delta_mean_cv=full_mean_cv,
                abbr_delta_mean_mean=abbr_mean_mean,
                abbr_delta_mean_std=abbr_mean_std,
                abbr_delta_mean_cv=abbr_mean_cv,
                full_spearman_mean=_mean(full_corrs),
                abbr_spearman_mean=_mean(abbr_corrs),
                full_noise_mean=full_noise_mean,
                abbr_noise_mean=abbr_noise_mean,
                full_signal_sd=full_signal_sd,
                abbr_signal_sd=abbr_signal_sd,
                full_snr=full_snr,
                abbr_snr=abbr_snr,
            )
        )

    return rows


def write_csv(path: Path, rows: List[StabilityRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "baseline",
                "method",
                "metric",
                "n_species",
                "full_delta_mean_mean",
                "full_delta_mean_std",
                "full_delta_mean_cv",
                "abbr_delta_mean_mean",
                "abbr_delta_mean_std",
                "abbr_delta_mean_cv",
                "full_spearman_mean",
                "abbr_spearman_mean",
                "full_noise_mean",
                "abbr_noise_mean",
                "full_signal_sd",
                "abbr_signal_sd",
                "full_snr",
                "abbr_snr",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.baseline,
                    r.method,
                    r.metric,
                    r.n_species,
                    f"{r.full_delta_mean_mean:.12g}",
                    f"{r.full_delta_mean_std:.12g}",
                    f"{r.full_delta_mean_cv:.12g}",
                    f"{r.abbr_delta_mean_mean:.12g}",
                    f"{r.abbr_delta_mean_std:.12g}",
                    f"{r.abbr_delta_mean_cv:.12g}",
                    f"{r.full_spearman_mean:.12g}",
                    f"{r.abbr_spearman_mean:.12g}",
                    f"{r.full_noise_mean:.12g}",
                    f"{r.abbr_noise_mean:.12g}",
                    f"{r.full_signal_sd:.12g}",
                    f"{r.abbr_signal_sd:.12g}",
                    f"{r.full_snr:.12g}",
                    f"{r.abbr_snr:.12g}",
                ]
            )


def write_md(path: Path, seeds: List[int], rows: List[StabilityRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(x: float) -> str:
        if x is None or not math.isfinite(x):
            return "nan"
        return f"{x:.6g}"

    lines = []
    lines.append("# Seed stability report\n")
    lines.append(
        "This report quantifies whether results are stable across random seeds.\n"
    )
    lines.append(f"Seeds: {', '.join(map(str, seeds))}\n")

    for baseline in sorted({r.baseline for r in rows}):
        lines.append(f"## Baseline: {baseline}\n")
        lines.append("Columns:\n")
        lines.append(
            "- mean(delta): average per-species delta above baseline (higher = stronger species signal)\n"
        )
        lines.append(
            "- Spearman: rank correlation across seeds (1.0 = identical ranking)\n"
        )
        lines.append(
            "- noise: mean per-species std across seeds (lower = less seed noise)\n"
        )
        lines.append(
            "- signal_sd: std across species of the across-seed mean deltas (higher = more between-species structure)\n"
        )
        lines.append(
            "- SNR: signal_sd / noise (higher = clearer signal vs seed noise)\n"
        )

        lines.append(
            "| method | n | mean(delta_full) | sd_seed(meandelta_full) | CV | Spearman_full | noise_full | signal_sd_full | SNR_full | mean(delta_abbr) | sd_seed(meandelta_abbr) | CV | Spearman_abbr | noise_abbr | signal_sd_abbr | SNR_abbr |\n"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for r in [x for x in rows if x.baseline == baseline]:
            lines.append(
                "| {method} | {n} | {mf} | {sdf} | {cvf} | {sp} | {nf} | {sf} | {snrf} | {ma} | {sda} | {cva} | {spa} | {na} | {sa} | {snra} |\n".format(
                    method=r.method,
                    n=r.n_species,
                    mf=fmt(r.full_delta_mean_mean),
                    sdf=fmt(r.full_delta_mean_std),
                    cvf=fmt(r.full_delta_mean_cv),
                    sp=fmt(r.full_spearman_mean),
                    nf=fmt(r.full_noise_mean),
                    sf=fmt(r.full_signal_sd),
                    snrf=fmt(r.full_snr),
                    ma=fmt(r.abbr_delta_mean_mean),
                    sda=fmt(r.abbr_delta_mean_std),
                    cva=fmt(r.abbr_delta_mean_cv),
                    spa=fmt(r.abbr_spearman_mean),
                    na=fmt(r.abbr_noise_mean),
                    sa=fmt(r.abbr_signal_sd),
                    snra=fmt(r.abbr_snr),
                )
            )
        lines.append("\n")

    path.write_text("".join(lines), encoding="utf-8")


def print_console(rows: List[StabilityRow]) -> None:
    # compact view similar to other scripts
    hdr = (
        "baseline,method,n_species,full_meandelta,sd_seed(full_meandelta),spearman_full,full_noise_mean,full_SNR,"
        "abbr_meandelta,sd_seed(abbr_meandelta),spearman_abbr,abbr_noise_mean,abbr_SNR"
    )
    print(hdr)
    for r in rows:
        print(
            f"{r.baseline},{r.method},{r.n_species},"
            f"{r.full_delta_mean_mean:.12g},{r.full_delta_mean_std:.12g},{r.full_spearman_mean:.6g},{r.full_noise_mean:.6g},{r.full_snr:.6g},"
            f"{r.abbr_delta_mean_mean:.12g},{r.abbr_delta_mean_std:.12g},{r.abbr_spearman_mean:.6g},{r.abbr_noise_mean:.6g},{r.abbr_snr:.6g}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Quantify seed stability (signal vs noise) across TEA runs"
    )
    ap.add_argument(
        "--workdir", type=str, default="work", help="Work directory (default: work)"
    )
    ap.add_argument(
        "--seeds", type=str, required=True, help="Comma-separated seeds, e.g. 42,43,44"
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="none,abtt,whiten",
        help="Comma-separated methods (default: none,abtt,whiten)",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="one_minus_cos",
        help="Metric name used in summaries (default: one_minus_cos)",
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default="both",
        choices=["synonym", "random", "both"],
        help="Which baseline deltas to analyze",
    )
    ap.add_argument("--out-csv", type=str, default="", help="Optional CSV output path")
    ap.add_argument(
        "--out-md", type=str, default="", help="Optional Markdown output path"
    )

    a = ap.parse_args()

    workdir = Path(a.workdir)
    seeds = [int(x.strip()) for x in a.seeds.split(",") if x.strip()]
    methods = [x.strip() for x in a.methods.split(",") if x.strip()]

    baselines = ["synonym", "random"] if a.baseline == "both" else [a.baseline]

    all_rows: List[StabilityRow] = []

    for baseline in baselines:
        deltas = collect_seed_method_deltas(
            workdir=workdir,
            seeds=seeds,
            methods=methods,
            baseline=baseline,
            metric=a.metric,
        )
        rows = compute_stability(
            deltas=deltas,
            seeds=seeds,
            methods=methods,
            baseline=baseline,
            metric=a.metric,
        )
        all_rows.extend(rows)

        # write per-baseline outputs unless user specified custom names
        if a.out_csv:
            out_csv = Path(a.out_csv)
        else:
            out_csv = (
                workdir
                / "summaries_from_cache"
                / f"seed_stability_{baseline}_{a.metric}.csv"
            )
        if a.out_md:
            out_md = Path(a.out_md)
        else:
            out_md = (
                workdir
                / "summaries_from_cache"
                / f"seed_stability_{baseline}_{a.metric}.md"
            )

        write_csv(out_csv, rows)
        write_md(out_md, seeds, rows)

    print_console(all_rows)


if __name__ == "__main__":
    main()
