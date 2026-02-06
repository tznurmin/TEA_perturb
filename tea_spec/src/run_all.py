from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import List


def _parse_seeds(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _run(cmd: List[str], env: dict) -> None:
    print("[RUN] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _limited_env() -> dict:
    # constrain BLAS thread counts to safely run this
    env = dict(os.environ)
    for k in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        env[k] = "1"
    return env


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    default_workdir = repo_root / "work"

    p = argparse.ArgumentParser(
        description="One-command pipeline: build inputs -> run 3 seeds -> write stability report."
    )
    p.add_argument("--workdir", type=str, default=str(default_workdir))
    p.add_argument("--seeds", type=str, default="42,43,44")

    # build steps
    p.add_argument(
        "--speclist",
        type=str,
        default=str(repo_root / "tea_spec" / "data" / "speclist.txt"),
        help="UniProt speclist.txt (default: tea_spec/data/speclist.txt)",
    )
    p.add_argument("--anchor", type=str, default="Staphylococcus aureus")
    p.add_argument("--regular-seed", type=int, default=42)

    # embedding and caching
    p.add_argument("--model", type=str, default="dmis-lab/biobert-base-cased-v1.2")
    p.add_argument("--pooling", type=str, default="mean_last2")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-species", type=int, default=5000)
    p.add_argument("--pp-max-fit-rows", type=int, default=200_000)
    p.add_argument(
        "--no-rand", action="store_true", help="Disable wordfreq random baseline"
    )

    # workspace bootstrap
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download TEA_curated_data; fail if curated material is missing.",
    )
    p.add_argument(
        "--tea-url",
        type=str,
        default=None,
        help="Override TEA_curated_data tarball URL used by setup_workdir.py",
    )

    return p


def main() -> None:
    a = build_parser().parse_args()
    workdir = Path(a.workdir)
    env = _limited_env()

    # bootstrap workdir and curated data: downloads TEA_curated_data v1.1 if missing
    cmd_setup = [
        "python",
        str(Path(__file__).with_name("setup_workdir.py")),
        "--workdir",
        str(workdir),
    ]
    if a.skip_download:
        cmd_setup.append("--skip-download")
    if a.tea_url:
        cmd_setup.extend(["--tea-url", str(a.tea_url)])
    _run(cmd_setup, env=env)

    curated_root = workdir / "TEA_curated_data"

    species_all = workdir / "species" / "all_species.txt"
    regular = workdir / f"regular_{a.regular_seed}.json"
    templates_flat = (
        workdir / "templates" / f"cluster_all_examples_flat_{a.regular_seed}.json"
    )
    templates_grouped = (
        workdir / "templates" / f"cluster_all_examples_{a.regular_seed}.json"
    )
    species_in_corpus = workdir / "species" / "species_in_corpus.txt"

    # 1: species list
    if not species_all.exists():
        _run(
            [
                "python",
                str(Path(__file__).with_name("build_species_list.py")),
                "--speclist",
                str(Path(a.speclist)),
                "--out",
                str(species_all),
            ],
            env=env,
        )

    # 2: regular dataset
    if not regular.exists():
        _run(
            [
                "python",
                str(Path(__file__).with_name("build_regular_dataset.py")),
                "--curated-root",
                str(curated_root),
                "--seed",
                str(int(a.regular_seed)),
                "--out",
                str(regular),
            ],
            env=env,
        )

    # 3: templates
    if not templates_flat.exists():
        _run(
            [
                "python",
                str(Path(__file__).with_name("generate_templates_from_curated.py")),
                "--workdir",
                str(workdir),
                "--curated",
                str(regular),
                "--species",
                str(species_all),
                "--anchor",
                str(a.anchor),
                "--out-flat",
                str(templates_flat),
                "--out-grouped",
                str(templates_grouped),
            ],
            env=env,
        )

    # 4: species-in-corpus
    if not species_in_corpus.exists():
        _run(
            [
                "python",
                str(Path(__file__).with_name("extract_species_in_corpus.py")),
                "--regular",
                str(regular),
                "--all-species",
                str(species_all),
                "--out",
                str(species_in_corpus),
            ],
            env=env,
        )

    # 5: seed sweep (embeddings, summaries, comparisons)
    seeds = ",".join(str(x) for x in _parse_seeds(a.seeds))
    cmd_seed_sweep = [
        "python",
        str(Path(__file__).with_name("run_seed_sweep.py")),
        "--workdir",
        str(workdir),
        "--dataset",
        str(templates_flat),
        "--species",
        str(species_in_corpus),
        "--seeds",
        seeds,
        "--methods",
        "none,abtt,whiten",
        "--compare-baselines",
        "both" if not a.no_rand else "synonym",
        "--batch-size",
        str(int(a.batch_size)),
        "--pooling",
        str(a.pooling),
        "--max-species",
        str(int(a.max_species)),
        "--pp-max-fit-rows",
        str(int(a.pp_max_fit_rows)),
    ]
    if a.no_rand:
        cmd_seed_sweep.append("--no-rand")

    _run(cmd_seed_sweep, env=env)

    # 6: seed stability report
    cmd_stability = [
        "python",
        str(Path(__file__).with_name("report_seed_stability.py")),
        "--workdir",
        str(workdir),
        "--seeds",
        seeds,
        "--methods",
        "none,abtt,whiten",
        "--baseline",
        "synonym" if a.no_rand else "both",
        "--metric",
        "one_minus_cos",
    ]
    _run(cmd_stability, env=env)


if __name__ == "__main__":
    main()
