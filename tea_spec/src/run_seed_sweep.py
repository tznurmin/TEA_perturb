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
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from io_utils import slugify_model_id


def _parse_seeds(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_methods(s: str) -> List[str]:
    return [part.strip() for part in s.split(",") if part.strip()]


def _compare_run_entries(out_summaries: Path, methods: List[str]) -> List[str]:
    entries: List[str] = []
    for method in methods:
        label = "white" if method == "whiten" else method
        entries.append(f"{label}={out_summaries / method}")
    return entries


def _selected_baselines(compare_baselines: str, no_rand: bool) -> List[str]:
    if no_rand:
        if compare_baselines == "random":
            raise SystemExit("--compare-baselines random is incompatible with --no-rand")
        return ["synonym"]

    if compare_baselines == "both":
        return ["synonym", "random"]

    return [compare_baselines]


def _limited_env() -> dict:
    # Constrain BLAS thread counts
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


def _run(cmd: List[str], env: dict):
    print("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def build_parser():
    repo_root = Path(__file__).resolve().parents[2]
    default_workdir = repo_root / "work"

    p = argparse.ArgumentParser(
        description="Seed sweep: run multiple seeds end-to-end (embed -> summarise -> compare)."
    )
    p.add_argument(
        "--workdir",
        type=str,
        default=str(default_workdir),
        help="Work directory root (default: ./work at repo root)",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seeds (default: 42,43,44)",
    )

    # embed_cache arguments (minimal surface)
    p.add_argument(
        "--dataset",
        default=None,
        help="Template dataset JSON (default: <workdir>/templates/cluster_all_examples_flat_42.json)",
    )
    p.add_argument(
        "--species",
        default=None,
        help="Species list (default: <workdir>/species/species_in_corpus.txt if it exists, else all_species.txt)",
    )
    p.add_argument(
        "--placeholder",
        default="Staphylococcus aureus",
        help="Placeholder species string that appears in the template dataset",
    )
    p.add_argument(
        "--model",
        default="dmis-lab/biobert-base-cased-v1.2",
        help="HF model name to embed with",
    )
    p.add_argument("--pooling", default="mean_last2", help="Pooling strategy")
    p.add_argument("--batch-size", type=int, default=3000)
    p.add_argument("--max-species", type=int, default=5000)
    p.add_argument("--sample-species", type=int, default=0)
    p.add_argument("--sample-species-seed", type=int, default=0)

    # random baseline toggle
    p.add_argument(
        "--no-rand", action="store_true", help="Disable wordfreq random-word baseline"
    )

    # summarize_from_cache args
    p.add_argument("--methods", default="none,abtt,whiten", help="Comma-separated list")
    p.add_argument("--abtt-k", type=int, default=10)
    p.add_argument("--whiten-eps", type=float, default=1e-6)
    p.add_argument("--pp-max-fit-rows", type=int, default=200_000)

    # comparison output
    p.add_argument(
        "--compare-baselines",
        choices=["synonym", "random", "both"],
        default="both",
        help="Which baselines to generate compare reports for",
    )
    return p


def main():
    args = build_parser().parse_args()
    workdir = Path(args.workdir)
    py = sys.executable

    model_slug = slugify_model_id(args.model)

    seeds = _parse_seeds(args.seeds)
    methods = _parse_methods(args.methods)

    dataset = args.dataset or str(
        workdir / "templates" / "cluster_all_examples_flat_42.json"
    )

    if args.species is None:
        cand = workdir / "species" / "species_in_corpus.txt"
        if cand.exists():
            species_path = str(cand)
        else:
            species_path = str(workdir / "species" / "all_species.txt")
    else:
        species_path = args.species

    env = _limited_env()

    for seed in seeds:
        out_npz = (
            workdir
            / "cache"
            / "models"
            / model_slug
            / f"species_embeddings_seed{seed}.npz"
        )
        out_summaries = (
            workdir
            / "summaries_from_cache"
            / "models"
            / model_slug
            / f"seed{seed}"
        )

        out_npz.parent.mkdir(parents=True, exist_ok=True)
        out_summaries.mkdir(parents=True, exist_ok=True)

        # 1: cache embeddings
        cmd_embed = [
            py,
            str(Path(__file__).with_name("embed_cache.py")),
            "--workdir",
            str(workdir),
            "--dataset",
            dataset,
            "--species",
            species_path,
            "--out-npz",
            str(out_npz),
            "--placeholder",
            args.placeholder,
            "--model",
            args.model,
            "--pooling",
            args.pooling,
            "--batch-size",
            str(int(args.batch_size)),
            "--max-species",
            str(int(args.max_species)),
            "--sample-species",
            str(int(args.sample_species)),
            "--sample-species-seed",
            str(int(args.sample_species_seed)),
            "--syn-seed",
            str(int(seed)),
            "--syn-write-manifest",
            "--rand-seed",
            str(int(seed)),
            "--rand-write-manifest",
        ]
        if args.no_rand:
            cmd_embed.append("--no-rand-enabled")

        _run(cmd_embed, env=env)

        # 2: summarize
        cmd_sum = [
            py,
            str(Path(__file__).with_name("summarize_from_cache.py")),
            "--workdir",
            str(workdir),
            "--cache",
            str(out_npz),
            "--out",
            str(out_summaries),
            "--methods",
            args.methods,
            "--abtt-k",
            str(int(args.abtt_k)),
            "--whiten-eps",
            str(float(args.whiten_eps)),
            "--pp-max-fit-rows",
            str(int(args.pp_max_fit_rows)),
            "--pp-seed",
            str(int(seed)),
        ]
        _run(cmd_sum, env=env)

        # 3: compare within this seed
        runs = _compare_run_entries(out_summaries, methods)

        baselines = _selected_baselines(args.compare_baselines, args.no_rand)

        for baseline in baselines:
            cmd_cmp = [
                py,
                str(Path(__file__).with_name("compare_runs.py")),
                "--workdir",
                str(workdir),
                "--runs",
                *runs,
                "--metric",
                "one_minus_cos",
                "--baseline",
                baseline,
                "--out",
                str(out_summaries / f"compare_all_{baseline}"),
                "--robust-fold",
                "2.0",
                "--penalty-min",
                "0.10",
            ]
            _run(cmd_cmp, env=env)


if __name__ == "__main__":
    main()
