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

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import compare_runs  # noqa: E402
import summarize_from_cache  # noqa: E402


def _write_fake_cache(npz_path: Path) -> None:
    rng = np.random.default_rng(0)

    species = ["Staphylococcus aureus", "Bacillus subtilis"]
    S, N, D = 2, 4, 8

    z_full = rng.normal(size=(S, N, D)).astype(np.float32)
    z_abbr = z_full + 0.01 * rng.normal(size=(S, N, D)).astype(np.float32)

    # synonym baseline: small perturbation
    z_syn_full = z_full + 0.005 * rng.normal(size=(S, N, D)).astype(np.float32)
    z_syn_abbr = z_abbr + 0.005 * rng.normal(size=(S, N, D)).astype(np.float32)

    # random baseline: larger perturbation
    z_rand_full = z_full + 0.02 * rng.normal(size=(S, N, D)).astype(np.float32)
    z_rand_abbr = z_abbr + 0.02 * rng.normal(size=(S, N, D)).astype(np.float32)

    syn_ok_full = np.ones((S, N), dtype=bool)
    syn_ok_abbr = np.ones((S, N), dtype=bool)
    rand_ok_full = np.ones((S, N), dtype=bool)
    rand_ok_abbr = np.ones((S, N), dtype=bool)

    meta = {
        "species": species,
        "placeholder": "Staphylococcus aureus",
        "n_sentences": N,
        "pooling": "mean_last2",
        "model_name": "FAKE",
        "synonym_swap": {"seed": 0},
        "random_swap": {"enabled": True, "seed": 0},
    }

    np.savez_compressed(
        npz_path,
        z_full=z_full,
        z_abbr=z_abbr,
        z_syn_full=z_syn_full,
        z_syn_abbr=z_syn_abbr,
        syn_ok_full=syn_ok_full,
        syn_ok_abbr=syn_ok_abbr,
        z_rand_full=z_rand_full,
        z_rand_abbr=z_rand_abbr,
        rand_ok_full=rand_ok_full,
        rand_ok_abbr=rand_ok_abbr,
        meta=json.dumps(meta),
    )


def test_random_baseline_rows_are_summarized_and_comparable(tmp_path: Path) -> None:
    npz = tmp_path / "fake_cache.npz"
    _write_fake_cache(npz)

    out_root = tmp_path / "summaries"

    summarize_from_cache.run_from_cache(
        npz_path=str(npz),
        out_root=str(out_root),
        methods=["none"],
        abtt_k=10,
        whiten_eps=1e-6,
        max_fit_rows=1000,
        seed=0,
        bootstrap=20,
        ci=0.95,
    )

    run_dir = out_root / "none"
    csv_path = run_dir / "minimal_summary.csv"
    assert csv_path.exists()

    # ensure random baseline rows exist
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    conds = {r["condition"] for r in rows}
    assert "random_full" in conds
    assert "random_abbrev" in conds
    assert "species_full_delta_vs_rand" in conds
    assert "species_abbrev_delta_vs_rand" in conds

    # ensure synonym baseline rows still exist
    assert "synonym_full" in conds
    assert "synonym_abbrev" in conds
    assert "species_full_delta_vs_syn" in conds
    assert "species_abbrev_delta_vs_syn" in conds

    # compare-runs parser must accept baseline=random
    r = compare_runs.RunData(
        "none", str(run_dir), metric="one_minus_cos", baseline="random"
    )
    agg = r.aggregate()
    assert "n_full" in agg and agg["n_full"] > 0


def test_compare_runs_rejects_missing_requested_baseline(tmp_path: Path) -> None:
    run_dir = tmp_path / "none"
    run_dir.mkdir()
    (run_dir / "minimal_summary.csv").write_text(
        "\n".join(
            [
                "condition,stat,anchor,mean,ci_lo,ci_hi",
                "synonym_full,one_minus_cos,Staphylococcus aureus,0.1,0.09,0.11",
                "species_full_delta_vs_syn,one_minus_cos,Staphylococcus aureus,0.2,0.19,0.21",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run = compare_runs.RunData(
        "none", str(run_dir), metric="one_minus_cos", baseline="random"
    )

    with pytest.raises(SystemExit, match="No usable 'random' baseline"):
        compare_runs._validate_runs_have_baseline([run], "random")
