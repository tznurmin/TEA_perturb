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
import json
import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from metrics import pairwise_compare
from postprocess import fit as pp_fit
from postprocess import transform as pp_transform
from utils import summarize


def _torch(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32, copy=False))


def _as_np(x) -> np.ndarray:
    # accept torch.Tensor or np.ndarray and return a numpy array
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_cache(npz_path: str):
    data = np.load(npz_path)
    meta = json.loads(str(data["meta"]))

    syn_ok_full = data["syn_ok_full"] if "syn_ok_full" in data.files else None
    syn_ok_abbr = data["syn_ok_abbr"] if "syn_ok_abbr" in data.files else None

    z_rand_full = data["z_rand_full"] if "z_rand_full" in data.files else None
    z_rand_abbr = data["z_rand_abbr"] if "z_rand_abbr" in data.files else None
    rand_ok_full = data["rand_ok_full"] if "rand_ok_full" in data.files else None
    rand_ok_abbr = data["rand_ok_abbr"] if "rand_ok_abbr" in data.files else None

    return (
        data["z_full"],
        data["z_abbr"],
        data["z_syn_full"],
        data["z_syn_abbr"],
        syn_ok_full,
        syn_ok_abbr,
        z_rand_full,
        z_rand_abbr,
        rand_ok_full,
        rand_ok_abbr,
        meta,
    )


def _baseline_arrays(
    a: np.ndarray,
    b: np.ndarray,
    ok_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Return (baseline_values, baseline_values_l2, baseline_fullarr, baseline_l2_fullarr)
    # - baseline_values are masked (only ok entries)
    # - baseline_fullarr keeps shape (N,) with NaN where invalid

    cos, l2 = pairwise_compare(_torch(a), _torch(b))
    omc = 1.0 - _as_np(cos)
    l2n = _as_np(l2)

    if ok_mask is None:
        ok = np.ones_like(omc, dtype=bool)
    else:
        ok = ok_mask.astype(bool)

    omc_full = omc.astype(np.float64, copy=True)
    l2_full = l2n.astype(np.float64, copy=True)
    omc_full[~ok] = np.nan
    l2_full[~ok] = np.nan
    return omc[ok], l2n[ok], omc_full, l2_full


def _init_delta_accumulators(S: int):
    return {
        i: {
            "full_1mcos_delta": [],
            "full_L2_delta": [],
            "abbr_1mcos_delta": [],
            "abbr_L2_delta": [],
        }
        for i in range(S)
    }


_CSV_HEADER = [
    "anchor",
    "n_sentences",
    "n_targets",
    "condition",
    "stat",
    "mean",
    "ci_lo",
    "ci_hi",
    "n",
    "reanchored_from",
]


def _cat(parts: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts, axis=0) if parts else np.array([])


def _csv_row(
    anchor: str, n_sentences: int, n_targets: int, cond: str, stat_name: str, stat_obj: dict
) -> List[Any]:
    return [
        anchor,
        int(n_sentences),
        int(n_targets),
        cond,
        stat_name,
        stat_obj["mean"],
        stat_obj["ci95"][0],
        stat_obj["ci95"][1],
        stat_obj["n"],
        "",
    ]


def _summary_filename(species_name: str) -> str:
    return f"summary_{species_name.lower().replace(' ', '_')}.json"


def _species_summary_from_stats(
    anchor: str,
    n_sentences: int,
    n_targets: int,
    stats: dict,
    have_rand: bool,
) -> Tuple[List[List[Any]], dict]:
    rows = [
        _csv_row(
            anchor,
            n_sentences,
            n_targets,
            "synonym_full",
            "one_minus_cos",
            stats["syn_full"]["one_minus_cos"],
        ),
        _csv_row(
            anchor,
            n_sentences,
            n_targets,
            "synonym_full",
            "unit_L2",
            stats["syn_full"]["unit_L2"],
        ),
        _csv_row(
            anchor,
            n_sentences,
            n_targets,
            "synonym_abbrev",
            "one_minus_cos",
            stats["syn_abbrev"]["one_minus_cos"],
        ),
        _csv_row(
            anchor,
            n_sentences,
            n_targets,
            "synonym_abbrev",
            "unit_L2",
            stats["syn_abbrev"]["unit_L2"],
        ),
    ]

    if have_rand:
        rows += [
            _csv_row(
                anchor,
                n_sentences,
                n_targets,
                "random_full",
                "one_minus_cos",
                stats["rand_full"]["one_minus_cos"],
            ),
            _csv_row(
                anchor,
                n_sentences,
                n_targets,
                "random_full",
                "unit_L2",
                stats["rand_full"]["unit_L2"],
            ),
            _csv_row(
                anchor,
                n_sentences,
                n_targets,
                "random_abbrev",
                "one_minus_cos",
                stats["rand_abbrev"]["one_minus_cos"],
            ),
            _csv_row(
                anchor,
                n_sentences,
                n_targets,
                "random_abbrev",
                "unit_L2",
                stats["rand_abbrev"]["unit_L2"],
            ),
        ]

    rows += [
        _csv_row(
            anchor,
            n_sentences,
            n_targets,
            "species_full_delta_vs_syn",
            "one_minus_cos",
            stats["full_1_syn"],
        ),
        _csv_row(
            anchor,
            n_sentences,
            n_targets,
            "species_full_delta_vs_syn",
            "unit_L2",
            stats["full_2_syn"],
        ),
        _csv_row(
            anchor,
            n_sentences,
            n_targets,
            "species_abbrev_delta_vs_syn",
            "one_minus_cos",
            stats["abbr_1_syn"],
        ),
        _csv_row(
            anchor,
            n_sentences,
            n_targets,
            "species_abbrev_delta_vs_syn",
            "unit_L2",
            stats["abbr_2_syn"],
        ),
    ]

    if have_rand:
        rows += [
            _csv_row(
                anchor,
                n_sentences,
                n_targets,
                "species_full_delta_vs_rand",
                "one_minus_cos",
                stats["full_1_rand"],
            ),
            _csv_row(
                anchor,
                n_sentences,
                n_targets,
                "species_full_delta_vs_rand",
                "unit_L2",
                stats["full_2_rand"],
            ),
            _csv_row(
                anchor,
                n_sentences,
                n_targets,
                "species_abbrev_delta_vs_rand",
                "one_minus_cos",
                stats["abbr_1_rand"],
            ),
            _csv_row(
                anchor,
                n_sentences,
                n_targets,
                "species_abbrev_delta_vs_rand",
                "unit_L2",
                stats["abbr_2_rand"],
            ),
        ]

    payload = {
        "_meta": {
            "anchor": anchor,
            "n_sentences": int(n_sentences),
            "n_targets": int(n_targets),
        },
        "synonym_full": {
            "one_minus_cos": stats["syn_full"]["one_minus_cos"],
            "unit_L2": stats["syn_full"]["unit_L2"],
        },
        "synonym_abbrev": {
            "one_minus_cos": stats["syn_abbrev"]["one_minus_cos"],
            "unit_L2": stats["syn_abbrev"]["unit_L2"],
        },
        "species_full": {
            "delta_vs_syn": {
                "one_minus_cos": stats["full_1_syn"],
                "unit_L2": stats["full_2_syn"],
            }
        },
        "species_abbrev": {
            "delta_vs_syn": {
                "one_minus_cos": stats["abbr_1_syn"],
                "unit_L2": stats["abbr_2_syn"],
            }
        },
    }

    if have_rand:
        payload.update(
            {
                "random_full": {
                    "one_minus_cos": stats["rand_full"]["one_minus_cos"],
                    "unit_L2": stats["rand_full"]["unit_L2"],
                },
                "random_abbrev": {
                    "one_minus_cos": stats["rand_abbrev"]["one_minus_cos"],
                    "unit_L2": stats["rand_abbrev"]["unit_L2"],
                },
            }
        )
        payload["species_full"]["delta_vs_rand"] = {
            "one_minus_cos": stats["full_1_rand"],
            "unit_L2": stats["full_2_rand"],
        }
        payload["species_abbrev"]["delta_vs_rand"] = {
            "one_minus_cos": stats["abbr_1_rand"],
            "unit_L2": stats["abbr_2_rand"],
        }

    return rows, payload


def _compute_species_result(
    i: int,
    z_full: np.ndarray,
    z_abbr: np.ndarray,
    z_syn_full: np.ndarray,
    z_syn_abbr: np.ndarray,
    species: List[str],
    bootstrap: int,
    ci: float,
    syn_ok_full: Optional[np.ndarray] = None,
    syn_ok_abbr: Optional[np.ndarray] = None,
    z_rand_full: Optional[np.ndarray] = None,
    z_rand_abbr: Optional[np.ndarray] = None,
    rand_ok_full: Optional[np.ndarray] = None,
    rand_ok_abbr: Optional[np.ndarray] = None,
) -> Tuple[int, List[List[Any]], str, dict]:
    S, N, _D = z_full.shape
    have_rand = z_rand_full is not None and z_rand_abbr is not None

    ok_f = syn_ok_full[i] if syn_ok_full is not None else None
    ok_a = syn_ok_abbr[i] if syn_ok_abbr is not None else None

    syn_omc_v, syn_l2_v, syn_base_omc, syn_base_l2 = _baseline_arrays(
        z_full[i], z_syn_full[i], ok_f
    )
    syn_abbr_omc_v, syn_abbr_l2_v, _syn_abbr_omc, _syn_abbr_l2 = _baseline_arrays(
        z_abbr[i], z_syn_abbr[i], ok_a
    )
    syn_mask = np.isfinite(syn_base_omc)

    if have_rand:
        assert z_rand_full is not None and z_rand_abbr is not None
        ok_rf = rand_ok_full[i] if rand_ok_full is not None else None
        ok_ra = rand_ok_abbr[i] if rand_ok_abbr is not None else None
        rand_omc_v, rand_l2_v, rand_base_omc, rand_base_l2 = _baseline_arrays(
            z_full[i], z_rand_full[i], ok_rf
        )
        rand_abbr_omc_v, rand_abbr_l2_v, _rand_abbr_omc, _rand_abbr_l2 = (
            _baseline_arrays(z_abbr[i], z_rand_abbr[i], ok_ra)
        )
        rand_mask = np.isfinite(rand_base_omc)
    else:
        rand_omc_v = rand_l2_v = rand_abbr_omc_v = rand_abbr_l2_v = None
        rand_base_omc = rand_base_l2 = rand_mask = None

    Zi = _torch(z_full[i])
    full_1_syn_parts: List[np.ndarray] = []
    full_2_syn_parts: List[np.ndarray] = []
    abbr_1_syn_parts: List[np.ndarray] = []
    abbr_2_syn_parts: List[np.ndarray] = []
    full_1_rand_parts: List[np.ndarray] = []
    full_2_rand_parts: List[np.ndarray] = []
    abbr_1_rand_parts: List[np.ndarray] = []
    abbr_2_rand_parts: List[np.ndarray] = []

    # Preserve the historical per-species target order:
    # full targets 0..i-1 use the old pair orientation (target, anchor),
    # then targets i+1..S-1 use (anchor, target).
    for j in range(i):
        Zj = _torch(z_full[j])
        cos, l2 = pairwise_compare(Zj, Zi)
        omc = 1.0 - _as_np(cos)
        l2n = _as_np(l2)
        full_1_syn_parts.append((omc - syn_base_omc)[syn_mask])
        full_2_syn_parts.append((l2n - syn_base_l2)[syn_mask])
        if have_rand:
            assert rand_base_omc is not None
            assert rand_base_l2 is not None
            assert rand_mask is not None
            full_1_rand_parts.append((omc - rand_base_omc)[rand_mask])
            full_2_rand_parts.append((l2n - rand_base_l2)[rand_mask])

    for j in range(i + 1, S):
        Zj = _torch(z_full[j])
        cos, l2 = pairwise_compare(Zi, Zj)
        omc = 1.0 - _as_np(cos)
        l2n = _as_np(l2)
        full_1_syn_parts.append((omc - syn_base_omc)[syn_mask])
        full_2_syn_parts.append((l2n - syn_base_l2)[syn_mask])
        if have_rand:
            assert rand_base_omc is not None
            assert rand_base_l2 is not None
            assert rand_mask is not None
            full_1_rand_parts.append((omc - rand_base_omc)[rand_mask])
            full_2_rand_parts.append((l2n - rand_base_l2)[rand_mask])

    for j in range(S):
        if i == j:
            continue
        ZjA = _torch(z_abbr[j])
        cos, l2 = pairwise_compare(Zi, ZjA)
        omc = 1.0 - _as_np(cos)
        l2n = _as_np(l2)
        abbr_1_syn_parts.append((omc - syn_base_omc)[syn_mask])
        abbr_2_syn_parts.append((l2n - syn_base_l2)[syn_mask])
        if have_rand:
            assert rand_base_omc is not None
            assert rand_base_l2 is not None
            assert rand_mask is not None
            abbr_1_rand_parts.append((omc - rand_base_omc)[rand_mask])
            abbr_2_rand_parts.append((l2n - rand_base_l2)[rand_mask])

    full_1_syn = _cat(full_1_syn_parts)
    full_2_syn = _cat(full_2_syn_parts)
    abbr_1_syn = _cat(abbr_1_syn_parts)
    abbr_2_syn = _cat(abbr_2_syn_parts)

    stats = {
        "syn_full": {
            "one_minus_cos": summarize(syn_omc_v, n_boot=bootstrap, level=ci),
            "unit_L2": summarize(syn_l2_v, n_boot=bootstrap, level=ci),
        },
        "syn_abbrev": {
            "one_minus_cos": summarize(syn_abbr_omc_v, n_boot=bootstrap, level=ci),
            "unit_L2": summarize(syn_abbr_l2_v, n_boot=bootstrap, level=ci),
        },
        "full_1_syn": summarize(full_1_syn, n_boot=bootstrap, level=ci),
        "full_2_syn": summarize(full_2_syn, n_boot=bootstrap, level=ci),
        "abbr_1_syn": summarize(abbr_1_syn, n_boot=bootstrap, level=ci),
        "abbr_2_syn": summarize(abbr_2_syn, n_boot=bootstrap, level=ci),
    }

    if have_rand:
        stats.update(
            {
                "rand_full": {
                    "one_minus_cos": summarize(
                        rand_omc_v, n_boot=bootstrap, level=ci
                    ),
                    "unit_L2": summarize(rand_l2_v, n_boot=bootstrap, level=ci),
                },
                "rand_abbrev": {
                    "one_minus_cos": summarize(
                        rand_abbr_omc_v, n_boot=bootstrap, level=ci
                    ),
                    "unit_L2": summarize(rand_abbr_l2_v, n_boot=bootstrap, level=ci),
                },
                "full_1_rand": summarize(
                    _cat(full_1_rand_parts), n_boot=bootstrap, level=ci
                ),
                "full_2_rand": summarize(
                    _cat(full_2_rand_parts), n_boot=bootstrap, level=ci
                ),
                "abbr_1_rand": summarize(
                    _cat(abbr_1_rand_parts), n_boot=bootstrap, level=ci
                ),
                "abbr_2_rand": summarize(
                    _cat(abbr_2_rand_parts), n_boot=bootstrap, level=ci
                ),
            }
        )

    rows, payload = _species_summary_from_stats(
        anchor=species[i],
        n_sentences=N,
        n_targets=len(species) - 1,
        stats=stats,
        have_rand=have_rand,
    )
    return i, rows, _summary_filename(species[i]), payload


_SPAWN_DATA: dict[str, Any] = {}


def _save_parallel_inputs(
    tmpdir: Path,
    z_full: np.ndarray,
    z_abbr: np.ndarray,
    z_syn_full: np.ndarray,
    z_syn_abbr: np.ndarray,
    syn_ok_full: Optional[np.ndarray],
    syn_ok_abbr: Optional[np.ndarray],
    z_rand_full: Optional[np.ndarray],
    z_rand_abbr: Optional[np.ndarray],
    rand_ok_full: Optional[np.ndarray],
    rand_ok_abbr: Optional[np.ndarray],
) -> dict[str, Optional[str]]:
    arrays = {
        "z_full": z_full,
        "z_abbr": z_abbr,
        "z_syn_full": z_syn_full,
        "z_syn_abbr": z_syn_abbr,
        "syn_ok_full": syn_ok_full,
        "syn_ok_abbr": syn_ok_abbr,
        "z_rand_full": z_rand_full,
        "z_rand_abbr": z_rand_abbr,
        "rand_ok_full": rand_ok_full,
        "rand_ok_abbr": rand_ok_abbr,
    }

    paths: dict[str, Optional[str]] = {}
    for name, arr in arrays.items():
        if arr is None:
            paths[name] = None
            continue
        path = tmpdir / f"{name}.npy"
        np.save(path, arr, allow_pickle=False)
        paths[name] = str(path)
    return paths


def _load_memmap(path: Optional[str]) -> Optional[np.ndarray]:
    return np.load(path, mmap_mode="c") if path is not None else None


def _set_single_thread_worker_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")


def _init_spawn_worker(
    array_paths: dict[str, Optional[str]],
    species: List[str],
    bootstrap: int,
    ci: float,
) -> None:
    _set_single_thread_worker_env()
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _SPAWN_DATA.clear()
    _SPAWN_DATA.update(
        {
            "z_full": _load_memmap(array_paths["z_full"]),
            "z_abbr": _load_memmap(array_paths["z_abbr"]),
            "z_syn_full": _load_memmap(array_paths["z_syn_full"]),
            "z_syn_abbr": _load_memmap(array_paths["z_syn_abbr"]),
            "species": species,
            "bootstrap": bootstrap,
            "ci": ci,
            "syn_ok_full": _load_memmap(array_paths["syn_ok_full"]),
            "syn_ok_abbr": _load_memmap(array_paths["syn_ok_abbr"]),
            "z_rand_full": _load_memmap(array_paths["z_rand_full"]),
            "z_rand_abbr": _load_memmap(array_paths["z_rand_abbr"]),
            "rand_ok_full": _load_memmap(array_paths["rand_ok_full"]),
            "rand_ok_abbr": _load_memmap(array_paths["rand_ok_abbr"]),
        }
    )


def _compute_species_result_from_spawn_worker(
    i: int,
) -> Tuple[int, List[List[Any]], str, dict]:
    return _compute_species_result(i=i, **_SPAWN_DATA)


def _write_parallel_results(out_dir: str, iterator, expected_count: int) -> None:
    csv_path = os.path.join(out_dir, "minimal_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_CSV_HEADER)
        for expected_i, (i, rows, json_name, payload) in enumerate(iterator):
            if i != expected_i:
                raise RuntimeError(f"parallel result order mismatch: got {i}, expected {expected_i}")
            w.writerows(rows)
            with open(os.path.join(out_dir, json_name), "w", encoding="utf-8") as jf:
                json.dump(payload, jf, ensure_ascii=False, indent=2)

    if expected_count == 0:
        raise RuntimeError("no species were summarized")
    print(f"[SAVE] {csv_path}")


def _compute_stats_parallel(
    z_full: np.ndarray,
    z_abbr: np.ndarray,
    z_syn_full: np.ndarray,
    z_syn_abbr: np.ndarray,
    out_dir: str,
    species: List[str],
    bootstrap: int,
    ci: float,
    workers: int,
    syn_ok_full: Optional[np.ndarray] = None,
    syn_ok_abbr: Optional[np.ndarray] = None,
    z_rand_full: Optional[np.ndarray] = None,
    z_rand_abbr: Optional[np.ndarray] = None,
    rand_ok_full: Optional[np.ndarray] = None,
    rand_ok_abbr: Optional[np.ndarray] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    S = z_full.shape[0]
    workers = max(1, min(int(workers), S))

    if workers == 1:
        iterator = (
            _compute_species_result(
                i,
                z_full,
                z_abbr,
                z_syn_full,
                z_syn_abbr,
                species,
                bootstrap,
                ci,
                syn_ok_full=syn_ok_full,
                syn_ok_abbr=syn_ok_abbr,
                z_rand_full=z_rand_full,
                z_rand_abbr=z_rand_abbr,
                rand_ok_full=rand_ok_full,
                rand_ok_abbr=rand_ok_abbr,
            )
            for i in range(S)
        )
        _write_parallel_results(out_dir, iterator, S)
        return

    if "spawn" not in mp.get_all_start_methods():
        raise RuntimeError("--workers > 1 requires a multiprocessing spawn start method")

    _set_single_thread_worker_env()
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="tea_summarize_arrays_") as tmp:
        array_paths = _save_parallel_inputs(
            Path(tmp),
            z_full,
            z_abbr,
            z_syn_full,
            z_syn_abbr,
            syn_ok_full,
            syn_ok_abbr,
            z_rand_full,
            z_rand_abbr,
            rand_ok_full,
            rand_ok_abbr,
        )
        with ctx.Pool(
            processes=workers,
            initializer=_init_spawn_worker,
            initargs=(array_paths, species, bootstrap, ci),
        ) as pool:
            _write_parallel_results(
                out_dir,
                pool.imap(_compute_species_result_from_spawn_worker, range(S), chunksize=1),
                S,
            )


def _compute_stats(
    z_full: np.ndarray,
    z_abbr: np.ndarray,
    z_syn_full: np.ndarray,
    z_syn_abbr: np.ndarray,
    out_dir: str,
    species: List[str],
    bootstrap: int,
    ci: float,
    syn_ok_full: Optional[np.ndarray] = None,
    syn_ok_abbr: Optional[np.ndarray] = None,
    z_rand_full: Optional[np.ndarray] = None,
    z_rand_abbr: Optional[np.ndarray] = None,
    rand_ok_full: Optional[np.ndarray] = None,
    rand_ok_abbr: Optional[np.ndarray] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    S, N, D = z_full.shape

    have_rand = z_rand_full is not None and z_rand_abbr is not None

    syn_full_vals = {}
    syn_abbr_vals = {}
    syn_full_base_1 = {}
    syn_full_base_2 = {}

    for i in range(S):
        ok_f = syn_ok_full[i] if syn_ok_full is not None else None
        ok_a = syn_ok_abbr[i] if syn_ok_abbr is not None else None

        omc_v, l2_v, omc_fullarr, l2_fullarr = _baseline_arrays(
            z_full[i], z_syn_full[i], ok_f
        )
        syn_full_vals[i] = {"one_minus_cos": omc_v, "unit_L2": l2_v}
        syn_full_base_1[i] = omc_fullarr
        syn_full_base_2[i] = l2_fullarr

        omc_v, l2_v, _omc_fullarr, _l2_fullarr = _baseline_arrays(
            z_abbr[i], z_syn_abbr[i], ok_a
        )
        syn_abbr_vals[i] = {"one_minus_cos": omc_v, "unit_L2": l2_v}

    rand_full_vals = {}
    rand_abbr_vals = {}
    rand_full_base_1 = {}
    rand_full_base_2 = {}

    if have_rand:
        assert z_rand_full is not None and z_rand_abbr is not None
        for i in range(S):
            ok_f = rand_ok_full[i] if rand_ok_full is not None else None
            ok_a = rand_ok_abbr[i] if rand_ok_abbr is not None else None

            omc_v, l2_v, omc_fullarr, l2_fullarr = _baseline_arrays(
                z_full[i], z_rand_full[i], ok_f
            )
            rand_full_vals[i] = {"one_minus_cos": omc_v, "unit_L2": l2_v}
            rand_full_base_1[i] = omc_fullarr
            rand_full_base_2[i] = l2_fullarr

            omc_v, l2_v, _omc_fullarr, _l2_fullarr = _baseline_arrays(
                z_abbr[i], z_rand_abbr[i], ok_a
            )
            rand_abbr_vals[i] = {"one_minus_cos": omc_v, "unit_L2": l2_v}

    # accumulators: delta vs synonym baseline
    per_species_syn = _init_delta_accumulators(S)

    # accumulators: delta vs random baseline
    per_species_rand = _init_delta_accumulators(S) if have_rand else None

    for i in range(S):
        Zi = _torch(z_full[i])

        syn_base_i_1 = syn_full_base_1[i]
        syn_base_i_2 = syn_full_base_2[i]
        syn_mask_i = np.isfinite(syn_base_i_1)

        if have_rand:
            r_base_i_1 = rand_full_base_1[i]
            r_base_i_2 = rand_full_base_2[i]
            r_mask_i = np.isfinite(r_base_i_1)

        for j in range(i + 1, S):
            Zj = _torch(z_full[j])
            cos, l2 = pairwise_compare(Zi, Zj)
            omc = 1.0 - _as_np(cos)
            l2n = _as_np(l2)

            d1 = (omc - syn_base_i_1)[syn_mask_i]
            d2 = (l2n - syn_base_i_2)[syn_mask_i]
            per_species_syn[i]["full_1mcos_delta"].append(d1)
            per_species_syn[i]["full_L2_delta"].append(d2)

            syn_base_j_1 = syn_full_base_1[j]
            syn_base_j_2 = syn_full_base_2[j]
            syn_mask_j = np.isfinite(syn_base_j_1)
            d1j = (omc - syn_base_j_1)[syn_mask_j]
            d2j = (l2n - syn_base_j_2)[syn_mask_j]
            per_species_syn[j]["full_1mcos_delta"].append(d1j)
            per_species_syn[j]["full_L2_delta"].append(d2j)

            if have_rand and per_species_rand is not None:
                dr1 = (omc - r_base_i_1)[r_mask_i]
                dr2 = (l2n - r_base_i_2)[r_mask_i]
                per_species_rand[i]["full_1mcos_delta"].append(dr1)
                per_species_rand[i]["full_L2_delta"].append(dr2)

                r_base_j_1 = rand_full_base_1[j]
                r_base_j_2 = rand_full_base_2[j]
                r_mask_j = np.isfinite(r_base_j_1)
                dr1j = (omc - r_base_j_1)[r_mask_j]
                dr2j = (l2n - r_base_j_2)[r_mask_j]
                per_species_rand[j]["full_1mcos_delta"].append(dr1j)
                per_species_rand[j]["full_L2_delta"].append(dr2j)

    for i in range(S):
        Zi = _torch(z_full[i])

        syn_base_i_1 = syn_full_base_1[i]
        syn_base_i_2 = syn_full_base_2[i]
        syn_mask_i = np.isfinite(syn_base_i_1)

        if have_rand:
            r_base_i_1 = rand_full_base_1[i]
            r_base_i_2 = rand_full_base_2[i]
            r_mask_i = np.isfinite(r_base_i_1)

        for j in range(S):
            if i == j:
                continue
            ZjA = _torch(z_abbr[j])
            cos, l2 = pairwise_compare(Zi, ZjA)
            omc = 1.0 - _as_np(cos)
            l2n = _as_np(l2)

            per_species_syn[i]["abbr_1mcos_delta"].append(
                (omc - syn_base_i_1)[syn_mask_i]
            )
            per_species_syn[i]["abbr_L2_delta"].append((l2n - syn_base_i_2)[syn_mask_i])

            if have_rand and per_species_rand is not None:
                per_species_rand[i]["abbr_1mcos_delta"].append(
                    (omc - r_base_i_1)[r_mask_i]
                )
                per_species_rand[i]["abbr_L2_delta"].append(
                    (l2n - r_base_i_2)[r_mask_i]
                )

    csv_path = os.path.join(out_dir, "minimal_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "anchor",
                "n_sentences",
                "n_targets",
                "condition",
                "stat",
                "mean",
                "ci_lo",
                "ci_hi",
                "n",
                "reanchored_from",
            ]
        )

        for i, s in enumerate(species):

            def cat(which: dict, name: str) -> np.ndarray:
                arrs = which[i][name]
                return np.concatenate(arrs, axis=0) if arrs else np.array([])

            # synonym-delta aggregates
            full_1_syn = cat(per_species_syn, "full_1mcos_delta")
            full_2_syn = cat(per_species_syn, "full_L2_delta")
            abbr_1_syn = cat(per_species_syn, "abbr_1mcos_delta")
            abbr_2_syn = cat(per_species_syn, "abbr_L2_delta")

            # random-delta aggregates
            if have_rand and per_species_rand is not None:
                full_1_rand = cat(per_species_rand, "full_1mcos_delta")
                full_2_rand = cat(per_species_rand, "full_L2_delta")
                abbr_1_rand = cat(per_species_rand, "abbr_1mcos_delta")
                abbr_2_rand = cat(per_species_rand, "abbr_L2_delta")
            else:
                full_1_rand = full_2_rand = abbr_1_rand = abbr_2_rand = None

            stats = {
                "syn_full": {
                    "one_minus_cos": summarize(
                        syn_full_vals[i]["one_minus_cos"], n_boot=bootstrap, level=ci
                    ),
                    "unit_L2": summarize(
                        syn_full_vals[i]["unit_L2"], n_boot=bootstrap, level=ci
                    ),
                },
                "syn_abbrev": {
                    "one_minus_cos": summarize(
                        syn_abbr_vals[i]["one_minus_cos"], n_boot=bootstrap, level=ci
                    ),
                    "unit_L2": summarize(
                        syn_abbr_vals[i]["unit_L2"], n_boot=bootstrap, level=ci
                    ),
                },
                "full_1_syn": summarize(full_1_syn, n_boot=bootstrap, level=ci),
                "full_2_syn": summarize(full_2_syn, n_boot=bootstrap, level=ci),
                "abbr_1_syn": summarize(abbr_1_syn, n_boot=bootstrap, level=ci),
                "abbr_2_syn": summarize(abbr_2_syn, n_boot=bootstrap, level=ci),
            }

            if have_rand and per_species_rand is not None:
                stats.update(
                    {
                        "rand_full": {
                            "one_minus_cos": summarize(
                                rand_full_vals[i]["one_minus_cos"],
                                n_boot=bootstrap,
                                level=ci,
                            ),
                            "unit_L2": summarize(
                                rand_full_vals[i]["unit_L2"], n_boot=bootstrap, level=ci
                            ),
                        },
                        "rand_abbrev": {
                            "one_minus_cos": summarize(
                                rand_abbr_vals[i]["one_minus_cos"],
                                n_boot=bootstrap,
                                level=ci,
                            ),
                            "unit_L2": summarize(
                                rand_abbr_vals[i]["unit_L2"], n_boot=bootstrap, level=ci
                            ),
                        },
                        "full_1_rand": summarize(
                            full_1_rand, n_boot=bootstrap, level=ci
                        ),
                        "full_2_rand": summarize(
                            full_2_rand, n_boot=bootstrap, level=ci
                        ),
                        "abbr_1_rand": summarize(
                            abbr_1_rand, n_boot=bootstrap, level=ci
                        ),
                        "abbr_2_rand": summarize(
                            abbr_2_rand, n_boot=bootstrap, level=ci
                        ),
                    }
                )

            n_targets = len(species) - 1

            def row(cond: str, stat_name: str, stat_obj: dict):
                w.writerow(
                    [
                        s,
                        int(N),
                        int(n_targets),
                        cond,
                        stat_name,
                        stat_obj["mean"],
                        stat_obj["ci95"][0],
                        stat_obj["ci95"][1],
                        stat_obj["n"],
                        "",
                    ]
                )

            # baselines
            row("synonym_full", "one_minus_cos", stats["syn_full"]["one_minus_cos"])
            row("synonym_full", "unit_L2", stats["syn_full"]["unit_L2"])
            row("synonym_abbrev", "one_minus_cos", stats["syn_abbrev"]["one_minus_cos"])
            row("synonym_abbrev", "unit_L2", stats["syn_abbrev"]["unit_L2"])

            if have_rand and per_species_rand is not None:
                row("random_full", "one_minus_cos", stats["rand_full"]["one_minus_cos"])
                row("random_full", "unit_L2", stats["rand_full"]["unit_L2"])
                row(
                    "random_abbrev",
                    "one_minus_cos",
                    stats["rand_abbrev"]["one_minus_cos"],
                )
                row("random_abbrev", "unit_L2", stats["rand_abbrev"]["unit_L2"])

            # deltas (delta vs syn)
            row("species_full_delta_vs_syn", "one_minus_cos", stats["full_1_syn"])
            row("species_full_delta_vs_syn", "unit_L2", stats["full_2_syn"])
            row("species_abbrev_delta_vs_syn", "one_minus_cos", stats["abbr_1_syn"])
            row("species_abbrev_delta_vs_syn", "unit_L2", stats["abbr_2_syn"])

            # deltas (delta vs rand)
            if have_rand and per_species_rand is not None:
                row("species_full_delta_vs_rand", "one_minus_cos", stats["full_1_rand"])
                row("species_full_delta_vs_rand", "unit_L2", stats["full_2_rand"])
                row(
                    "species_abbrev_delta_vs_rand",
                    "one_minus_cos",
                    stats["abbr_1_rand"],
                )
                row("species_abbrev_delta_vs_rand", "unit_L2", stats["abbr_2_rand"])

            j = {
                "_meta": {
                    "anchor": s,
                    "n_sentences": int(N),
                    "n_targets": int(n_targets),
                },
                "synonym_full": {
                    "one_minus_cos": stats["syn_full"]["one_minus_cos"],
                    "unit_L2": stats["syn_full"]["unit_L2"],
                },
                "synonym_abbrev": {
                    "one_minus_cos": stats["syn_abbrev"]["one_minus_cos"],
                    "unit_L2": stats["syn_abbrev"]["unit_L2"],
                },
                "species_full": {
                    "delta_vs_syn": {
                        "one_minus_cos": stats["full_1_syn"],
                        "unit_L2": stats["full_2_syn"],
                    }
                },
                "species_abbrev": {
                    "delta_vs_syn": {
                        "one_minus_cos": stats["abbr_1_syn"],
                        "unit_L2": stats["abbr_2_syn"],
                    }
                },
            }

            if have_rand and per_species_rand is not None:
                j.update(
                    {
                        "random_full": {
                            "one_minus_cos": stats["rand_full"]["one_minus_cos"],
                            "unit_L2": stats["rand_full"]["unit_L2"],
                        },
                        "random_abbrev": {
                            "one_minus_cos": stats["rand_abbrev"]["one_minus_cos"],
                            "unit_L2": stats["rand_abbrev"]["unit_L2"],
                        },
                    }
                )
                j["species_full"]["delta_vs_rand"] = {
                    "one_minus_cos": stats["full_1_rand"],
                    "unit_L2": stats["full_2_rand"],
                }
                j["species_abbrev"]["delta_vs_rand"] = {
                    "one_minus_cos": stats["abbr_1_rand"],
                    "unit_L2": stats["abbr_2_rand"],
                }

            with open(
                os.path.join(out_dir, f"summary_{s.lower().replace(' ', '_')}.json"),
                "w",
                encoding="utf-8",
            ) as jf:
                json.dump(j, jf, ensure_ascii=False, indent=2)

    print(f"[SAVE] {csv_path}")


def run_from_cache(
    npz_path: str,
    out_root: str,
    methods: List[str],
    abtt_k: int,
    whiten_eps: float,
    max_fit_rows: int,
    seed: int,
    bootstrap: int,
    ci: float,
    workers: int = 1,
):
    (
        z_full,
        z_abbr,
        z_sfull,
        z_sabbr,
        syn_ok_full,
        syn_ok_abbr,
        z_rfull,
        z_rabbr,
        rand_ok_full,
        rand_ok_abbr,
        meta,
    ) = _load_cache(npz_path)

    species = meta["species"]
    S, N, D = z_full.shape
    os.makedirs(out_root, exist_ok=True)

    # mats for fitting transforms (flatten species & sentences)
    mats = [
        z_full.reshape(-1, D),
        z_abbr.reshape(-1, D),
        z_sfull.reshape(-1, D),
        z_sabbr.reshape(-1, D),
    ]

    have_rand = z_rfull is not None and z_rabbr is not None
    if have_rand:
        mats += [z_rfull.reshape(-1, D), z_rabbr.reshape(-1, D)]

    for m in [m.lower() for m in methods]:
        out_dir = os.path.join(out_root, m)
        if m == "none":
            compute = _compute_stats_parallel if workers != 1 else _compute_stats
            kwargs = {"workers": workers} if workers != 1 else {}
            compute(
                z_full,
                z_abbr,
                z_sfull,
                z_sabbr,
                out_dir,
                species,
                bootstrap,
                ci,
                syn_ok_full=syn_ok_full,
                syn_ok_abbr=syn_ok_abbr,
                z_rand_full=z_rfull,
                z_rand_abbr=z_rabbr,
                rand_ok_full=rand_ok_full,
                rand_ok_abbr=rand_ok_abbr,
                **kwargs,
            )
        elif m in ("abtt", "whiten"):
            params = pp_fit(
                m,
                mats,
                abtt_k=abtt_k,
                whiten_eps=whiten_eps,
                max_fit_rows=max_fit_rows,
                seed=seed,
            )
            t_mats = pp_transform(mats, params)

            # unpack back to blocks
            t_full = t_mats[0].reshape(S, N, D)
            t_abbr = t_mats[1].reshape(S, N, D)
            t_sfull = t_mats[2].reshape(S, N, D)
            t_sabbr = t_mats[3].reshape(S, N, D)

            if have_rand:
                t_rfull = t_mats[4].reshape(S, N, D)
                t_rabbr = t_mats[5].reshape(S, N, D)
            else:
                t_rfull = None
                t_rabbr = None

            compute = _compute_stats_parallel if workers != 1 else _compute_stats
            kwargs = {"workers": workers} if workers != 1 else {}
            compute(
                t_full,
                t_abbr,
                t_sfull,
                t_sabbr,
                out_dir,
                species,
                bootstrap,
                ci,
                syn_ok_full=syn_ok_full,
                syn_ok_abbr=syn_ok_abbr,
                z_rand_full=t_rfull,
                z_rand_abbr=t_rabbr,
                rand_ok_full=rand_ok_full,
                rand_ok_abbr=rand_ok_abbr,
                **kwargs,
            )
        else:
            raise SystemExit(f"Unknown method: {m}")
        print(f"[DONE] {m} -> {out_dir}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Summarize none/abtt/whiten from a cached embedding NPZ (no re-encoding)."
    )
    p.add_argument(
        "--workdir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "work"),
        help="Work directory root (default: ./work at repo root)",
    )
    p.add_argument(
        "--cache",
        required=False,
        default=None,
        help="Path to NPZ created by embed_cache.py (default: <workdir>/cache/species_embeddings.npz)",
    )
    p.add_argument(
        "--out",
        required=False,
        default=None,
        help="Root output dir; subdirs per method will be created (default: <workdir>/summaries_from_cache)",
    )
    p.add_argument("--methods", default="none,abtt,whiten", help="Comma-separated list")
    p.add_argument("--abtt-k", type=int, default=10)
    p.add_argument("--whiten-eps", type=float, default=1e-6)
    p.add_argument("--pp-max-fit-rows", type=int, default=100_000)
    p.add_argument("--pp-seed", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of spawn worker processes for per-species summarization (default: 1).",
    )
    return p


def main():
    a = build_parser().parse_args()
    workdir = Path(a.workdir)

    cache_path = a.cache or str(workdir / "cache" / "species_embeddings.npz")
    out_root = a.out or str(workdir / "summaries_from_cache")

    run_from_cache(
        npz_path=cache_path,
        out_root=out_root,
        methods=[m.strip() for m in a.methods.split(",") if m.strip()],
        abtt_k=a.abtt_k,
        whiten_eps=a.whiten_eps,
        max_fit_rows=a.pp_max_fit_rows,
        seed=a.pp_seed,
        bootstrap=a.bootstrap,
        ci=a.ci,
        workers=a.workers,
    )


if __name__ == "__main__":
    main()
