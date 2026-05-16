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
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from encode import encode_sentences, load_model
from io_utils import load_placeholder_templates, read_species_list
from utils import (abbreviate_binomial, make_random_word_swapper,
                   make_synonym_swapper)


def _replace_placeholder(
    templates: List[str], placeholder: str, value: str
) -> List[str]:
    # Replace only the first occurrence.
    # Loader can enforce single-occurrence templates
    return [t.replace(placeholder, value, 1) for t in templates]


def _syn_manifest_path(out_npz: str) -> str:
    # sidecar manifest for auditability
    return out_npz + ".synswap.jsonl"


def _rand_manifest_path(out_npz: str) -> str:
    # sidecar manifest for auditability
    return out_npz + ".randswap.jsonl"


def _select_species(
    species: List[str],
    max_species: int,
    sample_species: int,
    sample_species_seed: int,
) -> List[str]:
    """Optionally downselect the species list for tractable runs.

    The embedding cache scales as O(|species| * |templates|). With a full UniProt
    list this becomes impractical, so the script supports a deterministic slice
    (max_species) or deterministic sampling (sample_species).

    Selection is performed in-file order (typically alphabetical) so the output
    remains stable across runs.
    """

    if max_species and max_species > 0:
        return species[:max_species]

    if sample_species and sample_species > 0:
        if sample_species > len(species):
            raise SystemExit(
                f"[ERROR] --sample-species={sample_species} exceeds available species ({len(species)})"
            )
        rng = np.random.RandomState(sample_species_seed)
        idx = rng.choice(len(species), size=sample_species, replace=False)
        idx_sorted = sorted(int(i) for i in idx)
        return [species[i] for i in idx_sorted]

    return species


def build_and_save_cache(
    dataset_file: str,
    species_file: str,
    out_npz: str,
    placeholder: str,
    require_single: bool,
    model_name: str,
    batch_size: int,
    pooling: str,
    # synonym control baseline
    syn_seed: int,
    syn_max_attempts: int,
    syn_success_min: float,
    syn_abort_on_protected_fail: bool,
    syn_write_manifest: bool,
    # random control baseline
    rand_enabled: bool,
    rand_seed: int,
    rand_max_attempts: int,
    rand_success_min: float,
    rand_write_manifest: bool,
    rand_wordlist_size: int,
    # scale controls
    max_species: int,
    sample_species: int,
    sample_species_seed: int,
    allow_huge: bool,
):
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)

    species = read_species_list(species_file)
    templates = load_placeholder_templates(dataset_file, placeholder, require_single)
    if not species:
        raise SystemExit(f"[ERROR] No species in {species_file}")
    if not templates:
        raise SystemExit(f"[ERROR] No templates with placeholder '{placeholder}'")

    species = _select_species(
        species,
        max_species=max_species,
        sample_species=sample_species,
        sample_species_seed=sample_species_seed,
    )

    print(
        f"[INFO] species={len(species)}  sentences={len(templates)}  model={model_name}  pooling={pooling}"
    )

    model, tok = load_model(model_name)

    S, N = len(species), len(templates)
    D = int(model.config.hidden_size)

    # Cache holds float32 tensors of shape (S, N, D).
    # Base: full + abbr + syn_full + syn_abbr (4)
    n_variants = 4 + (2 if rand_enabled else 0)
    est_bytes = S * N * D * n_variants * 4
    est_gb = est_bytes / 1e9
    print(
        f"[INFO] estimated_cache_raw_gb={est_gb:.2f}  ({n_variants}xfloat32 variants)"
    )
    if est_gb > 12.0 and not allow_huge:
        raise SystemExit(
            "[ERROR] Cache would be too large. "
            f"estimated_cache_raw_gb={est_gb:.2f}. "
            "Limit species/templates (e.g. --max-species or --sample-species), "
            "or pass --allow-huge if you really want to proceed."
        )

    zs_full = np.empty((S, N, D), np.float32)
    zs_abbr = np.empty_like(zs_full)
    zs_sfull = np.empty_like(zs_full)
    zs_sabbr = np.empty_like(zs_full)

    syn_ok_full = np.zeros((S, N), dtype=np.uint8)
    syn_ok_abbr = np.zeros((S, N), dtype=np.uint8)

    zs_rfull = None
    zs_rabbr = None
    rand_ok_full = None
    rand_ok_abbr = None

    if rand_enabled:
        zs_rfull = np.empty_like(zs_full)
        zs_rabbr = np.empty_like(zs_full)
        rand_ok_full = np.zeros((S, N), dtype=np.uint8)
        rand_ok_abbr = np.zeros((S, N), dtype=np.uint8)

    # manifests
    syn_manifest_f = None
    syn_manifest_path = _syn_manifest_path(out_npz)
    if syn_write_manifest:
        syn_manifest_f = open(syn_manifest_path, "w", encoding="utf-8")

    rand_manifest_f = None
    rand_manifest_path = _rand_manifest_path(out_npz)
    if rand_enabled and rand_write_manifest:
        rand_manifest_f = open(rand_manifest_path, "w", encoding="utf-8")

    # global stats
    syn_total_ok = 0
    syn_total_attempts = 0
    syn_total_rows = 0
    syn_reasons: Dict[str, int] = {}

    rand_total_ok = 0
    rand_total_attempts = 0
    rand_total_rows = 0
    rand_reasons: Dict[str, int] = {}

    def _bump_reason(d: Dict[str, int], r: str):
        d[r] = d.get(r, 0) + 1

    try:
        for i, s in enumerate(species):
            full_texts = _replace_placeholder(templates, placeholder, s)
            ab = abbreviate_binomial(s)
            abbr_texts = _replace_placeholder(templates, placeholder, ab)

            # ---------- synonym swap baseline ----------
            swap_full = make_synonym_swapper(
                [s], seed=syn_seed, max_attempts=syn_max_attempts
            )
            swap_abbr = make_synonym_swapper(
                [ab], seed=syn_seed, max_attempts=syn_max_attempts
            )

            syn_full_texts: List[str] = []
            syn_abbr_texts: List[str] = []

            ok_here = 0
            unchanged_ok_here = 0
            protected_fail_here = 0

            # build synonym-swapped full
            for t_idx, x in enumerate(full_texts):
                res = swap_full(x)
                syn_total_rows += 1
                syn_total_attempts += res.attempts
                _bump_reason(syn_reasons, res.reason)

                if res.ok:
                    ok_here += 1
                    syn_total_ok += 1
                    syn_ok_full[i, t_idx] = 1
                    y = res.text_out or x
                    if y == x:
                        unchanged_ok_here += 1
                    syn_full_texts.append(y)
                else:
                    syn_ok_full[i, t_idx] = 0
                    syn_full_texts.append(x)
                    if res.reason == "protected_modified":
                        protected_fail_here += 1

                if syn_manifest_f is not None:
                    syn_manifest_f.write(
                        json.dumps(
                            {
                                "species": s,
                                "variant": "syn_full",
                                "template_index": t_idx,
                                "ok": bool(res.ok),
                                "reason": res.reason,
                                "attempts": int(res.attempts),
                                "changed_token": res.changed_token,
                                "replacement": res.replacement,
                                "protected": s,
                                "text_in": x,
                                "text_out": res.text_out,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            # build synonym-swapped abbr
            for t_idx, x in enumerate(abbr_texts):
                res = swap_abbr(x)
                syn_total_rows += 1
                syn_total_attempts += res.attempts
                _bump_reason(syn_reasons, res.reason)

                if res.ok:
                    ok_here += 1
                    syn_total_ok += 1
                    syn_ok_abbr[i, t_idx] = 1
                    y = res.text_out or x
                    if y == x:
                        unchanged_ok_here += 1
                    syn_abbr_texts.append(y)
                else:
                    syn_ok_abbr[i, t_idx] = 0
                    syn_abbr_texts.append(x)
                    if res.reason == "protected_modified":
                        protected_fail_here += 1

                if syn_manifest_f is not None:
                    syn_manifest_f.write(
                        json.dumps(
                            {
                                "species": s,
                                "variant": "syn_abbr",
                                "template_index": t_idx,
                                "ok": bool(res.ok),
                                "reason": res.reason,
                                "attempts": int(res.attempts),
                                "changed_token": res.changed_token,
                                "replacement": res.replacement,
                                "protected": ab,
                                "text_in": x,
                                "text_out": res.text_out,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            # per-species gating
            denom = 2 * N
            success_rate = ok_here / float(denom) if denom else 0.0

            if unchanged_ok_here > 0:
                raise SystemExit(
                    f"[ERROR] synonym swap invariant violated (unchanged output accepted) for {s}: {unchanged_ok_here}"
                )

            if syn_abort_on_protected_fail and protected_fail_here > 0:
                raise SystemExit(
                    f"[ERROR] synonym swap invariant violated (protected phrase modified) for {s}: {protected_fail_here}"
                )

            if success_rate < syn_success_min:
                raise SystemExit(
                    f"[ERROR] synonym swap success rate too low for {s}: {success_rate:.3f} < {syn_success_min:.3f}"
                )

            # report duplicates among ok rows (non-fatal)
            ok_full = syn_ok_full[i].astype(bool)
            ok_abbr = syn_ok_abbr[i].astype(bool)
            dup_full = 0.0
            dup_abbr = 0.0
            if ok_full.any():
                dup_full = 1.0 - (
                    len(set([syn_full_texts[k] for k in range(N) if ok_full[k]]))
                    / float(ok_full.sum())
                )
            if ok_abbr.any():
                dup_abbr = 1.0 - (
                    len(set([syn_abbr_texts[k] for k in range(N) if ok_abbr[k]]))
                    / float(ok_abbr.sum())
                )
            print(
                f"[SYN] {s}: success={success_rate:.3f} dup_full={dup_full:.3f} dup_abbr={dup_abbr:.3f}"
            )

            # random swap baseline
            rand_full_texts: List[str] = []
            rand_abbr_texts: List[str] = []

            rand_success_rate = float("nan")
            if rand_enabled:
                swap_rfull = make_random_word_swapper(
                    [s],
                    seed=rand_seed,
                    max_attempts=rand_max_attempts,
                    wordlist_size=rand_wordlist_size,
                )
                swap_rabbr = make_random_word_swapper(
                    [ab],
                    seed=rand_seed,
                    max_attempts=rand_max_attempts,
                    wordlist_size=rand_wordlist_size,
                )

                ok_rand_here = 0
                unchanged_rand_ok_here = 0

                for t_idx, x in enumerate(full_texts):
                    res = swap_rfull(x)
                    rand_total_rows += 1
                    rand_total_attempts += res.attempts
                    _bump_reason(rand_reasons, res.reason)

                    if res.ok:
                        ok_rand_here += 1
                        rand_total_ok += 1
                        assert rand_ok_full is not None
                        rand_ok_full[i, t_idx] = 1
                        y = res.text_out or x
                        if y == x:
                            unchanged_rand_ok_here += 1
                        rand_full_texts.append(y)
                    else:
                        assert rand_ok_full is not None
                        rand_ok_full[i, t_idx] = 0
                        rand_full_texts.append(x)

                    if rand_manifest_f is not None:
                        rand_manifest_f.write(
                            json.dumps(
                                {
                                    "species": s,
                                    "variant": "rand_full",
                                    "template_index": t_idx,
                                    "ok": bool(res.ok),
                                    "reason": res.reason,
                                    "attempts": int(res.attempts),
                                    "changed_token": res.changed_token,
                                    "replacement": res.replacement,
                                    "protected": s,
                                    "text_in": x,
                                    "text_out": res.text_out,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                for t_idx, x in enumerate(abbr_texts):
                    res = swap_rabbr(x)
                    rand_total_rows += 1
                    rand_total_attempts += res.attempts
                    _bump_reason(rand_reasons, res.reason)

                    if res.ok:
                        ok_rand_here += 1
                        rand_total_ok += 1
                        assert rand_ok_abbr is not None
                        rand_ok_abbr[i, t_idx] = 1
                        y = res.text_out or x
                        if y == x:
                            unchanged_rand_ok_here += 1
                        rand_abbr_texts.append(y)
                    else:
                        assert rand_ok_abbr is not None
                        rand_ok_abbr[i, t_idx] = 0
                        rand_abbr_texts.append(x)

                    if rand_manifest_f is not None:
                        rand_manifest_f.write(
                            json.dumps(
                                {
                                    "species": s,
                                    "variant": "rand_abbr",
                                    "template_index": t_idx,
                                    "ok": bool(res.ok),
                                    "reason": res.reason,
                                    "attempts": int(res.attempts),
                                    "changed_token": res.changed_token,
                                    "replacement": res.replacement,
                                    "protected": ab,
                                    "text_in": x,
                                    "text_out": res.text_out,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                rand_success_rate = ok_rand_here / float(denom) if denom else 0.0

                if unchanged_rand_ok_here > 0:
                    raise SystemExit(
                        f"[ERROR] random swap invariant violated (unchanged output accepted) for {s}: {unchanged_rand_ok_here}"
                    )

                if rand_success_rate < rand_success_min:
                    raise SystemExit(
                        f"[ERROR] random swap success rate too low for {s}: {rand_success_rate:.3f} < {rand_success_min:.3f}"
                    )

                ok_rf = rand_ok_full[i].astype(bool)
                ok_ra = rand_ok_abbr[i].astype(bool)
                rdup_full = 0.0
                rdup_abbr = 0.0
                if ok_rf.any():
                    rdup_full = 1.0 - (
                        len(set([rand_full_texts[k] for k in range(N) if ok_rf[k]]))
                        / float(ok_rf.sum())
                    )
                if ok_ra.any():
                    rdup_abbr = 1.0 - (
                        len(set([rand_abbr_texts[k] for k in range(N) if ok_ra[k]]))
                        / float(ok_ra.sum())
                    )
                print(
                    f"[RAND] {s}: success={rand_success_rate:.3f} dup_full={rdup_full:.3f} dup_abbr={rdup_abbr:.3f}"
                )

            # ---------- encode ----------
            zs_full[i] = (
                encode_sentences(
                    model, tok, full_texts, batch_size=batch_size, pooling=pooling
                )
                .cpu()
                .numpy()
            )
            zs_abbr[i] = (
                encode_sentences(
                    model, tok, abbr_texts, batch_size=batch_size, pooling=pooling
                )
                .cpu()
                .numpy()
            )
            zs_sfull[i] = (
                encode_sentences(
                    model, tok, syn_full_texts, batch_size=batch_size, pooling=pooling
                )
                .cpu()
                .numpy()
            )
            zs_sabbr[i] = (
                encode_sentences(
                    model, tok, syn_abbr_texts, batch_size=batch_size, pooling=pooling
                )
                .cpu()
                .numpy()
            )

            if rand_enabled:
                assert zs_rfull is not None and zs_rabbr is not None
                zs_rfull[i] = (
                    encode_sentences(
                        model,
                        tok,
                        rand_full_texts,
                        batch_size=batch_size,
                        pooling=pooling,
                    )
                    .cpu()
                    .numpy()
                )
                zs_rabbr[i] = (
                    encode_sentences(
                        model,
                        tok,
                        rand_abbr_texts,
                        batch_size=batch_size,
                        pooling=pooling,
                    )
                    .cpu()
                    .numpy()
                )

    finally:
        if syn_manifest_f is not None:
            syn_manifest_f.close()
        if rand_manifest_f is not None:
            rand_manifest_f.close()

    # ---------- global reports ----------
    syn_overall_success = (
        syn_total_ok / float(syn_total_rows) if syn_total_rows else 0.0
    )
    syn_avg_attempts = (
        syn_total_attempts / float(syn_total_rows) if syn_total_rows else 0.0
    )

    print(
        f"[SYN] overall_success={syn_overall_success:.3f} avg_attempts={syn_avg_attempts:.2f} total_rows={syn_total_rows}"
    )
    for k in sorted(syn_reasons.keys()):
        print(f"[SYN] reason[{k}]={syn_reasons[k]}")

    rand_overall_success = None
    rand_avg_attempts = None
    if rand_enabled:
        rand_overall_success = (
            rand_total_ok / float(rand_total_rows) if rand_total_rows else 0.0
        )
        rand_avg_attempts = (
            rand_total_attempts / float(rand_total_rows) if rand_total_rows else 0.0
        )
        print(
            f"[RAND] overall_success={rand_overall_success:.3f} avg_attempts={rand_avg_attempts:.2f} total_rows={rand_total_rows}"
        )
        for k in sorted(rand_reasons.keys()):
            print(f"[RAND] reason[{k}]={rand_reasons[k]}")

    meta = {
        "species": species,
        "placeholder": placeholder,
        "n_sentences": N,
        "pooling": pooling,
        "model_name": model_name,
        "synonym_swap": {
            "seed": syn_seed,
            "max_attempts": syn_max_attempts,
            "success_min": syn_success_min,
            "abort_on_protected_fail": bool(syn_abort_on_protected_fail),
            "write_manifest": bool(syn_write_manifest),
            "manifest_path": syn_manifest_path if syn_write_manifest else None,
            "overall_success": float(syn_overall_success),
            "avg_attempts": float(syn_avg_attempts),
            "reasons": syn_reasons,
        },
        "random_swap": {
            "enabled": bool(rand_enabled),
            "seed": rand_seed,
            "max_attempts": rand_max_attempts,
            "success_min": rand_success_min,
            "write_manifest": bool(rand_write_manifest),
            "manifest_path": rand_manifest_path
            if (rand_enabled and rand_write_manifest)
            else None,
            "wordlist_size": int(rand_wordlist_size),
            "overall_success": float(rand_overall_success)
            if rand_overall_success is not None
            else None,
            "avg_attempts": float(rand_avg_attempts)
            if rand_avg_attempts is not None
            else None,
            "reasons": rand_reasons if rand_enabled else {},
        },
    }

    save_kwargs = {
        "z_full": zs_full,
        "z_abbr": zs_abbr,
        "z_syn_full": zs_sfull,
        "z_syn_abbr": zs_sabbr,
        "syn_ok_full": syn_ok_full,
        "syn_ok_abbr": syn_ok_abbr,
        "meta": json.dumps(meta),
    }

    if rand_enabled:
        assert zs_rfull is not None and zs_rabbr is not None
        assert rand_ok_full is not None and rand_ok_abbr is not None
        save_kwargs.update(
            {
                "z_rand_full": zs_rfull,
                "z_rand_abbr": zs_rabbr,
                "rand_ok_full": rand_ok_full,
                "rand_ok_abbr": rand_ok_abbr,
            }
        )

    np.savez_compressed(out_npz, **save_kwargs)
    print(f"[SAVE] {out_npz}  (~{os.path.getsize(out_npz) / 1e6:.1f} MB)")
    if syn_write_manifest:
        print(f"[SAVE] {syn_manifest_path}")
    if rand_enabled and rand_write_manifest:
        print(f"[SAVE] {rand_manifest_path}")


def build_parser():
    repo_root = Path(__file__).resolve().parents[2]
    default_workdir = repo_root / "work"

    p = argparse.ArgumentParser(
        description="Encode all variants once and cache embeddings."
    )
    p.add_argument(
        "--workdir",
        type=str,
        default=str(default_workdir),
        help="Work directory root (default: ./work at repo root)",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Template dataset JSON (default: <workdir>/templates/cluster_all_examples_flat_42.json)",
    )
    p.add_argument(
        "--species",
        default=None,
        help="Species list text file (default: <workdir>/species/all_species.txt)",
    )
    p.add_argument(
        "--out-npz",
        default=None,
        help="Output NPZ path (default: <workdir>/cache/species_embeddings.npz)",
    )
    p.add_argument("--placeholder", default="Staphylococcus aureus")
    p.add_argument("--allow-multiple", action="store_true")
    p.add_argument("--model", default="dmis-lab/biobert-base-cased-v1.2")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument(
        "--pooling", choices=["mean_last2", "cls", "mean"], default="mean_last2"
    )

    # scale controls
    p.add_argument(
        "--max-species",
        type=int,
        default=0,
        help="Take first K species from the species list (0 means no limit)",
    )
    p.add_argument(
        "--sample-species",
        type=int,
        default=0,
        help="Deterministically sample K species from the species list (0 means no sampling)",
    )
    p.add_argument(
        "--sample-species-seed", type=int, default=0, help="Seed for --sample-species"
    )
    p.add_argument(
        "--allow-huge",
        action="store_true",
        help="Allow very large caches (may allocate tens/hundreds of GB)",
    )

    # synonym swap safeguards
    p.add_argument("--syn-seed", type=int, default=42)
    p.add_argument("--syn-max-attempts", type=int, default=250)
    p.add_argument("--syn-success-min", type=float, default=0.95)
    p.add_argument(
        "--syn-abort-on-protected-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--syn-write-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # random word swap baseline (wordfreq)
    p.add_argument(
        "--rand-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute the random-word control baseline (wordfreq).",
    )
    p.add_argument("--rand-seed", type=int, default=42)
    p.add_argument("--rand-max-attempts", type=int, default=250)
    p.add_argument("--rand-success-min", type=float, default=0.95)
    p.add_argument(
        "--rand-write-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--rand-wordlist-size",
        type=int,
        default=50_000,
        help="Number of top wordfreq English words to sample from",
    )

    return p


def main():
    a = build_parser().parse_args()
    workdir = Path(a.workdir)

    dataset_file = a.dataset or str(
        workdir / "templates" / "cluster_all_examples_flat_42.json"
    )
    species_file = a.species or str(workdir / "species" / "all_species.txt")
    out_npz = a.out_npz or str(workdir / "cache" / "species_embeddings.npz")

    build_and_save_cache(
        dataset_file=dataset_file,
        species_file=species_file,
        out_npz=out_npz,
        placeholder=a.placeholder,
        require_single=not a.allow_multiple,
        model_name=a.model,
        batch_size=a.batch_size,
        pooling=a.pooling,
        syn_seed=a.syn_seed,
        syn_max_attempts=a.syn_max_attempts,
        syn_success_min=a.syn_success_min,
        syn_abort_on_protected_fail=a.syn_abort_on_protected_fail,
        syn_write_manifest=a.syn_write_manifest,
        rand_enabled=a.rand_enabled,
        rand_seed=a.rand_seed,
        rand_max_attempts=a.rand_max_attempts,
        rand_success_min=a.rand_success_min,
        rand_write_manifest=a.rand_write_manifest,
        rand_wordlist_size=a.rand_wordlist_size,
        max_species=a.max_species,
        sample_species=a.sample_species,
        sample_species_seed=a.sample_species_seed,
        allow_huge=a.allow_huge,
    )


if __name__ == "__main__":
    main()
