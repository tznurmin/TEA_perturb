# Changelog

## v1.0.0 - 2026-05-16

Stable public release of the TEA perturbation pipeline.

- Runs controlled species-name perturbation experiments over curated TEA templates
- Compares taxonomic replacement shifts against synonym and random-word controls
- Supports BioBERT by default and accepts alternate Hugging Face encoder models via `--model`
- Writes model-scoped caches and summaries under `work/cache/models/<model>` and `work/summaries_from_cache/models/<model>`
- Includes reporting scripts for summary statistics, per-species effects, seed stability, and shift components
- Licensed under Apache License 2.0
- Fixes model-slug resolution in reporting commands for model-scoped outputs
- Fixes template filtering so ordinary sentence-initial phrases are not rejected as unknown species
- Exposes curated-data checksum verification through `run_all.py --tea-sha256`
- Makes `run_seed_sweep.py --methods` control the downstream comparison inputs
- Makes `run_seed_sweep.py --no-rand` produce synonym-only comparisons and reject incompatible random-only comparison requests
- Makes swap-manifest inspection find model-scoped cache manifests by default
