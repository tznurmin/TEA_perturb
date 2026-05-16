# Taxonomic perturbations of biomedical sentence representations

This repository implements a perturbation pipeline for probing how biomedical sentence encoders respond to **taxonomic species name replacement**. Starting from curated sentence templates that contain exactly one species mention, the pipeline generates controlled variants by substituting that mention with alternative species names while keeping the surrounding context fixed.

To separate taxon-driven effects from non-specific sensitivity to lexical edits, two control perturbations are included that do not modify the species phrase: (i) a synonym substitution control (approximate meaning preservation) and (ii) a random word substitution control (uncontrolled semantic disruption). Embedding shifts induced by species substitution are compared to these controls across multiple random seeds.

Default model: `dmis-lab/biobert-base-cased-v1.2`. Other Hugging Face text encoders can be selected with `--model`; outputs are stored in model-scoped directories. The current release has been tested with BioBERT v1.2 and BioLinkBERT base.

## Results summary

Across 565 taxa and three random seeds, species substitution produces an embedding shift that exceeds the synonym control and is stable across seeds. The ordering of taxa by effect size is highly reproducible and the effect persists under common embedding post-processing.

Configuration:
- Seeds: 42, 43, 44
- Taxa: n = 565
- Distance metric: `one_minus_cos`
- Tested encoders: BioBERT v1.2, `dmis-lab/biobert-base-cased-v1.2`; BioLinkBERT base, `michiyasunaga/BioLinkBERT-base`

Post-processing:
- `none`: no post-processing
- `abtt`: all-but-the-top component removal
- `whiten`: whitening transform

Statistics:
- `fold`: (baseline + delta) / baseline  
  `delta` = species-replacement shift minus baseline-perturbation shift
- `Spearman`: cross-seed rank stability of per-taxon `delta`

### Baseline: synonym substitution

| encoder | method | median fold (FULL) | median fold (ABBR) | Spearman (FULL) | Spearman (ABBR) |
|---|---|---:|---:|---:|---:|
| BioBERT | none | 3.83x | 3.45x | 0.9889 | 0.9954 |
| BioBERT | abtt | 3.78x | 3.39x | 0.9896 | 0.9947 |
| BioBERT | whiten | 2.76x | 2.66x | 0.9794 | 0.9871 |
| BioLinkBERT | none | 1.62x | 2.45x | 0.9710 | 0.9506 |
| BioLinkBERT | abtt | 2.00x | 2.93x | 0.9815 | 0.9724 |
| BioLinkBERT | whiten | 1.55x | 2.09x | 0.9839 | 0.9658 |

### Baseline: random word substitution

| encoder | method | median fold (FULL) | median fold (ABBR) | Spearman (FULL) | Spearman (ABBR) |
|---|---|---:|---:|---:|---:|
| BioBERT | none | 2.24x | 2.02x | 0.9850 | 0.9939 |
| BioBERT | abtt | 2.29x | 2.06x | 0.9879 | 0.9934 |
| BioBERT | whiten | 1.91x | 1.84x | 0.9846 | 0.9895 |
| BioLinkBERT | none | 0.98x | 1.48x | 0.9602 | 0.9333 |
| BioLinkBERT | abtt | 1.28x | 1.88x | 0.9729 | 0.9592 |
| BioLinkBERT | whiten | 1.13x | 1.52x | 0.9873 | 0.9747 |

Related repositories:
- TEA (augmentation method targeting taxonomic names): https://github.com/tznurmin/TEA
- TEA_ft (fine-tuning experiments using taxonomic augmentation): https://github.com/tznurmin/TEA_ft

## Repository structure

- `tea_spec/src` - pipeline and reporting scripts
- `tea_spec/data` - UniProt species list
- `work` - runtime outputs; this directory is generated locally and is not tracked

## Installation

This is a script-oriented repository. Install the runtime requirements, then run the scripts from the repository root.

Install dependencies:

    pip install -r requirements.txt

The default pipeline requires WordNet for the synonym baseline.

Install WordNet once:

    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

The random-word baseline uses `wordfreq`. It can be disabled with `--no-rand`.

## Pipeline overview

The end-to-end run is orchestrated by `tea_spec/src/run_all.py`. Individual stages can be executed as standalone scripts.

Model-scoped cache and summary paths are created by `run_all.py` and `run_seed_sweep.py`. When running `embed_cache.py` or `summarize_from_cache.py` directly, pass `--out` and `--cache` explicitly if you want the same layout.

`run_all.py` downloads `TEA_curated_data` v1.1 into `work/TEA_curated_data` when the curated material is missing. That external dataset is separately licensed and is not covered by this repository's Apache-2.0 code licence; see the `TEA_curated_data` repository for its data and source-text licensing terms. The default v1.1 download is checksum-verified.

1) Work directory setup and external data checks
- `tea_spec/src/setup_workdir.py`  
  Creates `work` scaffolding and ensures required TEA curated material is present.

2) Species name list
- `tea_spec/src/build_species_list.py`  
  Builds `work/species/all_species.txt` from the bundled species list.
- `tea_spec/src/extract_species_in_corpus.py`  
  Filters to taxa observed in the curated corpus, writes `work/species/species_in_corpus.txt`.

3) Curated material to regularised dataset
- `tea_spec/src/build_regular_dataset.py`  
  Compiles curated annotations into `work/regular_<seed>.json`.

4) Template generation
- `tea_spec/src/generate_templates_from_curated.py`  
  Extracts sentences with a single taxon mention and writes template datasets under `work/templates`.

5) Perturbation, embedding cache, and per-seed summaries
- `tea_spec/src/run_seed_sweep.py`  
  Runs the sweep across seeds, post-processing methods, and the selected encoder model.
- `tea_spec/src/embed_cache.py`  
  Generates perturbations (taxon replacement plus controls), encodes all variants, and writes NPZ caches under `work/cache/models/<model>`.
- `tea_spec/src/summarize_from_cache.py`  
  Computes per-taxon deltas and confidence intervals and writes per-run summaries under `work/summaries_from_cache/models/<model>/seed<seed>/<method>`.

6) Reports
- `tea_spec/src/report_summary_statistics.py`  
  Prints compact cross-seed summary tables.
- `tea_spec/src/report_species_effect.py`  
  Reports the effect profile for a selected species.
- `tea_spec/src/report_seed_stability.py`  
  Computes cross-seed stability and signal-to-noise summaries.
- `tea_spec/src/report_shift_components.py`  
  Breaks down baseline shift and species-replacement delta components.

## Usage

Run the full pipeline with the default BioBERT encoder:

    python tea_spec/src/run_all.py --seeds 42,43,44

By default, `run_all.py` reuses existing derived input files under `work`. Pass `--force-build` to rebuild the species list, regularised dataset, templates, and in-corpus species list.

The built-in `TEA_curated_data` v1.1 download is pinned to SHA-256:

    bb811506fdaaf78a1fe9f1c4e0f3b89cefd75c65c70c421dbc309ec76946aec4

For a custom curated-data tarball, pass the matching checksum:

    python tea_spec/src/run_all.py --seeds 42,43,44 --tea-sha256 <sha256>

Run BioLinkBERT instead:

    python tea_spec/src/run_all.py --model michiyasunaga/BioLinkBERT-base --seeds 42,43,44

Print a compact cross-seed summary:

    python tea_spec/src/report_summary_statistics.py --workdir work --model michiyasunaga/BioLinkBERT-base --seeds 42,43,44 --metric one_minus_cos

Inspect a single species:

    python tea_spec/src/report_species_effect.py --workdir work --model michiyasunaga/BioLinkBERT-base --species "Staphylococcus aureus" --seeds 42,43,44

Generate a cross-seed stability report:

    python tea_spec/src/report_seed_stability.py --workdir work --model michiyasunaga/BioLinkBERT-base --seeds 42,43,44 --metric one_minus_cos

Inspect the contribution of baseline and taxonomic deltas:

    python tea_spec/src/report_shift_components.py --workdir work --model michiyasunaga/BioLinkBERT-base --seeds 42,43,44

## Outputs

All outputs are written under `work`:

- `work/species` - species lists and in-corpus extraction outputs
- `work/templates` - perturbation templates derived from curated text
- `work/cache/models/<model>` - embedding caches (`*.npz`)
- `work/summaries_from_cache/models/<model>` - per-run and per-seed summary tables

## License

Code: Apache License 2.0. See `LICENSE` and `NOTICE`.

Vendored UniProt organism list: CC BY 4.0. See `tea_spec/data/attribution.txt`.

Downloaded `TEA_curated_data` material is external and separately licensed.
