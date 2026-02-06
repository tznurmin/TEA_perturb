# Taxonomic perturbations of biomedical sentence representations

This repository implements a perturbation pipeline for probing how biomedical sentence encoders respond to **taxonomic species name replacement**. Starting from curated sentence templates that contain exactly one species mention, the pipeline generates controlled variants by substituting that mention with alternative species names while keeping the surrounding context fixed.

To separate taxon-driven effects from non-specific sensitivity to lexical edits, two control perturbations are included that do not modify the species phrase: (i) a synonym substitution control (approximate meaning preservation) and (ii) a random word substitution control (uncontrolled semantic disruption). Embedding shifts induced by species substitution are compared to these controls across multiple random seeds.

Embedding model: BioBERT v1.2 (cased), `dmis-lab/biobert-base-cased-v1.2`.

## Results summary

Across 565 taxa and three random seeds, species substitution produces an embedding shift that exceeds both lexical controls and is stable across seeds. The ordering of taxa by effect size is highly reproducible (Spearman approximately 0.98 to 0.995) and the effect persists under common embedding post-processing.

Configuration:
- Seeds: 42, 43, 44
- Taxa: n = 565
- Distance metric: `one_minus_cos`

Post-processing:
- `none`: no post-processing
- `abtt`: all-but-the-top component removal
- `whiten`: whitening transform

Statistics:
- `fold`: (baseline + delta) / baseline  
  `delta` = species-replacement shift minus baseline-perturbation shift
- `Spearman`: cross-seed rank stability of per-taxon `delta`

### Baseline: synonym substitution

| method | median fold (FULL) | median fold (ABBR) | Spearman (FULL) | Spearman (ABBR) |
|---|---:|---:|---:|---:|
| none | 3.83x | 3.45x | 0.9889 | 0.9954 |
| abtt | 3.78x | 3.39x | 0.9896 | 0.9947 |
| whiten | 2.76x | 2.66x | 0.9794 | 0.9871 |

### Baseline: random word substitution

| method | median fold (FULL) | median fold (ABBR) | Spearman (FULL) | Spearman (ABBR) |
|---|---:|---:|---:|---:|
| none | 2.24x | 2.02x | 0.9850 | 0.9939 |
| abtt | 2.29x | 2.06x | 0.9879 | 0.9934 |
| whiten | 1.91x | 1.84x | 0.9846 | 0.9895 |

Related repositories:
- TEA (augmentation method targeting taxonomic names): https://github.com/tznurmin/TEA
- TEA_ft (fine-tuning experiments using taxonomic augmentation): https://github.com/tznurmin/TEA_ft

## Repository structure

- `tea_spec/src` - pipeline and reporting scripts
- `tea_spec/data` - UniProt species list
- `work` - runtime outputs (approximately 9 GB for three seeds)

## Installation

Install dependencies:

    pip install -r requirements.txt

Synonym control uses WordNet. If WordNet is not available, the synonym baseline is skipped.

Install WordNet once:

    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

## Pipeline overview

The end-to-end run is orchestrated by `tea_spec/src/run_all.py`. Individual stages can be executed as standalone scripts.

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
  Runs the sweep across seeds and post-processing methods.
- `tea_spec/src/embed_cache.py`  
  Generates perturbations (taxon replacement plus controls), encodes all variants, and writes NPZ caches under `work/cache`.
- `tea_spec/src/summarize_from_cache.py`  
  Computes per-taxon deltas and confidence intervals and writes per-run summaries under `work/summaries_from_cache/seed<seed>/<method>`.

## Usage

Run the full pipeline:

    python tea_spec/src/run_all.py --seeds 42,43,44

Print a compact cross-seed summary:

    python tea_spec/src/report_summary_statistics.py --workdir work --seeds 42,43,44 --metric one_minus_cos

Inspect a single species:

    python tea_spec/src/report_species_effect.py --workdir work --species "Staphylococcus aureus" --seeds 42,43,44

Generate a cross-seed stability report:

    python tea_spec/src/report_seed_stability.py --workdir work --seeds 42,43,44 --metric one_minus_cos

## Outputs

All outputs are written under `work`:

- `work/species` - species lists and in-corpus extraction outputs
- `work/templates` - perturbation templates derived from curated text
- `work/cache` - embedding caches (`*.npz`)
- `work/summaries_from_cache` - per-run and per-seed summary tables
