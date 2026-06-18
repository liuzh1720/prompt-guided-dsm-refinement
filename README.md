# Prompt-Guided DSM Refinement

Official code for the master's thesis:

**"Prompt-guided DSM Refinement Using High-resolution RGB Imagery and Coarse DSM Prompts"**

Zhenghao Liu, KTH Royal Institute of Technology, 2026

## Overview

This repository implements prompt-guided Digital Surface Model (DSM) refinement
using high-resolution RGB imagery and coarse DSM prompts.  The model is adapted from
a PromptDA-style framework and tested across three urban regions: Hong Kong, Austin,
and Paris.

## Repository Structure

```
├── configs/              YAML configuration files
├── datasets/             CSV-based dataset loader
├── preprocessing/        Coarse prompt generation and split creation
├── training/             Training scripts (HK-only and mixed-region)
├── evaluation/
│   ├── whole_patch/      Whole-patch evaluation
│   ├── building_region/  Building-region evaluation
│   └── baseline/         PromptDA baseline evaluation
├── splits/               Train/eval split CSVs (relative paths)
├── results/              Evaluation metrics and metadata
├── docs/                 Documentation
├── third_party/          External dependency notes
└── examples/             Example data
```

## Quick Start

### 1. Install dependencies

```bash
conda env create -f environment.yml
conda activate dsm-refine
```

### 2. Install PromptDA

Clone PromptDA and download the pretrained checkpoint.
See [third_party/README.md](third_party/README.md).

### 3. Prepare data

Download and organize data following [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md).
Generate coarse DSM prompts and split CSVs:

```bash
python preprocessing/generate_coarse_prompt.py \
    --input-dir data/HK/dsm --output-dir data/HK/prompt_dsm_30m

python preprocessing/create_splits.py \
    --data-root data --hk-split data/HK/fixed_evaluation_group/split.csv
```

### 4. Train

```bash
# HK-only
python training/train_hk_only.py --config configs/hk_only.yaml \
    --promptda-path /path/to/PromptDA --data-root data

# Mixed-region
python training/train_mixed.py --config configs/hk_austin.yaml \
    --promptda-path /path/to/PromptDA --data-root data
```

### 5. Evaluate

```bash
# Whole-patch
python evaluation/whole_patch/evaluate.py --config configs/eval_whole_patch.yaml \
    --checkpoint training_outputs/hk_only/best_model.pth \
    --eval-csv splits/HK_eval.csv --output-dir results/hk_only/hk \
    --promptda-path /path/to/PromptDA

# Building-region
python evaluation/building_region/evaluate.py --config configs/eval_building_region.yaml \
    --checkpoint training_outputs/hk_only/best_model.pth \
    --eval-csv splits/HK_eval.csv --footprint-dir data/HK/building_footprints \
    --output-dir results/building_region/hk --promptda-path /path/to/PromptDA

# PromptDA baseline
python evaluation/baseline/evaluate_promptda.py --config configs/eval_whole_patch.yaml \
    --eval-csv splits/HK_eval.csv --output-dir results/baseline/hk \
    --promptda-path /path/to/PromptDA
```

## Expected Split Counts

| City   | Train | Eval | Total |
|--------|-------|------|-------|
| HK     | 139   | 72   | 211   |
| Austin | 311   | 78   | 389   |
| Paris  | 136   | 34   | 170   |

## Citation

```bibtex
@mastersthesis{liu2026promptdsm,
  title  = {Prompt-guided DSM Refinement Using High-resolution RGB Imagery and Coarse DSM Prompts},
  author = {Liu, Zhenghao},
  school = {KTH Royal Institute of Technology},
  year   = {2026},
}
```

## License

Code in this repository is provided for research reproducibility.
Data sources have their own licenses; see [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md).
PromptDA is an external dependency with its own license terms.
