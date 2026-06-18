# Reproduction Guide

## Prerequisites

1. Conda environment (see `docs/ENVIRONMENT.md`)
2. PromptDA cloned and checkpoint downloaded (see `third_party/README.md`)
3. Data prepared (see `docs/DATA_PREPARATION.md`)
4. Split CSVs generated:
   ```bash
   python preprocessing/create_splits.py --data-root data --hk-split data/HK/fixed_evaluation_group/split.csv
   ```

## Step 1: HK-only Training

```bash
python training/train_hk_only.py \
    --config configs/hk_only.yaml \
    --promptda-path /path/to/PromptDA \
    --data-root data
```

Expected: ~34 epochs with early stopping.
Output: `training_outputs/hk_only/best_model.pth`

## Step 2: Mixed-region Training

Run each configuration separately:

```bash
python training/train_mixed.py --config configs/hk_austin.yaml --promptda-path /path/to/PromptDA --data-root data
python training/train_mixed.py --config configs/hk_paris.yaml --promptda-path /path/to/PromptDA --data-root data
python training/train_mixed.py --config configs/austin_paris.yaml --promptda-path /path/to/PromptDA --data-root data
```

## Step 3: Whole-patch Evaluation

```bash
# HK-only model on all three cities
python evaluation/whole_patch/evaluate.py \
    --config configs/eval_whole_patch.yaml \
    --checkpoint training_outputs/hk_only/best_model.pth \
    --eval-csv splits/HK_eval.csv --output-dir results/hk_only/hk \
    --promptda-path /path/to/PromptDA

python evaluation/whole_patch/evaluate.py \
    --config configs/eval_whole_patch.yaml \
    --checkpoint training_outputs/hk_only/best_model.pth \
    --eval-csv splits/Austin_eval.csv --output-dir results/hk_only/austin \
    --promptda-path /path/to/PromptDA

python evaluation/whole_patch/evaluate.py \
    --config configs/eval_whole_patch.yaml \
    --checkpoint training_outputs/hk_only/best_model.pth \
    --eval-csv splits/Paris_eval.csv --output-dir results/hk_only/paris \
    --promptda-path /path/to/PromptDA
```

## Step 4: PromptDA Baseline Evaluation

```bash
python evaluation/baseline/evaluate_promptda.py \
    --config configs/eval_whole_patch.yaml \
    --eval-csv splits/HK_eval.csv --output-dir results/baseline/hk \
    --promptda-path /path/to/PromptDA
```

## Step 5: Building-region Evaluation

```bash
python evaluation/building_region/evaluate.py \
    --config configs/eval_building_region.yaml \
    --checkpoint training_outputs/hk_only/best_model.pth \
    --eval-csv splits/HK_eval.csv \
    --footprint-dir data/HK/building_footprints \
    --output-dir results/building_region/hk \
    --promptda-path /path/to/PromptDA
```

## Expected Outputs

Each evaluation produces:
- `metrics_per_patch.csv` — per-patch metrics
- `metrics_summary.csv` — aggregated statistics (MAE: median±NMAD, RMSE: median±std, R²: median±std, LE90: median)
- `run_metadata.json` — experiment provenance
- `used_eval_split.csv` — copy of the eval split used

See thesis Tables 4.2–4.5 for expected metric values.
