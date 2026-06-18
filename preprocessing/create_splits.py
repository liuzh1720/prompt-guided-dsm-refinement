"""
Generate train/eval split CSVs for all cities and mixed-training combinations.

HK uses an existing fixed split CSV.  Austin and Paris use random splits.
Output CSVs use relative paths from data_root.

Usage:
    python preprocessing/create_splits.py \
        --data-root data \
        --hk-split data/HK/fixed_evaluation_group/split.csv \
        --output-dir splits
"""
import argparse
import random
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Create train/eval split CSVs")
    p.add_argument("--data-root", required=True, help="Root data directory")
    p.add_argument("--hk-split", required=True, help="Existing HK fixed split CSV")
    p.add_argument("--output-dir", default="splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-ratio", type=float, default=0.20,
                   help="Eval ratio for Austin and Paris (default: 0.20)")
    return p.parse_args()


def extract_patch_id(name: str) -> str:
    """Extract numeric patch ID from filename or path."""
    stem = Path(name).stem
    nums = re.findall(r"\d+", stem)
    return nums[-1].zfill(6) if nums else stem


def list_tif_files(folder: Path):
    """List real .tif/.tiff files, excluding .aux.tif."""
    files = []
    for p in folder.glob("*"):
        if p.suffix.lower() in (".tif", ".tiff"):
            if p.name.lower().endswith(".aux.tif") or p.name.lower().endswith(".aux.tiff"):
                continue
            files.append(p)
    return sorted(files)


def collect_city_samples(city: str, data_root: Path, rgb_sub: str, dsm_sub: str, prompt_sub: str) -> pd.DataFrame:
    """Collect matched RGB/DSM/prompt samples for a city."""
    rgb_dir = data_root / rgb_sub
    dsm_dir = data_root / dsm_sub
    prompt_dir = data_root / prompt_sub

    for d, label in [(rgb_dir, "RGB"), (dsm_dir, "DSM"), (prompt_dir, "Prompt")]:
        if not d.exists():
            raise FileNotFoundError(f"[{city}] {label} dir not found: {d}")

    rgb_map = {}
    for p in list_tif_files(rgb_dir):
        pid = extract_patch_id(str(p))
        if pid not in rgb_map:
            rgb_map[pid] = p.relative_to(data_root)

    dsm_map = {}
    for p in list_tif_files(dsm_dir):
        pid = extract_patch_id(str(p))
        if pid not in dsm_map:
            dsm_map[pid] = p.relative_to(data_root)

    prompt_map = {}
    for p in list_tif_files(prompt_dir):
        pid = extract_patch_id(str(p))
        # Strip _prompt suffix if present
        stem = p.stem
        if stem.endswith("_prompt"):
            pid = extract_patch_id(stem[:-7])
        if pid not in prompt_map:
            prompt_map[pid] = p.relative_to(data_root)

    common = sorted(set(rgb_map) & set(dsm_map) & set(prompt_map))
    rows = []
    for pid in common:
        rows.append({
            "city": city,
            "patch_id": pid,
            "rgb_path": str(rgb_map[pid]).replace("\\", "/"),
            "dsm_path": str(dsm_map[pid]).replace("\\", "/"),
            "prompt_path": str(prompt_map[pid]).replace("\\", "/"),
        })
    df = pd.DataFrame(rows)
    print(f"[{city}] {len(df)} matched samples")
    return df


def load_hk_split(all_hk_df: pd.DataFrame, split_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse existing HK fixed split."""
    split_df = pd.read_csv(split_csv)
    split_df["split"] = split_df["split"].astype(str).str.lower().str.strip()
    train_ids = set(split_df[split_df["split"].isin(["training", "train"])]["patch_id"])
    eval_ids = set(split_df[split_df["split"].isin(["evaluation", "eval", "test", "val", "validation"])]["patch_id"])

    all_hk_ids = set(all_hk_df["patch_id"])
    hk_train = all_hk_df[all_hk_df["patch_id"].isin(train_ids & all_hk_ids)].copy()
    hk_eval = all_hk_df[all_hk_df["patch_id"].isin(eval_ids & all_hk_ids)].copy()
    hk_train["split"] = "training"
    hk_eval["split"] = "evaluation"
    print(f"[HK] train: {len(hk_train)}, eval: {len(hk_eval)} (reused existing split)")
    return hk_train, hk_eval


def random_split_city(df: pd.DataFrame, eval_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly split a city into train and eval."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_eval = max(1, int(round(len(df) * eval_ratio)))
    eval_df = df.iloc[:n_eval].copy()
    train_df = df.iloc[n_eval:].copy()
    train_df["split"] = "training"
    eval_df["split"] = "evaluation"
    return train_df, eval_df


def check_no_overlap(train_df, eval_df, label):
    train_keys = set(zip(train_df["city"], train_df["patch_id"]))
    eval_keys = set(zip(eval_df["city"], eval_df["patch_id"]))
    overlap = train_keys & eval_keys
    if overlap:
        raise RuntimeError(f"[{label}] Train/eval overlap: {list(overlap)[:10]}")


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved: {path} ({len(df)} rows)")


def main():
    args = parse_args()
    random.seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # City config: {city: (rgb_subdir, dsm_subdir, prompt_subdir)}
    city_config = {
        "HK":       ("HK/rgb",       "HK/dsm",       "HK/prompt_dsm_30m"),
        "Austin":   ("Austin/rgb",   "Austin/dsm",   "Austin/prompt_dsm_30m"),
        "Paris":    ("Paris/rgb",    "Paris/dsm",    "Paris/prompt_dsm_30m"),
    }

    city_train, city_eval = {}, {}

    for city, (rgb_s, dsm_s, prompt_s) in city_config.items():
        print(f"\n--- {city} ---")
        df = collect_city_samples(city, data_root, rgb_s, dsm_s, prompt_s)

        if city == "HK":
            hk_split = Path(args.hk_split)
            if not hk_split.exists():
                raise FileNotFoundError(f"HK split CSV not found: {hk_split}")
            train_df, eval_df = load_hk_split(df, hk_split)
        else:
            train_df, eval_df = random_split_city(df, args.eval_ratio, args.seed)
            print(f"[{city}] random split: train={len(train_df)}, eval={len(eval_df)}")

        check_no_overlap(train_df, eval_df, city)
        city_train[city] = train_df
        city_eval[city] = eval_df
        save_csv(train_df, output_dir / f"{city}_train.csv")
        save_csv(eval_df, output_dir / f"{city}_eval.csv")

    # Combined full split
    all_df = pd.concat(list(city_train.values()) + list(city_eval.values()), ignore_index=True)
    save_csv(all_df, output_dir / "all_city_fixed_split.csv")

    # Mixed training groups
    mix_pairs = [("HK", "Austin"), ("HK", "Paris"), ("Austin", "Paris")]
    for a, b in mix_pairs:
        name = f"{a}_{b}"
        mix_train = pd.concat([city_train[a], city_train[b]], ignore_index=True)
        save_csv(mix_train, output_dir / f"{name}_train.csv")

    # Summary
    rows = []
    for c in city_config:
        rows.append({"group": c, "train": len(city_train[c]), "eval": len(city_eval[c]),
                      "total": len(city_train[c]) + len(city_eval[c])})
    for a, b in mix_pairs:
        name = f"{a}_{b}"
        rows.append({"group": name, "train": len(city_train[a]) + len(city_train[b]),
                      "eval": sum(len(city_eval[x]) for x in city_config), "total": ""})
    pd.DataFrame(rows).to_csv(output_dir / "split_summary.csv", index=False)
    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
