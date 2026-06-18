"""
PromptDA baseline evaluation.
Evaluates the official PromptDA model (no fine-tuning) on eval patches.
Uses the same eval CSVs, metrics, and nodata mask as the proposed model.

Usage:
    python evaluation/baseline/evaluate_promptda.py \
        --config configs/eval_whole_patch.yaml \
        --eval-csv splits/HK_eval.csv \
        --output-dir results/baseline/hk
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets import CSVPromptDataset


# Reuse the same helper functions as whole-patch evaluate.py
def parse_args():
    p = argparse.ArgumentParser(description="PromptDA baseline evaluation")
    p.add_argument("--config", required=True)
    p.add_argument("--eval-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--promptda-path", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default="auto")
    p.add_argument("--save-predictions", action="store_true")
    return p.parse_args()


def get_device(pref):
    if pref == "cuda": return "cuda"
    if pref == "cpu": return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_valid_mask(gt, nodata=-9990):
    return gt > nodata


def compute_metrics(pred_np, gt_np, valid_mask):
    if valid_mask.sum() == 0:
        return None
    p, g = pred_np[valid_mask], gt_np[valid_mask]
    diff = p - g
    ae = np.abs(diff)
    mae = float(np.mean(ae))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    le90 = float(np.percentile(ae, 90))
    vr = float(valid_mask.mean())
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((g - np.mean(g)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2, "le90": le90, "valid_ratio": vr}


def nmad(vals):
    a = np.array(vals, dtype=np.float32)
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if a.size else np.nan


def safe_summary(vals):
    a = np.array(vals, dtype=np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"count": 0, "median": np.nan, "nmad": np.nan, "mean": np.nan, "std": np.nan}
    return {"count": int(a.size), "median": float(np.median(a)),
            "nmad": nmad(a), "mean": float(np.mean(a)), "std": float(np.std(a))}


def save_geotiff(arr, ref_path, save_path, nodata=None):
    with rasterio.open(ref_path) as src:
        p = src.profile.copy()
    p.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=nodata)
    with rasterio.open(save_path, "w", **p) as dst:
        dst.write(arr.astype(np.float32), 1)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    print(f"Device: {device}")

    # PromptDA
    promptda_path = Path(args.promptda_path)
    sys.path.insert(0, str(promptda_path))
    from promptda.promptda import PromptDA

    # Dataset
    eval_csv = Path(args.eval_csv)
    dataset = CSVPromptDataset(eval_csv, data_root=Path(args.data_root))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Eval samples: {len(dataset)}")

    # Model (baseline = pretrained, no fine-tuning)
    model = PromptDA(encoder="vitl",
                     ckpt_path=str(promptda_path / "checkpoints" / "model.ckpt")).to(device)
    model.eval()
    print("Loaded PromptDA baseline (no fine-tuning)")

    pred_dir = output_dir / "predictions"
    if args.save_predictions:
        pred_dir.mkdir(exist_ok=True)

    all_mae, all_rmse, all_r2, all_le90, all_vr = [], [], [], [], []

    with open(output_dir / "metrics_per_patch.csv", "w", newline="") as f:
        csv.writer(f).writerow(["name", "city", "mae", "rmse", "r2", "le90", "valid_ratio"])

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            prompt = batch["prompt_depth"].to(device)
            gt = batch["gt_dsm"].to(device)

            pred = model(rgb, prompt)

            gt_np = gt[0, 0].cpu().numpy()
            pred_np = pred[0, 0].cpu().numpy()
            vm = compute_valid_mask(gt_np)
            if vm.sum() == 0:
                continue

            m = compute_metrics(pred_np, gt_np, vm)
            if m is None:
                continue

            pid = batch.get("patch_id", [""])
            city = batch.get("city", [""])
            pid_str = str(pid[0]) if isinstance(pid, (list, tuple)) else str(pid)
            city_str = str(city[0]) if isinstance(city, (list, tuple)) else str(city)
            name = f"{city_str}_{pid_str}"

            all_mae.append(m["mae"]); all_rmse.append(m["rmse"]); all_le90.append(m["le90"])
            if np.isfinite(m["r2"]):
                all_r2.append(m["r2"])
            all_vr.append(m["valid_ratio"])

            with open(output_dir / "metrics_per_patch.csv", "a", newline="") as f:
                csv.writer(f).writerow([name, city_str, m["mae"], m["rmse"], m["r2"], m["le90"], m["valid_ratio"]])

            if args.save_predictions:
                dsm_path = batch["dsm_path"]
                dsm_str = str(dsm_path[0]) if isinstance(dsm_path, (list, tuple)) else str(dsm_path)
                ref = str(Path(args.data_root) / dsm_str)
                save_geotiff(pred_np, ref, str(pred_dir / f"{name}_pred.tif"))

    # Summary
    summary = {
        "mae": safe_summary(all_mae), "rmse": safe_summary(all_rmse),
        "r2": safe_summary(all_r2), "le90": safe_summary(all_le90),
        "valid_ratio": safe_summary(all_vr),
    }

    with open(output_dir / "metrics_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "count", "median", "nmad_or_std", "mean", "std", "report_format"])
        fmt = {"mae": "median ± NMAD", "rmse": "median ± std",
               "r2": "median ± std", "le90": "median", "valid_ratio": ""}
        for k, s in summary.items():
            spread = s["nmad"] if "NMAD" in fmt.get(k, "") else s["std"]
            w.writerow([k, s["count"], s["median"], spread, s["mean"], s["std"], fmt.get(k, "")])

    print(f"\nBaseline MAE: {summary['mae']['median']:.2f} ± {summary['mae']['nmad']:.2f}")
    print(f"Baseline RMSE: {summary['rmse']['median']:.2f}")
    print(f"Baseline R²: {summary['r2']['median']:.3f}")
    print(f"Patches: {summary['mae']['count']}")

    meta = {
        "model": "promptda_baseline",
        "promptda_path": str(promptda_path),
        "eval_csv": str(eval_csv),
        "patch_count": summary["mae"]["count"],
        "device": device,
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    dataset.df.to_csv(output_dir / "used_eval_split.csv", index=False)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
