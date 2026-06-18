"""
Building-region evaluation for a trained model.
Evaluates DSM predictions only within building footprint masks.
Uses eval CSV for patch selection and per-patch building footprint files.

Usage:
    python evaluation/building_region/evaluate.py \
        --config configs/eval_building_region.yaml \
        --checkpoint path/to/best_model.pth \
        --eval-csv splits/HK_eval.csv \
        --footprint-dir data/HK/building_footprints \
        --output-dir results/building_region/hk
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
import yaml
from rasterio.features import rasterize
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets import CSVPromptDataset


def parse_args():
    p = argparse.ArgumentParser(description="Building-region evaluation")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-csv", required=True)
    p.add_argument("--footprint-dir", required=True, help="Dir with per-patch building footprint files")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--promptda-path", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default="auto")
    p.add_argument("--min-building-pixel-ratio", type=float, default=0.01)
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


def find_footprint_file(patch_name: str, fp_dir: Path) -> Path | None:
    """Find building footprint file for a patch by name."""
    for ext in [".geojson", ".gpkg", ".shp"]:
        for pattern in [f"{patch_name}_buildings{ext}", f"{patch_name}{ext}"]:
            p = fp_dir / pattern
            if p.exists():
                return p
    return None


def rasterize_building_mask(fp_path: Path, ref_raster_path: str) -> np.ndarray | None:
    """
    Rasterize building footprints to match the reference DSM grid.

    Returns a boolean mask with the same shape as the reference raster,
    or None if no valid building geometries are found.
    """
    gdf = gpd.read_file(fp_path)
    if gdf.empty:
        return None
    with rasterio.open(ref_raster_path) as src:
        transform = src.transform
        shape = (src.height, src.width)
        crs_r = src.crs
    if gdf.crs is None:
        return None
    if gdf.crs != crs_r:
        gdf = gdf.to_crs(crs_r)
    geoms = [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        return None
    mask = rasterize(geoms, out_shape=shape, transform=transform, fill=0, dtype=np.uint8)
    return mask.astype(bool)


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
    fp_dir = Path(args.footprint_dir)
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

    # Model
    model = PromptDA(encoder="vitl",
                     ckpt_path=str(promptda_path / "checkpoints" / "model.ckpt")).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Metrics
    all_mae, all_rmse, all_r2, all_le90, all_bpr = [], [], [], [], []
    skipped_nf, skipped_nb, skipped_tf, saved = 0, 0, 0, 0

    pred_dir = output_dir / "predictions"
    if args.save_predictions:
        pred_dir.mkdir(exist_ok=True)

    with open(output_dir / "building_metrics_per_patch.csv", "w", newline="") as f:
        csv.writer(f).writerow([
            "name", "city", "mae_building", "rmse_building", "r2_building",
            "le90_building", "building_valid_ratio", "building_pixel_ratio",
            "building_pixel_count"
        ])

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            prompt = batch["prompt_depth"].to(device)
            gt = batch["gt_dsm"].to(device)

            # Get patch name (match footprint filename)
            pid = batch["patch_id"]
            pid_str = str(pid[0]) if isinstance(pid, (list, tuple)) else str(pid)
            city = batch["city"]
            city_str = str(city[0]) if isinstance(city, (list, tuple)) else str(city)

            # Try multiple name formats for footprint matching
            for name_fmt in [pid_str, f"{city_str}_{pid_str}"]:
                fp_path = find_footprint_file(name_fmt, fp_dir)
                if fp_path:
                    break

            if fp_path is None:
                skipped_nf += 1
                continue

            # Resolve reference raster path
            dsm_path = batch["dsm_path"]
            dsm_str = str(dsm_path[0]) if isinstance(dsm_path, (list, tuple)) else str(dsm_path)
            ref_path = str(Path(args.data_root) / dsm_str)

            bmask = rasterize_building_mask(fp_path, ref_path)
            if bmask is None or bmask.sum() == 0:
                skipped_nb += 1
                continue

            bpr = float(bmask.mean())
            if bpr < args.min_building_pixel_ratio:
                skipped_tf += 1
                continue

            pred = model(rgb, prompt)
            gt_np = gt[0, 0].cpu().numpy()
            pred_np = pred[0, 0].cpu().numpy()
            gvm = compute_valid_mask(gt_np)
            vm = gvm & bmask

            if vm.sum() == 0:
                skipped_nb += 1
                continue

            m = compute_metrics(pred_np, gt_np, vm)
            if m is None:
                continue

            all_mae.append(m["mae"]); all_rmse.append(m["rmse"])
            all_le90.append(m["le90"]); all_bpr.append(bpr)
            if np.isfinite(m["r2"]):
                all_r2.append(m["r2"])

            patch_name = f"{city_str}_{pid_str}"
            with open(output_dir / "building_metrics_per_patch.csv", "a", newline="") as f:
                csv.writer(f).writerow([
                    patch_name, city_str, m["mae"], m["rmse"], m["r2"],
                    m["le90"], m["valid_ratio"], bpr, int(bmask.sum())
                ])

            if args.save_predictions:
                save_geotiff(pred_np, ref_path, str(pred_dir / f"{patch_name}_pred.tif"))

            saved += 1

    # Summary
    summaries = {
        "mae_building": safe_summary(all_mae),
        "rmse_building": safe_summary(all_rmse),
        "r2_building": safe_summary(all_r2),
        "le90_building": safe_summary(all_le90),
        "building_pixel_ratio": safe_summary(all_bpr),
    }

    with open(output_dir / "building_metrics_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "count", "median", "nmad_or_std", "mean", "std", "report_format"])
        fmt_map = {
            "mae_building": "median ± NMAD",
            "rmse_building": "median ± std",
            "r2_building": "median ± std",
            "le90_building": "median",
            "building_pixel_ratio": "",
        }
        for k, s in summaries.items():
            spread = s["nmad"] if "NMAD" in fmt_map.get(k, "") else s["std"]
            w.writerow([k, s["count"], s["median"], spread, s["mean"], s["std"], fmt_map.get(k, "")])

    print(f"\nBuilding MAE: {summaries['mae_building']['median']:.2f} ± {summaries['mae_building']['nmad']:.2f}")
    print(f"Building R²: {summaries['r2_building']['median']:.3f}")
    print(f"Patches: {saved}  (skipped: no_fp={skipped_nf} no_bld={skipped_nb} too_few={skipped_tf})")

    # Metadata
    meta = {
        "checkpoint": str(args.checkpoint),
        "eval_csv": str(eval_csv),
        "footprint_dir": str(fp_dir),
        "patch_count": saved,
        "skipped_no_footprint": skipped_nf,
        "skipped_no_building": skipped_nb,
        "skipped_low_ratio": skipped_tf,
        "min_building_pixel_ratio": args.min_building_pixel_ratio,
        "device": device,
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    dataset.df.to_csv(output_dir / "used_eval_split.csv", index=False)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
