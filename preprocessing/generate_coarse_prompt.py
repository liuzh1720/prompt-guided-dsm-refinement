"""
Generate coarse DSM prompts from reference DSM patches.
Downsamples each DSM to a target spatial resolution and upsamples back
to the original grid, producing a smoothed height field.

Usage:
    python preprocessing/generate_coarse_prompt.py \
        --input-dir data/HK/dsm \
        --output-dir data/HK/prompt_dsm_30m \
        --target-resolution 30 \
        --patch-size 1036 \
        --pixel-size 0.6
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import rasterio
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Generate coarse DSM prompts")
    p.add_argument("--input-dir", required=True, help="Directory with reference DSM patches")
    p.add_argument("--output-dir", required=True, help="Output directory for prompt DSMs")
    p.add_argument("--target-resolution", type=float, default=30.0,
                   help="Target spatial resolution in metres (default: 30)")
    p.add_argument("--patch-size", type=int, default=1036,
                   help="Patch size in pixels (default: 1036)")
    p.add_argument("--pixel-size", type=float, default=0.6,
                   help="Pixel size in metres (default: 0.6)")
    p.add_argument("--nodata", type=float, default=-9999)
    p.add_argument("--suffix", default="_prompt", help="Suffix appended to output filenames")
    return p.parse_args()


def fill_nodata(dsm, nodata_value):
    """Inpaint nodata regions for prompt construction."""
    invalid = (~np.isfinite(dsm)) | (dsm <= nodata_value + 1)
    if not np.any(invalid):
        return dsm.copy(), invalid
    valid = ~invalid
    if not np.any(valid):
        return dsm.copy(), invalid
    valid_vals = dsm[valid]
    dmin, dmax = valid_vals.min(), valid_vals.max()
    if dmax - dmin < 1e-6:
        return np.where(valid, dsm, dmin).astype(np.float32), invalid
    dsm_norm = np.zeros_like(dsm, dtype=np.float32)
    dsm_norm[valid] = (dsm[valid] - dmin) / (dmax - dmin)
    dsm_8u = np.clip(dsm_norm * 255, 0, 255).astype(np.uint8)
    mask_8u = invalid.astype(np.uint8) * 255
    filled_8u = cv2.inpaint(dsm_8u, mask_8u, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    filled_norm = filled_8u.astype(np.float32) / 255.0
    return (filled_norm * (dmax - dmin) + dmin).astype(np.float32), invalid


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coarse_size = round((args.patch_size * args.pixel_size) / args.target_resolution)
    print(f"Target resolution: {args.target_resolution} m")
    print(f"Coarse grid size: {coarse_size} x {coarse_size}")

    dsm_files = sorted(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")))
    print(f"Found {len(dsm_files)} DSM files")

    for dsm_path in tqdm(dsm_files, desc="Generating prompts"):
        with rasterio.open(dsm_path) as src:
            dsm = src.read(1).astype(np.float32)
            profile = src.profile.copy()
            src_nodata = src.nodata

        h, w = dsm.shape
        nodata_val = args.nodata if src_nodata is None else src_nodata

        dsm_filled, invalid = fill_nodata(dsm, nodata_val)
        valid = ~invalid
        if not np.any(valid):
            print(f"Warning: all pixels invalid in {dsm_path.name}")
            continue

        valid_vals = dsm_filled[valid]
        dmin, dmax = valid_vals.min(), valid_vals.max()

        if dmax - dmin < 1e-6:
            prompt = dsm_filled.copy()
        else:
            dsm_norm = (dsm_filled - dmin) / (dmax - dmin)
            dsm_coarse = cv2.resize(dsm_norm, (coarse_size, coarse_size),
                                    interpolation=cv2.INTER_AREA)
            prompt_norm = cv2.resize(dsm_coarse, (w, h), interpolation=cv2.INTER_LINEAR)
            prompt = prompt_norm * (dmax - dmin) + dmin

        profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=None)

        stem = dsm_path.stem
        out_name = f"{stem}{args.suffix}.tif"
        out_path = output_dir / out_name

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(prompt.astype(np.float32), 1)

    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
