from pathlib import Path
import re
import subprocess
import shutil
import requests
import rasterio
import numpy as np
from tqdm import tqdm


# ============================================================
# 1. Paths
# ============================================================

# Paths — update these before running
MNS_LINK_FILE = Path("data/Paris/Liens_de_telechargement.txt")  # file with MNS download URLs
OUT_DIR = Path("data/Paris")  # output root for Paris data

MNS_DIR = OUT_DIR / "mns_raw"
WORK_DIR = OUT_DIR / "work"

# 已经生成好的中间文件
RGB_0P6_PATH = WORK_DIR / "rgb_0p6.tif"
DSM_0P6_PATH = WORK_DIR / "dsm_0p6.tif"

RGB_PATCH_DIR = OUT_DIR / "patches" / "rgb"
DSM_PATCH_DIR = OUT_DIR / "patches" / "dsm"

for d in [MNS_DIR, WORK_DIR, RGB_PATCH_DIR, DSM_PATCH_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Area
# ============================================================

X_MIN = 648000
X_MAX = 659000
Y_MIN = 6858000
Y_MAX = 6866000

TARGET_RES = 0.6
PATCH_SIZE = 1036
STRIDE = 1036
CRS = "EPSG:2154"


# ============================================================
# 3. Utils
# ============================================================

def run_cmd(cmd):
    print("\n[CMD]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def find_exe(name):
    exe = shutil.which(name)
    if exe:
        return exe
    raise RuntimeError(f"Cannot find {name}")


def download_file(url, out_path):
    if out_path.exists() and out_path.stat().st_size > 0:
        return True

    try:
        with requests.get(url, stream=True, timeout=120) as r:
            if r.status_code != 200:
                print(f"[FAILED] {url}")
                return False

            with open(out_path, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

        print(f"[OK] {out_path.name}")
        return True

    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def get_filename(url):
    m = re.search(r"FILENAME=([^&]+)", url)
    if m:
        return m.group(1)
    return "tile.tif"


# ============================================================
# 4. Download MNS
# ============================================================

def download_mns_tiles():
    print("\n=== Downloading MNS ===")

    if not MNS_LINK_FILE.exists():
        raise RuntimeError(f"Link file not found: {MNS_LINK_FILE}")

    with open(MNS_LINK_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Total links: {len(urls)}")

    for url in tqdm(urls):
        name = get_filename(url)
        out_path = MNS_DIR / name
        download_file(url, out_path)


def prepare_mns_list():
    files = sorted(MNS_DIR.rglob("*.tif")) + sorted(MNS_DIR.rglob("*.tiff"))

    if not files:
        raise RuntimeError("No MNS tif files found.")

    path = WORK_DIR / "mns.txt"

    with open(path, "w", encoding="utf-8") as f:
        for p in files:
            f.write(str(p) + "\n")

    print(f"MNS files: {len(files)}")
    return path


# ============================================================
# 5. Build DSM 0.6m only if needed
# ============================================================

def build_dsm_if_needed():
    if DSM_0P6_PATH.exists() and DSM_0P6_PATH.stat().st_size > 0:
        print(f"\n[SKIP] Existing DSM 0.6m found: {DSM_0P6_PATH}")
        return DSM_0P6_PATH

    gdalbuildvrt = find_exe("gdalbuildvrt")
    gdalwarp = find_exe("gdalwarp")

    mns_list = prepare_mns_list()
    mns_vrt = WORK_DIR / "mns.vrt"

    run_cmd([
        gdalbuildvrt,
        "-overwrite",
        "-input_file_list", str(mns_list),
        str(mns_vrt)
    ])

    run_cmd([
        gdalwarp,
        "-overwrite",
        "-t_srs", CRS,
        "-te", str(X_MIN), str(Y_MIN), str(X_MAX), str(Y_MAX),
        "-tr", str(TARGET_RES), str(TARGET_RES),
        "-r", "bilinear",
        "-multi",
        "-wo", "NUM_THREADS=ALL_CPUS",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        str(mns_vrt),
        str(DSM_0P6_PATH)
    ])

    return DSM_0P6_PATH


# ============================================================
# 6. Patch cutting
# ============================================================

def clear_old_patches():
    for folder in [RGB_PATCH_DIR, DSM_PATCH_DIR]:
        for f in folder.glob("*.tif"):
            f.unlink()


def cut(rgb_path, dsm_path):
    print("\n=== Cutting patches ===")

    count = 0
    skipped_rgb = 0
    skipped_dsm = 0
    skipped_flat_dsm = 0

    with rasterio.open(rgb_path) as r, rasterio.open(dsm_path) as d:
        if r.width != d.width or r.height != d.height:
            raise RuntimeError(
                f"RGB and DSM size mismatch: "
                f"RGB={r.width}x{r.height}, DSM={d.width}x{d.height}"
            )

        print(f"Raster size: {r.width} × {r.height}")

        for row in range(0, r.height - PATCH_SIZE + 1, STRIDE):
            for col in range(0, r.width - PATCH_SIZE + 1, STRIDE):
                win = rasterio.windows.Window(col, row, PATCH_SIZE, PATCH_SIZE)

                rgb = r.read(window=win)
                dsm = d.read(1, window=win)

                rgb_valid = np.any(rgb > 0, axis=0)
                rgb_valid_ratio = rgb_valid.sum() / rgb_valid.size

                if rgb_valid_ratio < 0.95:
                    skipped_rgb += 1
                    continue

                dsm_valid = np.isfinite(dsm)
                dsm_valid_ratio = dsm_valid.sum() / dsm_valid.size

                if dsm_valid_ratio < 0.95:
                    skipped_dsm += 1
                    continue

                dsm_valid_values = dsm[dsm_valid]
                if np.nanmax(dsm_valid_values) <= np.nanmin(dsm_valid_values):
                    skipped_flat_dsm += 1
                    continue

                count += 1

                rgb_meta = r.meta.copy()
                rgb_meta.update({
                    "driver": "GTiff",
                    "height": PATCH_SIZE,
                    "width": PATCH_SIZE,
                    "count": r.count,
                    "dtype": rgb.dtype,
                    "transform": r.window_transform(win),
                    "compress": "lzw",
                    "tiled": True,
                })

                dsm_meta = d.meta.copy()
                dsm_meta.update({
                    "driver": "GTiff",
                    "height": PATCH_SIZE,
                    "width": PATCH_SIZE,
                    "count": 1,
                    "dtype": dsm.dtype,
                    "transform": d.window_transform(win),
                    "compress": "lzw",
                    "tiled": True,
                })

                rgb_out = RGB_PATCH_DIR / f"paris_rgb_{count:04d}.tif"
                dsm_out = DSM_PATCH_DIR / f"paris_dsm_{count:04d}.tif"

                with rasterio.open(rgb_out, "w", **rgb_meta) as dst:
                    dst.write(rgb)

                with rasterio.open(dsm_out, "w", **dsm_meta) as dst:
                    dst.write(dsm, 1)

    print(f"\nDone. patches: {count}")
    print(f"Skipped by RGB valid ratio: {skipped_rgb}")
    print(f"Skipped by DSM valid ratio: {skipped_dsm}")
    print(f"Skipped by flat DSM: {skipped_flat_dsm}")
    print(f"RGB patches: {RGB_PATCH_DIR}")
    print(f"DSM patches: {DSM_PATCH_DIR}")


# ============================================================
# 7. main
# ============================================================

def main():
    print("=== Paris RGB + MNS DSM patch pipeline ===")
    print(f"MNS_LINK_FILE: {MNS_LINK_FILE}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"RGB_0P6_PATH: {RGB_0P6_PATH}")
    print(f"Area: X {X_MIN}-{X_MAX}, Y {Y_MIN}-{Y_MAX}")
    print(f"Resolution: {TARGET_RES} m")
    print(f"Patch size: {PATCH_SIZE}")

    if not RGB_0P6_PATH.exists():
        raise RuntimeError(
            f"RGB_0P6_PATH does not exist: {RGB_0P6_PATH}\n"
            "你之前已经生成过 rgb_0p6.tif，所以这里默认直接复用它。"
        )

    download_mns_tiles()
    dsm_path = build_dsm_if_needed()

    clear_old_patches()
    cut(RGB_0P6_PATH, dsm_path)


if __name__ == "__main__":
    main()