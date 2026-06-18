from pathlib import Path
import os

import geopandas as gpd
import rasterio
from shapely.geometry import box


# ============================================================
# 1. 路径设置
# ============================================================

# Paths — update these before running
dsm_dir = Path("data/Paris/dsm")  # reference DSM patches
global_footprint_path = Path("data/Paris/building_footprints_ign/paris_bdtopo_buildings_clipped_to_dsm_roi.gpkg")  # input: clipped footprints
out_dir = Path("data/Paris/building_footprints")  # output: per-patch footprints

out_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 参数设置
# ============================================================

# True = 每个 patch 都保存一个 gpkg，即使没有建筑
# False = 没有建筑的 patch 不保存
SAVE_EMPTY_PATCH_FILES = True

# 每次运行前是否清空旧的 per-patch footprint
CLEAR_OLD_OUTPUTS = True


# ============================================================
# 3. 工具函数
# ============================================================

def list_dsm_files(folder):
    files = []

    for suffix in ["*.tif", "*.tiff"]:
        files.extend(folder.glob(suffix))

    clean_files = []

    for file_path in files:
        if ".aux" in file_path.name.lower():
            continue

        clean_files.append(file_path)

    return sorted(clean_files)


def get_patch_id(file_path):
    return file_path.stem


def clean_old_outputs(folder):
    if not folder.exists():
        return

    print("Cleaning old per-patch footprint files...")

    for file_path in folder.glob("*_buildings.gpkg"):
        try:
            file_path.unlink()
        except Exception as e:
            print(f"Warning: failed to delete {file_path}: {e}")


def clean_gdf(gdf):
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if len(gdf) == 0:
        return gdf

    def fix_geom(geom):
        if geom is None or geom.is_empty:
            return None

        if not geom.is_valid:
            try:
                geom = geom.buffer(0)
            except Exception:
                return None

        if geom is None or geom.is_empty:
            return None

        return geom

    gdf["geometry"] = gdf.geometry.apply(fix_geom)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    gdf = gdf[
        gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    return gdf


def prepare_gdf_for_writing(gdf):
    """
    把复杂字段转成安全格式，避免写 GPKG 时报错。
    """
    gdf = gdf.copy()

    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue

        if str(gdf[col].dtype).startswith("datetime"):
            gdf[col] = gdf[col].astype(str)
            continue

        if gdf[col].dtype == "object":
            gdf[col] = gdf[col].apply(
                lambda x: "" if x is None else str(x)
            )

        gdf[col] = gdf[col].replace(["nan", "NaT", "None"], "")

    return gdf


def make_empty_like(gdf, crs):
    """
    创建一个和原始 footprint 字段结构类似的空 GeoDataFrame。
    """
    empty = gdf.iloc[0:0].copy()
    empty = empty.set_crs(crs, allow_override=True)
    return empty


def save_patch_gpkg(gdf, out_path):
    if out_path.exists():
        try:
            out_path.unlink()
        except Exception as e:
            print(f"Warning: failed to delete old file {out_path}: {e}")

    gdf = prepare_gdf_for_writing(gdf)

    gdf.to_file(
        out_path,
        layer="buildings",
        driver="GPKG"
    )


# ============================================================
# 4. 主程序
# ============================================================

def main():
    print("Paris BD TOPO per-patch footprint splitting")
    print(f"DSM folder: {dsm_dir}")
    print(f"Global footprint: {global_footprint_path}")
    print(f"Output folder: {out_dir}")
    print()

    if not dsm_dir.exists():
        raise RuntimeError(f"DSM folder does not exist: {dsm_dir}")

    if not global_footprint_path.exists():
        raise RuntimeError(f"Global footprint file does not exist: {global_footprint_path}")

    if CLEAR_OLD_OUTPUTS:
        clean_old_outputs(out_dir)

    dsm_files = list_dsm_files(dsm_dir)

    if len(dsm_files) == 0:
        raise RuntimeError(f"No DSM patch files found in: {dsm_dir}")

    print(f"DSM patches found: {len(dsm_files)}")

    print("Reading global building footprints...")
    buildings = gpd.read_file(global_footprint_path)

    if buildings.empty:
        raise RuntimeError("Global building footprint file is empty.")

    buildings = clean_gdf(buildings)

    if buildings.empty:
        raise RuntimeError("No valid polygon building footprints after cleaning.")

    print(f"Global building footprints loaded: {len(buildings)}")
    print(f"Global footprint CRS: {buildings.crs}")

    saved_count = 0
    saved_empty_count = 0
    skipped_empty_count = 0

    for idx, dsm_file in enumerate(dsm_files):
        patch_id = get_patch_id(dsm_file)

        with rasterio.open(dsm_file) as src:
            patch_crs = src.crs
            bounds = src.bounds

        if buildings.crs is None:
            raise RuntimeError("Global footprint CRS is None.")

        if buildings.crs != patch_crs:
            buildings_in_crs = buildings.to_crs(patch_crs)
        else:
            buildings_in_crs = buildings

        patch_geom = box(
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top
        )

        patch_gdf = gpd.GeoDataFrame(
            {"patch_id": [patch_id]},
            geometry=[patch_geom],
            crs=patch_crs
        )

        # spatial index 预筛选，避免每次 clip 全量数据太慢
        try:
            idxs = list(buildings_in_crs.sindex.query(patch_geom, predicate="intersects"))
            candidates = buildings_in_crs.iloc[idxs].copy()
        except Exception:
            candidates = buildings_in_crs[buildings_in_crs.intersects(patch_geom)].copy()

        if candidates.empty:
            if SAVE_EMPTY_PATCH_FILES:
                out_path = out_dir / f"{patch_id}_buildings.gpkg"
                empty = make_empty_like(buildings_in_crs, patch_crs)
                save_patch_gpkg(empty, out_path)
                saved_empty_count += 1
            else:
                skipped_empty_count += 1

        else:
            try:
                clipped = gpd.clip(candidates, patch_gdf)
            except Exception:
                clipped = candidates[candidates.intersects(patch_geom)].copy()
                clipped["geometry"] = clipped.geometry.intersection(patch_geom)

            clipped = clean_gdf(clipped)

            out_path = out_dir / f"{patch_id}_buildings.gpkg"

            if clipped.empty:
                if SAVE_EMPTY_PATCH_FILES:
                    empty = make_empty_like(buildings_in_crs, patch_crs)
                    save_patch_gpkg(empty, out_path)
                    saved_empty_count += 1
                else:
                    skipped_empty_count += 1
            else:
                save_patch_gpkg(clipped, out_path)
                saved_count += 1

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(dsm_files)} patches")

    print()
    print("Done.")
    print(f"DSM patches processed: {len(dsm_files)}")
    print(f"Patches with building footprints saved: {saved_count}")
    print(f"Empty patch files saved: {saved_empty_count}")
    print(f"Empty patches skipped: {skipped_empty_count}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()