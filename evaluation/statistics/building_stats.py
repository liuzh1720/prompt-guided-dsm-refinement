from pathlib import Path
import re
import warnings
import shutil

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from shapely.ops import unary_union
import matplotlib.pyplot as plt


# ============================================================
# 1. 三个城市路径配置
# ============================================================

CONFIGS = {
    "hongkong": {
        "dsm_dir": Path("data/HK/dsm"),
        "footprint_dir": Path("data/HK/building_footprints"),
        "out_dir": Path("results/statistics/building_height_stats/HK"),
        "nodata_values": [-9999, -32768],
    },

    "austin": {
        "dsm_dir": Path("data/Austin/dsm"),
        "footprint_dir": Path("data/Austin/building_footprints"),
        "out_dir": Path("results/statistics/building_height_stats/Austin"),
        "nodata_values": [-9999, -32768],
    },

    "paris": {
        "dsm_dir": Path("data/Paris/dsm"),
        "footprint_dir": Path("data/Paris/building_footprints"),
        "out_dir": Path("results/statistics/building_height_stats/Paris"),
        "nodata_values": [-9999, -32768],
    },
}


# ============================================================
# 2. 参数设置
# ============================================================

ROOF_EROSION_M = 1.2
GROUND_INNER_BUFFER_M = 3.0
GROUND_OUTER_BUFFER_M = 15.0

GROUND_PERCENTILE = 10
ROOF_PERCENTILE = 50

MIN_ROOF_PIXELS = 10
MIN_GROUND_PIXELS = 20

MIN_REASONABLE_HEIGHT = -5.0
MAX_REASONABLE_HEIGHT = 500.0

SAVE_PLOTS = True
SAVE_COMBINED = True
CLEAR_OLD_OUTPUTS = True

COMBINED_OUT_DIR = Path("results/statistics/building_height_stats_all_cities")


# ============================================================
# 3. 文件过滤与宽松匹配函数
# ============================================================

def is_valid_main_file(file_path, allowed_suffixes):
    file_path = Path(file_path)
    name_lower = file_path.name.lower()

    if not file_path.exists():
        return False

    if not file_path.is_file():
        return False

    allowed_suffixes = [s.lower() for s in allowed_suffixes]

    bad_endings = [
        ".aux.xml",
        ".ovr",
        ".qmd",
        ".qml",
        ".xml",
        ".sld",
        ".lock",
        ".tmp",
        ".bak",
        ".old",
    ]

    for ending in bad_endings:
        if name_lower.endswith(ending):
            return False

    if file_path.suffix.lower() not in allowed_suffixes:
        return False

    return True


def extract_number_id(name):
    stem = Path(str(name)).stem
    match = re.search(r"(\d+)", stem)

    if match is None:
        return None

    return match.group(1)


def normalize_number(number_text):
    if number_text is None:
        return None

    try:
        return str(int(number_text))
    except Exception:
        return number_text


def list_dsm_files(dsm_dir):
    dsm_dir = Path(dsm_dir)
    files = []

    for file_path in sorted(dsm_dir.iterdir()):
        if is_valid_main_file(file_path, [".tif", ".tiff"]):
            files.append(file_path)

    return files


def get_patch_id(file_path):
    return Path(file_path).stem


def find_footprint_file(footprint_dir, patch_id):
    footprint_dir = Path(footprint_dir)

    if not footprint_dir.exists():
        return None

    allowed_suffixes = [".gpkg", ".geojson", ".shp"]

    direct_candidates = [
        footprint_dir / f"{patch_id}_buildings.gpkg",
        footprint_dir / f"{patch_id}_buildings.geojson",
        footprint_dir / f"{patch_id}_buildings.shp",
    ]

    for candidate in direct_candidates:
        if is_valid_main_file(candidate, allowed_suffixes):
            return candidate

    target_num = normalize_number(extract_number_id(patch_id))

    if target_num is None:
        return None

    candidates = []

    for file_path in footprint_dir.iterdir():
        if not is_valid_main_file(file_path, allowed_suffixes):
            continue

        stem_lower = file_path.stem.lower()

        if "building" not in stem_lower and "buildings" not in stem_lower:
            continue

        file_num = normalize_number(extract_number_id(file_path.name))

        if file_num == target_num:
            candidates.append(file_path)

    if len(candidates) == 0:
        return None

    suffix_priority = {
        ".gpkg": 0,
        ".geojson": 1,
        ".shp": 2,
    }

    candidates = sorted(
        candidates,
        key=lambda p: (
            suffix_priority.get(p.suffix.lower(), 99),
            0 if p.stem == f"{patch_id}_buildings" else 1,
            len(p.stem),
            p.name,
        )
    )

    return candidates[0]


# ============================================================
# 4. 基础统计与几何函数
# ============================================================

def clean_dsm_array(dsm, nodata_values):
    arr = dsm.astype(np.float32)

    valid = np.isfinite(arr)

    for nodata in nodata_values:
        valid = valid & (arr != nodata)

    valid = valid & (arr > -1000) & (arr < 10000)

    return arr, valid


def safe_stats(values, prefix):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_p95": np.nan,
        }

    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
    }


def make_valid_geometry(geom):
    if geom is None:
        return None

    if geom.is_empty:
        return None

    try:
        if not geom.is_valid:
            geom = geom.buffer(0)
    except Exception:
        return None

    if geom is None or geom.is_empty:
        return None

    if geom.geom_type not in ["Polygon", "MultiPolygon"]:
        return None

    return geom


def clean_building_gdf(gdf):
    if gdf.empty:
        return gdf

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        return gdf

    gdf["geometry"] = gdf.geometry.apply(make_valid_geometry)

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    gdf = gdf[
        gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    return gdf


def rasterize_geom(geom, src, all_touched=True):
    if geom is None or geom.is_empty:
        return np.zeros((src.height, src.width), dtype=np.uint8)

    mask = rasterize(
        shapes=[(geom, 1)],
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched
    )

    return mask


def load_patch_buildings(footprint_path, raster_crs):
    try:
        gdf = gpd.read_file(footprint_path)
    except Exception as e:
        warnings.warn(f"Failed to read footprint file {footprint_path}: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=raster_crs)

    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=raster_crs)

    if gdf.crs is None:
        warnings.warn(f"Footprint CRS is None: {footprint_path}")
        return gpd.GeoDataFrame(geometry=[], crs=raster_crs)

    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    gdf = clean_building_gdf(gdf)

    return gdf


# ============================================================
# 5. 建筑高度估计
# ============================================================

def estimate_one_building_height(
    building_geom,
    all_buildings_union,
    patch_geom,
    src,
    dsm_arr,
    valid_mask
):
    building_geom = make_valid_geometry(building_geom)

    if building_geom is None:
        return None

    building_geom = building_geom.intersection(patch_geom)
    building_geom = make_valid_geometry(building_geom)

    if building_geom is None:
        return None

    footprint_area = float(building_geom.area)

    roof_geom = building_geom.buffer(-ROOF_EROSION_M)

    if roof_geom is None or roof_geom.is_empty:
        roof_geom = building_geom

    roof_geom = roof_geom.intersection(patch_geom)
    roof_geom = make_valid_geometry(roof_geom)

    if roof_geom is None:
        return None

    outer = building_geom.buffer(GROUND_OUTER_BUFFER_M)
    inner = building_geom.buffer(GROUND_INNER_BUFFER_M)

    ground_ring = outer.difference(inner)
    ground_ring = ground_ring.intersection(patch_geom)

    if all_buildings_union is not None and not all_buildings_union.is_empty:
        ground_ring = ground_ring.difference(all_buildings_union)

    ground_ring = make_valid_geometry(ground_ring)

    if ground_ring is None:
        return None

    roof_mask = rasterize_geom(roof_geom, src, all_touched=True)
    ground_mask = rasterize_geom(ground_ring, src, all_touched=True)

    roof_valid = (roof_mask == 1) & valid_mask
    ground_valid = (ground_mask == 1) & valid_mask

    roof_pixels = int(np.sum(roof_valid))
    ground_pixels = int(np.sum(ground_valid))

    if roof_pixels < MIN_ROOF_PIXELS:
        return {
            "valid_height": False,
            "invalid_reason": "too_few_roof_pixels",
            "footprint_area": footprint_area,
            "roof_pixels": roof_pixels,
            "ground_pixels": ground_pixels,
        }

    if ground_pixels < MIN_GROUND_PIXELS:
        return {
            "valid_height": False,
            "invalid_reason": "too_few_ground_pixels",
            "footprint_area": footprint_area,
            "roof_pixels": roof_pixels,
            "ground_pixels": ground_pixels,
        }

    roof_values = dsm_arr[roof_valid]
    ground_values = dsm_arr[ground_valid]

    roof_ref = float(np.percentile(roof_values, ROOF_PERCENTILE))
    ground_ref = float(np.percentile(ground_values, GROUND_PERCENTILE))

    estimated_height = roof_ref - ground_ref

    valid_height = (
        np.isfinite(estimated_height)
        and estimated_height >= MIN_REASONABLE_HEIGHT
        and estimated_height <= MAX_REASONABLE_HEIGHT
    )

    invalid_reason = ""

    if not valid_height:
        invalid_reason = "unreasonable_height"

    return {
        "valid_height": bool(valid_height),
        "invalid_reason": invalid_reason,
        "footprint_area": footprint_area,
        "roof_pixels": roof_pixels,
        "ground_pixels": ground_pixels,
        "roof_dsm_ref": roof_ref,
        "ground_dsm_ref": ground_ref,
        "estimated_height": float(estimated_height),
        "roof_dsm_mean": float(np.mean(roof_values)),
        "roof_dsm_median": float(np.median(roof_values)),
        "ground_dsm_mean": float(np.mean(ground_values)),
        "ground_dsm_median": float(np.median(ground_values)),
        "ground_dsm_p10": float(np.percentile(ground_values, 10)),
        "ground_dsm_p20": float(np.percentile(ground_values, 20)),
    }


# ============================================================
# 6. 绘图与输出清理
# ============================================================

def plot_hist(values, out_path, title, xlabel):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50, density=True, alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_boxplot(data_dict, out_path, title, ylabel):
    labels = []
    data = []

    for name, values in data_dict.items():
        arr = np.asarray(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]

        if arr.size > 0:
            labels.append(name)
            data.append(arr)

    if len(data) == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def clean_output_dir(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not CLEAR_OLD_OUTPUTS:
        return

    targets = [
        "estimated_building_height_per_building.csv",
        "building_height_stats_per_patch.csv",
        "building_height_stats_summary.csv",
        "estimated_building_height_distribution.png",
        "building_footprint_area_distribution.png",
        "building_coverage_ratio_distribution.png",
    ]

    for name in targets:
        path = out_dir / name
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                print(f"Warning: failed to delete old output {path}: {e}")


def clean_combined_outputs():
    COMBINED_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CLEAR_OLD_OUTPUTS:
        return

    targets = [
        "estimated_building_height_per_building_all_cities.csv",
        "building_height_stats_per_patch_all_cities.csv",
        "building_height_stats_summary_all_cities.csv",
        "estimated_building_height_boxplot_by_city.png",
        "building_footprint_area_boxplot_by_city.png",
    ]

    for name in targets:
        path = COMBINED_OUT_DIR / name
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                print(f"Warning: failed to delete old combined output {path}: {e}")


# ============================================================
# 7. 单城市处理
# ============================================================

def append_empty_patch_record(per_patch_records, city_name, patch_id, dsm_file, valid_pixels):
    per_patch_records.append({
        "city": city_name,
        "patch_id": patch_id,
        "dsm_file": dsm_file.name,
        "valid_pixels": valid_pixels,
        "building_pixels": 0,
        "building_ratio": 0.0 if valid_pixels > 0 else np.nan,
        "num_intersecting_buildings": 0,
        "num_valid_height_buildings": 0,
        "num_invalid_height_buildings": 0,

        # Building footprint area statistics.
        # Unit depends on the CRS unit. For projected CRS, this is normally m².
        "footprint_area_count": 0,
        "footprint_area_mean": np.nan,
        "footprint_area_median": np.nan,
        "footprint_area_std": np.nan,
        "footprint_area_min": np.nan,
        "footprint_area_max": np.nan,
        "footprint_area_p25": np.nan,
        "footprint_area_p75": np.nan,
        "footprint_area_p90": np.nan,
        "footprint_area_p95": np.nan,
        "total_footprint_area": 0.0,
    })


def process_city(city_name, config):
    dsm_dir = config["dsm_dir"]
    footprint_dir = config["footprint_dir"]
    out_dir = config["out_dir"]
    nodata_values = config.get("nodata_values", [-9999, -32768])

    clean_output_dir(out_dir)

    print()
    print("=" * 90)
    print(f"Processing city: {city_name}")
    print(f"DSM folder: {dsm_dir}")
    print(f"Footprint folder: {footprint_dir}")
    print(f"Output folder: {out_dir}")

    if not dsm_dir.exists():
        raise RuntimeError(f"DSM folder does not exist: {dsm_dir}")

    if not footprint_dir.exists():
        raise RuntimeError(f"Footprint folder does not exist: {footprint_dir}")

    dsm_files = list_dsm_files(dsm_dir)

    if len(dsm_files) == 0:
        raise RuntimeError(f"No DSM tif files found in: {dsm_dir}")

    print(f"DSM patches found: {len(dsm_files)}")

    per_building_records = []
    per_patch_records = []

    first_patch_crs = None
    building_global_id = 0

    missing_footprint_count = 0
    empty_footprint_count = 0

    for patch_index, dsm_file in enumerate(dsm_files):
        patch_id = get_patch_id(dsm_file)
        footprint_path = find_footprint_file(footprint_dir, patch_id)

        try:
            with rasterio.open(dsm_file) as src:
                if first_patch_crs is None:
                    first_patch_crs = src.crs
                    print(f"DSM CRS: {first_patch_crs}")

                    if src.crs is not None and src.crs.is_geographic:
                        warnings.warn(
                            "DSM CRS is geographic. Buffer distances are not meters. "
                            "Please reproject DSM/footprints to a projected CRS before using this method."
                        )

                dsm = src.read(1)
                dsm_arr, valid_mask = clean_dsm_array(dsm, nodata_values)
                valid_pixels = int(np.sum(valid_mask))

                if footprint_path is None:
                    missing_footprint_count += 1
                    append_empty_patch_record(
                        per_patch_records,
                        city_name,
                        patch_id,
                        dsm_file,
                        valid_pixels
                    )
                    continue

                buildings = load_patch_buildings(footprint_path, src.crs)

                if buildings.empty:
                    empty_footprint_count += 1
                    append_empty_patch_record(
                        per_patch_records,
                        city_name,
                        patch_id,
                        dsm_file,
                        valid_pixels
                    )
                    continue

                patch_geom = box(
                    src.bounds.left,
                    src.bounds.bottom,
                    src.bounds.right,
                    src.bounds.top
                )

                clipped_geoms = []

                for geom in buildings.geometry:
                    geom = make_valid_geometry(geom)

                    if geom is None:
                        continue

                    clipped = geom.intersection(patch_geom)
                    clipped = make_valid_geometry(clipped)

                    if clipped is not None:
                        clipped_geoms.append(clipped)

                if len(clipped_geoms) == 0:
                    append_empty_patch_record(
                        per_patch_records,
                        city_name,
                        patch_id,
                        dsm_file,
                        valid_pixels
                    )
                    continue

                all_buildings_union = unary_union(clipped_geoms)

                building_mask = rasterize_geom(
                    all_buildings_union,
                    src,
                    all_touched=True
                )

                building_valid_mask = (building_mask == 1) & valid_mask
                building_pixels = int(np.sum(building_valid_mask))

                if valid_pixels > 0:
                    building_ratio = building_pixels / valid_pixels
                else:
                    building_ratio = np.nan

                patch_height_values = []
                patch_roof_values = []
                patch_ground_values = []
                patch_footprint_areas = []

                num_valid_height_buildings = 0
                num_invalid_height_buildings = 0

                for local_idx, building_geom in enumerate(clipped_geoms):
                    building_global_id += 1

                    result = estimate_one_building_height(
                        building_geom=building_geom,
                        all_buildings_union=all_buildings_union,
                        patch_geom=patch_geom,
                        src=src,
                        dsm_arr=dsm_arr,
                        valid_mask=valid_mask
                    )

                    if result is None:
                        num_invalid_height_buildings += 1

                        per_building_records.append({
                            "city": city_name,
                            "patch_id": patch_id,
                            "building_global_id": building_global_id,
                            "building_local_id": local_idx,
                            "valid_height": False,
                            "invalid_reason": "geometry_failed",
                        })

                        continue

                    valid_height = result.get("valid_height", False)

                    if "footprint_area" in result and np.isfinite(result["footprint_area"]):
                        patch_footprint_areas.append(result["footprint_area"])

                    if valid_height:
                        num_valid_height_buildings += 1
                        patch_height_values.append(result["estimated_height"])
                        patch_roof_values.append(result["roof_dsm_ref"])
                        patch_ground_values.append(result["ground_dsm_ref"])
                    else:
                        num_invalid_height_buildings += 1

                    record = {
                        "city": city_name,
                        "patch_id": patch_id,
                        "building_global_id": building_global_id,
                        "building_local_id": local_idx,
                        "footprint_file": footprint_path.name,
                    }

                    record.update(result)
                    per_building_records.append(record)

                patch_record = {
                    "city": city_name,
                    "patch_id": patch_id,
                    "dsm_file": dsm_file.name,
                    "footprint_file": footprint_path.name,
                    "valid_pixels": valid_pixels,
                    "building_pixels": building_pixels,
                    "building_ratio": building_ratio,
                    "num_intersecting_buildings": len(clipped_geoms),
                    "num_valid_height_buildings": num_valid_height_buildings,
                    "num_invalid_height_buildings": num_invalid_height_buildings,
                }

                patch_record.update(
                    safe_stats(patch_footprint_areas, "footprint_area")
                )

                patch_record["total_footprint_area"] = (
                    float(np.sum(patch_footprint_areas))
                    if len(patch_footprint_areas) > 0 else 0.0
                )

                patch_record.update(
                    safe_stats(patch_height_values, "estimated_height")
                )
                patch_record.update(
                    safe_stats(patch_roof_values, "roof_dsm_ref")
                )
                patch_record.update(
                    safe_stats(patch_ground_values, "ground_dsm_ref")
                )

                per_patch_records.append(patch_record)

        except Exception as e:
            warnings.warn(f"Failed to process {dsm_file.name}: {e}")

        if (patch_index + 1) % 50 == 0:
            print(f"Processed {patch_index + 1}/{len(dsm_files)} patches")

    per_building_df = pd.DataFrame(per_building_records)
    per_patch_df = pd.DataFrame(per_patch_records)

    per_building_csv = out_dir / "estimated_building_height_per_building.csv"
    per_patch_csv = out_dir / "building_height_stats_per_patch.csv"

    per_building_df.to_csv(per_building_csv, index=False, encoding="utf-8-sig")
    per_patch_df.to_csv(per_patch_csv, index=False, encoding="utf-8-sig")

    if "valid_height" in per_building_df.columns:
        valid_buildings_df = per_building_df[
            per_building_df["valid_height"] == True
        ].copy()
    else:
        valid_buildings_df = pd.DataFrame()

    if "estimated_height" in valid_buildings_df.columns:
        estimated_heights = valid_buildings_df["estimated_height"].to_numpy(dtype=np.float32)
    else:
        estimated_heights = np.array([], dtype=np.float32)

    if "roof_dsm_ref" in valid_buildings_df.columns:
        roof_refs = valid_buildings_df["roof_dsm_ref"].to_numpy(dtype=np.float32)
    else:
        roof_refs = np.array([], dtype=np.float32)

    if "ground_dsm_ref" in valid_buildings_df.columns:
        ground_refs = valid_buildings_df["ground_dsm_ref"].to_numpy(dtype=np.float32)
    else:
        ground_refs = np.array([], dtype=np.float32)

    if "footprint_area" in per_building_df.columns:
        footprint_areas = per_building_df["footprint_area"].to_numpy(dtype=np.float32)
        footprint_areas = footprint_areas[np.isfinite(footprint_areas)]
    else:
        footprint_areas = np.array([], dtype=np.float32)

    total_patches = len(per_patch_df)

    if "building_pixels" in per_patch_df.columns:
        building_patches = int((per_patch_df["building_pixels"] > 0).sum())
        total_building_pixels = int(per_patch_df["building_pixels"].sum())
    else:
        building_patches = 0
        total_building_pixels = 0

    if "valid_pixels" in per_patch_df.columns:
        total_valid_pixels = int(per_patch_df["valid_pixels"].sum())
    else:
        total_valid_pixels = 0

    if total_valid_pixels > 0:
        total_building_ratio = total_building_pixels / total_valid_pixels
    else:
        total_building_ratio = np.nan

    total_buildings = len(per_building_df)
    valid_height_buildings = len(valid_buildings_df)
    invalid_height_buildings = total_buildings - valid_height_buildings

    summary = {
        "city": city_name,
        "total_patches": total_patches,
        "building_patches": building_patches,
        "nonbuilding_patches": total_patches - building_patches,
        "missing_footprint_patches": missing_footprint_count,
        "empty_footprint_patches": empty_footprint_count,
        "total_valid_pixels": total_valid_pixels,
        "total_building_pixels": total_building_pixels,
        "total_building_ratio": total_building_ratio,
        "total_buildings_intersecting_patches": total_buildings,
        "valid_height_buildings": valid_height_buildings,
        "invalid_height_buildings": invalid_height_buildings,
        "valid_height_building_ratio": (
            valid_height_buildings / total_buildings
            if total_buildings > 0 else np.nan
        ),
        "roof_erosion_m": ROOF_EROSION_M,
        "ground_inner_buffer_m": GROUND_INNER_BUFFER_M,
        "ground_outer_buffer_m": GROUND_OUTER_BUFFER_M,
        "roof_percentile": ROOF_PERCENTILE,
        "ground_percentile": GROUND_PERCENTILE,
    }

    summary.update(
        safe_stats(footprint_areas, "footprint_area")
    )

    summary["total_footprint_area"] = (
        float(np.sum(footprint_areas))
        if footprint_areas.size > 0 else 0.0
    )

    summary.update(
        safe_stats(estimated_heights, "estimated_building_height")
    )
    summary.update(
        safe_stats(roof_refs, "roof_dsm_ref")
    )
    summary.update(
        safe_stats(ground_refs, "ground_dsm_ref")
    )

    summary_df = pd.DataFrame([summary])

    summary_csv = out_dir / "building_height_stats_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    if SAVE_PLOTS:
        plot_hist(
            estimated_heights,
            out_dir / "estimated_building_height_distribution.png",
            f"{city_name.capitalize()} Estimated Building Height Distribution",
            "Estimated building height: roof DSM - nearby ground DSM"
        )

        plot_hist(
            footprint_areas,
            out_dir / "building_footprint_area_distribution.png",
            f"{city_name.capitalize()} Building Footprint Area Distribution",
            "Building footprint area (m²)"
        )

        if "building_ratio" in per_patch_df.columns:
            plot_hist(
                per_patch_df["building_ratio"].to_numpy(dtype=np.float32),
                out_dir / "building_coverage_ratio_distribution.png",
                f"{city_name.capitalize()} Building Coverage Ratio per Patch",
                "Building pixels / valid DSM pixels"
            )

    print()
    print(f"Saved per-building CSV: {per_building_csv}")
    print(f"Saved per-patch CSV: {per_patch_csv}")
    print(f"Saved summary CSV: {summary_csv}")

    print()
    print("City summary:")
    print(f"Total patches: {total_patches}")
    print(f"Building patches: {building_patches}")
    print(f"Missing footprint patches: {missing_footprint_count}")
    print(f"Empty footprint patches: {empty_footprint_count}")
    print(f"Total intersecting buildings: {total_buildings}")
    print(f"Valid height buildings: {valid_height_buildings}")
    print(f"Total building ratio: {total_building_ratio:.4f}")

    if footprint_areas.size > 0:
        print(f"Mean footprint area: {np.nanmean(footprint_areas):.3f}")
        print(f"Median footprint area: {np.nanmedian(footprint_areas):.3f}")
        print(f"Total footprint area: {np.nansum(footprint_areas):.3f}")

    if estimated_heights.size > 0:
        print(f"Estimated height median: {np.nanmedian(estimated_heights):.3f}")
        print(f"Estimated height p90: {np.nanpercentile(estimated_heights, 90):.3f}")
        print(f"Estimated height p95: {np.nanpercentile(estimated_heights, 95):.3f}")

    return per_building_df, per_patch_df, summary_df, estimated_heights, footprint_areas


# ============================================================
# 8. 主程序
# ============================================================

def main():
    if SAVE_COMBINED:
        clean_combined_outputs()

    combined_building_dfs = []
    combined_patch_dfs = []
    combined_summary_dfs = []

    heights_by_city = {}
    areas_by_city = {}

    for city_name, config in CONFIGS.items():
        per_building_df, per_patch_df, summary_df, estimated_heights, footprint_areas = process_city(
            city_name,
            config
        )

        combined_building_dfs.append(per_building_df)
        combined_patch_dfs.append(per_patch_df)
        combined_summary_dfs.append(summary_df)

        heights_by_city[city_name] = estimated_heights
        areas_by_city[city_name] = footprint_areas

    if SAVE_COMBINED:
        all_buildings_df = pd.concat(combined_building_dfs, ignore_index=True)
        all_patches_df = pd.concat(combined_patch_dfs, ignore_index=True)
        all_summary_df = pd.concat(combined_summary_dfs, ignore_index=True)

        all_buildings_csv = COMBINED_OUT_DIR / "estimated_building_height_per_building_all_cities.csv"
        all_patches_csv = COMBINED_OUT_DIR / "building_height_stats_per_patch_all_cities.csv"
        all_summary_csv = COMBINED_OUT_DIR / "building_height_stats_summary_all_cities.csv"

        all_buildings_df.to_csv(all_buildings_csv, index=False, encoding="utf-8-sig")
        all_patches_df.to_csv(all_patches_csv, index=False, encoding="utf-8-sig")
        all_summary_df.to_csv(all_summary_csv, index=False, encoding="utf-8-sig")

        plot_boxplot(
            heights_by_city,
            COMBINED_OUT_DIR / "estimated_building_height_boxplot_by_city.png",
            "Estimated Building Height by City",
            "Estimated building height: roof DSM - nearby ground DSM"
        )

        plot_boxplot(
            areas_by_city,
            COMBINED_OUT_DIR / "building_footprint_area_boxplot_by_city.png",
            "Building Footprint Area by City",
            "Building footprint area (m²)"
        )

        print()
        print("=" * 90)
        print("Combined outputs saved:")
        print(f"All-city per-building CSV: {all_buildings_csv}")
        print(f"All-city per-patch CSV: {all_patches_csv}")
        print(f"All-city summary CSV: {all_summary_csv}")
        print(f"All-city height boxplot: {COMBINED_OUT_DIR / 'estimated_building_height_boxplot_by_city.png'}")
        print(f"All-city footprint area boxplot: {COMBINED_OUT_DIR / 'building_footprint_area_boxplot_by_city.png'}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()