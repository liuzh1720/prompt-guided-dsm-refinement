from pathlib import Path
import os

import geopandas as gpd
import rasterio
from shapely.geometry import box


# ============================================================
# 1. 路径设置
# ============================================================

# Paths — update these before running
dsm_dir = Path("data/Paris/dsm")  # reference DSM patches to define ROI
extracted_root = Path("data/Paris/BDTOPO_extracted")  # extracted BD TOPO data
out_dir = Path("data/Paris/building_footprints_ign")  # intermediate output
)

out_dir.mkdir(parents=True, exist_ok=True)

roi_gpkg_path = out_dir / "paris_dsm_roi_bbox.gpkg"
out_gpkg_path = out_dir / "paris_bdtopo_buildings_clipped_to_dsm_roi.gpkg"

out_shp_dir = out_dir / "shapefile"
out_shp_dir.mkdir(parents=True, exist_ok=True)
out_shp_path = out_shp_dir / "paris_bdtopo_buildings_clipped_to_dsm_roi.shp"


# ============================================================
# 2. 删除旧输出
# ============================================================

def delete_file_if_exists(path):
    path = Path(path)

    if path.exists():
        try:
            path.unlink()
            print(f"Deleted old file: {path}")
        except Exception as e:
            print(f"Warning: failed to delete old file: {path}")
            print(e)


def delete_shapefile_set(shp_path):
    """
    删除 shapefile 的所有相关文件。
    """
    shp_path = Path(shp_path)
    stem = shp_path.with_suffix("")

    suffixes = [
        ".shp",
        ".shx",
        ".dbf",
        ".prj",
        ".cpg",
        ".qpj",
        ".sbn",
        ".sbx",
        ".fix",
        ".shp.xml",
    ]

    for suffix in suffixes:
        delete_file_if_exists(stem.with_suffix(suffix))


def clean_previous_outputs():
    print()
    print("Cleaning previous output files...")

    delete_file_if_exists(roi_gpkg_path)
    delete_file_if_exists(out_gpkg_path)
    delete_shapefile_set(out_shp_path)


# ============================================================
# 3. DSM ROI 工具函数
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


def compute_dsm_bounds_and_crs(dsm_files):
    lefts = []
    bottoms = []
    rights = []
    tops = []

    first_crs = None

    for tif_path in dsm_files:
        with rasterio.open(tif_path) as src:
            if first_crs is None:
                first_crs = src.crs

            if src.crs != first_crs:
                raise RuntimeError(
                    f"CRS mismatch: {tif_path.name} has {src.crs}, "
                    f"but first file has {first_crs}"
                )

            bounds = src.bounds

            lefts.append(bounds.left)
            bottoms.append(bounds.bottom)
            rights.append(bounds.right)
            tops.append(bounds.top)

    total_bounds = (
        min(lefts),
        min(bottoms),
        max(rights),
        max(tops),
    )

    return total_bounds, first_crs


def save_dsm_roi(bounds, crs):
    roi_gdf = gpd.GeoDataFrame(
        {"name": ["paris_dsm_roi"]},
        geometry=[box(*bounds)],
        crs=crs
    )

    roi_gdf.to_file(
        roi_gpkg_path,
        layer="dsm_roi",
        driver="GPKG"
    )

    return roi_gdf


# ============================================================
# 4. 搜索 BATIMENT 图层
# ============================================================

def find_building_vector_files(folder):
    """
    在解压后的 BD TOPO 文件夹中查找 BATIMENT 图层。
    使用 os.walk，避免 pathlib.rglob 在 Windows 长路径或异常目录时报错。
    """
    folder = Path(folder)

    if not folder.exists():
        raise RuntimeError(f"Extracted folder does not exist: {folder}")

    vector_files = []

    def on_walk_error(error):
        print(f"Warning: skipped folder due to error: {error}")

    for root, dirs, files in os.walk(folder, onerror=on_walk_error):
        root_path = Path(root)

        for file_name in files:
            lower_name = file_name.lower()

            if not (
                lower_name.endswith(".shp")
                or lower_name.endswith(".gpkg")
                or lower_name.endswith(".geojson")
            ):
                continue

            file_path = root_path / file_name
            vector_files.append(file_path)

    if len(vector_files) == 0:
        raise RuntimeError(f"No vector files found under: {folder}")

    candidates = []

    for file_path in vector_files:
        text = str(file_path).upper()
        name = file_path.name.upper()

        score = 0

        # 最优先：文件名正好是 BATIMENT.SHP
        if name == "BATIMENT.SHP":
            score += 1000

        if "BATIMENT" in name:
            score += 500

        if "BATIMENT" in text:
            score += 300

        if "BATI" in text:
            score += 120

        if "BUILDING" in text:
            score += 120

        # 排除明显不相关图层
        bad_keywords = [
            "ADMINISTRATIF",
            "ROUTE",
            "TRONCON",
            "HYDRO",
            "OROGRAPHIE",
            "ZONE_ACTIVITE",
            "OCCUPATION_DU_SOL",
            "VEGETATION",
            "TRANSPORT",
            "LIEU_DIT",
            "TOPONYMIE",
            "ADRESSE",
            "EQUIPEMENT",
        ]

        for bad in bad_keywords:
            if bad in text:
                score -= 200

        if file_path.suffix.lower() == ".shp":
            score += 50

        if file_path.suffix.lower() == ".gpkg":
            score += 30

        if score > 0:
            candidates.append((score, file_path))

    candidates = sorted(candidates, reverse=True, key=lambda x: x[0])

    print()
    print("Building vector candidates:")
    for score, file_path in candidates[:30]:
        print(f"[score={score}] {file_path}")

    if len(candidates) == 0:
        print()
        print("No BATIMENT candidate found.")
        print("Some vector files found:")
        for file_path in vector_files[:80]:
            print(file_path)

        raise RuntimeError("Could not find BATIMENT / BATI vector layer.")

    selected = candidates[0][1]

    print()
    print("Selected building vector:")
    print(selected)

    return selected


# ============================================================
# 5. GeoDataFrame 清理和裁剪
# ============================================================

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

    return gdf


def keep_polygon_geometries(gdf):
    """
    gpd.clip 可能会在 ROI 边界生成 Point 或 LineString。
    这些不是建筑 footprint，需要过滤掉。
    """
    gdf = gdf.copy()

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    gdf = gdf[
        gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    return gdf


def prepare_gdf_for_writing(gdf):
    """
    保存前清理字段类型。
    避免 old dates、mixed object、None、NaT 等导致 pyogrio 写出失败。
    """
    gdf = gdf.copy()

    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue

        # datetime 转字符串
        if str(gdf[col].dtype).startswith("datetime"):
            gdf[col] = gdf[col].astype(str)
            continue

        # object 全部安全转字符串
        if gdf[col].dtype == "object":
            gdf[col] = gdf[col].apply(
                lambda x: "" if x is None else str(x)
            )

        # 清理常见空值字符串
        gdf[col] = gdf[col].replace(["nan", "NaT", "None"], "")

    return gdf


def clip_buildings_to_roi(vector_path, roi_gdf, target_crs):
    print()
    print("Reading building vector:")
    print(vector_path)

    buildings = gpd.read_file(vector_path)

    print(f"Loaded features: {len(buildings)}")
    print(f"Building CRS: {buildings.crs}")

    if buildings.empty:
        raise RuntimeError("Building layer is empty.")

    if buildings.crs is None:
        raise RuntimeError(
            "Building layer CRS is None. "
            "Please define CRS manually before clipping."
        )

    if buildings.crs != target_crs:
        print(f"Reprojecting buildings to DSM CRS: {target_crs}")
        buildings = buildings.to_crs(target_crs)

    buildings = clean_gdf(buildings)
    buildings = keep_polygon_geometries(buildings)

    print(f"Polygon buildings before clipping: {len(buildings)}")

    print("Clipping buildings to DSM ROI...")

    clipped = gpd.clip(buildings, roi_gdf)
    clipped = clean_gdf(clipped)
    clipped = keep_polygon_geometries(clipped)

    if clipped.empty:
        raise RuntimeError("No polygon buildings intersect DSM ROI after clipping.")

    return clipped


# ============================================================
# 6. 保存输出
# ============================================================

def save_outputs(clipped):
    print()
    print("Saving outputs...")

    clipped = keep_polygon_geometries(clipped)

    print(f"Polygon buildings after geometry filtering: {len(clipped)}")

    if clipped.empty:
        raise RuntimeError("No polygon building geometries remain after filtering.")

    clipped_out = prepare_gdf_for_writing(clipped)

    # 先保存 GPKG，后续分析主要用这个
    clipped_out.to_file(
        out_gpkg_path,
        layer="buildings",
        driver="GPKG"
    )

    print(f"Saved GeoPackage: {out_gpkg_path}")

    # Shapefile 是备用输出，字段更少，避免字段名/类型/几何限制
    safe_cols = []

    for col in clipped_out.columns:
        if col == clipped_out.geometry.name:
            safe_cols.append(col)
            continue

        upper_col = col.upper()

        if upper_col in [
            "ID",
            "ID_BDTOPO",
            "CLEABS",
            "NATURE",
            "USAGE1",
            "USAGE2",
            "HAUTEUR",
            "Z_MIN",
            "Z_MAX",
        ]:
            safe_cols.append(col)

    if clipped_out.geometry.name not in safe_cols:
        safe_cols.append(clipped_out.geometry.name)

    clipped_shp = clipped_out[safe_cols].copy()
    clipped_shp = keep_polygon_geometries(clipped_shp)

    try:
        clipped_shp.to_file(
            out_shp_path,
            driver="ESRI Shapefile",
            encoding="utf-8"
        )

        print(f"Saved Shapefile: {out_shp_path}")

    except Exception as e:
        print()
        print("Warning: failed to save Shapefile.")
        print("This is not critical because GeoPackage was saved successfully.")
        print(f"Shapefile error: {e}")


# ============================================================
# 7. 主程序
# ============================================================

def main():
    clean_previous_outputs()

    print()
    print("Computing DSM ROI...")

    dsm_files = list_dsm_files(dsm_dir)

    if len(dsm_files) == 0:
        raise RuntimeError(f"No DSM files found in: {dsm_dir}")

    print(f"DSM patches found: {len(dsm_files)}")

    dsm_bounds, dsm_crs = compute_dsm_bounds_and_crs(dsm_files)

    if dsm_crs is None:
        raise RuntimeError("DSM CRS is None.")

    print()
    print("DSM CRS:")
    print(dsm_crs)

    print()
    print("DSM bounds:")
    print(dsm_bounds)

    roi_gdf = save_dsm_roi(dsm_bounds, dsm_crs)

    print()
    print("Saved DSM ROI bbox:")
    print(roi_gpkg_path)

    print()
    print("Searching for BATIMENT layer in extracted BD TOPO folder...")
    print(extracted_root)

    building_vector = find_building_vector_files(extracted_root)

    clipped = clip_buildings_to_roi(
        vector_path=building_vector,
        roi_gdf=roi_gdf,
        target_crs=dsm_crs
    )

    print()
    print(f"Buildings after clipping: {len(clipped)}")

    save_outputs(clipped)

    print()
    print("Done.")
    print(f"Output GeoPackage: {out_gpkg_path}")
    print(f"Output Shapefile: {out_shp_path}")
    print(f"DSM ROI bbox: {roi_gpkg_path}")

    print()
    print("Columns:")
    print(list(clipped.columns))

    possible_height_cols = [
        "hauteur",
        "HAUTEUR",
        "hauteur_m",
        "HAUTEUR_M",
        "altitude_minimale_sol",
        "altitude_maximale_toit",
        "ALTITUDE_MINIMALE_SOL",
        "ALTITUDE_MAXIMALE_TOIT",
        "z_min",
        "z_max",
        "Z_MIN",
        "Z_MAX",
    ]

    for col in possible_height_cols:
        if col in clipped.columns:
            print()
            print(f"Column found: {col}")
            print(clipped[col].describe())


if __name__ == "__main__":
    main()