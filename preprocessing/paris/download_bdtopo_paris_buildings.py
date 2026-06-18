from pathlib import Path
import re
import zipfile
import shutil
import subprocess
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse, unquote
from email.message import Message

import requests
import geopandas as gpd
import rasterio
from shapely.geometry import box


# ============================================================
# 1. 路径设置
# ============================================================

# Paths — update these before running
dsm_dir = Path("data/Paris/dsm")  # reference DSM patches to define ROI
out_dir = Path("data/Paris/building_footprints_ign")  # intermediate output

out_dir.mkdir(parents=True, exist_ok=True)

download_dir = out_dir / "downloads"
extract_dir = out_dir / "extracted"

download_dir.mkdir(parents=True, exist_ok=True)
extract_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. BD TOPO API 设置
# ============================================================

RESOURCE_NAME = "BDTOPO"
BASE_RESOURCE_URL = f"https://data.geopf.fr/telechargement/resource/{RESOURCE_NAME}"

TARGET_ZONE = "D075"          # Paris
TARGET_CRS_FILTER = "epsg:2154"
LIMIT = 50

# 只接受近年数据
MIN_YEAR = 2024
MAX_YEAR = 2026

# 如果 True，只打印候选，不下载。
# 第一次建议设为 True，确认候选正确后再改 False。
DRY_RUN = False


# ============================================================
# 3. 输出路径
# ============================================================

roi_gpkg_path = out_dir / "paris_dsm_roi_bbox.gpkg"
out_gpkg_path = out_dir / "paris_bdtopo_buildings_clipped_to_dsm_roi.gpkg"

out_shp_dir = out_dir / "shapefile"
out_shp_dir.mkdir(parents=True, exist_ok=True)
out_shp_path = out_shp_dir / "paris_bdtopo_buildings_clipped_to_dsm_roi.shp"


# ============================================================
# 4. DSM ROI 函数
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

    return (
        min(lefts),
        min(bottoms),
        max(rights),
        max(tops),
    ), first_crs


def save_dsm_roi(bounds, crs):
    roi_gdf = gpd.GeoDataFrame(
        {"name": ["paris_dsm_roi"]},
        geometry=[box(*bounds)],
        crs=crs
    )

    roi_gdf.to_file(roi_gpkg_path, layer="dsm_roi", driver="GPKG")

    return roi_gdf


# ============================================================
# 5. Atom/XML 解析
# ============================================================

def fetch_atom(url, params=None):
    print()
    print("Fetching Atom:")
    print(url)

    if params is not None:
        print("Params:")
        print(params)

    response = requests.get(url, params=params, timeout=120)

    if response.status_code != 200:
        print(response.text[:1000])
        raise RuntimeError(f"Request failed. Status code: {response.status_code}")

    text = response.text

    if "<feed" not in text and "<entry" not in text:
        print(text[:1000])
        raise RuntimeError("Response does not look like Atom/XML.")

    return text, response.url


def local_name(tag):
    if "}" in tag:
        return tag.split("}", 1)[1]

    return tag


def parse_atom(xml_text):
    root = ET.fromstring(xml_text)

    entries = []
    feed_links = []

    for child in root:
        name = local_name(child.tag)

        if name == "link":
            feed_links.append(child.attrib)

        if name == "entry":
            title = ""
            links = []

            for elem in child.iter():
                elem_name = local_name(elem.tag)

                if elem_name == "title" and elem.text:
                    title = elem.text.strip()

                if elem_name == "link":
                    links.append(elem.attrib)

            entries.append({
                "title": title,
                "links": links,
            })

    return entries, feed_links


def find_next_link(feed_links):
    for link in feed_links:
        rel = link.get("rel", "")
        href = link.get("href", "")

        if rel == "next" and href:
            return href

    return None


def collect_entries_paginated(url, params=None, max_pages=30):
    all_entries = []
    page = 1
    current_url = url
    current_params = params

    while True:
        page_params = {} if current_params is None else dict(current_params)
        page_params["page"] = page
        page_params["limit"] = LIMIT

        xml_text, final_url = fetch_atom(current_url, params=page_params)
        entries, feed_links = parse_atom(xml_text)

        print(f"Page {page}: entries = {len(entries)}")

        all_entries.extend(entries)

        next_href = find_next_link(feed_links)

        if next_href:
            current_url = urljoin(final_url, next_href)
            current_params = None
            page += 1

            if page > max_pages:
                break

            continue

        if len(entries) < LIMIT:
            break

        page += 1

        if page > max_pages:
            break

    return all_entries


def best_link_from_entry(entry):
    links = entry.get("links", [])

    if len(links) == 0:
        return None

    scored = []

    for link in links:
        href = link.get("href", "")

        if not href:
            continue

        rel = link.get("rel", "")
        type_value = link.get("type", "")

        score = 0

        if rel in ["alternate", "enclosure"]:
            score += 5

        if "atom" in type_value.lower():
            score += 3

        if "download" in href.lower():
            score += 10

        if "resource" in href.lower():
            score += 4

        scored.append((score, href))

    if len(scored) == 0:
        return None

    scored = sorted(scored, reverse=True, key=lambda x: x[0])

    return scored[0][1]


def entry_text(entry):
    text = entry.get("title", "")

    for link in entry.get("links", []):
        text += " " + link.get("href", "")

    return text


def extract_years(text):
    years = [int(y) for y in re.findall(r"20\d{2}", text)]

    return years


def has_recent_year(text):
    years = extract_years(text)

    for year in years:
        if MIN_YEAR <= year <= MAX_YEAR:
            return True

    return False


def latest_year(text):
    years = extract_years(text)

    if len(years) == 0:
        return None

    return max(years)


# ============================================================
# 6. 选择子资源和下载文件
# ============================================================

def score_subresource(entry):
    text = entry_text(entry).upper()

    score = 0

    # 强制排除不该用的数据
    if "ADMINISTRATIF" in text:
        return -9999

    if "DIFF" in text or "DIFFERENTIEL" in text:
        return -9999

    if "D075" not in text and "PARIS" not in text:
        return -9999

    if not has_recent_year(text):
        return -9999

    # 区域
    if "D075" in text:
        score += 200

    if "PARIS" in text:
        score += 150

    # 数据类型
    if "TOUSTHEMES" in text or "TOUS_THEMES" in text:
        score += 80

    if "BATI" in text or "BATIMENT" in text:
        score += 80

    # 格式
    if "SHP" in text:
        score += 60

    if "GPKG" in text:
        score += 50

    # 坐标系
    if "LAMB93" in text or "LAMBERT93" in text or "2154" in text:
        score += 50

    # 数据源
    if "BDTOPO" in text or "BD_TOPO" in text:
        score += 40

    year = latest_year(text)

    if year is not None:
        score += (year - 2020) * 20

    return score


def choose_subresource_entry(entries):
    scored = []

    for entry in entries:
        link = best_link_from_entry(entry)

        if link is None:
            continue

        score = score_subresource(entry)
        scored.append((score, entry.get("title", ""), link, entry_text(entry)))

    scored = sorted(scored, reverse=True, key=lambda x: x[0])

    print()
    print("Top subresource candidates:")
    for score, title, link, text in scored[:30]:
        print(f"[score={score}] {title}")
        print(f"  {link}")

    valid_candidates = [item for item in scored if item[0] > 0]

    if len(valid_candidates) == 0:
        raise RuntimeError(
            "No recent Paris D075 BD TOPO subresource found. "
            "The printed candidates do not include a valid 2024–2026 D075 package."
        )

    best_score, best_title, best_link, _ = valid_candidates[0]

    if best_score < 200:
        raise RuntimeError(
            "Could not confidently select a recent Paris D075 BD TOPO subresource. "
            "Please paste the printed candidates."
        )

    print()
    print("Selected subresource:")
    print(f"[score={best_score}] {best_title}")
    print(best_link)

    return best_link


def score_file_entry(entry):
    text = entry_text(entry).upper()

    score = 0

    if "ADMINISTRATIF" in text:
        return -9999

    if "DIFF" in text or "DIFFERENTIEL" in text:
        return -9999

    if not has_recent_year(text):
        return -9999

    if ".7Z" in text:
        score += 80

    if ".ZIP" in text:
        score += 70

    if "DOWNLOAD" in text or "TELECHARGEMENT" in text:
        score += 40

    if "SHP" in text:
        score += 50

    if "GPKG" in text:
        score += 45

    if "D075" in text:
        score += 60

    if "PARIS" in text:
        score += 60

    if "BATI" in text or "BATIMENT" in text:
        score += 40

    if "TOUSTHEMES" in text or "TOUS_THEMES" in text:
        score += 40

    if "2154" in text or "LAMB93" in text:
        score += 30

    year = latest_year(text)

    if year is not None:
        score += (year - 2020) * 20

    return score


def choose_download_file_entry(entries):
    scored = []

    for entry in entries:
        link = best_link_from_entry(entry)

        if link is None:
            continue

        text = entry_text(entry).upper()

        if not (
            ".7Z" in text
            or ".ZIP" in text
            or "DOWNLOAD" in text
            or "TELECHARGEMENT" in text
        ):
            continue

        score = score_file_entry(entry)
        scored.append((score, entry.get("title", ""), link, entry_text(entry)))

    scored = sorted(scored, reverse=True, key=lambda x: x[0])

    print()
    print("Top file download candidates:")
    for score, title, link, text in scored[:30]:
        print(f"[score={score}] {title}")
        print(f"  {link}")

    valid_candidates = [item for item in scored if item[0] > 0]

    if len(valid_candidates) == 0:
        raise RuntimeError(
            "No recent downloadable ZIP/7Z file found in selected subresource. "
            "Please paste the printed candidates."
        )

    best_score, best_title, best_link, _ = valid_candidates[0]

    if best_score < 150:
        raise RuntimeError(
            "Could not confidently select a recent BD TOPO file. "
            "Please paste the printed candidates."
        )

    print()
    print("Selected file:")
    print(f"[score={best_score}] {best_title}")
    print(best_link)

    return best_link


def discover_download_url():
    resource_params = {
        "lang": "fre",
        "zone": TARGET_ZONE,
        "format": "SHP",
        "crs": TARGET_CRS_FILTER,
    }

    entries = collect_entries_paginated(BASE_RESOURCE_URL, params=resource_params)

    if len(entries) == 0:
        print("No entries with full filters. Trying without CRS filter.")

        resource_params = {
            "lang": "fre",
            "zone": TARGET_ZONE,
            "format": "SHP",
        }

        entries = collect_entries_paginated(BASE_RESOURCE_URL, params=resource_params)

    if len(entries) == 0:
        print("No entries with zone/format filters. Trying raw resource list.")

        entries = collect_entries_paginated(
            BASE_RESOURCE_URL,
            params={"lang": "fre"}
        )

    if len(entries) == 0:
        raise RuntimeError("No BDTOPO subresources returned.")

    subresource_link = choose_subresource_entry(entries)
    subresource_link = urljoin(BASE_RESOURCE_URL + "/", subresource_link)

    file_entries = collect_entries_paginated(
        subresource_link,
        params={"lang": "fre"}
    )

    if len(file_entries) == 0:
        raise RuntimeError("No file entries returned inside selected subresource.")

    download_link = choose_download_file_entry(file_entries)
    download_link = urljoin(subresource_link + "/", download_link)

    return download_link


# ============================================================
# 7. 下载和解压
# ============================================================

def filename_from_response(url, response):
    cd = response.headers.get("content-disposition")

    if cd:
        msg = Message()
        msg["content-disposition"] = cd
        filename = msg.get_filename()

        if filename:
            return unquote(filename)

    parsed = urlparse(response.url)
    filename = Path(parsed.path).name

    if filename:
        return unquote(filename)

    parsed = urlparse(url)
    filename = Path(parsed.path).name

    if filename:
        return unquote(filename)

    return "bdtopo_download"


def infer_archive_suffix(file_path):
    with open(file_path, "rb") as f:
        magic = f.read(8)

    if magic.startswith(b"PK"):
        return ".zip"

    if magic.startswith(b"7z\xbc\xaf\x27\x1c"):
        return ".7z"

    return ""


def download_file(url, out_folder):
    print()
    print("Downloading real file:")
    print(url)

    with requests.get(url, stream=True, timeout=300) as response:
        if response.status_code != 200:
            print(response.text[:1000])
            raise RuntimeError(f"Download failed. Status code: {response.status_code}")

        content_type = response.headers.get("content-type", "")
        filename = filename_from_response(url, response)

        out_path = out_folder / filename

        if out_path.exists() and out_path.stat().st_size > 0:
            print("File already exists, using existing file:")
            print(out_path)
            return out_path

        with open(out_path, "wb") as f:
            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\rDownloaded {pct:.1f}%", end="")

    print()

    if out_path.suffix.lower() not in [".zip", ".7z"]:
        inferred = infer_archive_suffix(out_path)

        if inferred:
            new_path = out_path.with_suffix(inferred)

            if new_path.exists():
                new_path.unlink()

            out_path.rename(new_path)
            out_path = new_path
        else:
            first_text = out_path.read_bytes()[:300].decode("utf-8", errors="ignore")
            print("Downloaded file content-type:", content_type)
            print("First 300 bytes:")
            print(first_text)
            raise RuntimeError("Downloaded file is not a ZIP/7Z archive.")

    print("Download complete:")
    print(out_path)

    return out_path


def extract_archive(archive_path, extract_folder):
    target_dir = extract_folder / archive_path.stem

    if target_dir.exists():
        print()
        print("Extraction folder already exists, removing old folder:")
        print(target_dir)
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("Extracting:")
    print(archive_path)

    suffix = archive_path.suffix.lower()

    if suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(target_dir)

    elif suffix == ".7z":
        try:
            subprocess.run(
                ["7z", "x", str(archive_path), f"-o{target_dir}", "-y"],
                check=True
            )
        except FileNotFoundError:
            raise RuntimeError(
                "This archive is .7z, but the '7z' command was not found. "
                "Install 7-Zip and add 7z.exe to PATH, or download a ZIP version."
            )

    else:
        raise RuntimeError(f"Unsupported archive type: {archive_path.suffix}")

    print("Extracted to:")
    print(target_dir)

    return target_dir


# ============================================================
# 8. 查找 BATIMENT 图层并裁剪
# ============================================================

def find_building_vector_files(folder):
    vector_files = []

    for pattern in ["*.shp", "*.gpkg", "*.geojson"]:
        vector_files.extend(folder.rglob(pattern))

    candidates = []

    for file_path in vector_files:
        text = str(file_path).upper()
        score = 0

        if "BATIMENT" in text:
            score += 200

        if "BATI" in text:
            score += 80

        if "BUILDING" in text:
            score += 80

        if "ADMINISTRATIF" in text:
            score -= 200

        if file_path.suffix.lower() == ".shp":
            score += 20

        if file_path.suffix.lower() == ".gpkg":
            score += 15

        if score > 0:
            candidates.append((score, file_path))

    candidates = sorted(candidates, reverse=True, key=lambda x: x[0])

    print()
    print("Building vector candidates:")
    for score, file_path in candidates[:30]:
        print(f"[score={score}] {file_path}")

    if len(candidates) == 0:
        print()
        print("Some vector files found:")
        for file_path in vector_files[:50]:
            print(file_path)

        raise RuntimeError("Could not find BATIMENT / BATI vector layer.")

    return candidates[0][1]


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
        raise RuntimeError("Building layer CRS is None.")

    if buildings.crs != target_crs:
        print(f"Reprojecting buildings to DSM CRS: {target_crs}")
        buildings = buildings.to_crs(target_crs)

    buildings = clean_gdf(buildings)

    print("Clipping to DSM ROI...")
    clipped = gpd.clip(buildings, roi_gdf)
    clipped = clean_gdf(clipped)

    if clipped.empty:
        raise RuntimeError("No buildings intersect DSM ROI after clipping.")

    return clipped


# ============================================================
# 9. 主程序
# ============================================================

def main():
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
    print("Discovering recent BD TOPO Paris D075 download URL...")
    download_url = discover_download_url()

    print()
    print("Final selected download URL:")
    print(download_url)

    if DRY_RUN:
        print()
        print("DRY_RUN=True, stop before downloading.")
        return

    archive_path = download_file(download_url, download_dir)

    extracted_folder = extract_archive(archive_path, extract_dir)

    building_vector = find_building_vector_files(extracted_folder)

    clipped = clip_buildings_to_roi(building_vector, roi_gdf, dsm_crs)

    print()
    print(f"Buildings after clipping: {len(clipped)}")

    print()
    print("Saving outputs...")
    clipped.to_file(out_gpkg_path, layer="buildings", driver="GPKG")
    clipped.to_file(out_shp_path, driver="ESRI Shapefile", encoding="utf-8")

    print()
    print("Done.")
    print(f"Output GeoPackage: {out_gpkg_path}")
    print(f"Output Shapefile: {out_shp_path}")
    print(f"DSM ROI bbox: {roi_gpkg_path}")

    print()
    print("Columns:")
    print(list(clipped.columns))

    for col in [
        "hauteur",
        "HAUTEUR",
        "altitude_minimale_sol",
        "altitude_maximale_toit",
        "ALTITUDE_MINIMALE_SOL",
        "ALTITUDE_MAXIMALE_TOIT",
    ]:
        if col in clipped.columns:
            print()
            print(f"Column found: {col}")
            print(clipped[col].describe())


if __name__ == "__main__":
    main()