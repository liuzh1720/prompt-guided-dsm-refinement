# Paris Data Preparation Scripts

These scripts document the Paris data preparation pipeline used in the thesis.
They contain hardcoded paths from the original working environment and may
require adaptation before running.

## Scripts

1. `prepare_paris_rgb_mns_patches.py`
   - Downloads and aligns RGB (BD ORTHO) and DSM (MNS LiDAR HD) data
   - Generates 0.6m aligned patches
   - Requires: download links file, GDAL, rasterio

2. `download_bdtopo_paris_buildings.py`
   - Downloads IGN BD TOPO building footprints

3. `clip_extracted_bdtopo_paris_buildings.py`
   - Clips BD TOPO footprints to the DSM region of interest

4. `split_paris_bdtopo_footprints_by_patch.py`
   - Splits clipped footprints into per-patch GeoJSON/GeoPackage files

## Usage

Adapt the hardcoded paths and download URLs before running.
See `docs/DATA_PREPARATION.md` for the overall data preparation workflow.
