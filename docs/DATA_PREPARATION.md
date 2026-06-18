# Data Preparation

## Data Sources

### Hong Kong
- **RGB**: True Digital Orthophoto (TDOP), Lands Department HKSAR, 0.25m resampled to 0.6m
  - Download: CSDI Portal (https://portal.csdi.gov.hk/)
- **DSM**: Territory-wide airborne LiDAR survey, CEDD HKSAR, 2019-2020
  - Download: CEDD Spatial Data Portal (https://sdportal.cedd.gov.hk/)
- **Building footprints**: OpenStreetMap

### Austin
- **RGB**: NAIP orthophotos, USDA, 2020, 0.6m (EPSG:26914)
  - Download: USGS EarthExplorer (https://earthexplorer.usgs.gov/)
- **DSM**: USGS 3DEP LiDAR (TX Central B1 2017), 0.5m resampled to 0.6m
  - Download: USGS National Map (https://apps.nationalmap.gov/downloader/)
- **Building footprints**: OpenStreetMap

### Paris
- **RGB**: BD ORTHO, IGN, 2024 vintage, Department 75, 0.20m resampled to 0.6m
  - Download: IGN Géoservices (https://geoservices.ign.fr/bdortho)
- **DSM**: MNS LiDAR HD, IGN, 0.50m resampled to 0.6m (EPSG:2154)
  - Download: data.geopf.fr
- **Building footprints**: IGN BD TOPO v3.5

## Directory Layout

Organize data as follows under `data/`:

```
data/
├── HK/
│   ├── rgb/                # RGB orthophoto patches (.tif)
│   ├── dsm/                # Reference DSM patches (.tif)
│   ├── prompt_dsm_30m/     # Coarse DSM prompts (generated)
│   ├── building_footprints/ # Per-patch building footprint files
│   └── fixed_evaluation_group/
│       └── split.csv       # Original HK fixed split
├── Austin/
│   ├── rgb/
│   ├── dsm/
│   ├── prompt_dsm_30m/
│   └── building_footprints/
└── Paris/
    ├── rgb/
    ├── dsm/
    ├── prompt_dsm_30m/
    └── building_footprints/
```

## Processing Steps

1. Align RGB and DSM to matching grid (0.6m, same extent)
2. Generate patches (1036×1036 pixels, ~622m per side)
3. Generate coarse DSM prompts:
   ```bash
   python preprocessing/generate_coarse_prompt.py --input-dir data/HK/dsm --output-dir data/HK/prompt_dsm_30m
   ```
4. Prepare building footprints (per-patch GeoJSON/GeoPackage)
5. Generate split CSVs:
   ```bash
   python preprocessing/create_splits.py --data-root data --hk-split data/HK/fixed_evaluation_group/split.csv
   ```

## Paris Pipeline

Paris data preparation scripts are preserved in `preprocessing/paris/`:
- `prepare_paris_rgb_mns_patches.py` - Generate aligned RGB+DSM patches
- `download_bdtopo_paris_buildings.py` - Download BD TOPO footprints
- `clip_extracted_bdtopo_paris_buildings.py` - Clip to DSM ROI
- `split_paris_bdtopo_footprints_by_patch.py` - Split by patch

These scripts require local data files and download URLs. Adapt paths before running.

## License Notes

- NAIP (Austin RGB): Public domain (USDA)
- 3DEP (Austin DSM): Public domain (USGS)
- OpenStreetMap (HK, Austin footprints): ODbL, attribution required
- IGN data (Paris): Check IGN license terms before redistribution
- HKSAR data (HK RGB, DSM): Check terms of use
