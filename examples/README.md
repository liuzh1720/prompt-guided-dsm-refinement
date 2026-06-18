# Example Data

Place a small set of example patches here for quick testing.

Expected structure:
```
examples/
├── sample_patch/
│   ├── rgb.tif          # 3-band RGB orthophoto patch
│   ├── dsm.tif          # Reference DSM patch
│   ├── prompt.tif       # Coarse DSM prompt
│   └── buildings.geojson # Building footprints
```

Example data should be small enough for GitHub (< 1 MB per file recommended).
Choose a representative patch with visible buildings.

Note: Example files are excluded from the `.gitignore` data rules
via the `!examples/**` exception.
