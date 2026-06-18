# Results

This directory stores evaluation results as metrics CSV files and metadata.
Large prediction GeoTIFFs and PNGs are excluded from Git (see `.gitignore`).

## Expected structure per run

```
results/{experiment}/{city}/
├── metrics_per_patch.csv
├── metrics_summary.csv
├── run_metadata.json
├── used_eval_split.csv
└── README.md
```

## Example metadata

See `example_metadata.json` for the expected `run_metadata.json` format.
