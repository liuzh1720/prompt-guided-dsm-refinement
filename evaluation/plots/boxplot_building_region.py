from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIG
# ============================================================

OUTPUT_DIR = Path("results/plots/building_region")

CSV_CONFIGS = [
    {
        "model": "HK-only",
        "eval_city": "Hong Kong",
        "csv_path": "results/building_region/hk/building_metrics_per_patch.csv",
    },
    {
        "model": "HK-only",
        "eval_city": "Austin",
        "csv_path": "results/building_region/austin/building_metrics_per_patch.csv",
    },
    {
        "model": "HK-only",
        "eval_city": "Paris",
        "csv_path": "results/building_region/paris/building_metrics_per_patch.csv",
    },
]

CITY_ORDER = ["Hong Kong", "Austin", "Paris"]

METRICS_TO_PLOT = ["mae_building", "r2_building"]

METRIC_LABELS = {
    "mae_building": "Building-region MAE (m)",
    "r2_building": "Building-region R²",
}

DPI = 300


# ============================================================
# 2. Helper functions
# ============================================================

def normalize_city_name(x: str) -> str:
    x = str(x).strip()

    mapping = {
        "HK": "Hong Kong",
        "HongKong": "Hong Kong",
        "Hong Kong": "Hong Kong",
        "Austin": "Austin",
        "Paris": "Paris",
    }

    return mapping.get(x, x)


def nmad(values):
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.nan

    med = np.median(arr)
    return float(1.4826 * np.median(np.abs(arr - med)))


def load_one_csv(config: dict) -> pd.DataFrame:
    csv_path = Path(config["csv_path"])

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = [
        "mae_building",
        "rmse_building",
        "r2_building",
        "le90_building",
        "building_valid_ratio",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(
            f"CSV is missing required building-region columns: {missing_cols}\n"
            f"CSV path: {csv_path}\n"
            f"Actual columns: {list(df.columns)}"
        )

    out_df = pd.DataFrame(index=df.index)

    out_df["model"] = config["model"]
    out_df["eval_city"] = normalize_city_name(config["eval_city"])

    if "name" in df.columns:
        out_df["patch_id"] = df["name"].astype(str)
    elif "patch_id" in df.columns:
        out_df["patch_id"] = df["patch_id"].astype(str)
    else:
        out_df["patch_id"] = np.arange(len(df)).astype(str)

    metric_cols = [
        "mae_building",
        "rmse_building",
        "r2_building",
        "le90_building",
        "building_valid_ratio",
        "building_pixel_ratio",
        "building_pixel_count",
    ]

    for col in metric_cols:
        if col in df.columns:
            out_df[col] = pd.to_numeric(df[col], errors="coerce")

    out_df = out_df.replace([np.inf, -np.inf], np.nan)
    out_df = out_df.dropna(subset=["mae_building", "r2_building"])

    print(f"Loaded: {config['model']} | {config['eval_city']}")
    print(f"  Path: {csv_path}")
    print(f"  Valid rows: {len(out_df)}")
    print("  eval_city value counts:")
    print(out_df["eval_city"].value_counts())
    print()

    return out_df


def load_all_results(configs: list) -> pd.DataFrame:
    dfs = []

    for cfg in configs:
        df = load_one_csv(cfg)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    all_df["eval_city"] = pd.Categorical(
        all_df["eval_city"],
        categories=CITY_ORDER,
        ordered=True
    )

    return all_df


def save_summary_table(df: pd.DataFrame, output_path: Path):
    rows = []

    for city, g in df.groupby("eval_city", observed=True):
        row = {
            "model": "HK-only",
            "eval_city": city,
            "n_patches": len(g),
        }

        for metric in [
            "mae_building",
            "rmse_building",
            "r2_building",
            "le90_building",
            "building_valid_ratio",
            "building_pixel_ratio",
            "building_pixel_count",
        ]:
            if metric not in g.columns:
                continue

            values = g[metric].dropna().to_numpy()

            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_std"] = float(np.std(values)) if len(values) else np.nan
            row[f"{metric}_median"] = float(np.median(values)) if len(values) else np.nan
            row[f"{metric}_nmad"] = nmad(values) if len(values) else np.nan
            row[f"{metric}_min"] = float(np.min(values)) if len(values) else np.nan
            row[f"{metric}_max"] = float(np.max(values)) if len(values) else np.nan

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved summary table: {output_path}")


def make_boxplot_by_city(
    df: pd.DataFrame,
    metric: str,
    output_path: Path
):
    plot_df = df.dropna(subset=[metric]).copy()

    labels = []
    data = []

    for city in CITY_ORDER:
        values = plot_df[
            plot_df["eval_city"].astype(str) == city
        ][metric].dropna().to_numpy()

        if len(values) == 0:
            continue

        labels.append(city)
        data.append(values)

    if not data:
        print(f"No data for metric: {metric}")
        print("Available eval_city values:")
        print(plot_df["eval_city"].value_counts(dropna=False))
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        patch_artist=False
    )

    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"HK-only: patch-level {METRIC_LABELS.get(metric, metric)}")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_path}")


def make_scatter_mae_r2(df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))

    has_data = False

    for city in CITY_ORDER:
        g = df[df["eval_city"].astype(str) == city]

        if len(g) == 0:
            continue

        has_data = True

        ax.scatter(
            g["mae_building"],
            g["r2_building"],
            label=city,
            alpha=0.7,
            s=22
        )

    ax.set_xlabel("Building-region MAE (m)")
    ax.set_ylabel("Building-region R²")
    ax.set_title("HK-only: building-region MAE vs R²")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    if has_data:
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_path}")


# ============================================================
# 3. Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_df = load_all_results(CSV_CONFIGS)

    print("\nCombined dataframe check:")
    print(all_df.head())
    print(all_df["eval_city"].value_counts(dropna=False))

    merged_csv_path = OUTPUT_DIR / "merged_hk_only_building_region_patch_level_metrics.csv"
    all_df.to_csv(merged_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved merged csv: {merged_csv_path}")

    summary_csv_path = OUTPUT_DIR / "hk_only_building_region_metric_summary.csv"
    save_summary_table(all_df, summary_csv_path)

    for metric in METRICS_TO_PLOT:
        fig_path = OUTPUT_DIR / f"boxplot_{metric}_hk_only_by_city.png"
        make_boxplot_by_city(all_df, metric, fig_path)

    scatter_path = OUTPUT_DIR / "scatter_building_mae_vs_r2_hk_only_by_city.png"
    make_scatter_mae_r2(all_df, scatter_path)

    print("\nDone.")
    print(f"All figures saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()