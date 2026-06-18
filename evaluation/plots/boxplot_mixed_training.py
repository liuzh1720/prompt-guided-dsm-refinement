import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIG
# ============================================================

# Output folder for figures and summary csv
OUTPUT_DIR = Path("results/plots/mixed_training")

CSV_CONFIGS = [
    {
        "training_setting": "Hong Kong–Austin",
        "csv_path": "results/mixed/hk_austin/all/metrics_per_patch.csv",
        "eval_city": None,
    },
    {
        "training_setting": "Hong Kong–Paris",
        "csv_path": "results/mixed/hk_paris/all/metrics_per_patch.csv",
        "eval_city": None,
    },
    {
        "training_setting": "Austin–Paris",
        "csv_path": "results/mixed/austin_paris/all/metrics_per_patch.csv",
        "eval_city": None,
    },
    {
        "training_setting": "Hong Kong-only",
        "csv_path": "results/hk_only/hk/metrics_per_patch.csv",
        "eval_city": "Hong Kong",
    },
    {
        "training_setting": "Hong Kong-only",
        "csv_path": "results/hk_only/austin/metrics_per_patch.csv",
        "eval_city": "Austin",
    },
    {
        "training_setting": "Hong Kong-only",
        "csv_path": "results/hk_only/paris/metrics_per_patch.csv",
        "eval_city": "Paris",
    },
]

CITY_ORDER = ["Hong Kong", "Austin", "Paris"]

TRAINING_ORDER = [
    "Hong Kong-only",
    "Hong Kong–Austin",
    "Hong Kong–Paris",
    "Austin–Paris",
]

METRICS_TO_PLOT = ["mae", "r2"]

METRIC_LABELS = {
    "mae": "MAE (m)",
    "r2": "R²",
}

# Figure DPI
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


def load_one_csv(config: dict) -> pd.DataFrame:
    csv_path = Path(config["csv_path"])

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["mae", "r2"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(
            f"CSV is missing required columns: {missing_cols}\n"
            f"CSV path: {csv_path}\n"
            f"Actual columns: {list(df.columns)}"
        )

    df["training_setting"] = config["training_setting"]

    if "city" in df.columns:
        df["eval_city"] = df["city"].apply(normalize_city_name)
    else:
        if config.get("eval_city") is None:
            raise ValueError(
                f"CSV has no 'city' column. Please provide eval_city in CSV_CONFIGS.\n"
                f"CSV path: {csv_path}"
            )
        df["eval_city"] = normalize_city_name(config["eval_city"])

    if "patch_id" not in df.columns:
        if "name" in df.columns:
            df["patch_id"] = df["name"].astype(str)
        else:
            df["patch_id"] = np.arange(len(df)).astype(str)

    for col in ["mae", "rmse", "r2", "le90", "valid_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["mae", "r2"])

    keep_cols = [
        "training_setting",
        "eval_city",
        "patch_id",
        "mae",
        "r2",
    ]

    for optional_col in ["rmse", "le90", "valid_ratio"]:
        if optional_col in df.columns:
            keep_cols.append(optional_col)

    return df[keep_cols].copy()


def load_all_results(configs: list) -> pd.DataFrame:
    dfs = []

    for cfg in configs:
        df = load_one_csv(cfg)
        dfs.append(df)

        print(f"Loaded: {cfg['training_setting']}")
        print(f"  path: {cfg['csv_path']}")
        print("  city counts:")
        print(df["eval_city"].value_counts())
        print()

    all_df = pd.concat(dfs, ignore_index=True)

    all_df["eval_city"] = pd.Categorical(
        all_df["eval_city"],
        categories=CITY_ORDER,
        ordered=True
    )

    existing_training_order = [
        x for x in TRAINING_ORDER
        if x in set(all_df["training_setting"])
    ]

    other_training = [
        x for x in sorted(all_df["training_setting"].unique())
        if x not in existing_training_order
    ]

    final_training_order = existing_training_order + other_training

    all_df["training_setting"] = pd.Categorical(
        all_df["training_setting"],
        categories=final_training_order,
        ordered=True
    )

    return all_df


def nmad(values):
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.nan

    med = np.median(arr)
    return 1.4826 * np.median(np.abs(arr - med))


def save_summary_table(df: pd.DataFrame, output_path: Path):
    rows = []

    group_cols = ["training_setting", "eval_city"]

    for (training_setting, eval_city), g in df.groupby(group_cols, observed=True):
        row = {
            "training_setting": training_setting,
            "eval_city": eval_city,
            "n_patches": len(g),
        }

        for metric in ["mae", "rmse", "r2", "le90", "valid_ratio"]:
            if metric not in g.columns:
                continue

            values = g[metric].dropna().to_numpy()

            row[f"{metric}_mean"] = np.mean(values) if len(values) else np.nan
            row[f"{metric}_std"] = np.std(values) if len(values) else np.nan
            row[f"{metric}_median"] = np.median(values) if len(values) else np.nan
            row[f"{metric}_nmad"] = nmad(values) if len(values) else np.nan
            row[f"{metric}_min"] = np.min(values) if len(values) else np.nan
            row[f"{metric}_max"] = np.max(values) if len(values) else np.nan

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved summary table: {output_path}")


def make_boxplot_by_training_and_city(
    df: pd.DataFrame,
    metric: str,
    output_path: Path
):
    """
    One box for each training-setting + city combination.
    Useful for a full overview.
    """
    plot_df = df.dropna(subset=[metric]).copy()

    labels = []
    data = []

    training_settings = [
        x for x in plot_df["training_setting"].cat.categories
        if x in set(plot_df["training_setting"])
    ]

    cities = [
        x for x in CITY_ORDER
        if x in set(plot_df["eval_city"].astype(str))
    ]

    for training in training_settings:
        for city in cities:
            values = plot_df[
                (plot_df["training_setting"] == training) &
                (plot_df["eval_city"].astype(str) == city)
            ][metric].dropna().to_numpy()

            if len(values) == 0:
                continue

            labels.append(f"{training}\n{city}")
            data.append(values)

    if not data:
        print(f"No data for metric: {metric}")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(data) * 0.8), 6))

    ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        patch_artist=False
    )

    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"Patch-level {METRIC_LABELS.get(metric, metric)} distribution")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_path}")


def make_boxplot_one_city(
    df: pd.DataFrame,
    metric: str,
    city: str,
    output_path: Path
):
    """
    For one evaluation city, compare all training settings.
    This is usually the cleanest figure for slides.
    """
    plot_df = df[
        (df["eval_city"].astype(str) == city)
    ].dropna(subset=[metric]).copy()

    if len(plot_df) == 0:
        print(f"No data for city={city}, metric={metric}")
        return

    labels = []
    data = []

    training_settings = [
        x for x in plot_df["training_setting"].cat.categories
        if x in set(plot_df["training_setting"])
    ]

    for training in training_settings:
        values = plot_df[
            plot_df["training_setting"] == training
        ][metric].dropna().to_numpy()

        if len(values) == 0:
            continue

        labels.append(training)
        data.append(values)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        patch_artist=False
    )

    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{city}: patch-level {METRIC_LABELS.get(metric, metric)}")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_path}")


def make_boxplot_one_training(
    df: pd.DataFrame,
    metric: str,
    training_setting: str,
    output_path: Path
):
    """
    For one model, compare its performance across cities.
    Useful for explaining region-dependent performance.
    """
    plot_df = df[
        (df["training_setting"].astype(str) == training_setting)
    ].dropna(subset=[metric]).copy()

    if len(plot_df) == 0:
        print(f"No data for training={training_setting}, metric={metric}")
        return

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
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        patch_artist=False
    )

    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{training_setting}: patch-level {METRIC_LABELS.get(metric, metric)}")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

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

    merged_csv_path = OUTPUT_DIR / "merged_patch_level_metrics.csv"
    all_df.to_csv(merged_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved merged csv: {merged_csv_path}")

    summary_csv_path = OUTPUT_DIR / "patch_level_metric_summary.csv"
    save_summary_table(all_df, summary_csv_path)

    # --------------------------------------------------------
    # A. Full overview boxplots
    # --------------------------------------------------------
    for metric in METRICS_TO_PLOT:
        fig_path = OUTPUT_DIR / f"boxplot_{metric}_all_models_all_cities.png"
        make_boxplot_by_training_and_city(all_df, metric, fig_path)

    # --------------------------------------------------------
    # B. Per-city comparison boxplots
    # One figure per city and per metric
    # --------------------------------------------------------
    for metric in METRICS_TO_PLOT:
        for city in CITY_ORDER:
            fig_path = OUTPUT_DIR / f"boxplot_{metric}_{city.replace(' ', '_')}_compare_models.png"
            make_boxplot_one_city(all_df, metric, city, fig_path)

    # --------------------------------------------------------
    # C. Per-model city comparison boxplots
    # One figure per training setting and per metric
    # --------------------------------------------------------
    training_settings = [
        x for x in all_df["training_setting"].cat.categories
        if x in set(all_df["training_setting"])
    ]

    for metric in METRICS_TO_PLOT:
        for training in training_settings:
            safe_name = (
                training
                .replace(" ", "_")
                .replace("–", "_")
                .replace("+", "_")
            )
            fig_path = OUTPUT_DIR / f"boxplot_{metric}_{safe_name}_compare_cities.png"
            make_boxplot_one_training(all_df, metric, training, fig_path)

    print("\nDone.")
    print(f"All figures saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()