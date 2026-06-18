from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIG
# ============================================================

# Output directory (created automatically)
OUTPUT_DIR = Path("results/plots/hk_vs_baseline")

# CSV configs point to evaluation outputs in results/.
# Run evaluation first, then run this script.
CSV_CONFIGS = [
    {
        "model": "HK-only",
        "eval_city": "Hong Kong",
        "csv_path": "results/hk_only/hk/metrics_per_patch.csv",
    },
    {
        "model": "HK-only",
        "eval_city": "Austin",
        "csv_path": "results/hk_only/austin/metrics_per_patch.csv",
    },
    {
        "model": "HK-only",
        "eval_city": "Paris",
        "csv_path": "results/hk_only/paris/metrics_per_patch.csv",
    },
    {
        "model": "PromptDA",
        "eval_city": "Hong Kong",
        "csv_path": "results/baseline/hk/metrics_per_patch.csv",
    },
    {
        "model": "PromptDA",
        "eval_city": "Austin",
        "csv_path": "results/baseline/austin/metrics_per_patch.csv",
    },
    {
        "model": "PromptDA",
        "eval_city": "Paris",
        "csv_path": "results/baseline/paris/metrics_per_patch.csv",
    },
]

CITY_ORDER = ["Hong Kong", "Austin", "Paris"]
MODEL_ORDER = ["HK-only", "PromptDA"]

METRICS_TO_PLOT = ["mae", "r2"]

METRIC_LABELS = {
    "mae": "MAE (m)",
    "r2": "R²",
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

    required_cols = ["mae", "r2"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(
            f"CSV is missing required columns: {missing_cols}\n"
            f"CSV path: {csv_path}\n"
            f"Actual columns: {list(df.columns)}"
        )

    df["model"] = config["model"]
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
        "model",
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

        print(f"Loaded: {cfg['model']} | {cfg['eval_city']}")
        print(f"  Path: {cfg['csv_path']}")
        print(f"  Patches: {len(df)}")
        print()

    all_df = pd.concat(dfs, ignore_index=True)

    all_df["eval_city"] = pd.Categorical(
        all_df["eval_city"],
        categories=CITY_ORDER,
        ordered=True
    )

    all_df["model"] = pd.Categorical(
        all_df["model"],
        categories=MODEL_ORDER,
        ordered=True
    )

    return all_df


def save_summary_table(df: pd.DataFrame, output_path: Path):
    rows = []

    for (city, model), g in df.groupby(["eval_city", "model"], observed=True):
        row = {
            "eval_city": city,
            "model": model,
            "n_patches": len(g),
        }

        for metric in ["mae", "rmse", "r2", "le90", "valid_ratio"]:
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


def make_grouped_boxplot(
    df: pd.DataFrame,
    metric: str,
    output_path: Path
):
    """
    Grouped boxplot:
    For each city, compare HK-only and PromptDA.
    X-axis only shows city names.
    Legend shows model names.
    """
    plot_df = df.dropna(subset=[metric]).copy()

    fig, ax = plt.subplots(figsize=(8.5, 5))

    # positions for grouped boxplots
    city_centers = np.arange(len(CITY_ORDER)) * 3.0
    offset = 0.45
    width = 0.7

    model_positions = {
        "HK-only": city_centers - offset,
        "PromptDA": city_centers + offset,
    }

    model_styles = {
        "HK-only": {"facecolor": "#d9edf7"},
        "PromptDA": {"facecolor": "#f2dede"},
    }

    legend_handles = []

    for model in MODEL_ORDER:
        data = []
        positions = []

        for i, city in enumerate(CITY_ORDER):
            values = plot_df[
                (plot_df["eval_city"].astype(str) == city) &
                (plot_df["model"].astype(str) == model)
            ][metric].dropna().to_numpy()

            if len(values) == 0:
                continue

            data.append(values)
            positions.append(model_positions[model][i])

        if not data:
            continue

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            showmeans=True,
            patch_artist=True,
            manage_ticks=False
        )

        # style boxes
        for patch in bp["boxes"]:
            patch.set_facecolor(model_styles[model]["facecolor"])
            patch.set_alpha(0.8)

        # keep other parts black for clarity
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color("black")

        for mean in bp["means"]:
            mean.set_marker("^")
            mean.set_markerfacecolor("#2ca02c")
            mean.set_markeredgecolor("#2ca02c")
            mean.set_markersize(8)

        # create legend handle
        legend_handles.append(
            plt.Line2D(
                [0], [0],
                color="black",
                marker="s",
                markersize=10,
                markerfacecolor=model_styles[model]["facecolor"],
                linestyle="None",
                label=model
            )
        )

    ax.set_xticks(city_centers)
    ax.set_xticklabels(CITY_ORDER)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"HK-only vs PromptDA: patch-level {METRIC_LABELS.get(metric, metric)}")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.legend(handles=legend_handles, title="Model", loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_path}")


def make_per_city_boxplots(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path
):
    """
    Optional: one figure per city.
    Each figure compares HK-only vs PromptDA.
    """
    model_styles = {
        "HK-only": {"facecolor": "#d9edf7"},
        "PromptDA": {"facecolor": "#f2dede"},
    }

    for city in CITY_ORDER:
        plot_df = df[
            df["eval_city"].astype(str) == city
        ].dropna(subset=[metric]).copy()

        if len(plot_df) == 0:
            continue

        data = []
        labels = []

        for model in MODEL_ORDER:
            values = plot_df[
                plot_df["model"].astype(str) == model
            ][metric].dropna().to_numpy()

            if len(values) == 0:
                continue

            data.append(values)
            labels.append(model)

        if not data:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))

        bp = ax.boxplot(
            data,
            labels=labels,
            showmeans=True,
            patch_artist=True
        )

        for patch, model in zip(bp["boxes"], labels):
            patch.set_facecolor(model_styles[model]["facecolor"])
            patch.set_alpha(0.8)

        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color("black")

        for mean in bp["means"]:
            mean.set_marker("^")
            mean.set_markerfacecolor("#2ca02c")
            mean.set_markeredgecolor("#2ca02c")
            mean.set_markersize(8)

        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(f"{city}: HK-only vs PromptDA {METRIC_LABELS.get(metric, metric)}")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

        plt.tight_layout()

        safe_city = city.replace(" ", "_")
        fig_path = output_dir / f"boxplot_{metric}_{safe_city}_hk_only_vs_promptda.png"

        plt.savefig(fig_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved figure: {fig_path}")


# ============================================================
# 3. Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_df = load_all_results(CSV_CONFIGS)

    merged_csv_path = OUTPUT_DIR / "merged_hk_only_vs_promptda_patch_level_metrics.csv"
    all_df.to_csv(merged_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved merged csv: {merged_csv_path}")

    summary_csv_path = OUTPUT_DIR / "hk_only_vs_promptda_metric_summary.csv"
    save_summary_table(all_df, summary_csv_path)

    for metric in METRICS_TO_PLOT:
        fig_path = OUTPUT_DIR / f"boxplot_{metric}_hk_only_vs_promptda_by_city.png"
        make_grouped_boxplot(all_df, metric, fig_path)

    # Optional cleaner per-city figures
    for metric in METRICS_TO_PLOT:
        make_per_city_boxplots(all_df, metric, OUTPUT_DIR)

    print("\nDone.")
    print(f"All figures saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()