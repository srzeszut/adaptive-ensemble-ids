import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.labels import abbreviate

DEFAULT_METRICS = ["AUC-ROC", "VUS-PR", "Standard-F1"]


def boxplot(df: pd.DataFrame, metric: str, path: str) -> None:
    order = df.groupby("strategy")[metric].median().sort_values(ascending=False).index
    plt.figure(figsize=(12, 6))
    bp = plt.boxplot(
        [df.loc[df["strategy"] == s, metric].dropna() for s in order],
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        widths=0.6,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
    )
    cmap = plt.get_cmap("Blues_r")
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(cmap(i / len(order)))
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)
    plt.xticks(range(1, len(order) + 1), [abbreviate(s) for s in order], rotation=35, ha="right")
    plt.xlabel("Strategy")
    plt.ylabel(metric)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def recovery_barplot(df: pd.DataFrame, path: str) -> None:
    df = df.dropna(subset=["Recovery_Batches"]).copy()
    if df.empty:
        return
    df["failure"] = df["experiment_case"].str.extract(r"\((.*?)\)", expand=False)
    df["strategy"] = df["strategy"].map(abbreviate)
    grouped = df.groupby(["failure", "strategy"])["Recovery_Batches"]
    grouped.mean().unstack().plot.bar(
        yerr=grouped.std().unstack(), figsize=(10, 6), colormap="Blues_r", edgecolor="black"
    )
    plt.xlabel("Failure mode")
    plt.ylabel("Recovery batches")
    plt.xticks(rotation=0)
    plt.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Box plots of ensemble metrics from a results directory."
    )
    parser.add_argument("results_dir")
    parser.add_argument("--case", choices=["clean", "failures"], default="clean")
    parser.add_argument(
        "--strategies",
        nargs="+",
        help="Full strategy names to plot (default: all except BASELINE_*).",
    )
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.results_dir, "results.csv"))
    clean = df["experiment_case"] == "Clean"
    df = df[clean if args.case == "clean" else ~clean]
    if args.strategies:
        df = df[df["strategy"].isin(args.strategies)]
    else:
        df = df[~df["strategy"].str.startswith("BASELINE_")]
    if df.empty:
        raise SystemExit("No rows match the selected case and strategies.")

    out_dir = os.path.join(args.results_dir, "analysis_plots")
    os.makedirs(out_dir, exist_ok=True)
    for metric in args.metrics:
        boxplot(
            df, metric, os.path.join(out_dir, f"{args.case}_boxplot_{metric.replace('-', '_')}.png")
        )
    if args.case == "failures":
        recovery_barplot(df, os.path.join(out_dir, "recovery_batches_barplot.png"))
    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
