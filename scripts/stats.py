import argparse

import pandas as pd
from scipy.stats import wilcoxon

from src.utils.labels import abbreviate


def aggregate_and_test(csv_paths, metric: str) -> None:
    combined = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    clean = combined[combined["experiment_case"] == "Clean"]
    summary = clean.groupby(["dataset", "strategy"])[metric].agg(["mean", "std"]).reset_index()
    mean = summary.pivot(index="dataset", columns="strategy", values="mean")
    std = summary.pivot(index="dataset", columns="strategy", values="std")

    table = pd.DataFrame(
        {
            abbreviate(c): mean[c].map("{:.4f}".format) + " ± " + std[c].map("{:.4f}".format)
            for c in mean.columns
        }
    )
    print(f"=== {metric}: mean ± std over {len(csv_paths)} run(s) ===")
    print(table.to_string())

    dynamic = [s for s in mean.columns if s.startswith("DYNAMIC_")]
    static = [s for s in mean.columns if not s.startswith("DYNAMIC_")]
    if len(mean) < 6:
        print(f"\nWarning: only {len(mean)} dataset(s); the Wilcoxon test has little power.")

    rows = []
    for dyn in dynamic:
        for stat in static:
            diff = mean[dyn].mean() - mean[stat].mean()
            try:
                w, p = wilcoxon(mean[dyn], mean[stat])
            except ValueError:
                w, p = float("nan"), 1.0
            rows.append(
                {
                    "Dynamic": abbreviate(dyn),
                    "Static": abbreviate(stat),
                    "W": f"{w:.1f}",
                    "p-value": f"{p:.4f}",
                    "Sig.": "*" if p < 0.05 else "",
                    "Mean diff": f"{diff:+.4f}",
                }
            )

    print(f"\n=== Wilcoxon signed-rank over datasets: dynamic vs static ({metric}) ===")
    print(
        pd.DataFrame(rows).to_string(index=False) if rows else "No dynamic/static pairs to compare."
    )
    print("\n* p < 0.05")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate results.csv files from repeated runs and test significance."
    )
    parser.add_argument("csv_paths", nargs="+")
    parser.add_argument("--metric", default="AUC-PR")
    args = parser.parse_args()
    aggregate_and_test(args.csv_paths, args.metric)
