import argparse
import os
import sys

import numpy as np
import pandas as pd

CLEAN = "Clean"
EPS_METRIC = 5e-4
EPS_ENERGY_REL = 0.10

EQUIVALENT_COST_PAIRS = [
    ("DYNAMIC_META_EQUAL_START", "DYNAMIC_META_CALIBRATED_START"),
    ("DYNAMIC_HEUR_EQUAL_START", "DYNAMIC_HEUR_CALIBRATED_START"),
]


def check(df: pd.DataFrame) -> int:
    issues = 0

    def warn(tag: str, msg: str) -> None:
        nonlocal issues
        issues += 1
        print(f"  [{tag}] {msg}")

    for dataset, ds in df.groupby("dataset"):
        print(f"\n=== {dataset} ===")

        ratio = ds["anomaly_ratio"].iloc[0] if "anomaly_ratio" in ds else np.nan
        if np.isnan(ratio):
            warn("NO-BASELINE", "anomaly_ratio missing -- rerun with the current runner")
        else:
            trivial_f1 = 2 * ratio / (1 + ratio)
            print(
                f"  anomaly ratio={ratio:.4f}  random AUC-PR={ratio:.4f}  "
                f"all-positive F1={trivial_f1:.4f}"
            )

        clean = ds[ds["experiment_case"] == CLEAN]

        for _, row in clean.iterrows():
            strat = row["strategy"]
            if row.get("AUC-ROC", 1.0) < 0.5:
                warn(
                    "INVERTED",
                    f"{strat}: AUC-ROC={row['AUC-ROC']:.3f} < 0.5 -- the score ranks "
                    "anomalies below normal points (sign/normalisation bug)",
                )
            if not np.isnan(ratio) and row.get("AUC-PR", 1.0) <= ratio + EPS_METRIC:
                warn(
                    "AT-CHANCE",
                    f"{strat}: AUC-PR={row['AUC-PR']:.3f} <= random baseline {ratio:.3f}",
                )
            if (
                not np.isnan(ratio)
                and abs(row.get("PA-F1", 0) - 2 * ratio / (1 + ratio)) < EPS_METRIC
            ):
                warn(
                    "TRIVIAL-F1",
                    f"{strat}: PA-F1={row['PA-F1']:.4f} equals the all-positive "
                    "classifier -- the metric measures nothing here",
                )
            if "VUS-PR" in row and abs(row["VUS-PR"] - row["AUC-PR"]) < EPS_METRIC:
                warn(
                    "VUS==AUC",
                    f"{strat}: VUS-PR ({row['VUS-PR']:.4f}) == AUC-PR "
                    f"({row['AUC-PR']:.4f}) -- VUS carries no extra information",
                )

        for case, sub in ds[ds["experiment_case"] != CLEAN].groupby("experiment_case"):
            for _, row in sub.iterrows():
                base = clean[clean["strategy"] == row["strategy"]]
                if base.empty:
                    continue
                delta = row.get("AUC-PR", np.nan) - base["AUC-PR"].iloc[0]
                if delta > EPS_METRIC:
                    warn(
                        "FAILURE-BETTER",
                        f"{row['strategy']} under '{case}': AUC-PR "
                        f"{row['AUC-PR']:.3f} > clean {base['AUC-PR'].iloc[0]:.3f} "
                        f"(+{delta:.3f}) -- degrading detectors improved the result",
                    )

        if "strategy_eval_energy_kwh" in ds.columns and not clean.empty:
            e = clean.set_index("strategy")["strategy_eval_energy_kwh"]
            for a, b in EQUIVALENT_COST_PAIRS:
                if a in e.index and b in e.index and max(e[a], e[b]) > 0:
                    rel = abs(e[a] - e[b]) / max(e[a], e[b])
                    if rel > EPS_ENERGY_REL:
                        warn(
                            "ENERGY-NOISE",
                            f"{a} ({e[a]:.3e}) vs {b} ({e[b]:.3e}) differ by "
                            f"{rel:.0%} although they perform identical work -- "
                            "the energy measurement is dominated by noise",
                        )

        if "Recovery_Rate" in ds.columns:
            rec = ds.dropna(subset=["Recovery_Rate"])
            for strat, sub in rec.groupby("strategy"):
                rate = sub["Recovery_Rate"].mean()
                if rate < 0.999:
                    print(
                        f"  [INFO] {strat}: recovered in {rate:.0%} of suppressed "
                        f"windows ({int(sub['Recovery_Windows'].sum())} windows) -- "
                        "report this next to R_batches, the mean covers only "
                        "windows that did recover"
                    )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check a results directory; exits 1 if any issue is found."
    )
    parser.add_argument("results_dir")
    args = parser.parse_args()

    path = os.path.join(args.results_dir, "results.csv")
    if not os.path.exists(path):
        sys.exit(f"No results.csv in {args.results_dir}")

    manifest = os.path.join(args.results_dir, "run_manifest.yaml")
    status = "present" if os.path.exists(manifest) else "MISSING -- run provenance unknown"
    print(f"manifest: {status}")

    issues = check(pd.read_csv(path))
    print(f"\n{'=' * 60}\n{issues} issue(s) found.")
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
