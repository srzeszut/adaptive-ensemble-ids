import argparse
import itertools
import os

import pandas as pd
from hydra import compose, initialize

from src.utils.silence import apply_silence

apply_silence()

from src.pipeline.runner import run_pipeline

NORMALIZATIONS = ["mad", "robust", "percentile_minmax", "zscore_sigmoid"]
TEMPERATURES = [0.10, 0.25, 0.50]
WEIGHT_COMBINATIONS = [
    (0.80, 0.00, 0.20),
    (0.50, 0.25, 0.25),
    (1.00, 0.00, 0.00),
    (0.40, 0.00, 0.60),
    (0.33, 0.33, 0.34),
]
PARAMS = ["Normalization", "Temp", "Sens_W", "Comp_W", "Disc_W"]
METRICS = ["VUS-ROC", "VUS-PR", "AUC-ROC", "AUC-PR", "Standard-F1"]


def run_tuning(mode: str) -> None:
    tune_dir = os.path.join("results", f"tuning_{mode}")
    os.makedirs(tune_dir, exist_ok=True)
    raw_path = os.path.join(tune_dir, "tuning_raw_details.csv")
    if os.path.exists(raw_path):
        os.remove(raw_path)

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name=mode,
            overrides=[
                f"output_dir={tune_dir}",
                "tuning=true",
                "run_failure_experiments=false",
                "track_energy=false",
            ],
        )

    grid = list(itertools.product(NORMALIZATIONS, TEMPERATURES, WEIGHT_COMBINATIONS))
    rows = []
    for i, (norm, temp, (w_sens, w_comp, w_disc)) in enumerate(grid, 1):
        print(f"\n[{i}/{len(grid)}] norm={norm} T={temp} sens={w_sens} comp={w_comp} disc={w_disc}")
        cfg.normalization = norm
        cfg.calibration.temperature = temp
        cfg.calibration.sensitivity_weight = w_sens
        cfg.calibration.competence_weight = w_comp
        cfg.calibration.discriminability_weight = w_disc

        iteration = [
            {
                "Dataset": res["dataset"],
                "Strategy": res["strategy"],
                **dict(zip(PARAMS, [norm, temp, w_sens, w_comp, w_disc], strict=True)),
                **{m: res[m] for m in METRICS},
            }
            for res in run_pipeline(cfg)
            if res["strategy"].startswith(("WEIGHTED", "DYNAMIC"))
        ]
        pd.DataFrame(iteration).to_csv(
            raw_path, mode="a", header=not os.path.exists(raw_path), index=False
        )
        rows.extend(iteration)

    summary = (
        pd.DataFrame(rows)
        .groupby(PARAMS)[METRICS]
        .mean()
        .reset_index()
        .sort_values(by=["AUC-PR", "AUC-ROC", "VUS-PR"], ascending=False)
    )
    summary_path = os.path.join(tune_dir, "tuning_best_params.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nBest hyper-parameters ({mode}):")
    print(summary.head(10).to_string(index=False))
    print(f"\nRaw: {raw_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid-search ensemble hyper-parameters on validation data."
    )
    parser.add_argument("--mode", required=True, choices=["online", "offline"])
    run_tuning(parser.parse_args().mode)
