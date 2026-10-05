import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.labels import abbreviate

BLUES = ["#08306B", "#2171B5", "#4292C6", "#6BAED6", "#9ECAE1", "#C6DBEF"]
NOISE_WINDOW_COLOR = "#9ECAE1"


def plot_noise_experiment_weights(
    weights_history, model_names, broken_list, dataset_name, title, windows, output_dir
):
    if weights_history is None or len(weights_history) == 0:
        return

    plt.figure(figsize=(14, 5))
    for i, name in enumerate(model_names):
        is_broken = name in broken_list
        plt.plot(
            weights_history[:, i],
            label=f"{abbreviate(name)} (broken)" if is_broken else abbreviate(name),
            linewidth=1.8,
            linestyle="--" if is_broken else "-",
            color=BLUES[i % len(BLUES)],
        )

    for idx, (start, end) in enumerate(windows):
        plt.axvspan(
            start,
            end,
            facecolor=NOISE_WINDOW_COLOR,
            alpha=0.45,
            hatch="//",
            edgecolor=BLUES[0],
            label="Injected failure window" if idx == 0 else None,
        )

    plt.ylim(0, 1)
    plt.xlabel("Sample index", fontsize=12)
    plt.ylabel("Model weight", fontsize=12)
    plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    plot_dir = os.path.join(output_dir, "plots", "weights")
    os.makedirs(plot_dir, exist_ok=True)
    safe_name = re.sub(r"[^\w\-]+", "_", f"{dataset_name}_{title}")
    plt.savefig(os.path.join(plot_dir, f"{safe_name}_weights.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_weights_snapshot(
    weights_dir,
    safe_case,
    strat_name,
    history,
    model_names,
    y_true,
    broken_list,
    windows,
    dataset_stem,
    title,
):
    os.makedirs(weights_dir, exist_ok=True)
    n = min(len(history), len(y_true))
    df = pd.DataFrame(history[:n], columns=model_names)
    df["true_label"] = np.asarray(y_true)[:n]

    base = os.path.join(weights_dir, f"{safe_case}_{strat_name}_weights")
    df.to_csv(f"{base}.csv", index=False)
    with open(f"{base}.meta.json", "w") as f:
        json.dump(
            {
                "dataset_stem": dataset_stem,
                "strategy": strat_name,
                "title": title,
                "model_names": list(model_names),
                "broken_models": list(broken_list),
                "windows": [[int(x) for x in w] for w in windows],
            },
            f,
            indent=2,
        )


def load_weights_snapshot(csv_path):
    meta_path = csv_path.removesuffix(".csv") + ".meta.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata sidecar for {csv_path}: {meta_path}")

    df = pd.read_csv(csv_path)
    with open(meta_path) as f:
        meta = json.load(f)

    return {
        "history": df[meta["model_names"]].to_numpy(),
        "model_names": meta["model_names"],
        "broken_list": meta["broken_models"],
        "windows": [tuple(w) for w in meta["windows"]],
        "dataset_stem": meta["dataset_stem"],
        "title": meta["title"],
    }
