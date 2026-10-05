import argparse
import glob
import os

from src.utils.plotting import load_weights_snapshot, plot_noise_experiment_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate weight plots from *_weights.csv snapshots."
    )
    parser.add_argument(
        "path", help="A *_weights.csv file or a directory such as <output_dir>/dynamic_weights."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Plots go to <output-dir>/plots/weights/."
    )
    args = parser.parse_args()

    if os.path.isdir(args.path):
        csv_files = sorted(glob.glob(os.path.join(args.path, "*_weights.csv")))
    else:
        csv_files = [args.path]
    if not csv_files:
        raise SystemExit(f"No *_weights.csv snapshots in {args.path}")

    for csv_path in csv_files:
        snap = load_weights_snapshot(csv_path)
        plot_noise_experiment_weights(
            snap["history"],
            snap["model_names"],
            snap["broken_list"],
            snap["dataset_stem"],
            snap["title"],
            snap["windows"],
            args.output_dir,
        )
        print(f"[OK] {os.path.basename(csv_path)}")


if __name__ == "__main__":
    main()
