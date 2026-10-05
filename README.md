# Data-Driven Calibration and Online Adaptive Ensemble Weighting for Network Intrusion Detection

## Setup

```bash
make install
```

## Data

Place all datasets in `data/`. `data/InSDN_tr_53383.csv` is included: the
[InSDN](https://doi.org/10.1109/ACCESS.2020.3022633) dataset as a time series.

## Run

```bash
make tune-online
make tune-offline

make run-online
make run-offline
make run-seeds MODE=online SEEDS="42 43 44"
make run-seeds MODE=offline SEEDS="42 43 44"

make verify DIR=results/online
make analyze-online
make analyze-offline
make replot-weights DIR=results/online
make stats MODE=online
```

Override any config value:

```bash
.venv/bin/python scripts/run_experiment.py --config-name online \
    output_dir=results/exp calibration.temperature=0.25
```

## License

Code: [MIT](LICENSE).

`data/InSDN_tr_53383.csv` is derived from InSDN and licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Changes: numeric flow
features and binary label only, as a time series.

M. S. Elsayed, N.-A. Le-Khac, A. D. Jurcut, "InSDN: A Novel SDN Intrusion Dataset",
IEEE Access, vol. 8, pp. 165263–165284, 2020, doi:
[10.1109/ACCESS.2020.3022633](https://doi.org/10.1109/ACCESS.2020.3022633).
