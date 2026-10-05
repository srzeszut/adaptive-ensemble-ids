import copy
import os
import random
import subprocess
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_dataset, train_index_from_filename
from src.data.synthetic import inject_synthetic_anomalies
from src.ensemble.calibration import calibrate_weights
from src.ensemble.static import run_static_ensemble
from src.ensemble.weighted import apply_weighted_ensemble, run_cv_dynamic_ensemble
from src.models.offline import build_offline_model
from src.models.online import build_online_model
from src.pipeline.tracker import DummyTracker, EnergyTracker
from src.utils.metrics import get_metrics
from src.utils.noise import FAILURE_MODES, corrupt_test_scores
from src.utils.plotting import plot_noise_experiment_weights, save_weights_snapshot
from src.utils.scoring import safe_window_and_batch, score_model


def prequential_pass(
    model,
    data_stream,
    batch_size,
    scaler,
    poison_threshold=None,
    labels_stream=None,
    threshold_alpha: float = 0.05,
):
    scores = []
    stats = {
        "skipped_total": 0,
        "true_anom_batches": 0,
        "blocked_true_anom": 0,
        "blocked_false_anom": 0,
        "leaked_anom_batches": 0,
    }

    for start in range(0, len(data_stream), batch_size):
        batch_raw = data_stream[start : start + batch_size]
        has_anomaly = labels_stream is not None and np.any(
            labels_stream[start : start + batch_size] == 1
        )
        stats["true_anom_batches"] += int(has_anomaly)

        batch_scaled = scaler.transform(batch_raw)
        batch_scores = model.decision_function(batch_scaled)
        scores.extend(batch_scores)

        poisoned = False
        if poison_threshold is not None:
            batch_mean = np.mean(batch_scores)
            batch_cv = np.std(batch_scores) / (np.abs(batch_mean) + 1e-8)
            batch_p95 = np.percentile(batch_scores, 95.0)
            poisoned = batch_p95 > poison_threshold or (
                batch_cv < 0.05 and batch_mean > poison_threshold
            )

        if poisoned:
            stats["skipped_total"] += 1
            stats["blocked_true_anom" if has_anomaly else "blocked_false_anom"] += 1
            continue

        model.partial_fit(batch_scaled)
        scaler.partial_fit(batch_raw)
        stats["leaked_anom_batches"] += int(has_anomaly)
        if poison_threshold is not None:
            poison_threshold = (
                1 - threshold_alpha
            ) * poison_threshold + threshold_alpha * batch_p95

    return np.array(scores), stats


def calculate_recovery_batches(
    w_history: np.ndarray, windows: list, broken_idx: int, batch_size: int
) -> tuple[float, float, int]:
    recovered, n_valid = [], 0
    for noise_start, noise_end in windows:
        safe_start = max(0, noise_start - 10 * batch_size)
        if safe_start == noise_start:
            continue

        target_w = np.mean(w_history[safe_start:noise_start, broken_idx]) * 0.90
        if np.min(w_history[noise_start:noise_end, broken_idx]) >= target_w:
            continue

        n_valid += 1
        hits = np.flatnonzero(w_history[noise_end:, broken_idx] >= target_w)
        if hits.size:
            recovered.append(float(hits[0]) / batch_size)

    if n_valid == 0:
        return np.nan, np.nan, 0
    mean_rec = float(np.mean(recovered)) if recovered else np.nan
    return mean_rec, len(recovered) / n_valid, n_valid


def vus_sliding_window(labels: np.ndarray) -> int:
    if not np.any(labels == 1):
        return 100
    diffs = np.diff(np.concatenate([[0], labels, [0]]))
    lengths = np.flatnonzero(diffs == -1) - np.flatnonzero(diffs == 1)
    return max(10, min(int(np.median(lengths)), 200))


def write_run_manifest(cfg: DictConfig) -> None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = "unknown", False

    manifest = OmegaConf.to_container(cfg, resolve=True)
    manifest["_git_commit"] = sha
    manifest["_git_dirty"] = dirty
    manifest["_started_at"] = datetime.now().isoformat(timespec="seconds")

    path = os.path.join(cfg.output_dir, "run_manifest.yaml")
    with open(path, "w") as fh:
        fh.write(OmegaConf.to_yaml(OmegaConf.create(manifest)))
    print(f"[MANIFEST] {path} (commit {sha[:8]}{', DIRTY' if dirty else ''})")


def run_pipeline(cfg: DictConfig) -> list:
    os.makedirs(cfg.output_dir, exist_ok=True)
    write_run_manifest(cfg)
    all_results = []
    online = cfg.mode == "online"
    Tracker = EnergyTracker if cfg.track_energy else DummyTracker

    for file in sorted(os.listdir(cfg.dataset_dir)):
        if not file.endswith(".csv"):
            continue
        try:
            train_index = train_index_from_filename(file)
        except (ValueError, IndexError):
            print(f"[{file}] skipped: no '_tr_<index>_' in filename")
            continue

        print(f"\n[{file}] Starting dataset processing...")
        t_start_dataset = time.time()

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        data, label = load_dataset(os.path.join(cfg.dataset_dir, file))
        dataset_stem = file.removesuffix(".csv")
        data_train_raw, data_test_raw = data[:train_index], data[train_index:]
        label_test = label[train_index:]
        feats = data_train_raw.shape[1]

        split = int(len(data_train_raw) * (1 - cfg.val_ratio))
        win_safe, bs_safe = safe_window_and_batch(split)
        bs = cfg.get("batch_size", bs_safe)
        win = cfg.get("window_size", win_safe)

        data_core_raw = data_train_raw[:split]
        data_val_raw = data_train_raw[split:]

        if cfg.tuning:
            calib_end = int(len(data_val_raw) * 0.7)
            data_tune_raw = data_val_raw[calib_end:]
            data_val_raw = data_val_raw[:calib_end]
            if len(data_tune_raw) < 10 or len(data_val_raw) < 10:
                print(f"   [SKIP] {file}: validation block too small for tuning split.")
                continue
            data_test_raw, label_test = inject_synthetic_anomalies(
                data_tune_raw, contamination=cfg.tuning_contamination, seed=cfg.tuning_seed
            )
            print(
                f"   [TUNING] Selection stream: {len(data_test_raw)} samples, "
                f"{label_test.mean():.3f} anomaly ratio (test set untouched)."
            )

        sliding_window = vus_sliding_window(label_test)
        n_test = len(label_test)
        anomaly_ratio = float(np.mean(label_test)) if n_test else float("nan")
        dataset_stats = {
            "n_train": int(train_index),
            "n_test": n_test,
            "n_features": int(feats),
            "anomaly_ratio": anomaly_ratio,
            "random_auc_pr": anomaly_ratio,
            "all_positive_f1": 2 * anomaly_ratio / (1 + anomaly_ratio)
            if anomaly_ratio > 0
            else 0.0,
        }
        print(
            f"   [DATA] n_train={train_index} n_test={n_test} feats={feats} "
            f"anomaly_ratio={anomaly_ratio:.4f} vus_window={sliding_window}"
        )

        base_scaler = StandardScaler().fit(data_core_raw)
        data_core_scaled = base_scaler.transform(data_core_raw)
        data_val_scaled = base_scaler.transform(data_val_raw)
        data_test_scaled = base_scaler.transform(data_test_raw)

        trained_models, model_scalers, train_energy = {}, {}, {}
        train_scores, val_scores, test_scores = {}, {}, {}

        print(f"   --- PHASE 1: TRAINING ({cfg.mode.upper()}) ---")
        for name in cfg.models:
            print(f"   [TRAIN] {name}")
            with Tracker(f"train_{name}") as tracker:
                if online:
                    model = build_online_model(name, feats, cfg.seed)
                    model_scalers[name] = copy.deepcopy(base_scaler)
                    extra_passes = 3 if name == "StreamCNN" else 0
                    for _ in range(extra_passes):
                        prequential_pass(model, data_core_raw, bs, model_scalers[name])
                    tr_sc, _ = prequential_pass(model, data_core_raw, bs, model_scalers[name])
                else:
                    model = build_offline_model(name, feats, win, bs, cfg.seed)
                    model.fit(data_core_scaled)
                    tr_sc = score_model(model, data_core_scaled, batch_size=bs)

            trained_models[name] = model
            train_scores[name] = tr_sc
            train_energy[name] = tracker.emissions_kwh

        print("\n   --- PHASE 2: CALIBRATION ---")
        t_phase2 = time.time()
        valid_names = list(trained_models)

        with Tracker("val_scoring") as val_tracker:
            for name in valid_names:
                val_input = (
                    model_scalers[name].transform(data_val_raw) if online else data_val_scaled
                )
                val_scores[name] = score_model(trained_models[name], val_input, batch_size=bs)

        with Tracker("calibration") as calib_tracker:
            weights, meta_model, meta_model_energy = calibrate_weights(
                trained_models,
                train_scores,
                val_scores,
                data_val_scaled,
                valid_names,
                cfg.calibration,
                cache_dir=os.path.join(cfg.output_dir, "saved_synthetic_scores"),
                dataset_stem=dataset_stem,
                tracker_cls=Tracker,
                train_meta=online,
                batch_size=bs,
                seed=cfg.seed,
            )
        calib_energy = val_tracker.emissions_kwh + calib_tracker.emissions_kwh
        print(f"   [CALIBRATION] Phase completed in {time.time() - t_phase2:.2f}s")

        print(f"\n   --- PHASE 3: TESTING ({cfg.mode.upper()}) ---")
        blocked_stats = {}
        infer_energy = {}
        for name, model in trained_models.items():
            with Tracker(f"infer_{name}") as infer_tracker:
                if online:
                    prequential_pass(model, data_val_raw, bs, model_scalers[name])
                    ref_scores = np.asarray(train_scores[name])[cfg.warmup_skip :]
                    if len(ref_scores) < 10:
                        ref_scores = np.asarray(train_scores[name])
                    poison_threshold = (
                        np.percentile(ref_scores, cfg.poison_percentile) * cfg.poison_margin
                    )
                    te_sc, blocked_stats[name] = prequential_pass(
                        model,
                        data_test_raw,
                        bs,
                        model_scalers[name],
                        poison_threshold=poison_threshold,
                        labels_stream=label_test,
                        threshold_alpha=cfg.poison_threshold_alpha,
                    )
                else:
                    te_sc = score_model(model, data_test_scaled, batch_size=bs)

            infer_energy[name] = infer_tracker.emissions_kwh
            test_scores[name] = te_sc

        if blocked_stats:
            pd.DataFrame(
                [
                    {
                        "Dataset": dataset_stem,
                        "Model": name,
                        "Total_Skipped_Batches": s["skipped_total"],
                        "Total_True_Anomaly_Batches": s["true_anom_batches"],
                        "Blocked_True_Anomalies (TP)": s["blocked_true_anom"],
                        "Blocked_Normal_Data (FP)": s["blocked_false_anom"],
                        "Leaked_Anomalies (FN)": s["leaked_anom_batches"],
                    }
                    for name, s in blocked_stats.items()
                ]
            ).to_csv(
                os.path.join(cfg.output_dir, f"{dataset_stem}_blocked_batches_summary.csv"),
                index=False,
            )

        print("\n   --- PHASE 4: ENSEMBLE STRATEGIES ---")
        historical_scores = {
            name: np.concatenate([train_scores[name], val_scores[name]]) for name in valid_names
        }
        equal_weights = {n: 1.0 / len(valid_names) for n in valid_names}
        norm = cfg.normalization

        def evaluate_all_strategies(scores_in, case_name, broken_list, windows, track_energy=False):
            strategies = {
                "STATIC_MAX": lambda: run_static_ensemble(
                    historical_scores, scores_in, valid_names, "max", norm, bs
                ),
                "STATIC_MEDIAN": lambda: run_static_ensemble(
                    historical_scores, scores_in, valid_names, "median", norm, bs
                ),
                "STATIC_MEAN": lambda: run_static_ensemble(
                    historical_scores, scores_in, valid_names, "mean", norm, bs
                ),
                "WEIGHTED_STATIC_MEAN": lambda: apply_weighted_ensemble(
                    historical_scores, scores_in, valid_names, weights, norm, bs, "mean"
                ),
                "WEIGHTED_STATIC_MAX": lambda: apply_weighted_ensemble(
                    historical_scores, scores_in, valid_names, weights, norm, bs, "max"
                ),
            }
            for name in valid_names:
                strategies[f"BASELINE_{name}"] = lambda n=name: run_static_ensemble(
                    historical_scores, scores_in, [n], "mean", norm, bs
                )

            if online:

                def dynamic(start_w, meta, agg):
                    return lambda: run_cv_dynamic_ensemble(
                        historical_scores, scores_in, valid_names, start_w, bs, meta, norm, agg
                    )

                strategies.update(
                    {
                        "DYNAMIC_HEUR_EQUAL_START": dynamic(equal_weights, None, "mean"),
                        "DYNAMIC_HEUR_CALIBRATED_START": dynamic(weights, None, "mean"),
                        "DYNAMIC_META_EQUAL_START": dynamic(equal_weights, meta_model, "mean"),
                        "DYNAMIC_META_CALIBRATED_START": dynamic(weights, meta_model, "mean"),
                        "DYNAMIC_HEUR_MAX_EQUAL": dynamic(equal_weights, None, "max"),
                        "DYNAMIC_HEUR_MAX_CAL": dynamic(weights, None, "max"),
                        "DYNAMIC_META_MAX_EQUAL": dynamic(equal_weights, meta_model, "max"),
                        "DYNAMIC_META_MAX_CAL": dynamic(weights, meta_model, "max"),
                    }
                )

            EvalTracker = Tracker if track_energy else DummyTracker
            for strat_name, fn in strategies.items():
                t_strat = time.time()
                with EvalTracker(f"eval_{strat_name}") as tracker:
                    scores, pred, _, history = fn()
                elapsed = time.time() - t_strat
                metrics = get_metrics(scores, label_test, pred=pred, sliding_window=sliding_window)

                recovery_batches, recovery_rate, recovery_windows = np.nan, np.nan, 0
                if history is not None and broken_list and windows:
                    per_model = [
                        calculate_recovery_batches(history, windows, valid_names.index(b), bs)
                        for b in broken_list
                    ]
                    observed = [m for m in per_model if m[2] > 0]
                    if observed:
                        recovery_windows = sum(m[2] for m in observed)
                        recovery_rate = float(np.mean([m[1] for m in observed]))
                        means = [m[0] for m in observed if not np.isnan(m[0])]
                        recovery_batches = float(np.mean(means)) if means else np.nan

                if history is not None:
                    safe_case = (
                        case_name.replace(" ", "_")
                        .replace("'", "")
                        .replace("[", "")
                        .replace("]", "")
                    )
                    title = f"{case_name} ({strat_name})"
                    save_weights_snapshot(
                        os.path.join(cfg.output_dir, "dynamic_weights"),
                        safe_case,
                        strat_name,
                        history,
                        valid_names,
                        label_test,
                        broken_list,
                        windows,
                        dataset_stem,
                        title,
                    )
                    plot_noise_experiment_weights(
                        history,
                        valid_names,
                        broken_list,
                        dataset_stem,
                        title,
                        windows,
                        cfg.output_dir,
                    )

                all_results.append(
                    {
                        "dataset": file,
                        "experiment_case": case_name,
                        "broken_models": str(broken_list),
                        "strategy": strat_name,
                        "infer_time_sec": elapsed,
                        "strategy_eval_energy_kwh": tracker.emissions_kwh,
                        "train_energy_kwh_total": sum(train_energy.values()),
                        "models_infer_energy_kwh_total": sum(infer_energy.values()),
                        "calib_energy_kwh_total": calib_energy,
                        "meta_model_energy_kwh": meta_model_energy,
                        "Recovery_Batches": recovery_batches,
                        "Recovery_Rate": recovery_rate,
                        "Recovery_Windows": recovery_windows,
                        **dataset_stats,
                        **metrics,
                    }
                )

                rec_str = "N/A" if np.isnan(recovery_batches) else f"{recovery_batches:.2f}"
                rate_str = "N/A" if np.isnan(recovery_rate) else f"{recovery_rate:.0%}"
                print(f"      [OK] {strat_name:<30s} (Recovery: {rec_str}, recovered: {rate_str})")

        print("\n   => Clean evaluation")
        evaluate_all_strategies(test_scores, "Clean", [], [], track_energy=True)

        if online and cfg.run_failure_experiments:
            print("\n   => Single-detector failures")
            for broken in valid_names:
                for mode, mode_name in FAILURE_MODES.items():
                    corrupted, windows = corrupt_test_scores(
                        test_scores, [broken], mode, label_test, cfg.seed
                    )
                    evaluate_all_strategies(
                        corrupted, f"Failure - {broken} ({mode_name})", [broken], windows
                    )

            broken_many = [n for n in valid_names if n != "StreamCNN"]
            if broken_many:
                print("\n   => Multi-detector failure (CONFUSION)")
                corrupted, windows = corrupt_test_scores(
                    test_scores, broken_many, 1, label_test, cfg.seed
                )
                evaluate_all_strategies(
                    corrupted, f"Multiple Failure (CONFUSION) - {broken_many}", broken_many, windows
                )

        pd.DataFrame(all_results).to_csv(os.path.join(cfg.output_dir, "results.csv"), index=False)
        print(f"[{file}] Dataset processing finished in {time.time() - t_start_dataset:.2f}s.\n")

    return all_results
