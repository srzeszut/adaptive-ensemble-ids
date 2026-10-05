import os
import pickle
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.data.synthetic import inject_synthetic_anomalies
from src.ensemble.weighted import extract_meta_features, reference_stats
from src.utils.scoring import score_model


def _skip(n: int) -> int:
    return min(250, n // 5)


def _collect_synthetic_scores(
    trained_models: dict[str, Any],
    data_val: np.ndarray,
    names: list[str],
    cache_path: str,
    n_seeds: int,
    contamination_levels: tuple[float, ...],
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if all(n in cached["score_matrix"] for n in names):
            return cached["score_matrix"], cached["labels"]

    all_scores = {name: [] for name in names}
    all_labels = []
    for seed_i in range(n_seeds):
        for contamination in contamination_levels:
            aug_data, aug_labels = inject_synthetic_anomalies(
                data_val, contamination=contamination, seed=seed_i * 17 + seed
            )
            for name in names:
                if hasattr(trained_models[name], "reset_buffers"):
                    trained_models[name].reset_buffers()
                sc = score_model(trained_models[name], aug_data, batch_size=512)
                all_scores[name].append(sc[: len(aug_labels)])
            all_labels.append(aug_labels)

    score_matrix = {name: np.concatenate(all_scores[name]) for name in names}
    labels = np.concatenate(all_labels)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"score_matrix": score_matrix, "labels": labels}, f)
    return score_matrix, labels


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return 0.5


def _competence(tr: np.ndarray, va: np.ndarray) -> float:
    skip = _skip(len(tr))
    tr, va = tr[skip:], va[skip:]
    if len(tr) < 10 or len(va) < 10:
        return 0.5

    drift_penalty = abs(np.mean(va) - np.mean(tr)) / (np.std(tr) + 1e-8)
    false_alarm_rate = np.mean(va > np.percentile(tr, 99.0))
    shock_penalty = max(0.0, false_alarm_rate - 0.01) * 10.0
    variance_penalty = max(0.0, np.std(va) / (np.mean(va) + 1e-8) - 1.0)

    score = 1.0 / (1.0 + drift_penalty + shock_penalty + variance_penalty)
    return float(np.clip(score, 0.01, 1.0))


def _discriminability(scores: np.ndarray, labels: np.ndarray) -> float:
    skip = _skip(len(labels))
    sc, labels = scores[skip:], labels[skip:]
    anom, norm = labels == 1, labels == 0
    if anom.sum() < 2 or norm.sum() < 2:
        return 0.5
    snr = max(0.0, (sc[anom].mean() - sc[norm].mean()) / (sc[norm].std() + 1e-8))
    return snr / (1.0 + snr)


def train_meta_model(
    train_scores: dict[str, np.ndarray],
    val_scores: dict[str, np.ndarray],
    synthetic_scores: dict[str, np.ndarray],
    model_names: list[str],
    batch_size: int,
    seed: int,
) -> HistGradientBoostingClassifier:
    rng = np.random.RandomState(seed)
    X_meta, y_meta = [], []

    def add(batch: np.ndarray, stats: dict, healthy: int) -> None:
        X_meta.append(extract_meta_features(batch, stats))
        y_meta.append(healthy)

    def batches(scores: np.ndarray):
        for i in range(0, len(scores), batch_size):
            batch = scores[i : i + batch_size]
            if len(batch) >= 10:
                yield batch

    for name in model_names:
        stats = reference_stats(np.concatenate([train_scores[name], val_scores[name]]))

        for batch in batches(val_scores[name]):
            add(batch, stats, 1)
        for batch in batches(synthetic_scores[name]):
            add(batch, stats, 1)

        for batch in batches(val_scores[name]):
            for multiplier in [2.0, 5.0, 10.0]:
                add(batch * rng.uniform(multiplier, multiplier * 1.5), stats, 0)
            for noise_scale in [2, 5, 10, 20]:
                noise = rng.normal(
                    stats["p95"] * (noise_scale / 2), stats["iqr"] * noise_scale, len(batch)
                )
                add(np.abs(batch + noise), stats, 0)
            add(rng.uniform(0.0, 1e-4, len(batch)), stats, 0)

    meta_model = HistGradientBoostingClassifier(
        max_iter=100, max_depth=7, learning_rate=0.05, random_state=seed
    )
    meta_model.fit(np.array(X_meta), np.array(y_meta))
    return meta_model


def calibrate_weights(
    trained_models: dict[str, Any],
    train_scores: dict[str, np.ndarray],
    val_scores: dict[str, np.ndarray],
    data_val: np.ndarray,
    model_names: list[str],
    config: dict,
    cache_dir: str,
    dataset_stem: str,
    tracker_cls,
    train_meta: bool,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, float], Any, float]:
    names = [n for n in model_names if n in trained_models]
    n_seeds = int(config.n_seeds)
    contamination_levels = tuple(config.contamination_levels)

    levels = "-".join(str(c) for c in contamination_levels)
    cache_path = os.path.join(cache_dir, f"{dataset_stem}_seed{seed}_s{n_seeds}_c{levels}.pkl")
    score_matrix, labels = _collect_synthetic_scores(
        trained_models,
        data_val,
        names,
        cache_path,
        n_seeds,
        contamination_levels,
        seed,
    )

    skip = _skip(len(labels))
    combined = []
    for name in names:
        sensitivity = _roc_auc(score_matrix[name][skip:], labels[skip:])
        competence = _competence(
            np.asarray(train_scores[name]).ravel(), np.asarray(val_scores[name]).ravel()
        )
        discriminability = _discriminability(np.asarray(score_matrix[name]).ravel(), labels)
        gate = max(0.0, (sensitivity - 0.5) * 2.0)
        c = gate * (
            config.sensitivity_weight
            + config.competence_weight * competence
            + config.discriminability_weight * discriminability
        )
        combined.append(float(np.clip(c, 0.001, 3.0)))

    logits = np.array(combined) / config.temperature
    w = np.exp(logits - np.max(logits))
    w = w / w.sum()
    w = np.maximum(w, 1.0 / (len(names) * 5))
    w = w / w.sum()
    weights = {n: float(w[i]) for i, n in enumerate(names)}
    print(f"   [CALIBRATION] Prior weights: {weights}")

    if not train_meta:
        return weights, None, 0.0

    with tracker_cls() as meta_tracker:
        meta_model = train_meta_model(
            train_scores, val_scores, score_matrix, names, batch_size, seed
        )
    return weights, meta_model, meta_tracker.emissions_kwh
