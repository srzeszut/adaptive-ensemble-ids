from typing import Any

import numpy as np

from src.utils.normalization import normalize_all

SATURATION_RATIO = 3.0
SATURATION_CV = 0.10
META_MULTIPLIER_RANGE = (0.01, 1.3)
WEIGHT_EMA_ALPHA = 0.15
THRESHOLD_PERCENTILE = 95.0


def reference_stats(scores: np.ndarray) -> dict:
    p25, p50, p75, p95 = np.percentile(scores, [25, 50, 75, 95])
    return {"median": p50 + 1e-8, "iqr": p75 - p25 + 1e-8, "p95": p95 + 1e-8}


def extract_meta_features(batch_raw: np.ndarray, stats: dict) -> np.ndarray:
    b_p25, b_p50, b_p75, b_p95 = np.percentile(batch_raw, [25, 50, 75, 95])
    b_iqr = max(b_p75 - b_p25, 1e-8)

    feats = np.array(
        [
            np.log1p(b_iqr / (stats["iqr"] + 1e-8)),
            np.log1p(b_p95 / (stats["p95"] + 1e-8)),
            np.log1p(b_p50 / (stats["median"] + 1e-8)),
            b_iqr / (np.abs(b_p50) + 0.1),
            float(np.mean(batch_raw > stats["p95"])),
        ]
    )
    return np.clip(feats, -10.0, 10.0)


def _heuristic_multiplier(batch_raw: np.ndarray, stats: dict) -> float:
    b_p25, b_p50, b_p75, b_p95 = np.percentile(batch_raw, [25, 50, 75, 95])
    b_iqr = b_p75 - b_p25 + 1e-8
    b_robust_cv = b_iqr / (np.abs(b_p50) + 0.1)

    saturated = b_p50 > stats["p95"] * SATURATION_RATIO and b_robust_cv < SATURATION_CV
    flatlined = b_p95 < 1e-3 and stats["p95"] > 1e-2

    if saturated or flatlined:
        return 0.2
    if b_robust_cv > 1.2 and b_iqr > stats["iqr"] * 4.0:
        return max(0.1, 1.0 / b_robust_cv)
    if b_p50 > stats["p95"] and b_robust_cv < 0.8:
        return 1.2
    if b_p50 <= stats["p95"] and b_p95 > stats["p95"] * 2.0:
        return min(1.3, 1.0 + (b_p95 / (b_p50 + 0.1)) * 0.05)
    return 1.0


def _aggregate(norm_scores: np.ndarray, weights: np.ndarray, aggregation: str) -> np.ndarray:
    weighted = norm_scores * weights[:, None]
    return np.max(weighted, axis=0) if aggregation == "max" else np.sum(weighted, axis=0)


def run_cv_dynamic_ensemble(
    train_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    model_names: list[str],
    base_weights: dict[str, float],
    batch_size: int,
    meta_model: Any,
    normalize: str,
    aggregation: str = "mean",
) -> tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray]:
    valid = [n for n in model_names if n in base_weights]
    min_len_tr = min(len(train_scores[n]) for n in valid)
    n_samples = min(len(test_scores[n]) for n in valid)

    tr_norm, te_norm = normalize_all(train_scores, test_scores, valid, normalize, batch_size)

    hist_stats = {name: reference_stats(train_scores[name][:min_len_tr]) for name in valid}

    final_scores = np.zeros(n_samples)
    weights_history = np.zeros((n_samples, len(valid)))

    base_w = np.array([base_weights[n] for n in valid], dtype=np.float64)
    base_w = base_w / np.sum(base_w)
    current_w = base_w.copy()

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        calculated_w = np.zeros(len(valid))

        for i, name in enumerate(valid):
            batch_raw = test_scores[name][start:end]
            if meta_model is not None:
                features = extract_meta_features(batch_raw, hist_stats[name])
                p_healthy = meta_model.predict_proba(features.reshape(1, -1))[0, 1]
                multiplier = float(np.clip(p_healthy, *META_MULTIPLIER_RANGE))
            else:
                multiplier = _heuristic_multiplier(batch_raw, hist_stats[name])
            calculated_w[i] = base_w[i] * multiplier

        sum_w = np.sum(calculated_w)
        norm_w = base_w.copy() if sum_w < 1e-5 else calculated_w / sum_w

        current_w = (1.0 - WEIGHT_EMA_ALPHA) * current_w + WEIGHT_EMA_ALPHA * norm_w
        current_w = current_w / np.sum(current_w)

        final_scores[start:end] = _aggregate(te_norm[:, start:end], current_w, aggregation)
        weights_history[start:end] = current_w

    final_tr = _aggregate(tr_norm, base_w, aggregation)
    pred = (final_scores >= np.percentile(final_tr, THRESHOLD_PERCENTILE)).astype(int)

    mean_weights = {name: float(np.mean(weights_history[:, j])) for j, name in enumerate(valid)}
    return final_scores, pred, mean_weights, weights_history


def apply_weighted_ensemble(
    train_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    model_names: list[str],
    weights: dict[str, float],
    normalize: str,
    batch_size: int,
    aggregation: str = "mean",
) -> tuple[np.ndarray, np.ndarray, dict[str, float], None]:
    valid = [n for n in model_names if n in weights]
    tr_norm, te_norm = normalize_all(train_scores, test_scores, valid, normalize, batch_size)

    w = np.array([weights[n] for n in valid], dtype=np.float64)
    w = w / w.sum()

    final_tr = _aggregate(tr_norm, w, aggregation)
    final_te = _aggregate(te_norm, w, aggregation)
    pred = (final_te >= np.percentile(final_tr, THRESHOLD_PERCENTILE)).astype(int)
    return final_te, pred, dict(zip(valid, w.tolist(), strict=True)), None
