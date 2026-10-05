import numpy as np

MAD_SCALE = 6.0
MAD_INLIER_SIGMA = 3.0
MAD_BATCH_ALPHA = 0.05


def normalize_scores(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    method: str,
    batch_size: int,
    alpha: float = MAD_BATCH_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    tr = np.asarray(train_scores, dtype=np.float64).ravel()
    te = np.asarray(test_scores, dtype=np.float64).ravel()

    if method == "mad":
        med = np.median(tr)
        mad = max(np.median(np.abs(tr - med)), np.abs(med) * 0.01, 1e-3)
        tr_norm = 1.0 - np.exp(-np.maximum(tr - med, 0.0) / (MAD_SCALE * mad))

        te_norm = np.empty_like(te)
        cur_med, cur_mad = med, mad
        step = max(int(batch_size), 1)
        for start in range(0, len(te), step):
            chunk = te[start : start + step]
            dist = np.maximum(chunk - cur_med, 0.0) / cur_mad
            te_norm[start : start + len(chunk)] = 1.0 - np.exp(-dist / MAD_SCALE)

            inliers = chunk[np.abs(chunk - cur_med) <= MAD_INLIER_SIGMA * cur_mad]
            if inliers.size >= 2:
                b_med = np.median(inliers)
                b_mad = np.median(np.abs(inliers - b_med))
                cur_med = (1.0 - alpha) * cur_med + alpha * b_med
                cur_mad = max((1.0 - alpha) * cur_mad + alpha * b_mad, 1e-3)

        return np.clip(tr_norm, 0.0, 1.0), np.clip(te_norm, 0.0, 1.0)

    if method == "robust":
        med = np.median(tr)
        q75, q25 = np.percentile(tr, [75, 25])
        iqr = (q75 - q25) + 1e-8
        return (tr - med) / iqr, (te - med) / iqr

    if method == "percentile_minmax":
        lo, hi = np.percentile(tr, 2), np.percentile(tr, 98)
        denom = (hi - lo) + 1e-8
        return np.clip((tr - lo) / denom, 0.0, 1.0), np.clip((te - lo) / denom, 0.0, 1.0)

    if method == "zscore_sigmoid":
        mu, sigma = np.mean(tr), np.std(tr) + 1e-8
        return 1.0 / (1.0 + np.exp(-(tr - mu) / sigma)), 1.0 / (1.0 + np.exp(-(te - mu) / sigma))

    raise ValueError(f"Unknown normalization method: {method}")


def normalize_all(
    train_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    names: list[str],
    method: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    min_tr = min(len(train_scores[n]) for n in names)
    min_te = min(len(test_scores[n]) for n in names)
    pairs = [
        normalize_scores(train_scores[n][:min_tr], test_scores[n][:min_te], method, batch_size)
        for n in names
    ]
    return (
        np.array([p[0] for p in pairs], dtype=np.float64),
        np.array([p[1] for p in pairs], dtype=np.float64),
    )
