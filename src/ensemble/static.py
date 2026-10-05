import numpy as np

from src.utils.normalization import normalize_all


def run_static_ensemble(
    train_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    model_names: list[str],
    strategy: str,
    normalize: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], None]:
    valid = [n for n in model_names if n in train_scores and n in test_scores]
    tr_norm, te_norm = normalize_all(train_scores, test_scores, valid, normalize, batch_size)

    reduce = {"max": np.max, "median": np.median, "mean": np.mean}[strategy]
    final_tr = reduce(tr_norm, axis=0)
    final_te = reduce(te_norm, axis=0)

    pred = (final_te >= np.percentile(final_tr, 95.0)).astype(int)
    weights = {n: 1.0 / len(valid) for n in valid}
    return final_te, pred, weights, None
