from typing import Any

import numpy as np


def safe_window_and_batch(
    n_samples: int,
    desired_window: int = 100,
    desired_batch: int = 128,
    model_val_ratio: float = 0.2,
    pred_len: int = 1,
    min_windows: int = 4,
) -> tuple[int, int]:
    val_size = int(n_samples * model_val_ratio)
    train_size = n_samples - val_size

    max_win = min(val_size, train_size) - pred_len - min_windows + 1
    win = max(min(desired_window, max(max_win, 3)), 3)

    n_windows = max(min(val_size, train_size) - win - pred_len + 1, 1)
    bs = max(min(desired_batch, max(n_windows, 2)), 2)
    return int(win), int(bs)


def score_model(model: Any, data: np.ndarray, batch_size: int) -> np.ndarray:
    win = getattr(model, "window_size", 0)
    step = max(batch_size, win + 1) if win > 0 else batch_size
    out = []
    for i in range(0, len(data), step):
        batch = data[i : i + step]
        try:
            sc = np.asarray(model.decision_function(batch)).ravel()
            out.extend(np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0))
        except Exception as e:
            print(f"      [SCORE FAIL] {type(model).__name__} batch {i}: {e}")
            out.extend(np.zeros(len(batch)))
    return np.array(out[: len(data)])
