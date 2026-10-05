import numpy as np
from TSB_AD.evaluation.metrics import get_metrics as tsbad_get_metrics


def get_metrics(
    score: np.ndarray, labels: np.ndarray, pred: np.ndarray, sliding_window: int
) -> dict[str, float]:
    return tsbad_get_metrics(score, labels, pred=pred, slidingWindow=sliding_window)
