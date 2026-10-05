import numpy as np
from deep_river.anomaly import Autoencoder as DeepRiverAE
from river import cluster as river_cluster
from torch import nn


def _as_dict(row: np.ndarray) -> dict:
    return {i: float(v) for i, v in enumerate(row)}


class RiverKMeansWrapper:
    def __init__(self, n_clusters: int, seed: int):
        self.kmeans = river_cluster.KMeans(n_clusters=n_clusters, halflife=0.4, sigma=3, seed=seed)

    def partial_fit(self, X: np.ndarray) -> None:
        for row in np.asarray(X):
            self.kmeans.learn_one(_as_dict(row))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.kmeans.centers:
            return np.zeros(len(X))
        scores = []
        for row in np.asarray(X):
            features = _as_dict(row)
            scores.append(
                min(
                    sum((features[k] - center[k]) ** 2 for k in features)
                    for center in self.kmeans.centers.values()
                )
            )
        return np.array(scores)


class RiverZScoreWrapper:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._buf = None

    def partial_fit(self, X: np.ndarray) -> None:
        X = np.asarray(X)
        self._buf = X if self._buf is None else np.vstack([self._buf, X])
        self._buf = self._buf[-self.window_size :]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if self._buf is None or len(self._buf) < 2:
            return np.zeros(len(X))
        means = np.mean(self._buf, axis=0)
        stds = np.maximum(np.std(self._buf, axis=0), 1e-3)
        return np.max(np.abs(X - means) / stds, axis=1)


class _Conv1DAutoencoder(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Unflatten(1, (1, n_features)),
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 8, 5, padding=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(8, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 1, 5, padding=2),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class StreamCNNWrapper:
    def __init__(self, window_size: int, feats: int, seed: int, lr: float = 0.001):
        self.window_size = window_size
        self.feats = feats
        self.model = DeepRiverAE(
            module=_Conv1DAutoencoder,
            lr=lr,
            loss_fn="mse",
            optimizer_fn="adam",
            seed=seed,
            n_features=window_size * feats,
        )
        self.reset_buffers()

    def reset_buffers(self) -> None:
        self._train_buf = np.empty((0, self.feats))
        self._eval_buf = np.empty((0, self.feats))

    def _windows(self, buf: np.ndarray):
        for i in range(max(0, len(buf) - self.window_size + 1)):
            yield _as_dict(buf[i : i + self.window_size].ravel())

    def _tail(self, buf: np.ndarray) -> np.ndarray:
        return buf[-(self.window_size - 1) :] if len(buf) >= self.window_size else buf

    def partial_fit(self, X: np.ndarray) -> None:
        buf = np.vstack([self._train_buf, X])
        for window in self._windows(buf):
            self.model.learn_one(window)
        self._train_buf = self._tail(buf)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        buf = np.vstack([self._eval_buf, X])
        raw = [self.model.score_one(window) for window in self._windows(buf)]
        self._eval_buf = self._tail(buf)
        scores = np.zeros(len(X))
        if raw:
            scores[-len(raw) :] = raw
        return scores


def build_online_model(name: str, feats: int, seed: int):
    if name == "StreamCNN":
        return StreamCNNWrapper(window_size=100, feats=feats, seed=seed)
    if name == "RiverKMeans":
        return RiverKMeansWrapper(n_clusters=min(max(5, feats // 4), 20), seed=seed)
    if name == "RiverZScore":
        return RiverZScoreWrapper(window_size=100)
    raise ValueError(f"Unknown online model: {name}")
