import numpy as np


def inject_synthetic_anomalies(
    data_normal: np.ndarray, contamination: float = 0.10, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    data = data_normal.copy()
    n, d = data.shape
    labels = np.zeros(n, dtype=int)

    target = int(n * contamination)
    injected = 0

    col_med = np.median(data, axis=0)
    col_mad = np.median(np.abs(data - col_med), axis=0)
    col_mad = np.where(col_mad == 0, 1.0, col_mad)

    while injected < target:
        mode = rng.randint(0, 5)
        cs = rng.choice(d, max(1, int(d * 0.3)), replace=False)

        if mode == 0:
            length = rng.randint(5, 20)
            s = rng.randint(0, max(1, n - length))
            e = s + length
            for c in cs:
                data[s:e, c] = data[s:e, c] * rng.uniform(3.0, 8.0)

        elif mode == 1:
            length = rng.randint(100, 500)
            s = rng.randint(0, max(1, n - length))
            e = s + length
            for c in cs:
                data[s:e, c] += col_mad[c] * rng.uniform(4.0, 10.0) * rng.choice([-1, 1])

        elif mode == 2:
            s = rng.randint(0, n)
            e = s + 1
            for c in cs:
                data[s:e, c] += col_mad[c] * rng.uniform(10.0, 20.0) * rng.choice([-1, 1])

        elif mode == 3:
            length = rng.randint(50, 200)
            s = rng.randint(0, max(1, n - length))
            e = s + length
            src_s = rng.randint(0, max(1, n - length))
            for c in cs:
                segment = data_normal[src_s : src_s + length, c][::-1].copy()
                segment = segment + col_mad[c] * rng.uniform(2.0, 5.0) * rng.choice([-1, 1])
                data[s:e, c] = segment

        else:
            length = rng.randint(100, 300)
            s = rng.randint(0, max(1, n - length))
            e = s + length
            for c in cs:
                data[s:e, c] += rng.normal(col_med[c] * 2, col_mad[c] * 4, e - s)

        e = min(e, n)
        injected += int(np.sum(labels[s:e] == 0))
        labels[s:e] = 1

    return data, labels
