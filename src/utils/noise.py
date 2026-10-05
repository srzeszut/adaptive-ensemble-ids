import numpy as np

FAILURE_MODES = {0: "PANIC", 1: "CONFUSION", 2: "FLATLINE"}


def corrupt_test_scores(
    test_scores: dict[str, np.ndarray],
    models_to_break: list[str],
    mode: int,
    labels: np.ndarray,
    seed: int,
    noise_fraction: float = 0.15,
) -> tuple[dict[str, np.ndarray], list[tuple[int, int]]]:
    rng = np.random.RandomState(seed)
    n_samples = len(next(iter(test_scores.values())))
    win_size = int((n_samples * noise_fraction) / 2)

    s1 = rng.randint(int(n_samples * 0.1), int(n_samples * 0.4))
    s2 = rng.randint(int(n_samples * 0.6), int(n_samples * 0.8))
    second_half = int(n_samples * 0.5)
    normal_idx = np.where(labels[second_half:-win_size] == 0)[0]
    if len(normal_idx) > 0:
        s2 = second_half + rng.choice(normal_idx)
    windows = [(s1, s1 + win_size), (s2, s2 + win_size)]

    corrupted = {name: sc.copy() for name, sc in test_scores.items()}
    for name in models_to_break:
        sc = corrupted[name]
        p99 = np.percentile(sc, 99.0) + 1e-5
        for start, end in windows:
            size = end - start
            if mode == 0:
                sc[start:end] = rng.normal(p99 * 5.0, p99 * 5.0 * 0.001, size)
            elif mode == 1:
                sc[start:end] = rng.normal(loc=0.0, scale=1.0, size=size) * p99
            else:
                sc[start:end] = rng.uniform(0.0, 1e-5, size)

    print(f"      [FAILURE] {FAILURE_MODES[mode]} in {models_to_break}, windows {windows}")
    return corrupted, windows
