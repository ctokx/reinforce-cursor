import numpy as np

def generate_movement_pairs(n: int, rng: np.random.Generator,
                            screen_res=(1920, 1080)) -> list:
    pairs = []
    for _ in range(n):
        while True:
            sx = rng.integers(100, screen_res[0] - 100)
            sy = rng.integers(100, screen_res[1] - 100)
            ex = rng.integers(100, screen_res[0] - 100)
            ey = rng.integers(100, screen_res[1] - 100)
            dist = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
            if 100 < dist < 800:
                pairs.append(((int(sx), int(sy)), (int(ex), int(ey))))
                break
    return pairs

def sample_human_trajectories(db, pairs: list,
                              tolerance: float = 0.3) -> list:
    distances = db.get_feature("distance")
    all_indices = np.arange(len(distances))
    rng = np.random.default_rng(42)
    trajectories = []

    for start, end in pairs:
        target_dist = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        lo = target_dist * (1 - tolerance)
        hi = target_dist * (1 + tolerance)
        mask = (distances >= lo) & (distances <= hi)
        candidates = all_indices[mask]

        if len(candidates) == 0:
            idx = int(np.argmin(np.abs(distances - target_dist)))
        else:
            idx = int(rng.choice(candidates))

        traj = db.get_trajectory(idx)
        trajectories.append(traj.points)

    return trajectories

def generate_linear_trajectory(start: np.ndarray, end: np.ndarray,
                               duration: float = 0.5,
                               dt: float = 0.01) -> np.ndarray:
    n_steps = max(int(duration / dt), 5)
    t = np.linspace(0, duration, n_steps)
    frac = t / duration
    x = start[0] + (end[0] - start[0]) * frac
    y = start[1] + (end[1] - start[1]) * frac
    return np.column_stack([x, y, t])

def generate_bezier_trajectory(start: np.ndarray, end: np.ndarray,
                               duration: float = 0.5,
                               dt: float = 0.01,
                               rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    n_steps = max(int(duration / dt), 5)
    t = np.linspace(0, duration, n_steps)

    s = t / duration
    s_eased = 3 * s ** 2 - 2 * s ** 3

    mid = (start + end) / 2
    spread = np.linalg.norm(end - start) * 0.3
    perp = np.array([-(end[1] - start[1]), end[0] - start[0]])
    perp_norm = np.linalg.norm(perp)
    if perp_norm > 1e-9:
        perp = perp / perp_norm
    else:
        perp = np.array([0.0, 1.0])

    cp1 = start + (end - start) * 0.33 + perp * rng.uniform(-spread, spread)
    cp2 = start + (end - start) * 0.66 + perp * rng.uniform(-spread, spread)

    u = s_eased
    x = ((1 - u) ** 3 * start[0] + 3 * (1 - u) ** 2 * u * cp1[0] +
         3 * (1 - u) * u ** 2 * cp2[0] + u ** 3 * end[0])
    y = ((1 - u) ** 3 * start[1] + 3 * (1 - u) ** 2 * u * cp1[1] +
         3 * (1 - u) * u ** 2 * cp2[1] + u ** 3 * end[1])

    return np.column_stack([x, y, t])
