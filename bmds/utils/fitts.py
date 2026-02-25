import numpy as np
from typing import Tuple


def index_of_difficulty(distance: float, target_width: float) -> float:
    return np.log2(distance / max(target_width, 1e-6) + 1)


def predicted_movement_time(distance: float, target_width: float,
                            a: float, b: float) -> float:
    id_val = index_of_difficulty(distance, target_width)
    return a + b * id_val


def fit_fitts_law(distances: np.ndarray, target_widths: np.ndarray,
                  movement_times: np.ndarray) -> Tuple[float, float, float]:
    ids = np.array([index_of_difficulty(d, w) for d, w in zip(distances, target_widths)])


    A = np.column_stack([np.ones_like(ids), ids])
    result, residuals, _, _ = np.linalg.lstsq(A, movement_times, rcond=None)
    a, b = result


    ss_res = np.sum((movement_times - (a + b * ids)) ** 2)
    ss_tot = np.sum((movement_times - np.mean(movement_times)) ** 2)
    r_squared = 1 - ss_res / max(ss_tot, 1e-12)

    return float(a), float(b), float(r_squared)
