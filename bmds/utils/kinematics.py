import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class KinematicProfile:
    timestamps: np.ndarray
    positions: np.ndarray
    velocity: np.ndarray
    speed: np.ndarray
    acceleration: np.ndarray
    accel_magnitude: np.ndarray
    jerk: np.ndarray
    jerk_magnitude: np.ndarray
    curvature: np.ndarray
    path_length: float
    straight_line_distance: float
    path_efficiency: float
    duration: float
    peak_speed: float
    mean_speed: float
    max_acceleration: float
    max_deceleration: float
    mean_jerk: float


def compute_kinematics(points: np.ndarray) -> KinematicProfile:
    assert points.shape[1] == 3, "Expected (N, 3) array of [x, y, t]"
    assert len(points) >= 4, "Need at least 4 points for jerk computation"

    x = points[:, 0]
    y = points[:, 1]
    t = points[:, 2]

    dt = np.diff(t)

    dt = np.maximum(dt, 0.005)


    vx = np.diff(x) / dt
    vy = np.diff(y) / dt
    velocity = np.column_stack([vx, vy])
    speed = np.sqrt(vx**2 + vy**2)


    dt_a = dt[1:]
    ax = np.diff(vx) / dt_a
    ay = np.diff(vy) / dt_a
    acceleration = np.column_stack([ax, ay])
    accel_magnitude = np.sqrt(ax**2 + ay**2)


    dt_j = dt_a[1:]
    jx = np.diff(ax) / dt_j
    jy = np.diff(ay) / dt_j
    jerk = np.column_stack([jx, jy])
    jerk_magnitude = np.sqrt(jx**2 + jy**2)


    vx_c = vx[1:]
    vy_c = vy[1:]
    speed_cubed = (vx_c**2 + vy_c**2) ** 1.5
    speed_cubed = np.maximum(speed_cubed, 1e-12)
    curvature = (vx_c * ay - vy_c * ax) / speed_cubed


    segment_lengths = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    path_length = float(np.sum(segment_lengths))
    straight_line = float(np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2))
    path_efficiency = straight_line / max(path_length, 1e-9)


    tangential_accel = np.diff(speed) / dt_a
    max_accel = float(np.max(tangential_accel)) if len(tangential_accel) > 0 else 0.0
    max_decel = float(np.min(tangential_accel)) if len(tangential_accel) > 0 else 0.0

    return KinematicProfile(
        timestamps=t,
        positions=points[:, :2],
        velocity=velocity,
        speed=speed,
        acceleration=acceleration,
        accel_magnitude=accel_magnitude,
        jerk=jerk,
        jerk_magnitude=jerk_magnitude,
        curvature=curvature,
        path_length=path_length,
        straight_line_distance=straight_line,
        path_efficiency=path_efficiency,
        duration=float(t[-1] - t[0]),
        peak_speed=float(np.max(speed)),
        mean_speed=float(np.mean(speed)),
        max_acceleration=max_accel,
        max_deceleration=max_decel,
        mean_jerk=float(np.mean(jerk_magnitude)) if len(jerk_magnitude) > 0 else 0.0,
    )


def count_submovements(speed_profile: np.ndarray, min_prominence: float = 0.1) -> int:
    if len(speed_profile) < 3:
        return 1

    peak_speed = np.max(speed_profile)
    if peak_speed < 1e-6:
        return 0

    threshold = peak_speed * min_prominence


    peaks = []
    for i in range(1, len(speed_profile) - 1):
        if (speed_profile[i] > speed_profile[i-1] and
                speed_profile[i] > speed_profile[i+1] and
                speed_profile[i] > threshold):
            peaks.append(i)

    return max(len(peaks), 1) if peak_speed > threshold else 0


def normalize_speed_profile(speed: np.ndarray, n_samples: int = 100) -> np.ndarray:
    if len(speed) < 2:
        return np.zeros(n_samples)

    peak = np.max(speed)
    if peak < 1e-9:
        return np.zeros(n_samples)

    normalized = speed / peak
    x_orig = np.linspace(0, 1, len(normalized))
    x_new = np.linspace(0, 1, n_samples)
    return np.interp(x_new, x_orig, normalized)
