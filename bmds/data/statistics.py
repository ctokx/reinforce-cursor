import json
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from bmds.config import DATA_PROCESSED_DIR
from bmds.data.trajectory_db import TrajectoryDatabase
from bmds.utils.fitts import fit_fitts_law

DEFAULT_STATS_PATH = DATA_PROCESSED_DIR / "human_motion_stats.json"


@dataclass
class HumanMotionStatistics:


    speed_mean: float = 0.0
    speed_std: float = 0.0
    speed_p5: float = 0.0
    speed_p25: float = 0.0
    speed_p50: float = 0.0
    speed_p75: float = 0.0
    speed_p95: float = 0.0
    speed_max: float = 0.0


    accel_max: float = 0.0
    decel_max: float = 0.0
    accel_mean: float = 0.0


    jerk_mean: float = 0.0
    jerk_p95: float = 0.0
    jerk_max: float = 0.0


    efficiency_mean: float = 0.0
    efficiency_std: float = 0.0
    efficiency_min: float = 0.0


    submovement_mean: float = 0.0
    submovement_std: float = 0.0


    fitts_a: float = 0.0
    fitts_b: float = 0.0
    fitts_r_squared: float = 0.0


    velocity_profile_template: list = field(default_factory=lambda: [0.0] * 100)


    duration_mean: float = 0.0
    duration_std: float = 0.0
    duration_p5: float = 0.0
    duration_p95: float = 0.0


    curvature_mean: float = 0.0
    curvature_p95: float = 0.0


def compute_statistics(db: TrajectoryDatabase,
                       verbose: bool = True) -> HumanMotionStatistics:
    stats = HumanMotionStatistics()
    n = len(db)

    if verbose:
        print(f"Computing statistics from {n} trajectories...")


    peak_speeds = db.get_feature("peak_speed")
    mean_speeds = db.get_feature("mean_speed")
    durations = db.get_feature("duration")
    distances = db.get_feature("distance")
    efficiencies = db.get_feature("path_efficiency")
    fitts_ids = db.get_feature("fitts_id")
    submovements = db.get_feature("num_submovements")
    max_accels = db.get_feature("max_acceleration")
    max_decels = db.get_feature("max_deceleration")
    mean_jerks = db.get_feature("mean_jerk")
    max_jerks = db.get_feature("max_jerk")
    mean_curvatures = db.get_feature("mean_curvature")
    profiles = db.get_feature("normalized_profiles")


    valid = (durations > 0) & (distances > 0) & np.isfinite(peak_speeds)
    if verbose:
        print(f"  Valid trajectories: {np.sum(valid)} / {n}")

    peak_speeds = peak_speeds[valid]
    mean_speeds = mean_speeds[valid]
    durations = durations[valid]
    distances = distances[valid]
    efficiencies = efficiencies[valid]
    fitts_ids = fitts_ids[valid]
    submovements = submovements[valid]
    max_accels = max_accels[valid]
    max_decels = max_decels[valid]
    mean_jerks = mean_jerks[valid]
    max_jerks = max_jerks[valid]
    mean_curvatures = mean_curvatures[valid]
    profiles = profiles[valid]


    all_speeds = peak_speeds
    stats.speed_mean = float(np.mean(all_speeds))
    stats.speed_std = float(np.std(all_speeds))
    stats.speed_p5 = float(np.percentile(all_speeds, 5))
    stats.speed_p25 = float(np.percentile(all_speeds, 25))
    stats.speed_p50 = float(np.percentile(all_speeds, 50))
    stats.speed_p75 = float(np.percentile(all_speeds, 75))
    stats.speed_p95 = float(np.percentile(all_speeds, 95))
    stats.speed_max = float(np.percentile(all_speeds, 99))


    stats.accel_max = float(np.percentile(max_accels[max_accels > 0], 99)) if np.any(max_accels > 0) else 0.0
    stats.decel_max = float(np.abs(np.percentile(max_decels[max_decels < 0], 1))) if np.any(max_decels < 0) else 0.0
    stats.accel_mean = float(np.mean(np.abs(max_accels)))


    valid_jerk = mean_jerks[np.isfinite(mean_jerks) & (mean_jerks > 0)]
    if len(valid_jerk) > 0:
        stats.jerk_mean = float(np.mean(valid_jerk))
        stats.jerk_p95 = float(np.percentile(valid_jerk, 95))
        stats.jerk_max = float(np.percentile(valid_jerk, 99))


    valid_eff = efficiencies[(efficiencies > 0) & (efficiencies <= 1)]
    stats.efficiency_mean = float(np.mean(valid_eff))
    stats.efficiency_std = float(np.std(valid_eff))
    stats.efficiency_min = float(np.percentile(valid_eff, 5))


    stats.submovement_mean = float(np.mean(submovements))
    stats.submovement_std = float(np.std(submovements))


    target_widths = np.full_like(distances, 10.0)
    valid_fitts = (durations > 0) & (distances > 20) & np.isfinite(fitts_ids)
    if np.sum(valid_fitts) > 10:
        a, b, r2 = fit_fitts_law(
            distances[valid_fitts],
            target_widths[valid_fitts],
            durations[valid_fitts],
        )
        stats.fitts_a = a
        stats.fitts_b = b
        stats.fitts_r_squared = r2


    valid_profiles = profiles[np.all(np.isfinite(profiles), axis=1)]
    if len(valid_profiles) > 0:
        avg_profile = np.mean(valid_profiles, axis=0)
        stats.velocity_profile_template = avg_profile.tolist()


    stats.duration_mean = float(np.mean(durations))
    stats.duration_std = float(np.std(durations))
    stats.duration_p5 = float(np.percentile(durations, 5))
    stats.duration_p95 = float(np.percentile(durations, 95))


    valid_curv = mean_curvatures[np.isfinite(mean_curvatures)]
    if len(valid_curv) > 0:
        stats.curvature_mean = float(np.mean(valid_curv))
        stats.curvature_p95 = float(np.percentile(valid_curv, 95))

    if verbose:
        print(f"  Speed: mean={stats.speed_mean:.0f}, p95={stats.speed_p95:.0f} px/s")
        print(f"  Accel: max(p99)={stats.accel_max:.0f} px/s^2")
        print(f"  Efficiency: mean={stats.efficiency_mean:.3f}")
        print(f"  Fitts's law: MT = {stats.fitts_a:.3f} + {stats.fitts_b:.3f} * ID  (R²={stats.fitts_r_squared:.3f})")
        print(f"  Duration: mean={stats.duration_mean:.3f}s, p95={stats.duration_p95:.3f}s")

    return stats


def save_statistics(stats: HumanMotionStatistics,
                    path: Optional[Path] = None):
    path = Path(path or DEFAULT_STATS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(stats), f, indent=2)
    print(f"Statistics saved to {path}")


def load_statistics(path: Optional[Path] = None) -> HumanMotionStatistics:
    path = Path(path or DEFAULT_STATS_PATH)
    with open(path, "r") as f:
        data = json.load(f)
    stats = HumanMotionStatistics(**data)
    return stats
