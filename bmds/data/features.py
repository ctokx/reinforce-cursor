import numpy as np
from dataclasses import dataclass
from typing import List

from bmds.data.parser import Trajectory
from bmds.utils.kinematics import (
    compute_kinematics, count_submovements, normalize_speed_profile,
    KinematicProfile,
)
from bmds.utils.fitts import index_of_difficulty


@dataclass
class TrajectoryFeatures:

    speed_profile: np.ndarray
    peak_speed: float
    mean_speed: float
    speed_std: float


    max_acceleration: float
    max_deceleration: float
    mean_acceleration: float


    mean_jerk: float
    max_jerk: float


    mean_curvature: float
    max_curvature: float


    path_efficiency: float
    path_length: float
    straight_line_distance: float


    duration: float
    num_submovements: int


    fitts_id: float


    normalized_speed_profile: np.ndarray


class TrajectoryFeatureExtractor:

    def __init__(self, target_width_px: float = 10.0):
        self.target_width = target_width_px

    def extract(self, traj: Trajectory) -> TrajectoryFeatures:
        if traj.n_points < 4:
            return self._empty_features(traj)

        kin = compute_kinematics(traj.points)
        n_sub = count_submovements(kin.speed)
        norm_profile = normalize_speed_profile(kin.speed)

        return TrajectoryFeatures(
            speed_profile=kin.speed,
            peak_speed=kin.peak_speed,
            mean_speed=kin.mean_speed,
            speed_std=float(np.std(kin.speed)),
            max_acceleration=kin.max_acceleration,
            max_deceleration=kin.max_deceleration,
            mean_acceleration=float(np.mean(kin.accel_magnitude)),
            mean_jerk=kin.mean_jerk,
            max_jerk=float(np.max(kin.jerk_magnitude)) if len(kin.jerk_magnitude) > 0 else 0.0,
            mean_curvature=float(np.mean(np.abs(kin.curvature))),
            max_curvature=float(np.max(np.abs(kin.curvature))),
            path_efficiency=kin.path_efficiency,
            path_length=kin.path_length,
            straight_line_distance=kin.straight_line_distance,
            duration=kin.duration,
            num_submovements=n_sub,
            fitts_id=index_of_difficulty(traj.distance, self.target_width),
            normalized_speed_profile=norm_profile,
        )

    def extract_batch(self, trajectories: List[Trajectory],
                      verbose: bool = True) -> List[TrajectoryFeatures]:
        from tqdm import tqdm

        features = []
        iterator = tqdm(trajectories, desc="Extracting features") if verbose else trajectories

        for traj in iterator:
            try:
                feat = self.extract(traj)
                features.append(feat)
            except Exception as e:

                if verbose:
                    print(f"Warning: skipped trajectory ({e})")
                continue

        return features

    def _empty_features(self, traj: Trajectory) -> TrajectoryFeatures:
        return TrajectoryFeatures(
            speed_profile=np.array([0.0]),
            peak_speed=0.0, mean_speed=0.0, speed_std=0.0,
            max_acceleration=0.0, max_deceleration=0.0, mean_acceleration=0.0,
            mean_jerk=0.0, max_jerk=0.0,
            mean_curvature=0.0, max_curvature=0.0,
            path_efficiency=0.0,
            path_length=0.0,
            straight_line_distance=traj.distance,
            duration=traj.duration,
            num_submovements=0,
            fitts_id=0.0,
            normalized_speed_profile=np.zeros(100),
        )
