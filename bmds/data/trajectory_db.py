import h5py
import numpy as np
from pathlib import Path
from typing import List, Optional

from bmds.config import DATA_PROCESSED_DIR
from bmds.data.parser import Trajectory
from bmds.data.features import TrajectoryFeatures, TrajectoryFeatureExtractor

DEFAULT_DB_PATH = DATA_PROCESSED_DIR / "trajectory_db.h5"


class TrajectoryDatabase:

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)

    def build(self, trajectories: List[Trajectory],
              features: Optional[List[TrajectoryFeatures]] = None,
              verbose: bool = True):
        if features is None:
            extractor = TrajectoryFeatureExtractor()
            features = extractor.extract_batch(trajectories, verbose=verbose)


        assert len(trajectories) == len(features),\
            f"Mismatch: {len(trajectories)} trajectories, {len(features)} features"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        n = len(trajectories)

        if verbose:
            print(f"Building trajectory database with {n} entries at {self.db_path}")

        with h5py.File(self.db_path, "w") as f:

            traj_grp = f.create_group("trajectories")
            for i, traj in enumerate(trajectories):
                ds = traj_grp.create_dataset(str(i), data=traj.points,
                                             compression="gzip")
                ds.attrs["user_id"] = traj.user_id
                ds.attrs["session_id"] = traj.session_id


            feat_grp = f.create_group("features")
            feat_grp.create_dataset("peak_speed",
                                    data=[ft.peak_speed for ft in features])
            feat_grp.create_dataset("mean_speed",
                                    data=[ft.mean_speed for ft in features])
            feat_grp.create_dataset("duration",
                                    data=[ft.duration for ft in features])
            feat_grp.create_dataset("distance",
                                    data=[ft.straight_line_distance for ft in features])
            feat_grp.create_dataset("path_length",
                                    data=[ft.path_length for ft in features])
            feat_grp.create_dataset("path_efficiency",
                                    data=[ft.path_efficiency for ft in features])
            feat_grp.create_dataset("fitts_id",
                                    data=[ft.fitts_id for ft in features])
            feat_grp.create_dataset("num_submovements",
                                    data=[ft.num_submovements for ft in features])
            feat_grp.create_dataset("max_acceleration",
                                    data=[ft.max_acceleration for ft in features])
            feat_grp.create_dataset("max_deceleration",
                                    data=[ft.max_deceleration for ft in features])
            feat_grp.create_dataset("mean_jerk",
                                    data=[ft.mean_jerk for ft in features])
            feat_grp.create_dataset("max_jerk",
                                    data=[ft.max_jerk for ft in features])
            feat_grp.create_dataset("mean_curvature",
                                    data=[ft.mean_curvature for ft in features])


            profiles = np.array([ft.normalized_speed_profile for ft in features])
            feat_grp.create_dataset("normalized_profiles", data=profiles,
                                    compression="gzip")


            meta = f.create_group("metadata")
            meta.attrs["count"] = n
            meta.attrs["users"] = list(set(t.user_id for t in trajectories))

        if verbose:
            print(f"Database built: {n} trajectories")

    def __len__(self) -> int:
        with h5py.File(self.db_path, "r") as f:
            return int(f["metadata"].attrs["count"])

    def get_trajectory(self, idx: int) -> Trajectory:
        with h5py.File(self.db_path, "r") as f:
            ds = f[f"trajectories/{idx}"]
            points = ds[:]
            user_id = ds.attrs.get("user_id", "")
            session_id = ds.attrs.get("session_id", "")
        return Trajectory(points=points, user_id=user_id, session_id=session_id)

    def get_feature(self, feature_name: str) -> np.ndarray:
        with h5py.File(self.db_path, "r") as f:
            return f[f"features/{feature_name}"][:]

