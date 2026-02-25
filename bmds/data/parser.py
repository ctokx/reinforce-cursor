import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from bmds.config import (
    PAUSE_THRESHOLD_MS, MIN_TRAJECTORY_POINTS, MIN_TRAJECTORY_DISTANCE,
)


@dataclass
class Trajectory:
    points: np.ndarray
    user_id: str = ""
    session_id: str = ""

    @property
    def start_pos(self) -> Tuple[int, int]:
        return (int(self.points[0, 0]), int(self.points[0, 1]))

    @property
    def end_pos(self) -> Tuple[int, int]:
        return (int(self.points[-1, 0]), int(self.points[-1, 1]))

    @property
    def duration(self) -> float:
        return float(self.points[-1, 2] - self.points[0, 2])

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.points[-1, :2] - self.points[0, :2]))

    @property
    def n_points(self) -> int:
        return len(self.points)

    def __repr__(self):
        return (f"Trajectory(n={self.n_points}, dist={self.distance:.0f}px, "
                f"dur={self.duration:.3f}s, user={self.user_id})")


class BalabitParser:

    def __init__(self,
                 pause_threshold_ms: float = PAUSE_THRESHOLD_MS,
                 min_points: int = MIN_TRAJECTORY_POINTS,
                 min_distance: float = MIN_TRAJECTORY_DISTANCE):
        self.pause_threshold_s = pause_threshold_ms / 1000.0
        self.min_points = min_points
        self.min_distance = min_distance

    def parse_session(self, filepath: Path, user_id: str = "",
                      session_id: str = "") -> List[Trajectory]:
        if not session_id:
            session_id = filepath.stem

        df = self._read_session_file(filepath)
        if df is None or len(df) < self.min_points:
            return []


        df = df[df["state"] == "Move"].copy()
        if len(df) < self.min_points:
            return []


        df = df.sort_values("client_timestamp").reset_index(drop=True)


        segments = self._segment_trajectories(df)

        trajectories = []
        for seg in segments:
            if len(seg) < self.min_points:
                continue

            points = seg[["x", "y", "client_timestamp"]].values.astype(np.float64)

            points[:, 2] -= points[0, 2]


            dt = np.diff(points[:, 2])
            valid_mask = np.concatenate([[True], dt > 0.001])
            points = points[valid_mask]

            if len(points) < self.min_points:
                continue

            traj = Trajectory(
                points=points,
                user_id=user_id,
                session_id=session_id,
            )

            if traj.distance >= self.min_distance and traj.n_points >= self.min_points:
                trajectories.append(traj)

        return trajectories

    def _read_session_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        try:


            df = pd.read_csv(
                filepath,
                sep=",",
                on_bad_lines="skip",
            )


            df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]


            col_map = {}
            for col in df.columns:
                if "record" in col and "timestamp" in col:
                    col_map[col] = "record_timestamp"
                elif "client" in col and "timestamp" in col:
                    col_map[col] = "client_timestamp"
                elif col in ("button",):
                    col_map[col] = "button"
                elif col in ("state",):
                    col_map[col] = "state"
                elif col == "x":
                    col_map[col] = "x"
                elif col == "y":
                    col_map[col] = "y"
            df = df.rename(columns=col_map)

            required = ["client_timestamp", "state", "x", "y"]
            if not all(c in df.columns for c in required):

                df = pd.read_csv(
                    filepath, sep=",", header=None,
                    names=["record_timestamp", "client_timestamp",
                           "button", "state", "x", "y"],
                    on_bad_lines="skip",
                )


            for col in ["client_timestamp", "x", "y"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["x", "y", "client_timestamp"])
            if len(df) > 0:
                return df

        except Exception as e:
            print(f"Warning: Could not parse {filepath}: {e}")
        return None

    def _segment_trajectories(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        timestamps = df["client_timestamp"].values
        dt = np.diff(timestamps)


        pause_mask = dt > self.pause_threshold_s


        x = df["x"].values
        y = df["y"].values
        dx = np.diff(x)
        dy = np.diff(y)
        stationary = (dx == 0) & (dy == 0)


        extended_stationary = np.zeros(len(stationary), dtype=bool)
        run_len = 0
        for i in range(len(stationary)):
            if stationary[i]:
                run_len += 1
                if run_len >= 3:
                    extended_stationary[i] = True
            else:
                run_len = 0

        boundaries = pause_mask | extended_stationary


        segments = []
        seg_start = 0
        for i, is_boundary in enumerate(boundaries):
            if is_boundary:
                segment = df.iloc[seg_start:i+1]
                if len(segment) >= self.min_points:
                    segments.append(segment)
                seg_start = i + 1


        if seg_start < len(df):
            segment = df.iloc[seg_start:]
            if len(segment) >= self.min_points:
                segments.append(segment)

        return segments

    def parse_all_sessions(self, session_files: List[Tuple[str, Path]],
                           verbose: bool = True) -> List[Trajectory]:
        from tqdm import tqdm

        all_trajectories = []
        iterator = tqdm(session_files, desc="Parsing sessions") if verbose else session_files

        for user_id, filepath in iterator:
            trajectories = self.parse_session(filepath, user_id=user_id)
            all_trajectories.extend(trajectories)

        if verbose:
            print(f"Parsed {len(all_trajectories)} trajectories from "
                  f"{len(session_files)} sessions")

        return all_trajectories
