import numpy as np
from typing import Dict, List, Optional

from bmds.data.statistics import HumanMotionStatistics
from bmds.env.sim2screen import Sim2ScreenMapper
from bmds.utils.fitts import index_of_difficulty
from bmds.utils.kinematics import (
    normalize_speed_profile, count_submovements,
)

class BiomechanicalReward:

    def __init__(self, human_stats: HumanMotionStatistics,
                 mapper: Sim2ScreenMapper,
                 weights: Optional[Dict[str, float]] = None):
        self.stats = human_stats
        self.mapper = mapper
        self.weights = weights or {
            "reach": 1.0,
            "velocity_match": 0.5,
            "acceleration_penalty": 0.3,
            "jerk_penalty": 0.2,
            "effort": 0.1,
            "profile_shape": 1.0,
            "fitts_compliance": 2.0,
            "path_efficiency": 1.5,
            "submovements": 0.5,
            "approach_bonus": 1.0,
            "velocity_damping": 1.0,
            "running_path_efficiency": 0.5,
        }

        self._buffer: List[tuple] = []
        self._prev_screen_vel = np.zeros(2)
        self._prev_screen_accel = np.zeros(2)

        self._template = np.array(human_stats.velocity_profile_template)

    def reset(self):
        self._buffer = []
        self._prev_screen_vel = np.zeros(2)
        self._prev_screen_accel = np.zeros(2)

    def step_reward(self, obs_dict: Dict, action: Optional[np.ndarray] = None) -> Dict[str, float]:
        mouse_pos = obs_dict["mouse_pos"]
        mouse_vel = obs_dict["mouse_vel"]
        target_pos = obs_dict["target_pos"]
        t = float(obs_dict["time"][0])

        screen_pos = np.array(self.mapper.desk_to_screen(mouse_pos[0], mouse_pos[1]),
                              dtype=np.float64)
        screen_vel = np.array(self.mapper.desk_vel_to_screen_vel(mouse_vel[0], mouse_vel[1]))

        self._buffer.append((screen_pos[0], screen_pos[1], t))

        rewards = {}

        reach_dist = np.linalg.norm(mouse_pos - target_pos)
        rewards["reach"] = -reach_dist * 10.0

        rewards["approach_bonus"] = 3.0 * np.exp(-reach_dist / 0.015)

        if reach_dist < 0.02:
            damp = 1.0 - reach_dist / 0.02
            rewards["velocity_damping"] = -np.linalg.norm(mouse_vel) * damp * 3.0
        else:
            rewards["velocity_damping"] = 0.0

        if len(self._buffer) >= 5:
            _traj = np.array(self._buffer)
            _straight = np.linalg.norm(_traj[-1, :2] - _traj[0, :2])
            _segs = np.hypot(np.diff(_traj[:, 0]), np.diff(_traj[:, 1]))
            _path_len = _segs.sum()
            if _path_len > 1.0:
                _run_eff = _straight / _path_len
                _eff_target = self.stats.efficiency_mean + 0.2   # tolerance band
                rewards["running_path_efficiency"] = -max(0.0, _run_eff - _eff_target)
            else:
                rewards["running_path_efficiency"] = 0.0
        else:
            rewards["running_path_efficiency"] = 0.0

        speed = np.linalg.norm(screen_vel)
        speed_p5 = self.stats.speed_p5
        speed_p95 = self.stats.speed_p95

        if speed_p95 > 0:
            if speed > speed_p95:
                rewards["velocity_match"] = -(speed - speed_p95) / speed_p95
            elif speed < speed_p5 and reach_dist > 0.01:
                rewards["velocity_match"] = -(speed_p5 - speed) / max(speed_p5, 1.0)
            else:
                rewards["velocity_match"] = 0.0
        else:
            rewards["velocity_match"] = 0.0

        if len(self._buffer) >= 3:
            dt = self._buffer[-1][2] - self._buffer[-2][2]
            if dt > 1e-6:
                accel = (screen_vel - self._prev_screen_vel) / dt
                accel_mag = np.linalg.norm(accel)

                accel_limit = self.stats.accel_max
                if accel_limit > 0 and accel_mag > accel_limit:
                    rewards["acceleration_penalty"] = -(accel_mag - accel_limit) / accel_limit
                else:
                    rewards["acceleration_penalty"] = 0.0

                if len(self._buffer) >= 4:
                    jerk = (accel - self._prev_screen_accel) / dt
                    jerk_mag = np.linalg.norm(jerk)

                    jerk_limit = self.stats.jerk_p95
                    if jerk_limit > 0 and jerk_mag > jerk_limit:
                        rewards["jerk_penalty"] = -(jerk_mag - jerk_limit) / jerk_limit
                    else:
                        rewards["jerk_penalty"] = 0.0

                self._prev_screen_accel = accel
        else:
            rewards["acceleration_penalty"] = 0.0
            rewards["jerk_penalty"] = 0.0

        self._prev_screen_vel = screen_vel

        if action is not None:
            rewards["effort"] = -float(np.mean(action ** 2))
        else:
            rewards["effort"] = 0.0

        return rewards

    def episode_reward(self) -> Dict[str, float]:
        rewards = {}

        if len(self._buffer) < 4:
            return {"profile_shape": 0.0, "fitts_compliance": 0.0,
                    "path_efficiency": 0.0, "submovements": 0.0}

        traj = np.array(self._buffer)

        dt = np.diff(traj[:, 2])
        dt = np.maximum(dt, 1e-9)
        dx = np.diff(traj[:, 0])
        dy = np.diff(traj[:, 1])
        speed = np.sqrt(dx**2 + dy**2) / dt

        norm_profile = normalize_speed_profile(speed)
        if len(self._template) == len(norm_profile) and np.any(self._template > 0):
            correlation = np.corrcoef(norm_profile, self._template)[0, 1]
            if np.isfinite(correlation):

                rewards["profile_shape"] = max(correlation - 0.5, -1.0)
            else:
                rewards["profile_shape"] = 0.0
        else:
            rewards["profile_shape"] = 0.0

        distance = np.linalg.norm(traj[-1, :2] - traj[0, :2])
        duration = traj[-1, 2] - traj[0, 2]
        target_width = 10.0
        id_val = index_of_difficulty(max(distance, 1.0), target_width)

        if self.stats.fitts_b > 0 and duration > 0:
            expected_mt = self.stats.fitts_a + self.stats.fitts_b * id_val
            if expected_mt > 0:
                fitts_error = abs(duration - expected_mt) / expected_mt
                rewards["fitts_compliance"] = -fitts_error
            else:
                rewards["fitts_compliance"] = 0.0
        else:
            rewards["fitts_compliance"] = 0.0

        segment_lengths = np.sqrt(dx**2 + dy**2)
        path_length = np.sum(segment_lengths)
        straight_line = distance
        efficiency = straight_line / max(path_length, 1e-6)

        eff_mean = self.stats.efficiency_mean
        eff_std = max(self.stats.efficiency_std, 0.01)
        if efficiency > eff_mean + eff_std:
            rewards["path_efficiency"] = -(efficiency - (eff_mean + eff_std)) / eff_std
        elif efficiency < eff_mean - 2 * eff_std:
            rewards["path_efficiency"] = -(eff_mean - 2 * eff_std - efficiency) / eff_std
        else:
            rewards["path_efficiency"] = 0.0

        n_sub = count_submovements(speed)
        expected = self.stats.submovement_mean
        if expected > 0:
            rewards["submovements"] = -abs(n_sub - expected) / expected
        else:
            rewards["submovements"] = 0.0

        return rewards

    def compute_total_reward(self, step_rewards: Dict[str, float],
                             episode_rewards: Optional[Dict[str, float]] = None) -> float:
        total = 0.0
        for key, value in step_rewards.items():
            w = self.weights.get(key, 0.0)
            total += w * value

        if episode_rewards is not None:
            for key, value in episode_rewards.items():
                w = self.weights.get(key, 0.0)
                total += w * value

        return total
