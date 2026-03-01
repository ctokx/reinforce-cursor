import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

from bmds.config import CONTROL_TIMESTEP, DATA_PROCESSED_DIR
from bmds.data.parser import Trajectory
from bmds.data.trajectory_db import TrajectoryDatabase
from bmds.env.sim2screen import Sim2ScreenMapper
from bmds.env.mouse_reach_env import MouseReachEnv
from bmds.reward.biomechanical_reward import BiomechanicalReward

DEFAULT_DATASET_PATH = DATA_PROCESSED_DIR / "offline_rl_dataset.npz"

class DatasetBuilder:
    def __init__(self,
                 env: MouseReachEnv,
                 mapper: Sim2ScreenMapper,
                 reward_fn: Optional[BiomechanicalReward] = None,
                 kp: float = 8.0,
                 kd: float = 2.0,
                 max_desired_speed: float = 3.0,
                 reward_clip: Tuple[float, float] = (-50.0, 60.0)):
        self.env = env
        self.mapper = mapper
        self.reward_fn = reward_fn
        self.kp = kp
        self.kd = kd
        self.max_desired_speed = max_desired_speed
        self.reward_clip = reward_clip

    def build_dataset(self, db: TrajectoryDatabase,
                      max_trajectories: Optional[int] = None,
                      verbose: bool = True) -> dict:
        all_obs, all_actions, all_rewards, all_terminals, all_timeouts = [], [], [], [], []

        n = min(len(db), max_trajectories) if max_trajectories else len(db)
        iterator = tqdm(range(n), desc="Building RL dataset") if verbose else range(n)

        success_count = 0
        for i in iterator:
            traj = db.get_trajectory(i)
            try:
                result = self._convert_trajectory(traj)
                if result is not None:
                    obs_seq, act_seq, rew_seq, term_seq, timeout_seq = result
                    all_obs.extend(obs_seq)
                    all_actions.extend(act_seq)
                    all_rewards.extend(rew_seq)
                    all_terminals.extend(term_seq)
                    all_timeouts.extend(timeout_seq)
                    success_count += 1
            except Exception:
                continue

        if verbose:
            print(f"Converted {success_count}/{n} trajectories, "
                  f"{len(all_obs)} total transitions")

        return {
            "observations": np.array(all_obs, dtype=np.float32),
            "actions": np.array(all_actions, dtype=np.float32),
            "rewards": np.array(all_rewards, dtype=np.float32),
            "terminals": np.array(all_terminals, dtype=np.float32),
            "timeouts": np.array(all_timeouts, dtype=np.float32),
        }

    def _convert_trajectory(self, traj: Trajectory) -> Optional[tuple]:
        points = traj.points
        if len(points) < 4:
            return None

        screen_start = points[0, :2]
        screen_end = points[-1, :2]
        desk_start = self.mapper.screen_to_desk(int(screen_start[0]), int(screen_start[1]))
        desk_end = self.mapper.screen_to_desk(int(screen_end[0]), int(screen_end[1]))

        total_time = points[-1, 2] - points[0, 2]
        if total_time < 0.02:
            return None

        n_steps = max(int(total_time / CONTROL_TIMESTEP), 2)
        t_resampled = np.linspace(0, total_time, n_steps)
        t_orig = points[:, 2] - points[0, 2]
        x_interp = np.interp(t_resampled, t_orig, points[:, 0])
        y_interp = np.interp(t_resampled, t_orig, points[:, 1])

        desk_positions = self.mapper.screen_to_desk_array(
            np.column_stack([x_interp, y_interp])
        )

        obs, _ = self.env.reset(start_pos=desk_start, target_pos=desk_end)
        if self.reward_fn is not None:
            self.reward_fn.reset()

        obs_seq, act_seq, rew_seq, term_seq, timeout_seq = [], [], [], [], []

        for step_idx in range(n_steps - 1):
            desired_pos = desk_positions[step_idx + 1]
            mouse_pos = self.env.obs_dict["mouse_pos"]
            mouse_vel = self.env.obs_dict["mouse_vel"]

            pos_error = desired_pos - mouse_pos
            vel_desired = (desk_positions[min(step_idx + 2, n_steps - 1)] - desired_pos) / CONTROL_TIMESTEP
            vel_norm = np.linalg.norm(vel_desired)
            if vel_norm > self.max_desired_speed and vel_norm > 1e-9:
                vel_desired = vel_desired * (self.max_desired_speed / vel_norm)
            vel_error = vel_desired - mouse_vel

            action = np.clip(
                self.kp * pos_error + self.kd * vel_error,
                self.env.action_space.low,
                self.env.action_space.high,
            ).astype(np.float32)

            obs_seq.append(obs.copy())
            act_seq.append(action)

            obs, env_reward, terminated, truncated, info = self.env.step(action)

            if self.reward_fn is not None:
                step_rwd = self.reward_fn.step_reward(self.env.obs_dict, action)
                reward = self.reward_fn.compute_total_reward(step_rwd)
                if terminated:
                    reward += 50.0
                    ep = self.reward_fn.episode_reward()
                    reward += (
                        ep.get("fitts_compliance",  0.0) * 5.0
                        + ep.get("path_efficiency", 0.0) * 4.0
                        + ep.get("profile_shape",   0.0) * 2.0
                        + ep.get("submovements",    0.0) * 1.0
                    )
            else:
                reward = env_reward
            reward = float(np.clip(reward, self.reward_clip[0], self.reward_clip[1]))

            is_last = (step_idx == n_steps - 2)
            rew_seq.append(reward)
            term_seq.append(1.0 if (terminated or is_last) else 0.0)
            timeout_seq.append(1.0 if truncated else 0.0)

            if terminated:
                break

        if len(obs_seq) < 2:
            return None

        return obs_seq, act_seq, rew_seq, term_seq, timeout_seq

    def save_dataset(self, dataset: dict, path: Optional[Path] = None):
        path = Path(path or DEFAULT_DATASET_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **dataset)
        print(f"Dataset saved to {path} ({dataset['observations'].shape[0]} transitions)")

    @staticmethod
    def load_dataset(path: Optional[Path] = None) -> dict:
        path = Path(path or DEFAULT_DATASET_PATH)
        data = np.load(path)
        return {k: data[k] for k in data.files}
