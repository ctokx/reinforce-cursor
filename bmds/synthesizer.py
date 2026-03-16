import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from bmds.config import MODELS_DIR, DEFAULT_SCREEN_RESOLUTION, DESK_X_RANGE, DESK_Y_RANGE
from bmds.env.sim2screen import Sim2ScreenMapper
from bmds.training.model_loader import load_policy, infer_algorithm_from_model_path

class BMDSSynthesizer:
    def __init__(self, policy, env, mapper: Sim2ScreenMapper):
        self.policy = policy
        self.env = env
        self.mapper = mapper

    @classmethod
    def load(cls, model_path: Optional[str] = None,
             screen_resolution: Tuple[int, int] = DEFAULT_SCREEN_RESOLUTION,
             algorithm: Optional[str] = None) -> "BMDSSynthesizer":
        if model_path is None:
            model_path = str(MODELS_DIR / "bmds_cql_policy.d3")
        algo = algorithm or infer_algorithm_from_model_path(model_path, default="cql")
        policy = load_policy(model_path, algorithm=algo, use_gpu=False)
        from bmds.env.mouse_reach_env import MouseReachEnv
        env = MouseReachEnv(screen_resolution=screen_resolution)
        mapper = Sim2ScreenMapper(
            desk_bounds_m=(DESK_X_RANGE, DESK_Y_RANGE),
            screen_resolution=screen_resolution,
        )
        return cls(policy, env, mapper)

    @classmethod
    def load_untrained(cls,
                       screen_resolution: Tuple[int, int] = DEFAULT_SCREEN_RESOLUTION) -> "BMDSSynthesizer":
        from bmds.env.mouse_reach_env import MouseReachEnv
        env = MouseReachEnv(screen_resolution=screen_resolution)
        mapper = Sim2ScreenMapper(
            desk_bounds_m=(DESK_X_RANGE, DESK_Y_RANGE),
            screen_resolution=screen_resolution,
        )
        return cls(policy=None, env=env, mapper=mapper)

    def generate(self, start: Tuple[int, int], end: Tuple[int, int],
                 screen_resolution: Optional[Tuple[int, int]] = None,
                 noise_seed: Optional[int] = None,
                 max_steps: int = 500,
                 ou_sigma: float = 0.0,
                 ou_theta: float = 3.0) -> List[Tuple[int, int, float]]:
        if noise_seed is not None:
            np.random.seed(noise_seed)
        ou_rng = np.random.default_rng(noise_seed)
        ou_state = np.zeros(2)
        dt = getattr(self.env, 'dt', 0.01)

        mapper = self.mapper
        if screen_resolution is not None and screen_resolution != (mapper.screen_w, mapper.screen_h):
            mapper = Sim2ScreenMapper(
                desk_bounds_m=((mapper.desk_x_min, mapper.desk_x_max),
                               (mapper.desk_y_min, mapper.desk_y_max)),
                screen_resolution=screen_resolution,
            )
        desk_start = mapper.screen_to_desk(*start)
        desk_end = mapper.screen_to_desk(*end)
        obs, _ = self.env.reset(start_pos=desk_start, target_pos=desk_end)
        trajectory = [(start[0], start[1], 0.0)]
        time_offset = 0.0
        done = False
        step = 0
        while not done and step < max_steps:
            if self.policy is not None:
                action = self.policy.predict(np.expand_dims(obs, 0))[0]
                if ou_sigma > 0.0:
                    ou_state += ou_theta * (0.0 - ou_state) * dt + ou_sigma * np.sqrt(dt) * ou_rng.standard_normal(2)
                    action = np.clip(action + ou_state, -1.0, 1.0).astype(np.float32)
            else:
                action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            step += 1
            time_offset += self.env.dt
            screen_pos = info["screen_pos"]
            trajectory.append((screen_pos[0], screen_pos[1], round(time_offset, 6)))
        return trajectory

    def generate_batch(self, movements, **kwargs):
        return [self.generate(start=m[0], end=m[1], **kwargs) for m in movements]

    def generate_to_numpy(self, start, end, **kwargs) -> np.ndarray:
        return np.array(self.generate(start, end, **kwargs), dtype=np.float64)
