import numpy as np
from pathlib import Path
from typing import Dict, Optional
from scipy import stats as sp_stats

from bmds.config import MODELS_DIR
from bmds.data.statistics import HumanMotionStatistics
from bmds.env.mouse_reach_env import MouseReachEnv
from bmds.env.sim2screen import Sim2ScreenMapper
from bmds.utils.kinematics import compute_kinematics, count_submovements
from bmds.utils.fitts import fit_fitts_law


class PolicyEvaluator:

    def __init__(self, env: MouseReachEnv, mapper: Sim2ScreenMapper,
                 human_stats: HumanMotionStatistics):
        self.env = env
        self.mapper = mapper
        self.stats = human_stats

    def evaluate(self, policy, n_episodes: int = 200,
                 verbose: bool = True) -> Dict:
        from tqdm import tqdm

        all_peak_speeds = []
        all_durations = []
        all_distances = []
        all_efficiencies = []
        all_submovements = []
        all_max_accels = []
        all_mean_jerks = []
        successes = 0

        iterator = tqdm(range(n_episodes), desc="Evaluating") if verbose else range(n_episodes)

        for ep in iterator:
            obs, info = self.env.reset()
            done = False
            trajectory = []
            start_screen = self.mapper.desk_to_screen(*info["start_pos"])
            trajectory.append((start_screen[0], start_screen[1], 0.0))

            while not done:
                action = policy.predict(np.expand_dims(obs, 0))[0]
                obs, reward, terminated, truncated, step_info = self.env.step(action)
                done = terminated or truncated

                screen_pos = step_info["screen_pos"]
                t = len(trajectory) * self.env.dt
                trajectory.append((screen_pos[0], screen_pos[1], t))

            if terminated:
                successes += 1

            traj_arr = np.array(trajectory)
            if len(traj_arr) >= 4:
                kin = compute_kinematics(traj_arr)
                all_peak_speeds.append(kin.peak_speed)
                all_durations.append(kin.duration)
                all_distances.append(kin.straight_line_distance)
                all_efficiencies.append(kin.path_efficiency)
                all_submovements.append(count_submovements(kin.speed))
                all_max_accels.append(kin.max_acceleration)
                all_mean_jerks.append(kin.mean_jerk)

        all_peak_speeds = np.array(all_peak_speeds)
        all_durations = np.array(all_durations)
        all_distances = np.array(all_distances)
        all_efficiencies = np.array(all_efficiencies)

        results = {
            "success_rate": successes / n_episodes,
            "n_episodes": n_episodes,
        }


        if len(all_peak_speeds) > 10 and self.stats.speed_mean > 0:

            human_speeds = np.random.normal(
                self.stats.speed_mean, self.stats.speed_std, size=1000
            )
            ks_stat, ks_pvalue = sp_stats.ks_2samp(all_peak_speeds, human_speeds)
            results["speed_ks_statistic"] = float(ks_stat)
            results["speed_ks_pvalue"] = float(ks_pvalue)

        results["speed_mean"] = float(np.mean(all_peak_speeds)) if len(all_peak_speeds) > 0 else 0.0
        results["speed_std"] = float(np.std(all_peak_speeds)) if len(all_peak_speeds) > 0 else 0.0


        if len(all_max_accels) > 0 and self.stats.accel_max > 0:
            within_bounds = np.sum(np.array(all_max_accels) <= self.stats.accel_max * 1.1)
            results["accel_within_bounds_pct"] = float(within_bounds / len(all_max_accels))


        results["efficiency_mean"] = float(np.mean(all_efficiencies)) if len(all_efficiencies) > 0 else 0.0
        results["efficiency_std"] = float(np.std(all_efficiencies)) if len(all_efficiencies) > 0 else 0.0


        if len(all_distances) > 20:
            target_widths = np.full_like(all_distances, 10.0)
            a, b, r2 = fit_fitts_law(all_distances, target_widths, all_durations)
            results["fitts_a"] = a
            results["fitts_b"] = b
            results["fitts_r_squared"] = r2


        results["duration_mean"] = float(np.mean(all_durations)) if len(all_durations) > 0 else 0.0

        if verbose:
            print("\n=== Evaluation Results ===")
            print(f"Success rate: {results['success_rate']:.1%}")
            print(f"Speed: mean={results.get('speed_mean', 0):.0f} px/s "
                  f"(human: {self.stats.speed_mean:.0f})")
            if "speed_ks_pvalue" in results:
                print(f"Speed KS test: stat={results['speed_ks_statistic']:.3f}, "
                      f"p={results['speed_ks_pvalue']:.4f}")
            print(f"Efficiency: {results.get('efficiency_mean', 0):.3f} "
                  f"(human: {self.stats.efficiency_mean:.3f})")
            if "fitts_r_squared" in results:
                print(f"Fitts's law: R²={results['fitts_r_squared']:.3f} "
                      f"(human: {self.stats.fitts_r_squared:.3f})")
            if "accel_within_bounds_pct" in results:
                print(f"Accel within bounds: {results['accel_within_bounds_pct']:.1%}")

        return results
