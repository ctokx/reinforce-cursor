#!/usr/bin/env python3

import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Live MuJoCo 3D viewer for trained BMDS policy"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .d3 model file")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of movements to demonstrate (default: 8)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (0.5=slow, 2.0=fast)")
    parser.add_argument("--untrained", action="store_true",
                        help="Use untrained (random) policy for comparison")
    parser.add_argument("--pause-between", type=float, default=1.0,
                        help="Pause seconds between movements (default: 1.0)")
    args = parser.parse_args()

    import mujoco
    import mujoco.viewer
    from bmds.config import MODELS_DIR, DESK_X_RANGE, DESK_Y_RANGE
    from bmds.env.mouse_reach_env import MouseReachEnv, _DESK_MODEL_PATH
    from bmds.env.sim2screen import Sim2ScreenMapper
    from bmds.training.model_loader import (
        load_policy,
        infer_algorithm_from_model_path,
    )

    print("=" * 60)
    print("BMDS — Live MuJoCo 3D Viewer")
    print("=" * 60)


    if args.untrained:
        policy = None
        print("Mode: UNTRAINED (random actions)")
    else:
        model_path = args.model or str(MODELS_DIR / "bmds_cql_policy.d3")
        algo = infer_algorithm_from_model_path(model_path, default="cql")
        print(f"Loading model: {model_path}")
        policy = load_policy(model_path, algorithm=algo, use_gpu=False)
        print("Policy loaded successfully!")


    env = MouseReachEnv(mode="standalone")
    mapper = Sim2ScreenMapper()


    test_movements = [
        ((100, 100), (800, 500)),
        ((960, 100), (200, 800)),
        ((100, 540), (1800, 540)),
        ((960, 50), (960, 1000)),
        ((200, 200), (400, 300)),
        ((1500, 800), (300, 200)),
        ((500, 500), (1400, 500)),
        ((960, 540), (1600, 200)),
    ]


    if args.n > len(test_movements):
        rng = np.random.default_rng(42)
        while len(test_movements) < args.n:
            start = (rng.integers(50, 1870), rng.integers(50, 1030))
            end = (rng.integers(50, 1870), rng.integers(50, 1030))
            test_movements.append((start, end))
    test_movements = test_movements[:args.n]

    print(f"\nWill demonstrate {len(test_movements)} movements")
    print(f"Playback speed: {args.speed}x")
    print(f"\nOpening MuJoCo viewer window...")
    print("(Close the window or press Ctrl+C to exit)\n")


    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45
        viewer.cam.distance = 0.6
        viewer.cam.lookat[:] = [0.15, 0.125, 0.76]

        for i, (start_px, end_px) in enumerate(test_movements):
            if not viewer.is_running():
                break


            desk_start = mapper.screen_to_desk(*start_px)
            desk_end = mapper.screen_to_desk(*end_px)


            obs, _ = env.reset(start_pos=desk_start, target_pos=desk_end)
            viewer.sync()

            dist_px = np.sqrt(
                (end_px[0] - start_px[0]) ** 2 + (end_px[1] - start_px[1]) ** 2
            )
            print(f"Movement {i+1}/{len(test_movements)}: "
                  f"({start_px[0]},{start_px[1]}) -> ({end_px[0]},{end_px[1]})  "
                  f"[{dist_px:.0f}px]")


            t_pause = time.time()
            while time.time() - t_pause < args.pause_between and viewer.is_running():
                viewer.sync()
                time.sleep(0.016)


            done = False
            step = 0
            t_start = time.time()

            while not done and step < 500 and viewer.is_running():
                if policy is not None:
                    action = policy.predict(np.expand_dims(obs, 0))[0]
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1


                viewer.sync()
                sleep_time = env.dt / args.speed
                time.sleep(max(0, sleep_time))

            elapsed = time.time() - t_start
            reach_err = info.get("reach_err", 0)
            reached = "REACHED" if terminated else "TIMEOUT"
            print(f"  {reached} | {step} steps | {elapsed:.2f}s real | "
                  f"err={reach_err*1000:.1f}mm")

        print(f"\nDone! Viewer closing.")
    env.close()


if __name__ == "__main__":
    main()
