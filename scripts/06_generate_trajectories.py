#!/usr/bin/env python3

import sys
import argparse
import json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
from bmds.synthesizer import BMDSSynthesizer


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic mouse trajectories")
    parser.add_argument("--model", type=str, default=None, help="Path to .d3 model file")
    parser.add_argument("--start", type=int, nargs=2, default=[100, 100],
                        help="Start pixel coordinates (x y)")
    parser.add_argument("--end", type=int, nargs=2, default=[800, 500],
                        help="End pixel coordinates (x y)")
    parser.add_argument("--n", type=int, default=1, help="Number of trajectories to generate")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 1080],
                        help="Screen resolution (width height)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: print to stdout)")
    parser.add_argument("--untrained", action="store_true",
                        help="Use untrained (random) policy for testing")
    parser.add_argument("--plot", action="store_true", help="Plot the trajectory")
    args = parser.parse_args()

    print("=" * 60)
    print("BMDS - Generate Trajectories")
    print("=" * 60)

    resolution = tuple(args.resolution)

    if args.untrained:
        synth = BMDSSynthesizer.load_untrained(screen_resolution=resolution)
        print("Using untrained (random) policy")
    else:
        synth = BMDSSynthesizer.load(model_path=args.model, screen_resolution=resolution)
        print("Using trained policy")

    start = tuple(args.start)
    end = tuple(args.end)
    print(f"Generating {args.n} trajectory(ies): {start} -> {end}")
    print(f"Screen resolution: {resolution}")

    all_trajectories = []
    for i in range(args.n):
        traj = synth.generate(start=start, end=end, noise_seed=i)
        all_trajectories.append(traj)

        n_pts = len(traj)
        duration = traj[-1][2] if traj else 0
        distance = np.sqrt((traj[-1][0] - traj[0][0])**2 +
                           (traj[-1][1] - traj[0][1])**2) if len(traj) > 1 else 0
        print(f"  Trajectory {i+1}: {n_pts} points, {duration:.3f}s, {distance:.0f}px")


    output_data = {
        "start": list(start),
        "end": list(end),
        "screen_resolution": list(resolution),
        "trajectories": [
            [[p[0], p[1], p[2]] for p in traj]
            for traj in all_trajectories
        ],
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to {args.output}")
    else:

        if all_trajectories:
            traj = all_trajectories[0]
            print(f"\nFirst trajectory ({len(traj)} points):")
            for pt in traj[:10]:
                print(f"  ({pt[0]:6d}, {pt[1]:6d}, {pt[2]:.4f})")
            if len(traj) > 10:
                print(f"  ... ({len(traj) - 10} more points)")


    if args.plot:
        from bmds.utils.visualization import plot_trajectory
        for i, traj in enumerate(all_trajectories):
            arr = np.array(traj)
            plot_trajectory(arr, title=f"Trajectory {i+1}")


if __name__ == "__main__":
    main()
