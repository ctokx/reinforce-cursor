#!/usr/bin/env python3

import sys
import os
import time
import argparse
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def print_banner(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_system_info():
    import torch
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                     ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                     ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                     ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                     ('sullAvailExtendedVirtual', ctypes.c_ulonglong)]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(stat)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))

    print(f"  RAM: {stat.ullTotalPhys / 1024**3:.1f} GB total, "
          f"{stat.ullAvailPhys / 1024**3:.1f} GB free ({stat.dwMemoryLoad}% used)")
    print(f"  CPU: {os.cpu_count()} cores")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        print("  GPU: None (will use CPU)")
    print()


def phase1_download_and_parse():
    from bmds.data.download import download_balabit, get_session_files, BALABIT_DIR
    from bmds.data.parser import BalabitParser
    from bmds.data.features import TrajectoryFeatureExtractor
    from bmds.data.trajectory_db import TrajectoryDatabase
    from bmds.data.statistics import compute_statistics, save_statistics

    print_banner("PHASE 1: Download & Parse Balabit Mouse Dynamics Dataset")

    dataset_dir = download_balabit()
    sessions = get_session_files(dataset_dir)
    print(f"Found {len(sessions)} session files across users:")
    users = {}
    for uid, _ in sessions:
        users[uid] = users.get(uid, 0) + 1
    for uid, count in sorted(users.items()):
        print(f"  {uid}: {count} sessions")

    print(f"\nParsing sessions into trajectory segments...")
    parser = BalabitParser()
    trajectories = parser.parse_all_sessions(sessions)

    if not trajectories:
        print("ERROR: No trajectories extracted!")
        sys.exit(1)

    print(f"\nExtracting kinematic features...")
    extractor = TrajectoryFeatureExtractor()
    features = extractor.extract_batch(trajectories)

    print(f"\nBuilding HDF5 trajectory database...")
    db = TrajectoryDatabase()
    db.build(trajectories, features)
    print(f"Database: {len(db)} trajectories")

    print(f"\nComputing human motion statistics...")
    stats = compute_statistics(db)
    save_statistics(stats)

    print(f"\nSaving sample trajectory visualizations...")
    vis_dir = PROJECT_ROOT / "output" / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Sample Human Mouse Trajectories (Balabit Dataset)", fontsize=16)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(db), size=min(6, len(db)), replace=False)

    for idx, ax in zip(sample_indices, axes.flat):
        traj = db.get_trajectory(int(idx))
        pts = traj.points
        ax.plot(pts[:, 0], pts[:, 1], "b-", linewidth=1.2, alpha=0.8)
        ax.plot(pts[0, 0], pts[0, 1], "go", markersize=8)
        ax.plot(pts[-1, 0], pts[-1, 1], "rs", markersize=8)
        ax.set_title(f"User {traj.user_id} | {traj.distance:.0f}px | {traj.duration:.2f}s")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(vis_dir / "01_sample_human_trajectories.png", dpi=120)
    plt.close()
    print(f"  Saved: {vis_dir / '01_sample_human_trajectories.png'}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Human Motion Statistics (Balabit Dataset)", fontsize=14)

    peak_speeds = db.get_feature("peak_speed")
    valid = peak_speeds[np.isfinite(peak_speeds) & (peak_speeds > 0)]
    axes[0].hist(valid, bins=80, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].axvline(stats.speed_p95, color="red", linestyle="--", label=f"P95={stats.speed_p95:.0f}")
    axes[0].set_xlabel("Peak Speed (px/s)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Speed Distribution")
    axes[0].legend()

    efficiencies = db.get_feature("path_efficiency")
    valid_eff = efficiencies[(efficiencies > 0) & (efficiencies <= 1)]
    axes[1].hist(valid_eff, bins=50, color="coral", alpha=0.7, edgecolor="white")
    axes[1].axvline(stats.efficiency_mean, color="red", linestyle="--",
                    label=f"Mean={stats.efficiency_mean:.2f}")
    axes[1].set_xlabel("Path Efficiency")
    axes[1].set_title("Path Efficiency Distribution")
    axes[1].legend()

    durations = db.get_feature("duration")
    valid_dur = durations[(durations > 0) & (durations < 5)]
    axes[2].hist(valid_dur, bins=80, color="mediumpurple", alpha=0.7, edgecolor="white")
    axes[2].set_xlabel("Duration (s)")
    axes[2].set_title("Movement Duration Distribution")

    plt.tight_layout()
    plt.savefig(vis_dir / "02_human_motion_statistics.png", dpi=120)
    plt.close()
    print(f"  Saved: {vis_dir / '02_human_motion_statistics.png'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    template = np.array(stats.velocity_profile_template)
    ax.plot(np.linspace(0, 1, len(template)), template, "b-", linewidth=2)
    ax.fill_between(np.linspace(0, 1, len(template)), 0, template, alpha=0.2)
    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("Normalized Speed")
    ax.set_title("Average Human Velocity Profile Template (Bell-Shaped)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(vis_dir / "03_velocity_profile_template.png", dpi=120)
    plt.close()
    print(f"  Saved: {vis_dir / '03_velocity_profile_template.png'}")

    return db, stats


def phase2_build_rl_dataset(db, stats, max_trajectories=None):
    from bmds.env.mouse_reach_env import MouseReachEnv
    from bmds.env.sim2screen import Sim2ScreenMapper
    from bmds.reward.biomechanical_reward import BiomechanicalReward
    from bmds.training.dataset_builder import DatasetBuilder

    print_banner("PHASE 2: Build Offline RL Dataset")

    env = MouseReachEnv()
    mapper = Sim2ScreenMapper()
    reward_fn = BiomechanicalReward(human_stats=stats, mapper=mapper)

    builder = DatasetBuilder(env=env, mapper=mapper, reward_fn=reward_fn)

    n_trajs = max_trajectories or len(db)
    print(f"Processing {n_trajs} trajectories through MuJoCo physics...")

    dataset = builder.build_dataset(db, max_trajectories=max_trajectories)
    builder.save_dataset(dataset)

    env.close()

    print(f"\nDataset summary:")
    print(f"  Transitions: {dataset['observations'].shape[0]}")
    print(f"  Observation dim: {dataset['observations'].shape[1]}")
    print(f"  Action dim: {dataset['actions'].shape[1]}")
    print(f"  Reward range: [{dataset['rewards'].min():.2f}, {dataset['rewards'].max():.2f}]")
    print(f"  Terminal rate: {dataset['terminals'].mean():.3f}")

    return dataset


def phase3_train(n_steps, n_steps_per_epoch, algorithm="cql", alpha=5.0):
    from bmds.training.train_cql import train_cql, train_iql

    print_banner(f"PHASE 3: Train {algorithm.upper()} Policy on GPU")

    est_time = n_steps / 28
    print(f"Configuration:")
    print(f"  Algorithm: {algorithm.upper()}")
    print(f"  Total gradient steps: {n_steps:,}")
    print(f"  Steps/epoch (checkpoint interval): {n_steps_per_epoch:,}")
    print(f"  Estimated time: ~{est_time/60:.0f} minutes on RTX 4060")
    if algorithm == "cql":
        print(f"  CQL conservatism (alpha): {alpha}")
    print()

    tb_dir = str(PROJECT_ROOT / "output" / "tensorboard")
    print(f"TensorBoard: tensorboard --logdir {tb_dir}\n")

    start_time = time.time()

    if algorithm == "cql":
        model_path = train_cql(
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            alpha=alpha,
            use_gpu=True,
            verbose=True,
            tensorboard_dir=tb_dir,
        )
    elif algorithm == "iql":
        model_path = train_iql(
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            use_gpu=True,
            verbose=True,
            tensorboard_dir=tb_dir,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'cql' or 'iql'.")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Steps/second: {n_steps/elapsed:.0f}")
    print(f"Model saved: {model_path}")

    return model_path


def phase4_evaluate_and_visualize(model_path, stats):
    import d3rlpy
    from bmds.env.mouse_reach_env import MouseReachEnv
    from bmds.env.sim2screen import Sim2ScreenMapper
    from bmds.utils.kinematics import compute_kinematics, normalize_speed_profile

    print_banner("PHASE 4: Evaluate Trained Policy & Visualize")

    vis_dir = PROJECT_ROOT / "output" / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}")
    from bmds.training.model_loader import load_policy, infer_algorithm_from_model_path
    algo_name = infer_algorithm_from_model_path(model_path, default="cql")
    model = load_policy(str(model_path), algorithm=algo_name, use_gpu=False)

    env = MouseReachEnv()
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

    generated_trajectories = []
    print(f"\nGenerating {len(test_movements)} test trajectories...")

    for i, (start, end) in enumerate(test_movements):
        desk_start = mapper.screen_to_desk(*start)
        desk_end = mapper.screen_to_desk(*end)

        obs, _ = env.reset(start_pos=desk_start, target_pos=desk_end)
        trajectory = [(start[0], start[1], 0.0)]
        done = False
        step = 0

        while not done and step < 500:
            action = model.predict(np.expand_dims(obs, 0))[0]
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            sp = info["screen_pos"]
            trajectory.append((sp[0], sp[1], step * env.dt))

        traj_arr = np.array(trajectory)
        generated_trajectories.append(traj_arr)
        dist = np.linalg.norm(traj_arr[-1, :2] - traj_arr[0, :2])
        print(f"  {start} -> {end}: {len(trajectory)} pts, "
              f"{trajectory[-1][2]:.2f}s, {dist:.0f}px, "
              f"reached={'yes' if terminated else 'no'}")

    env.close()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Generated Synthetic Mouse Trajectories (Trained CQL Policy)", fontsize=16)

    for idx, (ax, traj_arr) in enumerate(zip(axes.flat, generated_trajectories)):
        x, y = traj_arr[:, 0], traj_arr[:, 1]
        if len(traj_arr) >= 4:
            kin = compute_kinematics(traj_arr)
            colors = kin.speed / max(kin.peak_speed, 1)
            for j in range(len(x) - 1):
                c = plt.cm.plasma(colors[j] if j < len(colors) else 0)
                ax.plot(x[j:j+2], y[j:j+2], color=c, linewidth=2)
        else:
            ax.plot(x, y, "b-", linewidth=1.5)

        ax.plot(x[0], y[0], "go", markersize=10)
        ax.plot(x[-1], y[-1], "rs", markersize=10)
        start, end = test_movements[idx]
        ax.set_title(f"{start}->{end}", fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(vis_dir / "04_generated_trajectories.png", dpi=120)
    plt.close()
    print(f"\nSaved: {vis_dir / '04_generated_trajectories.png'}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("Velocity Profiles: Generated (blue) vs Human Template (red dashed)", fontsize=14)

    human_template = np.array(stats.velocity_profile_template)

    for ax, traj_arr in zip(axes.flat, generated_trajectories):
        if len(traj_arr) >= 4:
            kin = compute_kinematics(traj_arr)
            norm_speed = normalize_speed_profile(kin.speed)
            t = np.linspace(0, 1, len(norm_speed))
            ax.plot(t, norm_speed, "b-", linewidth=2, label="Generated")
            ax.fill_between(t, 0, norm_speed, alpha=0.15, color="blue")

        t_h = np.linspace(0, 1, len(human_template))
        ax.plot(t_h, human_template, "r--", linewidth=1.5, alpha=0.7, label="Human avg")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(vis_dir / "05_velocity_profile_comparison.png", dpi=120)
    plt.close()
    print(f"Saved: {vis_dir / '05_velocity_profile_comparison.png'}")

    all_peak_speeds = []
    for traj_arr in generated_trajectories:
        if len(traj_arr) >= 4:
            kin = compute_kinematics(traj_arr)
            all_peak_speeds.append(kin.peak_speed)

    fig, ax = plt.subplots(figsize=(10, 5))
    if all_peak_speeds:
        ax.hist(all_peak_speeds, bins=20, alpha=0.6, color="blue", label="Generated", density=True)
    human_speeds = np.random.normal(stats.speed_mean, stats.speed_std, 1000)
    human_speeds = human_speeds[human_speeds > 0]
    ax.hist(human_speeds, bins=50, alpha=0.4, color="red", label="Human (sampled)", density=True)
    ax.set_xlabel("Peak Speed (px/s)")
    ax.set_ylabel("Density")
    ax.set_title("Peak Speed Distribution: Generated vs Human")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(vis_dir / "06_speed_distribution_comparison.png", dpi=120)
    plt.close()
    print(f"Saved: {vis_dir / '06_speed_distribution_comparison.png'}")

    print(f"\nAll visualizations saved to: {vis_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="phantom-hand Full Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training.py                         # Default: ~30 min training
  python run_training.py --epochs 50             # Shorter training
  python run_training.py --algorithm iql         # Use IQL instead of CQL
  python run_training.py --max-trajectories 500  # Quick test with less data
        """,
    )
    parser.add_argument("--steps", type=int, default=100_000,
                        help="Total gradient steps (default: 100000)")
    parser.add_argument("--steps-per-epoch", type=int, default=10_000,
                        help="Steps per checkpoint epoch (default: 10000)")
    parser.add_argument("--algorithm", choices=["cql", "iql", "bc"], default="cql",
                        help="RL algorithm (default: cql)")
    parser.add_argument("--alpha", type=float, default=5.0,
                        help="CQL conservatism weight (default: 5.0)")
    parser.add_argument("--max-trajectories", type=int, default=None,
                        help="Limit # of trajectories for dataset building")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (use existing)")
    parser.add_argument("--skip-dataset-build", action="store_true",
                        help="Skip RL dataset build (use existing offline_rl_dataset.npz)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training (just evaluate existing model)")
    args = parser.parse_args()

    print_banner("phantom-hand — Biomechanical Mouse Dynamics Synthesizer")
    print_system_info()

    total_start = time.time()

    if not args.skip_download:
        db, stats = phase1_download_and_parse()
    else:
        from bmds.data.trajectory_db import TrajectoryDatabase
        from bmds.data.statistics import load_statistics
        db = TrajectoryDatabase()
        stats = load_statistics()
        print(f"Using existing data: {len(db)} trajectories")

    skip_build = args.skip_training or args.skip_dataset_build
    if not skip_build:
        phase2_build_rl_dataset(db, stats, max_trajectories=args.max_trajectories)

    if not args.skip_training:
        model_path = phase3_train(
            n_steps=args.steps,
            n_steps_per_epoch=args.steps_per_epoch,
            algorithm=args.algorithm,
            alpha=args.alpha,
        )
    else:
        from bmds.config import MODELS_DIR
        model_path = MODELS_DIR / f"bmds_{args.algorithm}_policy.d3"

    phase4_evaluate_and_visualize(model_path, stats)

    total_elapsed = time.time() - total_start
    print_banner("PIPELINE COMPLETE")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Model: {model_path}")
    print(f"Visualizations: {PROJECT_ROOT / 'output' / 'visualizations'}")
    print(f"\nTo generate trajectories:")
    print(f"  python scripts/06_generate_trajectories.py --start 100 100 --end 800 500 --plot")
    print(f"\nTo view in 3D MuJoCo sim:")
    print(f"  python scripts/07_live_mujoco_viewer.py")
    print(f"\nTo view 2D screen animation:")
    print(f"  python scripts/08_live_screen_animation.py")


if __name__ == "__main__":
    main()
