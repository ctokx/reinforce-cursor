#!/usr/bin/env python3
"""
Usage:
    python scripts/12_visualize_postprocessed.py
    python scripts/12_visualize_postprocessed.py --n 6 --seed 7 --speed 1.5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bmds.synthesizer import BMDSSynthesizer
from bmds.data.trajectory_db import TrajectoryDatabase
from bmds.config import MODELS_DIR

def _add_lateral_deviation(traj: np.ndarray, frac: float,
                           rng: np.random.Generator) -> np.ndarray:
    if len(traj) < 4 or frac == 0.0:
        return traj
    result = traj.copy()
    n = len(result)
    dx = result[-1, 0] - result[0, 0]
    dy = result[-1, 1] - result[0, 1]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if dist < 1e-6:
        return traj
    perp = np.array([-dy, dx]) / dist
    side = rng.choice([-1, 1])
    t_vals = np.linspace(0, 1, n)
    deviation = side * frac * dist * np.sin(np.pi * t_vals)
    result[:, 0] += perp[0] * deviation
    result[:, 1] += perp[1] * deviation
    return result

def _truncate_oscillation(traj: np.ndarray,
                          threshold_px: float = 45.0) -> np.ndarray:
    if len(traj) < 8:
        return traj
    target = traj[-1, :2].copy()
    distances = np.linalg.norm(traj[:, :2] - target, axis=1)
    min_idx = max(int(len(traj) * 0.15), 4)
    close_mask = distances[min_idx:] < threshold_px
    if not np.any(close_mask):
        return traj
    first_close_idx = min_idx + int(np.argmax(close_mask))
    truncated = traj[:first_close_idx + 1].copy()
    if len(truncated) >= 2:
        dt_cur = max(float(truncated[-1, 2] - truncated[-2, 2]), 0.005)
        v_cur = (truncated[-1, :2] - truncated[-2, :2]) / dt_cur
        speed_cur = float(np.linalg.norm(v_cur))
    else:
        speed_cur = 0.0
    if speed_cur > 5.0:
        n_decel = 8
        dt_decel = 0.012
        direction = v_cur / speed_cur
        t_last = float(truncated[-1, 2])
        pos_last = truncated[-1, :2].copy()
        decel_rows = []
        cum_pos = pos_last.copy()
        for k in range(1, n_decel + 1):
            frac = k / n_decel
            speed_k = speed_cur * 0.5 * (1.0 + np.cos(np.pi * frac))
            cum_pos = cum_pos + direction * speed_k * dt_decel
            decel_rows.append([cum_pos[0], cum_pos[1], t_last + k * dt_decel])
        truncated = np.vstack([truncated, np.array(decel_rows)])
    return truncated

def postprocess(traj: np.ndarray, rng: np.random.Generator,
                lateral_frac: float = None,
                target_duration: float = None) -> np.ndarray:
    if traj is None or len(traj) < 4:
        return traj
    result = traj.astype(float).copy()
    result = _truncate_oscillation(result, threshold_px=45.0)
    if len(result) < 4:
        return traj
    if len(result) >= 8:
        pin = min(3, len(result) // 4)
        fp = result[:pin, :2].copy()
        lp = result[-pin:, :2].copy()
        result[:, 0] = gaussian_filter1d(result[:, 0], sigma=2.5, mode='nearest')
        result[:, 1] = gaussian_filter1d(result[:, 1], sigma=2.5, mode='nearest')
        result[:pin, :2] = fp
        result[-pin:, :2] = lp
    if lateral_frac is None:
        lateral_frac = float(rng.uniform(0.10, 0.28))
    result = _add_lateral_deviation(result, lateral_frac, rng)
    if target_duration is None:
        target_duration = float(np.clip(rng.normal(2.2, 0.7), 0.8, 4.5))
    current_dur = float(result[-1, 2] - result[0, 2])
    remaining = max(0.0, target_duration - current_dur)
    start_pause = float(np.clip(remaining * 0.45 + rng.uniform(0.03, 0.07), 0.04, 0.55))
    end_pause   = float(np.clip(remaining * 0.55 + rng.uniform(0.05, 0.12), 0.06, 0.55))
    tremor_amp = 0.28

    n_start = max(int(start_pause / 0.01) + 1, 3)
    t0 = float(result[0, 2])
    t_sp = np.linspace(t0 - start_pause, t0 - 0.005, n_start)
    js = rng.standard_normal((n_start, 2)) * tremor_amp
    js[:3] = 0.0
    js[-3:] = 0.0
    start_seg = np.column_stack([
        np.full(n_start, result[0, 0]) + js[:, 0],
        np.full(n_start, result[0, 1]) + js[:, 1],
        t_sp])
    result = np.vstack([start_seg, result])

    n_end = max(int(end_pause / 0.01) + 1, 3)
    t_end = float(result[-1, 2])
    t_ep = np.linspace(t_end + 0.005, t_end + end_pause, n_end)
    je = rng.standard_normal((n_end, 2)) * tremor_amp
    je[:3] = 0.0
    je[-3:] = 0.0
    end_seg = np.column_stack([
        np.full(n_end, result[-1, 0]) + je[:, 0],
        np.full(n_end, result[-1, 1]) + je[:, 1],
        t_ep])
    result = np.vstack([result, end_seg])
    return result

def main():
    parser = argparse.ArgumentParser(description="Visualize postprocessed CQL trajectories")
    parser.add_argument("--n", type=int, default=5, help="Number of movements (default 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default 1.0)")
    parser.add_argument("--fps", type=int, default=30, help="GIF fps (default 30)")
    parser.add_argument("--out", type=str,
                        default="output/visualizations/postprocessed_demo.gif")
    parser.add_argument("--show-human", action="store_true",
                        help="Also show a matched human trajectory alongside")
    args = parser.parse_args()

    RES = (1920, 1080)
    rng = np.random.default_rng(args.seed)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading IQL policy...")
    synth = BMDSSynthesizer.load(screen_resolution=RES)

    movements = [
        ((200, 200),  (900, 600)),
        ((1700, 800), (600, 300)),
        ((960, 100),  (300, 800)),
        ((150, 540),  (1750, 540)),
        ((500, 900),  (1400, 200)),
        ((800, 400),  (1200, 700)),
    ]
    movements = movements[:args.n]

    print("Generating postprocessed CQL trajectories...")
    trajs = []
    for i, (start, end) in enumerate(movements):
        raw = np.array(synth.generate(start=start, end=end, noise_seed=i))
        proc = postprocess(raw, rng)
        trajs.append(proc)
        dur = proc[-1, 2] - proc[0, 2]
        print(f"  [{i+1}] {start} -> {end}: {len(proc)} pts, {dur:.2f}s")

    if args.show_human:
        print("Loading human trajectories for comparison...")
        db_path = PROJECT_ROOT / "data" / "processed" / "trajectory_db.h5"
        db = TrajectoryDatabase(db_path)

    COLS = 2 if args.n > 3 else 1
    ROWS = int(np.ceil(args.n / COLS))
    fig_w = 7 * COLS
    fig_h = 4 * ROWS
    fig, axes = plt.subplots(ROWS, COLS, figsize=(fig_w, fig_h),
                             facecolor="#0f0f16")
    if args.n == 1:
        axes = np.array([[axes]])
    elif COLS == 1:
        axes = axes.reshape(-1, 1)
    elif ROWS == 1:
        axes = axes.reshape(1, -1)

    ax_list = [axes[r, c] for r in range(ROWS) for c in range(COLS)
               if r * COLS + c < args.n]

    for ax in ax_list:
        ax.set_facecolor("#0f0f16")
        ax.set_xlim(0, RES[0])
        ax.set_ylim(RES[1], 0)
        ax.axis("off")

    for r in range(ROWS):
        for c in range(COLS):
            if r * COLS + c >= args.n:
                axes[r, c].set_visible(False)

    speed_arrays = []
    for traj in trajs:
        if len(traj) > 1:
            dx = np.diff(traj[:, 0])
            dy = np.diff(traj[:, 1])
            dt = np.maximum(np.diff(traj[:, 2]), 1e-6)
            speed_arrays.append(np.sqrt(dx**2 + dy**2) / dt)
        else:
            speed_arrays.append(np.array([0.0]))

    global_max_speed = max(np.max(s) for s in speed_arrays)

    def speed_color(sp):
        t = min(sp / max(global_max_speed, 1.0), 1.0)
        if t < 0.5:
            r = 0.24 + t * (0.24 - 0.24)
            g = 0.47 + t * (0.86 - 0.47)
            b = 1.00 + t * (0.39 - 1.00)
        else:
            t2 = (t - 0.5) * 2
            r = 0.24 + t2 * (1.00 - 0.24)
            g = 0.86 + t2 * (0.31 - 0.86)
            b = 0.39 + t2 * (0.24 - 0.39)
        return (max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b)))

    TRAIL = 60
    line_collections = []
    cursor_circles = []
    start_markers = []
    end_markers = []
    time_texts = []

    for idx, (ax, traj) in enumerate(zip(ax_list, trajs)):
        sx, sy = traj[0, 0], traj[0, 1]
        ex, ey = traj[-1, 0], traj[-1, 1]
        s_circ = Circle((sx, sy), 12, color="#4cde8f", fill=False, lw=2, zorder=5)
        e_circ = Circle((ex, ey), 15, color="#ff5050", fill=False, lw=2, zorder=5)
        ax.add_patch(s_circ)
        ax.add_patch(e_circ)
        ax.plot([ex - 8, ex + 8], [ey, ey], color="#ff5050", lw=1.5, zorder=5)
        ax.plot([ex, ex], [ey - 8, ey + 8], color="#ff5050", lw=1.5, zorder=5)

        cur_circ = Circle((sx, sy), 8, color="white", zorder=10)
        ax.add_patch(cur_circ)
        cursor_circles.append(cur_circ)

        tt = ax.text(20, 30, "", color="#aaaacc", fontsize=9,
                     fontfamily="monospace", zorder=15)
        time_texts.append(tt)

        start_markers.append((sx, sy))
        end_markers.append((ex, ey))

    trail_lines_per_ax = [[] for _ in ax_list]
    target_fps = args.fps
    dt_frame = args.speed / target_fps

    max_durations = [t[-1, 2] - t[0, 2] for t in trajs]
    max_total = max(max_durations) + 0.5

    n_frames_total = int(max_total / dt_frame) + 1

    print(f"Rendering {n_frames_total} frames at {target_fps} fps...")

    def animate(frame_i):
        t_sim = frame_i * dt_frame
        for idx, (ax, traj) in enumerate(zip(ax_list, trajs)):
            for ln in trail_lines_per_ax[idx]:
                ln.remove()
            trail_lines_per_ax[idx].clear()

            t0 = float(traj[0, 2])
            t_abs = t0 + t_sim
            fi = int(np.searchsorted(traj[:, 2], t_abs))
            fi = min(fi, len(traj) - 1)

            trail_start = max(0, fi - TRAIL)
            spd = speed_arrays[idx]
            for j in range(trail_start, min(fi, len(traj) - 1)):
                age = (fi - j) / TRAIL
                alpha = max(0.05, 1.0 - age)
                sp = spd[j] if j < len(spd) else 0
                col = speed_color(sp)
                ln, = ax.plot([traj[j, 0], traj[j + 1, 0]],
                               [traj[j, 1], traj[j + 1, 1]],
                               color=col, alpha=alpha,
                               lw=max(1.0, 2.5 * alpha), zorder=3)
                trail_lines_per_ax[idx].append(ln)

            cx, cy = traj[fi, 0], traj[fi, 1]
            cursor_circles[idx].center = (cx, cy)

            rel_t = float(traj[fi, 2]) - t0
            time_texts[idx].set_text(f"t = {rel_t:.2f}s")

        return cursor_circles + time_texts + [
            ln for lines in trail_lines_per_ax for ln in lines
        ]

    fig.suptitle("BMDS — Postprocessed CQL Agent Trajectories\n"
                 "(OCSVM: 40% | DELBOT: 1% | GradBoost: 0%)",
                 color="#ccccdd", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames_total,
        interval=int(1000 / target_fps), blit=False
    )

    print(f"Saving GIF -> {out_path} ...")
    writer = animation.PillowWriter(fps=target_fps)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    print(f"Done! Saved: {out_path}")
    print(f"  {n_frames_total} frames, {target_fps} fps, {n_frames_total/target_fps:.1f}s")

if __name__ == "__main__":
    main()
