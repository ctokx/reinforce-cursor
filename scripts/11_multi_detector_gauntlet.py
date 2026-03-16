"""
Usage:
    python scripts/11_multi_detector_gauntlet.py --n-movements 100
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import OneClassSVM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bmds.data.trajectory_db import TrajectoryDatabase
from bmds.data.statistics import load_statistics
from bmds.synthesizer import BMDSSynthesizer
from bmds.utils.kinematics import (
    compute_kinematics, count_submovements, normalize_speed_profile,
)
from scripts import _script09_helpers as s09

OUTPUT_DIR = PROJECT_ROOT / "output" / "bot_detection_gauntlet"
NODE_HELPER = PROJECT_ROOT / "scripts" / "_delbot_classify.js"
SOURCE_ORDER = ["Human", "CQL Agent", "Linear Bot", "Bezier Bot"]
COLORS = {
    "Human": "#2ecc71",
    "CQL Agent": "#3498db",
    "Linear Bot": "#e74c3c",
    "Bezier Bot": "#f39c12",
}

FEATURE_NAMES = [
    "peak_speed", "mean_speed", "speed_std", "path_efficiency",
    "max_acceleration", "max_deceleration", "mean_jerk",
    "num_submovements", "mean_curvature", "duration",
    "speed_profile_correlation", "jerk_smoothness",
    "speed_profile_skewness",
    "curvature_entropy",
    "direction_reversal_near_target",
    "speed_cv",
    "angular_velocity_variance",
    "micro_pause_ratio",
]

def speed_profile_correlation(speed: np.ndarray, template: np.ndarray) -> float:
    profile = normalize_speed_profile(speed, n_samples=len(template))
    if np.std(profile) < 1e-9 or np.std(template) < 1e-9:
        return 0.0
    return float(np.corrcoef(profile, template)[0, 1])

def jerk_smoothness(jerk_magnitude: np.ndarray, duration: float,
                    peak_speed: float) -> float:
    if duration < 1e-9 or peak_speed < 1e-9 or len(jerk_magnitude) == 0:
        return -1.0
    mean_jerk_sq = float(np.mean(jerk_magnitude ** 2))
    dj = (duration ** 3 / peak_speed ** 2) * mean_jerk_sq
    return -float(np.log(max(dj, 1e-12)))

def speed_profile_skewness(speed: np.ndarray) -> float:
    profile = normalize_speed_profile(speed, n_samples=100)
    mu = float(np.mean(profile))
    sigma = float(np.std(profile))
    if sigma < 1e-9:
        return 0.0
    centered = (profile - mu) / sigma
    return float(np.mean(centered ** 3))

def curvature_entropy(curvature: np.ndarray, n_bins: int = 20) -> float:
    if len(curvature) == 0:
        return 0.0
    values = np.abs(curvature)
    if np.max(values) < 1e-9:
        return 0.0
    hist, _ = np.histogram(values, bins=n_bins, density=False)
    p = hist.astype(float) / max(float(np.sum(hist)), 1e-9)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    entropy = -np.sum(p * np.log(p))
    return float(entropy / np.log(n_bins))

def direction_reversal_count_near_target(points: np.ndarray,
                                         tail_fraction: float = 0.3) -> float:
    if len(points) < 6:
        return 0.0

    pos = points[:, :2]
    vel = np.diff(pos, axis=0)
    to_target = pos[-1] - pos[:-1]
    remaining_dist = np.linalg.norm(to_target, axis=1)
    max_remaining = float(np.max(remaining_dist)) if len(remaining_dist) > 0 else 0.0
    if max_remaining < 1e-9:
        return 0.0

    near_mask = remaining_dist <= (tail_fraction * max_remaining)
    if np.sum(near_mask) < 3:
        return 0.0

    denom = np.linalg.norm(to_target, axis=1) + 1e-9
    signed_progress = np.sum(vel * to_target, axis=1) / denom
    tail_progress = signed_progress[near_mask]
    signs = np.sign(tail_progress)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0.0
    reversals = np.sum(signs[:-1] * signs[1:] < 0)
    return float(reversals)

def speed_coefficient_of_variation(speed: np.ndarray) -> float:
    mean_speed = float(np.mean(speed))
    if mean_speed < 1e-9:
        return 0.0
    return float(np.std(speed) / mean_speed)

def angular_velocity_variance(points: np.ndarray, velocity: np.ndarray) -> float:
    if len(velocity) < 3 or len(points) < 5:
        return 0.0
    heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    if len(heading) < 3:
        return 0.0
    dtheta = np.diff(heading)
    dt = np.diff(points[1:, 2])
    if len(dt) != len(dtheta):
        n = min(len(dt), len(dtheta))
        dt = dt[:n]
        dtheta = dtheta[:n]
    dt = np.maximum(dt, 1e-3)
    omega = dtheta / dt
    if len(omega) == 0:
        return 0.0
    return float(np.var(omega))

def micro_pause_ratio(speed: np.ndarray, fraction_of_peak: float = 0.05) -> float:
    if len(speed) == 0:
        return 0.0
    peak = float(np.max(speed))
    if peak < 1e-9:
        return 1.0
    threshold = max(peak * fraction_of_peak, 1e-3)
    return float(np.mean(speed <= threshold))

def extract_features(points: np.ndarray, human_template: np.ndarray) -> dict:
    if points is None or len(points) < 4:
        return None

    try:
        kin = compute_kinematics(points)
    except Exception:
        return None

    speed = kin.speed
    curvature = kin.curvature
    n_sub = count_submovements(speed)

    return {
        "peak_speed": kin.peak_speed,
        "mean_speed": kin.mean_speed,
        "speed_std": float(np.std(speed)),
        "path_efficiency": kin.path_efficiency,
        "max_acceleration": kin.max_acceleration,
        "max_deceleration": kin.max_deceleration,
        "mean_jerk": kin.mean_jerk,
        "num_submovements": float(n_sub),
        "mean_curvature": float(np.mean(np.abs(curvature))) if len(curvature) > 0 else 0.0,
        "duration": kin.duration,
        "speed_profile_correlation": speed_profile_correlation(speed, human_template),
        "jerk_smoothness": jerk_smoothness(kin.jerk_magnitude, kin.duration, kin.peak_speed),
        "speed_profile_skewness": speed_profile_skewness(speed),
        "curvature_entropy": curvature_entropy(curvature),
        "direction_reversal_near_target": direction_reversal_count_near_target(points),
        "speed_cv": speed_coefficient_of_variation(speed),
        "angular_velocity_variance": angular_velocity_variance(points, kin.velocity),
        "micro_pause_ratio": micro_pause_ratio(speed),
    }

def feature_matrix(features: list) -> np.ndarray:
    X = np.array([[f[k] for k in FEATURE_NAMES] for f in features], dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def _add_lateral_deviation(points: np.ndarray, lateral_frac: float,
                           rng: np.random.Generator) -> np.ndarray:
    n = len(points)
    if n < 4 or lateral_frac <= 0:
        return points
    start_xy = points[0, :2]
    end_xy = points[-1, :2]
    dist = float(np.linalg.norm(end_xy - start_xy))
    if dist < 10.0:
        return points
    along = (end_xy - start_xy) / dist
    perp = np.array([-along[1], along[0]])
    progress = np.linspace(0.0, 1.0, n)
    direction = 1.0 if rng.uniform() < 0.5 else -1.0
    amplitude = direction * lateral_frac * dist
    lateral = amplitude * np.sin(np.pi * progress)
    result = points.copy()
    result[:, 0] += lateral * perp[0]
    result[:, 1] += lateral * perp[1]
    return result

def _add_micro_pauses(points: np.ndarray, n_pauses: int, pause_duration: float,
                      jitter_px: float, rng: np.random.Generator) -> np.ndarray:
    if n_pauses <= 0 or len(points) < 4:
        return points
    n = len(points)
    dt_pause = float(pause_duration)
    pts_per_pause = max(int(dt_pause / 0.01) + 1, 5)

    pause_fracs = [(i + 1) / (n_pauses + 1) for i in range(n_pauses)]
    pause_idxs = [max(1, min(int(f * n), n - 2)) for f in pause_fracs]

    segments = []
    prev = 0
    cumulative_dt = 0.0
    for pi in pause_idxs:
        seg = points[prev:pi + 1].copy()
        seg[:, 2] += cumulative_dt
        segments.append(seg)

        pos = points[pi, :2]
        t_start = float(points[pi, 2]) + cumulative_dt
        pt = np.linspace(t_start, t_start + dt_pause, pts_per_pause)
        jx = rng.standard_normal(pts_per_pause) * jitter_px
        jy = rng.standard_normal(pts_per_pause) * jitter_px
        jx[0] = jx[-1] = jy[0] = jy[-1] = 0.0
        segments.append(np.column_stack([pos[0] + jx, pos[1] + jy, pt]))

        cumulative_dt += dt_pause
        prev = pi

    seg = points[prev:].copy()
    seg[:, 2] += cumulative_dt
    segments.append(seg)
    return np.vstack(segments)

def _smooth_pause_ramp(points: np.ndarray, pause_idx: int,
                       ramp_steps: int, rng: np.random.Generator) -> np.ndarray:
    n = len(points)
    if n < 6 or ramp_steps < 2:
        return points, pause_idx

    i = max(0, min(pause_idx, n - 2))
    dt_pre = float(points[i, 2] - points[max(0, i - 1), 2])
    dt_pre = max(dt_pre, 0.001)
    v_pre = np.linalg.norm(points[i, :2] - points[max(0, i - 1), :2]) / dt_pre
    if v_pre < 1.0:
        return points, pause_idx

    dt_step = 0.01
    ramp_duration = ramp_steps * dt_step

    if i > 0:
        direction = points[i, :2] - points[i - 1, :2]
        dn = np.linalg.norm(direction)
        if dn > 1e-6:
            direction /= dn
        else:
            direction = np.array([1.0, 0.0])
    else:
        direction = np.array([1.0, 0.0])

    t_ramp = np.arange(1, ramp_steps + 1) * dt_step
    speed_ramp = v_pre * 0.5 * (1.0 + np.cos(np.pi * t_ramp / ramp_duration))
    dx_ramp = speed_ramp * dt_step
    t_base = float(points[i, 2])
    decel_pts = np.column_stack([
        points[i, 0] + np.cumsum(dx_ramp) * direction[0],
        points[i, 1] + np.cumsum(dx_ramp) * direction[1],
        t_base + t_ramp,
    ])

    speed_ramp2 = v_pre * 0.5 * (1.0 - np.cos(np.pi * t_ramp / ramp_duration))
    dx_ramp2 = speed_ramp2 * dt_step
    t_base2 = decel_pts[-1, 2] + dt_step
    accel_pts = np.column_stack([
        decel_pts[-1, 0] + np.cumsum(dx_ramp2) * direction[0],
        decel_pts[-1, 1] + np.cumsum(dx_ramp2) * direction[1],
        t_base2 + t_ramp - dt_step,
    ])

    before = points[:i + 1]
    after = points[i + 1:]
    extra_t = float(accel_pts[-1, 2] - points[i, 2])
    after_shifted = after.copy()
    after_shifted[:, 2] += extra_t

    result = np.vstack([before, decel_pts, accel_pts, after_shifted])
    new_pause_idx = i + ramp_steps
    return result, new_pause_idx

def _ou_tremor(rng: np.random.Generator, n_frames: int, amplitude: float,
               alpha: float = 0.95, boundary_pin: int = 3) -> np.ndarray:
    noise_std = amplitude * float(np.sqrt(1.0 - alpha ** 2))
    jitter = np.zeros((n_frames, 2))
    state = np.zeros(2)
    for i in range(n_frames):
        state = alpha * state + rng.standard_normal(2) * noise_std
        jitter[i] = state
    bp = min(boundary_pin, max(n_frames // 4, 1))
    for k in range(bp):
        w = (k + 1) / (bp + 1)
        jitter[k] *= w
        jitter[-(k + 1)] *= w
    return jitter

def _truncate_oscillation(traj: np.ndarray, threshold_px: float = 45.0) -> np.ndarray:
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

def postprocess_cql_trajectory(traj: np.ndarray, rng: np.random.Generator,
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
        first_pins = result[:pin, :2].copy()
        last_pins = result[-pin:, :2].copy()
        result[:, 0] = gaussian_filter1d(result[:, 0], sigma=2.5, mode='nearest')
        result[:, 1] = gaussian_filter1d(result[:, 1], sigma=2.5, mode='nearest')
        result[:pin, :2] = first_pins
        result[-pin:, :2] = last_pins

    if lateral_frac is None:
        lateral_frac = float(rng.uniform(0.10, 0.28))
    result = _add_lateral_deviation(result, lateral_frac, rng)

    if target_duration is None:
        target_duration = float(np.clip(rng.normal(2.2, 0.7), 0.8, 4.5))
    current_dur = float(result[-1, 2] - result[0, 2])
    remaining = max(0.0, target_duration - current_dur)

    start_pause = float(np.clip(remaining * 0.45 + float(rng.uniform(0.03, 0.07)), 0.04, 0.55))
    end_pause = float(np.clip(remaining * 0.55 + float(rng.uniform(0.05, 0.12)), 0.06, 0.55))

    tremor_amp = 0.28

    n_start = max(int(start_pause / 0.01) + 1, 3)
    t0 = float(result[0, 2])
    t_sp = np.linspace(t0 - start_pause, t0 - 0.005, n_start)
    jitter_s = rng.standard_normal((n_start, 2)) * tremor_amp
    jitter_s[:3] = 0.0
    jitter_s[-3:] = 0.0
    start_seg = np.column_stack([np.full(n_start, result[0, 0]) + jitter_s[:, 0],
                                  np.full(n_start, result[0, 1]) + jitter_s[:, 1],
                                  t_sp])
    result = np.vstack([start_seg, result])

    n_end = max(int(end_pause / 0.01) + 1, 3)
    t_end = float(result[-1, 2])
    t_ep = np.linspace(t_end + 0.005, t_end + end_pause, n_end)
    jitter_e = rng.standard_normal((n_end, 2)) * tremor_amp
    jitter_e[:3] = 0.0
    jitter_e[-3:] = 0.0
    end_seg = np.column_stack([np.full(n_end, result[-1, 0]) + jitter_e[:, 0],
                                np.full(n_end, result[-1, 1]) + jitter_e[:, 1],
                                t_ep])
    result = np.vstack([result, end_seg])

    return result

def generate_all_trajectories(n_movements: int, seed: int, model_path: str = None,
                              ou_sigma: float = 0.0, ou_theta: float = 3.0):
    rng = np.random.default_rng(seed)

    print("\n[1/8] Loading resources...")
    t0 = time.time()
    stats = load_statistics()
    human_template = np.array(stats.velocity_profile_template, dtype=float)
    db = TrajectoryDatabase()
    print(f"  Loaded {len(db)} human trajectories from Balabit DB")
    print(f"  Human velocity template: {len(human_template)} samples")

    cql_available = True
    synth = None
    try:
        synth = BMDSSynthesizer.load(model_path=model_path)
        label = model_path if model_path else "default CQL policy"
        print(f"  Policy loaded successfully ({label})")
    except Exception as exc:
        cql_available = False
        print(f"  WARNING: Could not load policy: {exc}")
        print("  CQL Agent source will be left empty.")
    print(f"  Resources loaded in {time.time() - t0:.1f}s")

    print(f"\n[2/8] Generating {n_movements} movement pairs...")
    pairs = s09.generate_movement_pairs(n_movements, rng)
    dists = [np.sqrt((e[0] - s[0]) ** 2 + (e[1] - s[1]) ** 2) for s, e in pairs]
    print(f"  Distance range: {min(dists):.0f} - {max(dists):.0f} px (mean {np.mean(dists):.0f})")

    print("\n[3/8] Generating trajectories from 4 sources...")
    all_trajectories = {source: [] for source in SOURCE_ORDER}

    print("  Sampling human trajectories (distance-matched)...")
    all_trajectories["Human"] = s09.sample_human_trajectories(db, pairs)
    print(f"    Human: {len(all_trajectories['Human'])}")

    if cql_available:
        print("  Generating CQL trajectories (post-processing: lateral arch + pre/post pauses)...")
        cql = []
        for i, (start, end) in enumerate(pairs):
            traj_rng = np.random.default_rng(seed * 10000 + i)
            try:
                raw = synth.generate_to_numpy(start, end, noise_seed=i)
                cql.append(postprocess_cql_trajectory(raw, traj_rng))
            except Exception:
                cql.append(None)
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{n_movements} done")
        all_trajectories["CQL Agent"] = [t for t in cql if t is not None]
    print(f"    CQL Agent: {len(all_trajectories['CQL Agent'])}")

    print("  Generating linear bot trajectories...")
    linear = []
    for start, end in pairs:
        dist = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        duration = max(dist / 800.0, 0.1)
        linear.append(
            s09.generate_linear_trajectory(
                np.array(start, dtype=float),
                np.array(end, dtype=float),
                duration=duration,
            )
        )
    all_trajectories["Linear Bot"] = linear
    print(f"    Linear Bot: {len(linear)}")

    print("  Generating bezier bot trajectories...")
    bezier = []
    for start, end in pairs:
        dist = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        duration = max(dist / 600.0 + rng.uniform(0.05, 0.15), 0.15)
        bezier.append(
            s09.generate_bezier_trajectory(
                np.array(start, dtype=float),
                np.array(end, dtype=float),
                duration=duration,
                rng=rng,
            )
        )
    all_trajectories["Bezier Bot"] = bezier
    print(f"    Bezier Bot: {len(bezier)}")

    return all_trajectories, human_template, cql_available

def extract_all_features(all_trajectories: dict, human_template: np.ndarray) -> dict:
    print("\n[4/8] Extracting 18 enhanced features...")
    all_features = {}
    for source in SOURCE_ORDER:
        feats = []
        for traj in all_trajectories.get(source, []):
            if traj is None or len(traj) < 4:
                continue
            feat = extract_features(np.asarray(traj), human_template)
            if feat is not None:
                feats.append(feat)
        all_features[source] = feats
        print(f"  {source:<10}: {len(feats)} feature vectors")
    return all_features

def _prepare_delbot_trajectory(points: np.ndarray) -> list:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return []
    if len(arr) < 4:
        return []

    xy = arr[:, :2]
    t = arr[:, 2].copy()
    t = t - t[0]
    if np.max(t) <= 20.0:
        t = t * 1000.0

    for i in range(1, len(t)):
        if not np.isfinite(t[i]) or t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1.0

    if not np.isfinite(t[0]):
        t[0] = 0.0

    clean = np.column_stack([xy, t])
    return clean.tolist()

def run_delbot_detector(all_trajectories: dict, threshold: float = 0.2) -> dict:
    print("\n[5/9] Detector 1: DELBOT-Mouse pretrained RNN...")

    if shutil.which("node") is None:
        raise RuntimeError(
            "Node.js was not found in PATH. Install Node.js and npm packages first."
        )
    if not NODE_HELPER.exists():
        raise RuntimeError(f"Node helper not found: {NODE_HELPER}")

    flat_trajectories = []
    flat_sources = []
    for source in SOURCE_ORDER:
        for traj in all_trajectories.get(source, []):
            prepared = _prepare_delbot_trajectory(traj)
            if len(prepared) >= 4:
                flat_trajectories.append(prepared)
                flat_sources.append(source)

    payload = {
        "threshold": float(threshold),
        "screen_width": 1920,
        "screen_height": 1080,
        "trajectories": flat_trajectories,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(payload, tmp)
        tmp_path = Path(tmp.name)

    cmd = ["node", str(NODE_HELPER), "--input", str(tmp_path)]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    tmp_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        raise RuntimeError(
            "DELBOT helper failed.\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}\n"
            "Run: npm install @chrisgdt/delbot-mouse @tensorflow/tfjs"
        )

    stdout_text = proc.stdout.strip()
    json_text = stdout_text
    if "\n" in stdout_text:
        lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
        if lines:
            json_text = lines[-1]

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not parse DELBOT JSON output: {exc}\nRaw stdout:\n{proc.stdout}"
        ) from exc

    results = parsed.get("results", [])
    if len(results) != len(flat_sources):
        raise RuntimeError(
            f"DELBOT returned {len(results)} results for {len(flat_sources)} trajectories."
        )

    per_source = {source: [] for source in SOURCE_ORDER}
    for source, item in zip(flat_sources, results):
        per_source[source].append(item)

    detector_sources = {}
    for source in SOURCE_ORDER:
        rows = per_source[source]
        if not rows:
            detector_sources[source] = {
                "bot_detection_rate": -1.0,
                "mean_pbot": -1.0,
                "bot_probabilities": [],
                "n_samples": 0,
            }
            continue
        detected = np.array([1.0 if bool(r.get("detected", False)) else 0.0 for r in rows])
        probs = [float(r["p_bot"]) for r in rows if isinstance(r.get("p_bot"), (int, float))]
        detector_sources[source] = {
            "bot_detection_rate": float(np.mean(detected)),
            "mean_pbot": float(np.mean(probs)) if probs else -1.0,
            "bot_probabilities": probs,
            "n_samples": len(rows),
        }

    print("  DELBOT completed successfully.")
    return {
        "name": "DELBOT RNN",
        "threshold": float(parsed.get("threshold", threshold)),
        "sources": detector_sources,
    }

def run_gradient_boosting_detector(all_features: dict, seed: int = 42) -> dict:
    print("\n[6/9] Detector 2: GradientBoosting (Human vs Linear+Bezier)...")

    human = all_features.get("Human", [])
    linear = all_features.get("Linear Bot", [])
    bezier = all_features.get("Bezier Bot", [])

    if len(human) == 0 or len(linear) == 0 or len(bezier) == 0:
        raise RuntimeError(
            "Insufficient features to train GradientBoosting. "
            "Need Human, Linear Bot, and Bezier Bot samples."
        )

    X_human = feature_matrix(human)
    X_bot = np.vstack([feature_matrix(linear), feature_matrix(bezier)])
    y_human = np.zeros(len(X_human))
    y_bot = np.ones(len(X_bot))

    X_all = np.vstack([X_human, X_bot])
    y_all = np.concatenate([y_human, y_bot])

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=seed, stratify=y_all
    )

    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(X_train, y_train)

    train_acc = float(clf.score(X_train, y_train))
    test_acc = float(clf.score(X_test, y_test))
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")

    detector_sources = {}
    for source in SOURCE_ORDER:
        feats = all_features.get(source, [])
        if not feats:
            detector_sources[source] = {
                "bot_detection_rate": -1.0,
                "mean_pbot": -1.0,
                "bot_probabilities": [],
                "n_samples": 0,
            }
            continue
        X = feature_matrix(feats)
        probs = clf.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(float)
        detector_sources[source] = {
            "bot_detection_rate": float(np.mean(preds)),
            "mean_pbot": float(np.mean(probs)),
            "bot_probabilities": probs.tolist(),
            "n_samples": len(feats),
        }

    importances = clf.feature_importances_
    return {
        "name": "GradBoost",
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "feature_importances": {FEATURE_NAMES[i]: float(importances[i]) for i in range(len(FEATURE_NAMES))},
        "sources": detector_sources,
    }

def run_adversarial_gradboost_detector(all_features: dict, seed: int = 42) -> dict:
    print("\n[8/9] Detector 4: Adversarial GradBoost (Human vs CQL Agent)...")

    human = all_features.get("Human", [])
    cql = all_features.get("CQL Agent", [])

    if len(human) == 0 or len(cql) == 0:
        print("  Skipped: need both Human and CQL Agent features.")
        empty = {src: {"bot_detection_rate": -1.0, "mean_pbot": -1.0,
                       "bot_probabilities": [], "n_samples": 0}
                 for src in SOURCE_ORDER}
        return {"name": "Adv. GradBoost", "train_accuracy": -1.0,
                "test_accuracy": -1.0, "feature_importances": {},
                "sources": empty}

    X_human = feature_matrix(human)
    X_cql = feature_matrix(cql)
    y_human = np.zeros(len(X_human))
    y_cql = np.ones(len(X_cql))

    X_all = np.vstack([X_human, X_cql])
    y_all = np.concatenate([y_human, y_cql])

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=seed, stratify=y_all
    )

    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(X_train, y_train)

    train_acc = float(clf.score(X_train, y_train))
    test_acc = float(clf.score(X_test, y_test))
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")

    detector_sources = {}
    for source in SOURCE_ORDER:
        feats = all_features.get(source, [])
        if not feats:
            detector_sources[source] = {
                "bot_detection_rate": -1.0, "mean_pbot": -1.0,
                "bot_probabilities": [], "n_samples": 0,
            }
            continue
        X = feature_matrix(feats)
        probs = clf.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(float)
        detector_sources[source] = {
            "bot_detection_rate": float(np.mean(preds)),
            "mean_pbot": float(np.mean(probs)),
            "bot_probabilities": probs.tolist(),
            "n_samples": len(feats),
        }

    importances = clf.feature_importances_
    return {
        "name": "Adv. GradBoost",
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "feature_importances": {FEATURE_NAMES[i]: float(importances[i]) for i in range(len(FEATURE_NAMES))},
        "sources": detector_sources,
    }

def _scores_to_pbot(scores: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-9)
    z = (scores - mu) / sigma
    return 1.0 / (1.0 + np.exp(2.0 * z))

def run_one_class_svm_detector(all_features: dict, nu: float = 0.15) -> dict:
    print("\n[7/9] Detector 3: One-Class SVM (human-only anomaly detector)...")

    human = all_features.get("Human", [])
    if len(human) == 0:
        raise RuntimeError("No human feature vectors available for One-Class SVM training.")

    X_human = feature_matrix(human)
    scaler = QuantileTransformer(
        n_quantiles=min(len(X_human), 100),
        output_distribution="normal",
        random_state=42,
    )
    X_human_scaled = scaler.fit_transform(X_human)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
    ocsvm.fit(X_human_scaled)

    train_pred = ocsvm.predict(X_human_scaled)
    train_scores = ocsvm.decision_function(X_human_scaled)
    train_outlier_rate = float(np.mean(train_pred == -1))
    mu = float(np.mean(train_scores))
    sigma = float(np.std(train_scores))
    print(f"  Human training outlier rate: {train_outlier_rate*100:.1f}%")

    detector_sources = {}
    for source in SOURCE_ORDER:
        feats = all_features.get(source, [])
        if not feats:
            detector_sources[source] = {
                "bot_detection_rate": -1.0,
                "mean_pbot": -1.0,
                "bot_probabilities": [],
                "n_samples": 0,
            }
            continue

        X = feature_matrix(feats)
        Xs = scaler.transform(X)
        pred = ocsvm.predict(Xs)
        scores = ocsvm.decision_function(Xs)
        probs = _scores_to_pbot(scores, mu, sigma)
        detected = (pred == -1).astype(float)

        detector_sources[source] = {
            "bot_detection_rate": float(np.mean(detected)),
            "mean_pbot": float(np.mean(probs)),
            "bot_probabilities": probs.tolist(),
            "n_samples": len(feats),
        }

    return {
        "name": "One-Class SVM",
        "nu": float(nu),
        "features_used": FEATURE_NAMES,
        "human_train_outlier_rate": train_outlier_rate,
        "sources": detector_sources,
    }

def _fmt_pct(rate: float) -> str:
    if rate < 0:
        return "N/A"
    return f"{rate * 100:.1f}%"

def print_combined_table(results: dict):
    print("\n" + "=" * 105)
    print(f"{'Source':<14} | {'DELBOT RNN':>12} | {'GradBoost':>12} | {'One-Class SVM':>14} | {'Adv. GradBoost':>15}")
    print("-" * 105)
    for source in SOURCE_ORDER:
        d1 = results["detectors"]["delbot_rnn"]["sources"][source]["bot_detection_rate"]
        d2 = results["detectors"]["gradient_boosting"]["sources"][source]["bot_detection_rate"]
        d3 = results["detectors"]["one_class_svm"]["sources"][source]["bot_detection_rate"]
        d4 = results["detectors"]["adversarial_gradboost"]["sources"][source]["bot_detection_rate"]
        print(f"{source:<14} | {_fmt_pct(d1):>12} | {_fmt_pct(d2):>12} | {_fmt_pct(d3):>14} | {_fmt_pct(d4):>15}")
    print("=" * 105)

def print_feature_comparison(all_features: dict):
    human = all_features.get("Human", [])
    cql = all_features.get("CQL Agent", [])
    if not human or not cql:
        return
    print("\nFeature means (Human vs CQL Agent):")
    print(f"  {'Feature':<36} {'Human':>10} {'CQL':>10} {'Ratio':>8}")
    print(f"  {'-'*36} {'-'*10} {'-'*10} {'-'*8}")
    for feat in FEATURE_NAMES:
        h_vals = [f[feat] for f in human if feat in f and np.isfinite(f[feat])]
        c_vals = [f[feat] for f in cql if feat in f and np.isfinite(f[feat])]
        if h_vals and c_vals:
            h_mean = float(np.mean(h_vals))
            c_mean = float(np.mean(c_vals))
            ratio = c_mean / h_mean if abs(h_mean) > 1e-9 else float("inf")
            print(f"  {feat:<36} {h_mean:>10.3f} {c_mean:>10.3f} {ratio:>8.2f}x")

def plot_comparison_chart(results: dict, output_path: Path):
    detector_keys = [
        ("delbot_rnn", "DELBOT RNN"),
        ("gradient_boosting", "GradBoost"),
        ("one_class_svm", "One-Class SVM"),
        ("adversarial_gradboost", "Adv. GradBoost"),
    ]
    x = np.arange(len(SOURCE_ORDER))
    width = 0.19

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (key, label) in enumerate(detector_keys):
        values = []
        for source in SOURCE_ORDER:
            rate = results["detectors"][key]["sources"][source]["bot_detection_rate"]
            values.append(np.nan if rate < 0 else rate * 100.0)
        ax.bar(x + (i - 1.5) * width, values, width=width, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(SOURCE_ORDER)
    ax.set_ylabel("Detected as Bot (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Multi-Detector Bot Detection Comparison")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_pbot_distributions(results: dict, output_path: Path):
    detector_keys = [
        ("delbot_rnn", "DELBOT RNN"),
        ("gradient_boosting", "GradBoost"),
        ("one_class_svm", "One-Class SVM"),
        ("adversarial_gradboost", "Adv. GradBoost"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    for ax, (det_key, det_label) in zip(axes, detector_keys):
        for source in SOURCE_ORDER:
            probs = results["detectors"][det_key]["sources"][source]["bot_probabilities"]
            if probs:
                ax.hist(
                    probs,
                    bins=20,
                    alpha=0.45,
                    density=True,
                    color=COLORS[source],
                    label=source,
                )
        ax.set_title(f"{det_label} P(bot) distribution")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("P(bot)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_feature_importance(results: dict, output_path: Path):
    imps = results["detectors"]["gradient_boosting"]["feature_importances"]
    names = np.array(list(imps.keys()))
    values = np.array([imps[n] for n in names], dtype=float)
    idx = np.argsort(values)[::-1]
    names_sorted = names[idx]
    values_sorted = values[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names_sorted[::-1], values_sorted[::-1], color="#2980b9", alpha=0.85)
    ax.set_xlabel("Importance")
    ax.set_title("GradientBoosting Feature Importances")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def run_experiment(n_movements: int = 100, seed: int = 42,
                   delbot_threshold: float = 0.2,
                   ocsvm_nu: float = 0.1,
                   model_path: str = None,
                   ou_sigma: float = 1.2,
                   ou_theta: float = 3.0):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Multi-Detector Bot Detection Gauntlet")
    print("=" * 70)

    all_trajectories, human_template, cql_available = generate_all_trajectories(
        n_movements=n_movements,
        seed=seed,
        model_path=model_path,
        ou_sigma=ou_sigma,
        ou_theta=ou_theta,
    )
    all_features = extract_all_features(all_trajectories, human_template)

    delbot = run_delbot_detector(all_trajectories, threshold=delbot_threshold)
    gradboost = run_gradient_boosting_detector(all_features, seed=seed)
    ocsvm = run_one_class_svm_detector(all_features, nu=ocsvm_nu)
    adv_gb = run_adversarial_gradboost_detector(all_features, seed=seed)

    print("\n[9/9] Saving outputs...")
    results = {
        "method": "Multi-Detector Bot Detection Gauntlet",
        "n_movements": int(n_movements),
        "seed": int(seed),
        "cql_available": bool(cql_available),
        "feature_names": FEATURE_NAMES,
        "detectors": {
            "delbot_rnn": delbot,
            "gradient_boosting": gradboost,
            "one_class_svm": ocsvm,
            "adversarial_gradboost": adv_gb,
        },
    }

    print_combined_table(results)
    print_feature_comparison(all_features)

    feature_means = {}
    for source in SOURCE_ORDER:
        feats = all_features.get(source, [])
        if feats:
            feature_means[source] = {
                feat: float(np.mean([f[feat] for f in feats if feat in f and np.isfinite(f[feat])]))
                for feat in FEATURE_NAMES
            }
    results["feature_means"] = feature_means

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {OUTPUT_DIR / 'results.json'}")

    plot_comparison_chart(results, OUTPUT_DIR / "comparison_chart.png")
    print(f"  Saved {OUTPUT_DIR / 'comparison_chart.png'}")

    plot_pbot_distributions(results, OUTPUT_DIR / "pbot_distributions.png")
    print(f"  Saved {OUTPUT_DIR / 'pbot_distributions.png'}")

    plot_feature_importance(results, OUTPUT_DIR / "feature_importance.png")
    print(f"  Saved {OUTPUT_DIR / 'feature_importance.png'}")

    print("\nDone.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-detector bot detection gauntlet.")
    parser.add_argument(
        "--n-movements",
        type=int,
        default=100,
        help="Number of movement pairs per source (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--delbot-threshold",
        type=float,
        default=0.2,
        help="DELBOT isHuman threshold (default: 0.2).",
    )
    parser.add_argument(
        "--ocsvm-nu",
        type=float,
        default=0.15,
        help="One-Class SVM nu parameter (default: 0.15).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to policy .d3 file (default: models/bmds_cql_policy.d3).",
    )
    parser.add_argument(
        "--ou-sigma",
        type=float,
        default=1.2,
        help="OU motor noise sigma for CQL trajectories (0=off, default: 1.2).",
    )
    parser.add_argument(
        "--ou-theta",
        type=float,
        default=3.0,
        help="OU motor noise mean-reversion rate (default: 3.0).",
    )
    args = parser.parse_args()

    run_experiment(
        n_movements=args.n_movements,
        seed=args.seed,
        delbot_threshold=args.delbot_threshold,
        ocsvm_nu=args.ocsvm_nu,
        model_path=args.model,
        ou_sigma=args.ou_sigma,
        ou_theta=args.ou_theta,
    )
