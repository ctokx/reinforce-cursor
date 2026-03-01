"""
Usage:
    python scripts/09_bot_detection_test.py --n-movements 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bmds.data.trajectory_db import TrajectoryDatabase
from bmds.data.statistics import load_statistics
from bmds.synthesizer import BMDSSynthesizer
from bmds.utils.kinematics import (
    compute_kinematics, count_submovements, normalize_speed_profile,
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "bot_detection"

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

def generate_linear_trajectory(start: np.ndarray, end: np.ndarray,
                               duration: float = 0.5,
                               dt: float = 0.01) -> np.ndarray:
    n_steps = max(int(duration / dt), 5)
    t = np.linspace(0, duration, n_steps)
    frac = t / duration
    x = start[0] + (end[0] - start[0]) * frac
    y = start[1] + (end[1] - start[1]) * frac
    return np.column_stack([x, y, t])

def generate_bezier_trajectory(start: np.ndarray, end: np.ndarray,
                               duration: float = 0.5,
                               dt: float = 0.01,
                               rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    n_steps = max(int(duration / dt), 5)
    t = np.linspace(0, duration, n_steps)

    s = t / duration
    s_eased = 3 * s ** 2 - 2 * s ** 3

    mid = (start + end) / 2
    spread = np.linalg.norm(end - start) * 0.3
    perp = np.array([-(end[1] - start[1]), end[0] - start[0]])
    perp_norm = np.linalg.norm(perp)
    if perp_norm > 1e-9:
        perp = perp / perp_norm
    else:
        perp = np.array([0.0, 1.0])

    cp1 = start + (end - start) * 0.33 + perp * rng.uniform(-spread, spread)
    cp2 = start + (end - start) * 0.66 + perp * rng.uniform(-spread, spread)

    u = s_eased
    x = ((1 - u) ** 3 * start[0] + 3 * (1 - u) ** 2 * u * cp1[0] +
         3 * (1 - u) * u ** 2 * cp2[0] + u ** 3 * end[0])
    y = ((1 - u) ** 3 * start[1] + 3 * (1 - u) ** 2 * u * cp1[1] +
         3 * (1 - u) * u ** 2 * cp2[1] + u ** 3 * end[1])

    return np.column_stack([x, y, t])

FEATURE_NAMES = [
    "peak_speed", "mean_speed", "speed_std", "path_efficiency",
    "max_acceleration", "max_deceleration", "mean_jerk",
    "num_submovements", "mean_curvature", "duration",
    "speed_profile_correlation", "jerk_smoothness",
]

def extract_features(points: np.ndarray, human_template: np.ndarray) -> dict:
    if len(points) < 4:
        return None

    kin = compute_kinematics(points)
    n_sub = count_submovements(kin.speed)
    spc = speed_profile_correlation(kin.speed, human_template)
    js = jerk_smoothness(kin.jerk_magnitude, kin.duration, kin.peak_speed)

    return {
        "peak_speed": kin.peak_speed,
        "mean_speed": kin.mean_speed,
        "speed_std": float(np.std(kin.speed)),
        "path_efficiency": kin.path_efficiency,
        "max_acceleration": kin.max_acceleration,
        "max_deceleration": kin.max_deceleration,
        "mean_jerk": kin.mean_jerk,
        "num_submovements": float(n_sub),
        "mean_curvature": float(np.mean(np.abs(kin.curvature))) if len(kin.curvature) > 0 else 0.0,
        "duration": kin.duration,
        "speed_profile_correlation": spc,
        "jerk_smoothness": js,
    }

def generate_movement_pairs(n: int, rng: np.random.Generator,
                            screen_res=(1920, 1080)) -> list:
    pairs = []
    for _ in range(n):
        while True:
            sx = rng.integers(100, screen_res[0] - 100)
            sy = rng.integers(100, screen_res[1] - 100)
            ex = rng.integers(100, screen_res[0] - 100)
            ey = rng.integers(100, screen_res[1] - 100)
            dist = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
            if 100 < dist < 800:
                pairs.append(((int(sx), int(sy)), (int(ex), int(ey))))
                break
    return pairs

def sample_human_trajectories(db: TrajectoryDatabase, pairs: list,
                              tolerance: float = 0.3) -> list:
    distances = db.get_feature("distance")
    all_indices = np.arange(len(distances))
    rng = np.random.default_rng(42)
    trajectories = []

    for start, end in pairs:
        target_dist = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        lo = target_dist * (1 - tolerance)
        hi = target_dist * (1 + tolerance)
        mask = (distances >= lo) & (distances <= hi)
        candidates = all_indices[mask]

        if len(candidates) == 0:
            idx = int(np.argmin(np.abs(distances - target_dist)))
        else:
            idx = int(rng.choice(candidates))

        traj = db.get_trajectory(idx)
        trajectories.append(traj.points)

    return trajectories

def plot_detection_rates(results: dict, output_path: Path):
    sources = list(results.keys())
    rates = [results[s]["bot_detection_rate"] * 100 for s in sources]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(sources, rates, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Detected as Bot (%)")
    ax.set_title("Bot Detection Rates by Source")
    ax.set_ylim(0, 105)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_feature_distributions(all_features: dict, output_path: Path):
    sources = list(all_features.keys())
    colors = {"Human": "#2ecc71", "CQL Agent": "#3498db",
              "Linear Bot": "#e74c3c", "Bezier Bot": "#f39c12"}

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i, feat_name in enumerate(FEATURE_NAMES):
        ax = axes[i]
        for src in sources:
            vals = [f[feat_name] for f in all_features[src] if f is not None]
            if vals:
                ax.hist(vals, bins=20, alpha=0.5, label=src,
                        color=colors.get(src, "gray"), density=True)
        ax.set_title(feat_name, fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    plt.suptitle("Feature Distributions by Source", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_score_distributions(results: dict, output_path: Path):
    sources = list(results.keys())
    colors = {"Human": "#2ecc71", "CQL Agent": "#3498db",
              "Linear Bot": "#e74c3c", "Bezier Bot": "#f39c12"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for src in sources:
        probs = results[src]["bot_probabilities"]
        if probs:
            ax.hist(probs, bins=20, alpha=0.5, label=src,
                    color=colors.get(src, "gray"), density=True)

    ax.set_xlabel("P(bot)")
    ax.set_ylabel("Density")
    ax.set_title("Bot Probability Distributions by Source")
    ax.legend()
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def run_experiment(n_movements: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Bot Detection Test: CQL Agent vs Naive Bots vs Humans")
    print("=" * 60)

    print("\n[1/6] Loading resources...")
    t0 = time.time()
    stats = load_statistics()
    human_template = np.array(stats.velocity_profile_template)
    db = TrajectoryDatabase()
    print(f"  Loaded {len(db)} human trajectories from Balabit DB")
    print(f"  Human velocity template: {len(human_template)} samples")

    print("  Loading CQL policy...")
    try:
        synth = BMDSSynthesizer.load()
        cql_available = True
        print("  CQL policy loaded successfully")
    except Exception as e:
        print(f"  WARNING: Could not load CQL policy: {e}")
        print("  CQL Agent will be skipped")
        cql_available = False
    print(f"  Resources loaded in {time.time() - t0:.1f}s")

    print(f"\n[2/6] Generating {n_movements} movement pairs...")
    pairs = generate_movement_pairs(n_movements, rng)
    dists = [np.sqrt((e[0]-s[0])**2 + (e[1]-s[1])**2) for s, e in pairs]
    print(f"  Distance range: {min(dists):.0f} - {max(dists):.0f} px (mean {np.mean(dists):.0f})")

    all_trajectories = {}

    print("\n[3/6] Generating trajectories...")
    print("  Sampling human trajectories (distance-matched)...")
    all_trajectories["Human"] = sample_human_trajectories(db, pairs)
    print(f"    Got {len(all_trajectories['Human'])} human trajectories")

    if cql_available:
        print("  Generating CQL agent trajectories...")
        cql_trajs = []
        for i, (start, end) in enumerate(pairs):
            try:
                traj = synth.generate_to_numpy(start, end)
                cql_trajs.append(traj)
            except Exception as e:
                if i == 0:
                    print(f"    WARNING: CQL generation failed: {e}")
                cql_trajs.append(None)
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{n_movements} done")
        all_trajectories["CQL Agent"] = [t for t in cql_trajs if t is not None]
        print(f"    Got {len(all_trajectories['CQL Agent'])} CQL trajectories")

    print("  Generating linear bot trajectories...")
    linear_trajs = []
    for start, end in pairs:
        dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        duration = dist / 800.0  # constant 800 px/s
        traj = generate_linear_trajectory(
            np.array(start, dtype=float), np.array(end, dtype=float),
            duration=max(duration, 0.1))
        linear_trajs.append(traj)
    all_trajectories["Linear Bot"] = linear_trajs
    print(f"    Got {len(linear_trajs)} linear trajectories")

    print("  Generating bezier bot trajectories...")
    bezier_trajs = []
    for start, end in pairs:
        dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        duration = dist / 600.0 + rng.uniform(0.05, 0.15)
        traj = generate_bezier_trajectory(
            np.array(start, dtype=float), np.array(end, dtype=float),
            duration=max(duration, 0.15), rng=rng)
        bezier_trajs.append(traj)
    all_trajectories["Bezier Bot"] = bezier_trajs
    print(f"    Got {len(bezier_trajs)} bezier trajectories")

    print("\n[4/6] Extracting kinematic features...")
    all_features = {}
    for source, trajs in all_trajectories.items():
        features = []
        for traj in trajs:
            if traj is not None and len(traj) >= 4:
                feat = extract_features(traj, human_template)
                if feat is not None:
                    features.append(feat)
        all_features[source] = features
        print(f"  {source}: {len(features)} valid feature vectors")

    print("\n[5/6] Training RandomForest classifier (Human vs Linear Bot)...")
    human_feats = all_features["Human"]
    linear_feats = all_features["Linear Bot"]

    X_human = np.array([[f[k] for k in FEATURE_NAMES] for f in human_feats])
    X_linear = np.array([[f[k] for k in FEATURE_NAMES] for f in linear_feats])

    X_train_all = np.vstack([X_human, X_linear])
    y_train_all = np.concatenate([np.zeros(len(X_human)), np.ones(len(X_linear))])

    X_train_all = np.nan_to_num(X_train_all, nan=0.0, posinf=0.0, neginf=0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_all, y_train_all, test_size=0.3, random_state=seed, stratify=y_train_all)

    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")

    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("  Top features:")
    for i in sorted_idx[:5]:
        print(f"    {FEATURE_NAMES[i]}: {importances[i]:.3f}")

    print("\n[6/6] Evaluating all sources against detector...")
    results = {}
    for source, feats in all_features.items():
        if not feats:
            results[source] = {
                "bot_detection_rate": -1,
                "mean_human_likeness": -1,
                "bot_probabilities": [],
                "n_samples": 0,
            }
            continue

        X = np.array([[f[k] for k in FEATURE_NAMES] for f in feats])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        probs = clf.predict_proba(X)[:, 1]
        predictions = clf.predict(X)
        bot_rate = float(np.mean(predictions))
        human_likeness = float(1.0 - np.mean(probs))

        results[source] = {
            "bot_detection_rate": bot_rate,
            "mean_human_likeness": human_likeness,
            "bot_probabilities": probs.tolist(),
            "n_samples": len(feats),
        }

    print("\n" + "=" * 65)
    print(f"{'Source':<20} | {'Detected as Bot':>15} | {'Human-likeness':>15}")
    print("-" * 65)
    for source in all_features:
        r = results[source]
        if r["n_samples"] > 0:
            print(f"{source:<20} | {r['bot_detection_rate']*100:>14.1f}% | "
                  f"{r['mean_human_likeness']:>15.3f}")
        else:
            print(f"{source:<20} | {'N/A':>15} | {'N/A':>15}")
    print("=" * 65)

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    meta = {
        "classifier": "RandomForest",
        "n_estimators": 100,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "feature_names": FEATURE_NAMES,
        "feature_importances": {FEATURE_NAMES[i]: float(importances[i])
                                for i in range(len(FEATURE_NAMES))},
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(OUTPUT_DIR / "classifier_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Generating plots...")
    plot_detection_rates(results, OUTPUT_DIR / "bot_detection_rates.png")
    print(f"  Saved bot_detection_rates.png")

    plot_feature_distributions(all_features, OUTPUT_DIR / "feature_distributions.png")
    print(f"  Saved feature_distributions.png")

    plot_score_distributions(results, OUTPUT_DIR / "score_distributions.png")
    print(f"  Saved score_distributions.png")

    print("\nDone!")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bot Detection Test")
    parser.add_argument("--n-movements", type=int, default=100,
                        help="Number of movements per source (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    run_experiment(n_movements=args.n_movements, seed=args.seed)
