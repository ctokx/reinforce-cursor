"""
Usage:
    python scripts/10_third_party_bot_test.py --n-movements 100
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bmds.data.trajectory_db import TrajectoryDatabase
from bmds.synthesizer import BMDSSynthesizer

from scripts import _script09_helpers as s09

OUTPUT_DIR = PROJECT_ROOT / "output" / "bot_detection_v3"

def trajectory_to_image(points: np.ndarray, img_size: int = 299,
                        screen_res: tuple = (1920, 1080)) -> np.ndarray:
    img = np.zeros((img_size, img_size, 3), dtype=np.float32)

    x = points[:, 0]
    y = points[:, 1]

    x_img = (x - x.min()) / max(x.max() - x.min(), 1e-6) * (img_size - 1)
    y_img = (y - y.min()) / max(y.max() - y.min(), 1e-6) * (img_size - 1)

    for i in range(len(x_img) - 1):
        x0, y0 = x_img[i], y_img[i]
        x1, y1 = x_img[i + 1], y_img[i + 1]

        dist = max(abs(x1 - x0), abs(y1 - y0))
        n_steps = max(int(dist) + 1, 2)
        ts = np.linspace(0, 1, n_steps)

        for t in ts:
            px = int(np.clip(x0 + (x1 - x0) * t, 0, img_size - 1))
            py = int(np.clip(y0 + (y1 - y0) * t, 0, img_size - 1))
            img[py, px, 0] = 1.0

    return img

def trajectories_to_images(trajectories: list, label: str = "",
                           img_size: int = 299) -> list:
    images = []
    for traj in trajectories:
        if traj is None or len(traj) < 4:
            continue
        try:
            img = trajectory_to_image(traj, img_size=img_size)
            images.append(img)
        except Exception:
            continue
    if label:
        print(f"  {label}: {len(images)} images from {len(trajectories)} trajectories")
    return images

def build_inception_model():
    import tensorflow as tf

    base = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(299, 299, 3),
    )
    base.trainable = False

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model

def preprocess_for_inception(images: np.ndarray) -> np.ndarray:
    import tensorflow as tf
    return tf.keras.applications.inception_v3.preprocess_input(images * 255.0)

def train_model(model, X_train, y_train, X_val, y_val, epochs: int = 10):
    history = model.fit(
        preprocess_for_inception(X_train), y_train,
        validation_data=(preprocess_for_inception(X_val), y_val),
        epochs=epochs,
        batch_size=16,
        verbose=1,
    )
    return history

def plot_detection_rates(results: dict, output_path: Path):
    sources = list(results.keys())
    rates = [results[s]["bot_detection_rate"] * 100 for s in sources]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(sources, rates, color=colors[:len(sources)],
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Detected as Bot (%)")
    ax.set_title("InceptionV3 Bot Detection Rates (Mouse-BB-Team Method)")
    ax.set_ylim(0, 105)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_pbot_distributions(results: dict, output_path: Path):
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
    ax.set_title("InceptionV3 P(bot) Distributions by Source")
    ax.legend()
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_sample_images(all_images: dict, output_path: Path, n_samples: int = 4):
    sources = list(all_images.keys())
    n_sources = len(sources)

    fig, axes = plt.subplots(n_sources, n_samples,
                             figsize=(3 * n_samples, 3 * n_sources))
    if n_sources == 1:
        axes = axes[np.newaxis, :]

    for row, src in enumerate(sources):
        imgs = all_images[src]
        for col in range(n_samples):
            ax = axes[row, col]
            if col < len(imgs):
                ax.imshow(imgs[col])
            else:
                ax.imshow(np.zeros((299, 299, 3)))
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(src, fontsize=11, rotation=0, labelpad=70,
                              va="center")

    plt.suptitle("Sample Trajectory Images (Red Path on Black)", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

def run_experiment(n_movements: int = 100, seed: int = 42, epochs: int = 10):
    rng = np.random.default_rng(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  InceptionV3 Bot Detection (Mouse-BB-Team Method)")
    print("=" * 65)

    print("\n[1/7] Loading resources...")
    t0 = time.time()
    db = TrajectoryDatabase()
    print(f"  Loaded {len(db)} human trajectories from Balabit DB")

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

    print(f"\n[2/7] Generating {n_movements} movement pairs...")
    pairs = s09.generate_movement_pairs(n_movements, rng)
    dists = [np.sqrt((e[0]-s[0])**2 + (e[1]-s[1])**2) for s, e in pairs]
    print(f"  Distance range: {min(dists):.0f} - {max(dists):.0f} px "
          f"(mean {np.mean(dists):.0f})")

    print("\n[3/7] Generating trajectories...")
    all_trajectories = {}

    print("  Sampling human trajectories (distance-matched)...")
    all_trajectories["Human"] = s09.sample_human_trajectories(db, pairs)
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
        duration = dist / 800.0
        traj = s09.generate_linear_trajectory(
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
        traj = s09.generate_bezier_trajectory(
            np.array(start, dtype=float), np.array(end, dtype=float),
            duration=max(duration, 0.15), rng=rng)
        bezier_trajs.append(traj)
    all_trajectories["Bezier Bot"] = bezier_trajs
    print(f"    Got {len(bezier_trajs)} bezier trajectories")

    print("\n[4/7] Converting trajectories to 299x299 images...")
    t0 = time.time()
    all_images = {}
    for source, trajs in all_trajectories.items():
        all_images[source] = trajectories_to_images(trajs, label=source)
    print(f"  Image conversion done in {time.time() - t0:.1f}s")

    print(f"\n[5/7] Training InceptionV3 (Human vs Linear Bot, {epochs} epochs)...")
    t0 = time.time()

    human_imgs = np.array(all_images["Human"])
    linear_imgs = np.array(all_images["Linear Bot"])

    X_all = np.concatenate([human_imgs, linear_imgs], axis=0)
    y_all = np.concatenate([
        np.zeros(len(human_imgs)),
        np.ones(len(linear_imgs)),
    ])

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.3, random_state=seed, stratify=y_all)

    print(f"  Train: {len(X_train)} images  |  Val: {len(X_val)} images")

    model = build_inception_model()
    trainable = sum(p.numpy().size for p in model.trainable_weights)
    print(f"  Trainable parameters: {trainable:,}")

    history = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs)
    val_acc = history.history["val_accuracy"][-1]
    print(f"  Final val accuracy: {val_acc:.3f}")
    print(f"  Training done in {time.time() - t0:.1f}s")

    print("\n[6/7] Evaluating all sources against InceptionV3 detector...")
    results = {}
    for source, imgs in all_images.items():
        if not imgs:
            results[source] = {
                "bot_detection_rate": -1,
                "mean_pbot": -1,
                "bot_probabilities": [],
                "n_samples": 0,
            }
            continue

        X = preprocess_for_inception(np.array(imgs))
        probs = model.predict(X, verbose=0).flatten()  # P(bot)
        predictions = (probs >= 0.5).astype(int)
        bot_rate = float(np.mean(predictions))
        mean_pbot = float(np.mean(probs))

        results[source] = {
            "bot_detection_rate": bot_rate,
            "mean_pbot": mean_pbot,
            "bot_probabilities": probs.tolist(),
            "n_samples": len(imgs),
        }

    print("\n" + "=" * 70)
    print(f"{'Source':<20} | {'Detected as Bot':>15} | {'Mean P(bot)':>12} | {'N':>5}")
    print("-" * 70)
    for source in all_images:
        r = results[source]
        if r["n_samples"] > 0:
            print(f"{source:<20} | {r['bot_detection_rate']*100:>14.1f}% | "
                  f"{r['mean_pbot']:>12.3f} | {r['n_samples']:>5}")
        else:
            print(f"{source:<20} | {'N/A':>15} | {'N/A':>12} | {0:>5}")
    print("=" * 70)

    print("\n[7/7] Saving results and plots...")

    results_summary = {}
    for source, r in results.items():
        results_summary[source] = {
            "bot_detection_rate": r["bot_detection_rate"],
            "mean_pbot": r["mean_pbot"],
            "n_samples": r["n_samples"],
        }

    results_path = OUTPUT_DIR / "results.json"
    full_results = {
        "method": "InceptionV3 (Mouse-BB-Team transfer learning)",
        "architecture": "InceptionV3(frozen,imagenet) → GAP → Dense(30,relu) → Dense(1,sigmoid)",
        "trained_on": "Human vs Linear Bot",
        "epochs": epochs,
        "val_accuracy": float(val_acc),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "sources": results_summary,
    }
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"  Saved {results_path}")

    with open(OUTPUT_DIR / "results_full.json", "w") as f:
        json.dump({s: r for s, r in results.items()}, f, indent=2)

    plot_detection_rates(results, OUTPUT_DIR / "detection_rates.png")
    print("  Saved detection_rates.png")

    plot_pbot_distributions(results, OUTPUT_DIR / "pbot_distributions.png")
    print("  Saved pbot_distributions.png")

    plot_sample_images(all_images, OUTPUT_DIR / "sample_images.png")
    print("  Saved sample_images.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"], label="train")
    ax1.plot(history.history["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    ax2.plot(history.history["accuracy"], label="train")
    ax2.plot(history.history["val_accuracy"], label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_history.png", dpi=150)
    plt.close(fig)
    print("  Saved training_history.png")

    print("\nDone!")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="InceptionV3 Bot Detection (Mouse-BB-Team Method)")
    parser.add_argument("--n-movements", type=int, default=100,
                        help="Number of movements per source (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs (default: 10)")
    args = parser.parse_args()

    run_experiment(n_movements=args.n_movements, seed=args.seed,
                   epochs=args.epochs)
