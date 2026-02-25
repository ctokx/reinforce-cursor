import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import List, Optional, Tuple

from bmds.utils.kinematics import KinematicProfile, compute_kinematics


def plot_trajectory(points: np.ndarray, title: str = "Mouse Trajectory",
                    color_by_speed: bool = True, ax: Optional[plt.Axes] = None,
                    show: bool = True) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x, y = points[:, 0], points[:, 1]

    if color_by_speed and points.shape[1] == 3 and len(points) >= 4:
        kin = compute_kinematics(points)

        seg_points = np.column_stack([x[:-1], y[:-1], x[1:], y[1:]]).reshape(-1, 2, 2)
        lc = LineCollection(seg_points, cmap="plasma", linewidth=2)
        lc.set_array(kin.speed)
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label="Speed (px/s)")
        ax.set_xlim(x.min() - 20, x.max() + 20)
        ax.set_ylim(y.min() - 20, y.max() + 20)
    else:
        ax.plot(x, y, "b-", linewidth=1.5, alpha=0.8)

    ax.plot(x[0], y[0], "go", markersize=10, label="Start")
    ax.plot(x[-1], y[-1], "rs", markersize=10, label="End")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.invert_yaxis()

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_velocity_profile(points: np.ndarray, title: str = "Velocity Profile",
                          ax: Optional[plt.Axes] = None,
                          show: bool = True) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    kin = compute_kinematics(points)
    t = kin.timestamps[:-1]
    t_relative = t - t[0]

    ax.plot(t_relative, kin.speed, "b-", linewidth=1.5)
    ax.fill_between(t_relative, 0, kin.speed, alpha=0.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (px/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_kinematic_dashboard(points: np.ndarray, title: str = "Kinematic Dashboard",
                             show: bool = True) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    kin = compute_kinematics(points)
    t = kin.timestamps
    t0 = t[0]


    plot_trajectory(points, title="Trajectory", ax=axes[0, 0], show=False)


    axes[0, 1].plot(t[:-1] - t0, kin.speed, "b-", linewidth=1.2)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Speed (px/s)")
    axes[0, 1].set_title("Speed Profile")
    axes[0, 1].grid(True, alpha=0.3)


    axes[1, 0].plot(t[:-2] - t0, kin.accel_magnitude, "r-", linewidth=1.0)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Acceleration (px/s²)")
    axes[1, 0].set_title("Acceleration Magnitude")
    axes[1, 0].grid(True, alpha=0.3)


    axes[1, 1].plot(t[:-2] - t0, kin.curvature, "g-", linewidth=1.0)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Curvature (1/px)")
    axes[1, 1].set_title("Curvature")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def compare_trajectories(real: np.ndarray, synthetic: np.ndarray,
                         title: str = "Real vs Synthetic",
                         show: bool = True) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14)


    axes[0].plot(real[:, 0], real[:, 1], "b-", linewidth=1.5, label="Real", alpha=0.7)
    axes[0].plot(synthetic[:, 0], synthetic[:, 1], "r--", linewidth=1.5,
                 label="Synthetic", alpha=0.7)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_title("Trajectory Shape")
    axes[0].legend()
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()


    if real.shape[1] == 3 and len(real) >= 4:
        kin_real = compute_kinematics(real)
        normalized_real = kin_real.speed / max(kin_real.peak_speed, 1e-9)
        t_real = np.linspace(0, 1, len(normalized_real))
        axes[1].plot(t_real, normalized_real, "b-", label="Real", linewidth=1.5)

    if synthetic.shape[1] == 3 and len(synthetic) >= 4:
        kin_syn = compute_kinematics(synthetic)
        normalized_syn = kin_syn.speed / max(kin_syn.peak_speed, 1e-9)
        t_syn = np.linspace(0, 1, len(normalized_syn))
        axes[1].plot(t_syn, normalized_syn, "r--", label="Synthetic", linewidth=1.5)

    axes[1].set_xlabel("Normalized Time")
    axes[1].set_ylabel("Normalized Speed")
    axes[1].set_title("Speed Profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)


    if real.shape[1] == 3 and len(real) >= 4:
        axes[2].plot(np.linspace(0, 1, len(kin_real.accel_magnitude)),
                     kin_real.accel_magnitude, "b-", label="Real", linewidth=1.0)
    if synthetic.shape[1] == 3 and len(synthetic) >= 4:
        axes[2].plot(np.linspace(0, 1, len(kin_syn.accel_magnitude)),
                     kin_syn.accel_magnitude, "r--", label="Synthetic", linewidth=1.0)

    axes[2].set_xlabel("Normalized Time")
    axes[2].set_ylabel("Acceleration (px/s²)")
    axes[2].set_title("Acceleration Profile")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    return fig
