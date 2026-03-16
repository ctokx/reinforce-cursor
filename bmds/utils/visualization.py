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


