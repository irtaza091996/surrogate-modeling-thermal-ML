"""
Shared utilities for ROM+LSTM and PINN surrogate models.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_TXT    = Path(__file__).parent / "data" / "thermal_analysis_data.txt"
RESULTS_DIR = Path(__file__).parent / "results"

# ── Normalization constants (fit once, shared everywhere) ──────────────────
X_SCALE = 20.0      # x ∈ [0, 20]
Y_SCALE = 200.0     # y ∈ [0, 200]
T_SCALE = 2000.0    # t ∈ [0, 2000]
TEMP_MIN = 25.0
TEMP_MAX = 500.0
TEMP_RANGE = TEMP_MAX - TEMP_MIN   # 475

TRAIN_RATIO = 0.8   # temporal split (train: t=0-1600s, test: t=1600-2000s)


# ── Data loading ───────────────────────────────────────────────────────────

def load_data(path: Path = DATA_TXT) -> pd.DataFrame:
    data = pd.read_csv(path, sep=r"\s+")
    print(f"  Loaded {len(data):,} rows | "
          f"{data['Node_number'].nunique():,} nodes | "
          f"t in [{data['time'].min()}, {data['time'].max()}]s")
    return data


def make_snapshot_matrix(data: pd.DataFrame):
    """
    Returns
    -------
    T_matrix : (n_time, n_nodes)  — temperature snapshots
    times    : (n_time,)          — simulation time values
    node_ids : (n_nodes,)         — node numbers
    coords   : DataFrame          — X_coordinate, Y_coordinate per node
    """
    pivot = data.pivot_table(
        index="time", columns="Node_number", values="temperature"
    )
    T_matrix = pivot.values.astype(np.float32)
    times    = pivot.index.values.astype(np.float32)
    node_ids = pivot.columns.values

    coords = (
        data.groupby("Node_number")[["X_coordinate", "Y_coordinate"]]
        .first()
        .loc[node_ids]
    )
    return T_matrix, times, node_ids, coords


def temporal_split(T_matrix, times, ratio=TRAIN_RATIO):
    n_train = int(ratio * len(times))
    return (
        T_matrix[:n_train], T_matrix[n_train:],
        times[:n_train],    times[n_train:],
        n_train,
    )


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(T_true: np.ndarray, T_pred: np.ndarray) -> dict:
    """Both arrays shaped (n_time, n_nodes) or flat."""
    a, b = T_true.flatten(), T_pred.flatten()
    return {
        "R2":   r2_score(a, b),
        "RMSE": float(np.sqrt(mean_squared_error(a, b))),
        "MAE":  float(mean_absolute_error(a, b)),
    }


def per_node_metrics(T_true: np.ndarray, T_pred: np.ndarray) -> np.ndarray:
    """Returns MAE per node. Both (n_time, n_nodes)."""
    return np.mean(np.abs(T_true - T_pred), axis=0)


# ── Visualisation ──────────────────────────────────────────────────────────

def plot_field_comparison(
    T_true, T_pred, coords, times,
    snapshot_indices, model_name, save_path
):
    """2-row grid: FEM actual vs model predicted at given time indices."""
    n = len(snapshot_indices)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes[:, np.newaxis]

    vmin, vmax = float(T_true.min()), float(T_true.max())

    for col, idx in enumerate(snapshot_indices):
        for row, (T, label) in enumerate(
            [(T_true, "FEM Actual"), (T_pred, model_name)]
        ):
            if idx >= len(T):
                idx = len(T) - 1
            T_snap = T[idx]
            df_plot = pd.DataFrame({
                "X": coords["X_coordinate"].values,
                "Y": coords["Y_coordinate"].values,
                "T": T_snap,
            })
            pivot = df_plot.pivot_table(
                index="Y", columns="X", values="T", aggfunc="mean"
            )
            im = axes[row, col].imshow(
                pivot.values[::-1], cmap="coolwarm",
                vmin=vmin, vmax=vmax, aspect="auto",
                extent=[
                    pivot.columns.min(), pivot.columns.max(),
                    pivot.index.min(),   pivot.index.max(),
                ],
            )
            axes[row, col].set_title(
                f"{label}\nt = {times[idx]:.0f}s", fontweight="bold"
            )
            axes[row, col].set_xlabel("X")
            axes[row, col].set_ylabel("Y")
            plt.colorbar(im, ax=axes[row, col], label="°C", fraction=0.046)

    plt.suptitle(
        f"Temperature Field — FEM vs {model_name}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {Path(save_path).name}")
