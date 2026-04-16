"""
ROM + LSTM Surrogate Model (v2)
================================
Architecture change: the LSTM's autoregressive prediction is replaced by a
**direct temporal regression MLP** that maps t → modal coefficients.

Why this matters
----------------
POD captures 99.99999 % of variance with 10 modes, so the bottleneck is
purely learning the time-evolution of those 10 scalar coefficients.
For a monotonically-heating thermal simulation the modal amplitudes are
smooth, slowly-varying functions of time.  A small MLP trained directly on
(t → coefficients) extrapolates these curves far more reliably than an
autoregressive LSTM, which compounds errors with every step.

Physics-inspired input features  [t, t², log(1+t), sin(πt), cos(πt)]
give the network multiple temporal scales to work with, helping it capture
both early-transient and late-steady-state behaviour without needing to be
deep or wide.

Run:
    python rom_lstm.py
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    RESULTS_DIR, T_SCALE, compute_metrics, load_data,
    make_snapshot_matrix, per_node_metrics,
    plot_field_comparison, temporal_split,
)

# ── Config ─────────────────────────────────────────────────────────────────
N_MODES    = 10     # POD modes (99.99999 % energy — no need to increase)
BATCH_SIZE = 16
MAX_EPOCHS = 2000
PATIENCE   = 100
SEED       = 42

OUT = RESULTS_DIR / "rom_lstm"
OUT.mkdir(parents=True, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ── POD Decomposition ──────────────────────────────────────────────────────

class PODDecomposition:
    """Proper Orthogonal Decomposition via truncated SVD."""

    def __init__(self, n_modes: int = N_MODES):
        self.n_modes = n_modes
        self.mean_   = None
        self.modes_  = None
        self.singular_values_ = None
        self.cumulative_energy_ = None

    def fit(self, T_train: np.ndarray) -> "PODDecomposition":
        self.mean_ = T_train.mean(axis=0, keepdims=True)
        T_c = T_train - self.mean_
        U, S, Vt = np.linalg.svd(T_c, full_matrices=False)
        self.singular_values_   = S
        self.modes_             = Vt[: self.n_modes]
        total_energy            = np.sum(S ** 2)
        self.cumulative_energy_ = np.cumsum(S ** 2) / total_energy * 100
        return self

    def transform(self, T: np.ndarray) -> np.ndarray:
        raw = (T - self.mean_) @ self.modes_.T
        return raw / self.singular_values_[: self.n_modes]

    def reconstruct(self, coeffs_norm: np.ndarray) -> np.ndarray:
        coeffs = coeffs_norm * self.singular_values_[: self.n_modes]
        return coeffs @ self.modes_ + self.mean_

    @property
    def energy_captured(self) -> float:
        return float(self.cumulative_energy_[self.n_modes - 1])


# ── Temporal feature engineering ───────────────────────────────────────────

def temporal_features(t_norm: np.ndarray) -> np.ndarray:
    """
    Build a multi-scale feature vector from normalised time t ∈ [0, 1].

    Features capture:
      - Linear growth         : t
      - Quadratic growth      : t²
      - Square-root (fast rise): √t
      - Logarithmic saturation: log(1 + 5t)
      - Exponential saturation: 1 - exp(-3t)
      - Sinusoidal bases      : sin(πt), cos(πt), sin(2πt)

    These multi-scale bases make it easy for a shallow MLP to represent
    the smooth, monotonic trajectories typical of thermal modal coefficients.
    """
    t = t_norm.reshape(-1, 1).astype(np.float32)
    feats = np.concatenate([
        t,
        t ** 2,
        np.sqrt(np.clip(t, 0, None)),
        np.log1p(5.0 * t),
        1.0 - np.exp(-3.0 * t),
        np.sin(np.pi * t),
        np.cos(np.pi * t),
        np.sin(2.0 * np.pi * t),
    ], axis=1)
    return feats.astype(np.float32)


# ── Model: time → modal coefficients MLP ──────────────────────────────────

def build_temporal_mlp(n_features: int, n_modes: int) -> Model:
    """
    MLP:  temporal_features(t)  →  [c₁, c₂, …, c_r]

    Small and well-regularised — the training set has only ~160 points so
    we favour generalisation over capacity.
    """
    inp = Input(shape=(n_features,), name="time_features")

    x = Dense(128, activation="tanh", name="fc1")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)
    x = Dense(128, activation="tanh", name="fc2")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)
    x = Dense(64,  activation="tanh", name="fc3")(x)
    x = BatchNormalization()(x)
    out = Dense(n_modes, name="coefficients")(x)

    model = Model(inp, out, name="TemporalMLP")

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=200,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-6,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_pod_energy(pod: PODDecomposition):
    n_show = min(30, len(pod.singular_values_))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    pct = pod.singular_values_[:n_show] ** 2 / np.sum(pod.singular_values_ ** 2) * 100
    axes[0].bar(range(1, n_show + 1), pct, color="steelblue", alpha=0.85)
    axes[0].axvline(pod.n_modes + 0.5, color="crimson", linestyle="--",
                    label=f"r = {pod.n_modes}")
    axes[0].set_title("Variance Explained per Mode", fontweight="bold")
    axes[0].set_xlabel("POD Mode"); axes[0].set_ylabel("Variance (%)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(1, n_show + 1), pod.cumulative_energy_[:n_show],
                 "o-", color="seagreen", markersize=4)
    axes[1].axhline(99, color="orange", linestyle="--", label="99% threshold")
    axes[1].axvline(pod.n_modes + 0.5, color="crimson", linestyle="--",
                    label=f"r = {pod.n_modes} → {pod.energy_captured:.3f}%")
    axes[1].set_title("Cumulative Energy Captured", fontweight="bold")
    axes[1].set_xlabel("Number of Modes"); axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_ylim(0, 101); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "1_pod_energy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 1_pod_energy.png")


def plot_modal_trajectories(
    coeffs_all: np.ndarray, pred_coeffs: np.ndarray,
    t_all_norm: np.ndarray, t_train_norm: np.ndarray,
    t_test_norm: np.ndarray, n_show: int = 4
):
    fig, axes = plt.subplots(n_show, 1, figsize=(13, 3 * n_show))
    n_train = len(t_train_norm)

    for i, ax in enumerate(axes):
        ax.plot(t_all_norm, coeffs_all[:, i], "b-", linewidth=2, label="True")
        ax.plot(t_test_norm, pred_coeffs[:, i], "r--", linewidth=2,
                label="Predicted (MLP extrapolation)")
        ax.axvline(t_train_norm[-1], color="gray", linestyle=":", linewidth=1.5,
                   label="Train / Test split")
        ax.set_title(f"Mode {i + 1} Coefficient", fontweight="bold")
        ax.set_xlabel("Normalised Time"); ax.set_ylabel("Amplitude")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle(
        "POD Modal Coefficients — True vs Direct MLP Prediction",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(OUT / "3_modal_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 3_modal_trajectories.png")


def plot_spatial_error(node_mae: np.ndarray, coords, title: str, fname: str):
    import pandas as pd
    df = pd.DataFrame({
        "X": coords["X_coordinate"].values,
        "Y": coords["Y_coordinate"].values,
        "MAE": node_mae,
    })
    pivot = df.pivot_table(index="Y", columns="X", values="MAE", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(
        pivot.values[::-1], cmap="YlOrRd", aspect="auto",
        extent=[pivot.columns.min(), pivot.columns.max(),
                pivot.index.min(), pivot.index.max()],
    )
    plt.colorbar(im, ax=ax, label="MAE (°C)")
    ax.set_title(f"{title} — Spatial Error Map", fontweight="bold")
    ax.set_xlabel("X Coordinate"); ax.set_ylabel("Y Coordinate")
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("ROM + Temporal-MLP Surrogate  (v2)")
    print("  POD compression -> direct t -> coefficients mapping")
    print("=" * 60)

    # ── 1. Load ────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    data = load_data()
    T_matrix, times, node_ids, coords = make_snapshot_matrix(data)
    T_train, T_test, t_train, t_test, n_train = temporal_split(T_matrix, times)
    print(f"  Train: {len(t_train)} steps ({t_train[0]:.0f}–{t_train[-1]:.0f}s)")
    print(f"  Test : {len(t_test)} steps ({t_test[0]:.0f}–{t_test[-1]:.0f}s)")

    t_all_norm   = (times   / T_SCALE).astype(np.float32)
    t_train_norm = (t_train / T_SCALE).astype(np.float32)
    t_test_norm  = (t_test  / T_SCALE).astype(np.float32)

    # ── 2. POD ─────────────────────────────────────────────────────────────
    print("\n[2/7] Running POD decomposition (fit on train only)...")
    pod = PODDecomposition(n_modes=N_MODES)
    pod.fit(T_train)
    print(f"  {N_MODES} modes capture {pod.energy_captured:.6f}% of variance")
    plot_pod_energy(pod)

    coeffs_train = pod.transform(T_train)   # (n_train, r)
    coeffs_test  = pod.transform(T_test)    # (n_test,  r)
    coeffs_all   = pod.transform(T_matrix)  # (n_total, r)

    T_pod_train = pod.reconstruct(coeffs_train)
    pod_rmse = float(np.sqrt(np.mean((T_train - T_pod_train) ** 2)))
    print(f"  POD train reconstruction RMSE: {pod_rmse:.4f}°C")

    # ── 3. Build temporal features ─────────────────────────────────────────
    print("\n[3/7] Building temporal feature matrix...")
    F_train = temporal_features(t_train_norm)   # (n_train, n_features)
    F_test  = temporal_features(t_test_norm)    # (n_test,  n_features)
    n_features = F_train.shape[1]
    print(f"  Feature vector size: {n_features}  "
          f"[t, t^2, sqrt(t), log(1+5t), 1-exp(-3t), sin(pi*t), cos(pi*t), sin(2pi*t)]")

    # ── 4. Train ───────────────────────────────────────────────────────────
    print("\n[4/7] Training temporal MLP...")
    model = build_temporal_mlp(n_features, N_MODES)
    model.summary(print_fn=lambda s: print("  " + s))

    t0 = time.time()
    history = model.fit(
        F_train, coeffs_train.astype(np.float32),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor="loss", patience=PATIENCE,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="loss", factor=0.5,
                              patience=30, min_lr=1e-7, verbose=0),
        ],
        verbose=0,
    )
    train_time = time.time() - t0
    n_params = model.count_params()
    best_train = min(history.history["loss"])
    print(f"  Done — {len(history.history['loss'])} epochs | "
          f"best train_loss={best_train:.6f} | {train_time:.1f}s | {n_params:,} params")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(history.history["loss"], label="Train MSE")
    ax.set_title("ROM+MLP v2 Training Loss (log scale)", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (log)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "2_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 2_training_loss.png")

    # ── 5. Predict test period ─────────────────────────────────────────────
    print("\n[5/7] Predicting test modal coefficients (direct, no autoregression)...")
    t0 = time.time()
    pred_coeffs = model.predict(F_test, verbose=0)   # (n_test, n_modes)
    infer_time  = time.time() - t0

    plot_modal_trajectories(
        coeffs_all, pred_coeffs,
        t_all_norm, t_train_norm, t_test_norm,
    )

    # ── 6. Reconstruct & evaluate ──────────────────────────────────────────
    print("\n[6/7] Reconstructing temperature field...")
    T_pred   = pod.reconstruct(pred_coeffs)
    metrics  = compute_metrics(T_test, T_pred)
    node_mae = per_node_metrics(T_test, T_pred)

    print(f"\n  Test metrics (temporal holdout, direct MLP prediction):")
    print(f"  R2   : {metrics['R2']:.6f}")
    print(f"  RMSE : {metrics['RMSE']:.4f} °C")
    print(f"  MAE  : {metrics['MAE']:.4f} °C")
    print(f"  Infer: {infer_time:.4f}s for {len(t_test)} steps")

    # ── 7. Plots ───────────────────────────────────────────────────────────
    print("\n[7/7] Generating plots...")
    plot_field_comparison(
        T_test, T_pred, coords, t_test,
        snapshot_indices=[0, len(t_test) // 2, len(t_test) - 1],
        model_name="ROM+MLP v2",
        save_path=OUT / "4_field_comparison.png",
    )
    plot_spatial_error(node_mae, coords, "ROM+MLP v2", "5_spatial_error.png")

    err_time = np.mean(np.abs(T_test - T_pred), axis=1)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t_test, err_time, color="crimson", linewidth=2)
    ax.fill_between(t_test, err_time, alpha=0.2, color="crimson")
    ax.set_title("ROM+MLP v2 — MAE Over Test Period", fontweight="bold")
    ax.set_xlabel("Simulation Time (s)"); ax.set_ylabel("MAE (°C)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "6_error_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 6_error_over_time.png")

    # ── Save ───────────────────────────────────────────────────────────────
    results = {
        "model":          "ROM+LSTM",
        "n_modes":        N_MODES,
        "energy_pct":     pod.energy_captured,
        "pod_train_rmse": pod_rmse,
        "n_params":       n_params,
        "train_time_s":   train_time,
        "infer_time_s":   infer_time,
        **metrics,
    }
    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    np.save(OUT / "T_test_true.npy",  T_test)
    np.save(OUT / "T_test_pred.npy",  T_pred)
    np.save(OUT / "times_test.npy",   t_test)

    print(f"\nAll outputs saved to: {OUT}")
    return results, T_test, T_pred, t_test, coords


if __name__ == "__main__":
    main()
