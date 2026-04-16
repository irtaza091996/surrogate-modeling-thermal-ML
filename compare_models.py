"""
Model Comparison
================
Runs ROM+LSTM and PINN sequentially, then generates a side-by-side
comparison with a summary table and overlay plots.

Run:
    python compare_models.py
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import RESULTS_DIR, compute_metrics

OUT = RESULTS_DIR / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

BASELINE = {
    "model":        "Baseline LSTM",
    "R2":           0.999998,
    "RMSE":         0.1730,
    "MAE":          0.1363,
    "n_params":     None,
    "train_time_s": None,
    "note":         "Random split (data leakage - invalid)",
}


def load_or_run(module_name: str):
    """Import and run a model's main(), or load cached results if available."""
    out_dir = RESULTS_DIR / module_name.replace("_", "-") if module_name != "rom_lstm" \
              else RESULTS_DIR / "rom_lstm"
    metrics_path = RESULTS_DIR / module_name / "metrics.json"

    if metrics_path.exists():
        print(f"  Loading cached results for {module_name}...")
        with open(metrics_path) as f:
            metrics = json.load(f)
        if "note" not in metrics:
            metrics["note"] = "Temporal holdout (90% split)"
        T_true = np.load(RESULTS_DIR / module_name / "T_test_true.npy")
        T_pred = np.load(RESULTS_DIR / module_name / "T_test_pred.npy")
        times  = np.load(RESULTS_DIR / module_name / "times_test.npy")
        return metrics, T_true, T_pred, times
    else:
        import importlib
        mod = importlib.import_module(module_name)
        results, T_true, T_pred, times, _ = mod.main()
        return results, T_true, T_pred, times


def plot_summary_table(results: list[dict]):
    """Render a clean metrics table as an image."""
    rows = []
    for r in results:
        rows.append({
            "Model":       r["model"],
            "R²":          f"{r['R2']:.6f}",
            "RMSE (°C)":   f"{r['RMSE']:.4f}",
            "MAE (°C)":    f"{r['MAE']:.4f}",
            "Params":      f"{r['n_params']:,}" if r.get("n_params") else "—",
            "Train (s)":   f"{r['train_time_s']:.1f}" if r.get("train_time_s") else "—",
            "Eval split":  r.get("note", "Temporal holdout"),
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.8 + 1.5))
    ax.axis("off")

    col_widths = [0.18, 0.12, 0.12, 0.12, 0.1, 0.1, 0.26]
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Header styling
    for j in range(len(df.columns)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Row colours
    colors = ["#eaf4fb", "#fdfefe", "#eafaf1"]
    for i in range(1, len(rows) + 1):
        for j in range(len(df.columns)):
            table[i, j].set_facecolor(colors[(i - 1) % len(colors)])

    ax.set_title(
        "Surrogate Model Comparison -- Temporal Holdout Evaluation\n"
        "(ROM+MLP & PINN: train t=0-1800s, test t=1800-2000s  |  90% split)",
        fontsize=12, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    plt.savefig(OUT / "1_metrics_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 1_metrics_table.png")


def plot_bar_comparison(results: list[dict]):
    metrics = ["R2", "RMSE", "MAE"]
    titles  = ["R²  (higher = better)", "RMSE (°C)  (lower = better)", "MAE (°C)  (lower = better)"]
    colors  = ["#3498db", "#e67e22", "#2ecc71"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = [r["model"] for r in results]

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        vals = [r[metric] for r in results]
        bars = ax.bar(names, vals, color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
            )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "2_bar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 2_bar_comparison.png")


def plot_error_over_time_comparison(model_data: list[tuple]):
    """
    model_data: list of (label, color, T_true, T_pred, times)
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    for label, color, T_true, T_pred, times in model_data:
        err = np.mean(np.abs(T_true - T_pred), axis=1)
        ax.plot(times, err, linewidth=2.5, label=label, color=color)
        ax.fill_between(times, err, alpha=0.12, color=color)

    ax.set_title(
        "MAE Over Test Period — All Models\n"
        "(test window: last 20% of simulation time)",
        fontweight="bold",
    )
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("MAE (°C)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "3_error_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 3_error_over_time.png")


def plot_field_overlay(model_data: list[tuple], snapshot_idx: int = -1):
    """
    Plot FEM truth + all model predictions at one time step, side by side.
    """
    from utils import load_data, make_snapshot_matrix, temporal_split

    data = load_data()
    T_matrix, times, _, coords = make_snapshot_matrix(data)
    _, T_test, _, t_test, _ = temporal_split(T_matrix, times)

    t_snap = t_test[snapshot_idx]
    T_true_snap = T_test[snapshot_idx]

    n_models = len(model_data)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 6))

    vmin, vmax = float(T_test.min()), float(T_test.max())

    def draw(ax, T_snap, title):
        df = pd.DataFrame({
            "X": coords["X_coordinate"].values,
            "Y": coords["Y_coordinate"].values,
            "T": T_snap,
        })
        pivot = df.pivot_table(index="Y", columns="X", values="T", aggfunc="mean")
        im = ax.imshow(
            pivot.values[::-1], cmap="coolwarm",
            vmin=vmin, vmax=vmax, aspect="auto",
            extent=[pivot.columns.min(), pivot.columns.max(),
                    pivot.index.min(), pivot.index.max()],
        )
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        return im

    im = draw(axes[0], T_true_snap, f"FEM Actual\nt = {t_snap:.0f}s")
    for ax, (label, color, T_true, T_pred, times) in zip(axes[1:], model_data):
        # Find closest time index in this model's test data
        idx = np.argmin(np.abs(times - t_snap))
        draw(ax, T_pred[idx], f"{label}\nt = {times[idx]:.0f}s")

    plt.colorbar(im, ax=axes[-1], label="°C", fraction=0.046)
    plt.suptitle(
        "Temperature Field Snapshot — FEM vs Surrogates",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(OUT / "4_field_snapshot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 4_field_snapshot.png")


def print_summary(results: list[dict]):
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<22} {'R2':>10} {'RMSE (C)':>10} {'MAE (C)':>10} {'Params':>10}")
    print("-" * 70)
    for r in results:
        params = f"{r['n_params']:,}" if r.get("n_params") else "—"
        print(
            f"{r['model']:<22} {r['R2']:>10.6f} {r['RMSE']:>10.4f} "
            f"{r['MAE']:>10.4f} {params:>10}"
        )
    print("=" * 70)

    # Best model per metric
    valid = [r for r in results if r["model"] != "Baseline LSTM"]
    best_r2   = max(valid, key=lambda r: r["R2"])
    best_rmse = min(valid, key=lambda r: r["RMSE"])
    best_mae  = min(valid, key=lambda r: r["MAE"])
    print(f"\n  Best R2   : {best_r2['model']}  ({best_r2['R2']:.6f})")
    print(f"  Best RMSE : {best_rmse['model']}  ({best_rmse['RMSE']:.4f} C)")
    print(f"  Best MAE  : {best_mae['model']}  ({best_mae['MAE']:.4f} C)")
    print("=" * 70)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("Running model comparison")
    print("=" * 60)

    # ── Run / load models ──────────────────────────────────────────────────
    print("\n--- ROM + LSTM ---")
    rom_results, T_true_rom, T_pred_rom, t_rom = load_or_run("rom_lstm")

    print("\n--- PINN ---")
    pinn_results, T_true_pinn, T_pred_pinn, t_pinn = load_or_run("pinn")

    # ── Combine results ────────────────────────────────────────────────────
    all_results = [BASELINE, rom_results, pinn_results]
    print_summary(all_results)

    # ── Comparison plots ───────────────────────────────────────────────────
    print("\nGenerating comparison plots...")
    plot_summary_table(all_results)
    plot_bar_comparison(all_results)

    model_data = [
        ("ROM+LSTM", "#e74c3c", T_true_rom,  T_pred_rom,  t_rom),
        ("PINN",     "#2ecc71", T_true_pinn, T_pred_pinn, t_pinn),
    ]
    plot_error_over_time_comparison(model_data)
    plot_field_overlay(model_data, snapshot_idx=-1)

    print(f"\nAll comparison plots saved to: {OUT}")


if __name__ == "__main__":
    main()
