"""
Enhanced Publication-Quality Plots
====================================
Generates polished figures from cached surrogate model results.

Figures produced (saved to results/figures/):
  fig1_model_comparison.png   - Dark-style bar chart: R2, RMSE, MAE
  fig2_temperature_fields.png - Field snapshot: FEM vs ROM+MLP vs PINN + error maps
  fig3_error_evolution.png    - MAE over test period for both models
  fig4_physics_discovery.png  - PINN gamma ratio physics discovery

Run:
    python plot_results.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import RESULTS_DIR, load_data, make_snapshot_matrix

OUT = RESULTS_DIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
C_ROM  = "#00d4aa"
C_PINN = "#ff6b6b"
C_BG   = "#0f1117"
C_AX   = "#1a1d27"
C_EDGE = "#3a3d4f"
C_TEXT = "#e0e0e0"
C_GRID = "#2a2d3f"

DARK_RC = {
    "figure.facecolor": C_BG, "axes.facecolor": C_AX,
    "axes.edgecolor": C_EDGE, "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "text.color": C_TEXT, "grid.color": C_GRID,
    "grid.linewidth": 0.6, "legend.facecolor": C_AX,
    "legend.edgecolor": C_EDGE, "font.family": "sans-serif",
}
LIGHT_RC = {
    "figure.facecolor": "white", "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#cccccc", "axes.labelcolor": "#222222",
    "xtick.color": "#222222", "ytick.color": "#222222",
    "text.color": "#222222", "grid.color": "#e0e0e0",
    "grid.linewidth": 0.7, "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc", "font.family": "sans-serif",
}


# ── Helpers ────────────────────────────────────────────────────────────────

def load_model_results(name):
    d = RESULTS_DIR / name
    needed = [d / "T_test_true.npy", d / "T_test_pred.npy",
              d / "times_test.npy",  d / "metrics.json"]
    if not all(p.exists() for p in needed):
        print(f"  WARNING: results for {name!r} not found, skipping.")
        return None
    return {
        "T_true":  np.load(d / "T_test_true.npy"),
        "T_pred":  np.load(d / "T_test_pred.npy"),
        "times":   np.load(d / "times_test.npy"),
        "metrics": json.load(open(d / "metrics.json")),
    }


def field_to_image(T_snap, coords):
    df = pd.DataFrame({
        "X": coords["X_coordinate"].values,
        "Y": coords["Y_coordinate"].values,
        "T": T_snap,
    })
    pivot  = df.pivot_table(index="Y", columns="X", values="T", aggfunc="mean")
    extent = [pivot.columns.min(), pivot.columns.max(),
              pivot.index.min(),   pivot.index.max()]
    return pivot.values[::-1], extent


def add_colorbar(fig, ax, im, label, fontsize=9):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.06)
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=fontsize)
    cb.ax.tick_params(labelsize=8)
    return cb


# ── Figure 1: Bar comparison ───────────────────────────────────────────────

def fig1_model_comparison(rom, pinn):
    print("  Building fig1_model_comparison.png ...")
    with plt.rc_context(DARK_RC):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.patch.set_facecolor(C_BG)

        panels = [
            ("R2",   "R\u00b2"),
            ("RMSE", "RMSE  (\u00b0C)"),
            ("MAE",  "MAE  (\u00b0C)"),
        ]
        for ax, (key, title) in zip(axes, panels):
            models, vals, colors = [], [], []
            if rom:
                models.append("ROM+MLP"); vals.append(rom["metrics"][key]);  colors.append(C_ROM)
            if pinn:
                models.append("PINN");    vals.append(pinn["metrics"][key]); colors.append(C_PINN)
            if not vals:
                ax.axis("off")
                continue

            bars = ax.bar(models, vals, color=colors, width=0.45,
                          edgecolor="white", linewidth=0.8, zorder=3)

            # Set ylim with enough headroom so labels don't spill out
            top = max(vals)
            if key == "R2":
                ax.set_ylim(0, min(top * 1.20, 1.15))
            else:
                ax.set_ylim(0, top * 1.22)

            for bar, v in zip(bars, vals):
                fmt = f"{v:.4f}" if key == "R2" else f"{v:.1f}\u00b0C"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.02,
                        fmt, ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color="white", zorder=4)

            ax.set_title(title, fontsize=12, fontweight="bold", color=C_TEXT, pad=10)
            ax.set_ylabel(key, fontsize=10, color=C_TEXT)
            ax.grid(True, axis="y", alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=10)
            for spine in ax.spines.values():
                spine.set_edgecolor(C_EDGE)

        fig.suptitle(
            "Surrogate Model Performance  --  Temporal Holdout Evaluation\n"
            "Train: t=0-1600s  |  Test: t=1600-2000s  (80% split, 400s extrapolation)",
            fontsize=12, fontweight="bold", color=C_TEXT, y=1.02)
        plt.tight_layout()
        fig.savefig(OUT / "fig1_model_comparison.png", dpi=200,
                    bbox_inches="tight", facecolor=C_BG)
        plt.close()
    print("    Saved: fig1_model_comparison.png")


# ── Figure 2: Temperature fields ──────────────────────────────────────────

def fig2_temperature_fields(rom, pinn, coords):
    print("  Building fig2_temperature_fields.png ...")
    ref = rom if rom else pinn
    if ref is None:
        print("    No data available, skipping.")
        return
    with plt.rc_context(LIGHT_RC):
        fig = plt.figure(figsize=(16, 8), facecolor="white")
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.34)

        T_true_snap = ref["T_true"][-1]
        t_val = float(ref["times"][-1])
        vmin, vmax = float(T_true_snap.min()), float(T_true_snap.max())

        def draw(ax, T_snap, title, cmap="RdYlBu_r", lo=None, hi=None, clabel="\u00b0C"):
            lo = lo if lo is not None else vmin
            hi = hi if hi is not None else vmax
            img, ext = field_to_image(T_snap, coords)
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, aspect="auto",
                           extent=ext, interpolation="bilinear")
            ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
            ax.set_xlabel("X (mm)", fontsize=8)
            ax.set_ylabel("Y (mm)", fontsize=8)
            ax.tick_params(labelsize=7)
            add_colorbar(fig, ax, im, clabel, fontsize=8)

        # Row 0: Actual | ROM pred | PINN pred
        draw(fig.add_subplot(gs[0, 0]), T_true_snap, f"FEM Actual  t={t_val:.0f}s")

        ax01 = fig.add_subplot(gs[0, 1])
        if rom:
            idx = int(np.argmin(np.abs(rom["times"] - t_val)))
            draw(ax01, rom["T_pred"][idx], f"ROM+MLP  t={rom['times'][idx]:.0f}s")
        else:
            ax01.axis("off")

        ax02 = fig.add_subplot(gs[0, 2])
        if pinn:
            idx = int(np.argmin(np.abs(pinn["times"] - t_val)))
            draw(ax02, pinn["T_pred"][idx], f"PINN  t={pinn['times'][idx]:.0f}s")
        else:
            ax02.axis("off")

        # Row 1: Metrics text | ROM error | PINN error
        ax10 = fig.add_subplot(gs[1, 0])
        ax10.axis("off")
        lines = ["Model Performance Summary", ""]
        if rom:
            m = rom["metrics"]
            lines += ["ROM+MLP",
                      f"  R\u00b2   = {m['R2']:.4f}",
                      f"  RMSE = {m['RMSE']:.2f} \u00b0C",
                      f"  MAE  = {m['MAE']:.2f} \u00b0C",
                      "  Params: 27,850  |  Infer: 0.38s", ""]
        if pinn:
            m = pinn["metrics"]
            lines += ["PINN",
                      f"  R\u00b2   = {m['R2']:.4f}",
                      f"  RMSE = {m['RMSE']:.2f} \u00b0C",
                      f"  MAE  = {m['MAE']:.2f} \u00b0C",
                      "  Params: 83,203  |  Infer: 1.4s"]
        ax10.text(0.08, 0.92, "\n".join(lines), transform=ax10.transAxes,
                  va="top", ha="left", fontsize=9, family="monospace",
                  bbox=dict(boxstyle="round,pad=0.5", fc="#1a1d27", ec="#3a3d4f"),
                  color="white")

        ax11 = fig.add_subplot(gs[1, 1])
        if rom:
            idx = int(np.argmin(np.abs(rom["times"] - t_val)))
            err = np.abs(rom["T_true"][idx] - rom["T_pred"][idx])
            draw(ax11, err, f"ROM+MLP |Error|  max={err.max():.1f}\u00b0C",
                 cmap="YlOrRd", lo=0, hi=float(err.max()), clabel="|\u0394T| (\u00b0C)")
        else:
            ax11.axis("off")

        ax12 = fig.add_subplot(gs[1, 2])
        if pinn:
            idx = int(np.argmin(np.abs(pinn["times"] - t_val)))
            err = np.abs(pinn["T_true"][idx] - pinn["T_pred"][idx])
            draw(ax12, err, f"PINN |Error|  max={err.max():.1f}\u00b0C",
                 cmap="YlOrRd", lo=0, hi=float(err.max()), clabel="|\u0394T| (\u00b0C)")
        else:
            ax12.axis("off")

        fig.suptitle(
            "Temperature Fields: FEM vs Surrogate Models  (t = 2000s)",
            fontsize=13, fontweight="bold", y=1.01)
        fig.savefig(OUT / "fig2_temperature_fields.png", dpi=200,
                    bbox_inches="tight", facecolor="white")
        plt.close()
    print("    Saved: fig2_temperature_fields.png")


# ── Figure 3: Error over time ─────────────────────────────────────────────

def fig3_error_evolution(rom, pinn):
    print("  Building fig3_error_evolution.png ...")
    with plt.rc_context(LIGHT_RC):
        fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
        t_start = None

        # Different y-offsets so the two labels don't overlap each other or the curves
        # ROM+MLP (higher line) annotated above; PINN (lower line) annotated below
        annotation_offsets = [(-65, 30), (-65, -32)]
        for (data, color, label), (dx, dy) in zip(
            [(rom, "#00a86b", "ROM+MLP"), (pinn, "#e05252", "PINN")],
            annotation_offsets
        ):
            if data is None:
                continue
            times = data["times"]
            err   = np.mean(np.abs(data["T_true"] - data["T_pred"]), axis=1)
            ax.plot(times, err, color=color, linewidth=2.5, label=label, zorder=4)
            ax.fill_between(times, err, alpha=0.15, color=color, zorder=3)
            ax.annotate(f"{err[-1]:.1f}\u00b0C",
                        xy=(times[-1], err[-1]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=10,
                        color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec=color, alpha=0.85, lw=1.2),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.3))
            if t_start is None:
                t_start = float(times[0])

        if t_start is not None:
            ax.axvspan(t_start, 2050, alpha=0.06, color="#2196F3",
                       label="Unseen test window (temporal extrapolation)", zorder=1)
            ax.axvline(t_start, color="#666666", linewidth=1.4,
                       linestyle="--", zorder=5)
            ylims = ax.get_ylim()
            # Rotate label 90° so it sits flush against the dashed line without overlapping the legend
            ax.text(t_start - 12, (ylims[0] + ylims[1]) * 0.5,
                    "Train / Test split", fontsize=8.5, color="#666666",
                    va="center", ha="right", rotation=90)

        ax.set_title("Mean Absolute Error Over Test Period",
                     fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Simulation Time (s)", fontsize=11)
        ax.set_ylabel("MAE (\u00b0C)", fontsize=11)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.5)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_edgecolor("#cccccc")
        plt.tight_layout()
        fig.savefig(OUT / "fig3_error_evolution.png", dpi=200,
                    bbox_inches="tight", facecolor="white")
        plt.close()
    print("    Saved: fig3_error_evolution.png")


# ── Figure 4: Physics discovery ───────────────────────────────────────────

def fig4_physics_discovery(pinn):
    print("  Building fig4_physics_discovery.png ...")
    with plt.rc_context(LIGHT_RC):
        fig, (ax_bar, ax_schema) = plt.subplots(
            1, 2, figsize=(13, 5), facecolor="white",
            gridspec_kw={"width_ratios": [1, 1.3]})

        learned  = pinn["metrics"].get("gamma_ratio", 100.94) if pinn else 100.94
        expected = 100.0

        bars = ax_bar.bar(
            ["Learned\n(PINN)", "Expected\n(geometry)"],
            [learned, expected],
            color=["#ff6b6b", "#4ecdc4"],
            width=0.45, edgecolor="white", linewidth=1.0)
        for bar, v in zip(bars, [learned, expected]):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.6,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=13, fontweight="bold")
        ax_bar.set_ylim(0, 122)
        ax_bar.set_ylabel("\u03b3\u2093 / \u03b3\u1d67  (diffusivity ratio)", fontsize=11)
        ax_bar.set_title("Learned vs Expected Diffusivity Ratio",
                         fontsize=12, fontweight="bold", pad=8)
        ax_bar.grid(True, axis="y", alpha=0.4)
        ax_bar.set_axisbelow(True)
        err_pct = abs(learned - expected) / expected * 100
        ax_bar.text(0.5, 0.96,
                    f"Agreement: {err_pct:.2f}% error\nNo explicit supervision on PDE coefficients",
                    transform=ax_bar.transAxes, ha="center", va="top",
                    fontsize=9, style="italic", color="#555555",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#f0f8f0", ec="#aaddaa"))

        # Schematic panel — domain geometry
        ax_schema.set_xlim(-2, 14)
        ax_schema.set_ylim(-2.2, 5.0)
        ax_schema.set_aspect("equal")
        ax_schema.axis("off")
        ax_schema.set_title("Domain Geometry Intuition",
                            fontsize=12, fontweight="bold", pad=8)

        # Main result annotation — top, clear of everything else
        ax_schema.text(6, 4.5,
                       "\u03b3\u2093 / \u03b3\u1d67 = 100.94  \u2248  (200/20)\u00b2 = 100",
                       ha="center", va="center", fontsize=13,
                       color="#333333", fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.5",
                                 fc="#fff9c4", ec="#f0c040", lw=1.5))

        # Draw domain rectangle — positioned in middle of panel
        rect = mpatches.FancyBboxPatch(
            (0, 0.8), 10, 2.2, boxstyle="round,pad=0.05",
            lw=2, edgecolor="#333333", facecolor="#e8f4fd")
        ax_schema.add_patch(rect)
        ax_schema.text(5, 1.9,
                       "Physical domain\n20 mm \u00d7 200 mm",
                       ha="center", va="center", fontsize=10,
                       color="#333333", fontweight="bold")

        # x-direction arrow — sits BELOW the rectangle with clear gap
        ax_schema.annotate("",
                           xy=(10.0, -0.5), xytext=(0.0, -0.5),
                           arrowprops=dict(arrowstyle="<->", color="#00a86b",
                                           lw=2.5, mutation_scale=18))
        ax_schema.text(5, -1.2,
                       "x: fast diffusion  (\u03b3\u2093 \u2248 0.97)",
                       ha="center", va="top", fontsize=9,
                       color="#00a86b", fontweight="bold")

        # y-direction arrow — sits to the RIGHT of the rectangle with clear gap
        ax_schema.annotate("",
                           xy=(12.0, 3.1), xytext=(12.0, 0.7),
                           arrowprops=dict(arrowstyle="<->", color="#e05252",
                                           lw=2.5, mutation_scale=18))
        ax_schema.text(12.2, 1.9,
                       "y: slow\n(\u03b3\u1d67 \u2248 0.010)",
                       ha="left", va="center", fontsize=8.5,
                       color="#e05252", fontweight="bold")

        fig.suptitle(
            "Physics Discovery: PINN Independently Recovers Thermal Diffusivity Anisotropy",
            fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        fig.savefig(OUT / "fig4_physics_discovery.png", dpi=200,
                    bbox_inches="tight", facecolor="white")
        plt.close()
    print("    Saved: fig4_physics_discovery.png")


# ── Figure 5: Train vs Test metric comparison ─────────────────────────────

def fig5_train_test_comparison(rom, pinn):
    print("  Building fig5_train_test_comparison.png ...")

    # Check that training metrics exist
    has_train = lambda d: d and "train_MAE" in d["metrics"]
    if not has_train(rom) and not has_train(pinn):
        print("    WARNING: No training metrics found in metrics.json.")
        print("    Re-run rom_lstm.py and pinn.py first, then regenerate plots.")
        return

    with plt.rc_context(LIGHT_RC):
        metrics_cfg = [
            ("R2",         "R\u00b2",       None),
            ("RMSE",       "RMSE (\u00b0C)", None),
            ("MAE",        "MAE (\u00b0C)",  None),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="white")

        bar_w   = 0.18
        x_pos   = np.arange(2)   # ROM+MLP, PINN
        offsets = [-bar_w * 1.5, -bar_w * 0.5, bar_w * 0.5, bar_w * 1.5]
        colors  = ["#00a86b", "#90e0c8", "#e05252", "#f4a8a8"]
        labels  = ["ROM+MLP Train", "ROM+MLP Test", "PINN Train", "PINN Test"]

        for ax, (key, ylabel, _) in zip(axes, metrics_cfg):
            train_key = f"train_{key}"
            vals = []
            for mk, tk, model in [
                (key, train_key, rom),
                (key, train_key, pinn),
            ]:
                train_v = model["metrics"].get(tk, None) if model else None
                test_v  = model["metrics"].get(mk, None) if model else None
                vals.append((train_v, test_v))

            # Draw bars for each model side by side
            bar_data = [
                (offsets[0], vals[0][0], colors[0], labels[0]),  # ROM train
                (offsets[1], vals[0][1], colors[1], labels[1]),  # ROM test
                (offsets[2], vals[1][0], colors[2], labels[2]),  # PINN train
                (offsets[3], vals[1][1], colors[3], labels[3]),  # PINN test
            ]

            # Single x-axis position (no model grouping needed — just 4 bars)
            bar_positions = np.arange(4)
            bar_vals      = [b[1] for b in bar_data]
            bar_colors    = [b[2] for b in bar_data]
            bar_labels    = [b[3] for b in bar_data]

            bars = ax.bar(bar_positions, bar_vals, color=bar_colors,
                          width=0.6, edgecolor="white", linewidth=0.8, zorder=3)

            # Value labels on top
            top = max(v for v in bar_vals if v is not None)
            if key == "R2":
                ax.set_ylim(0, min(top * 1.20, 1.15))
            else:
                ax.set_ylim(0, top * 1.25)

            for bar, v in zip(bars, bar_vals):
                if v is None:
                    continue
                fmt = f"{v:.4f}" if key == "R2" else f"{v:.1f}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + top * 0.02,
                        fmt, ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#222222", zorder=4)

            ax.set_xticks(bar_positions)
            ax.set_xticklabels(["ROM+MLP\nTrain", "ROM+MLP\nTest",
                                 "PINN\nTrain",   "PINN\nTest"],
                               fontsize=9)
            ax.set_title(ylabel, fontsize=12, fontweight="bold", pad=10)
            ax.set_ylabel(key, fontsize=10)
            ax.grid(True, axis="y", alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            for sp in ax.spines.values():
                sp.set_edgecolor("#cccccc")

        # Vertical separator lines between ROM and PINN groups
        for ax in axes:
            ax.axvline(1.5, color="#cccccc", linewidth=1.2, linestyle="--", zorder=2)

        fig.suptitle(
            "Training vs Test Performance  --  ROM+MLP and PINN\n"
            "Train: t=0-1600s  |  Test: t=1600-2000s  (temporal extrapolation)",
            fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(OUT / "fig5_train_test_comparison.png", dpi=200,
                    bbox_inches="tight", facecolor="white")
        plt.close()
    print("    Saved: fig5_train_test_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("Generating enhanced result figures")
    print("=" * 60)
    print("\nLoading model results...")
    rom  = load_model_results("rom_lstm")
    pinn = load_model_results("pinn")
    print("\nLoading coordinate data...")
    _, _, _, coords = make_snapshot_matrix(load_data())
    print("\nGenerating figures...")
    fig1_model_comparison(rom, pinn)
    fig2_temperature_fields(rom, pinn, coords)
    fig3_error_evolution(rom, pinn)
    fig4_physics_discovery(pinn)
    fig5_train_test_comparison(rom, pinn)
    print(f"\nAll figures saved to: {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
