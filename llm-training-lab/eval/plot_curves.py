"""
Plot training/validation loss curves from MLflow for all 3 experiments.

Reads directly from the mlruns/ directory (no MLflow server needed).
Saves the plot to outputs/loss_curves.png.
"""

import os
import json
import glob

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving to file

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def read_mlflow_metric(run_path, metric_name):
    """Read a metric's history from MLflow's file-based store."""
    metric_file = os.path.join(run_path, "metrics", metric_name)
    if not os.path.exists(metric_file):
        return [], []

    steps = []
    values = []
    with open(metric_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                # Format: timestamp value step
                values.append(float(parts[1]))
                steps.append(int(parts[2]))

    return steps, values


def read_mlflow_param(run_path, param_name):
    """Read a parameter from MLflow's file-based store."""
    param_file = os.path.join(run_path, "params", param_name)
    if not os.path.exists(param_file):
        return None
    with open(param_file, "r") as f:
        return f.read().strip()


def find_run_path(experiment_name):
    """Find the run directory for a given experiment name."""
    # Scan experiment directories
    for exp_dir in glob.glob(os.path.join(MLRUNS_DIR, "*")):
        meta_file = os.path.join(exp_dir, "meta.yaml")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                content = f.read()
                if f"name: {experiment_name}" in content:
                    # Found the experiment, now find runs
                    for run_dir in glob.glob(os.path.join(exp_dir, "*")):
                        if os.path.isdir(run_dir) and os.path.exists(
                            os.path.join(run_dir, "metrics")
                        ):
                            return run_dir
    return None


def plot_curves():
    print("\n" + "=" * 60)
    print("Loss Curve Visualization")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("LLM Training Lab — Loss Curves Across All Phases", fontsize=14, fontweight="bold")

    summary = []

    # ─── Phase 1: Pretrain ───────────────────────────────────
    print("\nReading Phase 1 (pretrain_shakespeare) metrics...")
    run_path = find_run_path("pretrain_shakespeare")
    if run_path:
        train_steps, train_loss = read_mlflow_metric(run_path, "train_loss")
        val_steps, val_loss = read_mlflow_metric(run_path, "val_loss")

        ax = axes[0]
        if train_steps:
            ax.plot(train_steps, train_loss, label="Train Loss", color="#2196F3", alpha=0.8)
        if val_steps:
            ax.plot(val_steps, val_loss, label="Val Loss", color="#FF5722", alpha=0.8, linestyle="--")
        ax.set_title("Phase 1: Pre-Training (NanoGPT)\n10.8M params, Shakespeare", fontsize=11)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        final_train = train_loss[-1] if train_loss else None
        final_val = val_loss[-1] if val_loss else None
        summary.append(("pretrain_shakespeare", final_train, final_val))
        print(f"  Train loss: {final_train:.4f}" if final_train else "  No train loss found")
        print(f"  Val loss: {final_val:.4f}" if final_val else "  No val loss found")
    else:
        axes[0].text(0.5, 0.5, "No data found", ha="center", va="center", transform=axes[0].transAxes)
        print("  ⚠ Experiment not found in mlruns/")

    # ─── Phase 2: SFT ───────────────────────────────────────
    print("\nReading Phase 2 (sft_dolly) metrics...")
    run_path = find_run_path("sft_dolly")
    if run_path:
        train_steps, train_loss = read_mlflow_metric(run_path, "train_loss")
        val_steps, val_loss = read_mlflow_metric(run_path, "eval_loss")

        ax = axes[1]
        if train_steps:
            ax.plot(train_steps, train_loss, label="Train Loss", color="#2196F3", alpha=0.8)
        if val_steps:
            ax.plot(val_steps, val_loss, label="Eval Loss", color="#FF5722", alpha=0.8, linestyle="--")
        ax.set_title("Phase 2: SFT (SmolLM2-135M)\n135M params, Dolly-15k", fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        final_train = train_loss[-1] if train_loss else None
        final_val = val_loss[-1] if val_loss else None
        summary.append(("sft_dolly", final_train, final_val))
        print(f"  Train loss: {final_train:.4f}" if final_train else "  No train loss found")
        print(f"  Eval loss: {final_val:.4f}" if final_val else "  No eval loss found")
    else:
        axes[1].text(0.5, 0.5, "No data found", ha="center", va="center", transform=axes[1].transAxes)
        print("  ⚠ Experiment not found in mlruns/")

    # ─── Phase 3: LoRA ──────────────────────────────────────
    print("\nReading Phase 3 (lora_medical) metrics...")
    run_path = find_run_path("lora_medical")
    if run_path:
        train_steps, train_loss = read_mlflow_metric(run_path, "train_loss")
        val_steps, val_loss = read_mlflow_metric(run_path, "eval_loss")

        ax = axes[2]
        if train_steps:
            ax.plot(train_steps, train_loss, label="Train Loss", color="#2196F3", alpha=0.8)
        if val_steps:
            ax.plot(val_steps, val_loss, label="Eval Loss", color="#FF5722", alpha=0.8, linestyle="--")
        ax.set_title("Phase 3: LoRA (Medical Adapter)\n0.9M trainable / 135M total", fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add annotation for trainable params
        trainable = read_mlflow_param(run_path, "trainable_params")
        efficiency = read_mlflow_param(run_path, "param_efficiency_pct")
        if trainable and efficiency:
            ax.annotate(
                f"Trainable: {int(trainable):,} ({efficiency}%)",
                xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=8, color="gray",
            )

        final_train = train_loss[-1] if train_loss else None
        final_val = val_loss[-1] if val_loss else None
        summary.append(("lora_medical", final_train, final_val))
        print(f"  Train loss: {final_train:.4f}" if final_train else "  No train loss found")
        print(f"  Eval loss: {final_val:.4f}" if final_val else "  No eval loss found")
    else:
        axes[2].text(0.5, 0.5, "No data found", ha="center", va="center", transform=axes[2].transAxes)
        print("  ⚠ Experiment not found in mlruns/")

    # ─── Save ────────────────────────────────────────────────
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "loss_curves.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Plot saved to {output_path}")

    # ─── Summary table ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"{'Experiment':<25} {'Final Train Loss':>17} {'Final Val/Eval Loss':>20}")
    print(f"{'─' * 25} {'─' * 17} {'─' * 20}")
    for name, train, val in summary:
        t = f"{train:.4f}" if train else "N/A"
        v = f"{val:.4f}" if val else "N/A"
        print(f"{name:<25} {t:>17} {v:>20}")

    print(f"\n✅ Loss curves saved to {output_path}")


if __name__ == "__main__":
    plot_curves()
