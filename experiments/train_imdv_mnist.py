"""Train GeodesicIMDVv2 dictionary on MNIST — true online (sample-by-sample).

This is a standalone experiment script, not a node-graph component.

Matches GeodesicIMDVv2: identical learning to v1 (geodesic pull/push with
LOO shadow inhibition), but tracks and outputs LOO-inhibited activations
alongside raw activations. The shadow overlap (w_i · x_shadow_hat_i)
measures dendrite-level redundancy and produces sparser soma-level output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from mnist import MNIST
from tqdm import tqdm

try:
    import cupy as xp

    USING_CUPY = True
except Exception:
    import numpy as xp  # type: ignore[no-redef]

    USING_CUPY = False


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "experiments" / "r002"

# Dataset / run
SEED = 42
NUM_EPOCHS = 10
MAX_TRAIN_SAMPLES = 5000  # Set to an int for quick experiments.
SHUFFLE_EACH_EPOCH = True
PRELOAD_DATASET_ON_DEVICE = True

# IMDV hyperparameters
TEMPLATE_COUNT = 100
INPUT_DIM = 28 * 28
LEARNING_RATE = 0.25  # Geodesic step fraction — matches node's step_fraction default.
BALANCE = 0.50
PROGRESS_EVERY = 500  # Update progress bar every N samples.
EPS = 1e-8

# Metrics
ACTIVITY_THRESHOLD = 1e-6
SAVE_FULL_OVERLAP_MATRIX = True


def _resolve_mnist_path() -> Path:
    candidates = [
        ROOT / "data" / "mnist" / "mnist",
        ROOT / "data" / "mnist_data",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "MNIST dataset not found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_mnist_train() -> np.ndarray:
    mnist_path = _resolve_mnist_path()
    images, labels = MNIST(str(mnist_path)).load_training()
    x = np.asarray(images, dtype=np.float32) / 255.0
    labels = np.asarray(labels)
    if MAX_TRAIN_SAMPLES is not None:
        rng = np.random.RandomState(SEED)
        unique_labels = np.unique(labels)
        per_class = MAX_TRAIN_SAMPLES // len(unique_labels)
        remainder = MAX_TRAIN_SAMPLES % len(unique_labels)
        indices = []
        for i, lbl in enumerate(unique_labels):
            class_idx = np.where(labels == lbl)[0]
            n = per_class + (1 if i < remainder else 0)
            chosen = rng.choice(class_idx, size=min(n, len(class_idx)), replace=False)
            indices.append(chosen)
        indices = np.concatenate(indices)
        rng.shuffle(indices)
        x = x[indices]
    return x


def _to_numpy(array):
    if USING_CUPY:
        return xp.asnumpy(array)
    return np.asarray(array)


def _normalize_rows(array, eps: float):
    norms = xp.linalg.norm(array, axis=1, keepdims=True)
    return array / (norms + eps), norms


def _hoyer_sparsity_scalar(a, eps: float) -> float:
    """Hoyer sparsity for a single 1-D activation vector (on device)."""
    n = a.shape[0]
    if n <= 1:
        return 0.0
    l1 = float(_to_numpy(xp.sum(a)))
    l2 = float(_to_numpy(xp.linalg.norm(a)))
    if l2 <= eps:
        return 0.0
    raw = (math.sqrt(n) - (l1 / (l2 + eps))) / (math.sqrt(n) - 1.0)
    return max(0.0, min(1.0, raw))


def _gini_sparsity_scalar(a, eps: float) -> float:
    """Gini sparsity for a single 1-D activation vector (on device)."""
    n = a.shape[0]
    if n <= 1:
        return 0.0
    sorted_a = xp.sort(a)
    index = xp.arange(1, n + 1, dtype=a.dtype)
    numer = float(_to_numpy(xp.sum((2.0 * index - n - 1.0) * sorted_a)))
    denom = float(n * _to_numpy(xp.sum(sorted_a))) + eps
    if denom <= eps:
        return 0.0
    return max(0.0, min(1.0, numer / denom))


def _initialize_weights(template_count: int, dim: int, seed: int):
    if USING_CUPY:
        xp.random.seed(seed)
    else:
        np.random.seed(seed)
    w = xp.random.standard_normal((template_count, dim)).astype(xp.float32)
    w, _ = _normalize_rows(w, EPS)
    return w


def _coactivation_overlap(
    template_counts: np.ndarray, coactivation_counts: np.ndarray
) -> float:
    min_counts = np.minimum.outer(template_counts, template_counts)
    valid = min_counts > 0
    off_diag = ~np.eye(template_counts.shape[0], dtype=bool)
    mask = valid & off_diag
    if not np.any(mask):
        return 0.0
    overlap = np.zeros_like(coactivation_counts, dtype=np.float64)
    overlap[mask] = coactivation_counts[mask] / min_counts[mask]
    return float(np.mean(overlap[mask]))


def _winner_entropy(winner_counts: np.ndarray) -> float:
    total = float(np.sum(winner_counts))
    if total <= 0.0:
        return 0.0
    probs = winner_counts / total
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _imdv_online_step(x, w, balance: float, learning_rate: float):
    """Process one sample and update weights — mirrors GeodesicIMDVv2.process().

    Returns (w_new, (a_raw, a_inhibited, mse)) or (w, None) if input is near-zero.
    """
    eps = EPS
    n_templates, dim = w.shape

    # 1. Normalize input
    x_norm = float(_to_numpy(xp.linalg.norm(x)))
    if x_norm < eps:
        return w, None
    x_hat = x / (x_norm + eps)

    # Ensure unit weights
    w_norms = xp.linalg.norm(w, axis=1, keepdims=True)
    w = w / (w_norms + eps)

    # 2. Similarities & activities
    s = w @ x_hat                       # (T,)
    a = xp.maximum(s, 0.0)             # (T,)
    A = xp.sum(a)                       # scalar

    # 3. Global reconstruction
    x_recon = a @ w                     # (D,)

    # 4. LOO shadow
    x_shadow = x_recon[None, :] - a[:, None] * w              # (T, D)
    x_shadow_norm = xp.linalg.norm(x_shadow, axis=1, keepdims=True)
    x_shadow_hat = x_shadow / (x_shadow_norm + eps)

    # Inhibition
    inh = (A - a) / (A + eps)          # (T,)

    # 5. Tangents
    # Pull: geodesic tangent w_i -> x_hat
    cos_pull = xp.sum(w * x_hat[None, :], axis=1, keepdims=True)  # = s reshaped
    tangent_pull = x_hat[None, :] - cos_pull * w
    norm_pull = xp.linalg.norm(tangent_pull, axis=1, keepdims=True)
    tau_pull = xp.where(norm_pull > eps, tangent_pull / (norm_pull + eps),
                        xp.zeros_like(w))

    # Push: geodesic tangent w_i -> x_shadow_hat_i
    cos_push = xp.sum(w * x_shadow_hat, axis=1, keepdims=True)
    tangent_push = x_shadow_hat - cos_push * w
    norm_push = xp.linalg.norm(tangent_push, axis=1, keepdims=True)
    tau_push = xp.where(norm_push > eps, tangent_push / (norm_push + eps),
                        xp.zeros_like(w))

    # 6. Random tangent for inactive templates with zero pull
    inactive_mask = (a <= 0.0)
    zero_pull_mask = (norm_pull[:, 0] < eps) & inactive_mask
    if bool(_to_numpy(xp.any(zero_pull_mask))):
        n_zero = int(_to_numpy(xp.sum(zero_pull_mask)))
        rand = xp.random.standard_normal((n_zero, dim)).astype(xp.float32)
        w_zero = w[zero_pull_mask]
        rand = rand - xp.sum(rand * w_zero, axis=1, keepdims=True) * w_zero
        rand_norm = xp.linalg.norm(rand, axis=1, keepdims=True)
        tau_pull[zero_pull_mask] = rand / (rand_norm + eps)

    # 7. Net tangent
    pull_weight = 1.0 - balance
    push_weight = balance
    tau_net = xp.where(
        a[:, None] > 0.0,
        pull_weight * tau_pull - (push_weight * inh[:, None]) * tau_push,
        pull_weight * tau_pull,
    )

    # 8. Geodesic rotation
    tau_net_norm = xp.linalg.norm(tau_net, axis=1, keepdims=True)
    rotation_axis = xp.where(tau_net_norm > eps, tau_net / (tau_net_norm + eps),
                             xp.zeros_like(w))
    angle = learning_rate * tau_net_norm
    w_new = w * xp.cos(angle) + rotation_axis * xp.sin(angle)

    # Renormalize
    w_new_norms = xp.linalg.norm(w_new, axis=1, keepdims=True)
    w_new = w_new / (w_new_norms + eps)

    # 9. LOO-inhibited activations (v2 output)
    shadow_overlap = cos_push[:, 0]     # w_i · x_shadow_hat_i
    a_inhibited = xp.maximum(s - shadow_overlap, 0.0)

    # Reconstruction MSE (in original input space)
    recon = x_recon * x_norm
    mse = float(_to_numpy(xp.sum((recon - x) ** 2))) / dim

    return w_new, (a, a_inhibited, mse)


def _format_device() -> str:
    if not USING_CUPY:
        return "numpy-cpu"
    device_id = int(xp.cuda.runtime.getDevice())
    props = xp.cuda.runtime.getDeviceProperties(device_id)
    name = props["name"].decode("utf-8")
    return f"cupy-gpu:{device_id}:{name}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train_cpu = _load_mnist_train()
    num_samples, dim = x_train_cpu.shape
    if dim != INPUT_DIM:
        raise ValueError(f"Expected input dim {INPUT_DIM}, got {dim}")

    if USING_CUPY and PRELOAD_DATASET_ON_DEVICE:
        x_train_device = xp.asarray(x_train_cpu, dtype=xp.float32)
    else:
        x_train_device = None

    weights = _initialize_weights(TEMPLATE_COUNT, INPUT_DIM, SEED)
    history: list[dict[str, float]] = []

    print(
        json.dumps(
            {
                "mode": "online",
                "device": _format_device(),
                "num_samples": num_samples,
                "num_epochs": NUM_EPOCHS,
                "template_count": TEMPLATE_COUNT,
                "learning_rate": LEARNING_RATE,
                "balance": BALANCE,
                "preload_dataset_on_device": PRELOAD_DATASET_ON_DEVICE,
            },
            indent=2,
        )
    )

    for epoch in range(NUM_EPOCHS):
        if SHUFFLE_EACH_EPOCH:
            order = np.random.permutation(num_samples)
        else:
            order = np.arange(num_samples)

        epoch_mse_sum = 0.0
        # Raw activation metrics
        epoch_raw_hoyer_sum = 0.0
        epoch_raw_gini_sum = 0.0
        epoch_raw_active_fraction_sum = 0.0
        epoch_raw_active_count_sum = 0.0
        # LOO-inhibited activation metrics
        epoch_inh_hoyer_sum = 0.0
        epoch_inh_gini_sum = 0.0
        epoch_inh_active_fraction_sum = 0.0
        epoch_inh_active_count_sum = 0.0
        epoch_valid = 0
        epoch_inh_template_counts = np.zeros(TEMPLATE_COUNT, dtype=np.float64)
        epoch_inh_coactivation_counts = np.zeros(
            (TEMPLATE_COUNT, TEMPLATE_COUNT), dtype=np.float64
        )
        epoch_inh_winner_counts = np.zeros(TEMPLATE_COUNT, dtype=np.float64)

        progress = tqdm(
            range(num_samples),
            desc=f"epoch {epoch + 1}/{NUM_EPOCHS}",
            leave=True,
            mininterval=0.5,
        )

        for i in progress:
            idx = order[i]
            if x_train_device is not None:
                x_sample = x_train_device[idx]
            else:
                x_sample = xp.asarray(x_train_cpu[idx], dtype=xp.float32)

            weights, result = _imdv_online_step(
                x_sample, weights, BALANCE, LEARNING_RATE
            )

            if result is None:
                continue

            a_raw, a_inh, mse = result
            epoch_valid += 1
            epoch_mse_sum += mse

            # Raw activation metrics
            epoch_raw_hoyer_sum += _hoyer_sparsity_scalar(a_raw, EPS)
            epoch_raw_gini_sum += _gini_sparsity_scalar(a_raw, EPS)
            raw_active = _to_numpy(a_raw) > ACTIVITY_THRESHOLD
            n_raw_active = int(np.sum(raw_active))
            epoch_raw_active_fraction_sum += n_raw_active / TEMPLATE_COUNT
            epoch_raw_active_count_sum += n_raw_active

            # LOO-inhibited activation metrics
            epoch_inh_hoyer_sum += _hoyer_sparsity_scalar(a_inh, EPS)
            epoch_inh_gini_sum += _gini_sparsity_scalar(a_inh, EPS)
            inh_active = _to_numpy(a_inh) > ACTIVITY_THRESHOLD
            n_inh_active = int(np.sum(inh_active))
            epoch_inh_active_fraction_sum += n_inh_active / TEMPLATE_COUNT
            epoch_inh_active_count_sum += n_inh_active

            # Template & coactivation counts (based on inhibited activations)
            epoch_inh_template_counts += inh_active.astype(np.float64)
            if n_inh_active > 0:
                active_idx = np.where(inh_active)[0]
                epoch_inh_coactivation_counts[np.ix_(active_idx, active_idx)] += 1.0
                winner = int(_to_numpy(xp.argmax(a_inh)))
                epoch_inh_winner_counts[winner] += 1.0

            if (i + 1) % PROGRESS_EVERY == 0:
                progress.set_postfix(
                    mse=f"{epoch_mse_sum / epoch_valid:.5f}",
                    inh_hoyer=f"{epoch_inh_hoyer_sum / epoch_valid:.4f}",
                    inh_frac=f"{epoch_inh_active_fraction_sum / epoch_valid:.4f}",
                )

        if epoch_valid == 0:
            continue

        inh_overlap = _coactivation_overlap(
            epoch_inh_template_counts, epoch_inh_coactivation_counts
        )
        epoch_record = {
            "epoch": float(epoch + 1),
            "recon_mse": epoch_mse_sum / epoch_valid,
            # Raw activation metrics
            "raw_hoyer_sparsity": epoch_raw_hoyer_sum / epoch_valid,
            "raw_gini_sparsity": epoch_raw_gini_sum / epoch_valid,
            "raw_active_fraction": epoch_raw_active_fraction_sum / epoch_valid,
            "raw_mean_active_count": epoch_raw_active_count_sum / epoch_valid,
            # LOO-inhibited activation metrics
            "inh_hoyer_sparsity": epoch_inh_hoyer_sum / epoch_valid,
            "inh_gini_sparsity": epoch_inh_gini_sum / epoch_valid,
            "inh_active_fraction": epoch_inh_active_fraction_sum / epoch_valid,
            "inh_mean_active_count": epoch_inh_active_count_sum / epoch_valid,
            "inh_coactivation_overlap": inh_overlap,
            "inh_winner_entropy": _winner_entropy(epoch_inh_winner_counts),
            "inh_dead_template_fraction": float(np.mean(epoch_inh_template_counts <= 0.0)),
        }
        history.append(epoch_record)

        print(
            f"epoch={epoch + 1} "
            f"recon_mse={epoch_record['recon_mse']:.6f} "
            f"raw[hoyer={epoch_record['raw_hoyer_sparsity']:.4f} "
            f"act_frac={epoch_record['raw_active_fraction']:.4f}] "
            f"inh[hoyer={epoch_record['inh_hoyer_sparsity']:.4f} "
            f"act_frac={epoch_record['inh_active_fraction']:.4f} "
            f"active={epoch_record['inh_mean_active_count']:.1f} "
            f"overlap={epoch_record['inh_coactivation_overlap']:.4f} "
            f"dead={epoch_record['inh_dead_template_fraction']:.4f}]"
        )

        weights_cpu = _to_numpy(weights)
        np.savez(
            OUTPUT_DIR / "imdv_mnist_latest.npz",
            weights=weights_cpu,
            winner_counts=epoch_inh_winner_counts,
            template_counts=epoch_inh_template_counts,
            coactivation_counts=epoch_inh_coactivation_counts
            if SAVE_FULL_OVERLAP_MATRIX
            else np.empty((0, 0), dtype=np.float32),
        )

        with (OUTPUT_DIR / "imdv_mnist_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    visualize(history, weights_cpu, epoch_inh_winner_counts, epoch_inh_template_counts,
              epoch_inh_coactivation_counts)


def visualize(history: list[dict], weights_cpu: np.ndarray,
              winner_counts: np.ndarray, template_counts: np.ndarray,
              coactivation_counts: np.ndarray):
    """Generate and save all training diagnostics as a single figure."""
    epochs = [r["epoch"] for r in history]

    fig = plt.figure(figsize=(20, 16), constrained_layout=True)
    fig.suptitle("GeodesicIMDVv2 — MNIST Training Diagnostics (Online)", fontsize=14, y=1.01)
    gs = GridSpec(3, 4, figure=fig)

    # --- Row 1: epoch-level line plots ---

    # 1. Reconstruction MSE
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, [r["recon_mse"] for r in history], "o-", color="tab:red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction MSE")

    # 2. Sparsity (Hoyer & Gini) — raw vs inhibited
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, [r["raw_hoyer_sparsity"] for r in history], "s--", color="tab:blue",
            alpha=0.4, label="Raw Hoyer")
    ax.plot(epochs, [r["raw_gini_sparsity"] for r in history], "^--", color="tab:orange",
            alpha=0.4, label="Raw Gini")
    ax.plot(epochs, [r["inh_hoyer_sparsity"] for r in history], "s-", color="tab:blue",
            label="Inh Hoyer")
    ax.plot(epochs, [r["inh_gini_sparsity"] for r in history], "^-", color="tab:orange",
            label="Inh Gini")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sparsity")
    ax.set_title("Activation Sparsity (raw vs inhibited)")
    ax.legend(fontsize=7)

    # 3. Active fraction & mean active count — raw vs inhibited
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(epochs, [r["raw_active_fraction"] for r in history], "o--", color="tab:green",
            alpha=0.4, label="Raw active frac")
    ax.plot(epochs, [r["inh_active_fraction"] for r in history], "o-", color="tab:green",
            label="Inh active frac")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction")
    ax.set_title("Active Fraction (raw vs inhibited)")
    ax2 = ax.twinx()
    ax2.plot(epochs, [r["raw_mean_active_count"] for r in history], "s--", color="tab:purple",
             alpha=0.4, label="Raw active count")
    ax2.plot(epochs, [r["inh_mean_active_count"] for r in history], "s-", color="tab:purple",
             label="Inh active count")
    ax2.set_ylabel("Count")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6)

    # 4. Coactivation overlap, winner entropy, dead fraction (inhibited)
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(epochs, [r["inh_coactivation_overlap"] for r in history], "o-", label="Coactivation overlap")
    ax.plot(epochs, [r["inh_dead_template_fraction"] for r in history], "s-", label="Dead fraction")
    ax.set_xlabel("Epoch")
    ax.set_title("Dictionary Health (inhibited)")
    ax.legend(fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(epochs, [r["inh_winner_entropy"] for r in history], "^--", color="tab:orange",
             label="Winner entropy")
    ax2.set_ylabel("Entropy (nats)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    # --- Row 2: weight grid (learned templates as 28x28 images) ---
    n_templates = weights_cpu.shape[0]
    grid_cols = int(math.ceil(math.sqrt(n_templates)))
    grid_rows = int(math.ceil(n_templates / grid_cols))
    img_h, img_w = 28, 28

    # Verify unit-sphere constraint
    w_norms = np.linalg.norm(weights_cpu, axis=1)
    print(f"Weight norms — min: {w_norms.min():.6f}, max: {w_norms.max():.6f}, "
          f"mean: {w_norms.mean():.6f}, std: {w_norms.std():.2e}")

    # Per-template normalization: scale each template to [0, 1] for display,
    # same approach as the node editor's to_display_grid.
    canvas = np.full((grid_rows * img_h, grid_cols * img_w), np.nan)
    for idx in range(n_templates):
        r, c = divmod(idx, grid_cols)
        tile = weights_cpu[idx].reshape(img_h, img_w)
        t_min, t_max = tile.min(), tile.max()
        if t_max - t_min > 1e-12:
            tile = (tile - t_min) / (t_max - t_min)
        else:
            tile = np.full_like(tile, 0.5)
        canvas[r * img_h:(r + 1) * img_h, c * img_w:(c + 1) * img_w] = tile

    ax = fig.add_subplot(gs[1, :3])
    ax.imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(f"Learned Templates ({n_templates}) — per-template normalized")
    ax.axis("off")

    # --- Row 2 right: winner count histogram ---
    ax = fig.add_subplot(gs[1, 3])
    ax.bar(range(n_templates), winner_counts, color="tab:blue", width=1.0)
    ax.set_xlabel("Template")
    ax.set_ylabel("Winner count")
    ax.set_title("Winner Counts (last epoch)")

    # --- Row 3 left: template activation histogram ---
    ax = fig.add_subplot(gs[2, 0])
    ax.bar(range(n_templates), template_counts, color="tab:green", width=1.0)
    ax.set_xlabel("Template")
    ax.set_ylabel("Activation count")
    ax.set_title("Template Usage (last epoch)")

    # --- Row 3 right: coactivation heatmap ---
    ax = fig.add_subplot(gs[2, 1:3])
    if coactivation_counts.size > 0 and coactivation_counts.shape[0] > 1:
        min_counts = np.minimum.outer(template_counts, template_counts)
        with np.errstate(divide="ignore", invalid="ignore"):
            overlap_matrix = np.where(min_counts > 0,
                                      coactivation_counts / min_counts, 0.0)
        np.fill_diagonal(overlap_matrix, 0.0)
        im = ax.imshow(overlap_matrix, cmap="hot", interpolation="nearest",
                       vmin=0.0, vmax=min(1.0, overlap_matrix.max() + 0.05))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Coactivation Overlap")
    ax.set_xlabel("Template j")
    ax.set_ylabel("Template i")

    # --- Row 3 far right: activation count distribution ---
    ax = fig.add_subplot(gs[2, 3])
    ax.hist(template_counts, bins=30, color="tab:orange", edgecolor="black")
    ax.set_xlabel("Activation count")
    ax.set_ylabel("# Templates")
    ax.set_title("Usage Distribution")

    fig_path = OUTPUT_DIR / "imdv_mnist_diagnostics.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostics figure to {fig_path}")


if __name__ == "__main__":
    main()
