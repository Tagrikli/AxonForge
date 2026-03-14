"""Evaluate learned IMDV dictionaries on MNIST via a linear classifier.

Loads trained weights, encodes train/test images as activation vectors,
fits a linear classifier, and produces comprehensive classification metrics
(accuracy, per-class precision/recall/F1, confusion matrix, etc.).

Usage:
    python experiments/eval_imdv_mnist.py --weights experiments/r003/imdv_mnist_latest.npz
    python experiments/eval_imdv_mnist.py --weights experiments/r003/imdv_mnist_latest.npz --template-count 512
"""

from __future__ import annotations

import argparse
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-8


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


def _load_mnist():
    mnist_path = _resolve_mnist_path()
    loader = MNIST(str(mnist_path))

    x_train, y_train = loader.load_training()
    x_train = np.asarray(x_train, dtype=np.float32) / 255.0
    y_train = np.asarray(y_train)

    x_test, y_test = loader.load_testing()
    x_test = np.asarray(x_test, dtype=np.float32) / 255.0
    y_test = np.asarray(y_test)

    return x_train, y_train, x_test, y_test


def encode_raw(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Encode samples using raw IMDV activation rule: a = ReLU(w @ x_hat).

    Parameters
    ----------
    x : (N, D) raw pixel vectors (already in [0, 1]).
    weights : (T, D) unit-norm weight bank.

    Returns
    -------
    activations : (N, T)
    """
    n = x.shape[0]
    t = weights.shape[0]
    activations = np.empty((n, t), dtype=np.float32)
    batch_size = 1024

    for start in tqdm(range(0, n, batch_size), desc="encoding (raw)", leave=False):
        end = min(start + batch_size, n)
        batch = x[start:end]
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        batch_hat = batch / (norms + EPS)
        sims = batch_hat @ weights.T  # (B, T)
        activations[start:end] = np.maximum(sims, 0.0)

    return activations


def encode_loo_inhibited(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Encode samples using LOO-shadow-inhibited activations (v2).

    a_raw = ReLU(w @ x_hat)
    x_shadow_i = Σ_{j≠i} a_raw_j * w_j  (LOO reconstruction)
    shadow_overlap_i = w_i · x_shadow_hat_i
    a_i = ReLU(s_i - shadow_overlap_i)

    Parameters
    ----------
    x : (N, D) raw pixel vectors (already in [0, 1]).
    weights : (T, D) unit-norm weight bank.

    Returns
    -------
    activations : (N, T)
    """
    n = x.shape[0]
    t = weights.shape[0]
    activations = np.empty((n, t), dtype=np.float32)
    batch_size = 256  # smaller batches — LOO is more memory-intensive

    for start in tqdm(range(0, n, batch_size), desc="encoding (loo-inh)", leave=False):
        end = min(start + batch_size, n)
        batch = x[start:end].astype(np.float64)
        b = end - start

        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        batch_hat = batch / (norms + EPS)

        # Raw similarities and activations: (B, T)
        s = batch_hat @ weights.T
        a_raw = np.maximum(s, 0.0)

        # Global reconstruction per sample: (B, D)
        x_recon = a_raw @ weights

        # LOO shadow per sample per template: shadow_i = x_recon - a_raw_i * w_i
        # shadow_overlap_i = w_i · (shadow_i / ||shadow_i||)
        # We compute this efficiently without materializing the full (B, T, D) tensor
        shadow_overlap = np.empty((b, t), dtype=np.float64)
        for ti in range(t):
            # shadow for template ti: (B, D)
            shadow = x_recon - a_raw[:, ti:ti+1] * weights[ti:ti+1]
            shadow_norm = np.linalg.norm(shadow, axis=1, keepdims=True)
            shadow_hat = shadow / (shadow_norm + EPS)
            # w_ti · shadow_hat: (B,)
            shadow_overlap[:, ti] = shadow_hat @ weights[ti]

        a_inhibited = np.maximum(s - shadow_overlap, 0.0)
        activations[start:end] = a_inhibited.astype(np.float32)

    return activations


def train_and_evaluate(args):
    # --- Load weights ---
    data = np.load(args.weights, allow_pickle=True)
    weights = data["weights"]
    print(f"Loaded weights from {args.weights}  shape={weights.shape}")

    if args.template_count is not None and args.template_count < weights.shape[0]:
        weights = weights[: args.template_count]
        print(f"Using first {args.template_count} templates")

    n_templates, dim = weights.shape
    print(f"Templates: {n_templates}  Dim: {dim}")

    # --- Load MNIST ---
    print("Loading MNIST...")
    x_train, y_train, x_test, y_test = _load_mnist()
    print(f"Train: {x_train.shape[0]}  Test: {x_test.shape[0]}")

    # --- Encode ---
    encoding = args.encoding
    print(f"Encoding mode: {encoding}")
    if encoding == "loo-inhibited":
        encode_fn = encode_loo_inhibited
    else:
        encode_fn = encode_raw

    print("Encoding train set...")
    z_train = encode_fn(x_train, weights)
    print("Encoding test set...")
    z_test = encode_fn(x_test, weights)

    # Activation statistics
    train_active_frac = np.mean(z_train > 0)
    test_active_frac = np.mean(z_test > 0)
    print(f"Train active fraction: {train_active_frac:.4f}")
    print(f"Test  active fraction: {test_active_frac:.4f}")

    # --- Train linear classifier ---
    print("Training logistic regression...")
    clf = LogisticRegression(
        max_iter=args.max_iter,
        solver="lbfgs",
        C=args.regularization,
        verbose=1,
    )
    clf.fit(z_train, y_train)

    # --- Evaluate ---
    y_pred_train = clf.predict(z_train)
    y_pred_test = clf.predict(z_test)
    y_proba_test = clf.predict_proba(z_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    top3_acc = top_k_accuracy_score(y_test, y_proba_test, k=3)
    top5_acc = top_k_accuracy_score(y_test, y_proba_test, k=5)

    print(f"\n{'='*50}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {test_acc:.4f}")
    print(f"Top-3 accuracy: {top3_acc:.4f}")
    print(f"Top-5 accuracy: {top5_acc:.4f}")
    print(f"{'='*50}\n")

    report_str = classification_report(y_test, y_pred_test, digits=4)
    print(report_str)

    report_dict = classification_report(y_test, y_pred_test, output_dict=True)
    cm = confusion_matrix(y_test, y_pred_test)

    # --- Save results ---
    output_dir = Path(args.weights).parent
    results = {
        "weights_path": str(args.weights),
        "encoding": encoding,
        "template_count": n_templates,
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "top3_accuracy": top3_acc,
        "top5_accuracy": top5_acc,
        "train_active_fraction": float(train_active_frac),
        "test_active_fraction": float(test_active_frac),
        "regularization_C": args.regularization,
        "max_iter": args.max_iter,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }

    suffix = f"_{encoding}" if encoding != "raw" else ""
    results_path = output_dir / f"eval_classification_results{suffix}.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")

    # --- Plot ---
    visualize(cm, report_dict, results, output_dir, z_test, y_test, weights, suffix)


def visualize(
    cm: np.ndarray,
    report_dict: dict,
    results: dict,
    output_dir: Path,
    z_test: np.ndarray,
    y_test: np.ndarray,
    weights: np.ndarray,
    suffix: str = "",
):
    class_labels = list(range(10))
    n_templates = weights.shape[0]

    fig = plt.figure(figsize=(22, 14), constrained_layout=True)
    fig.suptitle(
        f"IMDV Linear Probe — {n_templates} templates  |  "
        f"Test acc: {results['test_accuracy']:.4f}",
        fontsize=14,
    )
    gs = GridSpec(2, 4, figure=fig)

    # 1. Confusion matrix (counts)
    ax = fig.add_subplot(gs[0, 0:2])
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(class_labels)
    ax.set_yticks(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (counts)")
    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)

    # 2. Normalized confusion matrix
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm_norm / row_sums, 0.0)

    ax = fig.add_subplot(gs[0, 2:4])
    im = ax.imshow(cm_norm, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(class_labels)
    ax.set_yticks(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized)")
    for i in range(10):
        for j in range(10):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    # 3. Per-class precision, recall, F1
    precisions = [report_dict[str(c)]["precision"] for c in class_labels]
    recalls = [report_dict[str(c)]["recall"] for c in class_labels]
    f1s = [report_dict[str(c)]["f1-score"] for c in class_labels]

    ax = fig.add_subplot(gs[1, 0])
    x_pos = np.arange(10)
    width = 0.25
    ax.bar(x_pos - width, precisions, width, label="Precision", color="tab:blue")
    ax.bar(x_pos, recalls, width, label="Recall", color="tab:orange")
    ax.bar(x_pos + width, f1s, width, label="F1", color="tab:green")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_labels)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # 4. Mean activation per class
    ax = fig.add_subplot(gs[1, 1])
    mean_act_per_class = []
    for c in class_labels:
        mask = y_test == c
        mean_act_per_class.append(z_test[mask].mean())
    ax.bar(class_labels, mean_act_per_class, color="tab:purple")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Mean activation")
    ax.set_title("Mean Activation by Class")

    # 5. Active templates per sample distribution
    ax = fig.add_subplot(gs[1, 2])
    active_per_sample = np.sum(z_test > 0, axis=1)
    ax.hist(active_per_sample, bins=50, color="tab:cyan", edgecolor="black")
    ax.set_xlabel("# Active templates")
    ax.set_ylabel("# Samples")
    ax.set_title("Active Templates per Sample")
    ax.axvline(active_per_sample.mean(), color="red", linestyle="--",
               label=f"mean={active_per_sample.mean():.1f}")
    ax.legend(fontsize=8)

    # 6. Per-class support
    ax = fig.add_subplot(gs[1, 3])
    supports = [report_dict[str(c)]["support"] for c in class_labels]
    ax.bar(class_labels, supports, color="tab:gray")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Support")
    ax.set_title("Test Set Class Distribution")

    fig_path = output_dir / f"eval_classification_diagnostics{suffix}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostics figure to {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IMDV dictionary on MNIST with a linear classifier"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to imdv_mnist_latest.npz",
    )
    parser.add_argument(
        "--template-count", type=int, default=None,
        help="Use only the first N templates (default: use all)",
    )
    parser.add_argument(
        "--regularization", type=float, default=1.0,
        help="Logistic regression inverse regularization strength C (default: 1.0)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000,
        help="Max iterations for logistic regression solver (default: 1000)",
    )
    parser.add_argument(
        "--encoding", type=str, default="loo-inhibited",
        choices=["raw", "loo-inhibited"],
        help="Activation encoding mode (default: loo-inhibited)",
    )
    args = parser.parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
