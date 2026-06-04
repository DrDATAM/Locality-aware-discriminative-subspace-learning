"""
Experiment: LADSL Noise Robustness on Extended Yale B
======================================================

Reproduces Figure 3 from the paper:

    "Locality-Aware Discriminative Subspace Learning for Image Classification"
    IEEE Transactions on Instrumentation and Measurement, 2022.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9812722

This is the most distinctive experiment in the paper — it shows that the
L2,1 sparse error term E in LADSL's objective actively absorbs structured
corruption, so the learned projection is far more robust to real-world noise
than methods that do not model errors explicitly.

Dataset
-------
Extended Yale B: 38 subjects, ~64 images each, 32 × 32 pixels (1024-dim).

Protocol  (Section IV-B-3 of the paper)
----------------------------------------
1. Randomly select 15 classes from the 38 available.
2. Use 30 training images per selected class.
3. Corrupt ALL images (train and test) with one of:
     - Salt-and-pepper noise at 10 % or 20 % density
     - Block occlusion of size 10×10 or 20×20 pixels placed randomly
4. Repeat the entire procedure 10 times; report the mean classification rate.

The experiment demonstrates that LADSL degrades gracefully as corruption
increases, outperforming methods without an explicit noise term.

Usage
-----
    python experiment_noise_robustness.py --data_path <path_to_YaleB_32x32.mat>

The data file is in the repository's Dataset_WPX/ folder.
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ladsl import LADSL, normalize_columns, knn_classifier
from ladsl.data import load_yale_b, train_test_split_by_class

# ---------------------------------------------------------------------------
# Hyperparameters — Yale B settings from the paper
# ---------------------------------------------------------------------------
N_COMPONENTS  = 140
ALPHA         = 1e-2
BETA          = 1e-3
LAM           = 1e-3
MAX_ITER      = 100
N_CLASSES_USE = 15      # randomly select 15 of the 38 classes each run
N_TRAIN       = 30      # training images per class
N_RUNS        = 10      # repetitions

IMG_SIZE      = (32, 32)   # each sample reshaped to this for corruption


# ---------------------------------------------------------------------------
# Corruption functions
# ---------------------------------------------------------------------------

def add_salt_and_pepper(X: np.ndarray, density: float, rng: np.random.Generator) -> np.ndarray:
    """Corrupt a (d, n) data matrix with salt-and-pepper noise.

    Each pixel in each image is independently flipped to the maximum (salt)
    or minimum (pepper) value with probability `density / 2` each.

    Args:
        X:       (d, n) data matrix — each column is a flattened image.
        density: Fraction of pixels to corrupt (e.g. 0.10 for 10 %).
        rng:     Random number generator.

    Returns:
        X_noisy: Same shape as X.
    """
    X_noisy = X.copy()
    d, n = X_noisy.shape
    noise = rng.random((d, n))
    vmin, vmax = X.min(), X.max()
    X_noisy[noise < density / 2] = vmin          # pepper
    X_noisy[(noise >= density / 2) & (noise < density)] = vmax  # salt
    return X_noisy


def add_block_occlusion(X: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Occlude each image with a randomly placed zero-filled block.

    Args:
        X:          (d, n) data matrix — each column is a flattened image.
        block_size: Side length of the square occluding block in pixels.
        rng:        Random number generator.

    Returns:
        X_occ: Same shape as X with one block zeroed per column.
    """
    X_occ = X.copy()
    h, w = IMG_SIZE
    d, n = X_occ.shape
    for i in range(n):
        r0 = rng.integers(0, h - block_size)
        c0 = rng.integers(0, w - block_size)
        img = X_occ[:, i].reshape(h, w)
        img[r0:r0 + block_size, c0:c0 + block_size] = 0.0
        X_occ[:, i] = img.ravel()
    return X_occ


# ---------------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------------

def run_one(
    X: np.ndarray,
    y: np.ndarray,
    corruption_fn,
    seed: int,
) -> float:
    """One repetition: subsample classes, corrupt, train, evaluate.

    Args:
        X:             (d, n_total) clean data.
        y:             (n_total,) labels.
        corruption_fn: Callable(X, rng) -> X_corrupted.
        seed:          Random seed for reproducibility.

    Returns:
        Classification accuracy (%).
    """
    rng = np.random.default_rng(seed)

    # Step 1: randomly select N_CLASSES_USE classes
    all_classes = np.unique(y)
    chosen = rng.choice(all_classes, size=N_CLASSES_USE, replace=False)
    mask = np.isin(y, chosen)
    X_sub, y_sub = X[:, mask], y[mask]

    # Re-label chosen classes to consecutive integers 1..N_CLASSES_USE
    label_map = {c: i + 1 for i, c in enumerate(sorted(chosen))}
    y_sub = np.array([label_map[yi] for yi in y_sub])

    # Step 2: apply corruption to all data BEFORE splitting
    X_corrupted = corruption_fn(X_sub, rng)

    # Step 3: stratified train/test split
    X_tr, y_tr, X_te, y_te = train_test_split_by_class(
        X_corrupted, y_sub, N_TRAIN, rng
    )

    X_tr = normalize_columns(X_tr)
    X_te = normalize_columns(X_te)

    # Step 4: fit LADSL and classify
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LADSL(
            n_components=N_COMPONENTS,
            alpha=ALPHA, beta=BETA, lam=LAM,
            max_iter=MAX_ITER, verbose=False,
        )
        model.fit(X_tr, y_tr)

    proj_tr = normalize_columns(model.transform(X_tr))
    proj_te = normalize_columns(model.transform(X_te))
    acc, _ = knn_classifier(proj_tr, y_tr, proj_te, y_te, k=1)
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_path: str) -> None:
    print("=" * 65)
    print("LADSL — Noise Robustness on Extended Yale B")
    print("=" * 65)

    print(f"\nLoading data from: {data_path}")
    X, y = load_yale_b(data_path)
    n_classes = len(np.unique(y))
    print(f"Features: {X.shape[0]}  |  Samples: {X.shape[1]}  |  Classes: {n_classes}")
    print(f"Protocol: {N_CLASSES_USE} randomly selected classes, "
          f"{N_TRAIN} train images each, {N_RUNS} repetitions\n")

    # Define corruption conditions
    conditions = {
        "Clean":                lambda X, rng: X.copy(),
        "Salt & Pepper 10%":    lambda X, rng: add_salt_and_pepper(X, 0.10, rng),
        "Salt & Pepper 20%":    lambda X, rng: add_salt_and_pepper(X, 0.20, rng),
        "Block Occlusion 10x10": lambda X, rng: add_block_occlusion(X, 10, rng),
        "Block Occlusion 20x20": lambda X, rng: add_block_occlusion(X, 20, rng),
    }

    results = {}
    for label, fn in conditions.items():
        t0 = time.time()
        accs = [run_one(X, y, fn, seed=100 + r) for r in range(N_RUNS)]
        elapsed = time.time() - t0
        m, s = np.mean(accs), np.std(accs)
        results[label] = (m, s)
        print(f"  {label:<25} mean={m:5.2f}%  std={s:4.2f}%  ({elapsed:.1f}s)")

    # -----------------------------------------------------------------------
    # Bar chart
    # -----------------------------------------------------------------------
    labels = list(results.keys())
    means  = [results[k][0] for k in labels]
    stds   = [results[k][1] for k in labels]
    colors = ["#4C72B0", "#DD8452", "#C44E52", "#55A868", "#8172B2"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Mean classification rate (%)", fontsize=12)
    ax.set_title(
        "LADSL Robustness to Noise and Occlusion — Extended Yale B\n"
        f"(15 classes, Tr={N_TRAIN}, {N_RUNS} runs)",
        fontsize=11,
    )
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path = Path(__file__).parent / "noise_robustness_results.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nPlot saved to {out_path}")

    print("\nSummary")
    print(f"{'Condition':<26}  {'Mean (%)':>9}  {'Std':>6}")
    print("-" * 44)
    for k, (m, s) in results.items():
        print(f"{k:<26}  {m:>9.2f}  {s:>6.2f}")

    # -----------------------------------------------------------------------
    # Block occlusion sweep (varying occlusion size — mirrors Fig 3b)
    # -----------------------------------------------------------------------
    print("\nBlock occlusion size sweep ...")
    block_sizes = [5, 8, 10, 14, 18, 20]
    sweep_means, sweep_stds = [], []
    for bs in block_sizes:
        fn = lambda X, rng, b=bs: add_block_occlusion(X, b, rng)
        accs = [run_one(X, y, fn, seed=200 + r) for r in range(N_RUNS)]
        m, s = np.mean(accs), np.std(accs)
        sweep_means.append(m)
        sweep_stds.append(s)
        pct = round((bs * bs) / (32 * 32) * 100)
        print(f"  {bs}x{bs} block (~{pct}% of image): mean={m:.2f}%  std={s:.2f}%")

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.errorbar(block_sizes, sweep_means, yerr=sweep_stds,
                 marker="s", linewidth=2, capsize=4, color="#C44E52", label="LADSL")
    ax2.set_xlabel("Block occlusion side length (pixels)", fontsize=12)
    ax2.set_ylabel("Mean classification rate (%)", fontsize=12)
    ax2.set_title("LADSL — Effect of Increasing Block Occlusion (Yale B)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    out_path2 = Path(__file__).parent / "occlusion_sweep_results.png"
    plt.savefig(out_path2, dpi=120)
    print(f"Plot saved to {out_path2}")


if __name__ == "__main__":
    default_path = str(
        Path(__file__).parents[2]
        / "Dataset_WPX"
        / "YaleB_32x32.mat"
    )
    parser = argparse.ArgumentParser(description="LADSL noise robustness experiment")
    parser.add_argument(
        "--data_path", default=default_path,
        help="Path to YaleB_32x32.mat",
    )
    args = parser.parse_args()
    main(args.data_path)
