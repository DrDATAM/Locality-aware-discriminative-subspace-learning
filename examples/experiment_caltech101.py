"""
Experiment: LADSL on Caltech-101 (spatial pyramid features)
============================================================

Reproduces Table IV from the paper:

    "Locality-Aware Discriminative Subspace Learning for Image Classification"
    IEEE Transactions on Instrumentation and Measurement, 2022.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9812722

Dataset
-------
Caltech-101: 102 object categories, 9144 images, 3000-dimensional spatial
pyramid features.  Each class contains between 31 and 800 images.

Protocol
--------
Training size Tr is varied from 10 to 25 images per class.  For each Tr the
random train/test split is repeated 10 times and the mean classification rate
is reported, matching the paper's evaluation protocol exactly.  Classification
is done with 1-nearest-neighbour in the LADSL-projected subspace.

Usage
-----
    python experiment_caltech101.py --data_path <path_to_spatialpyramidfeatures4caltech101.mat>

The data file is included in the repository's Dataset_WPX/ folder.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ladsl import LADSL, normalize_columns, knn_classifier
from ladsl.data import load_caltech101_spatial, train_test_split_by_class

# ---------------------------------------------------------------------------
# Hyperparameters — taken from the paper's Caltech-101 settings
# ---------------------------------------------------------------------------
N_COMPONENTS = 100      # subspace dimension d
ALPHA        = 1e-2     # locality scatter weight
BETA         = 1e-2     # sparse error weight
LAM          = 1e-1     # low-rank (nuclear norm) weight
MAX_ITER     = 100
N_RUNS       = 10       # repetitions per training size
TRAIN_SIZES  = [10, 15, 20, 25]


def run_one(X: np.ndarray, y: np.ndarray, n_train: int, seed: int) -> float:
    """Single train/test split + fit + evaluate.  Returns accuracy (%)."""
    rng = np.random.default_rng(seed)
    X_tr, y_tr, X_te, y_te = train_test_split_by_class(X, y, n_train, rng)

    X_tr = normalize_columns(X_tr)
    X_te = normalize_columns(X_te)

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


def main(data_path: str) -> None:
    print("=" * 65)
    print("LADSL — Caltech-101 Object Recognition (Spatial Pyramid)")
    print("=" * 65)

    print(f"\nLoading data from: {data_path}")
    X, y = load_caltech101_spatial(data_path)
    n_classes = len(np.unique(y))
    print(f"Features: {X.shape[0]}  |  Samples: {X.shape[1]}  |  Classes: {n_classes}")

    mean_accs, std_accs = [], []

    for n_train in TRAIN_SIZES:
        t0 = time.time()
        accs = [run_one(X, y, n_train, seed=42 + r) for r in range(N_RUNS)]
        elapsed = time.time() - t0

        m, s = np.mean(accs), np.std(accs)
        mean_accs.append(m)
        std_accs.append(s)
        print(f"  Tr={n_train:2d} | mean={m:5.2f}%  std={s:4.2f}%  ({elapsed:.1f}s)")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(TRAIN_SIZES, mean_accs, yerr=std_accs,
                marker="o", linewidth=2, capsize=4, label="LADSL")
    ax.set_xlabel("Number of training samples per class (Tr)", fontsize=12)
    ax.set_ylabel("Mean classification rate (%)", fontsize=12)
    ax.set_title("Caltech-101 — LADSL (Spatial Pyramid Features)", fontsize=12)
    ax.set_xticks(TRAIN_SIZES)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    out_path = Path(__file__).parent / "caltech101_results.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nPlot saved to {out_path}")

    print("\nSummary")
    print(f"{'Tr':>4}  {'Mean (%)':>9}  {'Std':>6}")
    print("-" * 24)
    for tr, m, s in zip(TRAIN_SIZES, mean_accs, std_accs):
        print(f"{tr:>4}  {m:>9.2f}  {s:>6.2f}")


if __name__ == "__main__":
    default_path = str(
        Path(__file__).parents[2]
        / "Dataset_WPX"
        / "spatialpyramidfeatures4caltech101.mat"
    )
    parser = argparse.ArgumentParser(description="LADSL on Caltech-101")
    parser.add_argument(
        "--data_path", default=default_path,
        help="Path to spatialpyramidfeatures4caltech101.mat",
    )
    args = parser.parse_args()
    main(args.data_path)
