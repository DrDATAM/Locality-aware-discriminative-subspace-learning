"""
Dataset loaders for the experiments in the LADSL paper.

Each function returns (X, y) in column-major layout — shape (n_features, n_samples)
and (n_samples,) integer labels — ready to pass directly to normalize_columns()
and LADSL.fit().

The loaders handle the two mat-file formats used across these datasets:
- v5 .mat files (scipy.io.loadmat)
- v7.3 HDF .mat files (h5py)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np


def _loadmat(path: str | Path) -> dict:
    """Load a .mat file regardless of whether it is v5 or v7.3 (HDF5)."""
    import scipy.io as sio
    try:
        return sio.loadmat(str(path))
    except NotImplementedError:
        import h5py
        out = {}
        with h5py.File(str(path), "r") as f:
            for k in f.keys():
                if not k.startswith("#"):
                    out[k] = f[k][()]
        return out


def load_orl(mat_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the ORL face dataset (40 classes, 10 images each, 32x32 pixels).

    The paper uses Tr = 5–7 training images per class and d = 100.

    Args:
        mat_path: Path to ORL_32x32.mat.

    Returns:
        X: (1024, 400) float64, column-normalised.
        y: (400,) integer labels 1–40.
    """
    d = _loadmat(mat_path)
    fea = np.array(d["fea"], dtype=float)   # (400, 1024)
    gnd = np.array(d["gnd"], dtype=int).ravel()
    return fea.T, gnd


def load_yale_b(mat_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the Extended Yale B face dataset (38 classes, ~64 images each, 32x32).

    Used for both the standard classification experiment (Tr = 10–25, d = 140)
    and the noise / block-occlusion robustness experiment.

    Args:
        mat_path: Path to YaleB_32x32.mat.

    Returns:
        X: (1024, 2414) float64.
        y: (2414,) integer labels 1–38.
    """
    d = _loadmat(mat_path)
    fea = np.array(d["fea"], dtype=float)   # (2414, 1024)
    gnd = np.array(d["gnd"], dtype=int).ravel()
    return fea.T, gnd


def load_caltech101_spatial(mat_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the Caltech-101 spatial pyramid features dataset.

    The file contains 3000-dimensional spatial pyramid features for 9144 images
    from 102 object categories.  The label matrix is one-hot (102 x 9144); we
    convert it to integer labels here.

    The paper uses Tr = 10–25 training images per class and 10 random repetitions.

    Args:
        mat_path: Path to spatialpyramidfeatures4caltech101.mat.

    Returns:
        X: (3000, 9144) float64.
        y: (9144,) integer labels 1–102.
    """
    d = _loadmat(mat_path)
    fea = np.array(d["featureMat"], dtype=float)        # (3000, 9144) — already column-major
    label_mat = np.array(d["labelMat"], dtype=int)      # (102, 9144) one-hot
    gnd = np.argmax(label_mat, axis=0) + 1              # integer labels 1–102
    return fea, gnd


def train_test_split_by_class(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified random train/test split — n_train samples per class for training.

    This matches the paper's evaluation protocol exactly: for each class
    n_train images are randomly selected for training and the remaining images
    form the test set.

    Args:
        X:       (n_features, n_samples) data matrix.
        y:       (n_samples,) integer label vector.
        n_train: Number of training samples per class.
        rng:     Random number generator.  Created with default seed if None.

    Returns:
        X_train, y_train, X_test, y_test — all column-major.
    """
    if rng is None:
        rng = np.random.default_rng()

    classes = np.unique(y)
    X_tr, y_tr, X_te, y_te = [], [], [], []

    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) <= n_train:
            # skip classes with too few samples
            continue
        perm = rng.permutation(len(idx))
        tr_idx = idx[perm[:n_train]]
        te_idx = idx[perm[n_train:]]
        X_tr.append(X[:, tr_idx])
        y_tr.append(y[tr_idx])
        X_te.append(X[:, te_idx])
        y_te.append(y[te_idx])

    return (
        np.hstack(X_tr), np.concatenate(y_tr),
        np.hstack(X_te), np.concatenate(y_te),
    )
