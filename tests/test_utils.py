"""Tests for utility functions."""

import numpy as np
import pytest
from ladsl.utils import (
    euclidean_distance,
    l2_distance,
    normalize_columns,
    encode_labels,
    pca,
    knn_classifier,
)


def test_euclidean_distance_self():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 10))
    D = euclidean_distance(X)
    assert D.shape == (20, 20)
    assert np.allclose(np.diag(D), 0.0, atol=1e-10)
    assert np.all(D >= 0)
    assert np.allclose(D, D.T)


def test_euclidean_distance_cross():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((15, 8))
    B = rng.standard_normal((20, 8))
    D = euclidean_distance(A, B)
    assert D.shape == (15, 20)
    assert np.all(D >= 0)


def test_l2_distance_column_major():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((5, 10))  # 5 features, 10 samples
    B = rng.standard_normal((5, 12))
    D = l2_distance(A, B)
    assert D.shape == (10, 12)
    assert np.all(D >= 0)


def test_normalize_columns():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((8, 30))
    Xn = normalize_columns(X)
    col_norms = np.sqrt(np.sum(Xn ** 2, axis=0))
    assert np.allclose(col_norms, 1.0, atol=1e-10)


def test_encode_labels_shape():
    labels = np.array([1, 1, 2, 2, 3])
    H = encode_labels(labels)
    assert H.shape == (3, 5)
    assert np.all(H.sum(axis=0) == 1)


def test_encode_labels_values():
    labels = np.array([2, 1, 2, 3, 1])
    H = encode_labels(labels)
    # class 1 → row 0, class 2 → row 1, class 3 → row 2
    assert H[0, 1] == 1 and H[0, 4] == 1   # label-1 positions
    assert H[1, 0] == 1 and H[1, 2] == 1   # label-2 positions
    assert H[2, 3] == 1                     # label-3 position


def test_pca_output_shape():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((50, 20))
    P = pca(X, n_components=10)
    assert P.shape == (20, 10)


def test_pca_orthonormality():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((50, 15))
    P = pca(X, n_components=8)
    assert np.allclose(P.T @ P, np.eye(8), atol=1e-10)


def test_knn_classifier_perfect():
    # 2 well-separated classes, should classify perfectly
    rng = np.random.default_rng(6)
    X_train = np.hstack([rng.standard_normal((4, 20)), 10 + rng.standard_normal((4, 20))])
    y_train = np.array([1] * 20 + [2] * 20)
    X_test = np.hstack([rng.standard_normal((4, 10)), 10 + rng.standard_normal((4, 10))])
    y_test = np.array([1] * 10 + [2] * 10)

    acc, pred = knn_classifier(X_train, y_train, X_test, y_test, k=1)
    assert acc == 100.0
    assert len(pred) == 20
