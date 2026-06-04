"""Tests for the LADSL model."""

import numpy as np
import pytest
from ladsl import LADSL
from ladsl.utils import normalize_columns, knn_classifier


def _make_toy_dataset(n_classes=3, n_per_class=10, n_features=20, seed=0):
    """Synthetic dataset: Gaussian clusters, well separated."""
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for c in range(n_classes):
        centre = rng.standard_normal(n_features) * 5
        X_list.append(centre[:, None] + rng.standard_normal((n_features, n_per_class)) * 0.5)
        y_list.append(np.full(n_per_class, c + 1, dtype=int))
    X = np.hstack(X_list)
    y = np.concatenate(y_list)
    X = normalize_columns(X)
    return X, y


class TestLADSLFit:
    def test_fit_returns_self(self):
        X, y = _make_toy_dataset()
        model = LADSL(n_components=5, max_iter=10)
        result = model.fit(X, y)
        assert result is model

    def test_result_shapes(self):
        n_classes, n_per, n_feat = 3, 8, 20
        X, y = _make_toy_dataset(n_classes, n_per, n_feat)
        n = X.shape[1]
        d = 5

        model = LADSL(n_components=d, max_iter=15)
        model.fit(X, y)
        r = model.result_

        assert r.P.shape == (n_feat, d)
        assert r.D.shape == (d, n)
        assert r.W.shape == (n_classes, d)
        assert r.A.shape == (n_classes, n_feat)
        assert r.Z.shape == (n, n)
        assert r.E.shape == (d, n)

    def test_transform_shape(self):
        X, y = _make_toy_dataset(n_classes=3, n_per_class=8, n_features=20)
        model = LADSL(n_components=4, max_iter=10)
        model.fit(X, y)

        X_test, _ = _make_toy_dataset(n_classes=3, n_per_class=5, n_features=20, seed=99)
        out = model.transform(X_test)
        assert out.shape == (3, X_test.shape[1])

    def test_fit_transform_matches(self):
        X, y = _make_toy_dataset()
        model = LADSL(n_components=4, max_iter=10)
        out1 = model.fit_transform(X, y)
        out2 = model.transform(X)
        assert np.allclose(out1, out2)

    def test_objective_decreases_roughly(self):
        X, y = _make_toy_dataset(n_classes=4, n_per_class=10, n_features=30)
        model = LADSL(n_components=8, max_iter=50, verbose=False)
        model.fit(X, y)
        obj = model.result_.objective
        # Allow some non-monotonicity (ADMM) but the first half should be higher than the last
        assert obj[0] > obj[-1] or len(obj) < 5

    def test_transform_before_fit_raises(self):
        model = LADSL()
        with pytest.raises(RuntimeError):
            model.transform(np.zeros((10, 5)))

    def test_get_set_params(self):
        model = LADSL(alpha=0.5, n_components=10)
        params = model.get_params()
        assert params["alpha"] == 0.5
        assert params["n_components"] == 10
        model.set_params(alpha=0.1)
        assert model.alpha == 0.1


class TestLADSLClassification:
    """End-to-end classification test using knn_classifier."""

    def test_classification_accuracy_above_chance(self):
        n_classes = 4
        X_train, y_train = _make_toy_dataset(n_classes=n_classes, n_per_class=10, n_features=30, seed=42)
        X_test, y_test = _make_toy_dataset(n_classes=n_classes, n_per_class=8, n_features=30, seed=123)

        model = LADSL(n_components=n_classes, max_iter=30, alpha=1e-2, beta=1e-2, lam=1e-1)
        model.fit(X_train, y_train)

        proj_train = model.transform(X_train)
        proj_train = normalize_columns(proj_train)
        proj_test = model.transform(X_test)
        proj_test = normalize_columns(proj_test)

        acc, _ = knn_classifier(proj_train, y_train, proj_test, y_test, k=1)
        chance = 100.0 / n_classes
        assert acc > chance, f"Expected above-chance accuracy, got {acc:.1f}%"
