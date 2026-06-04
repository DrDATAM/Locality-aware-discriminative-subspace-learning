"""
Demonstration of LADSL on a synthetic multi-class dataset.

Generates well-separated Gaussian clusters, fits LADSL,
and reports classification accuracy using 1-NN in the learned subspace.
"""

import numpy as np
import matplotlib.pyplot as plt

from ladsl import LADSL, normalize_columns, knn_classifier


def make_synthetic(n_classes=5, n_train=15, n_test=10, n_features=60, seed=42):
    """Generate Gaussian clusters with the same centres for train and test."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((n_classes, n_features)) * 10

    X_train_parts, X_test_parts, y_train_parts, y_test_parts = [], [], [], []
    for c in range(n_classes):
        label = c + 1
        X_train_parts.append(centres[c, :, None] + rng.standard_normal((n_features, n_train)) * 0.8)
        X_test_parts.append(centres[c, :, None] + rng.standard_normal((n_features, n_test)) * 0.8)
        y_train_parts.append(np.full(n_train, label))
        y_test_parts.append(np.full(n_test, label))

    X_train = np.hstack(X_train_parts)
    X_test = np.hstack(X_test_parts)
    y_train = np.concatenate(y_train_parts)
    y_test = np.concatenate(y_test_parts)
    return X_train, y_train, X_test, y_test


def run_demo():
    print("=" * 60)
    print("LADSL Demo -- Synthetic Gaussian Clusters")
    print("=" * 60)

    n_classes = 5
    X_train, y_train, X_test, y_test = make_synthetic(
        n_classes=n_classes, n_train=15, n_test=10, n_features=60, seed=42
    )

    # L2-normalise columns (mirrors MATLAB pre-processing)
    X_train = normalize_columns(X_train)
    X_test = normalize_columns(X_test)

    print(f"\nTraining: {X_train.shape[1]} samples | Test: {X_test.shape[1]} samples")
    print(f"Feature dim: {X_train.shape[0]} | Classes: {n_classes}")

    # Fit LADSL
    model = LADSL(
        n_components=n_classes,
        alpha=1e-2,
        beta=1e-2,
        lam=1e-1,
        max_iter=100,
        verbose=True,
    )
    print("\nFitting LADSL ...")
    model.fit(X_train, y_train)

    # Project and classify
    proj_train = normalize_columns(model.transform(X_train))
    proj_test = normalize_columns(model.transform(X_test))

    acc, _ = knn_classifier(proj_train, y_train, proj_test, y_test, k=1)
    print(f"\n1-NN accuracy in LADSL subspace: {acc:.2f}%")
    print(f"(Chance level: {100.0 / n_classes:.1f}%)")

    # Plot convergence
    obj = model.result_.objective
    plt.figure(figsize=(7, 4))
    plt.plot(obj, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title("LADSL Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ladsl_convergence.png", dpi=100)
    print("Convergence plot saved to ladsl_convergence.png")


if __name__ == "__main__":
    run_demo()
