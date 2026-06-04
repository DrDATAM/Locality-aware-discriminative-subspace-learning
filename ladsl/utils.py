"""
Supporting utilities for the LADSL algorithm.

These functions implement the core building blocks — distance computation,
graph construction, PCA initialisation, label encoding, and classification —
used throughout the main optimisation loop.
"""

import numpy as np
from scipy.linalg import svd as scipy_svd
from sklearn.neighbors import KNeighborsClassifier


def euclidean_distance(A: np.ndarray, B: np.ndarray | None = None, sqrt: bool = True) -> np.ndarray:
    """Compute pairwise Euclidean distances between two sets of row-vectors.

    The computation avoids an explicit double-loop by exploiting the identity
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b, which reduces the whole operation
    to a single matrix multiply and a couple of broadcasts.

    When only one matrix is passed the result is a square, symmetric distance
    matrix of all pairwise self-distances.  A small diagonal fix is applied
    after computing the squared distances because floating-point arithmetic can
    leave tiny positive residuals on the diagonal (the difference between
    np.sum(x*x) and the BLAS dot product of x with itself after sqrt).

    Args:
        A:    Matrix of shape (n_a, d) where each row is one sample.
        B:    Matrix of shape (n_b, d).  When omitted, distances are computed
              within A (A vs. A).
        sqrt: Return actual Euclidean distances when True, squared distances
              when False.  Squared distances are faster and sufficient for
              nearest-neighbour comparisons.

    Returns:
        D: Distance matrix of shape (n_a, n_a) if B is None, or (n_a, n_b)
           otherwise.  All entries are non-negative.
    """
    aa = np.sum(A * A, axis=1, keepdims=True)
    if B is None:
        ab = A @ A.T
        D = aa + aa.T - 2 * ab
        D = np.maximum(D, 0)
        np.fill_diagonal(D, 0.0)  # remove floating-point noise on the diagonal
        if sqrt:
            D = np.sqrt(D)
        D = np.maximum(D, D.T)    # enforce exact symmetry
    else:
        bb = np.sum(B * B, axis=1, keepdims=True)
        ab = A @ B.T
        D = aa + bb.T - 2 * ab
        D = np.maximum(D, 0)
        if sqrt:
            D = np.sqrt(D)
    return D


def l2_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance between two sets of column-vectors.

    Used internally to compute the locality-aware weight updates inside the
    main ADMM loop.  The column-major convention (each column = one sample)
    is used throughout the codebase for consistency.

    A one-row edge case is handled by appending a zero row so that the
    computation does not collapse to a scalar.

    Args:
        A: Matrix of shape (d, n_a) — each column is a d-dimensional sample.
        B: Matrix of shape (d, n_b) — each column is a d-dimensional sample.

    Returns:
        D: Matrix of shape (n_a, n_b) containing squared Euclidean distances.
           All entries are non-negative (imaginary parts dropped, then clamped).
    """
    if A.shape[0] == 1:
        A = np.vstack([A, np.zeros((1, A.shape[1]))])
        B = np.vstack([B, np.zeros((1, B.shape[1]))])

    aa = np.sum(A * A, axis=0)
    bb = np.sum(B * B, axis=0)
    ab = A.T @ B
    D = aa[:, None] + bb[None, :] - 2 * ab
    D = np.real(D)           # guard against tiny imaginary residuals
    D = np.maximum(D, 0)
    return D


def normalize_columns(X: np.ndarray) -> np.ndarray:
    """Divide each column of X by its L2 norm.

    Before fitting LADSL (and before classifying with the learned subspace)
    each sample vector is normalised so it sits on the unit hypersphere.  This
    removes the effect of overall magnitude differences and keeps dot-product
    distances meaningful.

    A small epsilon (1e-12) is added to the denominator so that all-zero
    columns do not produce NaNs.

    Args:
        X: Matrix of shape (d, n) — each column is one sample vector.

    Returns:
        X_normed: Same shape as X, with each column having unit L2 norm.
    """
    norms = np.sqrt(np.sum(X ** 2, axis=0, keepdims=True))
    return X / (norms + 1e-12)


def encode_labels(labels: np.ndarray) -> np.ndarray:
    """Turn a vector of class labels into a binary indicator matrix H.

    The result is a matrix H of shape (n_classes, n) where H[c, i] = 1 if
    sample i belongs to class c and 0 otherwise.

    This indicator matrix is used throughout the algorithm: to initialise the
    same-class affinity matrix S, to define the discriminative target for W
    and D, and to compute the classification loss term.

    Works with any integer labels — they do not need to be consecutive or
    start from a particular value.

    Args:
        labels: 1-D array of shape (n,) containing integer class identifiers.

    Returns:
        H: Binary matrix of shape (n_classes, n).  Row order follows
           np.unique(labels), i.e. ascending sorted class values.
    """
    classes = np.unique(labels)
    H = np.zeros((len(classes), len(labels)), dtype=float)
    for i, c in enumerate(classes):
        H[i, labels == c] = 1.0
    return H


def pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """Compute the top principal components of a data matrix.

    The function mean-centres the data and finds the leading left singular
    vectors of the feature-by-sample matrix X^T.

    Inside LADSL, PCA is used only to warm-start the projection matrix P at
    the beginning of the first iteration.  A reasonable initialisation makes
    convergence noticeably faster than starting from a random or zero matrix.

    Args:
        X:            Data matrix of shape (n_samples, n_features) where
                      each row is one sample.
        n_components: How many principal components to keep.  Automatically
                      clamped to min(n_samples, n_features) if too large.

    Returns:
        P: Projection matrix of shape (n_features, n_components).  To project
           a new sample x (row vector) into the PCA subspace, compute x @ P.
    """
    X_centered = X - X.mean(axis=0)
    n_components = min(n_components, min(X_centered.shape))
    U, _, _ = scipy_svd(X_centered.T, full_matrices=False)
    return U[:, :n_components]


def construct_weight_matrix(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = 0,
    mode: str = "supervised",
    weight: str = "binary",
    t: float | None = None,
) -> np.ndarray:
    """Build a class-aware neighbourhood affinity matrix W.

    In supervised mode two samples can only be connected if they share the same
    class label.  Setting k=0 connects every pair within a class (a fully
    connected per-class graph).  Setting k>0 restricts each node to its k
    nearest same-class neighbours, which is more robust when classes are large
    and internally heterogeneous.

    The resulting matrix is symmetric: if i is a neighbour of j, j is also a
    neighbour of i.  Self-connections (diagonal) are always removed.

    Args:
        X:      Data matrix of shape (n_samples, n_features), rows = samples.
        labels: Integer label vector of shape (n_samples,).
        k:      Neighbourhood size within each class.  Use 0 for all same-class
                pairs (complete within-class graph).
        mode:   Graph construction strategy.  Currently only ``"supervised"``
                is supported.
        weight: How to assign edge weights.  ``"binary"`` gives 0/1 edges;
                ``"heat_kernel"`` uses a Gaussian kernel exp(-d^2 / 2t^2).
        t:      Bandwidth for the heat kernel.  When None it is estimated
                automatically as the mean of all pairwise squared distances.

    Returns:
        W: Symmetric affinity matrix of shape (n_samples, n_samples).
           Diagonal is zero.  Off-diagonal entries are non-negative.
    """
    n = X.shape[0]
    W = np.zeros((n, n))
    classes = np.unique(labels)

    if t is None and weight == "heat_kernel":
        D_all = euclidean_distance(X, sqrt=False)
        t = float(np.mean(D_all))

    for c in classes:
        idx = np.where(labels == c)[0]
        if k > 0 and len(idx) > k:
            # k-nearest same-class neighbours only
            D_c = euclidean_distance(X[idx], sqrt=False)
            order = np.argsort(D_c, axis=1)[:, 1 : k + 1]   # skip self (col 0)
            for local_i, global_i in enumerate(idx):
                neighbours = idx[order[local_i]]
                if weight == "binary":
                    W[global_i, neighbours] = 1.0
                else:
                    dists = D_c[local_i, order[local_i]]
                    W[global_i, neighbours] = np.exp(-dists / (2 * t ** 2))
        else:
            # fully connected within each class
            if weight == "binary":
                W[np.ix_(idx, idx)] = 1.0
            else:
                D_c = euclidean_distance(X[idx], sqrt=False)
                W[np.ix_(idx, idx)] = np.exp(-D_c / (2 * t ** 2))

    np.fill_diagonal(W, 0)
    W = np.maximum(W, W.T)   # enforce symmetry
    return W


def knn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int = 1,
) -> tuple[float, np.ndarray]:
    """Evaluate classification accuracy with a k-nearest-neighbour classifier.

    This is the standard evaluation protocol used in the paper: after projecting
    data into the learned subspace, 1-NN (k=1) is applied.  In a well-separated
    subspace the nearest neighbour is almost always from the correct class.

    Accepts the column-major layout used throughout the codebase (each column =
    one sample) and internally transposes before passing to scikit-learn.

    Args:
        X_train: Training features of shape (d, n_train).
        y_train: Training labels of shape (n_train,).
        X_test:  Test features of shape (d, n_test).
        y_test:  Ground-truth test labels of shape (n_test,).
        k:       Number of neighbours to consult.

    Returns:
        accuracy:  Classification rate as a percentage (0–100).
        predicted: Predicted label array of shape (n_test,).
    """
    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    clf.fit(X_train.T, y_train)
    predicted = clf.predict(X_test.T)
    accuracy = float(np.sum(predicted == y_test) / len(y_test) * 100)
    return accuracy, predicted
