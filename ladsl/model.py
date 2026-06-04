"""
LADSL: Locality-Aware Discriminative Subspace Learning

Python implementation of the algorithm presented in:

    "Locality-Aware Discriminative Subspace Learning for Image Classification"
    IEEE Transactions on Image Processing, 2022.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9812722

The algorithm learns a projection P, a discriminative code matrix D, a linear
classifier W, and a low-rank self-expression Z — all tied together by a
locality-preserving scatter term that dynamically re-weights within-class
neighbour pairs during optimisation.

Optimisation is carried out via ADMM (Alternating Direction Method of Multipliers).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import svd as scipy_svd

from .utils import pca, encode_labels, l2_distance


# ---------------------------------------------------------------------------
# Internal proximal operators
# ---------------------------------------------------------------------------

def _solve_l2(w: np.ndarray, lam: float) -> np.ndarray:
    """Proximal operator for a scaled L2-norm penalty applied to a single vector.

    Solves:  min_x  lam * ||x||_2  +  ||x - w||_2^2

    The closed-form solution is a vector shrinkage: the output has the same
    direction as w but its magnitude is reduced by lam.  If ||w|| <= lam the
    entire vector collapses to zero (the origin is the minimiser).

    This is used column-by-column inside _solve_l21 to handle the L2,1 norm
    on the error matrix E, which encourages whole columns (i.e. the full
    feature vector of a single sample) to be either present or exactly zero —
    a more structured form of sparsity than element-wise L1.

    Args:
        w:   Input vector of any length.
        lam: Shrinkage threshold — larger values push more columns to zero.

    Returns:
        x: Shrunk vector with the same shape as w.
    """
    nw = np.linalg.norm(w)
    if nw > lam:
        return (nw - lam) * w / nw
    return np.zeros_like(w)


def _solve_l21(W: np.ndarray, lam: float) -> np.ndarray:
    """Proximal operator for the L2,1 mixed norm, applied column-by-column.

    The L2,1 norm treats each column of W as a group and sums their L2 norms.
    Its proximal operator decouples across columns: each column is independently
    shrunk towards zero using _solve_l2.

    Args:
        W:   Matrix of shape (d, n).  Each column is treated as one group.
        lam: Shrinkage threshold applied to every column.

    Returns:
        E: Shrunk matrix of the same shape as W.
    """
    E = W.copy()
    for i in range(W.shape[1]):
        E[:, i] = _solve_l2(W[:, i], lam)
    return E


def _singular_value_threshold(M: np.ndarray, threshold: float) -> np.ndarray:
    """Proximal operator for the nuclear norm (sum of singular values).

    The nuclear norm is the convex relaxation of the matrix rank — minimising
    it encourages the self-expression coefficient matrix Z to be low-rank, which
    enforces that all same-class samples share a common underlying subspace.

    The operator soft-thresholds the singular values: values smaller than
    `threshold` are zeroed; larger ones are reduced by `threshold`.  The result
    is clamped to be non-negative, which keeps J on the non-negative orthant
    consistent with the paper's formulation.

    Args:
        M:         Matrix of shape (n, n) to threshold.
        threshold: Amount subtracted from each singular value (lam / mu in
                   the ADMM augmented Lagrangian).

    Returns:
        J: Low-rank, non-negative approximation of M with the same shape.
    """
    U, s, Vt = scipy_svd(M, full_matrices=False)
    svp = int(np.sum(s > threshold))
    if svp >= 1:
        s_thresh = s[:svp] - threshold
    else:
        svp = 1
        s_thresh = np.array([0.0])
    J = U[:, :svp] @ np.diag(s_thresh) @ Vt[:svp, :]
    return np.maximum(J, 0)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class LADSLResult:
    """All learned matrices returned after a completed optimisation run.

    Attributes:
        Z:         Self-expression coefficient matrix, shape (n, n).  Each
                   training sample is expressed as a weighted combination of the
                   other training samples.  The low-rank constraint on Z
                   enforces that all same-class samples share a common subspace.
        E:         Sparse error / noise matrix, shape (d, n).  Captures
                   structured corruption in the projected data.
        P:         Orthogonal projection matrix, shape (m, d).  Maps raw
                   m-dimensional features into the d-dimensional subspace.
        D:         Discriminative code matrix, shape (d, n).  Represents the
                   projected training data in the learned subspace.
        W:         Linear classifier weight matrix, shape (c, d).  Together
                   with P, forms the final mapping from raw features to class
                   scores: W @ P.T @ x.
        A:         Auxiliary weight matrix, shape (c, m).  Couples the
                   locality-preserving scatter term into the joint objective.
        objective: Objective function value recorded at every iteration.
                   Plot this to verify convergence.
    """
    Z: np.ndarray
    E: np.ndarray
    P: np.ndarray
    D: np.ndarray
    W: np.ndarray
    A: np.ndarray
    objective: list[float] = field(default_factory=list)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project a data matrix into the learned discriminative subspace.

        The final representation of a sample x is obtained by first projecting
        with P (the subspace projection) and then applying the linear classifier
        weights W.  The result lives in a c-dimensional space where c is the
        number of classes, which makes downstream 1-NN classification cheap and
        effective.

        Args:
            X: Data matrix of shape (m, n_samples) where each column is one
               sample.  m must match the feature dimension used during training.

        Returns:
            Projected matrix of shape (c, n_samples).
        """
        return self.W @ self.P.T @ X


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class LADSL:
    """Locality-Aware Discriminative Subspace Learning (LADSL).

    LADSL is a supervised subspace learning algorithm that jointly optimises
    five interconnected objectives:

    1. **Low-rank self-expression** — the training set should be representable
       as a linear combination of itself with a low-rank coefficient matrix Z.
    2. **Sparse noise modelling** — deviations from a clean low-rank signal are
       captured by a sparse error E (L2,1 norm encourages whole-column zeroes).
    3. **Discriminative coding** — a linear mapping (P, D) projects features
       into a subspace where class labels are linearly predictable via W.
    4. **Locality-preserving scatter** — same-class neighbours should map close
       together; the scatter matrix Sw captures this and is re-weighted at
       every iteration based on the current projected representation.
    5. **Orthogonality of P** — P is constrained to the Stiefel manifold via
       a Procrustes (SVD-based) update.

    All five are optimised simultaneously using ADMM, which introduces Lagrange
    multipliers Y1–Y4 and an augmented penalty mu that grows geometrically by
    a factor rho each iteration.

    After fitting, a test sample x is classified by:
        1. Computing the projection  f = W @ P.T @ x
        2. L2-normalising f
        3. Finding the 1-nearest-neighbour in the projected training set

    Parameters:
        n_components: Dimension of the learned subspace (d in the paper).
                      A value equal to the number of classes often works well
                      as a starting point.
        alpha:        Weight on the locality-preserving scatter term.  Larger
                      values push same-class samples closer together but can
                      over-constrain the solution.
        beta:         Weight on the L2,1 sparse error term.  Controls how much
                      structured noise is allowed in the reconstruction.
        lam:          Weight on the nuclear norm of Z.  Larger values encourage
                      a lower-rank self-expression (stronger subspace assumption).
        k_neighbours: Neighbourhood size used to seed the locality graph.
                      Kept for forward compatibility with the graph construction
                      utility; the adaptive S update uses all same-class pairs.
        max_iter:     Hard cap on the number of ADMM iterations.  Most datasets
                      from the paper converge within 30–80 iterations.
        tol:          Convergence is declared when the maximum relative residual
                      across all four constraint violations drops below this.
        rho:          Geometric growth factor for the ADMM penalty mu.  Values
                      slightly above 1 (e.g. 1.03) are typical.
        mu_init:      Starting value of the ADMM penalty parameter.
        mu_max:       Upper bound on mu; prevents numerical issues from an
                      unbounded penalty.
        verbose:      When True, prints obj and err at every iteration so you
                      can watch the algorithm converge in real time.
    """

    def __init__(
        self,
        n_components: int = 50,
        alpha: float = 1e-2,
        beta: float = 1e-2,
        lam: float = 1e-1,
        k_neighbours: int = 4,
        max_iter: int = 100,
        tol: float = 1e-5,
        rho: float = 1.03,
        mu_init: float = 1e-1,
        mu_max: float = 1e5,
        verbose: bool = False,
    ) -> None:
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.k_neighbours = k_neighbours
        self.max_iter = max_iter
        self.tol = tol
        self.rho = rho
        self.mu_init = mu_init
        self.mu_max = mu_max
        self.verbose = verbose

        self.result_: LADSLResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, labels: np.ndarray) -> "LADSL":
        """Run the ADMM optimisation to learn the subspace from labelled training data.

        The training data should be L2-normalised column-wise before being passed
        in — use ``ladsl.utils.normalize_columns`` for this.  This matches the
        pre-processing protocol described in the paper and keeps the inter-sample
        distances meaningful.

        Args:
            X:      Training data matrix of shape (m, n), where m is the number
                    of features and n is the number of training samples.  Each
                    column is one sample.
            labels: Integer class labels of shape (n,).  Can be 0-indexed or
                    1-indexed; the algorithm only needs them to be consistent.

        Returns:
            self — so the call can be chained with transform().
        """
        self.result_ = self._ladsl(X, labels)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data into the learned discriminative subspace.

        Must be called after fit().  The projection W @ P.T @ X first maps each
        sample into the d-dimensional subspace via P and then applies the learned
        classifier weights W to produce a c-dimensional representation where
        samples from the same class should cluster tightly together.

        After projecting it is good practice to L2-normalise the columns again
        before running nearest-neighbour classification.

        Args:
            X: Data matrix of shape (m, n_test).  m must match the feature
               dimension used during training.

        Returns:
            Projected matrix of shape (c, n_test).

        Raises:
            RuntimeError: If called before fit().
        """
        if self.result_ is None:
            raise RuntimeError("Call fit() before transform().")
        return self.result_.transform(X)

    def fit_transform(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit the model and immediately project the training data.

        Equivalent to calling fit() then transform() on the same data.  Useful
        when you need the projected training representations for a downstream
        classifier or for evaluating convergence.

        Args:
            X:      Training data, shape (m, n).
            labels: Integer labels, shape (n,).

        Returns:
            Projected training data of shape (c, n).
        """
        self.fit(X, labels)
        return self.transform(X)

    # ------------------------------------------------------------------
    # Core ADMM optimisation loop
    # ------------------------------------------------------------------

    def _ladsl(self, X: np.ndarray, labels: np.ndarray) -> LADSLResult:
        """Main ADMM optimisation loop.

        The loop alternates between eight variable updates per iteration.
        Variable naming follows the paper directly:

            W  — linear classifier weights       (closed-form least squares)
            D  — discriminative code matrix      (closed-form least squares)
            A  — auxiliary locality weights      (closed-form)
            Z  — self-expression coefficients    (linear solve)
            J  — low-rank surrogate for Z        (nuclear-norm proximal)
            E  — sparse error                    (L2,1 proximal)
            P  — orthogonal projection           (Procrustes / SVD)
            S  — locality weight matrix          (distance-based re-weighting)

        Convergence is checked after each full iteration via four constraint
        residuals normalised by the initial projection norm.  The penalty mu
        grows geometrically until the maximum is reached.

        Args:
            X:      Training data, shape (m, n).
            labels: Integer class labels, shape (n,).

        Returns:
            A LADSLResult containing all learned matrices and the objective trace.
        """
        d = self.n_components
        m, n = X.shape
        mu = self.mu_init
        max_mu = self.mu_max

        # Label indicator matrix H (c x n): H[c, i] = 1 iff sample i is class c.
        H = encode_labels(labels)
        c = H.shape[0]

        # S1 is the binary same-class adjacency (diagonal zeroed).
        # S is its row-normalised version — updated each iteration.
        S1 = H.T @ H
        np.fill_diagonal(S1, 0.0)
        row_sums = np.sum(S1, axis=1, keepdims=True)
        S = S1 / (row_sums + 1e-12)

        # Initialise primal and dual variables to zero.
        E  = np.zeros((d, n))
        Y1 = np.zeros((d, n))   # dual for  P'X = P'XZ + E
        Y2 = np.zeros((n, n))   # dual for  Z = J
        Y3 = np.zeros((d, n))   # dual for  D = P'X
        Y4 = np.zeros((c, m))   # dual for  A = W P'

        Z = np.zeros((n, n))
        J = np.zeros((n, n))
        W = np.zeros((c, d))
        A = np.zeros((c, m))

        # Warm-start P with the leading PCA directions — much faster to converge
        # than a random or zero initialisation.
        P = pca(X.T, d)   # X.T is (n, m); pca() expects rows = samples
        D = P.T @ X       # initial discriminative codes, (d, n)

        # Pre-factorise (X'X + I)^{-1} — used in the Z update every iteration.
        XX_inv = np.linalg.inv(X.T @ X + np.eye(n))

        normfX = np.linalg.norm(P.T @ X, "fro")
        objective: list[float] = []
        classes = np.unique(labels)

        for iteration in range(self.max_iter):
            PX = P.T @ X   # current projection of training data, shape (d, n)

            # ------------------------------------------------------------------
            # Update W — linear classifier weights (c x d)
            # Closed-form least-squares with an augmented Lagrangian term.
            # ------------------------------------------------------------------
            lhs = D @ D.T + (1 + mu) * np.eye(d)
            rhs = H @ D.T + mu * (Y4 / mu + A) @ P
            W = rhs @ np.linalg.inv(lhs)

            # ------------------------------------------------------------------
            # Build within-class scatter matrix Sw (m x m)
            # Sw penalises configurations where close same-class samples in
            # A-space drift apart.  The locality weights (sub_S) come from S,
            # which is updated at the end of this iteration so Sw adapts to
            # the evolving feature representation.
            # ------------------------------------------------------------------
            Sw = np.zeros((m, m))
            for cls in classes:
                idx = np.where(labels == cls)[0]
                sub_S = S[np.ix_(idx, idx)]
                Xc = X[:, idx]

                QQ = sub_S ** 2
                D1 = np.diag(QQ.sum(axis=1))
                D2 = np.diag(QQ.sum(axis=0))
                L_c = D1 + D2 - QQ - QQ.T   # per-class graph Laplacian
                Sw += len(idx) * (Xc @ L_c @ Xc.T)

            # ------------------------------------------------------------------
            # Update D — discriminative code matrix (d x n)
            # ------------------------------------------------------------------
            Arg_D = W.T @ W + mu * np.eye(d)
            D = np.linalg.solve(Arg_D, W.T @ H + mu * (PX - Y3 / mu))

            # ------------------------------------------------------------------
            # Update A — auxiliary locality weight matrix (c x m)
            # ------------------------------------------------------------------
            Arg_A = 2 * self.alpha * Sw + mu * np.eye(m)
            A = mu * (W @ P.T - Y4 / mu) @ np.linalg.inv(Arg_A)

            # ------------------------------------------------------------------
            # Update Z — self-expression coefficients (n x n)
            # ------------------------------------------------------------------
            B1 = PX - E + Y1 / mu
            B2 = J - Y2 / mu
            Z = XX_inv @ (X.T @ P @ B1 + B2)

            # ------------------------------------------------------------------
            # Update J — low-rank surrogate for Z (n x n)
            # Nuclear-norm proximal step: soft-threshold the singular values.
            # ------------------------------------------------------------------
            J = _singular_value_threshold(Z + Y2 / mu, self.lam / mu)

            # ------------------------------------------------------------------
            # Update E — sparse error matrix (d x n)
            # L2,1 proximal step: column-wise L2 shrinkage.
            # ------------------------------------------------------------------
            temp = PX - PX @ Z + Y1 / mu
            E = _solve_l21(temp, self.beta / mu)

            # ------------------------------------------------------------------
            # Update P — orthogonal projection (m x d), Stiefel manifold
            # The gradient w.r.t. P is assembled and then projected back onto
            # the manifold of orthonormal frames via a compact SVD (Procrustes).
            # ------------------------------------------------------------------
            AA = (
                (X - X @ Z) @ (E - Y1 / mu).T
                + X @ D.T
                + X @ (Y3 / mu).T
                + (A + Y4 / mu).T @ W
            )
            U_svd, _, Vt_svd = scipy_svd(AA + 1e-15, full_matrices=False)
            P = U_svd[:, :d] @ Vt_svd[:d, :]

            # ------------------------------------------------------------------
            # Update S — locality-aware affinity weights (n x n)
            # Invert the current A-space distances and mask by S1 so that only
            # same-class pairs get non-zero entries.  Row-normalise so that the
            # weights for each sample sum to 1.
            # ------------------------------------------------------------------
            WPX = A @ X   # (c, n) — training samples in the current A-space
            dist_WX = l2_distance(WPX, WPX)
            inv_dist = 1.0 / (dist_WX + 1e-12)
            Dis_WX = inv_dist * S1
            row_max = np.maximum(Dis_WX.sum(axis=1, keepdims=True), 1e-10)
            S = Dis_WX / row_max

            # ------------------------------------------------------------------
            # Objective — logged at every iteration for convergence diagnostics.
            # Rank(Z) is replaced by its nuclear-norm relaxation.
            # ------------------------------------------------------------------
            v1 = np.sqrt(np.sum(E ** 2, axis=1) + 1e-12)
            nuclear_Z = float(np.sum(scipy_svd(Z, compute_uv=False)))
            obj = (
                self.lam * nuclear_Z
                + 0.5 * np.linalg.norm(H - W @ D, "fro") ** 2
                + self.beta * float(np.sum(1.0 / v1))
                + self.alpha * float(np.trace(A @ Sw @ A.T))
            )
            objective.append(obj)

            # ------------------------------------------------------------------
            # Convergence check — four normalised constraint residuals.
            # ------------------------------------------------------------------
            dY1 = PX - PX @ Z - E
            dY2 = Z - J
            dY3 = D - PX
            dY4 = A - W @ P.T

            err1 = np.linalg.norm(dY1, "fro") / (normfX + 1e-12)
            err2 = np.linalg.norm(dY2, "fro") / (normfX + 1e-12)
            err3 = np.linalg.norm(dY3, "fro") / (normfX + 1e-12)
            err4 = np.linalg.norm(dY4, "fro") / (normfX + 1e-12)
            rec_err = max(err1, err2, err3, err4)

            if self.verbose:
                print(f"Iter {iteration + 1:4d} | obj={obj:.4e} | err={rec_err:.2e}")

            if rec_err < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}.")
                break

            # Standard ADMM dual variable and penalty updates.
            Y1 = Y1 + mu * dY1
            Y2 = Y2 + mu * dY2
            Y3 = Y3 + mu * dY3
            Y4 = Y4 + mu * dY4
            mu = min(max_mu, mu * self.rho)
        else:
            warnings.warn(
                f"LADSL did not converge within {self.max_iter} iterations "
                f"(final residual = {rec_err:.2e}).  "
                f"Try increasing max_iter or relaxing tol.",
                RuntimeWarning,
                stacklevel=2,
            )

        return LADSLResult(Z=Z, E=E, P=P, D=D, W=W, A=A, objective=objective)

    # ------------------------------------------------------------------
    # scikit-learn compatibility helpers
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> dict:
        """Return all constructor parameters as a dictionary.

        Follows the scikit-learn estimator convention so that LADSL can be
        wrapped in a Pipeline or used with GridSearchCV.

        Args:
            deep: Ignored (included for API compatibility).

        Returns:
            Dictionary mapping parameter names to their current values.
        """
        return {
            "n_components": self.n_components,
            "alpha": self.alpha,
            "beta": self.beta,
            "lam": self.lam,
            "k_neighbours": self.k_neighbours,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "rho": self.rho,
            "mu_init": self.mu_init,
            "mu_max": self.mu_max,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "LADSL":
        """Set one or more constructor parameters by name.

        Follows the scikit-learn estimator convention so that LADSL can be
        used with hyperparameter search utilities.

        Args:
            **params: Keyword arguments matching constructor parameter names.

        Returns:
            self — so calls can be chained.
        """
        for k, v in params.items():
            setattr(self, k, v)
        return self
