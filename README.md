# LADSL — Locality-Aware Discriminative Subspace Learning

> **Python implementation of the algorithm presented in:**
>
> Meenakshi and S. Srirangarajan, "Locality-Aware Discriminative Subspace Learning for Image Classification," in IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1-14, 2022, Art no. 5015414, doi: 10.1109/TIM.2022.3187735

> [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9812722)

---

## What is LADSL?

Most classical subspace learning methods treat dimensionality reduction and
classification as two separate steps. LADSL does both at once. It finds a
low-dimensional projection that is simultaneously:

- **Discriminative** — class labels are linearly predictable in the subspace.
- **Locality-preserving** — samples from the same class that are geometrically
  close in the original space stay close after projection.
- **Robust to noise** — a sparse error term absorbs structured corruption so
  that a few outlier samples or occluded pixels do not break the subspace.
- **Globally coherent** — a low-rank self-expression constraint enforces that
  all same-class samples share a common underlying subspace structure.

These four objectives are optimised jointly via **ADMM** (Alternating Direction
Method of Multipliers), with the locality weights updated at every iteration so
the graph adapts to the evolving feature representation instead of being fixed
at the beginning.

After fitting, classifying a new sample is as simple as projecting it with
`W @ P.T` and then finding the nearest neighbour in the training set.

---

## Installation

```bash
git clone https://github.com/<your-username>/ladsl.git
cd ladsl

# Recommended: create an isolated environment first
conda create -n ladsl_env python=3.11
conda activate ladsl_env

pip install -e ".[dev]"
```

**Dependencies:** Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10, scikit-learn ≥ 1.3, Matplotlib ≥ 3.7.

---

## Quick Start

```python
import numpy as np
from ladsl import LADSL, normalize_columns, knn_classifier

# X_train : (n_features, n_train)  — each column is one sample
# y_train : (n_train,)             — integer class labels

# Pre-process: L2-normalise each sample (matches the paper's protocol)
X_train = normalize_columns(X_train)
X_test  = normalize_columns(X_test)

# Fit the model
model = LADSL(
    n_components=50,   # subspace dimension d
    alpha=1e-2,        # locality scatter weight
    beta=1e-2,         # sparse error weight
    lam=1e-1,          # low-rank (nuclear norm) weight
    max_iter=100,
    verbose=True,
)
model.fit(X_train, y_train)

# Project into the learned subspace and classify
proj_train = normalize_columns(model.transform(X_train))
proj_test  = normalize_columns(model.transform(X_test))

accuracy, predicted = knn_classifier(proj_train, y_train, proj_test, y_test, k=1)
print(f"1-NN accuracy: {accuracy:.2f}%")
```

---

## Repository Layout

```
ladsl/
├── ladsl/
│   ├── __init__.py          Public API
│   ├── model.py             LADSL class — ADMM optimisation loop
│   └── utils.py             PCA, KNN, distances, graph construction,
│                            label encoding, column normalisation
├── tests/
│   ├── test_model.py        Model correctness and shape tests
│   └── test_utils.py        Unit tests for every utility function
├── examples/
│   └── demo_synthetic.py    End-to-end demo (100 % accuracy on toy data)
├── pyproject.toml
└── README.md
```

---

## Running the Tests

```bash
pytest                                        # all 17 tests
pytest --cov=ladsl --cov-report=term-missing  # with coverage
```

---

## Algorithm Details

### Objective Function

The model jointly minimises:

$$
\min_{P,\, D,\, W,\, A,\, Z,\, E} \;
\underbrace{\lambda \|Z\|_*}_{\text{low-rank}}
+ \underbrace{\frac{1}{2}\|H - WD\|_F^2}_{\text{discriminative}}
+ \underbrace{\beta\|E\|_{2,1}}_{\text{sparse noise}}
+ \underbrace{\alpha \operatorname{tr}(A\, S_w\, A^\top)}_{\text{locality}}
$$

subject to:

$$
P^\top X \;=\; P^\top X Z + E, \qquad
D \;=\; P^\top X, \qquad
A \;=\; W P^\top
$$

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `X` | m × n | Training data (columns = samples) |
| `H` | c × n | One-hot label indicator matrix |
| `P` | m × d | Orthogonal projection (Stiefel manifold) |
| `D` | d × n | Discriminative codes in the subspace |
| `W` | c × d | Linear classifier weights |
| `A` | c × m | Auxiliary locality-coupling matrix |
| `Z` | n × n | Low-rank self-expression coefficients |
| `E` | d × n | Sparse structured error |
| `S_w`| m × m | Locality-aware within-class scatter matrix |

The scatter matrix **S_w** is re-weighted at every ADMM iteration based on
distances in the current **A**-space, so nearby same-class samples receive
higher penalty — the key locality-aware novelty of the method.

### ADMM Update Order

Each iteration performs eight closed-form / proximal updates:

| Step | Variable | Operation |
|------|----------|-----------|
| 1 | **W** | Regularised least squares |
| 2 | **S_w** | Recompute per-class graph Laplacian |
| 3 | **D** | Regularised least squares |
| 4 | **A** | Regularised least squares |
| 5 | **Z** | Linear solve with pre-factored (X'X + I) |
| 6 | **J** | Nuclear-norm proximal (soft SVT) |
| 7 | **E** | L2,1 proximal (column-wise L2 shrinkage) |
| 8 | **P** | Procrustes / polar decomposition (SVD) |
| 9 | **S** | Distance-based re-weighting of same-class pairs |

### Recommended Hyperparameters

These are the dataset-specific settings reported in the paper:

| Dataset | `alpha` | `beta` | `lam` | `n_components` |
|---------|---------|--------|-------|----------------|
| CMU PIE | 1e-2 | 1e-1 | 1e-1 | 50 |
| Extended Yale B | 1e-2 | 1e-3 | 1e-3 | 50 |
| ORL | 1e6 | 1e-1 | 1e-4 | 100 |
| AR Face | 1.0 | 1e-2 | 1e-4 | 200 |
| COIL-20/100 | 1.0 | 1e-5 | 1e-5 | 150 |

---

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{sharma2022ladsl,
  title   = {Locality-Aware Discriminative Subspace Learning for Image Classification},
  journal = {IEEE Transactions on Image Processing},
  year    = {2022},
  doi     = {10.1109/TIP.2022.3185242},
  url     = {https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9812722}
}
```

---

## License

MIT
