"""
LADSL: Locality-Aware Discriminative Subspace Learning

A Python implementation of the LADSL algorithm for image classification,
translated from the original MATLAB code.

Reference:
    Locality-Aware Discriminative Subspace Learning for Image Classification
"""

from .model import LADSL
from .utils import (
    pca,
    construct_weight_matrix,
    knn_classifier,
    l2_distance,
    euclidean_distance,
    normalize_columns,
    encode_labels,
)
from .data import (
    load_orl,
    load_yale_b,
    load_caltech101_spatial,
    train_test_split_by_class,
)

__all__ = [
    "LADSL",
    "pca",
    "construct_weight_matrix",
    "knn_classifier",
    "l2_distance",
    "euclidean_distance",
    "normalize_columns",
    "encode_labels",
    "load_orl",
    "load_yale_b",
    "load_caltech101_spatial",
    "train_test_split_by_class",
]

__version__ = "1.0.0"
