"""CSP spatial filtering and log-variance EEG feature extraction."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _normalized_covariances(epochs: np.ndarray) -> np.ndarray:
    covariances = epochs @ np.swapaxes(epochs, 1, 2)
    traces = np.trace(covariances, axis1=1, axis2=2)
    if np.any(traces <= 0):
        raise ValueError("Every epoch must have positive signal energy")
    return covariances / traces[:, None, None]


class OneVsRestCSP(BaseEstimator, TransformerMixin):
    """Multi-class CSP using one binary generalized eigensystem per class.

    Only training epochs are used in ``fit``. Each binary model contributes
    ``n_components`` filters from both ends of its eigenvalue spectrum.
    """

    def __init__(self, n_components: int = 4, regularization: float = 1e-10):
        self.n_components = n_components
        self.regularization = regularization

    def fit(self, epochs: np.ndarray, labels: np.ndarray) -> "OneVsRestCSP":
        epochs = np.asarray(epochs, dtype=np.float64)
        labels = np.asarray(labels, dtype=int)
        if epochs.ndim != 3:
            raise ValueError("epochs must have shape (trials, channels, samples)")
        if epochs.shape[0] != labels.size:
            raise ValueError("The number of epochs and labels must match")
        self.classes_ = np.unique(labels)
        if self.classes_.size < 2:
            raise ValueError("CSP requires at least two classes")
        n_channels = epochs.shape[1]
        if not 1 <= self.n_components <= n_channels // 2:
            raise ValueError("n_components must be between 1 and channels // 2")
        covariances = _normalized_covariances(epochs)
        filters = []
        eigenvalues = []
        patterns = []
        for class_label in self.classes_:
            target = covariances[labels == class_label].mean(axis=0)
            rest = covariances[labels != class_label].mean(axis=0)
            composite = target + rest
            composite += self.regularization * np.trace(composite) / n_channels * np.eye(n_channels)
            values, vectors = eigh(target, composite, check_finite=True)
            selected = np.r_[np.arange(self.n_components), np.arange(n_channels - self.n_components, n_channels)]
            class_filters = vectors[:, selected].T
            filters.append(class_filters)
            eigenvalues.append(values[selected])
            patterns.append(np.linalg.pinv(class_filters))
        self.filters_ = np.stack(filters)
        self.eigenvalues_ = np.stack(eigenvalues)
        self.spatial_patterns_ = np.stack(patterns)
        self.n_features_out_ = self.classes_.size * 2 * self.n_components
        return self

    def transform(self, epochs: np.ndarray) -> np.ndarray:
        if not hasattr(self, "filters_"):
            raise RuntimeError("CSP must be fitted before transform")
        epochs = np.asarray(epochs, dtype=np.float64)
        class_features = []
        for filters in self.filters_:
            projected = np.einsum("kc,nct->nkt", filters, epochs, optimize=True)
            variances = np.var(projected, axis=2)
            variances = np.maximum(variances, np.finfo(float).tiny)
            normalized = variances / variances.sum(axis=1, keepdims=True)
            class_features.append(np.log(normalized))
        return np.concatenate(class_features, axis=1)


class LogVariance(BaseEstimator, TransformerMixin):
    """Non-spatial baseline: one log-variance feature per EEG channel."""

    def fit(self, epochs: np.ndarray, labels: np.ndarray | None = None) -> "LogVariance":
        return self

    def transform(self, epochs: np.ndarray) -> np.ndarray:
        variances = np.var(np.asarray(epochs, dtype=np.float64), axis=2)
        return np.log(np.maximum(variances, np.finfo(float).tiny))


def make_csp_classifier(n_components: int = 4):
    """Create the fixed, subject-specific CSP + shrinkage LDA pipeline."""
    return make_pipeline(
        OneVsRestCSP(n_components=n_components),
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def make_baseline_classifier():
    return make_pipeline(
        LogVariance(),
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
