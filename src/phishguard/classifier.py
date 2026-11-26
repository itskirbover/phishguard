"""From-scratch L2-regularized Logistic Regression & feature scaler in NumPy.

Security design note:
    Model weights are serialized exclusively as plain JSON (never Python
    ``pickle``), eliminating arbitrary code execution risks when loading
    untrusted model checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid avoiding overflow on large |z|."""
    clipped = np.clip(z, -500.0, 500.0)
    return np.where(
        clipped >= 0.0,
        1.0 / (1.0 + np.exp(-clipped)),
        np.exp(clipped) / (1.0 + np.exp(clipped)),
    )


class StandardScaler:
    """Zero-mean, unit-variance feature scaler with JSON serialization."""

    def __init__(self) -> None:
        self.mean_: np.ndarray = np.array([], dtype=np.float64)
        self.scale_: np.ndarray = np.array([], dtype=np.float64)

    def fit(self, x: np.ndarray) -> StandardScaler:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError("StandardScaler expects a non-empty 2D array.")
        self.mean_ = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        # Avoid division by zero for constant features
        self.scale_ = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_.size == 0:
            raise RuntimeError("StandardScaler must be fitted before transform().")
        arr = np.asarray(x, dtype=np.float64)
        return (arr - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StandardScaler:
        scaler = cls()
        scaler.mean_ = np.array(payload["mean"], dtype=np.float64)
        scaler.scale_ = np.array(payload["scale"], dtype=np.float64)
        return scaler


@dataclass(frozen=True)
class FeatureContribution:
    """Per-feature linear log-odds contribution for a single prediction."""

    feature: str
    value: float
    weight: float
    contribution: float


class LogisticRegressionClassifier:
    """Binary L2-regularized logistic regression trained with Adam in NumPy."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        l2_reg: float = 0.01,
        max_iter: int = 500,
        tol: float = 1e-6,
        balanced: bool = True,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if l2_reg < 0.0:
            raise ValueError("l2_reg must be non-negative.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")

        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.tol = tol
        self.balanced = balanced

        self.weights_: np.ndarray = np.array([], dtype=np.float64)
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        if not self.balanced:
            return np.ones_like(y, dtype=np.float64)
        n_samples = float(len(y))
        n_pos = float(np.sum(y == 1.0))
        n_neg = n_samples - n_pos
        if n_pos == 0.0 or n_neg == 0.0:
            return np.ones_like(y, dtype=np.float64)
        w_pos = n_samples / (2.0 * n_pos)
        w_neg = n_samples / (2.0 * n_neg)
        return np.where(y == 1.0, w_pos, w_neg)

    def fit(self, x: np.ndarray, y: Sequence[int] | np.ndarray) -> LogisticRegressionClassifier:
        """Fit weights and bias using Adam gradient descent with L2 regularization."""
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        if x_arr.ndim != 2 or x_arr.shape[0] == 0:
            raise ValueError("Feature matrix X must be a non-empty 2D array.")
        if y_arr.ndim != 1 or y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError("Target vector y must be 1D and match X rows.")

        n_samples, n_features = x_arr.shape
        self.weights_ = np.zeros(n_features, dtype=np.float64)
        self.bias_ = 0.0
        self.loss_history_ = []

        sample_weights = self._compute_sample_weights(y_arr)
        weight_norm = float(np.sum(sample_weights))

        # Adam moments
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_w = np.zeros(n_features, dtype=np.float64)
        v_w = np.zeros(n_features, dtype=np.float64)
        m_b = 0.0
        v_b = 0.0

        prev_loss = float("inf")
        for step in range(1, self.max_iter + 1):
            logits = x_arr @ self.weights_ + self.bias_
            probs = sigmoid(logits)

            # Numerically stable binary cross-entropy: log(1 + exp(z)) - y*z
            bce = np.logaddexp(0.0, logits) - y_arr * logits
            data_loss = float(np.sum(sample_weights * bce) / weight_norm)
            reg_loss = 0.5 * self.l2_reg * float(np.sum(self.weights_**2))
            total_loss = data_loss + reg_loss
            self.loss_history_.append(total_loss)

            # Gradients (L2 penalty applies only to weights, not bias)
            errors = (probs - y_arr) * sample_weights / weight_norm
            grad_w = (x_arr.T @ errors) + self.l2_reg * self.weights_
            grad_b = float(np.sum(errors))

            # Adam parameter update
            m_w = beta1 * m_w + (1.0 - beta1) * grad_w
            v_w = beta2 * v_w + (1.0 - beta2) * (grad_w**2)
            m_b = beta1 * m_b + (1.0 - beta1) * grad_b
            v_b = beta2 * v_b + (1.0 - beta2) * (grad_b**2)

            m_w_hat = m_w / (1.0 - beta1**step)
            v_w_hat = v_w / (1.0 - beta2**step)
            m_b_hat = m_b / (1.0 - beta1**step)
            v_b_hat = v_b / (1.0 - beta2**step)

            self.weights_ -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
            self.bias_ -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

            if abs(prev_loss - total_loss) < self.tol:
                break
            prev_loss = total_loss

        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Compute raw log-odds (z = X @ w + b)."""
        if self.weights_.size == 0:
            raise RuntimeError("Classifier must be fitted before inference.")
        x_arr = np.asarray(x, dtype=np.float64)
        return x_arr @ self.weights_ + self.bias_

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return probability of the positive (phishing) class for each row."""
        return sigmoid(self.decision_function(x))

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions (0 = benign, 1 = phishing)."""
        return (self.predict_proba(x) >= threshold).astype(np.int64)

    def explain_row(
        self,
        x_row: np.ndarray,
        feature_names: Sequence[str],
    ) -> tuple[float, list[FeatureContribution]]:
        """Compute exact additive log-odds contributions for a single feature vector.

        Satisfies the exact linear identity:
            raw_logit == bias + sum(c.contribution for c in contributions)
        """
        if self.weights_.size == 0:
            raise RuntimeError("Classifier must be fitted before explain_row().")
        row = np.asarray(x_row, dtype=np.float64).ravel()
        if row.shape[0] != self.weights_.shape[0]:
            raise ValueError("Feature row length does not match model weights.")
        if len(feature_names) != row.shape[0]:
            raise ValueError("feature_names length does not match feature row.")

        contributions = [
            FeatureContribution(
                feature=name,
                value=float(val),
                weight=float(weight),
                contribution=float(val * weight),
            )
            for name, val, weight in zip(feature_names, row, self.weights_)
        ]
        raw_logit = float(self.bias_ + np.dot(row, self.weights_))
        return raw_logit, contributions

    def to_dict(self) -> dict[str, Any]:
        """Serialize classifier parameters to a JSON-safe dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "l2_reg": self.l2_reg,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "balanced": self.balanced,
            "weights": self.weights_.tolist(),
            "bias": float(self.bias_),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LogisticRegressionClassifier:
        """Reconstruct a fitted classifier from a JSON dictionary."""
        clf = cls(
            learning_rate=float(payload.get("learning_rate", 0.05)),
            l2_reg=float(payload.get("l2_reg", 0.01)),
            max_iter=int(payload.get("max_iter", 500)),
            tol=float(payload.get("tol", 1e-6)),
            balanced=bool(payload.get("balanced", True)),
        )
        clf.weights_ = np.array(payload["weights"], dtype=np.float64)
        clf.bias_ = float(payload["bias"])
        return clf
