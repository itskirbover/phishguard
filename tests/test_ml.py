"""Unit tests for TF-IDF vectorizer, LogisticRegressionClassifier, and metrics."""

from __future__ import annotations

import json
import math

import numpy as np

from phishguard.classifier import LogisticRegressionClassifier, StandardScaler, sigmoid
from phishguard.metrics import (
    compute_pr_auc,
    compute_roc_auc,
    evaluate_predictions,
    find_optimal_threshold,
    stratified_kfold_indices,
)
from phishguard.vectorizer import CharNgramTfidfVectorizer


def test_tfidf_vectorizer_normalization_and_roundtrip() -> None:
    corpus = [
        "https://paypal-verify.tk/login",
        "https://paypal-secure.xyz/signin",
        "https://github.com/python/cpython",
        "https://docs.python.org/3/library/",
    ]
    vec = CharNgramTfidfVectorizer(ngram_range=(3, 4), max_features=64, min_df=1)
    matrix = vec.fit_transform(corpus)

    assert matrix.shape[0] == len(corpus)
    assert matrix.shape[1] <= 64
    # Every non-empty document should be L2-normalized to unit length
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    # JSON round-trip must produce identical vectors
    restored = CharNgramTfidfVectorizer.from_dict(json.loads(json.dumps(vec.to_dict())))
    matrix_restored = restored.transform(corpus)
    assert np.allclose(matrix, matrix_restored, atol=1e-12)
    assert restored.feature_names() == vec.feature_names()


def test_sigmoid_numerical_stability() -> None:
    extreme = np.array([-10000.0, -500.0, 0.0, 500.0, 10000.0])
    out = sigmoid(extreme)
    assert np.all(np.isfinite(out))
    assert math.isclose(float(out[2]), 0.5, rel_tol=1e-12)
    assert out[0] < 1e-100
    assert out[-1] > 0.999999


def test_standard_scaler_handles_constant_feature() -> None:
    x = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    assert np.all(np.isfinite(scaled))
    assert np.allclose(scaled[:, 1], 0.0)

    restored = StandardScaler.from_dict(scaler.to_dict())
    assert np.allclose(restored.transform(x), scaled)


def test_logistic_regression_converges_explains_and_serializes() -> None:
    rng = np.random.default_rng(7)
    x_neg = rng.normal(loc=-2.0, scale=0.7, size=(60, 4))
    x_pos = rng.normal(loc=2.0, scale=0.7, size=(60, 4))
    x = np.vstack([x_neg, x_pos])
    y = np.array([0] * 60 + [1] * 60)

    clf = LogisticRegressionClassifier(learning_rate=0.05, l2_reg=0.01, max_iter=300)
    clf.fit(x, y)

    assert clf.loss_history_[-1] < clf.loss_history_[0]
    preds = clf.predict(x)
    assert float(np.mean(preds == y)) >= 0.95

    # Exact additive log-odds explainability identity
    names = ["f0", "f1", "f2", "f3"]
    raw_logit, contribs = clf.explain_row(x[0], names)
    reconstructed = clf.bias_ + sum(c.contribution for c in contribs)
    assert math.isclose(raw_logit, reconstructed, rel_tol=1e-9, abs_tol=1e-9)

    # Lossless JSON round-trip
    restored = LogisticRegressionClassifier.from_dict(json.loads(json.dumps(clf.to_dict())))
    assert np.allclose(clf.predict_proba(x), restored.predict_proba(x), atol=1e-12)


def test_metrics_roc_pr_and_threshold_optimization() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_perfect = np.array([0.05, 0.1, 0.2, 0.25, 0.75, 0.8, 0.9, 0.95])
    assert math.isclose(compute_roc_auc(y_true, y_perfect), 1.0, rel_tol=1e-9)
    assert math.isclose(compute_pr_auc(y_true, y_perfect), 1.0, rel_tol=1e-9)

    metrics = evaluate_predictions(y_true, y_perfect, threshold=0.5)
    assert metrics.accuracy == 1.0
    assert metrics.f1 == 1.0
    assert metrics.fpr == 0.0

    # Ties in score must still yield 0.5 ROC-AUC for constant predictions
    assert math.isclose(compute_roc_auc(y_true, np.full(8, 0.5)), 0.5, rel_tol=1e-9)

    best_thr = find_optimal_threshold(y_true, y_perfect, max_fpr=0.0)
    assert 0.25 < best_thr <= 0.75


def test_stratified_kfold_preserves_class_balance() -> None:
    y = np.array([0] * 40 + [1] * 20)
    splits = stratified_kfold_indices(y, n_splits=5, seed=123)
    assert len(splits) == 5

    seen_val: list[int] = []
    for train_idx, val_idx in splits:
        assert len(set(train_idx) & set(val_idx)) == 0
        assert int(np.sum(y[val_idx] == 1)) == 4
        assert int(np.sum(y[val_idx] == 0)) == 8
        seen_val.extend(val_idx.tolist())
    assert sorted(seen_val) == list(range(60))
