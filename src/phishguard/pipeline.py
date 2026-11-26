"""End-to-end hybrid ML + lexical security pipeline for offline URL scanning.

Combines 18 standardized lexical security features with character n-gram
TF-IDF representations inside a single L2-regularized logistic regression
model, serialized exclusively as plain JSON (never pickle).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from phishguard import __version__
from phishguard.classifier import (
    FeatureContribution,
    LogisticRegressionClassifier,
    StandardScaler,
)
from phishguard.lexical import (
    LEXICAL_FEATURE_NAMES,
    SecuritySignal,
    extract_security_signals,
    lexical_feature_vector,
)
from phishguard.metrics import (
    ClassificationMetrics,
    evaluate_predictions,
    find_optimal_threshold,
    stratified_kfold_indices,
)
from phishguard.url_parser import parse_url
from phishguard.vectorizer import CharNgramTfidfVectorizer

MODEL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScanResult:
    """Explainable scan result for a single URL."""

    url: str
    normalized_url: str
    verdict: str  # "PHISHING", "SUSPICIOUS", "BENIGN"
    probability: float
    risk_score: int
    threshold: float
    raw_logit: float
    bias: float
    latency_ms: float
    top_risk_factors: list[FeatureContribution]
    top_benign_factors: list[FeatureContribution]
    security_signals: list[SecuritySignal]

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "normalized_url": self.normalized_url,
            "verdict": self.verdict,
            "probability": round(self.probability, 6),
            "risk_score": self.risk_score,
            "threshold": round(self.threshold, 4),
            "raw_logit": round(self.raw_logit, 6),
            "bias": round(self.bias, 6),
            "latency_ms": round(self.latency_ms, 3),
            "top_risk_factors": [
                {
                    "feature": c.feature,
                    "value": round(c.value, 4),
                    "weight": round(c.weight, 4),
                    "contribution": round(c.contribution, 4),
                }
                for c in self.top_risk_factors
            ],
            "top_benign_factors": [
                {
                    "feature": c.feature,
                    "value": round(c.value, 4),
                    "weight": round(c.weight, 4),
                    "contribution": round(c.contribution, 4),
                }
                for c in self.top_benign_factors
            ],
            "security_signals": [
                {
                    "rule_id": s.rule_id,
                    "severity": s.severity,
                    "title": s.title,
                    "detail": s.detail,
                }
                for s in self.security_signals
            ],
        }


@dataclass(frozen=True)
class CrossValidationReport:
    """Zero-leakage stratified K-fold cross-validation summary."""

    n_splits: int
    fold_metrics: list[ClassificationMetrics]
    aggregate_metrics: ClassificationMetrics
    recommended_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_splits": self.n_splits,
            "recommended_threshold": round(self.recommended_threshold, 6),
            "aggregate": self.aggregate_metrics.to_dict(),
            "folds": [m.to_dict() for m in self.fold_metrics],
        }


class PhishGuardPipeline:
    """Hybrid lexical + char n-gram TF-IDF phishing classifier pipeline."""

    def __init__(
        self,
        max_ngram_features: int = 384,
        ngram_range: tuple[int, int] = (3, 5),
        min_df: int = 2,
        learning_rate: float = 0.05,
        l2_reg: float = 0.015,
        max_iter: int = 450,
        threshold: float = 0.5,
    ) -> None:
        self.scaler = StandardScaler()
        self.vectorizer = CharNgramTfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_ngram_features,
            min_df=min_df,
        )
        self.classifier = LogisticRegressionClassifier(
            learning_rate=learning_rate,
            l2_reg=l2_reg,
            max_iter=max_iter,
            balanced=True,
        )
        self.threshold = threshold
        self.metadata: dict[str, Any] = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "phishguard_version": __version__,
        }

    def _build_features_fit(self, urls: Sequence[str]) -> np.ndarray:
        parsed_urls = [parse_url(u) for u in urls]
        lexical_raw = np.array(
            [lexical_feature_vector(p) for p in parsed_urls],
            dtype=np.float64,
        )
        lexical_scaled = self.scaler.fit_transform(lexical_raw)
        normalized_texts = [p.normalized or p.raw for p in parsed_urls]
        ngram_matrix = self.vectorizer.fit_transform(normalized_texts)
        return np.hstack([lexical_scaled, ngram_matrix])

    def _build_features_transform(self, urls: Sequence[str]) -> np.ndarray:
        parsed_urls = [parse_url(u) for u in urls]
        lexical_raw = np.array(
            [lexical_feature_vector(p) for p in parsed_urls],
            dtype=np.float64,
        )
        lexical_scaled = self.scaler.transform(lexical_raw)
        normalized_texts = [p.normalized or p.raw for p in parsed_urls]
        ngram_matrix = self.vectorizer.transform(normalized_texts)
        return np.hstack([lexical_scaled, ngram_matrix])

    def feature_names(self) -> list[str]:
        return list(LEXICAL_FEATURE_NAMES) + self.vectorizer.feature_names()

    def fit(
        self,
        urls: Sequence[str],
        labels: Sequence[int],
        metadata: dict[str, Any] | None = None,
    ) -> PhishGuardPipeline:
        """Fit scaler, TF-IDF vectorizer, and logistic regression on labeled URLs."""
        x = self._build_features_fit(urls)
        y = np.asarray(labels, dtype=np.int64)
        self.classifier.fit(x, y)
        self.metadata = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "phishguard_version": __version__,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(urls),
            "n_positive": int(np.sum(y == 1)),
            "n_negative": int(np.sum(y == 0)),
            "n_features": int(x.shape[1]),
        }
        if metadata:
            self.metadata.update(metadata)
        return self

    def predict_proba(self, urls: Sequence[str]) -> np.ndarray:
        """Return phishing probability for each URL."""
        x = self._build_features_transform(urls)
        return self.classifier.predict_proba(x)

    def predict_one(self, url: str, top_k: int = 6) -> ScanResult:
        """Analyze a single URL offline and return an explainable ScanResult."""
        t0 = time.perf_counter()
        parsed = parse_url(url)
        lexical_raw = np.array([lexical_feature_vector(parsed)], dtype=np.float64)
        lexical_scaled = self.scaler.transform(lexical_raw)
        ngram_vec = self.vectorizer.transform([parsed.normalized or parsed.raw])
        combined = np.hstack([lexical_scaled, ngram_vec])[0]

        names = self.feature_names()
        raw_logit, contributions = self.classifier.explain_row(combined, names)
        prob = float(1.0 / (1.0 + np.exp(-np.clip(raw_logit, -500.0, 500.0))))

        signals = extract_security_signals(parsed)
        has_high_signal = any(s.severity == "high" for s in signals)

        if prob >= self.threshold:
            verdict = "PHISHING"
        elif prob >= max(0.30, self.threshold - 0.15) or (prob >= 0.25 and has_high_signal):
            verdict = "SUSPICIOUS"
        else:
            verdict = "BENIGN"

        pos_factors = sorted(
            [c for c in contributions if c.contribution > 1e-4],
            key=lambda c: c.contribution,
            reverse=True,
        )[:top_k]
        neg_factors = sorted(
            [c for c in contributions if c.contribution < -1e-4],
            key=lambda c: c.contribution,
        )[:top_k]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ScanResult(
            url=url,
            normalized_url=parsed.normalized,
            verdict=verdict,
            probability=prob,
            risk_score=int(round(prob * 100)),
            threshold=self.threshold,
            raw_logit=raw_logit,
            bias=self.classifier.bias_,
            latency_ms=latency_ms,
            top_risk_factors=pos_factors,
            top_benign_factors=neg_factors,
            security_signals=signals,
        )

    def predict_batch(self, urls: Sequence[str], top_k: int = 5) -> list[ScanResult]:
        """Analyze a list of URLs and return ScanResult objects."""
        return [self.predict_one(url, top_k=top_k) for url in urls]

    def cross_validate(
        self,
        urls: Sequence[str],
        labels: Sequence[int],
        n_splits: int = 5,
        seed: int = 42,
        max_fpr: float | None = None,
    ) -> CrossValidationReport:
        """Run zero-leakage stratified K-fold cross-validation.

        Each fold fits its own scaler and TF-IDF vocabulary strictly on the
        training split before scoring the held-out validation fold.
        """
        y_arr = np.asarray(labels, dtype=np.int64)
        splits = stratified_kfold_indices(y_arr, n_splits=n_splits, seed=seed)

        oof_scores = np.zeros(len(urls), dtype=np.float64)
        fold_metrics: list[ClassificationMetrics] = []

        for train_idx, val_idx in splits:
            fold_pipe = PhishGuardPipeline(
                max_ngram_features=self.vectorizer.max_features,
                ngram_range=self.vectorizer.ngram_range,
                min_df=self.vectorizer.min_df,
                learning_rate=self.classifier.learning_rate,
                l2_reg=self.classifier.l2_reg,
                max_iter=self.classifier.max_iter,
                threshold=self.threshold,
            )
            train_urls = [urls[i] for i in train_idx]
            val_urls = [urls[i] for i in val_idx]
            fold_pipe.fit(train_urls, y_arr[train_idx].tolist())
            val_probs = fold_pipe.predict_proba(val_urls)
            oof_scores[val_idx] = val_probs
            fold_metrics.append(
                evaluate_predictions(y_arr[val_idx], val_probs, threshold=self.threshold)
            )

        recommended = find_optimal_threshold(y_arr, oof_scores, max_fpr=max_fpr)
        aggregate = evaluate_predictions(y_arr, oof_scores, threshold=recommended)
        return CrossValidationReport(
            n_splits=n_splits,
            fold_metrics=fold_metrics,
            aggregate_metrics=aggregate,
            recommended_threshold=recommended,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full pipeline to a JSON-safe dictionary."""
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "metadata": self.metadata,
            "threshold": float(self.threshold),
            "lexical_features": list(LEXICAL_FEATURE_NAMES),
            "scaler": self.scaler.to_dict(),
            "vectorizer": self.vectorizer.to_dict(),
            "classifier": self.classifier.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PhishGuardPipeline:
        """Reconstruct a trained pipeline from a JSON dictionary."""
        schema_version = int(payload.get("schema_version", 0))
        if schema_version != MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported model schema version: {schema_version} "
                f"(expected {MODEL_SCHEMA_VERSION})."
            )
        pipe = cls(threshold=float(payload.get("threshold", 0.5)))
        pipe.metadata = dict(payload.get("metadata", {}))
        pipe.scaler = StandardScaler.from_dict(payload["scaler"])
        pipe.vectorizer = CharNgramTfidfVectorizer.from_dict(payload["vectorizer"])
        pipe.classifier = LogisticRegressionClassifier.from_dict(payload["classifier"])
        return pipe

    def save(self, path: str | Path) -> Path:
        """Save pipeline to a plain JSON file (never pickle)."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> PhishGuardPipeline:
        """Load pipeline from a plain JSON file."""
        in_path = Path(path)
        with in_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    @classmethod
    def load_default(cls) -> PhishGuardPipeline:
        """Load the bundled pre-trained offline JSON model."""
        model_resource = resources.files("phishguard.data").joinpath("default_model.json")
        with model_resource.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)
