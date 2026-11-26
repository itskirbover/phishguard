"""From-scratch Character N-Gram TF-IDF Vectorizer implemented in Python + NumPy.

Character n-grams capture obfuscated phishing patterns (such as ``paypa1``,
``-login-``, ``.tk/``, ``webscr``) that word-level tokenizers miss, while
remaining compact enough for fast offline inference and JSON serialization.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Sequence

import numpy as np


class CharNgramTfidfVectorizer:
    """Character n-gram TF-IDF vectorizer with sublinear TF and L2 normalization."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (3, 5),
        max_features: int = 512,
        min_df: int = 2,
        sublinear_tf: bool = True,
    ) -> None:
        if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
            raise ValueError(f"Invalid ngram_range: {ngram_range}")
        if max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {max_features}")
        if min_df < 1:
            raise ValueError(f"min_df must be >= 1, got {min_df}")

        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf

        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray = np.array([], dtype=np.float64)

    def _extract_ngrams(self, text: str) -> Counter[str]:
        """Extract character n-grams from normalized lowercase text."""
        normalized = f"^{text.strip().lower()}$"
        counts: Counter[str] = Counter()
        min_n, max_n = self.ngram_range
        length = len(normalized)
        for n in range(min_n, max_n + 1):
            if length < n:
                continue
            for i in range(length - n + 1):
                counts[normalized[i : i + n]] += 1
        return counts

    def fit(self, documents: Sequence[str]) -> CharNgramTfidfVectorizer:
        """Learn character n-gram vocabulary and smooth IDF weights."""
        n_docs = len(documents)
        if n_docs == 0:
            raise ValueError("Cannot fit CharNgramTfidfVectorizer on empty corpus.")

        doc_freq: Counter[str] = Counter()
        total_freq: Counter[str] = Counter()

        for doc in documents:
            counts = self._extract_ngrams(doc)
            for gram, cnt in counts.items():
                doc_freq[gram] += 1
                total_freq[gram] += cnt

        # Filter by minimum document frequency; if corpus is tiny, relax gracefully.
        effective_min_df = min(self.min_df, n_docs)
        candidates = [
            (gram, df, total_freq[gram])
            for gram, df in doc_freq.items()
            if df >= effective_min_df
        ]
        if not candidates:
            candidates = [(gram, df, total_freq[gram]) for gram, df in doc_freq.items()]

        # Deterministic ranking: highest document frequency, then total frequency, then lex order.
        candidates.sort(key=lambda item: (-item[1], -item[2], item[0]))
        selected = candidates[: self.max_features]

        self.vocabulary_ = {gram: idx for idx, (gram, _, _) in enumerate(selected)}
        # Smooth IDF: log((1 + N) / (1 + df)) + 1.0 (never divides by zero)
        self.idf_ = np.array(
            [math.log((1.0 + n_docs) / (1.0 + df)) + 1.0 for _, df, _ in selected],
            dtype=np.float64,
        )
        return self

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        """Transform documents into an L2-normalized dense TF-IDF float64 matrix."""
        if not self.vocabulary_:
            raise RuntimeError("Vectorizer must be fitted before calling transform().")

        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        matrix = np.zeros((n_docs, n_features), dtype=np.float64)

        for row_idx, doc in enumerate(documents):
            counts = self._extract_ngrams(doc)
            for gram, cnt in counts.items():
                col_idx = self.vocabulary_.get(gram)
                if col_idx is not None:
                    tf = 1.0 + math.log(cnt) if self.sublinear_tf else float(cnt)
                    matrix[row_idx, col_idx] = tf * self.idf_[col_idx]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0.0
        matrix[nonzero] /= norms[nonzero]
        return matrix

    def fit_transform(self, documents: Sequence[str]) -> np.ndarray:
        """Fit to documents and return their TF-IDF matrix."""
        return self.fit(documents).transform(documents)

    def feature_names(self) -> list[str]:
        """Return ordered feature names prefixed with 'ngram:'."""
        ordered = [""] * len(self.vocabulary_)
        for gram, idx in self.vocabulary_.items():
            ordered[idx] = f"ngram:{gram}"
        return ordered

    def to_dict(self) -> dict[str, Any]:
        """Serialize vectorizer parameters to a JSON-safe dictionary."""
        return {
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "min_df": self.min_df,
            "sublinear_tf": self.sublinear_tf,
            "vocabulary": self.vocabulary_,
            "idf": self.idf_.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CharNgramTfidfVectorizer:
        """Reconstruct a fitted vectorizer from a JSON dictionary."""
        ngram_range = tuple(int(x) for x in payload["ngram_range"])
        vec = cls(
            ngram_range=(ngram_range[0], ngram_range[1]),
            max_features=int(payload["max_features"]),
            min_df=int(payload["min_df"]),
            sublinear_tf=bool(payload.get("sublinear_tf", True)),
        )
        vec.vocabulary_ = {str(k): int(v) for k, v in payload["vocabulary"].items()}
        vec.idf_ = np.array(payload["idf"], dtype=np.float64)
        return vec
