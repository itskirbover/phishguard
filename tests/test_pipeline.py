"""End-to-end tests for dataset loading, zero-leakage CV, and PhishGuardPipeline."""

from __future__ import annotations

from pathlib import Path

from phishguard.dataset import generate_benchmark_dataset, load_csv_dataset, save_csv_dataset
from phishguard.pipeline import PhishGuardPipeline


def test_csv_dataset_roundtrip_and_flexible_labels(tmp_path: Path) -> None:
    csv_file = tmp_path / "urls.csv"
    csv_file.write_text(
        "url,label\n"
        "https://github.com/python/cpython,benign\n"
        "https://docs.python.org/3/,0\n"
        "http://paypal.verify-account.tk/login,phishing\n"
        "http://198.51.100.20/apple/signin,1\n",
        encoding="utf-8",
    )
    ds = load_csv_dataset(csv_file)
    assert len(ds) == 4
    assert ds.labels == [0, 0, 1, 1]

    out_file = tmp_path / "out.csv"
    save_csv_dataset(ds, out_file)
    reloaded = load_csv_dataset(out_file)
    assert reloaded.urls == ds.urls
    assert reloaded.labels == ds.labels


def test_default_model_loads_and_distinguishes_hard_negatives() -> None:
    pipe = PhishGuardPipeline.load_default()

    benign_urls = [
        "https://github.com/SimplifyJobs/Summer2027-Internships",
        "https://accounts.google.com/signin/v2/identifier",
        "https://www.paypal.com/signin",
        "https://docs.python.org/3/library/index.html",
    ]
    phish_urls = [
        "https://paypal.com.verify-account.tk/login/confirm",
        "http://198.51.100.42:8080/microsoft/login/verify",
        "https://arnazon-security-update.xyz/account/billing",
        "https://apple.com@secure-login-9921.click/signin",
    ]

    for url in benign_urls:
        res = pipe.predict_one(url)
        assert res.verdict == "BENIGN", f"Expected BENIGN for {url}, got {res.verdict} ({res.probability})"
        assert res.probability < pipe.threshold

    for url in phish_urls:
        res = pipe.predict_one(url)
        assert res.verdict == "PHISHING", f"Expected PHISHING for {url}, got {res.verdict} ({res.probability})"
        assert res.probability >= pipe.threshold
        assert len(res.top_risk_factors) > 0
        assert len(res.security_signals) > 0


def test_pipeline_cv_training_and_json_persistence(tmp_path: Path) -> None:
    ds = generate_benchmark_dataset(n_samples=300, seed=99)
    pipe = PhishGuardPipeline(max_ngram_features=128, max_iter=200)
    cv = pipe.cross_validate(ds.urls, ds.labels, n_splits=3, seed=99)

    assert len(cv.fold_metrics) == 3
    assert cv.aggregate_metrics.roc_auc >= 0.95
    assert cv.aggregate_metrics.f1 >= 0.90

    pipe.threshold = cv.recommended_threshold
    pipe.fit(ds.urls, ds.labels)

    model_path = tmp_path / "model.json"
    pipe.save(model_path)
    loaded = PhishGuardPipeline.load(model_path)

    sample = "https://paypa1-verify.tk/account/login"
    r1 = pipe.predict_one(sample)
    r2 = loaded.predict_one(sample)
    assert abs(r1.probability - r2.probability) < 1e-12
    assert r1.verdict == r2.verdict
