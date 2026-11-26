# PhishGuard

[![CI](https://github.com/itskirbover/phishguard/actions/workflows/ci.yml/badge.svg)](https://github.com/itskirbover/phishguard/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies: NumPy only](https://img.shields.io/badge/dependencies-numpy%20only-green)
![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)

**PhishGuard** is an explainable, 100% offline phishing and malicious URL classifier built from scratch in Python and NumPy. It combines **18 domain-engineered lexical security features** with a **from-scratch character n-gram TF-IDF vectorizer** and **L2-regularized logistic regression (Adam)**, exports **JSON, SARIF 2.1.0, and single-file offline HTML reports**, and optionally runs a **zero-dependency local web UI** on `127.0.0.1` — no cloud server, API keys, or external ML frameworks required.

---

## Why PhishGuard?

Most URL detectors either rely on third-party threat-intel APIs (sending sensitive URLs off-device) or wrap opaque `scikit-learn` / `pickle` pipelines that are hard to explain and unsafe to deserialize. PhishGuard is designed to be compact, transparent, and auditable end-to-end:

- **100% Offline & Zero SSRF Risk**: Never performs DNS lookups or HTTP requests against untrusted URLs. Supports SOC-defanged indicators (`hxxps://`, `[.]`).
- **From-Scratch ML in NumPy**: Character 3–5 gram TF-IDF, feature standardization, Adam-optimized L2 logistic regression with class-balanced weights, stratified K-fold CV (strictly fit inside each fold to prevent data leakage), ROC-AUC/PR-AUC, and FPR-constrained threshold tuning.
- **Exact Additive Explainability**: Because the classifier is linear in standardized lexical + TF-IDF space, `phishguard explain` decomposes every prediction into exact per-feature log-odds contributions (`raw_logit == bias + sum(w_i * x_i)`) alongside human-readable `PG001`–`PG010` security signals.
- **Safe JSON Model Serialization (No `pickle`)**: The bundled 30 KB model and all custom trained checkpoints are stored as plain, schema-versioned JSON — eliminating Python `pickle` arbitrary-code-execution vectors.
- **Self-Contained Tooling**: Works as a standalone CLI, plugs into GitHub Code Scanning via **SARIF 2.1.0**, generates single-file offline HTML dashboards, or launches an optional local browser UI using Python's stdlib `http.server`.

---

## Architecture

```
Untrusted URL (raw or SOC-defanged)
        │
        ▼
┌──────────────────────┐
│ Safe Offline Parser  │  RFC 3986 scheme check, IPv4/IPv6/hex/dec IP,
│ (url_parser.py)      │  punycode/IDN, multi-part TLDs, userinfo (@)
└──────────┬───────────┘
           │
     ┌─────┴──────────────────────────┐
     ▼                                ▼
┌───────────────────────┐   ┌──────────────────────────┐
│ 18 Lexical Features   │   │ Char N-Gram TF-IDF (3-5) │
│ • Shannon entropy     │   │ • Sublinear TF           │
│ • Levenshtein typos   │   │ • Smooth IDF             │
│ • Brand lookalikes    │   │ • L2 row normalization   │
│ • Suspicious TLDs     │   └────────────┬─────────────┘
│ • Heuristic PG001-010 │                │
└──────────┬────────────┘                │
           │ StandardScaler              │
           └──────────────┬──────────────┘
                          ▼
            ┌───────────────────────────┐
            │ L2 Logistic Regression    │  Adam optimizer, balanced weights,
            │ (Pure NumPy + JSON model) │  exact per-feature log-odds
            └─────────────┬─────────────┘
                          ▼
   Terminal / JSON / SARIF 2.1.0 / Offline HTML / Localhost UI
```

---

## Quickstart

### Install locally

```bash
git clone https://github.com/itskirbover/phishguard.git
cd phishguard
pip install -e ".[dev]"
```

### Scan & Explain a URL

```bash
phishguard explain "https://paypal.com.verify-account.tk/login/confirm"
```

Example output:

```text
URL:        https://paypal.com.verify-account.tk/login/confirm
Verdict:    PHISHING  [######################--] 93/100 (p=0.9338, threshold=0.420)
Latency:    2.16 ms (100% offline)
Signals:
  - [HIGH  ] PG001: Brand impersonation or typosquatting detected — URL references or mimics protected brand 'paypal' on non-official domain 'verify-account.tk'.
  - [MEDIUM] PG005: High-abuse top-level domain — TLD '.tk' is frequently abused in phishing campaigns.
  - [MEDIUM] PG006: Multiple credential-lure keywords — Detected 4 sensitive account/auth keywords across the URL.
  - [LOW   ] PG009: High hostname entropy — Hostname Shannon entropy is 4.01 bits/char, consistent with generated or randomized phishing hosts.
Log-Odds:   raw_logit=+2.6464 (bias=+0.4266)
Top Risk Drivers (+ log-odds):
  + brand_impersonation_score    contrib=+1.3832 (val=+1.350, w=+1.025)
  + sensitive_keyword_count      contrib=+1.2645 (val=+1.175, w=+1.076)
  + suspicious_tld               contrib=+1.1101 (val=+1.365, w=+0.813)
  + host_shannon_entropy         contrib=+0.3545 (val=+1.287, w=+0.275)
  + hostname_length              contrib=+0.2899 (val=+0.845, w=+0.343)
```

### Batch Scan & Export Reports (SARIF / Offline HTML / JSON)

```bash
# Export an interactive single-file offline HTML dashboard
phishguard batch examples/sample_urls.txt --format html -o report.html

# Export SARIF 2.1.0 for GitHub Code Scanning / CI gates (exits 2 on threat)
phishguard batch examples/sample_urls.txt --format sarif -o results.sarif --fail-on-phish

# Pipe URLs from stdin as JSON
cat examples/sample_urls.txt | phishguard batch - --format json
```

### Launch the Optional Local Self-Hosted Web UI

```bash
phishguard serve --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` in your browser to inspect URLs interactively with zero external dependencies (runs strictly on localhost using Python's `http.server` with `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, and CSP headers).

---

## Training on Your Own Real-World Datasets

The bundled `src/phishguard/data/default_model.json` (30 KB) is trained on PhishGuard's deterministic offline benchmark generator (including hard negatives such as official `accounts.google.com/signin`, `login.microsoftonline.com`, and `www.paypal.com/signin` URLs so the classifier does not naively flag every login page).

For real-world deployment or experimentation, combine public threat feeds (e.g., **PhishTank** / **OpenPhish**) with benign domain lists (e.g., **Tranco**) into a CSV with `url,label` columns (`0`/`benign` or `1`/`phishing`) and train a custom portable JSON model with stratified K-fold cross-validation:

```bash
# Train with 5-fold zero-leakage CV and optimize threshold for <= 1% false-positive rate
phishguard train --dataset my_urls.csv --folds 5 --max-fpr 0.01 -o my_model.json

# Evaluate on a held-out test CSV
phishguard eval --dataset holdout_urls.csv --model my_model.json

# Run inference using your custom model
phishguard scan "https://example.com/login" --model my_model.json --explain
```

---

## Security & Engineering Design Decisions

| Area | Design Choice | Why It Matters |
|---|---|---|
| **Model Persistence** | Plain schema-versioned JSON (`to_dict` / `from_dict`) | Avoids Python `pickle` deserialization RCE vulnerabilities. |
| **Network Safety** | Pure static URL parsing (no `socket`/`urllib` fetches) | Eliminates SSRF, DNS leaks, and drive-by exposure during SOC triage. |
| **Zero Data Leakage** | Scaler & TF-IDF fit strictly inside each CV fold | Prevents optimistic bias from validation vocabulary/statistics leaking into training. |
| **Numerical Stability** | Clipped sigmoid + `np.logaddexp(0, z) - y*z` BCE | Prevents `exp` overflow and `log(0)` NaNs during gradient descent. |
| **Report & UI Hardening** | `html.escape()` in HTML reports, `textContent` DOM nodes, CSP/size caps in local server | Prevents stored/reflected XSS when scanning adversarial URLs containing `<script>` payloads. |

---

## Running Tests

```bash
pytest -v
```

---

## License

MIT — see [LICENSE](./LICENSE).
