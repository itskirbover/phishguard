"""Integration tests for CLI subcommands, SARIF/HTML reports, and local web UI."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

from phishguard.cli import main
from phishguard.pipeline import PhishGuardPipeline
from phishguard.server import create_server


def test_cli_scan_and_explain_json(capsys) -> None:
    rc = main(["scan", "https://github.com/python/cpython", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["benign"] == 1
    assert payload["results"][0]["verdict"] == "BENIGN"

    rc_phish = main(
        [
            "explain",
            "https://paypal.com.verify-account.tk/login/confirm",
            "--fail-on-phish",
        ]
    )
    assert rc_phish == 2
    out = capsys.readouterr().out
    assert "PHISHING" in out
    assert "Top Risk Drivers" in out


def test_cli_batch_sarif_and_html_exports(tmp_path: Path) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "# sample input\n"
        "https://docs.python.org/3/\n"
        "https://paypal.com.verify-account.tk/login/confirm\n",
        encoding="utf-8",
    )

    sarif_out = tmp_path / "report.sarif"
    rc = main(["batch", str(url_file), "--format", "sarif", "--output", str(sarif_out)])
    assert rc == 0
    sarif = json.loads(sarif_out.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    assert len(sarif["runs"][0]["results"]) == 1
    assert sarif["runs"][0]["results"][0]["properties"]["verdict"] == "PHISHING"

    html_out = tmp_path / "report.html"
    rc_html = main(
        [
            "batch",
            str(url_file),
            "--format",
            "html",
            "--output",
            str(html_out),
            "--fail-on-phish",
        ]
    )
    assert rc_html == 2
    html_text = html_out.read_text(encoding="utf-8")
    assert "PhishGuard Offline Scan Report" in html_text
    assert "https://paypal.com.verify-account.tk/login/confirm" in html_text


def test_cli_train_and_eval_roundtrip(tmp_path: Path, capsys) -> None:
    model_file = tmp_path / "custom.json"
    rc_train = main(
        [
            "train",
            "--samples",
            "200",
            "--folds",
            "3",
            "--max-ngrams",
            "64",
            "--max-iter",
            "120",
            "--output",
            str(model_file),
            "--json",
        ]
    )
    assert rc_train == 0
    train_summary = json.loads(capsys.readouterr().out)
    assert model_file.exists()
    assert train_summary["cv_aggregate"]["roc_auc"] >= 0.90

    rc_eval = main(
        [
            "eval",
            "--model",
            str(model_file),
            "--samples",
            "100",
            "--seed",
            "2027",
            "--json",
        ]
    )
    assert rc_eval == 0
    eval_summary = json.loads(capsys.readouterr().out)
    assert eval_summary["roc_auc"] >= 0.90


def test_local_server_health_ui_and_scan_api() -> None:
    pipe = PhishGuardPipeline.load_default()
    server = create_server(pipe, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"

        with urllib.request.urlopen(f"{base}/") as resp:
            assert resp.status == 200
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert resp.headers.get("X-Frame-Options") == "DENY"
            assert "PhishGuard Local Analyzer" in resp.read().decode("utf-8")

        with urllib.request.urlopen(f"{base}/api/health") as resp:
            assert resp.status == 200
            health = json.loads(resp.read().decode("utf-8"))
            assert health["status"] == "ok"

        req = urllib.request.Request(
            f"{base}/api/scan",
            data=json.dumps(
                {
                    "urls": [
                        "https://github.com/python/cpython",
                        "https://paypal.com.verify-account.tk/login",
                    ]
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode("utf-8"))
            assert len(body["results"]) == 2
            assert body["results"][0]["verdict"] == "BENIGN"
            assert body["results"][1]["verdict"] == "PHISHING"

        bad_req = urllib.request.Request(
            f"{base}/api/scan",
            data=b"{not-json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(bad_req)
            raise AssertionError("Expected HTTP 400 on malformed JSON")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
