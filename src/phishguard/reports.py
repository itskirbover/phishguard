"""Terminal, JSON, SARIF 2.1.0, and self-contained offline HTML report exporters."""

from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from typing import Any, Sequence

from phishguard import __version__
from phishguard.pipeline import ScanResult


def _risk_bar(probability: float, width: int = 24) -> str:
    filled = int(round(max(0.0, min(1.0, probability)) * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def format_text_report(result: ScanResult, explain: bool = False) -> str:
    """Format a single ScanResult as a clean, readable terminal report."""
    lines = [
        f"URL:        {result.url}",
        f"Verdict:    {result.verdict}  {_risk_bar(result.probability)} "
        f"{result.risk_score}/100 (p={result.probability:.4f}, threshold={result.threshold:.3f})",
        f"Latency:    {result.latency_ms:.2f} ms (100% offline)",
    ]

    if result.security_signals:
        lines.append("Signals:")
        for sig in result.security_signals:
            lines.append(f"  - [{sig.severity.upper():6s}] {sig.rule_id}: {sig.title} — {sig.detail}")
    else:
        lines.append("Signals:    None")

    if explain:
        lines.append(
            f"Log-Odds:   raw_logit={result.raw_logit:+.4f} (bias={result.bias:+.4f})"
        )
        if result.top_risk_factors:
            lines.append("Top Risk Drivers (+ log-odds):")
            for factor in result.top_risk_factors:
                lines.append(
                    f"  + {factor.feature:<28s} contrib={factor.contribution:+.4f} "
                    f"(val={factor.value:+.3f}, w={factor.weight:+.3f})"
                )
        if result.top_benign_factors:
            lines.append("Top Benign Drivers (- log-odds):")
            for factor in result.top_benign_factors:
                lines.append(
                    f"  - {factor.feature:<28s} contrib={factor.contribution:+.4f} "
                    f"(val={factor.value:+.3f}, w={factor.weight:+.3f})"
                )

    return "\n".join(lines)


def to_json_report(results: Sequence[ScanResult]) -> str:
    """Serialize scan results to formatted JSON."""
    payload: dict[str, Any] = {
        "tool": "phishguard",
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": len(results),
            "phishing": sum(1 for r in results if r.verdict == "PHISHING"),
            "suspicious": sum(1 for r in results if r.verdict == "SUSPICIOUS"),
            "benign": sum(1 for r in results if r.verdict == "BENIGN"),
        },
        "results": [r.to_dict() for r in results],
    }
    return json.dumps(payload, indent=2)


def to_sarif_report(
    results: Sequence[ScanResult],
    source_uri: str = "urls.txt",
) -> str:
    """Export findings in SARIF 2.1.0 format for GitHub Code Scanning / CI pipelines."""
    rules_by_id: dict[str, dict[str, Any]] = {
        "PG000": {
            "id": "PG000",
            "name": "PhishingUrlDetected",
            "shortDescription": {"text": "ML classifier flagged URL as phishing or suspicious"},
            "defaultConfiguration": {"level": "error"},
        }
    }
    sarif_results: list[dict[str, Any]] = []

    for idx, res in enumerate(results, start=1):
        if res.verdict == "BENIGN":
            continue

        level = "error" if res.verdict == "PHISHING" else "warning"
        rule_id = res.security_signals[0].rule_id if res.security_signals else "PG000"

        for sig in res.security_signals:
            if sig.rule_id not in rules_by_id:
                rules_by_id[sig.rule_id] = {
                    "id": sig.rule_id,
                    "name": sig.title.replace(" ", ""),
                    "shortDescription": {"text": sig.title},
                    "defaultConfiguration": {
                        "level": "error" if sig.severity == "high" else "warning"
                    },
                }

        top_reasons = ", ".join(
            f"{f.feature} ({f.contribution:+.2f})" for f in res.top_risk_factors[:3]
        )
        message = (
            f"{res.verdict} URL detected (risk={res.risk_score}/100, p={res.probability:.3f}): "
            f"{res.url}"
        )
        if top_reasons:
            message += f" | Top ML drivers: {top_reasons}"

        sarif_results.append(
            {
                "ruleId": rule_id,
                "level": level,
                "message": {"text": message},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": source_uri},
                            "region": {"startLine": idx},
                        }
                    }
                ],
                "properties": {
                    "verdict": res.verdict,
                    "probability": round(res.probability, 6),
                    "risk_score": res.risk_score,
                    "signals": [s.rule_id for s in res.security_signals],
                },
            }
        )

    sarif_doc = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "PhishGuard",
                        "version": __version__,
                        "informationUri": "https://github.com/itskirbover/phishguard",
                        "rules": list(rules_by_id.values()),
                    }
                },
                "results": sarif_results,
            }
        ],
    }
    return json.dumps(sarif_doc, indent=2)


def to_html_report(results: Sequence[ScanResult]) -> str:
    """Generate a self-contained offline HTML report (zero external CDNs)."""
    total = len(results)
    phish_cnt = sum(1 for r in results if r.verdict == "PHISHING")
    susp_cnt = sum(1 for r in results if r.verdict == "SUSPICIOUS")
    benign_cnt = sum(1 for r in results if r.verdict == "BENIGN")
    generated = html.escape(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

    rows_html: list[str] = []
    for res in results:
        verdict_cls = res.verdict.lower()
        escaped_url = html.escape(res.url)
        signals_html = "".join(
            f'<span class="tag tag-{html.escape(s.severity)}">'
            f"{html.escape(s.rule_id)}: {html.escape(s.title)}</span>"
            for s in res.security_signals
        ) or '<span class="muted">No heuristic alerts</span>'

        factors_html = "".join(
            f'<div class="factor"><code>{html.escape(f.feature)}</code> '
            f'<span class="pos">{f.contribution:+.3f}</span></div>'
            for f in res.top_risk_factors[:4]
        ) or '<span class="muted">—</span>'

        rows_html.append(
            f"""
            <tr data-verdict="{html.escape(verdict_cls)}">
              <td><span class="badge badge-{html.escape(verdict_cls)}">{html.escape(res.verdict)}</span></td>
              <td><strong>{res.risk_score}</strong>/100<br><span class="muted">p={res.probability:.4f}</span></td>
              <td class="url-cell"><code>{escaped_url}</code></td>
              <td>{signals_html}</td>
              <td>{factors_html}</td>
            </tr>
            """
        )

    table_body = "\n".join(rows_html)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PhishGuard Offline Scan Report</title>
  <style>
    :root {{
      --bg: #0d1117; --panel: #161b22; --border: #30363d;
      --text: #e6edf3; --muted: #8b949e;
      --red: #f85149; --amber: #d29922; --green: #3fb950; --blue: #58a6ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 2rem; font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.5;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    header {{ display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; }}
    h1 {{ margin: 0; font-size: 1.6rem; }}
    .muted {{ color: var(--muted); font-size: 0.85rem; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }}
    .card .num {{ font-size: 1.8rem; font-weight: 700; margin-top: 0.25rem; }}
    .controls {{ display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }}
    input, select {{
      background: var(--panel); color: var(--text); border: 1px solid var(--border);
      border-radius: 6px; padding: 0.5rem 0.75rem; font-size: 0.9rem;
    }}
    input {{ flex: 1; min-width: 220px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
    th, td {{ text-align: left; padding: 0.75rem; border-bottom: 1px solid var(--border); vertical-align: top; }}
    th {{ background: #1c2128; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }}
    .url-cell {{ max-width: 380px; word-break: break-all; font-size: 0.85rem; }}
    .badge {{ display: inline-block; padding: 0.2rem 0.55rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }}
    .badge-phishing {{ background: rgba(248,81,73,0.18); color: var(--red); border: 1px solid var(--red); }}
    .badge-suspicious {{ background: rgba(210,153,34,0.18); color: var(--amber); border: 1px solid var(--amber); }}
    .badge-benign {{ background: rgba(63,185,80,0.18); color: var(--green); border: 1px solid var(--green); }}
    .tag {{ display: inline-block; margin: 0 0.25rem 0.25rem 0; padding: 0.15rem 0.45rem; border-radius: 4px; font-size: 0.75rem; background: #21262d; border: 1px solid var(--border); }}
    .tag-high {{ border-color: var(--red); color: #ffa198; }}
    .tag-medium {{ border-color: var(--amber); color: #e3b341; }}
    .factor {{ font-size: 0.8rem; margin-bottom: 0.2rem; }}
    .pos {{ color: var(--red); font-weight: 600; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>PhishGuard Offline Scan Report</h1>
      <div class="muted">Generated {generated} &middot; PhishGuard v{html.escape(__version__)}</div>
    </header>
    <div class="cards">
      <div class="card"><div class="muted">Total URLs</div><div class="num">{total}</div></div>
      <div class="card"><div class="muted">Phishing</div><div class="num" style="color:var(--red)">{phish_cnt}</div></div>
      <div class="card"><div class="muted">Suspicious</div><div class="num" style="color:var(--amber)">{susp_cnt}</div></div>
      <div class="card"><div class="muted">Benign</div><div class="num" style="color:var(--green)">{benign_cnt}</div></div>
    </div>
    <div class="controls">
      <input id="search" type="search" placeholder="Filter by URL, rule ID, or feature...">
      <select id="verdictFilter">
        <option value="all">All Verdicts</option>
        <option value="phishing">Phishing</option>
        <option value="suspicious">Suspicious</option>
        <option value="benign">Benign</option>
      </select>
    </div>
    <table>
      <thead>
        <tr>
          <th>Verdict</th>
          <th>Risk Score</th>
          <th>URL</th>
          <th>Security Signals</th>
          <th>Top ML Risk Drivers</th>
        </tr>
      </thead>
      <tbody id="rows">
        {table_body}
      </tbody>
    </table>
  </div>
  <script>
    const search = document.getElementById('search');
    const verdict = document.getElementById('verdictFilter');
    const rows = Array.from(document.querySelectorAll('#rows tr'));
    function applyFilters() {{
      const q = search.value.toLowerCase();
      const v = verdict.value;
      for (const row of rows) {{
        const matchV = v === 'all' || row.dataset.verdict === v;
        const matchQ = !q || row.textContent.toLowerCase().includes(q);
        row.style.display = (matchV && matchQ) ? '' : 'none';
      }}
    }}
    search.addEventListener('input', applyFilters);
    verdict.addEventListener('change', applyFilters);
  </script>
</body>
</html>
"""
