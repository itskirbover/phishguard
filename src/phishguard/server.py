"""Optional zero-dependency self-hosted local web UI for PhishGuard.

Uses only Python's standard library ``http.server.ThreadingHTTPServer`` and
binds to ``127.0.0.1`` by default. No external server, database, or cloud
hosting is required.
"""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any
from urllib.parse import urlsplit

from phishguard import __version__
from phishguard.pipeline import PhishGuardPipeline

MAX_BODY_BYTES = 128 * 1024  # 128 KB request cap
MAX_BATCH_URLS = 200
MAX_URL_LENGTH = 4096

_UI_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PhishGuard Local Analyzer</title>
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
    .wrap {{ max-width: 960px; margin: 0 auto; }}
    h1 {{ margin: 0 0 0.25rem 0; font-size: 1.65rem; }}
    .sub {{ color: var(--muted); margin-bottom: 1.5rem; font-size: 0.9rem; }}
    .panel {{
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 10px; padding: 1.25rem; margin-bottom: 1.25rem;
    }}
    textarea {{
      width: 100%; min-height: 92px; resize: vertical;
      background: #0d1117; color: var(--text); border: 1px solid var(--border);
      border-radius: 6px; padding: 0.75rem; font-family: ui-monospace, monospace; font-size: 0.9rem;
    }}
    .actions {{ display: flex; gap: 0.75rem; margin-top: 0.75rem; align-items: center; flex-wrap: wrap; }}
    button {{
      background: #238636; color: #fff; border: none; border-radius: 6px;
      padding: 0.55rem 1.1rem; font-weight: 600; cursor: pointer; font-size: 0.9rem;
    }}
    button.secondary {{ background: #21262d; border: 1px solid var(--border); color: var(--text); }}
    button:hover {{ filter: brightness(1.08); }}
    .result {{
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1rem;
    }}
    .result-head {{ display: flex; justify-content: space-between; gap: 1rem; align-items: center; flex-wrap: wrap; }}
    .url-str {{ font-family: ui-monospace, monospace; word-break: break-all; font-size: 0.9rem; }}
    .badge {{ padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.78rem; font-weight: 700; }}
    .badge-phishing {{ background: rgba(248,81,73,0.18); color: var(--red); border: 1px solid var(--red); }}
    .badge-suspicious {{ background: rgba(210,153,34,0.18); color: var(--amber); border: 1px solid var(--amber); }}
    .badge-benign {{ background: rgba(63,185,80,0.18); color: var(--green); border: 1px solid var(--green); }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.85rem; }}
    @media (max-width: 720px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    .sec-title {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); margin-bottom: 0.4rem; }}
    .chip {{ display: inline-block; font-size: 0.78rem; background: #21262d; border: 1px solid var(--border); border-radius: 4px; padding: 0.2rem 0.45rem; margin: 0 0.35rem 0.35rem 0; }}
    .driver {{ display: flex; justify-content: space-between; font-size: 0.83rem; padding: 0.2rem 0; border-bottom: 1px dashed #21262d; font-family: ui-monospace, monospace; }}
    .pos {{ color: var(--red); }}
    .neg {{ color: var(--green); }}
    .err {{ color: var(--red); font-size: 0.9rem; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>PhishGuard Local Analyzer</h1>
    <div class="sub">100% offline ML + lexical security analysis &middot; v{__version__} &middot; Never fetches untrusted URLs</div>
    <div class="panel">
      <textarea id="urls" placeholder="Paste one or more URLs (one per line)..."></textarea>
      <div class="actions">
        <button id="scanBtn" type="button">Analyze Offline</button>
        <button id="sampleBtn" class="secondary" type="button">Load Demo URLs</button>
        <span id="status" class="sub" style="margin:0;"></span>
      </div>
    </div>
    <div id="output"></div>
  </div>
  <script>
    const urlsEl = document.getElementById('urls');
    const outEl = document.getElementById('output');
    const statusEl = document.getElementById('status');

    document.getElementById('sampleBtn').addEventListener('click', () => {{
      urlsEl.value = [
        'https://github.com/SimplifyJobs/Summer2027-Internships',
        'https://accounts.google.com/signin/v2/identifier',
        'https://paypal.com.verify-account.tk/login/confirm',
        'http://198.51.100.42:8080/microsoft/login/verify',
        'https://arnazon-security-update.xyz/account/billing',
        'https://apple.com@secure-login-9921.click/signin'
      ].join('\\n');
    }});

    function renderResult(r) {{
      const card = document.createElement('div');
      card.className = 'result';

      const head = document.createElement('div');
      head.className = 'result-head';
      const urlSpan = document.createElement('div');
      urlSpan.className = 'url-str';
      urlSpan.textContent = r.url;
      const badge = document.createElement('span');
      badge.className = 'badge badge-' + r.verdict.toLowerCase();
      badge.textContent = r.verdict + ' · ' + r.risk_score + '/100 (p=' + r.probability.toFixed(4) + ')';
      head.appendChild(urlSpan);
      head.appendChild(badge);
      card.appendChild(head);

      const sigWrap = document.createElement('div');
      sigWrap.style.marginTop = '0.65rem';
      if (r.security_signals.length === 0) {{
        const none = document.createElement('span');
        none.className = 'sub';
        none.textContent = 'No heuristic security alerts triggered.';
        sigWrap.appendChild(none);
      }} else {{
        for (const s of r.security_signals) {{
          const chip = document.createElement('span');
          chip.className = 'chip';
          chip.textContent = '[' + s.severity.toUpperCase() + '] ' + s.rule_id + ': ' + s.title;
          sigWrap.appendChild(chip);
        }}
      }}
      card.appendChild(sigWrap);

      const grid = document.createElement('div');
      grid.className = 'grid';

      const riskCol = document.createElement('div');
      const riskTitle = document.createElement('div');
      riskTitle.className = 'sec-title';
      riskTitle.textContent = 'Top ML Risk Drivers (+ log-odds)';
      riskCol.appendChild(riskTitle);
      for (const f of r.top_risk_factors) {{
        const row = document.createElement('div');
        row.className = 'driver';
        const k = document.createElement('span');
        k.textContent = f.feature;
        const v = document.createElement('span');
        v.className = 'pos';
        v.textContent = (f.contribution >= 0 ? '+' : '') + f.contribution.toFixed(4);
        row.appendChild(k);
        row.appendChild(v);
        riskCol.appendChild(row);
      }}

      const benignCol = document.createElement('div');
      const benignTitle = document.createElement('div');
      benignTitle.className = 'sec-title';
      benignTitle.textContent = 'Top Benign Drivers (- log-odds)';
      benignCol.appendChild(benignTitle);
      for (const f of r.top_benign_factors) {{
        const row = document.createElement('div');
        row.className = 'driver';
        const k = document.createElement('span');
        k.textContent = f.feature;
        const v = document.createElement('span');
        v.className = 'neg';
        v.textContent = f.contribution.toFixed(4);
        row.appendChild(k);
        row.appendChild(v);
        benignCol.appendChild(row);
      }}

      grid.appendChild(riskCol);
      grid.appendChild(benignCol);
      card.appendChild(grid);
      return card;
    }}

    document.getElementById('scanBtn').addEventListener('click', async () => {{
      const lines = urlsEl.value.split(/\\r?\\n/).map(s => s.trim()).filter(Boolean);
      outEl.textContent = '';
      if (lines.length === 0) {{
        statusEl.textContent = 'Enter at least one URL.';
        return;
      }}
      statusEl.textContent = 'Scanning...';
      try {{
        const resp = await fetch('/api/scan', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ urls: lines }})
        }});
        const data = await resp.json();
        if (!resp.ok) {{
          const err = document.createElement('div');
          err.className = 'err';
          err.textContent = data.error || 'Request failed.';
          outEl.appendChild(err);
          statusEl.textContent = '';
          return;
        }}
        for (const r of data.results) {{
          outEl.appendChild(renderResult(r));
        }}
        statusEl.textContent = 'Scanned ' + data.results.length + ' URL(s) offline.';
      }} catch (e) {{
        statusEl.textContent = 'Error connecting to local PhishGuard server.';
      }}
    }});
  </script>
</body>
</html>
"""


def create_server(
    pipeline: PhishGuardPipeline,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create a configured ThreadingHTTPServer bound to ``host:port``."""

    class PhishGuardHandler(BaseHTTPRequestHandler):
        def _send_security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'",
            )

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self._send_security_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Silence noisy default stderr logging during library/test usage.
            return

        def do_GET(self) -> None:  # noqa: N802
            route = urlsplit(self.path).path
            if route == "/" or route == "/index.html":
                body = _UI_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self._send_security_headers()
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if route == "/api/health":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "version": __version__,
                        "threshold": pipeline.threshold,
                        "metadata": pipeline.metadata,
                    },
                )
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            route = urlsplit(self.path).path
            if route != "/api/scan":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length"})
                return

            if content_length <= 0 or content_length > MAX_BODY_BYTES:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                    if content_length > MAX_BODY_BYTES
                    else HTTPStatus.BAD_REQUEST,
                    {"error": f"Request body must be between 1 and {MAX_BODY_BYTES} bytes."},
                )
                return

            raw = self.rfile.read(content_length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body."})
                return

            if not isinstance(data, dict):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "JSON payload must be an object."})
                return

            urls: list[str] = []
            if "url" in data and isinstance(data["url"], str):
                urls = [data["url"].strip()]
            elif "urls" in data and isinstance(data["urls"], list):
                urls = [str(item).strip() for item in data["urls"] if str(item).strip()]
            else:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Provide 'url' (string) or 'urls' (list of strings)."},
                )
                return

            if not urls or len(urls) > MAX_BATCH_URLS:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"Provide between 1 and {MAX_BATCH_URLS} non-empty URLs."},
                )
                return

            if any(len(u) > MAX_URL_LENGTH for u in urls):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"Individual URL length cannot exceed {MAX_URL_LENGTH} characters."},
                )
                return

            results = pipeline.predict_batch(urls)
            self._send_json(
                HTTPStatus.OK,
                {"results": [r.to_dict() for r in results]},
            )

    return ThreadingHTTPServer((host, port), PhishGuardHandler)
