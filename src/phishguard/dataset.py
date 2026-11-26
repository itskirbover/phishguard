"""Dataset loading (CSV) and reproducible offline benchmark corpus generation.

Supports importing real-world labeled URL datasets (such as PhishTank,
OpenPhish, and Tranco exports) as well as generating a balanced,
adversarial-aware benchmark dataset with hard negatives (legitimate login
pages, long documentation/API URLs) and diverse phishing tactics.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import random
import string

_POSITIVE_LABELS = frozenset({"1", "phishing", "phish", "malicious", "bad", "true", "yes"})
_NEGATIVE_LABELS = frozenset({"0", "benign", "legitimate", "legit", "good", "safe", "false", "no"})


@dataclass(frozen=True)
class Dataset:
    """Labeled collection of URLs (0 = benign, 1 = phishing)."""

    urls: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.urls)


def _parse_label(raw: str) -> int:
    normalized = raw.strip().lower()
    if normalized in _POSITIVE_LABELS:
        return 1
    if normalized in _NEGATIVE_LABELS:
        return 0
    raise ValueError(f"Unrecognized URL label: {raw!r}")


def load_csv_dataset(path: str | Path) -> Dataset:
    """Load a labeled URL dataset from a CSV file with ``url`` and ``label`` columns."""
    csv_path = Path(path)
    urls: list[str] = []
    labels: list[int] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV file '{csv_path}' is empty or missing headers.")

        field_map = {name.strip().lower(): name for name in reader.fieldnames if name}
        url_col = field_map.get("url") or field_map.get("uri") or field_map.get("link")
        label_col = field_map.get("label") or field_map.get("class") or field_map.get("target")
        if not url_col or not label_col:
            raise ValueError(
                f"CSV '{csv_path}' must include 'url' and 'label' columns "
                f"(found: {reader.fieldnames})."
            )

        for row in reader:
            url = (row.get(url_col) or "").strip()
            raw_label = (row.get(label_col) or "").strip()
            if not url or not raw_label:
                continue
            urls.append(url)
            labels.append(_parse_label(raw_label))

    if not urls:
        raise ValueError(f"No valid labeled rows found in '{csv_path}'.")
    return Dataset(urls=urls, labels=labels)


def save_csv_dataset(dataset: Dataset, path: str | Path) -> None:
    """Write a labeled URL dataset to a CSV file."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["url", "label"])
        for url, label in zip(dataset.urls, dataset.labels):
            writer.writerow([url, label])


# --- Reproducible offline benchmark generator ---

_BENIGN_DOMAINS = (
    "github.com",
    "docs.python.org",
    "wikipedia.org",
    "stackoverflow.com",
    "developer.mozilla.org",
    "npmjs.com",
    "pypi.org",
    "arxiv.org",
    "nytimes.com",
    "bbc.co.uk",
    "reuters.com",
    "mit.edu",
    "stanford.edu",
    "cmu.edu",
    "cloudflare.com",
    "stripe.com",
    "aws.amazon.com",
    "cloud.google.com",
    "learn.microsoft.com",
    "apple.com",
    "paypal.com",
    "netflix.com",
    "linkedin.com",
    "dropbox.com",
    "reddit.com",
    "ycombinator.com",
    "kernel.org",
    "debian.org",
    "rust-lang.org",
    "golang.org",
    "kubernetes.io",
    "docker.com",
    "mozilla.org",
    "w3.org",
    "nist.gov",
    "cisa.gov",
)

_BENIGN_PATHS = (
    "/",
    "/about",
    "/docs/getting-started",
    "/blog/2026/release-notes",
    "/en-US/docs/Web/HTTP/Overview",
    "/questions/74291820/python-type-hints",
    "/search?q=linear+regression+numpy",
    "/projects/security-guide/issues",
    "/packages/numpy/releases",
    "/articles/distributed-systems-design",
    "/careers/internships/software-engineering",
    "/pricing",
    "/status",
    "/changelog/v2-4-0",
    "/papers/2601.04812.pdf",
    "/courses/cs101/syllabus",
    "/api/v2/reference/endpoints",
    "/news/technology/cybersecurity-trends",
    "/wiki/Transport_Layer_Security",
    "/download/linux-amd64",
)

# Hard negatives: legitimate auth/account URLs on official domains so the model
# learns not to flag every URL containing "login" or "account".
_BENIGN_HARD_NEGATIVES = (
    "https://github.com/login",
    "https://accounts.google.com/signin/v2/identifier",
    "https://www.paypal.com/signin",
    "https://www.paypal.com/myaccount/summary",
    "https://login.microsoftonline.com/common/oauth2/authorize",
    "https://appleid.apple.com/sign-in",
    "https://www.amazon.com/ap/signin",
    "https://www.linkedin.com/login",
    "https://www.dropbox.com/login",
    "https://dashboard.stripe.com/login",
    "https://dash.cloudflare.com/login",
    "https://www.netflix.com/login",
    "https://www.coinbase.com/signin",
    "https://discord.com/login",
    "https://store.steampowered.com/login/",
    "https://account.docusign.com/oauth/auth",
    "https://secure.chase.com/web/auth/dashboard",
    "https://www.wellsfargo.com/help/online-banking/sign-on-faqs/",
    "https://www.bankofamerica.com/security-center/overview/",
    "https://myaccount.google.com/security",
)

_TARGET_BRANDS = (
    "paypal",
    "apple",
    "google",
    "microsoft",
    "amazon",
    "netflix",
    "github",
    "dropbox",
    "chase",
    "wellsfargo",
    "docusign",
    "linkedin",
    "okta",
    "coinbase",
    "binance",
    "steam",
    "discord",
    "stripe",
    "cloudflare",
    "instagram",
)

_TYPOSQUATS = (
    "paypa1",
    "paypall",
    "arnazon",
    "amaz0n",
    "micros0ft",
    "mircosoft",
    "app1e",
    "appple",
    "g00gle",
    "gooogle",
    "netf1ix",
    "githhub",
    "dropb0x",
    "c0inbase",
    "linked1n",
    "docuslgn",
    "chasse",
    "disc0rd",
    "str1pe",
    "c1oudflare",
)

_PHISH_TLDS = ("tk", "ml", "ga", "cf", "xyz", "top", "click", "buzz", "icu", "shop", "online", "site", "cfd", "sbs", "com", "net", "org")
_LURE_WORDS = ("verify", "secure", "login", "signin", "account", "update", "confirm", "unlock", "billing", "support", "recovery", "auth", "webscr", "wallet")
_PHISH_PATHS = (
    "/login/verify",
    "/account/update-billing",
    "/signin/confirm-identity",
    "/webscr?cmd=_login-run&dispatch={tok}",
    "/secure/auth/session?token={tok}",
    "/unlock/suspended-account?id={num}",
    "/verify/password-reset/{tok}",
    "/wallet/connect/recover?ref={tok}",
    "/oauth2/authorize/confirm?client={tok}",
    "/invoice/view/document.php?id={num}",
)


def _rand_token(rng: random.Random, length: int = 12) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def _generate_benign_url(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.14:
        base = rng.choice(_BENIGN_HARD_NEGATIVES)
        if rng.random() < 0.35:
            return f"{base}?ref=nav_{rng.randint(1, 99)}"
        return base

    domain = rng.choice(_BENIGN_DOMAINS)
    scheme = "https" if rng.random() < 0.94 else "http"
    prefix = rng.choice(["", "www.", "docs.", "blog.", "api.", "status."])
    if domain.count(".") >= 2 or domain.startswith(("docs.", "learn.", "cloud.", "developer.", "aws.")):
        prefix = ""
    path = rng.choice(_BENIGN_PATHS)

    # Occasionally add realistic query params or slug variations
    if rng.random() < 0.3 and "?" not in path:
        slug = rng.choice(["guide", "v2", "overview", "metrics", "reference", "2026"])
        path = f"{path.rstrip('/')}/{slug}"
    if rng.random() < 0.25 and "?" not in path:
        path = f"{path}?page={rng.randint(1, 12)}&sort=recent"

    return f"{scheme}://{prefix}{domain}{path}"


def _generate_phishing_url(rng: random.Random) -> str:
    tactic = rng.choices(
        ["subdomain_spoof", "typosquat", "ip_literal", "userinfo_at", "punycode", "dga_lure"],
        weights=[0.28, 0.24, 0.12, 0.12, 0.10, 0.14],
        k=1,
    )[0]

    brand = rng.choice(_TARGET_BRANDS)
    lure1 = rng.choice(_LURE_WORDS)
    lure2 = rng.choice(_LURE_WORDS)
    tld = rng.choice(_PHISH_TLDS)
    scheme = "https" if rng.random() < 0.45 else "http"
    tok = _rand_token(rng, rng.randint(10, 22))
    num = rng.randint(100000, 9999999)
    path = rng.choice(_PHISH_PATHS).format(tok=tok, num=num)

    if tactic == "subdomain_spoof":
        host = f"{brand}.{lure1}-account.{lure2}-portal.{tld}"
        return f"{scheme}://{host}{path}"

    if tactic == "typosquat":
        typo = rng.choice(_TYPOSQUATS)
        suffix = f"-{lure1}" if rng.random() < 0.65 else ""
        host = f"{typo}{suffix}.{tld}"
        return f"{scheme}://{host}{path}"

    if tactic == "ip_literal":
        ip = f"{rng.randint(11, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
        port = f":{rng.choice([8080, 8443, 8888, 9000])}" if rng.random() < 0.5 else ""
        return f"http://{ip}{port}/{brand}{path}"

    if tactic == "userinfo_at":
        attacker = f"{lure1}-{_rand_token(rng, 6)}.{tld}"
        return f"{scheme}://{brand}.com@{attacker}{path}"

    if tactic == "punycode":
        host = f"xn--{brand}-{_rand_token(rng, 4)}.{tld}"
        return f"{scheme}://{host}{path}"

    # dga_lure: high-entropy host + credential lure path
    rand_host = f"{_rand_token(rng, 10)}-{lure1}-{rng.randint(10, 999)}.{tld}"
    return f"{scheme}://{rand_host}{path}"


def generate_benchmark_dataset(n_samples: int = 1200, seed: int = 42) -> Dataset:
    """Generate a balanced, deterministic offline URL classification benchmark."""
    if n_samples < 20:
        raise ValueError("n_samples must be at least 20.")

    rng = random.Random(seed)
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    pairs: list[tuple[str, int]] = []
    for _ in range(n_neg):
        pairs.append((_generate_benign_url(rng), 0))
    for _ in range(n_pos):
        pairs.append((_generate_phishing_url(rng), 1))

    rng.shuffle(pairs)
    urls = [u for u, _ in pairs]
    labels = [y for _, y in pairs]
    return Dataset(urls=urls, labels=labels)
