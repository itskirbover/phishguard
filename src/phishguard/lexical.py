"""Lexical security feature extraction and human-readable heuristic signals.

All features are computed purely from URL syntax (no DNS/HTTP requests) and
return deterministic floating-point values suitable for linear ML models.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from phishguard.url_parser import ParsedURL, parse_url

# Ordered list of lexical features used by both the ML model and explainer.
LEXICAL_FEATURE_NAMES: tuple[str, ...] = (
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "subdomain_count",
    "digit_ratio",
    "hyphen_count_host",
    "special_char_count",
    "has_ip_host",
    "has_at_userinfo",
    "has_non_standard_port",
    "is_https",
    "has_punycode_or_idn",
    "host_shannon_entropy",
    "path_shannon_entropy",
    "suspicious_tld",
    "sensitive_keyword_count",
    "brand_impersonation_score",
)

# High-abuse TLDs frequently observed in phishing campaigns (Spamhaus / OpenPhish trends).
SUSPICIOUS_TLDS: frozenset[str] = frozenset(
    {
        "tk",
        "ml",
        "ga",
        "cf",
        "gq",
        "xyz",
        "top",
        "buzz",
        "click",
        "work",
        "zip",
        "mov",
        "icu",
        "loan",
        "rest",
        "shop",
        "online",
        "site",
        "cfd",
        "sbs",
        "pw",
        "cc",
        "ru",
    }
)

SENSITIVE_KEYWORDS: tuple[str, ...] = (
    "login",
    "signin",
    "sign-in",
    "verify",
    "verification",
    "account",
    "update",
    "secure",
    "security",
    "banking",
    "confirm",
    "password",
    "passwd",
    "wallet",
    "recover",
    "recovery",
    "suspend",
    "suspended",
    "unlock",
    "billing",
    "auth",
    "webscr",
    "support",
    "cmd",
    "invoice",
    "credential",
)

# Frequently targeted brands mapped to their legitimate registrable domains.
PROTECTED_BRANDS: dict[str, frozenset[str]] = {
    "paypal": frozenset({"paypal.com", "paypal.me"}),
    "apple": frozenset({"apple.com", "icloud.com"}),
    "google": frozenset({"google.com", "googleapis.com", "googleusercontent.com", "gmail.com"}),
    "microsoft": frozenset(
        {
            "microsoft.com",
            "microsoftonline.com",
            "live.com",
            "outlook.com",
            "office.com",
            "azure.com",
        }
    ),
    "amazon": frozenset({"amazon.com", "aws.amazon.com", "amzn.to"}),
    "netflix": frozenset({"netflix.com"}),
    "github": frozenset({"github.com", "githubusercontent.com"}),
    "dropbox": frozenset({"dropbox.com"}),
    "facebook": frozenset({"facebook.com", "fb.com"}),
    "instagram": frozenset({"instagram.com"}),
    "chase": frozenset({"chase.com"}),
    "wellsfargo": frozenset({"wellsfargo.com"}),
    "bankofamerica": frozenset({"bankofamerica.com"}),
    "docusign": frozenset({"docusign.com", "docusign.net"}),
    "linkedin": frozenset({"linkedin.com"}),
    "okta": frozenset({"okta.com"}),
    "coinbase": frozenset({"coinbase.com"}),
    "binance": frozenset({"binance.com"}),
    "steam": frozenset({"steampowered.com", "steamcommunity.com"}),
    "discord": frozenset({"discord.com", "discord.gg"}),
    "stripe": frozenset({"stripe.com"}),
    "cloudflare": frozenset({"cloudflare.com"}),
}

_LEET_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})
_SPECIAL_CHARS = frozenset("@?=%&_~;")
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class SecuritySignal:
    """Human-readable security finding triggered by lexical analysis."""

    rule_id: str
    severity: str  # "high", "medium", "low"
    title: str
    detail: str


def shannon_entropy(text: str) -> float:
    """Compute Shannon entropy (bits per character) of a string."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        prob = count / length
        entropy -= prob * math.log2(prob)
    return entropy


def levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings using O(min(m, n)) space."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        current = [i]
        for j, ch_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if ch_a == ch_b else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _normalize_lookalikes(token: str) -> str:
    """Fold common phishing lookalike substitutions (e.g. rn->m, 1->l, 0->o)."""
    folded = token.lower().translate(_LEET_MAP)
    folded = folded.replace("rn", "m").replace("vv", "w")
    return folded


def detect_brand_impersonation(parsed: ParsedURL) -> tuple[float, str | None]:
    """Detect brand spoofing in subdomains, paths, or typosquatted domains.

    Returns (score, matched_brand) where score is 1.0 if impersonation is
    detected and 0.0 otherwise.
    """
    registrable = parsed.registrable_domain
    domain_label = parsed.domain_label
    normalized_label = _normalize_lookalikes(domain_label)
    non_domain_surface = f"{parsed.subdomain}/{parsed.path}?{parsed.query}".lower()
    surface_tokens = [t for t in _TOKEN_SPLIT_RE.split(non_domain_surface) if t]
    domain_tokens = [t for t in _TOKEN_SPLIT_RE.split(domain_label.lower()) if t]

    for brand, official_domains in PROTECTED_BRANDS.items():
        if registrable in official_domains:
            continue

        # 1. Exact or lookalike brand name embedded in subdomain, path, or query
        if brand in non_domain_surface:
            return 1.0, brand
        if any(_normalize_lookalikes(tok) == brand for tok in surface_tokens):
            return 1.0, brand

        # 2. Brand embedded inside a compound domain like paypal-secure-login.com
        if brand in domain_label or brand in normalized_label:
            return 1.0, brand
        if any(_normalize_lookalikes(tok) == brand for tok in domain_tokens):
            return 1.0, brand

        # 3. Typosquatting via Levenshtein edit distance on the registrable domain label
        if len(brand) >= 5 and abs(len(normalized_label) - len(brand)) <= 2:
            if levenshtein_distance(normalized_label, brand) <= 1:
                return 1.0, brand
            if len(brand) >= 7 and levenshtein_distance(normalized_label, brand) <= 2:
                return 1.0, brand

    return 0.0, None


def extract_lexical_features(url: str | ParsedURL) -> dict[str, float]:
    """Extract the 18 named lexical security features for a URL."""
    parsed = url if isinstance(url, ParsedURL) else parse_url(url)

    subdomain_count = len([p for p in parsed.subdomain.split(".") if p]) if parsed.subdomain else 0
    host_and_path = f"{parsed.hostname}{parsed.path}"
    digit_ratio = (
        sum(ch.isdigit() for ch in host_and_path) / len(host_and_path) if host_and_path else 0.0
    )
    hyphen_count_host = float(parsed.hostname.count("-"))
    special_char_count = float(sum(ch in _SPECIAL_CHARS for ch in parsed.normalized))
    has_ip_host = 1.0 if parsed.is_ip else 0.0
    has_at_userinfo = 1.0 if bool(parsed.userinfo) else 0.0
    has_non_standard_port = (
        1.0 if (parsed.port is not None and parsed.port not in (80, 443)) else 0.0
    )
    is_https = 1.0 if parsed.scheme == "https" else 0.0
    has_punycode_or_idn = 1.0 if (parsed.has_punycode or parsed.has_non_ascii_host) else 0.0
    host_entropy = shannon_entropy(parsed.hostname)
    path_entropy = shannon_entropy(f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path)
    suspicious_tld = 1.0 if parsed.tld in SUSPICIOUS_TLDS else 0.0

    lower_surface = f"{parsed.hostname}{parsed.path}?{parsed.query}".lower()
    sensitive_keyword_count = float(sum(1 for kw in SENSITIVE_KEYWORDS if kw in lower_surface))

    brand_score, _ = detect_brand_impersonation(parsed)

    return {
        "url_length": float(len(parsed.normalized)),
        "hostname_length": float(len(parsed.hostname)),
        "path_length": float(len(parsed.path)),
        "query_length": float(len(parsed.query)),
        "subdomain_count": float(subdomain_count),
        "digit_ratio": float(digit_ratio),
        "hyphen_count_host": hyphen_count_host,
        "special_char_count": special_char_count,
        "has_ip_host": has_ip_host,
        "has_at_userinfo": has_at_userinfo,
        "has_non_standard_port": has_non_standard_port,
        "is_https": is_https,
        "has_punycode_or_idn": has_punycode_or_idn,
        "host_shannon_entropy": float(host_entropy),
        "path_shannon_entropy": float(path_entropy),
        "suspicious_tld": suspicious_tld,
        "sensitive_keyword_count": sensitive_keyword_count,
        "brand_impersonation_score": float(brand_score),
    }


def lexical_feature_vector(url: str | ParsedURL) -> list[float]:
    """Return lexical features in canonical LEXICAL_FEATURE_NAMES order."""
    feats = extract_lexical_features(url)
    return [feats[name] for name in LEXICAL_FEATURE_NAMES]


def extract_security_signals(url: str | ParsedURL) -> list[SecuritySignal]:
    """Generate human-readable heuristic alerts explaining risk indicators."""
    parsed = url if isinstance(url, ParsedURL) else parse_url(url)
    feats = extract_lexical_features(parsed)
    signals: list[SecuritySignal] = []

    brand_score, brand_name = detect_brand_impersonation(parsed)
    if brand_score > 0.0 and brand_name:
        signals.append(
            SecuritySignal(
                rule_id="PG001",
                severity="high",
                title="Brand impersonation or typosquatting detected",
                detail=(
                    f"URL references or mimics protected brand '{brand_name}' "
                    f"on non-official domain '{parsed.registrable_domain}'."
                ),
            )
        )

    if feats["has_at_userinfo"] > 0.0:
        signals.append(
            SecuritySignal(
                rule_id="PG002",
                severity="high",
                title="URL authority spoofing via '@' userinfo",
                detail=(
                    f"Embedded userinfo '{parsed.userinfo}@' can disguise the true "
                    f"destination host '{parsed.hostname}'."
                ),
            )
        )

    if feats["has_ip_host"] > 0.0:
        signals.append(
            SecuritySignal(
                rule_id="PG003",
                severity="high",
                title="Raw IP address used as hostname",
                detail=f"URL connects directly to IP literal '{parsed.hostname}' instead of a domain.",
            )
        )

    if feats["has_punycode_or_idn"] > 0.0:
        signals.append(
            SecuritySignal(
                rule_id="PG004",
                severity="high",
                title="Punycode / IDN homoglyph hostname",
                detail=f"Hostname '{parsed.hostname}' uses punycode or non-ASCII characters.",
            )
        )

    if feats["suspicious_tld"] > 0.0:
        signals.append(
            SecuritySignal(
                rule_id="PG005",
                severity="medium",
                title="High-abuse top-level domain",
                detail=f"TLD '.{parsed.tld}' is frequently abused in phishing campaigns.",
            )
        )

    if feats["sensitive_keyword_count"] >= 2.0:
        signals.append(
            SecuritySignal(
                rule_id="PG006",
                severity="medium",
                title="Multiple credential-lure keywords",
                detail=(
                    f"Detected {int(feats['sensitive_keyword_count'])} sensitive account/auth "
                    "keywords across the URL."
                ),
            )
        )

    if feats["subdomain_count"] >= 3.0:
        signals.append(
            SecuritySignal(
                rule_id="PG007",
                severity="medium",
                title="Excessive subdomain nesting",
                detail=f"Hostname contains {int(feats['subdomain_count'])} subdomain labels.",
            )
        )

    if feats["has_non_standard_port"] > 0.0:
        signals.append(
            SecuritySignal(
                rule_id="PG008",
                severity="medium",
                title="Non-standard HTTP/HTTPS port",
                detail=f"URL specifies non-standard port {parsed.port}.",
            )
        )

    if feats["host_shannon_entropy"] >= 3.8:
        signals.append(
            SecuritySignal(
                rule_id="PG009",
                severity="low",
                title="High hostname entropy",
                detail=(
                    f"Hostname Shannon entropy is {feats['host_shannon_entropy']:.2f} bits/char, "
                    "consistent with generated or randomized phishing hosts."
                ),
            )
        )

    if feats["is_https"] == 0.0 and feats["sensitive_keyword_count"] >= 1.0:
        signals.append(
            SecuritySignal(
                rule_id="PG010",
                severity="medium",
                title="Unencrypted HTTP with sensitive lure keywords",
                detail="URL uses plaintext HTTP while requesting login or account verification.",
            )
        )

    return signals
