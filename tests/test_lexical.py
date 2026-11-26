"""Unit tests for offline URL parsing and lexical security feature extraction."""

from __future__ import annotations

import math

from phishguard.lexical import (
    LEXICAL_FEATURE_NAMES,
    detect_brand_impersonation,
    extract_lexical_features,
    extract_security_signals,
    levenshtein_distance,
    lexical_feature_vector,
    shannon_entropy,
)
from phishguard.url_parser import parse_url, refang_url


def test_refang_soc_indicators() -> None:
    assert refang_url("hxxps://paypal[.]verify-account[.]tk/login") == (
        "https://paypal.verify-account.tk/login"
    )
    assert refang_url("hxxp://192[.]168[.]1[.]10/admin") == "http://192.168.1.10/admin"


def test_parse_url_handles_scheme_less_and_multipart_tld() -> None:
    parsed = parse_url("docs.example.co.uk/guide?page=2#top")
    assert parsed.scheme == "http"
    assert parsed.subdomain == "docs"
    assert parsed.domain_label == "example"
    assert parsed.tld == "co.uk"
    assert parsed.registrable_domain == "example.co.uk"
    assert parsed.path == "/guide"
    assert parsed.query == "page=2"
    assert parsed.fragment == "top"
    assert not parsed.is_ip


def test_parse_url_ip_userinfo_port_and_punycode() -> None:
    parsed_ip = parse_url("http://198.51.100.42:8443/login")
    assert parsed_ip.is_ip
    assert parsed_ip.port == 8443

    parsed_v6 = parse_url("http://[2001:db8::1]:8080/auth")
    assert parsed_v6.is_ip
    assert parsed_v6.hostname == "2001:db8::1"
    assert parsed_v6.port == 8080

    parsed_at = parse_url("https://paypal.com@evil-login.xyz/verify")
    assert parsed_at.userinfo == "paypal.com"
    assert parsed_at.hostname == "evil-login.xyz"

    parsed_idn = parse_url("https://xn--pple-43d.com/signin")
    assert parsed_idn.has_punycode


def test_parse_url_schemeless_with_nested_redirect_url() -> None:
    parsed = parse_url("evil.tk/redirect?next=https://paypal.com/signin")
    assert parsed.scheme == "http"
    assert parsed.hostname == "evil.tk"
    assert parsed.registrable_domain == "evil.tk"
    assert parsed.path == "/redirect"
    assert parsed.query == "next=https://paypal.com/signin"


def test_shannon_entropy_known_values() -> None:
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert math.isclose(shannon_entropy("ab"), 1.0, rel_tol=1e-6)
    assert math.isclose(shannon_entropy("abcd"), 2.0, rel_tol=1e-6)


def test_levenshtein_distance_known_pairs() -> None:
    assert levenshtein_distance("paypal", "paypal") == 0
    assert levenshtein_distance("paypal", "paypa") == 1
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("", "abc") == 3


def test_brand_impersonation_and_typosquatting() -> None:
    # Legitimate official domain must not trigger impersonation
    score, brand = detect_brand_impersonation(parse_url("https://www.paypal.com/myaccount"))
    assert score == 0.0
    assert brand is None

    # Legitimate Microsoft Entra/SSO domain must not trigger impersonation
    score, brand = detect_brand_impersonation(
        parse_url("https://login.microsoftonline.com/common/oauth2/authorize")
    )
    assert score == 0.0
    assert brand is None

    # Brand in subdomain on attacker domain
    score, brand = detect_brand_impersonation(
        parse_url("https://paypal.com.verify-user.tk/signin")
    )
    assert score == 1.0
    assert brand == "paypal"

    # Lookalike / leet-speak typosquatting
    score, brand = detect_brand_impersonation(parse_url("https://arnazon-security.com/login"))
    assert score == 1.0
    assert brand == "amazon"

    score, brand = detect_brand_impersonation(parse_url("https://paypa1.com/account"))
    assert score == 1.0
    assert brand == "paypal"


def test_lexical_features_and_security_signals() -> None:
    phish_url = "http://paypal.com@secure-login-verify.tk:8080/account/update?id=12345"
    feats = extract_lexical_features(phish_url)
    vec = lexical_feature_vector(phish_url)

    assert len(vec) == len(LEXICAL_FEATURE_NAMES) == 18
    assert feats["has_at_userinfo"] == 1.0
    assert feats["has_non_standard_port"] == 1.0
    assert feats["suspicious_tld"] == 1.0
    assert feats["is_https"] == 0.0
    assert feats["sensitive_keyword_count"] >= 3.0

    signals = extract_security_signals(phish_url)
    rule_ids = {s.rule_id for s in signals}
    assert {"PG002", "PG005", "PG006", "PG008", "PG010"}.issubset(rule_ids)
