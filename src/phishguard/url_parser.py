"""Safe, purely offline URL parser and normalizer.

This module deliberately avoids making any network or DNS calls so untrusted
URLs can be inspected without SSRF, tracking-pixel leaks, or drive-by risks.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import re
from urllib.parse import urlsplit

# Common multi-part public suffixes so registrable-domain splitting stays
# accurate without requiring an external PSL network download.
MULTI_PART_TLDS: frozenset[str] = frozenset(
    {
        "co.uk",
        "org.uk",
        "gov.uk",
        "ac.uk",
        "com.au",
        "net.au",
        "org.au",
        "co.jp",
        "ne.jp",
        "or.jp",
        "com.br",
        "com.cn",
        "com.mx",
        "co.in",
        "co.nz",
        "co.za",
        "com.sg",
        "com.tr",
        "com.ua",
        "co.kr",
    }
)

_DEFANG_SCHEME_RE = re.compile(r"^hxxp(s?)://", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedURL:
    """Structured offline representation of a URL."""

    raw: str
    normalized: str
    scheme: str
    userinfo: str
    hostname: str
    port: int | None
    subdomain: str
    domain_label: str
    tld: str
    registrable_domain: str
    path: str
    query: str
    fragment: str
    is_ip: bool
    has_punycode: bool
    has_non_ascii_host: bool


def refang_url(raw_url: str) -> str:
    """Convert common SOC defanged URLs (hxxp://, [.]) into standard syntax."""
    cleaned = raw_url.strip()
    cleaned = _DEFANG_SCHEME_RE.sub(r"http\1://", cleaned)
    cleaned = cleaned.replace("[.]", ".").replace("(.)", ".")
    return cleaned


def _is_ip_address(hostname: str) -> bool:
    """Return True if hostname is an IPv4 or IPv6 literal (including bracketed)."""
    candidate = hostname.strip("[]")
    if not candidate:
        return False
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        pass
    # Also catch hexadecimal or decimal-encoded IPv4 tricks like 0x7f000001
    if re.fullmatch(r"0x[0-9a-f]+", candidate, re.IGNORECASE):
        try:
            ipaddress.IPv4Address(int(candidate, 16))
            return True
        except ValueError:
            return False
    if candidate.isdigit():
        try:
            ipaddress.IPv4Address(int(candidate))
            return True
        except ValueError:
            return False
    return False


def _split_domain(hostname: str, is_ip: bool) -> tuple[str, str, str, str]:
    """Split hostname into (subdomain, domain_label, tld, registrable_domain)."""
    host = hostname.strip(".").lower()
    if not host or is_ip:
        return "", host, "", host

    parts = [p for p in host.split(".") if p]
    if len(parts) == 1:
        return "", parts[0], "", parts[0]

    last_two = ".".join(parts[-2:])
    if len(parts) >= 3 and last_two in MULTI_PART_TLDS:
        tld = last_two
        domain_label = parts[-3]
        subdomain = ".".join(parts[:-3])
        registrable = f"{domain_label}.{tld}"
        return subdomain, domain_label, tld, registrable

    tld = parts[-1]
    domain_label = parts[-2]
    subdomain = ".".join(parts[:-2])
    registrable = f"{domain_label}.{tld}"
    return subdomain, domain_label, tld, registrable


def parse_url(raw_url: str) -> ParsedURL:
    """Safely parse a URL without performing any network operations.

    Handles scheme-less URLs, defanged SOC indicators, IPv4/IPv6 literals,
    invalid ports, and internationalized domain names.
    """
    refanged = refang_url(raw_url)
    if not refanged:
        return ParsedURL(
            raw=raw_url,
            normalized="",
            scheme="",
            userinfo="",
            hostname="",
            port=None,
            subdomain="",
            domain_label="",
            tld="",
            registrable_domain="",
            path="",
            query="",
            fragment="",
            is_ip=False,
            has_punycode=False,
            has_non_ascii_host=False,
        )

    to_parse = refanged
    if "://" not in to_parse:
        to_parse = f"http://{to_parse}"

    split = urlsplit(to_parse)
    scheme = (split.scheme or "http").lower()

    # Extract userinfo manually so malformed credentials never break parsing.
    netloc = split.netloc
    userinfo = ""
    hostport = netloc
    if "@" in netloc:
        userinfo, hostport = netloc.rsplit("@", 1)

    hostname = ""
    port: int | None = None
    if hostport.startswith("["):
        # IPv6 literal: [2001:db8::1]:8080
        closing = hostport.find("]")
        if closing != -1:
            hostname = hostport[1:closing].lower()
            remainder = hostport[closing + 1 :]
            if remainder.startswith(":") and remainder[1:].isdigit():
                parsed_port = int(remainder[1:])
                if 0 < parsed_port <= 65535:
                    port = parsed_port
        else:
            hostname = hostport.strip("[]").lower()
    else:
        if ":" in hostport:
            host_part, port_part = hostport.rsplit(":", 1)
            hostname = host_part.lower()
            if port_part.isdigit():
                parsed_port = int(port_part)
                if 0 < parsed_port <= 65535:
                    port = parsed_port
        else:
            hostname = hostport.lower()

    is_ip = _is_ip_address(hostname)
    has_punycode = "xn--" in hostname
    has_non_ascii_host = any(ord(ch) > 127 for ch in hostname)

    subdomain, domain_label, tld, registrable = _split_domain(hostname, is_ip)

    path = split.path or ""
    query = split.query or ""
    fragment = split.fragment or ""

    normalized = f"{scheme}://{netloc}{path}"
    if query:
        normalized = f"{normalized}?{query}"
    if fragment:
        normalized = f"{normalized}#{fragment}"

    return ParsedURL(
        raw=raw_url,
        normalized=normalized,
        scheme=scheme,
        userinfo=userinfo,
        hostname=hostname,
        port=port,
        subdomain=subdomain,
        domain_label=domain_label,
        tld=tld,
        registrable_domain=registrable,
        path=path,
        query=query,
        fragment=fragment,
        is_ip=is_ip,
        has_punycode=has_punycode,
        has_non_ascii_host=has_non_ascii_host,
    )
