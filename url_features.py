import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlparse

import tldextract


STRUCTURE_COLUMNS = [
    "url_length",
    "domain_length",
    "is_https",
    "has_ip",
    "has_at_symbol",
    "has_hyphen",
    "double_slash_redirect",
    "subdomain_count",
    "is_shortened",
    "has_sensitive_keyword",
    "count_dots",
    "count_question",
    "count_equal",
    "count_hyphen",
    "count_slash",
    "count_digits",
    "digits_ratio",
    "url_entropy",
]


_TLD_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=None
)


SHORTENER_DOMAINS = {
    "bit.ly",
    "goo.gl",
    "tinyurl.com",
    "is.gd",
    "cli.gs",
    "t.co",
    "ow.ly",
    "adf.ly",
    "rb.gy",
    "reurl.cc",
    "cutt.ly",
    "rebrand.ly",
}


SENSITIVE_WORDS = {
    "login",
    "signin",
    "verify",
    "verification",
    "account",
    "password",
    "secure",
    "security",
    "update",
    "confirm",
    "bank",
    "wallet",
    "refund",
    "tax",
    "bonus",
    "prize",
    "payment",
    "authenticate",
    "authentication",
}


def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0

    counts = Counter(text)
    length = len(text)
    entropy = 0.0

    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return round(entropy, 4)


def normalize_url(url: str) -> str:
    if not isinstance(url, str):
        url = ""

    url = url.strip()

    if not url:
        return "http://unknown.com"

    url = (
        url
        .replace("[.]", ".")
        .replace("hxxps://", "https://")
        .replace("hxxp://", "http://")
    )

    if not url.lower().startswith(
        ("http://", "https://")
    ):
        url = "http://" + url

    return url


def is_shortened_domain(hostname: str) -> int:
    hostname = hostname.lower().strip(".")

    if hostname.startswith("www."):
        hostname = hostname[4:]

    for shortener in SHORTENER_DOMAINS:
        if (
            hostname == shortener
            or hostname.endswith("." + shortener)
        ):
            return 1

    return 0


def contains_sensitive_keyword(url: str) -> int:
    tokens = {
        token
        for token in re.split(
            r"[^a-z0-9]+",
            url.lower(),
        )
        if token
    }

    return int(
        bool(tokens.intersection(SENSITIVE_WORDS))
    )


def extract_universal_features(url: str) -> dict:
    url = normalize_url(url)

    digit_count = sum(
        character.isdigit()
        for character in url
    )

    features = {
        "url_length": len(url),
        "domain_length": 0,
        "is_https": 0,
        "has_ip": 0,
        "has_at_symbol": int("@" in url),
        "has_hyphen": 0,
        "double_slash_redirect": 0,
        "subdomain_count": 0,
        "is_shortened": 0,
        "has_sensitive_keyword": 0,
        "count_dots": url.count("."),
        "count_question": url.count("?"),
        "count_equal": url.count("="),
        "count_hyphen": url.count("-"),
        "count_slash": url.count("/"),
        "count_digits": digit_count,
        "digits_ratio": (
            round(digit_count / len(url), 4)
            if url
            else 0.0
        ),
        "url_entropy": calculate_entropy(url),
    }

    try:
        parsed = urlparse(url)
        hostname = (
            parsed.hostname
            or ""
        ).lower()

        features["domain_length"] = len(hostname)
        features["is_https"] = int(
            parsed.scheme.lower() == "https"
        )
        features["has_hyphen"] = int(
            "-" in hostname
        )

        scheme_end = url.find("://")

        if scheme_end >= 0:
            remaining_url = url[
                scheme_end + 3:
            ]

            features[
                "double_slash_redirect"
            ] = int(
                "//" in remaining_url
            )

        try:
            ipaddress.ip_address(hostname)
            features["has_ip"] = 1

        except ValueError:
            features["has_ip"] = 0

        if features["has_ip"] == 0:
            extracted = _TLD_EXTRACTOR(hostname)

            subdomain = (
                extracted.subdomain
                or ""
            ).strip(".")

            if subdomain:
                features["subdomain_count"] = len(
                    [
                        part
                        for part in subdomain.split(".")
                        if part
                    ]
                )

        features["is_shortened"] = (
            is_shortened_domain(hostname)
        )

        features["has_sensitive_keyword"] = (
            contains_sensitive_keyword(url)
        )

    except Exception:
        pass

    return features
