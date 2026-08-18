import re
import socket
import ssl
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import tldextract
import whois
from bs4 import BeautifulSoup


# =========================================================
# 1. 工具函式
# =========================================================

def ensure_url_scheme(url: str) -> str:
    url = str(url).strip()

    if not url.startswith(
        ("http://", "https://")
    ):
        url = "http://" + url

    return url


def safe_hostname(url: str) -> str:
    try:
        parsed = urlparse(
            ensure_url_scheme(url)
        )
        return parsed.hostname or ""
    except Exception:
        return ""


def is_ip_address(hostname: str) -> bool:
    pattern = (
        r"^\d{1,3}\."
        r"\d{1,3}\."
        r"\d{1,3}\."
        r"\d{1,3}$"
    )

    return bool(
        re.fullmatch(
            pattern,
            hostname
        )
    )


def fetch_html(
    url: str,
    timeout: int = 8
):

    try:
        response = requests.get(
            ensure_url_scheme(url),
            timeout=timeout,
            allow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            },
        )

        return response

    except Exception:
        return None


# =========================================================
# 2. 25 個特徵
#
# 值的方向沿用常見 phishing dataset：
#
#  1  = 正常
#  0  = 可疑
# -1  = 高風險
# =========================================================

def feature_having_ip(url: str) -> int:
    hostname = safe_hostname(url)
    return -1 if is_ip_address(hostname) else 1


def feature_url_length(url: str) -> int:
    length = len(url)

    if length < 54:
        return 1

    if length <= 75:
        return 0

    return -1


def feature_shortening_service(url: str) -> int:
    shorteners = [
        "bit.ly",
        "goo.gl",
        "tinyurl.com",
        "t.co",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "adf.ly",
        "tiny.cc",
    ]

    hostname = safe_hostname(url).lower()

    return (
        -1
        if any(
            item in hostname
            for item in shorteners
        )
        else 1
    )


def feature_at_symbol(url: str) -> int:
    return -1 if "@" in url else 1


def feature_double_slash_redirect(url: str) -> int:
    url = ensure_url_scheme(url)
    rest = url.split(
        "://",
        1
    )[-1]

    return -1 if "//" in rest else 1


def feature_prefix_suffix(url: str) -> int:
    hostname = safe_hostname(url)
    return -1 if "-" in hostname else 1


def feature_sub_domain(url: str) -> int:
    hostname = safe_hostname(url)

    if is_ip_address(hostname):
        return -1

    extracted = tldextract.extract(
        hostname
    )

    if not extracted.subdomain:
        return 1

    count = len(
        [
            part
            for part in extracted.subdomain.split(".")
            if part
        ]
    )

    if count == 1:
        return 0

    if count >= 2:
        return -1

    return 1


def feature_ssl_final_state(url: str) -> int:
    url = ensure_url_scheme(url)

    parsed = urlparse(url)

    if parsed.scheme.lower() != "https":
        return -1

    hostname = parsed.hostname

    if not hostname:
        return -1

    try:
        context = ssl.create_default_context()

        with socket.create_connection(
            (hostname, 443),
            timeout=5
        ) as sock:

            with context.wrap_socket(
                sock,
                server_hostname=hostname
            ):
                return 1

    except Exception:
        return 0


def feature_domain_registration_length(url: str) -> int:
    hostname = safe_hostname(url)

    if not hostname:
        return -1

    try:
        domain_info = whois.whois(
            hostname
        )

        creation_date = (
            domain_info.creation_date
        )

        expiration_date = (
            domain_info.expiration_date
        )

        if isinstance(
            creation_date,
            list
        ):
            creation_date = creation_date[0]

        if isinstance(
            expiration_date,
            list
        ):
            expiration_date = expiration_date[0]

        if (
            creation_date is None
            or
            expiration_date is None
        ):
            return 0

        days = (
            expiration_date
            - creation_date
        ).days

        return (
            1
            if days >= 365
            else -1
        )

    except Exception:
        return 0


def feature_favicon(
    url: str,
    soup
) -> int:

    hostname = safe_hostname(url)

    if soup is None:
        return 0

    icon = soup.find(
        "link",
        rel=lambda value:
            value
            and
            "icon" in str(value).lower()
    )

    if not icon:
        return 1

    href = icon.get(
        "href",
        ""
    )

    if not href:
        return 1

    if href.startswith(
        ("/", "./", "../")
    ):
        return 1

    icon_host = safe_hostname(
        href
    )

    if not icon_host:
        return 1

    return (
        1
        if icon_host == hostname
        else -1
    )


def feature_port(url: str) -> int:
    try:
        parsed = urlparse(
            ensure_url_scheme(url)
        )

        port = parsed.port

        if port is None:
            return 1

        return (
            1
            if port in (80, 443)
            else -1
        )

    except Exception:
        return 0


def feature_https_token(url: str) -> int:
    hostname = safe_hostname(url).lower()

    return (
        -1
        if "https" in hostname
        else 1
    )


def feature_request_url(
    url: str,
    soup
) -> int:

    if soup is None:
        return 0

    hostname = safe_hostname(url)

    resources = []

    for tag, attr in [
        ("img", "src"),
        ("audio", "src"),
        ("embed", "src"),
        ("iframe", "src"),
        ("script", "src"),
        ("link", "href"),
    ]:

        for item in soup.find_all(tag):

            value = item.get(
                attr
            )

            if value:
                resources.append(
                    value
                )

    if not resources:
        return 1

    external = 0
    total = 0

    for resource in resources:

        if resource.startswith(
            (
                "data:",
                "javascript:",
                "#",
            )
        ):
            continue

        total += 1

        resource_host = safe_hostname(
            resource
        )

        if (
            resource_host
            and
            resource_host != hostname
        ):
            external += 1

    if total == 0:
        return 1

    ratio = (
        external
        / total
        * 100
    )

    if ratio < 22:
        return 1

    if ratio <= 61:
        return 0

    return -1


def feature_anchor_url(
    url: str,
    soup
) -> int:

    if soup is None:
        return 0

    hostname = safe_hostname(url)

    anchors = soup.find_all(
        "a"
    )

    if not anchors:
        return 1

    unsafe = 0
    total = 0

    for anchor in anchors:

        href = str(
            anchor.get(
                "href",
                ""
            )
        ).strip()

        if not href:
            continue

        total += 1

        lower = href.lower()

        if (
            lower.startswith("#")
            or
            lower.startswith(
                "javascript:"
            )
            or
            lower.startswith(
                "mailto:"
            )
        ):
            unsafe += 1
            continue

        href_host = safe_hostname(
            href
        )

        if (
            href_host
            and
            href_host != hostname
        ):
            unsafe += 1

    if total == 0:
        return 1

    ratio = (
        unsafe
        / total
        * 100
    )

    if ratio < 31:
        return 1

    if ratio <= 67:
        return 0

    return -1


def feature_links_in_tags(
    url: str,
    soup
) -> int:

    if soup is None:
        return 0

    hostname = safe_hostname(url)

    links = []

    for tag in [
        "link",
        "script",
        "meta",
    ]:

        for item in soup.find_all(
            tag
        ):

            value = (
                item.get("href")
                or
                item.get("src")
                or
                item.get("content")
            )

            if value:
                links.append(
                    str(value)
                )

    if not links:
        return 1

    external = 0
    total = 0

    for link in links:

        if link.startswith(
            (
                "data:",
                "#",
            )
        ):
            continue

        total += 1

        link_host = safe_hostname(
            link
        )

        if (
            link_host
            and
            link_host != hostname
        ):
            external += 1

    if total == 0:
        return 1

    ratio = (
        external
        / total
        * 100
    )

    if ratio < 17:
        return 1

    if ratio <= 81:
        return 0

    return -1


def feature_sfh(
    url: str,
    soup
) -> int:

    if soup is None:
        return 0

    hostname = safe_hostname(url)

    forms = soup.find_all(
        "form"
    )

    if not forms:
        return 1

    for form in forms:

        action = str(
            form.get(
                "action",
                ""
            )
        ).strip()

        if not action:
            return -1

        if action.lower() in [
            "about:blank",
            "#",
        ]:
            return -1

        action_host = safe_hostname(
            action
        )

        if (
            action_host
            and
            action_host != hostname
        ):
            return 0

    return 1


def feature_submitting_to_email(
    soup
) -> int:

    if soup is None:
        return 0

    html = str(
        soup
    ).lower()

    if (
        "mailto:" in html
        or
        "mail(" in html
    ):
        return -1

    return 1


def feature_abnormal_url(
    url: str
) -> int:

    hostname = safe_hostname(url)

    if not hostname:
        return -1

    lower_url = url.lower()

    return (
        1
        if hostname.lower() in lower_url
        else -1
    )


def feature_redirect(
    response
) -> int:

    if response is None:
        return 0

    count = len(
        response.history
    )

    if count <= 1:
        return 1

    if count <= 4:
        return 0

    return -1


def feature_on_mouseover(
    soup
) -> int:

    if soup is None:
        return 0

    html = str(
        soup
    ).lower()

    return (
        -1
        if "onmouseover" in html
        else 1
    )


def feature_right_click(
    soup
) -> int:

    if soup is None:
        return 0

    html = str(
        soup
    ).lower()

    patterns = [
        "event.button==2",
        "event.button == 2",
        "contextmenu",
    ]

    return (
        -1
        if any(
            pattern in html
            for pattern in patterns
        )
        else 1
    )


def feature_popup_window(
    soup
) -> int:

    if soup is None:
        return 0

    html = str(
        soup
    ).lower()

    patterns = [
        "window.open",
        "alert(",
        "prompt(",
    ]

    return (
        -1
        if any(
            pattern in html
            for pattern in patterns
        )
        else 1
    )


def feature_iframe(
    soup
) -> int:

    if soup is None:
        return 0

    return (
        -1
        if soup.find("iframe")
        else 1
    )


def feature_age_of_domain(
    url: str
) -> int:

    hostname = safe_hostname(url)

    if not hostname:
        return -1

    try:
        domain_info = whois.whois(
            hostname
        )

        creation_date = (
            domain_info.creation_date
        )

        if isinstance(
            creation_date,
            list
        ):
            creation_date = creation_date[0]

        if creation_date is None:
            return 0

        from datetime import datetime

        age_days = (
            datetime.now()
            - creation_date.replace(
                tzinfo=None
            )
        ).days

        return (
            1
            if age_days >= 180
            else -1
        )

    except Exception:
        return 0


def feature_dns_record(
    url: str
) -> int:

    hostname = safe_hostname(url)

    if not hostname:
        return -1

    try:
        socket.gethostbyname(
            hostname
        )
        return 1

    except Exception:
        return -1


# =========================================================
# 3. 特徵主函式
# =========================================================

def extract_url_features_25(
    url: str,
    feature_columns
):

    url = ensure_url_scheme(
        url
    )

    response = fetch_html(
        url
    )

    soup = None

    if (
        response is not None
        and
        response.text
    ):
        try:
            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )
        except Exception:
            soup = None

    feature_values = {

        "having_IPhaving_IP_Address":
            feature_having_ip(url),

        "URLURL_Length":
            feature_url_length(url),

        "Shortining_Service":
            feature_shortening_service(url),

        "having_At_Symbol":
            feature_at_symbol(url),

        "double_slash_redirecting":
            feature_double_slash_redirect(url),

        "Prefix_Suffix":
            feature_prefix_suffix(url),

        "having_Sub_Domain":
            feature_sub_domain(url),

        "SSLfinal_State":
            feature_ssl_final_state(url),

        "Domain_registeration_length":
            feature_domain_registration_length(url),

        "Favicon":
            feature_favicon(
                url,
                soup
            ),

        "port":
            feature_port(url),

        "HTTPS_token":
            feature_https_token(url),

        "Request_URL":
            feature_request_url(
                url,
                soup
            ),

        "URL_of_Anchor":
            feature_anchor_url(
                url,
                soup
            ),

        "Links_in_tags":
            feature_links_in_tags(
                url,
                soup
            ),

        "SFH":
            feature_sfh(
                url,
                soup
            ),

        "Submitting_to_email":
            feature_submitting_to_email(
                soup
            ),

        "Abnormal_URL":
            feature_abnormal_url(url),

        "Redirect":
            feature_redirect(
                response
            ),

        "on_mouseover":
            feature_on_mouseover(
                soup
            ),

        "RightClick":
            feature_right_click(
                soup
            ),

        "popUpWidnow":
            feature_popup_window(
                soup
            ),

        "Iframe":
            feature_iframe(
                soup
            ),

        "age_of_domain":
            feature_age_of_domain(url),

        "DNSRecord":
            feature_dns_record(url),
    }

    missing = [
        column
        for column in feature_columns
        if column not in feature_values
    ]

    if missing:
        raise ValueError(
            "缺少模型需要的 URL 特徵："
            f"{missing}"
        )

    ordered_features = {
        column: feature_values[column]
        for column in feature_columns
    }

    feature_df = pd.DataFrame(
        [ordered_features],
        columns=feature_columns,
    )

    return feature_df
