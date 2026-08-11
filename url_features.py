import re
import socket
import ssl
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

import pandas as pd
import requests
import whois
from bs4 import BeautifulSoup


# =========================================================
# 網頁下載
# =========================================================

def fetch_page(url):
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/120 Safari/537.36"
            )
        }

        return requests.get(
            url,
            headers=headers,
            timeout=8,
            allow_redirects=True,
        )

    except Exception:
        return None


# =========================================================
# 判斷是否同網域
# =========================================================

def same_domain(base_url, target_url):
    try:
        base_host = (
            urlparse(base_url).hostname or ""
        ).lower()

        target_host = (
            urlparse(target_url).hostname or ""
        ).lower()

        if not target_host:
            return True

        return (
            target_host == base_host
            or target_host.endswith("." + base_host)
        )

    except Exception:
        return False


# =========================================================
# 1. having_IPhaving_IP_Address
# =========================================================

def having_ip_address(url):
    try:
        import ipaddress

        host = urlparse(url).hostname

        if not host:
            return 1

        ipaddress.ip_address(host)

        return -1

    except ValueError:
        return 1

    except Exception:
        return 1


# =========================================================
# 2. URLURL_Length
# =========================================================

def url_length(url):
    length = len(url)

    if length < 54:
        return 1

    if length <= 75:
        return 0

    return -1


# =========================================================
# 3. Shortining_Service
# =========================================================

def shortening_service(url):
    shortening_domains = [
        "bit.ly",
        "goo.gl",
        "tinyurl.com",
        "t.co",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "adf.ly",
        "bit.do",
        "cutt.ly",
        "rebrand.ly",
        "shorturl.at",
    ]

    host = (
        urlparse(url).hostname or ""
    ).lower()

    for domain in shortening_domains:
        if (
            host == domain
            or host.endswith("." + domain)
        ):
            return -1

    return 1


# =========================================================
# 4. having_At_Symbol
# =========================================================

def having_at_symbol(url):
    return -1 if "@" in url else 1


# =========================================================
# 5. double_slash_redirecting
# =========================================================

def double_slash_redirecting(url):
    position = url.rfind("//")

    if position > 7:
        return -1

    return 1


# =========================================================
# 6. Prefix_Suffix
# =========================================================

def prefix_suffix(url):
    host = urlparse(url).hostname or ""

    return -1 if "-" in host else 1


# =========================================================
# 7. having_Sub_Domain
# =========================================================

def having_sub_domain(url):
    host = urlparse(url).hostname or ""

    host = re.sub(
        r"^www\.",
        "",
        host,
    )

    dots = host.count(".")

    if dots == 1:
        return 1

    if dots == 2:
        return 0

    return -1


# =========================================================
# 8. SSLfinal_State
# =========================================================

def ssl_final_state(url):
    try:
        parsed = urlparse(url)
        host = parsed.hostname

        if not host:
            return -1

        if parsed.scheme.lower() != "https":
            return -1

        context = ssl.create_default_context()

        with socket.create_connection(
            (host, 443),
            timeout=5,
        ) as sock:

            with context.wrap_socket(
                sock,
                server_hostname=host,
            ) as secure_sock:

                certificate = (
                    secure_sock.getpeercert()
                )

        if certificate:
            return 1

        return 0

    except Exception:
        return -1


# =========================================================
# WHOIS
# =========================================================

def get_whois_info(url):
    try:
        host = urlparse(url).hostname

        if not host:
            return None

        return whois.whois(host)

    except Exception:
        return None


def normalize_date(value):
    if isinstance(value, list):
        if not value:
            return None

        value = value[0]

    if not isinstance(value, datetime):
        return None

    if value.tzinfo is None:
        value = value.replace(
            tzinfo=timezone.utc
        )

    return value


# =========================================================
# 9. Domain_registeration_length
# =========================================================

def domain_registration_length(whois_info):
    try:
        if whois_info is None:
            return 0

        creation = normalize_date(
            whois_info.creation_date
        )

        expiration = normalize_date(
            whois_info.expiration_date
        )

        if creation is None or expiration is None:
            return 0

        days = (
            expiration - creation
        ).days

        if days >= 365:
            return 1

        return -1

    except Exception:
        return 0


# =========================================================
# 10. Favicon
# =========================================================

def favicon_feature(url, soup):
    try:
        icon = soup.find(
            "link",
            rel=lambda x:
                x
                and "icon" in str(x).lower()
        )

        if not icon:
            return 1

        href = icon.get("href")

        if not href:
            return 1

        full_url = urljoin(
            url,
            href,
        )

        if same_domain(url, full_url):
            return 1

        return -1

    except Exception:
        return 0


# =========================================================
# 11. port
# =========================================================

def port_feature(url):
    try:
        parsed = urlparse(url)

        if parsed.port is None:
            return 1

        if parsed.port in [80, 443]:
            return 1

        return -1

    except Exception:
        return -1


# =========================================================
# 12. HTTPS_token
# =========================================================

def https_token(url):
    host = (
        urlparse(url).hostname or ""
    )

    if "https" in host.lower():
        return -1

    return 1


# =========================================================
# 13. Request_URL
# =========================================================

def request_url_feature(url, soup):
    try:
        resources = []

        tags = [
            ("img", "src"),
            ("script", "src"),
            ("audio", "src"),
            ("video", "src"),
            ("source", "src"),
        ]

        for tag, attribute in tags:
            for item in soup.find_all(tag):

                value = item.get(attribute)

                if value:
                    resources.append(
                        urljoin(url, value)
                    )

        if not resources:
            return 1

        external = sum(
            1
            for resource in resources
            if not same_domain(
                url,
                resource,
            )
        )

        ratio = external / len(resources)

        if ratio < 0.22:
            return 1

        if ratio <= 0.61:
            return 0

        return -1

    except Exception:
        return 0


# =========================================================
# 14. URL_of_Anchor
# =========================================================

def url_of_anchor_feature(url, soup):
    try:
        anchors = soup.find_all(
            "a",
            href=True,
        )

        if not anchors:
            return 1

        suspicious = 0

        for anchor in anchors:
            href = (
                anchor.get(
                    "href",
                    "",
                )
                .strip()
                .lower()
            )

            if (
                href == ""
                or href.startswith("#")
                or href.startswith(
                    "javascript:"
                )
                or href.startswith(
                    "mailto:"
                )
            ):
                suspicious += 1
                continue

            full_url = urljoin(
                url,
                href,
            )

            if not same_domain(
                url,
                full_url,
            ):
                suspicious += 1

        ratio = (
            suspicious
            / len(anchors)
        )

        if ratio < 0.31:
            return 1

        if ratio <= 0.67:
            return 0

        return -1

    except Exception:
        return 0


# =========================================================
# 15. Links_in_tags
# =========================================================

def links_in_tags_feature(url, soup):
    try:
        links = []

        for tag in soup.find_all(
            ["meta", "script", "link"]
        ):

            value = (
                tag.get("href")
                or tag.get("src")
                or tag.get("content")
            )

            if not value:
                continue

            value = str(value)

            if (
                value.startswith("http")
                or value.startswith("/")
            ):
                links.append(
                    urljoin(
                        url,
                        value,
                    )
                )

        if not links:
            return 1

        external = sum(
            1
            for link in links
            if not same_domain(
                url,
                link,
            )
        )

        ratio = external / len(links)

        if ratio < 0.17:
            return 1

        if ratio <= 0.81:
            return 0

        return -1

    except Exception:
        return 0


# =========================================================
# 16. SFH
# =========================================================

def sfh_feature(url, soup):
    try:
        forms = soup.find_all("form")

        if not forms:
            return 1

        for form in forms:

            action = (
                form.get("action")
                or ""
            ).strip()

            if action == "":
                return -1

            action_lower = action.lower()

            if (
                action_lower == "about:blank"
                or action_lower.startswith(
                    "javascript:"
                )
            ):
                return -1

            full_url = urljoin(
                url,
                action,
            )

            if not same_domain(
                url,
                full_url,
            ):
                return 0

        return 1

    except Exception:
        return 0


# =========================================================
# 17. Submitting_to_email
# =========================================================

def submitting_to_email(url):
    lower_url = url.lower()

    if (
        "mailto:" in lower_url
        or "mail(" in lower_url
        or "mail.php" in lower_url
    ):
        return -1

    return 1


# =========================================================
# 18. Abnormal_URL
# =========================================================

def abnormal_url(url):
    try:
        parsed = urlparse(url)

        host = parsed.hostname or ""

        if not host:
            return -1

        if host.lower() in url.lower():
            return 1

        return -1

    except Exception:
        return -1


# =========================================================
# 19. Redirect
# =========================================================

def redirect_feature(url):
    try:
        occurrences = url.count("//")

        if occurrences <= 1:
            return 1

        return -1

    except Exception:
        return -1


# =========================================================
# 20. on_mouseover
# =========================================================

def on_mouseover_feature(html):
    try:
        if "onmouseover" in html.lower():
            return -1

        return 1

    except Exception:
        return 0


# =========================================================
# 21. RightClick
# =========================================================

def right_click_feature(html):
    try:
        html_lower = html.lower()

        patterns = [
            "event.button==2",
            "event.button == 2",
            "contextmenu",
            "oncontextmenu",
        ]

        for pattern in patterns:
            if pattern in html_lower:
                return -1

        return 1

    except Exception:
        return 0


# =========================================================
# 22. popUpWidnow
# =========================================================

def popup_window_feature(html):
    try:
        html_lower = html.lower()

        if (
            "window.open" in html_lower
            or "alert(" in html_lower
        ):
            return -1

        return 1

    except Exception:
        return 0


# =========================================================
# 23. Iframe
# =========================================================

def iframe_feature(soup):
    try:
        if soup.find("iframe"):
            return -1

        return 1

    except Exception:
        return 0


# =========================================================
# 24. age_of_domain
# =========================================================

def age_of_domain(whois_info):
    try:
        if whois_info is None:
            return 0

        creation = normalize_date(
            whois_info.creation_date
        )

        if creation is None:
            return 0

        now = datetime.now(
            timezone.utc
        )

        age_days = (
            now - creation
        ).days

        if age_days >= 180:
            return 1

        return -1

    except Exception:
        return 0


# =========================================================
# 25. DNSRecord
# =========================================================

def dns_record(url):
    try:
        host = urlparse(url).hostname

        if not host:
            return -1

        socket.gethostbyname(host)

        return 1

    except Exception:
        return -1


# =========================================================
# 建立模型需要的 25 個特徵
# =========================================================

def extract_url_features_25(
    url,
    feature_columns,
):

    response = fetch_page(url)

    if response is not None:

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        html = response.text

    else:

        soup = BeautifulSoup(
            "",
            "html.parser",
        )

        html = ""

    whois_info = get_whois_info(url)

    features = {

        "having_IPhaving_IP_Address":
            having_ip_address(url),

        "URLURL_Length":
            url_length(url),

        "Shortining_Service":
            shortening_service(url),

        "having_At_Symbol":
            having_at_symbol(url),

        "double_slash_redirecting":
            double_slash_redirecting(url),

        "Prefix_Suffix":
            prefix_suffix(url),

        "having_Sub_Domain":
            having_sub_domain(url),

        "SSLfinal_State":
            ssl_final_state(url),

        "Domain_registeration_length":
            domain_registration_length(
                whois_info
            ),

        "Favicon":
            favicon_feature(
                url,
                soup,
            ),

        "port":
            port_feature(url),

        "HTTPS_token":
            https_token(url),

        "Request_URL":
            request_url_feature(
                url,
                soup,
            ),

        "URL_of_Anchor":
            url_of_anchor_feature(
                url,
                soup,
            ),

        "Links_in_tags":
            links_in_tags_feature(
                url,
                soup,
            ),

        "SFH":
            sfh_feature(
                url,
                soup,
            ),

        "Submitting_to_email":
            submitting_to_email(url),

        "Abnormal_URL":
            abnormal_url(url),

        "Redirect":
            redirect_feature(url),

        "on_mouseover":
            on_mouseover_feature(html),

        "RightClick":
            right_click_feature(html),

        "popUpWidnow":
            popup_window_feature(html),

        "Iframe":
            iframe_feature(soup),

        "age_of_domain":
            age_of_domain(
                whois_info
            ),

        "DNSRecord":
            dns_record(url),
    }

    dataframe = pd.DataFrame(
        [features]
    )

    dataframe = dataframe[
        feature_columns
    ]

    return dataframe
