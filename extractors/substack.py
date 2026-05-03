import requests
from bs4 import BeautifulSoup


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_substack_content(url: str) -> str:
    """Fetch the readable text of a Substack article."""
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, "lxml")

    # Remove nav, footer, script, style noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Substack article body lives in one of these containers
    for selector in [
        "div.available-content",
        "div.post-content",
        "article",
    ]:
        container = soup.select_one(selector)
        if container:
            return container.get_text(separator=" ", strip=True)

    # Fallback: whole body, capped to avoid token bloat
    body = soup.find("body")
    return body.get_text(separator=" ", strip=True)[:15_000] if body else ""
