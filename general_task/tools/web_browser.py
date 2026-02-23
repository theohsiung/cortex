"""Web browser tool using Playwright + requests fallback."""

from __future__ import annotations

from google.adk.tools import FunctionTool


def _extract_text_from_html(html: str, max_length: int = 8000) -> str:
    """Extract clean text from HTML content."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    if len(text) > max_length:
        text = text[:max_length] + f"\n\n... (truncated, total {len(text)} chars)"
    return text


def _web_browser_playwright(url: str, timeout_ms: int = 15000) -> str:
    """Fetch page using Playwright headless browser (handles JS-rendered pages)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=timeout_ms)

        # Auto-click common consent/accept buttons (for sites like USGS, cookie banners)
        consent_selectors = [
            "button:has-text('Accept')",
            "button:has-text('Agree')",
            "button:has-text('OK')",
            "button:has-text('I Agree')",
            "button:has-text('Got it')",
            "button:has-text('Continue')",
            "a:has-text('Accept')",
            "a:has-text('Agree')",
            "a:has-text('OK')",
            "[class*='accept']",
            "[class*='consent']",
            "[id*='accept']",
            "[id*='consent']",
        ]
        for selector in consent_selectors:
            try:
                btn = page.locator(selector).first
                if btn.is_visible(timeout=500):
                    btn.click(timeout=1000)
                    page.wait_for_load_state("networkidle", timeout=3000)
                    break
            except Exception:
                continue

        html = page.content()
        browser.close()
    return str(html)


def _web_browser_requests(url: str) -> str:
    """Fetch page using requests (fallback for static pages)."""
    import requests  # type: ignore[import-untyped]

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return str(response.text)


def web_browser(url: str, action: str = "navigate") -> str:
    """Web browser with Playwright headless (JS-rendering) + requests fallback.

    Fetches URL and extracts text content.

    Status: IMPLEMENTED
    Dependencies: playwright (preferred), requests + beautifulsoup4 (fallback)
    Install: pip install playwright && playwright install chromium
             pip install requests beautifulsoup4
    """
    try:
        import bs4  # noqa: F401 - needed by _extract_text_from_html
    except ImportError:
        return "[ERROR] beautifulsoup4 not installed. Run: pip install beautifulsoup4"

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        html = None
        method = "unknown"

        # Try Playwright first (handles JS-rendered pages)
        try:
            html = _web_browser_playwright(url)
            method = "playwright"
        except ImportError:
            pass  # Playwright not installed, fall through to requests
        except Exception as e:
            print(f"  [WARN] Playwright failed for {url}: {e}, falling back to requests")

        # Fallback to requests
        if html is None:
            try:
                html = _web_browser_requests(url)
                method = "requests"
            except Exception as e:
                last_error = e
                # Don't retry on 4xx client errors - they won't succeed on retry
                try:
                    import requests as _requests  # type: ignore[import-untyped]

                    if (
                        isinstance(e, _requests.HTTPError)
                        and e.response is not None
                        and e.response.status_code < 500
                    ):
                        return f"[ERROR] Web browser failed: {e} (client error, not retrying)"
                except ImportError:
                    pass
                if attempt < max_retries - 1:
                    import time

                    wait = 2**attempt  # 1s, 2s
                    print(
                        f"  [RETRY] Web browser attempt {attempt + 1} failed: {e},"
                        f" retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue
                return f"[ERROR] Web browser failed after {max_retries} attempts: {last_error}"

        try:
            text = _extract_text_from_html(html, max_length=8000)
            output = [
                f"URL: {url}",
                f"Action: {action}",
                f"Method: {method}",
                "\nExtracted Text:\n",
                text,
            ]
            return "\n".join(output)
        except Exception as e:
            return f"[ERROR] Web browser text extraction failed: {str(e)}"

    return f"[ERROR] Web browser failed after {max_retries} attempts: {last_error}"


web_browser_tool = FunctionTool(web_browser)
