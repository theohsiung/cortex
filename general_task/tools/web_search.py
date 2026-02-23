"""Web search tool using DuckDuckGo."""

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


def _fetch_page_text(url: str, max_length: int = 5000) -> str:
    """Fetch and extract text from a URL.

    Tries playwright (JS-rendered) first, falls back to requests.
    Reuses _extract_text_from_html for consistent text cleaning.
    """
    html = None

    # Try Playwright first for JS-rendered pages
    try:
        html = _web_browser_playwright(url, timeout_ms=10000)
    except (ImportError, Exception):
        pass

    # Fallback to requests
    if html is None:
        try:
            html = _web_browser_requests(url)
        except Exception:
            return ""

    try:
        return _extract_text_from_html(html, max_length=max_length)
    except Exception:
        return ""


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo.

    Returns: Title + Snippet for each result.
    Auto-expands the top 2 results by fetching their page content.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return "[ERROR] ddgs not installed. Run: pip install ddgs"

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            with DDGS(timeout=30) as ddgs:
                results = list(ddgs.text(query, max_results=max_results, backend="auto"))

            if not results:
                return f"[INFO] No results found for query: {query}"

            output_lines = [f"Search results for: '{query}'\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                snippet = r.get("body", "No snippet")
                url = r.get("href", "")
                output_lines.append(f"[{i}] {title}")
                output_lines.append(f"    {snippet}")
                if url:
                    output_lines.append(f"    URL: {url}")

                # Auto-expand top 3 results with deeper page content
                if i <= 3 and url:
                    page_text = _fetch_page_text(url, max_length=5000)
                    if page_text:
                        output_lines.append("    --- Expanded Content ---")
                        output_lines.append(f"    {page_text[:5000]}")
                        output_lines.append("    --- End Expanded ---")

                output_lines.append("")

            return "\n".join(output_lines)

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                import time

                wait = 2**attempt  # 1s, 2s
                print(
                    f"  [RETRY] Web search attempt {attempt + 1} failed: {e},"
                    f" retrying in {wait}s..."
                )
                time.sleep(wait)

    return f"[ERROR] Web search failed after {max_retries} attempts: {last_error}"


web_search_tool = FunctionTool(web_search)
