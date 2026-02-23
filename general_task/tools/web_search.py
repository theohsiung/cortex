"""Web search tool with fallback chain: DuckDuckGo → Tavily → graceful error."""

from __future__ import annotations

import os

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
    """Fetch page using requests (fallback for static pages).

    Raises requests.HTTPError on HTTP errors (caller decides whether to retry).
    """
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


def _format_results(query: str, results: list[dict], expand_top: int = 3) -> str:
    """Format search results into a readable string with optional page expansion."""
    output_lines = [f"Search results for: '{query}'\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        snippet = r.get("snippet", r.get("body", "No snippet"))
        url = r.get("url", r.get("href", ""))
        output_lines.append(f"[{i}] {title}")
        output_lines.append(f"    {snippet}")
        if url:
            output_lines.append(f"    URL: {url}")

        # Auto-expand top results with deeper page content
        if i <= expand_top and url:
            page_text = _fetch_page_text(url, max_length=5000)
            if page_text:
                output_lines.append("    --- Expanded Content ---")
                output_lines.append(f"    {page_text[:5000]}")
                output_lines.append("    --- End Expanded ---")

        output_lines.append("")

    return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# Search backends
# ---------------------------------------------------------------------------


def _search_ddg(query: str, max_results: int) -> list[dict] | None:
    """Search using DuckDuckGo. Returns list of results or None on failure."""
    try:
        from ddgs import DDGS
    except ImportError:
        return None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with DDGS(timeout=30) as ddgs:
                results = list(ddgs.text(query, max_results=max_results, backend="auto"))
            if results:
                # Normalize keys to {title, snippet, url}
                return [
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    }
                    for r in results
                ]
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                import time

                wait = 2**attempt
                print(f"  [DDG RETRY] attempt {attempt + 1} failed: {e}, retrying in {wait}s...")
                time.sleep(wait)
    return None


def _search_tavily(query: str, max_results: int) -> list[dict] | None:
    """Search using Tavily API. Returns list of results or None on failure."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        print("  [Tavily] TAVILY_API_KEY not set, skipping")
        return None

    print(f"  [Tavily] Searching: {query}")

    try:
        from tavily import TavilyClient  # type: ignore[import-untyped]
    except ImportError:
        # Fallback: call Tavily REST API directly
        print("  [Tavily] SDK not installed, using REST API fallback")
        try:
            import requests  # type: ignore[import-untyped]

            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": max_results,
                    "include_answer": False,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("results", [])
            if raw:
                print(f"  [Tavily] Got {len(raw)} results via REST API")
                return [
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("content", ""),
                        "url": r.get("url", ""),
                    }
                    for r in raw
                ]
        except Exception as e:
            print(f"  [Tavily] REST API failed: {e}")
        return None

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth="advanced", max_results=max_results)
        raw = response.get("results", [])
        if raw:
            print(f"  [Tavily] Got {len(raw)} results via SDK")
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("content", ""),
                    "url": r.get("url", ""),
                }
                for r in raw
            ]
    except Exception as e:
        print(f"  [Tavily] SDK search failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Main web_search function
# ---------------------------------------------------------------------------


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information.

    Fallback chain: DuckDuckGo → Tavily → graceful error message.
    Returns: Title + Snippet + URL for each result, with top results auto-expanded.
    """
    errors: list[str] = []

    # 1. Try DuckDuckGo
    print(f"  [web_search] Trying DuckDuckGo for: {query}")
    results = _search_ddg(query, max_results)
    if results:
        print(f"  [web_search] DuckDuckGo returned {len(results)} results")
        return _format_results(query, results)
    errors.append("DuckDuckGo: no results or unavailable")
    print("  [web_search] DuckDuckGo failed, trying Tavily...")

    # 2. Try Tavily
    results = _search_tavily(query, max_results)
    if results:
        print(f"  [web_search] Tavily returned {len(results)} results")
        return _format_results(query, results)
    if not os.environ.get("TAVILY_API_KEY"):
        errors.append("Tavily: TAVILY_API_KEY not set")
    else:
        errors.append("Tavily: no results or API error")

    # 3. All backends failed
    return (
        f"[ERROR] Web search failed for query: '{query}'\n"
        f"Tried: {'; '.join(errors)}.\n"
        "The search backends are currently unavailable. "
        "Try rephrasing the query or using a direct URL with"
        " the web_browser tool if you know the target site."
    )


web_search_tool = FunctionTool(web_search)
