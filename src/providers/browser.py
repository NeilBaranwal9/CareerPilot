import json
import logging
import os
import sys
import time
import urllib.parse
from typing import Any

import httpx
import urllib3
from bs4 import BeautifulSoup

# Suppress InsecureRequestWarning when verifying=False is used
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("recruiting-platform.providers.browser")

# User agent to simulate real browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class FetchError(RuntimeError):
    """HTTP fetch failure with the status code (None for network errors)."""

    def __init__(self, url: str, status: int | None, message: str):
        super().__init__(message)
        self.url = url
        self.status = status


class BrowserProvider:
    """
    Scraping and search provider. Integrates both httpx (lightweight) and Playwright (JS dynamic).
    """

    def __init__(self) -> None:
        self.playwright_active = False
        self.run_failures: dict[str, int] = {}
        self.run_successes: dict[str, int] = {}
        self.search_provider = "duckduckgo"
        self.serper_key = ""
        self.brave_key = ""
        self.min_search_interval = 1.0
        self._last_search_at = 0.0
        self._search_cache: dict[tuple[str, int, bool], list[dict[str, str]]] = {}
        self._playwright_ready: bool | None = None
        self.playwright_message = ""
        self._playwright_warned = False
        self._ddg_blocked_until = 0.0
        self._load_domain_stats()

    def configure_search(
        self,
        provider: str = "auto",
        serper_key: str = "",
        brave_key: str = "",
        min_interval_seconds: float = 1.0,
    ) -> None:
        """
        Selects the web search backend. "auto" prefers Serper (Google results) or Brave when an API key
        is configured, and otherwise uses keyless DuckDuckGo with a Yahoo fallback.
        """
        self.serper_key = serper_key
        self.brave_key = brave_key
        self.min_search_interval = max(0.0, min_interval_seconds)
        provider = (provider or "auto").lower()
        if provider == "auto":
            provider = "serper" if serper_key else "brave" if brave_key else "duckduckgo"
        self.search_provider = provider

    @staticmethod
    def is_placeholder_result(result: dict[str, str]) -> bool:
        """True for the synthetic result emitted when every search engine failed."""
        return result.get("url", "").startswith("https://example.com/search")

    def fetch_json(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        timeout: float = 20.0,
    ) -> Any:
        """Performs an HTTP request against a JSON API (ATS boards, GitHub, Hunter, Apollo) and returns parsed JSON."""
        request_headers = {"User-Agent": HEADERS["User-Agent"], "Accept": "application/json"}
        if headers:
            request_headers.update(headers)
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.request(method, url, headers=request_headers, params=params, json=json_body)
        if response.status_code >= 400:
            raise RuntimeError(f"{method} {url} returned HTTP {response.status_code}: {response.text[:200]}")
        return response.json()

    def _throttle_search(self) -> None:
        wait = self.min_search_interval - (time.monotonic() - self._last_search_at)
        if wait > 0:
            time.sleep(wait)
        self._last_search_at = time.monotonic()

    def _search_serper(self, query: str, num_results: int) -> list[dict[str, str]]:
        data = self.fetch_json(
            "https://google.serper.dev/search",
            method="POST",
            headers={"X-API-KEY": self.serper_key, "Content-Type": "application/json"},
            json_body={"q": query, "num": max(num_results, 10)},
        )
        return [
            {"title": str(r.get("title", "")), "url": str(r.get("link", "")), "snippet": str(r.get("snippet", ""))}
            for r in data.get("organic", [])
            if r.get("link")
        ]

    def _search_brave(self, query: str, num_results: int) -> list[dict[str, str]]:
        data = self.fetch_json(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": self.brave_key},
            params={"q": query, "count": min(max(num_results, 10), 20)},
        )
        return [
            {"title": str(r.get("title", "")), "url": str(r.get("url", "")), "snippet": str(r.get("description", ""))}
            for r in data.get("web", {}).get("results", [])
            if r.get("url")
        ]

    def is_disabled(self, domain: str) -> bool:
        """True for a disabled domain or any of its subdomains (e.g. in.linkedin.com for linkedin.com)."""
        return any(domain == d or domain.endswith("." + d) for d in self.disabled_domains)

    def _get_domain(self, url: str) -> str:
        """Helper to extract clean domain name from URL."""
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""

    def _load_domain_stats(self) -> None:
        """Loads persistent domain success/failure statistics."""
        self.stats_file = "data/domain_stats.json"
        self.domain_failures = {}
        self.domain_successes = {}
        self.disabled_domains = set()

        # Seed with known bot-blocking/scraper-hostile platforms to avoid wasting time/CPU
        self.disabled_domains.update([
            "linkedin.com", "zoominfo.com", "rocketreach.co", "indeed.com",
            "tracxn.com", "stackshare.io", "glassdoor.com", "crunchbase.com"
        ])

        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file) as f:
                    data = json.load(f)
                    for domain, stats in data.items():
                        fails = stats.get("failures", 0)
                        wins = stats.get("successes", 0)
                        self.domain_failures[domain] = fails
                        self.domain_successes[domain] = wins
                        # If a domain has failed >= 5 times and has 0 successes, permanently blacklist it
                        if fails >= 5 and wins == 0:
                            self.disabled_domains.add(domain)
                            logger.info(f"Permanently blacklisted domain: {domain} (0 successes, {fails} failures)")
            except Exception as e:
                logger.warning(f"Failed to load domain stats: {e}")

    def _save_domain_stats(self) -> None:
        """Saves domain statistics to data/domain_stats.json."""
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
        
        # Build stats structure
        stats_data = {}
        all_domains = set(self.domain_failures.keys()) | set(self.domain_successes.keys())
        for domain in all_domains:
            stats_data[domain] = {
                "successes": self.domain_successes.get(domain, 0),
                "failures": self.domain_failures.get(domain, 0),
            }
            
        try:
            with open(self.stats_file, "w") as f:
                json.dump(stats_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save domain stats: {e}")

    def _record_success(self, domain: str) -> None:
        if not domain:
            return
        self.domain_successes[domain] = self.domain_successes.get(domain, 0) + 1
        self.run_successes[domain] = self.run_successes.get(domain, 0) + 1
        self._save_domain_stats()

    def _record_failure(self, domain: str) -> None:
        if not domain:
            return
        self.domain_failures[domain] = self.domain_failures.get(domain, 0) + 1
        self.run_failures[domain] = self.run_failures.get(domain, 0) + 1
        
        # Check if it failed multiple times in the current run (e.g. 3 times)
        total_run_fails = self.run_failures[domain]
        if total_run_fails >= 3:
            if domain not in self.disabled_domains:
                self.disabled_domains.add(domain)
                logger.warning(
                    f"Domain {domain} failed {total_run_fails} times in the current run. "
                    f"Disabling it for the remainder of this run."
                )
        self._save_domain_stats()

    def fetch_page_http(self, url: str) -> str:
        """
        Fetches page content using curl_cffi with Chrome 120 TLS impersonation, falling back to httpx.
        429/503 responses get one retry after a short backoff (Retry-After honoured up to 10s).
        Raises FetchError carrying the HTTP status (None for network errors).
        """
        status: int | None = None
        last_error: Exception | None = None
        for attempt in range(2):
            status, retry_after = None, None
            try:
                from curl_cffi import requests as curl_requests

                response = curl_requests.get(
                    url, headers=HEADERS, impersonate="chrome120", timeout=15, verify=False, allow_redirects=True
                )
                if response.status_code < 400:
                    return str(response.text)
                status, retry_after = response.status_code, response.headers.get("Retry-After")
            except Exception as ce:
                last_error = ce
                logger.debug(f"curl_cffi fetch failed for {url}: {ce}")

            if status is None or status >= 500:
                try:
                    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=15.0, verify=False) as client:
                        httpx_resp = client.get(url)
                    if httpx_resp.status_code < 400:
                        return httpx_resp.text
                    status, retry_after = httpx_resp.status_code, httpx_resp.headers.get("Retry-After")
                except Exception as e:
                    last_error = e

            if status in (429, 503) and attempt == 0:
                try:
                    wait = min(float(retry_after), 10.0) if retry_after else 3.0
                except ValueError:
                    wait = 3.0
                logger.info(f"HTTP {status} from {url}; retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            break
        reason = f"status {status}" if status else f"network error: {last_error}"
        logger.debug(f"HTTP fetch failed for {url}: {reason}")
        raise FetchError(url, status, f"HTTP request to {url} failed ({reason})")

    def playwright_status(self) -> tuple[bool, str]:
        """(ready, message). Checked once per process; used to skip Playwright when its browser is not installed."""
        if self._playwright_ready is None:
            fix = f'"{sys.executable}" -m playwright install chromium'
            try:
                from playwright.sync_api import sync_playwright

                with sync_playwright() as p:
                    path = p.chromium.executable_path
                self._playwright_ready = bool(path) and os.path.exists(path)
                self.playwright_message = (
                    "Playwright Chromium ready"
                    if self._playwright_ready
                    else f"Playwright browser is not installed (JS-heavy pages will be skipped). Fix: {fix}"
                )
            except Exception as e:
                self._playwright_ready = False
                self.playwright_message = f"Playwright unavailable ({e}). Fix: {fix}"
        return bool(self._playwright_ready), self.playwright_message

    def fetch_page_playwright(self, url: str) -> str:
        """
        Fetches page content using Playwright to render JavaScript.
        """
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.set_extra_http_headers(HEADERS)
                # Shorter timeout (10s) and using 'domcontentloaded' to drastically reduce resources/hangs
                response = page.goto(url, wait_until="domcontentloaded", timeout=10000)
                if response and response.status >= 400:
                    raise RuntimeError(f"Playwright received HTTP status {response.status}")
                content = page.content()
                return content
            finally:
                browser.close()

    def fetch_page(self, url: str, use_playwright: bool = False) -> str:
        """
        Fetches a page over HTTP; falls back to Playwright (if installed) for network errors and 5xx responses.
        404/410 are "page not found" (no penalty); 403/429/999 count towards the domain circuit breaker.
        """
        domain = self._get_domain(url)
        if self.is_disabled(domain):
            raise RuntimeError(f"Domain {domain} is disabled due to previous failures.")

        if use_playwright:
            ready, message = self.playwright_status()
            if not ready:
                raise RuntimeError(message)
            try:
                res = self.fetch_page_playwright(url)
                self._record_success(domain)
                return res
            except Exception as e:
                self._record_failure(domain)
                raise e

        # Known block-heavy domains where Playwright fallback is a waste of CPU/time
        block_heavy = {"zoominfo.com", "rocketreach.co", "tracxn.com", "stackshare.io", "linkedin.com", "indeed.com", "glassdoor.com", "crunchbase.com"}
        is_block_heavy = any(bh in domain for bh in block_heavy)

        try:
            res = self.fetch_page_http(url)
            self._record_success(domain)
            return res
        except FetchError as e:
            if e.status in (404, 410):
                raise  # a missing page says nothing about the site's health
            if is_block_heavy or e.status in (401, 403, 429, 999):
                self._record_failure(domain)
                logger.info(f"{url} blocked or rate limited (status {e.status}); not retrying.")
                raise RuntimeError(f"Failed to fetch page {url} (blocked/hostile, status {e.status})") from e

            ready, message = self.playwright_status()
            if not ready:
                if not self._playwright_warned:
                    logger.warning(message)
                    self._playwright_warned = True
                self._record_failure(domain)
                raise RuntimeError(f"Failed to fetch page {url} ({e}); Playwright fallback unavailable") from e

            logger.info(f"Direct HTTP fetch failed for {url} ({e}). Retrying with Playwright...")
            try:
                res = self.fetch_page_playwright(url)
                self._record_success(domain)
                return res
            except Exception as pe:
                self._record_failure(domain)
                logger.warning(f"Both HTTP and Playwright fetch failed for {url}: {pe}")
                raise RuntimeError(f"Failed to fetch page {url}") from pe

    def extract_text(self, html: str) -> str:
        """Extracts text content from HTML, removing scripts, styles, etc."""
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()

        # Get text
        text = soup.get_text(separator="\n")

        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        cleaned_text = "\n".join(chunk for chunk in chunks if chunk)

        return cleaned_text

    def search_google(self, query: str, num_results: int = 5, include_blocked: bool = False) -> list[dict[str, str]]:
        """
        Runs a search and parses organic results.
        Uses Serper/Brave when configured, otherwise DuckDuckGo with a Yahoo Search fallback.
        With include_blocked=True, results on scraper-hostile domains (e.g. linkedin.com) are kept so their
        titles/snippets can be mined without fetching the pages themselves.
        """
        cache_key = (query, num_results, include_blocked)
        if cache_key in self._search_cache:
            return list(self._search_cache[cache_key])

        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        self._throttle_search()

        results: list[dict[str, str]] = []

        # 0. Keyed search APIs (more reliable, Google-quality results)
        if self.search_provider in ("serper", "brave"):
            try:
                api_results = (
                    self._search_serper(query, num_results)
                    if self.search_provider == "serper"
                    else self._search_brave(query, num_results)
                )
                for r in api_results:
                    if not include_blocked and self.is_disabled(self._get_domain(r["url"])):
                        continue
                    results.append(r)
                    if len(results) >= num_results:
                        break
            except Exception as e:
                logger.warning(f"{self.search_provider} search failed for '{query}': {e}. Falling back to DuckDuckGo.")
            if results:
                self._search_cache[cache_key] = list(results)
                return results

        # 1. Try DuckDuckGo (skipped for 15 minutes after it serves a bot challenge)
        try:
            if time.monotonic() < self._ddg_blocked_until:
                raise RuntimeError("DuckDuckGo temporarily skipped after a bot challenge")
            html = self.fetch_page_http(url)
            if "anomaly" in html.lower() and "challenge" in html.lower():
                self._ddg_blocked_until = time.monotonic() + 900
                logger.warning("DuckDuckGo served a bot challenge; using Yahoo for the next 15 minutes.")
                raise RuntimeError("DuckDuckGo bot challenge")
            soup = BeautifulSoup(html, "html.parser")

            # DuckDuckGo HTML layout
            links = soup.find_all("a", class_="result__snippet")
            for link in links:
                parent = link.find_parent("div", class_="result__body")
                if not parent:
                    continue
                title_elem = parent.find("a", class_="result__url")
                if not title_elem:
                    continue

                # `result__a` holds the page title; `result__url` only shows the display URL.
                title_link = parent.find("a", class_="result__a")
                title = (title_link or title_elem).get_text().strip()
                href = title_elem.get("href", "").strip()

                # Clean DuckDuckGo redirect URLs if necessary
                if "uddg=" in href:
                    parts = href.split("uddg=")
                    if len(parts) > 1:
                        target = parts[1].split("&")[0]
                        href = urllib.parse.unquote(target).strip()

                if href.startswith("//"):
                    href = "https:" + href

                # Filter out disabled domains
                domain = self._get_domain(href)
                if not include_blocked and self.is_disabled(domain):
                    continue

                snippet = link.get_text().strip()
                results.append({"title": title, "url": href, "snippet": snippet})
                if len(results) >= num_results:
                    break
        except Exception as e:
            logger.info(f"DuckDuckGo search raised exception for query '{query}': {e}")

        # 2. If DuckDuckGo returned 0 results, fall back to Yahoo Search
        if not results:
            logger.info(f"DuckDuckGo returned 0 results for '{query}'. Retrying with Yahoo Search...")
            try:
                yahoo_url = f"https://search.yahoo.com/search?p={encoded_query}"
                import re

                html = self.fetch_page_http(yahoo_url)
                soup = BeautifulSoup(html, "html.parser")
                seen = set()

                for a in soup.find_all("a"):
                    href = a.get("href", "")
                    if "r.search.yahoo.com" in href and "/RU=" in href:
                        match = re.search(r"/RU=([^/]+)", href)
                        if match:
                            dest_url = urllib.parse.unquote(match.group(1))
                            if dest_url in seen or "yahoo.com" in dest_url or "yahoo.co" in dest_url:
                                continue
                            seen.add(dest_url)
                            
                            # Filter out disabled domains
                            domain = self._get_domain(dest_url)
                            if not include_blocked and self.is_disabled(domain):
                                continue

                            title = a.get_text().strip()
                            # Yahoo nests the <h3> title inside the link (next to site/breadcrumb spans)
                            h3 = a.find("h3") or a.find_parent("h3")
                            if h3:
                                title = h3.get_text().strip()

                            snippet = ""
                            parent = a.find_parent("div")
                            if parent:
                                sib = parent.find_next_sibling()
                                if sib:
                                    snippet = sib.get_text().strip()
                            results.append({"title": title or dest_url, "url": dest_url, "snippet": snippet})
                            if len(results) >= num_results:
                                break
            except Exception as ye:
                logger.error(f"Yahoo Search also failed for query '{query}': {ye}")

        if results:
            self._search_cache[cache_key] = list(results)

        # 3. If both failed, generate placeholder search results to ensure continuity
        if not results:
            logger.warning(f"All search engines failed for query '{query}'. Generating placeholder results.")
            results = [
                {
                    "title": f"Search result for {query}",
                    "url": f"https://example.com/search?q={encoded_query}",
                    "snippet": f"Mock result description for pipeline continuation query: {query}",
                }
            ]

        return results
