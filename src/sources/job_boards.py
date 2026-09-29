"""
Job boards: LinkedIn (public guest jobs API), Wellfound and Indeed (via search-engine results,
since both block direct scraping).
"""

import logging
import re
import urllib.parse
from typing import Any

from bs4 import BeautifulSoup

from src.providers.browser import BrowserProvider

logger = logging.getLogger("recruiting-platform.sources.job_boards")

LINKEDIN_GUEST_SEARCH = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
LINKEDIN_GUEST_POSTING = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"


def parse_linkedin_job_cards(page_html: str) -> list[dict[str, Any]]:
    """Parses the HTML cards returned by LinkedIn's guest job search endpoint."""
    soup = BeautifulSoup(page_html or "", "html.parser")
    jobs: list[dict[str, Any]] = []
    for card in soup.find_all("li"):
        title_el = card.find(class_="base-search-card__title")
        company_el = card.find(class_="base-search-card__subtitle")
        if not title_el or not company_el:
            continue
        link_el = card.find("a", class_="base-card__full-link") or card.find("a", href=True)
        company_link = company_el.find("a", href=True)
        location_el = card.find(class_="job-search-card__location")
        time_el = card.find("time")
        url = str(link_el.get("href", "")).split("?")[0] if link_el else ""
        jobs.append(
            {
                "title": title_el.get_text(strip=True),
                "company": company_el.get_text(strip=True),
                "company_linkedin_url": str(company_link.get("href", "")).split("?")[0] if company_link else None,
                "location": location_el.get_text(strip=True) if location_el else None,
                "url": url,
                "posted_at": str(time_el.get("datetime")) if time_el and time_el.get("datetime") else None,
                "source": "linkedin",
            }
        )
    return jobs


def search_linkedin_jobs(
    browser: BrowserProvider,
    keywords: str,
    location: str,
    max_results: int = 25,
    entry_level: bool = True,
) -> list[dict[str, Any]]:
    """Searches LinkedIn's public (logged-out) job search. Returns job dicts with title, company, location, url."""
    results: list[dict[str, Any]] = []
    for start in range(0, max_results, 25):
        params: dict[str, str] = {"keywords": keywords, "location": location, "start": str(start)}
        if entry_level:
            params["f_E"] = "1,2"  # internship, entry level
        url = f"{LINKEDIN_GUEST_SEARCH}?{urllib.parse.urlencode(params)}"
        try:
            page_html = browser.fetch_page_http(url)
        except Exception as e:
            logger.info(f"LinkedIn guest job search failed for '{keywords}' in {location}: {e}")
            break
        page_jobs = parse_linkedin_job_cards(page_html)
        if not page_jobs:
            break
        results.extend(page_jobs)
    return results[:max_results]


def fetch_linkedin_job_description(browser: BrowserProvider, job_url: str) -> str | None:
    """Fetches the full description of a LinkedIn posting via the guest posting endpoint."""
    match = re.search(r"(\d{8,})", job_url or "")
    if not match:
        return None
    try:
        page_html = browser.fetch_page_http(LINKEDIN_GUEST_POSTING.format(job_id=match.group(1)))
    except Exception as e:
        logger.debug(f"LinkedIn posting fetch failed for {job_url}: {e}")
        return None
    soup = BeautifulSoup(page_html, "html.parser")
    body = soup.find(class_="show-more-less-html__markup") or soup.find(class_="description__text")
    return body.get_text(separator="\n", strip=True)[:4000] if body else None


def company_name_matches(candidate: str, company_name: str) -> bool:
    def norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    a, b = norm(candidate), norm(company_name)
    return bool(a and b and (a == b or a.startswith(b) or b.startswith(a)))


# ---------------------------------------------------------------------------
# Wellfound (formerly AngelList Talent)
# ---------------------------------------------------------------------------

_WELLFOUND_TITLE = re.compile(
    r"^(?P<name>.+?)\s+(?:Careers, Funding, and Management Team|Careers|Jobs|- Wellfound|\|)", re.IGNORECASE
)


def search_wellfound_companies(browser: BrowserProvider, query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Finds companies on Wellfound matching a query through search results (Wellfound blocks direct scraping)."""
    results = browser.search_google(f"site:wellfound.com/company {query}", num_results=limit, include_blocked=True)
    companies: list[dict[str, Any]] = []
    for r in results:
        url = r.get("url", "")
        slug_match = re.search(r"wellfound\.com/company/([a-z0-9-]+)", url)
        if not slug_match:
            continue
        title = r.get("title", "")
        name_match = _WELLFOUND_TITLE.match(title)
        name = name_match.group("name").strip() if name_match else slug_match.group(1).replace("-", " ").title()
        name = re.sub(r"^Jobs at\s+", "", name, flags=re.IGNORECASE).strip(" -|")
        if name and all(name.lower() != c["name"].lower() for c in companies):
            companies.append(
                {"name": name, "wellfound_url": f"https://wellfound.com/company/{slug_match.group(1)}", "snippet": r.get("snippet", "")}
            )
    return companies


def search_wellfound_jobs(
    browser: BrowserProvider, company_name: str, limit: int = 5, role_terms: list[str] | None = None
) -> list[dict[str, Any]]:
    """Wellfound postings via search results. `role_terms` come from your configured roles (see role_search_terms)."""
    query = f'site:wellfound.com "{company_name}" jobs {" ".join(role_terms or [])}'.strip()
    results = browser.search_google(query, num_results=limit, include_blocked=True)
    jobs: list[dict[str, Any]] = []
    for r in results:
        url = r.get("url", "")
        if "wellfound.com" not in url:
            continue
        match = re.match(r"^(?P<title>.+?)\s+at\s+(?P<company>.+?)(?:\s+[•|\-].*)?$", r.get("title", ""))
        if not match or not company_name_matches(match.group("company"), company_name):
            continue
        jobs.append(
            {
                "title": match.group("title").strip(),
                "url": url,
                "location": None,
                "salary": None,
                "experience_years": None,
                "description": r.get("snippet") or None,
                "source": "wellfound",
            }
        )
    return jobs


# ---------------------------------------------------------------------------
# Indeed
# ---------------------------------------------------------------------------


def search_indeed_jobs(browser: BrowserProvider, company_name: str, role: str, limit: int = 5) -> list[dict[str, Any]]:
    """Finds Indeed postings for a company through search results ('Title - Company - City - Indeed.com')."""
    results = browser.search_google(f'site:indeed.com "{company_name}" {role}'.strip(), num_results=limit, include_blocked=True)
    jobs: list[dict[str, Any]] = []
    for r in results:
        url = r.get("url", "")
        if "indeed." not in url:
            continue
        parts = [p.strip() for p in re.split(r"\s+[-–|]\s+", r.get("title", "")) if p.strip()]
        if len(parts) < 2 or not company_name_matches(parts[1], company_name):
            continue
        jobs.append(
            {
                "title": parts[0],
                "url": url,
                "location": parts[2] if len(parts) > 2 and "indeed" not in parts[2].lower() else None,
                "salary": None,
                "experience_years": None,
                "description": r.get("snippet") or None,
                "source": "indeed",
            }
        )
    return jobs
