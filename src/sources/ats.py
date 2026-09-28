"""
Applicant Tracking System (ATS) job boards with public JSON APIs:
Greenhouse, Lever, Ashby, Workable and SmartRecruiters.
"""

import html as html_lib
import logging
import re
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

from src.intel.scoring import title_relevance
from src.providers.browser import BrowserProvider

logger = logging.getLogger("recruiting-platform.sources.ats")

ATS_PROVIDERS = ("greenhouse", "lever", "ashby", "workable", "smartrecruiters")

ATS_URL_PATTERNS: dict[str, list[str]] = {
    "greenhouse": [
        r"boards\.greenhouse\.io/embed/job_board(?:/js)?\?for=([a-zA-Z0-9_-]+)",
        r"(?:job-)?boards(?:-api)?\.greenhouse\.io/(?:v1/boards/)?([a-zA-Z0-9_-]+)",
    ],
    "lever": [r"jobs\.(?:eu\.)?lever\.co/([a-zA-Z0-9_-]+)"],
    "ashby": [r"jobs\.ashbyhq\.com/([a-zA-Z0-9_.%-]+)"],
    "workable": [r"apply\.workable\.com/([a-zA-Z0-9_-]+)"],
    "smartrecruiters": [r"(?:careers|jobs)\.smartrecruiters\.com/([a-zA-Z0-9_-]+)"],
}

_IGNORED_TOKENS = {"embed", "api", "v1", "jobs", "job", "careers", "static", "assets", "js", "widget"}

_LEGAL_SUFFIXES = re.compile(
    r"\b(inc|llc|ltd|limited|pvt|private|corp|corporation|co|gmbh|plc|technologies|technology|labs|hq|app|ai|india)\b"
)


@dataclass
class AtsJob:
    title: str
    url: str
    location: str | None = None
    description: str | None = None
    posted_at: str | None = None
    department: str | None = None
    source: str = "ats"


def html_to_text(raw: str | None, limit: int = 4000) -> str:
    if not raw:
        return ""
    text = BeautifulSoup(html_lib.unescape(raw), "html.parser").get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)[:limit]


def detect_ats_from_html(page_html: str) -> tuple[str, str] | None:
    """Finds an embedded/linked ATS board in a careers page and returns (provider, board_token)."""
    for provider, patterns in ATS_URL_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, page_html or ""):
                token = match.group(1).strip().strip("/")
                if token and token.lower() not in _IGNORED_TOKENS:
                    return provider, token
    return None


def candidate_tokens(company_name: str, domain: str | None) -> list[str]:
    """Likely board tokens: domain root and slugified name variants ('Razorpay Software' -> razorpay, razorpaysoftware)."""
    tokens: list[str] = []
    if domain:
        root = domain.lower().replace("www.", "").split(".")[0]
        tokens.append(root)
    lowered = (company_name or "").lower()
    stripped = _LEGAL_SUFFIXES.sub(" ", lowered)
    for variant in (lowered, stripped):
        words = re.findall(r"[a-z0-9]+", variant)
        if words:
            tokens.extend(["".join(words), "-".join(words)])
    unique: list[str] = []
    for token in tokens:
        if token and token not in unique:
            unique.append(token)
    return unique


def _greenhouse(browser: BrowserProvider, token: str) -> list[AtsJob]:
    data = browser.fetch_json(f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs", params={"content": "true"})
    jobs = []
    for j in data.get("jobs", []):
        departments = j.get("departments") or []
        jobs.append(
            AtsJob(
                title=str(j.get("title", "")),
                url=str(j.get("absolute_url", "")),
                location=(j.get("location") or {}).get("name"),
                description=html_to_text(j.get("content")),
                posted_at=j.get("updated_at"),
                department=departments[0].get("name") if departments else None,
                source="greenhouse",
            )
        )
    return jobs


def _lever(browser: BrowserProvider, token: str) -> list[AtsJob]:
    data = browser.fetch_json(f"https://api.lever.co/v0/postings/{token}", params={"mode": "json"})
    jobs = []
    for j in data if isinstance(data, list) else []:
        categories = j.get("categories") or {}
        jobs.append(
            AtsJob(
                title=str(j.get("text", "")),
                url=str(j.get("hostedUrl", "")),
                location=categories.get("location"),
                description=(j.get("descriptionPlain") or "")[:4000],
                posted_at=str(j.get("createdAt")) if j.get("createdAt") else None,
                department=categories.get("team"),
                source="lever",
            )
        )
    return jobs


def _ashby(browser: BrowserProvider, token: str) -> list[AtsJob]:
    data = browser.fetch_json(
        f"https://api.ashbyhq.com/posting-api/job-board/{token}", params={"includeCompensation": "true"}
    )
    jobs = []
    for j in data.get("jobs", []):
        jobs.append(
            AtsJob(
                title=str(j.get("title", "")),
                url=str(j.get("jobUrl") or j.get("applyUrl") or ""),
                location=j.get("location"),
                description=(j.get("descriptionPlain") or html_to_text(j.get("descriptionHtml")))[:4000],
                posted_at=j.get("publishedAt"),
                department=j.get("department"),
                source="ashby",
            )
        )
    return jobs


def _workable(browser: BrowserProvider, token: str) -> list[AtsJob]:
    data = browser.fetch_json(f"https://apply.workable.com/api/v1/widget/accounts/{token}")
    jobs = []
    for j in data.get("jobs", []):
        location = ", ".join(x for x in (j.get("city"), j.get("country")) if x) or None
        jobs.append(
            AtsJob(
                title=str(j.get("title", "")),
                url=str(j.get("url") or j.get("application_url") or ""),
                location=location,
                description=html_to_text(j.get("description")),
                posted_at=j.get("published_on"),
                department=j.get("department"),
                source="workable",
            )
        )
    return jobs


def _smartrecruiters(browser: BrowserProvider, token: str) -> list[AtsJob]:
    data = browser.fetch_json(f"https://api.smartrecruiters.com/v1/companies/{token}/postings")
    jobs = []
    for j in data.get("content", []):
        loc = j.get("location") or {}
        location = ", ".join(x for x in (loc.get("city"), loc.get("country")) if x) or None
        jobs.append(
            AtsJob(
                title=str(j.get("name", "")),
                url=f"https://jobs.smartrecruiters.com/{token}/{j.get('id')}",
                location=location,
                posted_at=j.get("releasedDate"),
                department=(j.get("department") or {}).get("label"),
                source="smartrecruiters",
            )
        )
    return jobs


_FETCHERS = {
    "greenhouse": _greenhouse,
    "lever": _lever,
    "ashby": _ashby,
    "workable": _workable,
    "smartrecruiters": _smartrecruiters,
}


def fetch_ats_jobs(browser: BrowserProvider, provider: str, token: str) -> list[AtsJob]:
    fetcher = _FETCHERS.get(provider)
    if not fetcher:
        raise ValueError(f"Unsupported ATS provider: {provider}")
    return [j for j in fetcher(browser, token) if j.title and j.url]


def probe_ats(
    browser: BrowserProvider,
    company_name: str,
    domain: str | None,
    careers_html: str | None = None,
    known: tuple[str, str] | None = None,
) -> tuple[str, str, list[AtsJob]] | None:
    """
    Resolves a company's ATS board: known board -> board linked from the careers page -> probing likely tokens.
    Returns (provider, token, jobs) or None.
    """
    attempts: list[tuple[str, str]] = []
    if known:
        attempts.append(known)
    if careers_html:
        detected = detect_ats_from_html(careers_html)
        if detected:
            attempts.append(detected)
    for token in candidate_tokens(company_name, domain)[:3]:
        for provider in ("greenhouse", "lever", "ashby", "workable"):
            attempts.append((provider, token))

    seen: set[tuple[str, str]] = set()
    for provider, token in attempts:
        if (provider, token) in seen:
            continue
        seen.add((provider, token))
        try:
            jobs = fetch_ats_jobs(browser, provider, token)
        except Exception as e:
            logger.debug(f"ATS probe {provider}/{token} failed: {e}")
            continue
        if jobs or (provider, token) == known:
            logger.info(f"Detected {provider} board '{token}' for {company_name} with {len(jobs)} postings.")
            return provider, token, jobs
    return None


def location_matches(location: str | None, geographies: list[str]) -> bool:
    if not location:
        return True
    lowered = location.lower()
    geos = [g.lower() for g in geographies]
    if not geos:
        return True
    if "remote" in geos and ("remote" in lowered or "anywhere" in lowered):
        return True
    india_cities = ("bengaluru", "bangalore", "mumbai", "delhi", "gurgaon", "gurugram", "noida", "hyderabad", "pune", "chennai", "kolkata", "ahmedabad")
    if "india" in geos and any(city in lowered for city in india_cities):
        return True
    return any(g in lowered for g in geos)


def filter_relevant_jobs(
    jobs: list[AtsJob], roles: list[str], geographies: list[str], experience_years_max: float, limit: int = 3
) -> list[AtsJob]:
    """Keeps engineering roles that fit your level and locations, most relevant first."""
    scored: list[tuple[float, AtsJob]] = []
    for job in jobs:
        relevance = title_relevance(job.title, roles, experience_years_max)
        if relevance < 0.5 or not location_matches(job.location, geographies):
            continue
        scored.append((relevance, job))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [job for _score, job in scored[:limit]]


def parse_experience_years(text: str | None) -> float | None:
    """Extracts a minimum-experience requirement: '2+ years', '3-5 years of experience' -> 2.0 / 3.0."""
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:\+|-\s*\d+)?\s*(?:years|yrs)", text.lower())
    return float(match.group(1)) if match else None


def ats_job_to_dict(job: AtsJob) -> dict[str, Any]:
    return {
        "title": job.title,
        "url": job.url,
        "location": job.location,
        "salary": None,
        "experience_years": parse_experience_years(job.description),
        "description": (job.description or "")[:3000] or None,
        "source": job.source,
        "posted_at": job.posted_at,
    }
