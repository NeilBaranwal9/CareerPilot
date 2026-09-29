"""
Company discovery from multiple sources, driven by a DiscoverySpec (e.g. parsed from
"Find 200 fintech companies in India" or "Series A/B AI startups").

Sources:
- llm           : the LLM's own knowledge of companies matching the spec
- web_search    : "top X startups" list articles found via search, extracted with the LLM
- yc            : the open Y Combinator company directory (yc-oss API) filtered by sector/region/hiring
- linkedin_jobs : companies with live matching job postings on LinkedIn (strong hiring signal)
- wellfound     : Wellfound company pages found via search
- ats_boards    : companies you list explicitly by Greenhouse/Lever/Ashby board token
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.intel.classify import FUNDING_ORDER, SECTOR_KEYWORDS, classify_sector, keyword_in, normalize_sector
from src.pipeline.schemas import CompanyListResponse
from src.providers.browser import BrowserProvider
from src.providers.llm import BaseLLMProvider
from src.sources.job_boards import search_linkedin_jobs, search_wellfound_companies

logger = logging.getLogger("recruiting-platform.sources.companies")

YC_DATASET_URL = "https://yc-oss.github.io/api/companies/all.json"
YC_CACHE_PATH = os.path.join("data", "cache", "yc_companies.json")
YC_CACHE_TTL_SECONDS = 7 * 86400

AGGREGATOR_DOMAINS = {
    "linkedin.com", "crunchbase.com", "wikipedia.org", "glassdoor.com", "glassdoor.co.in", "tracxn.com",
    "wellfound.com", "angel.co", "ambitionbox.com", "facebook.com", "twitter.com", "x.com", "instagram.com",
    "youtube.com", "medium.com", "github.com", "g2.com", "zaubacorp.com", "tofler.in", "economictimes.indiatimes.com",
    "inc42.com", "yourstory.com", "techcrunch.com", "bloomberg.com", "naukri.com", "indeed.com", "ycombinator.com",
    "pitchbook.com", "cbinsights.com", "owler.com", "zoominfo.com", "apollo.io", "rocketreach.co", "startupindia.gov.in",
    "entrackr.com", "livemint.com", "moneycontrol.com", "forbes.com", "reuters.com", "producthunt.com", "example.com",
    "google.com", "bing.com", "duckduckgo.com", "yahoo.com", "reddit.com", "quora.com", "instahyre.com", "cutshort.io",
}

GEO_KEYWORDS = [
    "india", "bengaluru", "bangalore", "mumbai", "delhi", "new delhi", "gurgaon", "gurugram", "noida", "hyderabad",
    "pune", "chennai", "kolkata", "ahmedabad", "remote", "usa", "united states", "us", "uk", "united kingdom",
    "london", "singapore", "europe", "germany", "berlin", "canada", "dubai", "uae", "san francisco", "new york",
    "australia", "japan", "netherlands", "france", "israel",
]

_LEGAL_SUFFIX_RE = re.compile(
    r"[,.]?\s+(inc|llc|ltd|limited|pvt|private|corp|corporation|co|gmbh|plc)\.?$", re.IGNORECASE
)


@dataclass
class DiscoverySpec:
    query: str = ""
    sectors: list[str] = field(default_factory=list)
    geographies: list[str] = field(default_factory=list)
    funding_stages: list[str] = field(default_factory=list)
    min_employees: int | None = None
    max_employees: int | None = None
    count: int = 20
    keywords: list[str] = field(default_factory=list)
    role_titles: list[str] = field(default_factory=list)
    personas: list[str] = field(default_factory=list)
    # Soft preference used to steer searches/prompts (hard filtering uses `sectors`).
    preferred_sectors: list[str] = field(default_factory=list)
    # Discovery mode for a campaign (job_search | company_outreach | hybrid); "" = use discovery.mode from config.
    mode: str = ""

    def describe(self) -> str:
        parts = []
        if self.sectors:
            parts.append("sectors: " + ", ".join(self.sectors))
        elif self.preferred_sectors:
            parts.append("preferably in sectors: " + ", ".join(self.preferred_sectors))
        if self.funding_stages:
            parts.append("funding stages: " + ", ".join(self.funding_stages))
        if self.geographies:
            parts.append("locations: " + ", ".join(self.geographies))
        if self.min_employees or self.max_employees:
            parts.append(f"size: {self.min_employees or 1}-{self.max_employees or 'any'} employees")
        if self.keywords:
            parts.append("keywords: " + ", ".join(self.keywords))
        return "; ".join(parts) or "technology companies"

    def search_phrase(self) -> str:
        if self.query:
            cleaned = re.sub(r"^(find|get|list|discover|show)\s+(me\s+)?(\d+\s+)?", "", self.query.strip(), flags=re.I)
            # "..., use actual openings when available, otherwise ask ..." describes the outreach, not the companies.
            return _OUTREACH_CLAUSE.split(cleaned, maxsplit=1)[0].strip(" ,;.") or cleaned
        sector = " ".join((self.sectors or self.preferred_sectors)[:2]) or "tech"
        geo = self.geographies[0] if self.geographies else ""
        return f"{sector} startups {geo}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query, "sectors": self.sectors, "geographies": self.geographies,
            "funding_stages": self.funding_stages, "min_employees": self.min_employees,
            "max_employees": self.max_employees, "count": self.count, "keywords": self.keywords,
            "role_titles": self.role_titles, "personas": self.personas, "preferred_sectors": self.preferred_sectors,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoverySpec":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class CompanyCandidate:
    name: str
    domain: str | None = None
    industry: str | None = None
    sector: str | None = None
    description: str | None = None
    employee_count: int | None = None
    funding_stage: str | None = None
    location: str | None = None
    linkedin_url: str | None = None
    hiring: bool | None = None
    source: str = "llm"
    sources: list[str] = field(default_factory=list)
    jobs: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_company_name(name: str) -> str:
    cleaned = (name or "").strip()
    while True:  # strip stacked suffixes such as "Pvt Ltd" / "Private Limited"
        stripped = _LEGAL_SUFFIX_RE.sub("", cleaned).strip()
        if stripped == cleaned:
            break
        cleaned = stripped
    return re.sub(r"[^a-z0-9]", "", cleaned.lower())


def clean_domain(value: str | None) -> str | None:
    if not value:
        return None
    domain = value.strip().lower()
    domain = re.sub(r"^https?://", "", domain).split("/")[0].split("?")[0]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain if "." in domain else None


def is_aggregator(domain: str | None) -> bool:
    if not domain:
        return True
    return any(domain == agg or domain.endswith("." + agg) for agg in AGGREGATOR_DOMAINS)


def resolve_company_domain(browser: BrowserProvider, name: str) -> str | None:
    """Finds a company's official website domain via search results."""
    try:
        results = browser.search_google(f"{name} official website", num_results=6)
    except Exception:
        return None
    name_tokens = [t for t in re.findall(r"[a-z0-9]+", name.lower()) if len(t) > 2]
    for r in results:
        domain = clean_domain(r.get("url"))
        if not domain or is_aggregator(domain):
            continue
        root = domain.split(".")[0]
        if any(t in root for t in name_tokens) or normalize_company_name(name) in domain.replace(".", ""):
            return domain
    return None


_OUTREACH_CLAUSE = re.compile(
    r"[,;]?\s+(?:and\s+|then\s+)?(?:ask\b|use (?:actual|real|public) (?:job )?openings|otherwise\b)", re.IGNORECASE
)
_HYBRID_GOAL = re.compile(
    r"\bhybrid\b|(?:actual|real|public) (?:job )?openings? (?:when|where|if) (?:available|possible|they exist)|"
    r"(?:openings?|jobs?|postings?) (?:when|where|if) (?:available|possible).{0,80}\botherwise\b|"
    r"\botherwise\b.{0,60}\bask\b.{0,60}\b(?:intern\w*|opportunit\w*|openings?)",
    re.IGNORECASE,
)
_COMPANY_OUTREACH_GOAL = re.compile(
    r"\bask (?:\w+ ){0,3}(?:if|whether|about)\b.{0,60}\b(?:intern\w*|opportunit\w*|openings?|hiring)|"
    r"\b(?:speculative|cold) (?:outreach|inquir\w*|applications?)\b|"
    r"\b(?:even )?(?:if|when|where) (?:they have |there (?:are|is) )?no (?:public )?(?:openings?|jobs?|postings?)",
    re.IGNORECASE,
)


def infer_outreach_mode(goal: str) -> str:
    """
    Discovery mode implied by a goal, or "" when it doesn't say:
    "... ask if they have internship opportunities for me" -> company_outreach;
    "... use actual openings when available, otherwise ask companies about internships" -> hybrid.
    """
    if _HYBRID_GOAL.search(goal or ""):
        return "hybrid"
    if _COMPANY_OUTREACH_GOAL.search(goal or ""):
        return "company_outreach"
    return ""


def parse_spec_fallback(goal: str, default_count: int = 25) -> DiscoverySpec:
    """Rule-based parser for goals like 'Find 200 fintech companies in India' or 'Series A/B AI startups'."""
    lowered = goal.lower()
    spec = DiscoverySpec(query=goal.strip(), count=default_count)

    count_match = re.search(r"\b(\d{1,4})\s+(?:\w+\s+){0,3}(?:companies|startups|firms|orgs)", lowered)
    if count_match:
        spec.count = int(count_match.group(1))

    for sector in SECTOR_KEYWORDS:
        if keyword_in(lowered, sector) or (sector == "ai" and re.search(r"\bai\b|artificial intelligence", lowered)):
            spec.sectors.append(sector)
    if "trading" not in spec.sectors and re.search(r"trading|quant|hft", lowered):
        spec.sectors.append("trading")

    stage_match = re.search(r"series\s+([a-f](?:\s*(?:/|,|and|or|&)\s*[a-f])*)\b", lowered)
    if stage_match:
        for letter in re.findall(r"[a-f]", stage_match.group(1)):
            stage = f"series_{letter}" if letter in "abc" else "series_d_plus"
            if stage not in spec.funding_stages:
                spec.funding_stages.append(stage)
    if re.search(r"pre[- ]?seed", lowered):
        spec.funding_stages.append("pre_seed")
    elif re.search(r"\bseed\b", lowered):
        spec.funding_stages.append("seed")

    for geo in GEO_KEYWORDS:
        if keyword_in(lowered, geo) and geo != "us":
            spec.geographies.append(geo.title() if geo not in ("usa", "uk", "uae") else geo.upper())

    persona_map = {
        "engineering manager": "engineering_manager", "recruiter": "recruiter", "founder": "founder",
        "cto": "cto", "hiring manager": "hiring_manager", "tech lead": "tech_lead",
    }
    spec.personas = [v for k, v in persona_map.items() if k in lowered]

    size_match = re.search(r"(\d+)\s*[-–to]+\s*(\d+)\s*(?:employees|people)", lowered)
    if size_match:
        spec.min_employees, spec.max_employees = int(size_match.group(1)), int(size_match.group(2))
    spec.mode = infer_outreach_mode(goal)
    return spec


def merge_candidates(candidates: list[CompanyCandidate]) -> list[CompanyCandidate]:
    """Deduplicates by normalized name or domain, merging fields and recording every contributing source."""
    merged: list[CompanyCandidate] = []
    by_key: dict[str, CompanyCandidate] = {}
    for cand in candidates:
        if not cand.name or len(cand.name) > 80:
            continue
        cand.domain = clean_domain(cand.domain)
        if cand.domain and is_aggregator(cand.domain):
            cand.domain = None
        keys = [f"n:{normalize_company_name(cand.name)}"] + ([f"d:{cand.domain}"] if cand.domain else [])
        existing = next((by_key[k] for k in keys if k in by_key), None)
        if existing is None:
            cand.sources = cand.sources or [cand.source]
            merged.append(cand)
            for k in keys:
                by_key[k] = cand
            continue
        for attr in ("domain", "industry", "sector", "description", "employee_count", "funding_stage", "location", "linkedin_url"):
            if getattr(existing, attr) is None and getattr(cand, attr) is not None:
                setattr(existing, attr, getattr(cand, attr))
        existing.hiring = existing.hiring or cand.hiring
        existing.jobs.extend(j for j in cand.jobs if j not in existing.jobs)
        existing.extra.update({k: v for k, v in cand.extra.items() if k not in existing.extra})
        if cand.source not in existing.sources:
            existing.sources.append(cand.source)
        for k in keys:
            by_key[k] = existing
    return merged


def matches_spec(cand: CompanyCandidate, spec: DiscoverySpec) -> bool:
    """Cheap pre-filter before full research (research + fit scoring does the thorough check)."""
    text = " ".join(x for x in (cand.name, cand.industry, cand.sector, cand.description) if x)
    if spec.sectors:
        sector = normalize_sector(cand.sector) or normalize_sector(cand.industry)
        _primary, all_sectors = classify_sector(text)
        if sector not in spec.sectors and not set(all_sectors) & set(spec.sectors):
            # Unknown sector: keep it for research rather than discarding on missing data.
            if sector or all_sectors:
                return False
    if spec.funding_stages and cand.funding_stage and cand.funding_stage not in spec.funding_stages:
        return False
    if cand.employee_count is not None:
        if spec.max_employees and cand.employee_count > spec.max_employees * 1.5:
            return False
        if spec.min_employees and cand.employee_count < spec.min_employees * 0.5:
            return False
    return True


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def discover_from_llm(
    llm: BaseLLMProvider, spec: DiscoverySpec, exclude: list[str], count: int
) -> list[CompanyCandidate]:
    prompt = (
        f"List {count} real, currently operating companies matching this request: '{spec.search_phrase()}'.\n"
        f"Constraints: {spec.describe()}.\n"
        f"They must employ software engineers. Prefer companies likely to be hiring junior engineers.\n"
        f"DO NOT include any of these companies: {exclude[:150]}.\n"
        "For each company give its official website domain (not a LinkedIn/Crunchbase URL), approximate employee "
        "count, industry, canonical sector (fintech, trading, ai, healthtech, edtech, saas, ...), latest funding stage "
        "(seed, series_a, series_b, ...), a one-line description and headquarters location. Do not invent companies."
    )
    response = llm.generate_json(prompt, CompanyListResponse)
    assert isinstance(response, CompanyListResponse)
    return [
        CompanyCandidate(
            name=c.name,
            domain=c.domain,
            industry=c.industry,
            sector=normalize_sector(c.sector) or normalize_sector(c.industry),
            description=c.description,
            employee_count=c.employee_count,
            funding_stage=c.funding_stage,
            location=c.location,
            source="llm",
        )
        for c in response.companies
    ]


def discover_from_web(
    browser: BrowserProvider, llm: BaseLLMProvider, spec: DiscoverySpec, exclude: list[str], count: int
) -> list[CompanyCandidate]:
    year = datetime.now(UTC).year
    phrase = spec.search_phrase()
    queries = [f"top {phrase} list {year}", f"{phrase} funded startups {year}", f"best {phrase} to work for"]
    text = ""
    seen_urls: set[str] = set()
    for query in queries:
        for r in browser.search_google(query, num_results=4):
            url = r.get("url", "")
            if url in seen_urls or "example.com" in url:
                continue
            seen_urls.add(url)
            try:
                page = browser.fetch_page(url, use_playwright=False)
                text += f"\n--- {r.get('title', '')} ({url}) ---\n" + browser.extract_text(page)[:4000]
            except Exception as e:
                logger.debug(f"Skipping list page {url}: {e}")
            if len(text) > 14000:
                break
        if len(text) > 14000:
            break
    if not text:
        return []
    prompt = (
        f"From the following articles, extract up to {count} companies that match: '{phrase}' ({spec.describe()}).\n"
        f"Only include companies explicitly named in the text. Exclude: {exclude[:100]}.\n\n{text}"
    )
    response = llm.generate_json(prompt, CompanyListResponse)
    assert isinstance(response, CompanyListResponse)
    return [
        CompanyCandidate(
            name=c.name,
            domain=c.domain,
            industry=c.industry,
            sector=normalize_sector(c.sector) or normalize_sector(c.industry),
            description=c.description,
            employee_count=c.employee_count,
            funding_stage=c.funding_stage,
            location=c.location,
            source="web_search",
        )
        for c in response.companies
    ]


def _load_yc_dataset(browser: BrowserProvider) -> list[dict[str, Any]]:
    if os.path.exists(YC_CACHE_PATH) and time.time() - os.path.getmtime(YC_CACHE_PATH) < YC_CACHE_TTL_SECONDS:
        with open(YC_CACHE_PATH, encoding="utf-8") as f:
            cached: list[dict[str, Any]] = json.load(f)
            return cached
    data = browser.fetch_json(YC_DATASET_URL, timeout=60.0)
    companies = data if isinstance(data, list) else []
    os.makedirs(os.path.dirname(YC_CACHE_PATH), exist_ok=True)
    with open(YC_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(companies, f)
    return companies


def discover_from_yc(browser: BrowserProvider, spec: DiscoverySpec, exclude: set[str], count: int) -> list[CompanyCandidate]:
    companies = _load_yc_dataset(browser)
    geos = [g.lower() for g in spec.geographies if g.lower() != "remote"]
    scored: list[tuple[float, CompanyCandidate]] = []
    for c in companies:
        name = str(c.get("name") or "")
        if not name or normalize_company_name(name) in exclude:
            continue
        if str(c.get("status", "Active")).lower() not in ("active", "public"):
            continue
        text = " ".join(
            str(x) for x in (c.get("one_liner"), c.get("long_description"), c.get("industry"), c.get("subindustry"),
                             " ".join(c.get("tags") or []), " ".join(c.get("industries") or []))
            if x
        )
        primary, all_sectors = classify_sector(text)
        if spec.sectors and not set(all_sectors) & set(spec.sectors):
            continue
        locations = " ".join(str(x) for x in (c.get("all_locations"), " ".join(c.get("regions") or [])) if x).lower()
        if geos and not any(g in locations for g in geos):
            continue
        team_size = c.get("team_size")
        if isinstance(team_size, int):
            if spec.max_employees and team_size > spec.max_employees * 1.5:
                continue
            if spec.min_employees and team_size < spec.min_employees * 0.5:
                continue
        score = 1.0 + (1.0 if c.get("isHiring") else 0.0) + (0.5 if primary in spec.sectors else 0.0)
        cand = CompanyCandidate(
            name=name,
            domain=clean_domain(c.get("website")),
            industry=c.get("industry"),
            sector=primary if primary != "generic" else None,
            description=c.get("one_liner"),
            employee_count=team_size if isinstance(team_size, int) else None,
            location=c.get("all_locations"),
            hiring=bool(c.get("isHiring")),
            source="yc",
            extra={"yc_batch": c.get("batch"), "yc_url": c.get("url"), "yc_stage": c.get("stage")},
        )
        scored.append((score, cand))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [cand for _s, cand in scored[:count]]


def discover_from_linkedin_jobs(
    browser: BrowserProvider, spec: DiscoverySpec, roles: list[str], exclude: set[str], count: int
) -> list[CompanyCandidate]:
    """Companies currently posting matching roles on LinkedIn; their postings are kept for Stage 1."""
    locations = [g for g in spec.geographies if g.lower() != "remote"] or ["India"]
    sector_terms = (spec.sectors or spec.preferred_sectors)[:2] or [""]
    role_terms = (spec.role_titles or roles or ["Software Engineer"])[:2]
    by_company: dict[str, CompanyCandidate] = {}
    for location in locations[:2]:
        for sector in sector_terms:
            for role in role_terms:
                keywords = f"{role} {sector}".strip()
                for job in search_linkedin_jobs(browser, keywords, location, max_results=25):
                    key = normalize_company_name(job["company"])
                    if not key or key in exclude:
                        continue
                    cand = by_company.setdefault(
                        key,
                        CompanyCandidate(
                            name=job["company"],
                            linkedin_url=job.get("company_linkedin_url"),
                            location=job.get("location"),
                            hiring=True,
                            source="linkedin_jobs",
                        ),
                    )
                    if len(cand.jobs) < 3:
                        cand.jobs.append(job)
                if len(by_company) >= count:
                    break
    return list(by_company.values())[:count]


def discover_from_wellfound(browser: BrowserProvider, spec: DiscoverySpec, exclude: set[str], count: int) -> list[CompanyCandidate]:
    candidates = []
    for c in search_wellfound_companies(browser, spec.search_phrase(), limit=min(count, 20)):
        if normalize_company_name(c["name"]) in exclude:
            continue
        candidates.append(
            CompanyCandidate(
                name=c["name"],
                description=c.get("snippet"),
                source="wellfound",
                extra={"wellfound_url": c.get("wellfound_url")},
            )
        )
    return candidates


def discover_from_ats_boards(ats_boards: dict[str, list[str]], exclude: set[str]) -> list[CompanyCandidate]:
    candidates = []
    for provider, tokens in ats_boards.items():
        for token in tokens:
            if normalize_company_name(token) in exclude:
                continue
            candidates.append(
                CompanyCandidate(
                    name=token.replace("-", " ").title(),
                    source="ats_boards",
                    hiring=None,
                    extra={"ats_provider": provider, "ats_token": token},
                )
            )
    return candidates


def run_discovery(
    browser: BrowserProvider,
    llm: BaseLLMProvider,
    spec: DiscoverySpec,
    sources: list[str],
    exclude_names: list[str],
    roles: list[str],
    ats_boards: dict[str, list[str]] | None = None,
    allow_linkedin: bool = False,
) -> list[CompanyCandidate]:
    """Runs every enabled source, merges and pre-filters the candidates. Source failures are logged and skipped."""
    exclude_set = {normalize_company_name(n) for n in exclude_names}
    want = max(spec.count, 5)
    collected: list[CompanyCandidate] = []

    def attempt(name: str, fn: Any, *args: Any) -> None:
        if name not in sources:
            return
        try:
            found = fn(*args)
            logger.info(f"Discovery source '{name}' returned {len(found)} candidates.")
            collected.extend(found)
        except Exception as e:
            logger.warning(f"Discovery source '{name}' failed: {e}")

    attempt("ats_boards", discover_from_ats_boards, ats_boards or {}, exclude_set)
    attempt("yc", discover_from_yc, browser, spec, exclude_set, want)
    if allow_linkedin:
        attempt("linkedin_jobs", discover_from_linkedin_jobs, browser, spec, roles, exclude_set, want)
    elif "linkedin_jobs" in sources:
        logger.info("Skipping linkedin_jobs source (discovery.allow_linkedin is false).")
    attempt("wellfound", discover_from_wellfound, browser, spec, exclude_set, want)
    attempt("web_search", discover_from_web, browser, llm, spec, exclude_names, want)
    attempt("llm", discover_from_llm, llm, spec, exclude_names, min(want, 40))

    merged = [c for c in merge_candidates(collected) if normalize_company_name(c.name) not in exclude_set]
    filtered = [c for c in merged if matches_spec(c, spec)]
    # Prefer candidates confirmed by several sources and currently hiring.
    filtered.sort(key=lambda c: (len(c.sources), bool(c.hiring), c.funding_stage in FUNDING_ORDER), reverse=True)
    return filtered
