"""
Contact discovery from multiple sources, role classification and persona-aware ranking.

Sources:
- linkedin_search : public LinkedIn profile titles/snippets from search results (profiles are never scraped)
- team_page       : the company's own about/team/leadership/contact pages (plus any emails printed there)
- press           : press releases, interviews and news quoting founders/CTOs/engineering leaders
- github          : public members of the company's GitHub org, and commit emails for pattern inference
- hunter          : Hunter.io domain search (people, positions, emails, company email pattern)
- apollo          : Apollo.io people search (titles, LinkedIn, employment history)
"""

import logging
import re
from dataclasses import dataclass, field

from src.config import AppConfig
from src.db.models import Company
from src.intel.classify import classify_role, effective_headcount
from src.intel.learning import OutcomeStats, persona_multiplier
from src.pipeline.schemas import ContactListResponse
from src.providers.browser import BrowserProvider
from src.providers.llm import BaseLLMProvider
from src.sources.enrichment import ApolloClient, GitHubClient, HunterClient, PersonRecord
from src.utils.email_verifier import extract_emails_from_text, is_placeholder_name

logger = logging.getLogger("recruiting-platform.sources.contacts")

PERSONA_SEARCH_TERMS: dict[str, str] = {
    "engineering_manager": '"engineering manager"',
    "hiring_manager": '"hiring manager"',
    "recruiter": '(recruiter OR "talent acquisition")',
    "founder": "(founder OR co-founder)",
    "cto": '(CTO OR "chief technology officer")',
    "vp_engineering": '("head of engineering" OR "VP engineering" OR "director of engineering")',
    "tech_lead": '"tech lead"',
}

TEAM_PAGE_PATHS = ["/about", "/about-us", "/team", "/our-team", "/company", "/leadership", "/people", "/contact"]

SOURCE_CONFIDENCE = {
    "hunter": 0.8,
    "apollo": 0.8,
    "team_page": 0.75,
    "linkedin_search": 0.7,
    "github": 0.6,
    "press": 0.55,
    "web_search": 0.5,
    "provided": 0.95,
    "fallback": 0.2,
}


@dataclass
class ContactCandidate:
    name: str
    title: str
    source: str
    source_url: str | None = None
    linkedin_url: str | None = None
    github_url: str | None = None
    email: str | None = None
    email_confidence: float = 0.0
    background: str | None = None
    confidence: float = 0.5
    role_category: str = "other"
    seniority: str = "individual"
    rank_score: float = 0.0
    sources: list[str] = field(default_factory=list)


@dataclass
class ContactDiscoveryResult:
    candidates: list[ContactCandidate] = field(default_factory=list)
    email_samples: list[tuple[str, str]] = field(default_factory=list)
    observed_emails: list[str] = field(default_factory=list)
    email_pattern: str | None = None
    accept_all: bool | None = None
    context_text: str = ""


def looks_like_person_name(name: str) -> bool:
    tokens = name.strip().split()
    if not 2 <= len(tokens) <= 5 or len(name) > 60:
        return False
    if any(ch.isdigit() for ch in name) or "linkedin" in name.lower():
        return False
    return all(re.match(r"^[A-Za-zÀ-ÿ.'-]+$", t) for t in tokens) and not is_placeholder_name(name)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def parse_linkedin_result(result: dict[str, str], company_name: str) -> ContactCandidate | None:
    """Parses 'Priya Sharma - Engineering Manager - Razorpay | LinkedIn' style search results."""
    url = result.get("url", "")
    if "linkedin.com/in/" not in url:
        return None
    title = re.split(r"\s*\|\s*LinkedIn", result.get("title", ""), flags=re.IGNORECASE)[0]
    parts = [p.strip() for p in re.split(r"\s+[-–—]\s+", title) if p.strip()]
    if len(parts) < 2:
        return None
    name = parts[0]
    role = parts[1].split("|")[0].replace("...", "").strip()
    at_match = re.match(r"^(.+?)\s+(?:at\s+|@\s*)(.+)$", role)
    if at_match:
        role = at_match.group(1).strip()
    haystack = _norm(" ".join([title, result.get("snippet", "")]))
    if _norm(company_name) not in haystack or not looks_like_person_name(name):
        return None
    if _norm(role) == _norm(company_name):
        role = "Employee"
    snippet = result.get("snippet", "")
    return ContactCandidate(
        name=name,
        title=role,
        source="linkedin_search",
        source_url=url.split("?")[0],
        linkedin_url=url.split("?")[0],
        background=snippet[:300] or None,
    )


def _llm_extract(llm: BaseLLMProvider, company: Company, text: str, source: str) -> list[ContactCandidate]:
    prompt = (
        f"Identify real people who work at {company.name} (engineering managers, tech leads, recruiters, "
        f"founders, CTO, hiring managers) from the following text. Only include people explicitly named in the "
        f"text as working at {company.name}; include their exact role and a one-line background if present.\n\n{text}"
    )
    response = llm.generate_json(prompt, ContactListResponse)
    assert isinstance(response, ContactListResponse)
    return [
        ContactCandidate(
            name=c.name,
            title=c.role,
            source=source,
            linkedin_url=c.linkedin_url if c.linkedin_url and "linkedin.com/in/" in c.linkedin_url else None,
            background=c.background,
        )
        for c in response.contacts
        if c.name
    ]


def from_linkedin_search(browser: BrowserProvider, company: Company, personas: list[str]) -> list[ContactCandidate]:
    found: list[ContactCandidate] = []
    for persona in personas[:3]:
        terms = PERSONA_SEARCH_TERMS.get(persona)
        if not terms:
            continue
        query = f'site:linkedin.com/in "{company.name}" {terms}'
        for r in browser.search_google(query, num_results=6, include_blocked=True):
            cand = parse_linkedin_result(r, company.name)
            if cand:
                found.append(cand)
    return found


def from_team_pages(
    browser: BrowserProvider, llm: BaseLLMProvider, company: Company
) -> tuple[list[ContactCandidate], str, list[str]]:
    if not company.domain:
        return [], "", []
    text = ""
    emails: list[str] = []
    for path in TEAM_PAGE_PATHS:
        url = f"https://{company.domain}{path}"
        try:
            page = browser.fetch_page(url, use_playwright=False)
        except Exception:
            continue
        emails.extend(e for e in extract_emails_from_text(page, company.domain) if e not in emails)
        text += f"\n--- {url} ---\n" + browser.extract_text(page)[:2500]
        if len(text) > 7000:
            break
    if not text:
        return [], "", emails
    try:
        candidates = _llm_extract(llm, company, text, "team_page")
    except Exception as e:
        logger.warning(f"Team page contact extraction failed for {company.name}: {e}")
        candidates = []
    return candidates, text, emails


def from_press_and_web(
    browser: BrowserProvider, llm: BaseLLMProvider, company: Company
) -> tuple[list[ContactCandidate], str]:
    queries = [
        (f"'{company.name}' (CTO OR 'Engineering Manager' OR 'Tech Lead' OR 'Hiring Manager') linkedin contacts", "web_search"),
        (f'"{company.name}" founder OR CTO OR "head of engineering" interview OR announces OR said', "press"),
    ]
    candidates: list[ContactCandidate] = []
    all_text = ""
    for query, source in queries:
        text = ""
        for r in browser.search_google(query, num_results=3):
            if "example.com/search" in r.get("url", ""):
                continue
            text += f"\n--- {r.get('title', '')} ({r.get('url', '')}) ---\n{r.get('snippet', '')}\n"
            try:
                page = browser.fetch_page(r["url"], use_playwright=False)
                text += browser.extract_text(page)[:2000]
            except Exception as e:
                logger.debug(f"Error scraping {r.get('url')}: {e}")
        if not text.strip():
            continue
        all_text += text
        try:
            candidates.extend(_llm_extract(llm, company, text, source))
        except Exception as e:
            logger.warning(f"{source} contact extraction failed for {company.name}: {e}")
    return candidates, all_text


def _person_to_candidate(p: PersonRecord) -> ContactCandidate:
    return ContactCandidate(
        name=p.name,
        title=p.title or "Employee",
        source=p.source,
        linkedin_url=p.linkedin_url,
        github_url=p.github_url,
        email=p.email,
        email_confidence=p.email_confidence,
        background=p.background,
    )


def merge_contact_candidates(candidates: list[ContactCandidate]) -> list[ContactCandidate]:
    merged: dict[str, ContactCandidate] = {}
    for cand in candidates:
        key = _norm(cand.name)
        if not key:
            continue
        cand.confidence = max(cand.confidence, SOURCE_CONFIDENCE.get(cand.source, 0.5))
        if key not in merged:
            cand.sources = cand.sources or [cand.source]
            merged[key] = cand
            continue
        existing = merged[key]
        if cand.source not in existing.sources:
            existing.sources.append(cand.source)
            existing.confidence = min(0.98, existing.confidence + 0.1)
        if (not existing.title or existing.title == "Employee") and cand.title:
            existing.title = cand.title
        for attr in ("linkedin_url", "github_url", "source_url", "background"):
            if not getattr(existing, attr) and getattr(cand, attr):
                setattr(existing, attr, getattr(cand, attr))
        if cand.email and cand.email_confidence > existing.email_confidence:
            existing.email, existing.email_confidence = cand.email, cand.email_confidence
    for cand in merged.values():
        cand.role_category, cand.seniority = classify_role(cand.title)
    return list(merged.values())


def rank_contacts(
    candidates: list[ContactCandidate], company: Company, config: AppConfig, stats: OutcomeStats | None = None
) -> list[ContactCandidate]:
    """
    Orders contacts by who is most likely to reply and able to help:
    persona preference (size-aware in 'auto' mode, strict in 'ordered' mode) x source confidence x reachability,
    scaled by what has historically worked for you.
    """
    personas = config.contacts.personas
    for cand in candidates:
        if config.contacts.persona_strategy == "ordered":
            idx = personas.index(cand.role_category) if cand.role_category in personas else len(personas) + 2
            persona_weight = max(0.1, 1.0 - 0.12 * idx)
            if stats is not None:
                persona_weight *= stats.multiplier("persona", cand.role_category)
        else:
            headcount = effective_headcount(company.employee_count, company.funding_stage)
            persona_weight = persona_multiplier(cand.role_category, headcount, stats)
            persona_weight *= 1.1 if cand.role_category in personas else 0.7
        reach = 1.15 if cand.email else 1.0
        reach *= 1.05 if cand.linkedin_url else 1.0
        cand.rank_score = round(persona_weight * (0.5 + 0.5 * cand.confidence) * reach, 4)
    if config.contacts.persona_strategy == "ordered":
        # Strict persona order first; score only breaks ties within the same persona.
        def order(c: ContactCandidate) -> tuple[int, float]:
            idx = personas.index(c.role_category) if c.role_category in personas else len(personas)
            return idx, -c.rank_score

        return sorted(candidates, key=order)
    return sorted(candidates, key=lambda c: c.rank_score, reverse=True)


def discover_contacts(
    config: AppConfig, llm: BaseLLMProvider, browser: BrowserProvider, company: Company
) -> ContactDiscoveryResult:
    sources = config.contacts.sources
    personas = config.contacts.personas
    result = ContactDiscoveryResult()
    raw: list[ContactCandidate] = []
    context_parts: list[str] = []

    def guard(name: str) -> bool:
        return name in sources

    if guard("hunter") and company.domain and config.api_keys.get("hunter"):
        try:
            intel = HunterClient(browser, config.api_keys.get("hunter")).domain_search(company.domain)
            raw.extend(_person_to_candidate(p) for p in intel.people)
            result.email_samples.extend(intel.samples)
            result.email_pattern = intel.pattern or result.email_pattern
            result.accept_all = intel.accept_all
        except Exception as e:
            logger.warning(f"Hunter domain search failed for {company.domain}: {e}")

    if guard("apollo") and company.domain and config.api_keys.get("apollo"):
        try:
            people = ApolloClient(browser, config.api_keys.get("apollo")).search_people(company.domain, personas)
            raw.extend(_person_to_candidate(p) for p in people)
        except Exception as e:
            logger.warning(f"Apollo people search failed for {company.domain}: {e}")

    if guard("linkedin_search"):
        try:
            raw.extend(from_linkedin_search(browser, company, personas))
        except Exception as e:
            logger.warning(f"LinkedIn search contact discovery failed for {company.name}: {e}")

    if guard("team_page"):
        try:
            team, text, emails = from_team_pages(browser, llm, company)
            raw.extend(team)
            context_parts.append(text)
            result.observed_emails.extend(e for e in emails if e not in result.observed_emails)
        except Exception as e:
            logger.warning(f"Team page discovery failed for {company.name}: {e}")

    if guard("press") or guard("web_search"):
        try:
            press, text = from_press_and_web(browser, llm, company)
            raw.extend(press)
            context_parts.append(text)
        except Exception as e:
            logger.warning(f"Press/web contact discovery failed for {company.name}: {e}")

    if guard("github"):
        try:
            gh = GitHubClient(browser, config.api_keys.get("github"))
            org = company.github_org or gh.find_org(company.name, company.domain)
            if org:
                company.github_org = org
                raw.extend(_person_to_candidate(p) for p in gh.org_people(org))
                if company.domain:
                    result.email_samples.extend(gh.commit_email_samples(org, company.domain))
        except Exception as e:
            logger.info(f"GitHub contact discovery skipped for {company.name}: {e}")

    for _name, email in result.email_samples:
        if email not in result.observed_emails:
            result.observed_emails.append(email)

    merged = [c for c in merge_contact_candidates(raw) if looks_like_person_name(c.name) or c.source in ("hunter", "apollo")]
    # Attach emails observed for the same person (e.g. from GitHub commits or team pages).
    for cand in merged:
        if cand.email:
            continue
        for sample_name, sample_email in result.email_samples:
            if _norm(sample_name) == _norm(cand.name):
                cand.email, cand.email_confidence = sample_email, 0.85
                break
    result.candidates = merged
    result.context_text = "\n".join(p for p in context_parts if p)[:8000]
    return result
