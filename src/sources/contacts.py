"""
Contact discovery from multiple sources (LinkedIn-free by default), role classification and persona-aware ranking.

Sources, in priority order:
- team_page   : team / leadership / about / people pages found by following the company homepage's own links
- blog        : engineering and product blogs (post authors and their roles)
- press       : press releases, interviews and conference-speaker pages that name company leaders
- github      : public members of the company's GitHub org, and commit emails for pattern inference
- theorg      : The Org public org-chart pages (via search results)
- crunchbase  : Crunchbase person profiles (via search results; the site itself blocks scrapers)
- wellfound   : Wellfound company team pages (via search results)
- hunter/apollo: optional APIs when keys are configured
- linkedin_search: only when `discovery.allow_linkedin` is true (LinkedIn blocks automated access)

Anti-hallucination: any person proposed by the LLM must appear by name in the fetched text, otherwise dropped.
Every contact keeps the URL where it was found and a source-based confidence.
"""

import logging
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

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
    "product_manager": '"product manager"',
}

# Config persona names -> role categories produced by classify_role
PERSONA_ALIASES: dict[str, str] = {
    "director_engineering": "vp_engineering",
    "head_of_engineering": "vp_engineering",
    "staff_engineer": "tech_lead",
    "talent_acquisition": "recruiter",
    "hr": "recruiter",
    "head_of_product": "product_lead",
}

TEAM_LINK_WORDS = ("team", "leadership", "about", "people", "management", "founders", "our-story", "company", "who-we-are")
BLOG_LINK_WORDS = ("blog", "engineering", "tech-blog", "insights", "stories")
TEAM_PAGE_PATHS = ["/about", "/about-us", "/team", "/our-team", "/leadership", "/company", "/people", "/contact"]

SOURCE_CONFIDENCE = {
    "provided": 0.95,
    "hunter": 0.8,
    "apollo": 0.8,
    "team_page": 0.8,
    "theorg": 0.7,
    "linkedin_search": 0.7,
    "blog": 0.65,
    "press": 0.6,
    "crunchbase": 0.6,
    "wellfound": 0.6,
    "github": 0.55,
    "web_search": 0.5,
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
    sources_tried: dict[str, int] = field(default_factory=dict)  # source -> contacts found


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def looks_like_person_name(name: str) -> bool:
    tokens = name.strip().split()
    if not 2 <= len(tokens) <= 5 or len(name) > 60:
        return False
    if any(ch.isdigit() for ch in name) or "linkedin" in name.lower():
        return False
    return all(re.match(r"^[A-Za-zÀ-ÿ.'-]+$", t) for t in tokens) and not is_placeholder_name(name)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _fold(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()


def name_in_text(name: str, text: str) -> bool:
    """True when every meaningful token of the name appears in the text (accent/case-insensitive)."""
    folded = _fold(text)
    tokens = [t for t in re.findall(r"[a-z]+", _fold(name)) if len(t) > 1]
    return bool(tokens) and all(re.search(rf"\b{re.escape(t)}\b", folded) for t in tokens)


def person_key(name: str) -> str:
    """First + last name, ignoring middle names/initials: 'Ramkumar M Venkatesan' == 'Ramkumar Venkatesan'."""
    tokens = [t for t in re.findall(r"[a-z]+", _fold(name)) if len(t) > 1]
    if len(tokens) >= 2:
        return str(tokens[0]) + str(tokens[-1])
    return "".join(tokens)


def normalize_personas(personas: list[str]) -> list[str]:
    out: list[str] = []
    for p in personas:
        mapped = PERSONA_ALIASES.get(p, p)
        if mapped not in out:
            out.append(mapped)
    return out


def _same_site(url: str, domain: str) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower().replace("www.", "")
    return host == domain or host.endswith("." + domain)


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


def parse_theorg_result(result: dict[str, str], company_name: str) -> ContactCandidate | None:
    """Parses The Org results: 'Harshil Mathur - CEO at Razorpay | The Org'."""
    url = result.get("url", "")
    if "theorg.com" not in url:
        return None
    title = re.split(r"\s*\|\s*The Org", result.get("title", ""), flags=re.IGNORECASE)[0]
    match = re.match(r"^(?P<name>[^-–—|]+?)\s+[-–—]\s+(?P<role>.+?)(?:\s+(?:at|@)\s+(?P<company>.+))?$", title)
    if not match:
        return None
    name, role = match.group("name").strip(), match.group("role").strip()
    company = match.group("company") or result.get("snippet", "")
    if _norm(company_name) not in _norm(company + title) or not looks_like_person_name(name):
        return None
    return ContactCandidate(name=name, title=role, source="theorg", source_url=url.split("?")[0],
                            background=(result.get("snippet") or "")[:300] or None)


def _llm_extract(
    llm: BaseLLMProvider, company: Company, chunks: list[tuple[str, str]], source: str, focus: str = ""
) -> list[ContactCandidate]:
    """
    Asks the (local) LLM for people named in the chunks, then keeps only names that literally appear in the text
    and records the URL of the chunk where each name was found.
    """
    chunks = [(url, text) for url, text in chunks if text.strip()]
    if not chunks:
        return []
    joined = "\n".join(f"\n--- SOURCE: {url} ---\n{text}" for url, text in chunks)[:11000]
    prompt = (
        f"List the real people who work at {company.name} and are named in the text below{focus}. For each, give "
        f"their exact name, their role/title as written, a LinkedIn URL only if one is printed in the text, and a "
        f"one-line background if present. Only include people explicitly described as working at {company.name}. "
        f"Do not guess or invent anyone.\n{joined}"
    )
    response = llm.generate_json(prompt, ContactListResponse)
    assert isinstance(response, ContactListResponse)
    found: list[ContactCandidate] = []
    for c in response.contacts:
        if not c.name or not name_in_text(c.name, joined):
            if c.name:
                logger.info(f"Dropped '{c.name}' for {company.name}: name not present in the {source} text")
            continue
        url = next((u for u, text in chunks if name_in_text(c.name, text)), chunks[0][0])
        linkedin = c.linkedin_url if c.linkedin_url and "linkedin.com/in/" in c.linkedin_url and c.linkedin_url in joined else None
        found.append(
            ContactCandidate(name=c.name.strip(), title=c.role or "Employee", source=source, source_url=url,
                             linkedin_url=linkedin, background=c.background)
        )
    return found


def _fetch_text(browser: BrowserProvider, url: str, limit: int = 3000) -> tuple[str, str]:
    """(raw_html, text) or ("", "") on failure."""
    try:
        html = browser.fetch_page(url, use_playwright=False)
    except Exception as e:
        logger.debug(f"Could not fetch {url}: {e}")
        return "", ""
    return html, browser.extract_text(html)[:limit]


def discover_site_links(browser: BrowserProvider, company: Company) -> tuple[list[str], list[str]]:
    """Team-like and blog-like links from the company homepage (same site only)."""
    if not company.domain:
        return [], []
    html, _text = _fetch_text(browser, f"https://{company.domain}")
    if not html:
        return [], []
    soup = BeautifulSoup(html, "html.parser")
    team: list[str] = []
    blogs: list[str] = []
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(f"https://{company.domain}/", str(a.get("href", "")))
        if not href.startswith("http") or not _same_site(href, company.domain):
            continue
        href = href.split("#")[0].rstrip("/")
        label = f"{urllib.parse.urlparse(href).path} {a.get_text(' ', strip=True)}".lower()
        host = urllib.parse.urlparse(href).netloc.lower()
        if any(w in label for w in TEAM_LINK_WORDS) and href not in team and "career" not in label:
            team.append(href)
        elif (any(w in label for w in BLOG_LINK_WORDS) or host.startswith(("blog.", "engineering.", "tech."))) and href not in blogs:
            blogs.append(href)
    return team[:5], blogs[:3]


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def from_team_pages(
    browser: BrowserProvider, llm: BaseLLMProvider, company: Company, links: list[str] | None = None
) -> tuple[list[ContactCandidate], str, list[str]]:
    if not company.domain:
        return [], "", []
    urls = list(links or []) or [f"https://{company.domain}{p}" for p in TEAM_PAGE_PATHS]
    chunks: list[tuple[str, str]] = []
    emails: list[str] = []
    for url in urls[:6]:
        html, text = _fetch_text(browser, url, 2800)
        if not html:
            continue
        emails.extend(e for e in extract_emails_from_text(html, company.domain) if e not in emails)
        chunks.append((url, text))
        if sum(len(t) for _u, t in chunks) > 9000:
            break
    if not chunks:
        return [], "", emails
    try:
        candidates = _llm_extract(llm, company, chunks, "team_page", " (founders, leadership, team members)")
    except Exception as e:
        logger.warning(f"Team page contact extraction failed for {company.name}: {e}")
        candidates = []
    return candidates, "\n".join(t for _u, t in chunks), emails


def from_blogs(
    browser: BrowserProvider, llm: BaseLLMProvider, company: Company, links: list[str] | None = None
) -> list[ContactCandidate]:
    """Engineering/product blog authors (index page plus up to two recent posts)."""
    if not company.domain:
        return []
    candidates_urls = list(links or []) + [
        f"https://{company.domain}/blog", f"https://engineering.{company.domain}", f"https://blog.{company.domain}",
        f"https://{company.domain}/engineering", f"https://tech.{company.domain}",
    ]
    chunks: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index_url in candidates_urls:
        if index_url in seen or len(chunks) >= 3:
            continue
        seen.add(index_url)
        html, text = _fetch_text(browser, index_url, 2500)
        if not html:
            continue
        chunks.append((index_url, text))
        soup = BeautifulSoup(html, "html.parser")
        posts = []
        for a in soup.find_all("a", href=True):
            href = urllib.parse.urljoin(index_url + "/", str(a.get("href", ""))).split("#")[0]
            path = urllib.parse.urlparse(href).path
            if _same_site(href, company.domain) and href not in seen and path.count("/") >= 2 and len(path) > 12:
                posts.append(href)
        for post in posts[:2]:
            seen.add(post)
            _html, post_text = _fetch_text(browser, post, 1800)
            if post_text:
                chunks.append((post, post_text))
        break  # one blog is enough
    if not chunks:
        return []
    try:
        return _llm_extract(llm, company, chunks, "blog", " as blog post authors (with their job titles)")
    except Exception as e:
        logger.warning(f"Blog contact extraction failed for {company.name}: {e}")
        return []


def _search_chunks(
    browser: BrowserProvider, query: str, fetch: int = 1, num: int = 4, snippets_only_site: bool = False
) -> list[tuple[str, str]]:
    """
    Search results as (url, text) chunks. For sites that block scrapers (Crunchbase, Wellfound) we keep their
    results but read only the title/snippet (never fetch the page).
    """
    chunks: list[tuple[str, str]] = []
    results = (
        browser.search_google(query, num_results=num, include_blocked=True)
        if snippets_only_site
        else browser.search_google(query, num_results=num)
    )
    for i, r in enumerate(results):
        url = r.get("url", "")
        if not url or "example.com/search" in url:
            continue
        text = f"{r.get('title', '')}\n{r.get('snippet', '')}"
        if "linkedin.com" in url:
            continue  # LinkedIn is never used as a source; profile URLs are only kept when printed elsewhere
        if i < fetch:
            _html, page_text = _fetch_text(browser, url, 2200)
            text += "\n" + page_text
        chunks.append((url, text))
    return chunks


def from_web_sources(
    browser: BrowserProvider, llm: BaseLLMProvider, company: Company, sources: list[str]
) -> tuple[list[ContactCandidate], str]:
    """Press, conference speakers, The Org, Crunchbase and Wellfound — one LLM extraction over all snippets."""
    name = company.name
    found: list[ContactCandidate] = []
    chunks: list[tuple[str, str]] = []
    if "press" in sources:
        chunks += _search_chunks(browser, f'"{name}" (founder OR CTO OR "head of engineering" OR "engineering manager" OR "head of product") interview OR announces OR said', fetch=2)
        chunks += _search_chunks(browser, f'"{name}" ("head of talent" OR "talent acquisition" OR recruiter OR "people team")', fetch=0)
        chunks += _search_chunks(browser, f'"{name}" speaker conference (engineering OR product OR fintech)', fetch=1)
    if "theorg" in sources:
        for r in browser.search_google(f'site:theorg.com "{name}"', num_results=8, include_blocked=True):
            cand = parse_theorg_result(r, name)
            if cand:
                found.append(cand)
            elif "theorg.com" in r.get("url", ""):
                chunks.append((r["url"], f"{r.get('title', '')}\n{r.get('snippet', '')}"))
    if "crunchbase" in sources:
        chunks += _search_chunks(browser, f'site:crunchbase.com/person "{name}"', fetch=0, num=6, snippets_only_site=True)
    if "wellfound" in sources:
        chunks += _search_chunks(
            browser, f'site:wellfound.com "{name}" founder OR team OR "head of"', fetch=0, num=5, snippets_only_site=True
        )
    if chunks:
        try:
            for cand in _llm_extract(llm, company, chunks, "press"):
                url = cand.source_url or ""
                cand.source = (
                    "crunchbase" if "crunchbase.com" in url else "wellfound" if "wellfound.com" in url
                    else "theorg" if "theorg.com" in url else "press"
                )
                found.append(cand)
        except Exception as e:
            logger.warning(f"Web-source contact extraction failed for {name}: {e}")
    return found, "\n".join(t for _u, t in chunks)


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


def _person_to_candidate(p: PersonRecord) -> ContactCandidate:
    return ContactCandidate(
        name=p.name,
        title=p.title or "Employee",
        source=p.source,
        source_url=p.github_url or p.linkedin_url,
        linkedin_url=p.linkedin_url,
        github_url=p.github_url,
        email=p.email,
        email_confidence=p.email_confidence,
        background=p.background,
    )


# ---------------------------------------------------------------------------
# Merge & rank
# ---------------------------------------------------------------------------


def merge_contact_candidates(candidates: list[ContactCandidate]) -> list[ContactCandidate]:
    merged: dict[str, ContactCandidate] = {}
    for cand in candidates:
        key = person_key(cand.name)
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


# Who to ask about internships in each target role family (company-level inquiries), most suitable first.
FAMILY_PERSONAS: dict[str, list[str]] = {
    "product": ["product_lead", "product_manager", "recruiter"],
    "business": ["product_lead", "product_manager", "recruiter"],
    "data": ["data_lead", "product_lead", "product_manager", "recruiter"],
    "qa": ["qa_lead", "engineering_manager", "recruiter"],
    "software": ["engineering_manager", "vp_engineering", "hiring_manager", "recruiter"],
    "ai_ml": ["data_lead", "engineering_manager", "vp_engineering", "recruiter"],
}
EARLY_STAGE_HEADCOUNT = 50


def personas_for_families(families: list[str], company: Company) -> list[str]:
    """
    Recipient preference for a company-level inquiry, derived from your target role families (in your order):
    product/data -> product lead / PM / data lead / recruiter, engineering -> EM / head of engineering / recruiter,
    QA -> QA lead / EM / recruiter; founders first at early-stage startups.
    """
    preferred: list[str] = []
    headcount = effective_headcount(company.employee_count, company.funding_stage)
    if (headcount is not None and headcount <= EARLY_STAGE_HEADCOUNT) or company.funding_stage in ("pre_seed", "seed"):
        preferred.append("founder")
    for family in families:
        for persona in FAMILY_PERSONAS.get(family, []):
            if persona not in preferred:
                preferred.append(persona)
    for persona in ("hiring_manager", "recruiter"):
        if persona not in preferred:
            preferred.append(persona)
    return preferred


def rank_contacts(
    candidates: list[ContactCandidate],
    company: Company,
    config: AppConfig,
    stats: OutcomeStats | None = None,
    preferred_personas: list[str] | None = None,
) -> list[ContactCandidate]:
    """
    Orders contacts by who is most likely to reply and able to help:
    persona preference (size-aware in 'auto' mode, strict in 'ordered' mode) x source confidence x reachability,
    scaled by what has historically worked for you. `preferred_personas` (company-level inquiries) boosts, in 'auto'
    mode, those of your configured personas that suit your target roles; 'ordered' keeps your explicit order.
    """
    personas = normalize_personas(config.contacts.personas)
    preferred = [p for p in normalize_personas(preferred_personas or []) if p in personas]
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
            if cand.role_category in preferred:
                persona_weight *= max(1.0, 1.4 - 0.08 * preferred.index(cand.role_category))
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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def discover_contacts(
    config: AppConfig, llm: BaseLLMProvider, browser: BrowserProvider, company: Company
) -> ContactDiscoveryResult:
    sources = config.contacts.sources
    personas = normalize_personas(config.contacts.personas)
    result = ContactDiscoveryResult()
    raw: list[ContactCandidate] = []
    context_parts: list[str] = []

    def add(source: str, found: list[ContactCandidate]) -> None:
        raw.extend(found)
        result.sources_tried[source] = result.sources_tried.get(source, 0) + len(found)

    if "hunter" in sources and company.domain and config.api_keys.get("hunter"):
        try:
            intel = HunterClient(browser, config.api_keys.get("hunter")).domain_search(company.domain)
            add("hunter", [_person_to_candidate(p) for p in intel.people])
            result.email_samples.extend(intel.samples)
            result.email_pattern = intel.pattern or result.email_pattern
            result.accept_all = intel.accept_all
        except Exception as e:
            logger.warning(f"Hunter domain search failed for {company.domain}: {e}")

    if "apollo" in sources and company.domain and config.api_keys.get("apollo"):
        try:
            add("apollo", [_person_to_candidate(p) for p in ApolloClient(browser, config.api_keys.get("apollo")).search_people(company.domain, personas)])
        except Exception as e:
            logger.warning(f"Apollo people search failed for {company.domain}: {e}")

    team_links: list[str] = []
    blog_links: list[str] = []
    if ("team_page" in sources or "blog" in sources) and company.domain:
        try:
            team_links, blog_links = discover_site_links(browser, company)
        except Exception as e:
            logger.debug(f"Homepage link discovery failed for {company.name}: {e}")

    if "team_page" in sources:
        try:
            team, text, emails = from_team_pages(browser, llm, company, team_links)
            add("team_page", team)
            context_parts.append(text)
            result.observed_emails.extend(e for e in emails if e not in result.observed_emails)
        except Exception as e:
            logger.warning(f"Team page discovery failed for {company.name}: {e}")

    if "blog" in sources:
        try:
            add("blog", from_blogs(browser, llm, company, blog_links))
        except Exception as e:
            logger.warning(f"Blog discovery failed for {company.name}: {e}")

    web_sources = [s for s in ("press", "theorg", "crunchbase", "wellfound") if s in sources]
    if web_sources:
        try:
            found, text = from_web_sources(browser, llm, company, web_sources)
            add("press+web", found)
            context_parts.append(text)
        except Exception as e:
            logger.warning(f"Press/web contact discovery failed for {company.name}: {e}")

    if "github" in sources:
        try:
            gh = GitHubClient(browser, config.api_keys.get("github"))
            org = company.github_org or gh.find_org(company.name, company.domain)
            if org:
                company.github_org = org
                add("github", [_person_to_candidate(p) for p in gh.org_people(org)])
                if company.domain:
                    result.email_samples.extend(gh.commit_email_samples(org, company.domain))
        except Exception as e:
            logger.info(f"GitHub contact discovery skipped for {company.name}: {e}")

    if "linkedin_search" in sources and config.discovery.allow_linkedin:
        try:
            add("linkedin_search", from_linkedin_search(browser, company, personas))
        except Exception as e:
            logger.warning(f"LinkedIn search contact discovery failed for {company.name}: {e}")

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
    logger.info(f"Contact sources for {company.name}: {result.sources_tried}")
    return result
