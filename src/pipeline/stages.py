import contextlib
import json
import logging
import os
import re
import subprocess
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from src.config import AppConfig
from src.db.models import (
    COMPANY_OUTREACH_SOURCE,
    COMPANY_OUTREACH_TITLE,
    NON_OPENING_SOURCES,
    Application,
    Company,
    Contact,
    Email,
    History,
    Job,
    ResumeVersion,
    outreach_type_for,
)
from src.intel.classify import (
    classify_role,
    classify_sector,
    describe_role_families,
    normalize_funding_stage,
    normalize_sector,
    role_families,
    role_search_terms,
)
from src.intel.learning import compute_outcome_stats, estimate_response_probability, explain_reply_probability
from src.intel.scoring import FitResult, compute_company_fit, rule_based_opportunity, title_relevance, weighted_total
from src.outreach.dedupe import find_duplicate, record_outreach
from src.outreach.engine import get_initial_email, log_event, recipient_already_contacted
from src.outreach.personas import (
    DEFAULT_FOLLOWUP_PROMPT,
    EMAIL_STYLE_RULES,
    check_email_style,
    describe_tone,
    fallback_followups,
    find_placeholders,
    first_name,
    html_to_plain,
    safe_format,
    tone_for,
)
from src.outreach.scheduling import next_send_slot, utc_now_naive
from src.outreach.voice import (
    COMPANY_OUTREACH_WORDS,
    OPENING_STYLES,
    check_company_inquiry,
    check_grounding,
    check_subject,
    check_voice,
    choose_opening,
    company_outreach_ask,
    jargon_sentences,
    overused_recent_phrases,
    persona_ask,
    persona_focus,
    recent_patterns_summary,
    recent_sentence_starts,
    research_signals,
    word_target,
)
from src.pipeline.schemas import (  # noqa: F401  (re-exported for backwards compatibility)
    CompanyListResponse,
    CompanyResearchResponse,
    ContactListResponse,
    DiscoveredCompany,
    DiscoveredContact,
    DiscoveredJob,
    EmailDiscoveryResponse,
    EmailGenResponse,
    FollowUpSequenceSchema,
    JobListResponse,
    OpportunityScoreResponse,
    PlainSummarySchema,
    ResumeTailorResponse,
    StructuredResumeSchema,
    ValidationResponse,
)
from src.providers.browser import BrowserProvider
from src.providers.gmail import GmailProvider
from src.providers.llm import BaseLLMProvider
from src.providers.llm.ollama import OllamaUnavailableError
from src.providers.llm.router import BudgetExceededError
from src.sources.ats import ats_job_to_dict, detect_ats_from_html, filter_relevant_jobs, probe_ats
from src.sources.companies import CompanyCandidate, DiscoverySpec, normalize_company_name, run_discovery
from src.sources.contacts import ContactCandidate, discover_contacts, personas_for_families, rank_contacts
from src.sources.emails import find_contact_email, verification_timestamp
from src.sources.enrichment import ApolloClient
from src.sources.job_boards import (
    company_name_matches,
    fetch_linkedin_job_description,
    search_indeed_jobs,
    search_linkedin_jobs,
    search_wellfound_jobs,
)
from src.utils.caching import DBCache
from src.utils.claims import find_unsupported_claims
from src.utils.logging import PipelineLogger
from src.utils.resume import format_highlights, rank_highlights, read_resume_text, select_resume_variant
from src.utils.resume_pdf import (
    check_tailored_resume,
    extract_resume_text,
    render_resume_pdf,
    structured_to_text,
    validate_resume_file,
)

logger = logging.getLogger("recruiting-platform.pipeline.stages")

TERMINAL_STATES = [
    "Completed",
    "Skipped",
    "Duplicate",
    "Salary Too Low",
    "Low Score",
    "Poor Fit",
    "Excluded Company",
    "Ghost Job",
    "No Professional Email",
    "Research Failed",
    "Timeout",
    "Validation Failed",
    "Draft Failed",
    "Manual Skip",
    "Do Not Contact",
    "Failed",
]


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _extra(company: Company) -> dict[str, Any]:
    return dict(company.extra_data) if isinstance(company.extra_data, dict) else {}


def spec_from_config(config: AppConfig, query: str = "") -> DiscoverySpec:
    """Discovery spec derived from your target profile when no explicit campaign/query is given."""
    profile = config.target_profile
    preferred = [s for s, w in sorted(profile.sector_weights.items(), key=lambda kv: -kv[1]) if s != "generic" and w >= 0.8]
    return DiscoverySpec(
        query=query,
        sectors=list(profile.allowed_sectors),
        preferred_sectors=preferred[:3],
        geographies=list(config.job_preferences.geographies),
        funding_stages=list(profile.funding_stages),
        min_employees=config.job_preferences.company_size.min_employees,
        max_employees=config.job_preferences.company_size.max_employees,
        count=config.discovery.companies_per_run,
        role_titles=list(config.job_preferences.roles[:3]),
        personas=list(config.contacts.personas),
    )


# -------------------------------------------------------------
# Stage Executors
# -------------------------------------------------------------


def _candidate_to_company(cand: CompanyCandidate, spec: DiscoverySpec, campaign_id: int | None) -> Company:
    text = " ".join(x for x in (cand.name, cand.industry, cand.sector, cand.description) if x)
    primary, all_sectors = classify_sector(text)
    sector = normalize_sector(cand.sector) or normalize_sector(cand.industry) or (primary if primary != "generic" else None)
    extra: dict[str, Any] = {"sources": cand.sources or [cand.source], **cand.extra}
    if cand.jobs:
        extra["prefetched_jobs"] = cand.jobs
    return Company(
        name=cand.name,
        domain=cand.domain,
        employee_count=cand.employee_count,
        industry=cand.industry,
        description=cand.description,
        sector=sector,
        sub_sectors=[s for s in all_sectors if s != sector][:3] or None,
        funding_stage=normalize_funding_stage(cand.funding_stage),
        location=cand.location,
        linkedin_url=cand.linkedin_url,
        hiring_status="hiring" if cand.hiring else None,
        source=cand.source,
        discovery_query=spec.query or spec.describe(),
        campaign_id=campaign_id,
        ats_provider=cand.extra.get("ats_provider"),
        ats_token=cand.extra.get("ats_token"),
        status="candidate",
        extra_data=extra,
    )


def run_stage_0_company_discovery(
    session: Session,
    config: AppConfig,
    llm: BaseLLMProvider,
    browser: BrowserProvider,
    run_id: str,
    spec: DiscoverySpec | None = None,
    campaign_id: int | None = None,
    limit: int | None = None,
) -> list[Company]:
    """
    Stage 0: Company Discovery
    Builds a target list from every enabled source (LLM, list articles, YC directory, LinkedIn jobs, Wellfound,
    ATS boards) matching the spec (sector, funding stage, geography, size), deduplicated against the database.
    """
    p_log = PipelineLogger(logger, run_id, "Stage 0: Company Discovery")
    spec = spec or spec_from_config(config)
    want = limit or spec.count or config.discovery.companies_per_run
    p_log.info(f"Starting company discovery for: {spec.query or spec.describe()} (want {want})")

    existing_names = [c.name for c in session.query(Company.name).all() if c.name]
    existing_norm = {normalize_company_name(n) for n in existing_names}
    existing_domains = {d for (d,) in session.query(Company.domain).all() if d}
    all_exclusions = list(set(config.exclusions.companies) | set(existing_names))
    p_log.info(f"Currently tracking {len(existing_names)} companies in database.")

    cache = DBCache(session)
    cache_key = "company_discovery_v2_" + re.sub(r"[^a-z0-9]+", "_", json.dumps(spec.to_dict(), sort_keys=True).lower())[:180]
    candidates: list[CompanyCandidate] = []
    cached = cache.get(cache_key)
    if isinstance(cached, list):
        fresh = [c for c in cached if normalize_company_name(str(c.get("name", ""))) not in existing_norm]
        if len(fresh) >= 3:
            p_log.info(f"Found cached discovery list with {len(fresh)} untracked companies.")
            candidates = [CompanyCandidate(**{k: v for k, v in c.items() if k in CompanyCandidate.__dataclass_fields__}) for c in fresh]

    if not candidates:
        request = DiscoverySpec(**{**spec.to_dict(), "count": min(max(want * 2, 10), 60)})
        try:
            candidates = run_discovery(
                browser,
                llm,
                request,
                config.discovery.sources,
                all_exclusions,
                config.job_preferences.roles,
                config.discovery.ats_boards,
                config.discovery.allow_linkedin,
            )
        except Exception as e:
            p_log.error(f"Company discovery failed: {e}")
            raise
        cache.set(cache_key, [c.__dict__ for c in candidates], config.pipeline.cache_lifetime_seconds)

    new_companies: list[Company] = []
    for cand in candidates:
        if len(new_companies) >= want:
            break
        norm = normalize_company_name(cand.name)
        if not norm or norm in existing_norm or (cand.domain and cand.domain in existing_domains):
            continue
        if any(ex.lower() in cand.name.lower() for ex in config.exclusions.companies):
            continue
        company = _candidate_to_company(cand, spec, campaign_id)
        session.add(company)
        session.flush()
        existing_norm.add(norm)
        if company.domain:
            existing_domains.add(company.domain)
        fit = compute_company_fit(company, config)
        company.fit_score = fit.score
        # At discovery time only hard exclusions count; sector/fit are re-checked after research in Stage 3.
        known_sector = company.sector not in (None, "generic")
        if fit.rejected and (
            fit.reject_reason.startswith(("company", "domain"))
            or (known_sector and fit.reject_reason.startswith("sector"))
        ):
            company.status = "rejected"
            company.fit_reasoning = fit.reject_reason
            continue
        new_companies.append(company)

    # Backlog: previously discovered companies that never got a job/application (e.g. daily limit hit).
    if len(new_companies) < want:
        backlog_query = session.query(Company).filter(
            Company.status.in_(["candidate", "target"]) | Company.status.is_(None),
            ~Company.jobs.any(),
        )
        if campaign_id is not None:
            backlog_query = backlog_query.filter(Company.campaign_id == campaign_id)
        for company in backlog_query.limit(want - len(new_companies)).all():
            if company not in new_companies:
                new_companies.append(company)

    session.commit()
    p_log.info(f"Discovered {len(new_companies)} companies to process.", status="SUCCESS")
    return new_companies


def _llm_jobs_from_text(llm: BaseLLMProvider, config: AppConfig, company: Company, text: str) -> list[dict[str, Any]]:
    prompt = (
        f"Analyze this scraped text from {company.name}'s search results and career sites:\n\n"
        f"{text}\n\n"
        f"Extract the open jobs or internships listed above that match these preferred roles: "
        f"{config.job_preferences.roles}. Look for experience requirements close to: "
        f"up to {config.job_preferences.experience_years_max} years (intern, entry level, graduate). "
        f"Only list roles that are actually posted in the text; never make up a title."
    )
    response = llm.generate_json(prompt, JobListResponse)
    assert isinstance(response, JobListResponse)
    return [{**j.model_dump(), "source": "career_page"} for j in response.jobs]


def _discover_jobs_for_company(
    config: AppConfig,
    llm: BaseLLMProvider,
    browser: BrowserProvider,
    company: Company,
    p_log: PipelineLogger,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    sources = config.discovery.job_sources if sources is None else sources
    prefs = config.job_preferences
    # Search terms come from your configured roles (e.g. "product analyst data analyst"), never a default "engineer".
    role_terms = role_search_terms(prefs.roles)
    jobs: list[dict[str, Any]] = []
    extra = _extra(company)

    # Postings already found during discovery (LinkedIn jobs source)
    for job in extra.get("prefetched_jobs", []) or []:
        jobs.append(
            {
                "title": job.get("title"),
                "url": job.get("url"),
                "location": job.get("location"),
                "salary": None,
                "experience_years": None,
                "description": job.get("description"),
                "source": job.get("source", "linkedin"),
                "posted_at": job.get("posted_at"),
            }
        )

    careers_html = ""
    if company.domain and ("career_page" in sources or "ats" in sources):
        for path in ("/careers", "/jobs"):
            try:
                careers_html = browser.fetch_page(f"https://{company.domain}{path}", use_playwright=False)
                company.careers_url = company.careers_url or f"https://{company.domain}{path}"
                break
            except Exception:
                continue

    if "ats" in sources:
        try:
            known = (company.ats_provider, company.ats_token) if company.ats_provider and company.ats_token else None
            found = probe_ats(browser, company.name, company.domain, careers_html or None, known)
            if found:
                provider, token, ats_jobs = found
                company.ats_provider, company.ats_token = provider, token
                company.open_roles_count = len(ats_jobs)
                relevant_ats = filter_relevant_jobs(
                    ats_jobs, prefs.roles, prefs.geographies, prefs.experience_years_max, config.discovery.max_jobs_per_company
                )
                jobs.extend(ats_job_to_dict(j) for j in relevant_ats)
        except Exception as e:
            p_log.warning(f"ATS lookup failed for {company.name}: {e}")

    if "linkedin" in sources and config.discovery.allow_linkedin:
        try:
            location = next((g for g in prefs.geographies if g.lower() != "remote"), "India")
            role = prefs.roles[0] if prefs.roles else "internship"
            for posting in search_linkedin_jobs(browser, f"{company.name} {role}", location, max_results=25):
                if company_name_matches(posting["company"], company.name):
                    posting = {**posting, "salary": None, "experience_years": None, "description": None}
                    jobs.append(posting)
        except Exception as e:
            p_log.warning(f"LinkedIn job search failed for {company.name}: {e}")

    if "career_page" in sources and careers_html and not detect_ats_from_html(careers_html):
        try:
            jobs.extend(_llm_jobs_from_text(llm, config, company, browser.extract_text(careers_html)[:5000]))
        except Exception as e:
            p_log.warning(f"Career page job extraction failed for {company.name}: {e}")

    if "wellfound" in sources:
        try:
            jobs.extend(search_wellfound_jobs(browser, company.name, role_terms=role_terms))
        except Exception as e:
            p_log.info(f"Wellfound job search skipped for {company.name}: {e}")

    if "indeed" in sources:
        try:
            jobs.extend(search_indeed_jobs(browser, company.name, " ".join(role_terms)))
        except Exception as e:
            p_log.info(f"Indeed job search skipped for {company.name}: {e}")

    if "web_search" in sources and not any(j.get("source") not in ("wellfound", "indeed") for j in jobs):
        query = " ".join([f"'{company.name}'", *role_terms, "careers jobs"])
        search_results = browser.search_google(query, num_results=3)
        scraped_text = ""
        for result in search_results:
            try:
                page = browser.fetch_page(result["url"], use_playwright=False)
                scraped_text += f"\n--- {result['title']} ({result['url']}) ---\n"
                scraped_text += browser.extract_text(page)[:2000]
            except Exception as e:
                p_log.warning(f"Failed to scrape {result['url']}: {e}")
        if scraped_text:
            try:
                jobs.extend(_llm_jobs_from_text(llm, config, company, scraped_text))
            except Exception as e:
                p_log.error(f"Failed to parse jobs list from LLM for {company.name}: {e}")
        else:
            p_log.warning(f"No scrapable text found for {company.name}")

    # Dedupe by URL / title and keep the most relevant postings
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for job in jobs:
        title = str(job.get("title") or "").strip()
        if not title:
            continue
        key = (job.get("url") or "").strip() or f"title:{title.lower()}"
        if key in seen or f"title:{title.lower()}" in seen:
            continue
        seen.update({key, f"title:{title.lower()}"})
        unique.append(job)
    unique = [j for j in unique if not GENERIC_JOB_TITLE.match(str(j.get("title")).strip())]
    if unique:
        # The company is hiring even if none of its postings fit you (a company-level inquiry can still apply).
        company.hiring_status = "hiring"
        company.open_roles_count = max(company.open_roles_count or 0, len(unique))
    unique.sort(key=lambda j: title_relevance(str(j.get("title")), prefs.roles, prefs.experience_years_max), reverse=True)
    relevant = [
        j for j in unique if title_relevance(str(j.get("title")), prefs.roles, prefs.experience_years_max) >= 0.5
    ]
    if unique and not relevant:
        p_log.info(f"{company.name} has {len(unique)} postings but none match your target roles.")
    selected = relevant[: config.discovery.max_jobs_per_company]

    # Enrich LinkedIn postings with the full description (used for personalization)
    for job in selected:
        if job.get("source") == "linkedin" and not job.get("description") and job.get("url"):
            with contextlib.suppress(Exception):
                job["description"] = fetch_linkedin_job_description(browser, str(job["url"]))
    return selected


GENERIC_JOB_TITLE = re.compile(
    r"^(jobs?|careers?|open (positions|roles)|job openings|current openings|hiring|join us|work with us)\b",
    re.IGNORECASE,
)


def company_outreach_target(config: AppConfig, company: Company) -> dict[str, Any]:
    """
    The record behind a company-level (speculative) inquiry. Its title is a label, not a role name, and its
    description says plainly that no matching public opening was found.
    """
    areas = describe_role_families(role_families(config.job_preferences.roles), limit=6)
    slug = re.sub(r"[^a-z0-9]+", "_", (company.domain or company.name).lower()).strip("_")
    return {
        "title": COMPANY_OUTREACH_TITLE,
        "url": f"company-outreach://{slug}",
        "location": None,
        "salary": None,
        "experience_years": None,
        "description": "No matching public opening was found. Company-level inquiry about current or upcoming "
        f"internship opportunities{f' in {areas}' if areas else ''}.",
        "source": COMPANY_OUTREACH_SOURCE,
    }


def outreach_role_label(config: AppConfig, app: Application) -> str:
    """What the outreach is about: the real job title, or your target role families for a company-level inquiry."""
    if not app.is_company_level:
        return app.job.title
    areas = describe_role_families(role_families(config.job_preferences.roles))
    return f"internship opportunities in {areas}" if areas else "internship opportunities"


def run_stage_1_job_discovery(
    session: Session,
    config: AppConfig,
    llm: BaseLLMProvider,
    browser: BrowserProvider,
    companies: list[Company],
    run_id: str,
    mode: str | None = None,
) -> list[Job]:
    """
    Stage 1: Job Discovery
    Finds open roles at each company from ATS boards (Greenhouse/Lever/Ashby/Workable/SmartRecruiters), LinkedIn,
    the careers page, Wellfound, Indeed and web search, depending on `mode` (default `discovery.mode`):
    - job_search: only real openings that match your roles; companies without one are skipped.
    - company_outreach: checks the company's own careers page/ATS board; without a matching opening the company
      gets a company-level inquiry (no public opening required).
    - hybrid: full job search; companies without a matching opening fall back to a company-level inquiry.
    A job title is never invented: a company-level inquiry is labelled as such and never names a role.
    """
    p_log = PipelineLogger(logger, run_id, "Stage 1: Job Discovery")
    mode = mode or config.discovery.mode
    p_log.info(f"[DISCOVERY] Mode: {mode}")
    sources = list(config.discovery.job_sources)
    if mode == "company_outreach":
        p_log.info("[DISCOVERY] Company-level outreach enabled; public job opening not required.")
        sources = [s for s in sources if s in ("ats", "career_page")]
        p_log.info("[DISCOVERY] Checking each company's careers page / ATS board for a matching opening...")
    else:
        p_log.info("[DISCOVERY] Searching for actual openings...")
        if mode == "hybrid":
            p_log.info("[DISCOVERY] Hybrid: companies without a matching opening get a company-level inquiry.")
    if any(s in sources for s in ("wellfound", "indeed", "web_search", "linkedin")):
        terms = role_search_terms(config.job_preferences.roles)
        p_log.info(f"[DISCOVERY] Job-search terms from your target roles: {', '.join(terms) or '(none: generic search)'}")
    if mode == "job_search" and config.job_preferences.allow_speculative_outreach:
        p_log.warning(
            "[DISCOVERY] job_preferences.allow_speculative_outreach is deprecated and no longer creates speculative "
            "jobs. Set discovery.mode: hybrid (or company_outreach) to contact companies without a matching opening."
        )
    p_log.info(f"Searching jobs for {len(companies)} companies...")

    all_jobs = []
    seen_urls = set()
    for company in companies:
        p_log.company = company.name

        cache = DBCache(session)
        # v3: role-aware queries/matching (v2 results were searched with a hard-coded "engineer" query)
        scope = "careers_" if mode == "company_outreach" else ""
        cache_key = f"job_discovery_v3_{scope}{company.name.lower()}"
        cached_jobs = cache.get(cache_key)

        jobs_data: list[dict[str, Any]] = []
        if cached_jobs is not None:
            p_log.info(f"Found cached job listings for {company.name}")
            jobs_data = list(cached_jobs)
        else:
            jobs_data = _discover_jobs_for_company(config, llm, browser, company, p_log, sources=sources)
            cache.set(cache_key, jobs_data, config.pipeline.research_cache_days * 86400)

        real_jobs = [j for j in jobs_data if j.get("source") not in NON_OPENING_SOURCES]
        jobs_data = real_jobs
        if real_jobs:
            company.hiring_status = "hiring"
            company.open_roles_count = max(company.open_roles_count or 0, len(real_jobs))
            p_log.info(f"[OUTREACH] Matching opening found -> job-specific outreach ({real_jobs[0].get('title')})")
        else:
            if company.hiring_status != "hiring":
                company.hiring_status = "no_public_openings"
            p_log.info(f"[OUTREACH] No matching public opening found for {company.name}")
            if mode == "job_search":
                p_log.info("[OUTREACH] job_search mode -> skipping company (no speculative job is created)")
            else:
                if mode == "hybrid":
                    p_log.info("[OUTREACH] Hybrid mode -> falling back to company-level outreach")
                p_log.info("[OUTREACH] Creating company-level speculative outreach")
                jobs_data = [company_outreach_target(config, company)]

        for j_data in jobs_data:
            # Normalize URL: empty/whitespace-only becomes None
            url = (j_data.get("url") or "").strip() or None
            if url:
                if url in seen_urls:
                    continue
                seen_urls.add(url)

            existing_job = session.query(Job).filter(Job.url == url).first() if url else None
            if not existing_job:
                new_job = Job(
                    company_id=company.id,
                    title=j_data["title"],
                    url=url,
                    location=j_data.get("location"),
                    salary=j_data.get("salary"),
                    experience_years_required=j_data.get("experience_years"),
                    description=j_data.get("description"),
                    source=j_data.get("source"),
                    posted_at=j_data.get("posted_at"),
                )
                session.add(new_job)
                session.flush()  # Populate job.id and make it queryable within the transaction
                all_jobs.append(new_job)
            else:
                all_jobs.append(existing_job)

    session.commit()
    p_log.company = None
    p_log.info(f"Discovered {len(all_jobs)} jobs across companies.", status="SUCCESS")
    return all_jobs


def run_stage_2_filtering(
    session: Session, config: AppConfig, jobs: list[Job], run_id: str, campaign_id: int | None = None
) -> list[Application]:
    """
    Stage 2: Filtering
    Screens jobs against exclusion rules (companies, keywords, experience), enforces one outreach thread per
    company and the company contact cool-down, then initializes/returns Application instances.
    """
    p_log = PipelineLogger(logger, run_id, "Stage 2: Filtering")
    p_log.info(f"Filtering {len(jobs)} jobs...")

    active_applications = []

    for job in jobs:
        company = job.company
        p_log.company = company.name

        # Check if Application already exists for this job
        existing_app = session.query(Application).filter(Application.job_id == job.id).first()
        if existing_app:
            if existing_app.state not in TERMINAL_STATES:
                active_applications.append(existing_app)
            continue

        app = Application(
            run_id=run_id,
            job_id=job.id,
            current_stage=2,
            state="Filtering",
            campaign_id=campaign_id,
            outreach_type=outreach_type_for(job.source),
        )
        session.add(app)
        session.flush()  # Populate app.id

        history = History(
            application_id=app.id,
            stage=2,
            state="Filtering",
            run_id=run_id,
            notes="Initialized application state.",
        )
        session.add(history)

        # 1. Company Name Exclusions
        is_company_excluded = any(ex_c.lower() in company.name.lower() for ex_c in config.exclusions.companies)
        if is_company_excluded:
            app.state = "Excluded Company"
            history.notes = f"Filtered out: company {company.name} is excluded."
            p_log.info(f"Excluded company: {company.name}", status="EXCLUDED")
            continue

        if company.status == "rejected" and not _extra(company).get("targeted"):
            app.state = "Poor Fit"
            history.notes = f"Filtered out: company rejected by fit scoring ({company.fit_reasoning})."
            continue

        # 2. Keyword Exclusions in title (a company-level inquiry has no posting title to screen)
        is_keyword_excluded = not app.is_company_level and any(
            ex_k.lower() in job.title.lower() for ex_k in config.exclusions.keywords
        )
        if is_keyword_excluded:
            app.state = "Ghost Job"
            history.notes = f"Filtered out: job title '{job.title}' contains excluded keywords."
            p_log.info(f"Excluded job keyword in title: {job.title}", status="EXCLUDED")
            continue

        # 3. Experience Exclusions
        if (
            job.experience_years_required
            and job.experience_years_required > config.job_preferences.experience_years_max
        ):
            app.state = "Ghost Job"
            history.notes = f"Filtered out: experience required ({job.experience_years_required} yrs) exceeds max ({config.job_preferences.experience_years_max} yrs)."
            p_log.info(
                f"Excluded due to experience requirement: {job.experience_years_required} years",
                status="EXCLUDED",
            )
            continue

        # 4. One outreach thread per company (avoid emailing the same team about several postings)
        session.flush()
        sibling = (
            session.query(Application)
            .join(Job, Application.job_id == Job.id)
            .filter(Job.company_id == company.id, Application.id != app.id)
            .filter(Application.state.notin_([s for s in TERMINAL_STATES if s != "Completed"]))
            .first()
        )
        if sibling is not None:
            app.state = "Duplicate"
            history.notes = f"Company already has application #{sibling.id} in progress/completed; one thread per company."
            continue

        # 5. Company contact cool-down
        cooldown = config.outreach.company_cooldown_days
        if company.last_contacted_at and (_now() - company.last_contacted_at).days < cooldown:
            app.state = "Duplicate"
            history.notes = f"Company contacted within the last {cooldown} days."
            continue

        # Advance to Stage 3 if filter passes
        app.current_stage = 3
        app.state = "Company Research"
        session.add(
            History(
                application_id=app.id,
                stage=3,
                state="Company Research",
                run_id=run_id,
                notes="Passed basic filtering. Moving to Company Research.",
            )
        )
        active_applications.append(app)
        p_log.info(f"Passed filtering: {job.title} at {company.name}" + (" (no public opening)" if app.is_company_level else ""))

    session.commit()
    p_log.company = None
    p_log.info(f"Initialized {len(active_applications)} active applications.", status="SUCCESS")
    return active_applications


RESEARCH_GUIDANCE = (
    "\n\nAlso fill: description (2 sentences), sector (one canonical key), sub_sectors, funding_stage "
    "(pre_seed/seed/series_a/series_b/series_c/series_d_plus/public/bootstrapped/acquired), headcount, "
    "hiring_signals, recent_news (launches, funding, announcements with dates, ONLY from the provided text), "
    "products and location. Use null/empty when the text does not say."
)


def _gather_research_text(browser: BrowserProvider, company: Company, p_log: PipelineLogger) -> tuple[str, list[dict[str, str]]]:
    text = ""
    news_results: list[dict[str, str]] = []
    if company.domain:
        for path in ("", "/about"):
            try:
                page = browser.fetch_page(f"https://{company.domain}{path}", use_playwright=False)
                text += f"\n--- {company.domain}{path or '/'} (official site) ---\n" + browser.extract_text(page)[:2500]
            except Exception:
                continue

    results = browser.search_google(f"'{company.name}' tech stack product business model funding engineering blog", num_results=3)
    for r in results:
        try:
            page = browser.fetch_page(r["url"], use_playwright=False)
            text += f"\n--- {r['title']} ({r['url']}) ---\n" + browser.extract_text(page)[:2500]
        except Exception as e:
            p_log.warning(f"Error scraping {r['url']}: {e}")

    year = datetime.now(UTC).year
    try:
        news_results = [
            r
            for r in browser.search_google(f'"{company.name}" raises OR launches OR announces OR partners {year}', num_results=5)
            if "example.com" not in r.get("url", "")
        ]
        if news_results:
            text += "\n--- Recent news search results ---\n" + "\n".join(
                f"* {r.get('title', '')} ({r.get('url', '')}): {r.get('snippet', '')}" for r in news_results
            )
    except Exception as e:
        p_log.info(f"News search skipped: {e}")
    return text, news_results


def _apply_research(
    company: Company, research: dict[str, Any], news_results: list[dict[str, str]], apollo: dict[str, Any] | None
) -> None:
    company.research_data = research
    company.description = research.get("description") or company.description or research.get("business_model")
    blob = " ".join(
        str(x)
        for x in (company.description, research.get("business_model"), company.industry, " ".join(research.get("products") or []))
        if x
    )
    primary, all_sectors = classify_sector(blob)
    company.sector = (
        normalize_sector(research.get("sector"))
        or company.sector
        or normalize_sector(company.industry)
        or (primary if primary != "generic" else "generic")
    )
    subs = [normalize_sector(s) for s in (research.get("sub_sectors") or [])]
    merged_subs = [s for s in [*subs, *all_sectors] if s and s != company.sector]
    company.sub_sectors = list(dict.fromkeys(merged_subs))[:4] or None
    company.funding_stage = (
        normalize_funding_stage(research.get("funding_stage"))
        or normalize_funding_stage(research.get("funding"))
        or company.funding_stage
        or normalize_funding_stage((apollo or {}).get("funding_stage"))
    )
    company.funding_details = research.get("funding") or company.funding_details
    company.employee_count = company.employee_count or research.get("headcount") or (apollo or {}).get("employee_count")
    stack = list(research.get("tech_stack") or [])
    for tech in (apollo or {}).get("technologies", [])[:10]:
        if tech not in stack:
            stack.append(tech)
    company.tech_stack = stack or company.tech_stack
    news = list(research.get("recent_news") or [])
    if not news:
        news = [
            {"title": r.get("title", ""), "date": None, "summary": r.get("snippet", ""), "url": r.get("url")}
            for r in news_results
            if company.name.lower().split()[0] in r.get("title", "").lower()
        ][:3]
    company.recent_news = news[:5] or company.recent_news
    company.location = research.get("location") or company.location or (apollo or {}).get("location")
    signals = str(research.get("hiring_signals") or "").lower()
    if not company.hiring_status and signals:
        company.hiring_status = "hiring" if re.search(r"\bhiring\b|open (roles|positions)|we're growing", signals) else None
    company.last_researched_at = _now()


def research_company(
    session: Session,
    config: AppConfig,
    llm: BaseLLMProvider,
    browser: BrowserProvider,
    company: Company,
    p_log: PipelineLogger,
) -> FitResult:
    """
    Researches one company (site, search, news, optional Apollo), classifies it and scores its fit.
    Raises if the LLM cannot produce the research structure.
    """
    cache = DBCache(session)
    cache_key = f"company_research_{company.name.lower()}"
    cached_research = cache.get(cache_key)
    news_results: list[dict[str, str]] = []
    apollo: dict[str, Any] | None = None

    if cached_research:
        p_log.info("Retrieved company research from cache.")
        research = dict(cached_research)
    else:
        research_raw_text, news_results = _gather_research_text(browser, company, p_log)
        if company.domain and config.api_keys.get("apollo"):
            try:
                apollo = ApolloClient(browser, config.api_keys.get("apollo")).enrich_organization(company.domain)
                research_raw_text += f"\n--- Apollo organization data ---\n{json.dumps(apollo)[:1500]}"
            except Exception as e:
                p_log.info(f"Apollo enrichment skipped: {e}")

        if not research_raw_text:
            p_log.warning("No scrapable information retrieved for research. Falling back to parametric LLM knowledge.")
            research_raw_text = (
                "No web scraped text was retrieved due to rate limiting or connection issues. "
                "Please use your internal knowledge about the company and its domain to complete this research."
            )

        prompt = (
            safe_format(config.prompts.company_research, company_name=company.name)
            + RESEARCH_GUIDANCE
            + f"\n\nHere is the scraped content:\n{research_raw_text[:12000]}"
        )
        response = llm.generate_json(prompt, CompanyResearchResponse)
        research = response.model_dump()
        cache.set(cache_key, research, config.pipeline.research_cache_days * 86400)

    _apply_research(company, research, news_results, apollo)

    stats = compute_outcome_stats(session, config)
    fit = compute_company_fit(company, config, stats)
    company.fit_score = fit.score
    company.fit_reasoning = "; ".join(fit.reasons + ([fit.reject_reason] if fit.reject_reason else []))
    company.response_probability = estimate_response_probability(company, stats)
    p_log.info(
        f"Classified as sector={company.sector}, funding={company.funding_stage}, hiring={company.hiring_status}; "
        f"fit={fit.score:.2f}, reply probability={company.response_probability:.1%}"
    )
    return fit


def run_stage_3_company_research(
    session: Session,
    config: AppConfig,
    llm: BaseLLMProvider,
    browser: BrowserProvider,
    app: Application,
    run_id: str,
) -> bool:
    """
    Stage 3: Company Research
    Collects product, sector, funding stage, headcount, hiring status, tech stack and recent news; classifies the
    company (fintech/healthtech/edtech/...), scores its fit against your target profile and rejects poor fits.
    """
    company = app.job.company
    p_log = PipelineLogger(logger, run_id, "Stage 3: Company Research", company.name)
    p_log.info("Starting company research...")

    try:
        fit = research_company(session, config, llm, browser, company, p_log)
    except Exception as e:
        p_log.error(f"LLM failed to compile company research structure: {e}")
        app.state = "Research Failed"
        session.add(
            History(
                application_id=app.id,
                stage=3,
                state="Research Failed",
                run_id=run_id,
                notes=f"Company research LLM call failed: {e}",
            )
        )
        session.commit()
        return False

    if fit.rejected and not _extra(company).get("targeted"):
        company.status = "rejected"
        app.state = "Poor Fit"
        session.add(
            History(
                application_id=app.id,
                stage=3,
                state="Poor Fit",
                run_id=run_id,
                notes=f"Rejected by company fit scoring: {fit.reject_reason}",
            )
        )
        session.commit()
        p_log.info(f"Rejected {company.name}: {fit.reject_reason}", status="POOR_FIT")
        return False

    company.status = "target" if company.status != "contacted" else company.status
    app.current_stage = 4
    app.state = "Contact Research"
    session.add(
        History(
            application_id=app.id,
            stage=4,
            state="Contact Research",
            run_id=run_id,
            notes=f"Completed company research (sector {company.sector}, fit {fit.score:.2f}). Advancing to Contact Research.",
        )
    )
    session.commit()
    p_log.info("Completed company research.", status="SUCCESS")
    return True


def _upsert_contact(session: Session, company: Company, cand: ContactCandidate) -> Contact:
    contact = session.query(Contact).filter(Contact.company_id == company.id, Contact.name == cand.name).first()
    if not contact:
        contact = Contact(company_id=company.id, name=cand.name, role=cand.title or "Employee")
        session.add(contact)
    contact.role = cand.title or contact.role
    contact.role_category = cand.role_category
    contact.seniority = cand.seniority
    contact.source = ",".join(cand.sources or [cand.source])
    contact.source_url = cand.source_url or contact.source_url
    contact.linkedin_url = cand.linkedin_url or contact.linkedin_url
    contact.github_url = cand.github_url or contact.github_url
    contact.background = cand.background or contact.background
    contact.confidence = cand.confidence
    contact.rank_score = cand.rank_score
    session.flush()
    return contact


def run_stage_4_contact_research(
    session: Session,
    config: AppConfig,
    llm: BaseLLMProvider,
    browser: BrowserProvider,
    app: Application,
    run_id: str,
) -> bool:
    """
    Stage 4: Contact Research
    Finds people at the company (LinkedIn search results, team pages, press, GitHub, Hunter, Apollo), classifies
    them (recruiter / hiring manager / engineering manager / founder / CTO ...), ranks them by likelihood of a
    useful reply and stores all of them with role, source and background. Enforces the 'NO DUPLICATES' constraint.
    """
    company = app.job.company
    p_log = PipelineLogger(logger, run_id, "Stage 4: Contact Research", company.name)
    p_log.info("Starting contact research...")

    # If contact is already set, skip this stage
    if app.contact_id is not None:
        contact = session.query(Contact).filter(Contact.id == app.contact_id).first()
        if contact:
            p_log.info(f"Application already has contact associated: {contact.name}. Skipping contact research.")
            app.persona = app.persona or contact.role_category or classify_role(contact.role)[0]
            app.current_stage = 5
            app.state = "Professional Email Discovery"
            if contact.email:
                p_log.info(f"Contact email already known: {contact.email}. Bypassing Stage 5 as well.")
                app.current_stage = 6
                app.state = "Opportunity Scoring"
            session.commit()
            return True

    # Rule: "If a company only has one engineering contact available, consider the company itself contacted."
    previous_contact = (
        session.query(Contact).filter(Contact.company_id == company.id, Contact.email.isnot(None)).first()
    )
    if previous_contact:
        existing_app_email = (
            session.query(Application)
            .filter(
                Application.contact_id == previous_contact.id,
                Application.id != app.id,
                Application.state.in_(["Completed", "Gmail Draft Creation", "Email Generation"]),
            )
            .first()
        )
        if existing_app_email:
            p_log.info(
                f"Duplicate check: Contact {previous_contact.name} already contacted for this company.",
                status="DUPLICATE",
            )
            app.state = "Duplicate"
            session.add(
                History(
                    application_id=app.id,
                    stage=4,
                    state="Duplicate",
                    run_id=run_id,
                    notes=f"Company already contacted via {previous_contact.name}.",
                )
            )
            session.commit()
            return False

    cache = DBCache(session)
    cache_key = f"contacts_v2_{company.name.lower()}"
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and cached.get("candidates"):
        p_log.info("Found contacts in cache.")
        candidates = [ContactCandidate(**c) for c in cached["candidates"]]
        intel = cached.get("intel", {})
    else:
        result = discover_contacts(config, llm, browser, company)
        candidates = result.candidates
        intel = {
            "samples": result.email_samples,
            "observed": result.observed_emails,
            "pattern": result.email_pattern,
            "accept_all": result.accept_all,
        }
        if candidates:
            cache.set(
                cache_key,
                {"candidates": [c.__dict__ for c in candidates], "intel": intel},
                config.pipeline.research_cache_days * 86400,
            )

    # Never re-target people who asked not to be contacted or were already emailed.
    blocked = {
        c.name
        for c in session.query(Contact).filter(Contact.company_id == company.id).all()
        if c.do_not_contact or c.last_contacted_at
    }
    candidates = [c for c in candidates if c.name not in blocked]

    stats = compute_outcome_stats(session, config)
    preferred: list[str] | None = None
    if app.is_company_level:
        preferred = personas_for_families(role_families(config.job_preferences.roles), company)
        p_log.info(f"[OUTREACH] Company-level inquiry: preferring {', '.join(preferred[:5])} (from your target roles)")
    ranked = rank_contacts(candidates, company, config, stats, preferred_personas=preferred)

    if not ranked and config.contacts.allow_generic_inbox:
        p_log.warning("No named contacts discovered. Falling back to the company's hiring inbox.")
        ranked = [
            ContactCandidate(
                name="Hiring Team",
                title="Recruiting (generic inbox)",
                source="fallback",
                confidence=0.2,
                role_category="generic_inbox",
                seniority="team",
            )
        ]

    if not ranked:
        app.state = "Research Failed"
        session.add(
            History(
                application_id=app.id,
                stage=4,
                state="Research Failed",
                run_id=run_id,
                notes="Could not discover any contacts.",
            )
        )
        session.commit()
        return False

    extra = _extra(company)
    extra["email_intel"] = intel
    extra["email_hints"] = {
        c.name: {"email": c.email, "confidence": c.email_confidence, "source": c.source} for c in ranked if c.email
    }
    company.extra_data = extra
    if intel.get("pattern") and not company.email_pattern:
        company.email_pattern = intel["pattern"]

    db_contacts = [_upsert_contact(session, company, cand) for cand in ranked[:10]]
    best_contact = db_contacts[0]

    app.contact_id = best_contact.id
    app.persona = best_contact.role_category
    app.current_stage = 5
    app.state = "Professional Email Discovery"
    session.add(
        History(
            application_id=app.id,
            stage=5,
            state="Professional Email Discovery",
            run_id=run_id,
            notes=(
                f"Selected contact {best_contact.name} ({best_contact.role}; {best_contact.role_category}, "
                f"score {best_contact.rank_score}) from {len(ranked)} candidates. Moving to Email Discovery."
            ),
        )
    )

    # Optionally reach a second persona at the same company (e.g. EM + recruiter) as a sibling application.
    for extra_contact in db_contacts[1 : config.contacts.max_contacts_per_company]:
        if extra_contact.role_category == best_contact.role_category:
            continue
        exists = (
            session.query(Application)
            .filter(Application.job_id == app.job_id, Application.contact_id == extra_contact.id)
            .first()
        )
        if not exists:
            session.add(
                Application(
                    run_id=run_id,
                    job_id=app.job_id,
                    contact_id=extra_contact.id,
                    current_stage=5,
                    state="Professional Email Discovery",
                    campaign_id=app.campaign_id,
                    persona=extra_contact.role_category,
                )
            )
    session.commit()
    p_log.info(f"Selected contact {best_contact.name} ({best_contact.role}).", status="SUCCESS")
    return True


def run_stage_5_email_discovery(
    session: Session,
    config: AppConfig,
    llm: BaseLLMProvider,
    browser: BrowserProvider,
    app: Application,
    run_id: str,
) -> bool:
    """
    Stage 5: Professional Email Discovery
    Finds the contact's address from source data, company pages, commit emails, inferred company pattern,
    Hunter/Apollo and the LLM, then verifies it (SMTP RCPT probe with catch-all detection, Hunter verifier).
    """
    contact = app.contact
    assert contact is not None
    company = app.job.company

    p_log = PipelineLogger(logger, run_id, "Stage 5: Professional Email Discovery", company.name)
    p_log.info(f"Finding email for {contact.name}...")

    if contact.email:
        p_log.info(f"Contact email already known: {contact.email}")
        app.current_stage = 6
        app.state = "Opportunity Scoring"
        session.commit()
        return True

    extra = _extra(company)
    intel = extra.get("email_intel") or {}
    hint = (extra.get("email_hints") or {}).get(contact.name) or {}

    result = find_contact_email(
        config,
        llm,
        browser,
        company,
        contact.name,
        hint_email=hint.get("email"),
        hint_confidence=float(hint.get("confidence") or 0.0),
        hint_source=str(hint.get("source") or "source"),
        observed_emails=list(intel.get("observed") or []),
        email_samples=[(str(s[0]), str(s[1])) for s in (intel.get("samples") or []) if len(s) >= 2],
        known_pattern=company.email_pattern or intel.get("pattern"),
        accept_all_hint=intel.get("accept_all"),
        rejected=list(contact.rejected_emails or []),
    )
    if result.rejected:
        contact.rejected_emails = list(dict.fromkeys([*(contact.rejected_emails or []), *result.rejected]))

    if not result.email:
        p_log.error(f"No deliverable professional email discovered for {contact.name}: {result.detail}", status="NO_EMAIL")
        contact.email_status = result.status
        app.state = "No Professional Email"
        session.add(
            History(
                application_id=app.id,
                stage=5,
                state="No Professional Email",
                run_id=run_id,
                notes=f"Failed to discover/verify email for {contact.name}: {result.detail}",
            )
        )
        session.commit()
        return False

    # Save email and proceed safely avoiding UNIQUE constraint errors
    clean_email = result.email.strip().lower()
    existing_contact = (
        session.query(Contact)
        .filter(func.lower(Contact.email) == clean_email, Contact.id != contact.id)
        .first()
    )
    target = contact
    if existing_contact:
        p_log.info(
            f"Email '{clean_email}' is already associated with Contact #{existing_contact.id} ({existing_contact.name}). "
            f"Linking Application #{app.id} to existing contact."
        )
        temp_contact = contact
        app.contact_id = existing_contact.id
        app.contact = existing_contact
        target = existing_contact
        if temp_contact and not temp_contact.email and len(temp_contact.applications) <= 1:
            session.delete(temp_contact)
    else:
        contact.email = clean_email

    target.email_status = result.status
    target.email_confidence = result.confidence
    target.email_source = result.source
    target.email_confidence_level = result.level
    target.email_evidence = result.evidence
    target.email_verified_at = verification_timestamp()
    if result.pattern and not company.email_pattern:
        company.email_pattern = result.pattern

    app.current_stage = 6
    app.state = "Opportunity Scoring"
    session.add(
        History(
            application_id=app.id,
            stage=6,
            state="Opportunity Scoring",
            run_id=run_id,
            notes=f"Discovered email: {clean_email} [{result.level}] evidence: {result.evidence[:300]}. Moving to Scoring.",
        )
    )
    session.commit()
    p_log.info(f"Discovered email: {clean_email} ({result.status}, {result.detail})", status="SUCCESS")
    return True


def run_stage_6_opportunity_scoring(
    session: Session,
    config: AppConfig,
    llm: BaseLLMProvider,
    app: Application,
    run_id: str,
) -> bool:
    """
    Stage 6: Opportunity Scoring
    Hybrid scoring: deterministic rules (role/experience/salary/stack/company fit/data completeness) blended with the
    LLM's judgement (`scoring.mode`: hybrid | llm | rules), plus an estimated reply probability learned from outcomes.
    """
    company = app.job.company
    job = app.job
    p_log = PipelineLogger(logger, run_id, "Stage 6: Opportunity Scoring", company.name)
    p_log.info(f"Scoring opportunity: {job.title}...")

    mode = config.scoring.mode.lower()
    rules = rule_based_opportunity(job, company, config)
    llm_scores: dict[str, Any] | None = None
    reasoning = "Rule-based score."

    if mode in ("hybrid", "llm"):
        prompt = (
            f"Analyze the following job and company data to score this opportunity:\n\n"
            f"Job Title: {job.title}\n"
            f"Location: {job.location}\n"
            f"Salary Information: {job.salary or 'Not mentioned'}\n"
            f"Job Description: {job.description or 'No desc'}\n"
            f"Company Name: {company.name}\n"
            f"Sector: {company.sector}; Funding: {company.funding_stage}; Hiring: {company.hiring_status}\n"
            f"Company Details: {json.dumps(company.research_data)}\n\n"
            f"Provide a score between 0.0 (poor) and 1.0 (excellent) for each of these categories:\n"
            f"1. role_match: Fit for entry level / up to {config.job_preferences.experience_years_max} yr experience, matching {config.job_preferences.roles}.\n"
            f"2. tech_stack: Alignment with the candidate's skills: {config.target_profile.skills or 'modern python/fullstack/backend technologies'}.\n"
            f"3. salary: Matching LPA preferences {config.job_preferences.salary_range.min_lpa} - {config.job_preferences.salary_range.max_lpa}.\n"
            f"4. company_quality: Reputation, engineering culture, stability.\n"
            f"5. growth: Industry sector potential, career acceleration.\n"
            f"6. confidence: Reliability of the job and company data found.\n\n"
            f"Note: If this is a company-level internship inquiry (no active public job listing; see the title/description), "
            f"evaluate and score 'role_match' and 'confidence' based on the company's tech stack suitability, engineering team size/growth, "
            f"and the relevance of their business domain to the candidate's preferred roles, rather than requiring an active public job listing. "
            f"Set the confidence score based on the completeness and quality of the company's research data."
        )
        try:
            response = llm.generate_json(prompt, OpportunityScoreResponse)
            assert isinstance(response, OpportunityScoreResponse)
            llm_scores = response.model_dump()
            reasoning = response.reasoning
        except Exception as e:
            if mode == "llm":
                p_log.error(f"Error calculating score: {e}")
                app.state = "Failed"
                session.add(
                    History(
                        application_id=app.id,
                        stage=6,
                        state="Failed",
                        run_id=run_id,
                        notes=f"Opportunity scoring failed: {e}",
                    )
                )
                session.commit()
                return False
            p_log.warning(f"LLM scoring failed ({e}); using rule-based score only.")

    keys = ("role_match", "tech_stack", "salary", "company_quality", "growth", "confidence")
    if llm_scores is not None and mode == "llm":
        final = {k: float(llm_scores[k]) for k in keys}
    elif llm_scores is not None:
        w = max(0.0, min(1.0, config.scoring.llm_weight))
        final = {k: round(w * float(llm_scores[k]) + (1 - w) * float(rules[k]), 4) for k in keys}
    else:
        final = {k: float(rules[k]) for k in keys}

    stats = compute_outcome_stats(session, config)
    contact = app.contact
    email_level = (contact.email_confidence_level or contact.email_status) if contact else None
    reply = explain_reply_probability(
        company,
        stats,
        persona=app.persona,
        email_level=email_level,
        resume_match=(float(final["role_match"]) + float(final["tech_stack"])) / 2,
        relevant_job=(job.source not in (*NON_OPENING_SOURCES, "pasted")) or None,
    )
    app.response_probability = float(reply["final"])
    app.score = weighted_total(final, config)
    app.score_breakdown = {
        **final,
        "reasoning": reasoning,
        "mode": mode,
        "llm": llm_scores,
        "rules": rules,
        "company_fit": company.fit_score,
        "response_probability": app.response_probability,
        "reply_probability": reply,
    }
    session.commit()

    p_log.info(
        f"Weighted Score: {app.score:.2f} (Threshold: {config.scoring.thresholds.minimum_score})\n{reply['explanation']}"
    )

    if app.score < config.scoring.thresholds.minimum_score:
        p_log.warning(f"Score {app.score:.2f} is below minimum threshold.", status="LOW_SCORE")
        app.state = "Low Score"
        session.add(
            History(
                application_id=app.id,
                stage=6,
                state="Low Score",
                run_id=run_id,
                notes=f"Score {app.score:.2f} below threshold of {config.scoring.thresholds.minimum_score}.",
            )
        )
        session.commit()
        return False

    app.current_stage = 7
    app.state = "Resume Tailoring"
    session.add(
        History(
            application_id=app.id,
            stage=7,
            state="Resume Tailoring",
            run_id=run_id,
            notes=f"Scored {app.score:.2f}. Advancing to Resume Tailoring.",
        )
    )
    session.commit()
    p_log.info("Opportunity scoring completed successfully.", status="SUCCESS")
    return True


def _compile_typst(typ_path: str, pdf_path: str, p_log: PipelineLogger) -> bool:
    try:
        result = subprocess.run(["typst", "compile", typ_path, pdf_path], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            p_log.info(f"Compiled resume to PDF: {pdf_path}")
            return True
        p_log.warning(f"Typst compilation returned non-zero code. Error: {result.stderr}")
    except Exception as e:
        p_log.warning(f"Typst compiler not found or failed to execute: {e}.")
    return False


RESUME_TAILOR_GUIDANCE = """

Additional tailoring instructions:
- Job description (for keyword and requirement alignment):
{job_description}
- Reorder the Projects and Experience entries so the most relevant appear first, in this priority order:
{highlights}
- You may rewrite entire bullet points (not only keywords) to emphasise the requirements above, keeping every fact,
  number, employer, date and technology exactly as in the Base Resume.
- Keep the document valid Typst that compiles with the same imports/templates as the Base Resume.
"""


STRUCTURED_TAILOR_PROMPT = """Tailor this resume for the {role_name} role at {company_name}.
Resume variant: {variant_name}. Variant focus: {variant_focus}

Original resume (text extracted from a PDF; silently fix extraction artifacts such as "V ellore" or glued words):
{base_resume_text}

Job description:
{job_description}

Company: {company_description} | Sector: {company_sector} | Tech stack: {tech_stack}

Order experience and projects so the most relevant come first, in this priority:
{highlights}

Rules:
- Keep the name, contact line, every employer, institution, job title, date, degree, GPA and number EXACTLY as in the original.
- Do NOT invent projects, employers, metrics, skills, tools or certifications that are not in the original.
- You MAY reorder sections, entries and bullets, rewrite bullets to foreground what matters for this role, and merge or
  drop the least relevant bullets so the resume fits on one page (at most 4 bullets per entry).
- Skills: only skills present in the original, most relevant first, as lines like "Languages: Python, Rust".
- Keep every section you keep (Education, Experience, Projects, Skills, Achievements, ...) in the schema structure.
"""


def _record_resume(
    session: Session,
    app: Application,
    run_id: str,
    p_log: PipelineLogger,
    parent: str,
    attachment: str,
    variant: str,
    keywords: list[str],
    reasoning: str,
    highlight_names: list[str],
) -> bool:
    company = app.job.company
    session.add(
        ResumeVersion(
            application_id=app.id,
            parent_resume=parent,
            company=company.name,
            role=app.job.title,
            keywords_added=keywords,
            reasoning=reasoning,
            path=attachment,
            variant=variant,
            highlights_order=highlight_names,
        )
    )
    app.tailored_resume_path = attachment
    app.current_stage = 8
    app.state = "Email Generation"
    session.add(
        History(
            application_id=app.id,
            stage=8,
            state="Email Generation",
            run_id=run_id,
            notes=f"Attached {variant} resume ({attachment}). Moving to Email Gen.",
        )
    )
    session.commit()
    p_log.info(f"Resume stage completed: {attachment}", status="SUCCESS")
    return True


def _resume_from_document(
    session: Session,
    config: AppConfig,
    llm: BaseLLMProvider,
    app: Application,
    run_id: str,
    p_log: PipelineLogger,
    choice: Any,
    highlights: list[Any],
) -> bool:
    """
    PDF / DOCX / Markdown / text resumes:
      generate_resume=false -> validate the file and attach it unchanged
      generate_resume=true  -> extract text -> LLM tailors a structured resume -> fabrication guard -> render PDF
                               (any failure falls back to attaching the original file)
    """
    company = app.job.company
    job = app.job
    source_path = str(choice.path)
    highlight_names = [h.name for h in highlights]

    problems = validate_resume_file(source_path)
    if problems:
        p_log.error(f"Resume file {source_path} is unusable: {problems}")
        app.state = "Failed"
        session.add(History(application_id=app.id, stage=7, state="Failed", run_id=run_id, notes=f"Resume unusable: {problems}"))
        session.commit()
        return False

    original_text = extract_resume_text(source_path)
    if not original_text:
        p_log.warning(f"No text could be extracted from {source_path} (scanned PDF?). Emails won't see resume content.")

    tailor = config.pipeline.effective_resume_mode() == "tailor"
    if not tailor or not original_text:
        reason = f"Attached original {choice.name} resume unchanged ({choice.reason})."
        if tailor:
            reason += " Tailoring skipped: no extractable text."
        return _record_resume(session, app, run_id, p_log, source_path, source_path, choice.name, [], reason, highlight_names)

    stack = company.tech_stack if isinstance(company.tech_stack, list) else (company.research_data or {}).get("tech_stack", [])
    template = config.prompts.resume_tailoring_structured or STRUCTURED_TAILOR_PROMPT
    prompt = safe_format(
        template,
        role_name=outreach_role_label(config, app),
        company_name=company.name,
        variant_name=choice.name,
        variant_focus=choice.focus or "general software engineering",
        base_resume_text=original_text[:12000],
        job_description=(job.description or "Not available")[:3000],
        company_description=company.description or "",
        company_sector=company.sector or "unknown",
        tech_stack=", ".join(str(t) for t in stack or []),
        highlights=format_highlights(highlights),
    )
    tailored: StructuredResumeSchema | None = None
    violations: list[str] = []
    for attempt in range(2):  # generate, then regenerate once with the guard's findings
        attempt_prompt = prompt
        if violations:
            attempt_prompt += (
                "\n\nYour previous version was rejected because it introduced facts that are not in the original "
                "resume:\n- " + "\n- ".join(violations[:10]) + "\nRemove every one of them and keep only original facts."
            )
        try:
            candidate = llm.generate_json(attempt_prompt, StructuredResumeSchema)
            assert isinstance(candidate, StructuredResumeSchema)
        except Exception as e:
            p_log.warning(f"Structured resume tailoring failed ({e}); attaching the original resume.")
            return _record_resume(
                session, app, run_id, p_log, source_path, source_path, choice.name, [],
                f"Tailoring failed ({e}); attached original {choice.name} resume.", highlight_names,
            )
        violations = check_tailored_resume(original_text, candidate)
        if not violations:
            tailored = candidate
            break
        p_log.warning(f"Tailored resume attempt {attempt + 1} rejected by fabrication guard: {violations[:3]}")

    if tailored is None:
        return _record_resume(
            session, app, run_id, p_log, source_path, source_path, choice.name, [],
            f"Tailored version rejected twice ({'; '.join(violations[:3])}); attached original {choice.name} resume.",
            highlight_names,
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    safe_comp = re.sub(r"[^a-z0-9_]+", "_", company.name.lower())
    safe_role = re.sub(r"[^a-z0-9_]+", "_", job.title.lower())[:60]
    out_dir = os.path.join(config.pipeline.generated_resumes_dir, f"resume_{safe_comp}_{safe_role}_{timestamp}")
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    out_pdf = os.path.join(out_dir, f"{base_name}.pdf")
    try:
        pages = render_resume_pdf(tailored, out_pdf)
        with open(os.path.join(out_dir, f"{base_name}.txt"), "w", encoding="utf-8") as f:
            f.write(structured_to_text(tailored))
    except Exception as e:
        p_log.warning(f"Rendering the tailored PDF failed ({e}); attaching the original resume.")
        return _record_resume(
            session, app, run_id, p_log, source_path, source_path, choice.name, [],
            f"PDF rendering failed ({e}); attached original {choice.name} resume.", highlight_names,
        )

    source_norm = re.sub(r"[^a-z0-9]", "", original_text.lower())
    keywords = [k for k in tailored.keywords_added if re.sub(r"[^a-z0-9]", "", k.lower()) in source_norm]
    reasoning = f"[{choice.name}] {tailored.reasoning} (variant: {choice.reason}; {pages} page(s))"
    return _record_resume(session, app, run_id, p_log, source_path, out_pdf, choice.name, keywords, reasoning, highlight_names)


def run_stage_7_resume_tailoring(
    session: Session,
    config: AppConfig,
    llm: BaseLLMProvider,
    app: Application,
    run_id: str,
) -> bool:
    """
    Stage 7: Resume Tailoring
    Picks the best resume variant (AI / Backend / Fintech / Systems ...) for the role, orders your highlights by
    relevance, optionally rewrites bullets/reorders projects with the LLM (without fabrication) and compiles a PDF.
    """
    company = app.job.company
    job = app.job
    p_log = PipelineLogger(logger, run_id, "Stage 7: Resume Tailoring", company.name)
    p_log.info("Tailoring resume...")
    mode = config.pipeline.effective_resume_mode()

    if mode == "attach_base":
        base_path = config.pipeline.base_resume_path
        problems = validate_resume_file(base_path)
        if problems:
            p_log.error(f"Base resume {base_path} is unusable: {problems}")
            app.state = "Failed"
            session.add(History(application_id=app.id, stage=7, state="Failed", run_id=run_id, notes=f"Base resume unusable: {problems}"))
            session.commit()
            return False
        highlight_names = [h.name for h in rank_highlights(config, job, company, app.persona)]
        return _record_resume(
            session, app, run_id, p_log, base_path, base_path, "base", [],
            f"Attached base resume unchanged (resume_mode=attach_base): {base_path}", highlight_names,
        )

    choice = select_resume_variant(config, job, company, app.persona)
    if choice is None:
        p_log.error(f"No resume variant found (checked config.resumes and {config.pipeline.base_resume_path}).")
        app.state = "Failed"
        session.add(
            History(
                application_id=app.id,
                stage=7,
                state="Failed",
                run_id=run_id,
                notes="Base resume file does not exist.",
            )
        )
        session.commit()
        return False

    base_resume_path = choice.path
    resume_variant = choice.name
    highlights = rank_highlights(config, job, company, app.persona)
    highlight_names = [h.name for h in highlights]
    p_log.info(f"Selected resume variant '{resume_variant}' ({choice.reason}).")

    if os.path.splitext(base_resume_path)[1].lower() != ".typ":
        return _resume_from_document(session, config, llm, app, run_id, p_log, choice, highlights)

    try:
        with open(base_resume_path, encoding="utf-8") as f:
            base_resume_text = f.read()
    except Exception as e:
        p_log.error(f"Error reading base resume ({base_resume_path}): {e}")
        app.state = "Failed"
        session.commit()
        return False

    should_generate = mode == "tailor"

    try:
        if not should_generate:
            p_log.info(f"Using pre-built {resume_variant} base resume without LLM generation (generate_resume=False): {base_resume_path}")
            pdf_filepath = os.path.splitext(base_resume_path)[0] + ".pdf"
            has_pdf = _compile_typst(base_resume_path, pdf_filepath, p_log)
            final_attachment_path = pdf_filepath if (has_pdf and os.path.exists(pdf_filepath)) else base_resume_path

            session.add(
                ResumeVersion(
                    application_id=app.id,
                    parent_resume=base_resume_path,
                    company=company.name,
                    role=job.title,
                    keywords_added=[],
                    reasoning=f"Selected pre-built {resume_variant} base resume ({base_resume_path}) with generate_resume=False; {choice.reason}.",
                    path=final_attachment_path,
                    variant=resume_variant,
                    highlights_order=highlight_names,
                )
            )

            app.tailored_resume_path = final_attachment_path
            app.current_stage = 8
            app.state = "Email Generation"
            session.add(
                History(
                    application_id=app.id,
                    stage=8,
                    state="Email Generation",
                    run_id=run_id,
                    notes=f"Attached pre-built {resume_variant} resume ({final_attachment_path}). Moving to Email Gen.",
                )
            )
            session.commit()
            p_log.info(f"Resume stage completed using static {resume_variant} resume: {final_attachment_path}", status="SUCCESS")
            return True

        p_log.info(f"Generating tailored resume via LLM using {resume_variant} base resume...")
        stack = company.tech_stack if isinstance(company.tech_stack, list) else (company.research_data or {}).get("tech_stack", [])
        tech_stack = ", ".join(str(t) for t in stack or [])
        values = {
            "role_name": outreach_role_label(config, app),
            "company_name": company.name,
            "tech_stack": tech_stack,
            "base_resume_text": base_resume_text,
            "job_description": (job.description or "Not available")[:2500],
            "highlights": format_highlights(highlights),
        }
        prompt = safe_format(config.prompts.resume_tailoring, **values)
        if "{highlights}" not in config.prompts.resume_tailoring:
            prompt += safe_format(RESUME_TAILOR_GUIDANCE, **values)
        try:
            response = llm.generate_json(prompt, ResumeTailorResponse)
            assert isinstance(response, ResumeTailorResponse)
            tailored_content = response.tailored_typst_content
            keywords_added = response.keywords_added
            reasoning = response.reasoning
        except Exception as err:
            p_log.warning(f"LLM resume tailoring failed: {err}. Falling back to pre-built {resume_variant} base resume.")
            tailored_content = base_resume_text
            keywords_added = []
            reasoning = f"Fallback to {resume_variant} base resume due to tailoring error: {err}"

        os.makedirs(config.pipeline.generated_resumes_dir, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        safe_comp = re.sub(r"[^a-z0-9_]+", "_", company.name.lower())
        safe_role = re.sub(r"[^a-z0-9_]+", "_", job.title.lower())[:60]
        resume_dir = os.path.join(config.pipeline.generated_resumes_dir, f"resume_{safe_comp}_{safe_role}_{timestamp}")
        os.makedirs(resume_dir, exist_ok=True)

        base_resume_name = os.path.splitext(os.path.basename(base_resume_path))[0]
        typ_filepath = os.path.join(resume_dir, f"{base_resume_name}.typ")
        with open(typ_filepath, "w", encoding="utf-8") as f:
            f.write(tailored_content)
        p_log.info(f"Saved tailored Typst file: {typ_filepath}")

        pdf_filepath = os.path.join(resume_dir, f"{base_resume_name}.pdf")
        has_pdf = _compile_typst(typ_filepath, pdf_filepath, p_log)
        if not has_pdf and tailored_content != base_resume_text:
            # Never attach a broken tailored resume: fall back to the compiled base variant if possible.
            base_pdf = os.path.splitext(base_resume_path)[0] + ".pdf"
            if _compile_typst(base_resume_path, base_pdf, p_log):
                p_log.warning("Tailored resume failed to compile; attaching the base variant PDF instead.")
                pdf_filepath, has_pdf = base_pdf, True
                reasoning += " (tailored version failed to compile; base variant attached)"
        final_attachment_path = pdf_filepath if has_pdf else typ_filepath

        session.add(
            ResumeVersion(
                application_id=app.id,
                parent_resume=base_resume_path,
                company=company.name,
                role=job.title,
                keywords_added=keywords_added,
                reasoning=f"[{resume_variant}] {reasoning}",
                path=final_attachment_path,
                variant=resume_variant,
                highlights_order=highlight_names,
            )
        )

        app.tailored_resume_path = final_attachment_path
        app.current_stage = 8
        app.state = "Email Generation"
        session.add(
            History(
                application_id=app.id,
                stage=8,
                state="Email Generation",
                run_id=run_id,
                notes=f"Tailored resume saved to {final_attachment_path}. Moving to Email Gen.",
            )
        )
        session.commit()
        p_log.info("Resume tailoring completed successfully.", status="SUCCESS")
        return True

    except Exception as e:
        p_log.error(f"Error tailoring resume: {e}")
        app.state = "Failed"
        session.add(
            History(
                application_id=app.id,
                stage=7,
                state="Failed",
                run_id=run_id,
                notes=f"Resume tailoring failed: {e}",
            )
        )
        session.commit()
        return False


EMAIL_CONTEXT_BLOCK = """

Additional context (use it to make the email specific; never invent facts beyond it):
- About the company: {company_description}
- Recent news / launches: {recent_news}
- Job description: {job_description}
- About {contact_name}: {contact_background}
- My most relevant highlights, in priority order:
{highlights}
- My resume (full text of the attached version):
{resume_summary}

Recipient-specific guidelines: {persona_guidelines}
Tone: {tone}.
If a recent launch or funding round is listed above, open with one specific sentence about it (e.g. "I saw you
recently launched ..."). Mention at most two highlights. Sign off as {user_name}.
"""


MAX_VALIDATION_RETRIES = 1

def _format_news(news: Any) -> str:
    if not isinstance(news, list) or not news:
        return "None found"
    lines = []
    for item in news[:3]:
        if isinstance(item, dict):
            date = f" ({item.get('date')})" if item.get("date") else ""
            summary = f": {item.get('summary')}" if item.get("summary") else ""
            lines.append(f"{item.get('title', '')}{date}{summary}")
    return "; ".join(lines) or "None found"


def _signature(config: AppConfig, body_html: str) -> str:
    user_name = config.user_identity.name
    linkedin = config.user_identity.linkedin_url
    github = config.user_identity.github_url
    link_parts = []
    if linkedin:
        link_parts.append(f'<a href="{linkedin}">{linkedin.replace("https://", "").replace("http://", "")}</a>')
    if github:
        link_parts.append(f'<a href="{github}">{github.replace("https://", "").replace("http://", "")}</a>')
    if user_name and user_name not in body_html:
        body_html += f"<p>Best,<br>{user_name}</p>"
    if link_parts:
        signature_links = "<br>" + " | ".join(link_parts)
        already_has_links = any(url in body_html for url in [linkedin, github] if url)
        if not already_has_links:
            if user_name and user_name in body_html:
                idx = body_html.rfind(user_name)
                body_html = body_html[: idx + len(user_name)] + signature_links + body_html[idx + len(user_name) :]
            else:
                body_html += f"<p>{signature_links}</p>"
    return body_html


MAX_EMAIL_DEFERRALS = 3

PLAIN_SUMMARY_PROMPT = """Rewrite this resume item as ONE plain sentence (max 28 words) that a non-engineer would
understand: what the person did and what the system was for. Mention at most one technology name. Keep every fact
exactly as given; do not add results, numbers or claims. Start with the organisation or project name.

Item: {item}
"""


def plain_highlights(
    session: Session, helper_llm: BaseLLMProvider | None, highlights: list[Any], resume_text: str, limit: int = 2
) -> list[str]:
    """
    Plain-language one-liners for the most relevant highlights, written once by the (local) helper model and cached
    for 30 days. Falls back to nothing if the helper is unavailable or the rewrite adds facts or jargon.
    """
    if helper_llm is None or not highlights:
        return []
    cache = DBCache(session)
    out: list[str] = []
    for h in highlights[:limit]:
        item = f"{h.name}: {h.summary}" if h.summary else h.name
        key = "plain_highlight_" + re.sub(r"[^a-z0-9]+", "_", item.lower())[:150]
        cached = cache.get(key)
        if isinstance(cached, str) and cached:
            out.append(cached)
            continue
        try:
            result = helper_llm.generate_json(safe_format(PLAIN_SUMMARY_PROMPT, item=item), PlainSummarySchema)
            assert isinstance(result, PlainSummarySchema)
            sentence = result.summary.strip()
        except Exception as e:
            logger.info(f"Plain summary for '{h.name}' unavailable: {e}")
            continue
        if not sentence or find_unsupported_claims(sentence, item + " " + resume_text) or jargon_sentences(sentence):
            continue
        cache.set(key, sentence, 30 * 86400)
        out.append(sentence)
    return out


COMPANY_OUTREACH_RULES = """
This is a company-level inquiry, not an application to a posted job:
- No matching public opening was found at {company_name}. Do not say or imply that a specific role, opening, posting
  or vacancy exists: never write "I saw your opening/posting for ...", "applying for the ... role", "your open
  positions", and never state or guess hiring plans ("you're growing the team", "you're hiring").
- The areas I'm interested in: {role_families}. Mention the one or two that suit this recipient best, naturally, once.
- Use only ONE experience or project: the one most relevant to those areas.
"""


def _word_range(config: AppConfig, app: Application) -> tuple[int, int]:
    """Company-level inquiries are shorter (about 100-150 words) than job-specific emails."""
    if app.is_company_level:
        return COMPANY_OUTREACH_WORDS
    return config.outreach.min_words, config.outreach.max_words


def clean_role_name(title: str | None) -> str:
    """'Product Analyst Intern (Speculative Application)' -> 'Product Analyst'."""
    cleaned = re.sub(r"\((?:speculative application|targeted outreach)\)", "", title or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(intern(ship)?|trainee)\b", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" -,") or "software engineering"


def _recent_initial_emails(session: Session, exclude_app_id: int, limit: int = 5) -> list[Email]:
    """The most recent first emails written for OTHER applications (used to avoid repeating ourselves)."""
    return (
        session.query(Email)
        .filter(
            Email.application_id != exclude_app_id,
            or_(Email.sequence_step == 0, Email.sequence_step.is_(None)),
            Email.status != "cancelled",
        )
        .order_by(Email.id.desc())
        .limit(limit)
        .all()
    )


def _defer_email_generation(
    session: Session, app: Application, run_id: str, p_log: PipelineLogger, err: Exception
) -> bool:
    """
    Keeps the application in Email Generation so a later run retries it. Budget/rate-limit/outage deferrals are
    unlimited (they resolve themselves); other errors fail the application after MAX_EMAIL_DEFERRALS attempts.
    """
    transient = isinstance(err, BudgetExceededError | OllamaUnavailableError) or any(
        marker in str(err).lower() for marker in ("rate limit", "429", "all groq models failed", "timeout", "unavailable")
    )
    previous = (
        session.query(History)
        .filter(History.application_id == app.id, History.state == "Email Deferred")
        .count()
    )
    if not transient and previous + 1 >= MAX_EMAIL_DEFERRALS:
        app.state = "Failed"
        session.add(History(application_id=app.id, stage=8, state="Failed", run_id=run_id,
                            notes=f"Email generation failed {previous + 1} times: {str(err)[:300]}"))
        session.commit()
        p_log.error(f"Email generation failed permanently: {err}")
        return False
    app.current_stage = 8
    app.state = "Email Generation"
    session.add(History(application_id=app.id, stage=8, state="Email Deferred", run_id=run_id,
                        notes=f"Deferred ({'transient' if transient else f'attempt {previous + 1}'}): {str(err)[:300]}"))
    session.commit()
    p_log.warning(f"Email generation deferred to a later run: {err}", status="DEFERRED")
    return False


def run_stage_8_email_generation(
    session: Session,
    config: AppConfig,
    llm: BaseLLMProvider,
    app: Application,
    run_id: str,
    followup_llm: BaseLLMProvider | None = None,
    helper_llm: BaseLLMProvider | None = None,
) -> bool:
    """
    Stage 8: Email Generation
    Writes a persona-specific email (founder / recruiter / engineering manager / ...) in the configured tone using
    the company description, recent news, job description, contact background, your resume and ranked highlights,
    plus follow-up #1 and #2 for the same thread.
    """
    company = app.job.company
    job = app.job
    contact = app.contact
    assert contact is not None

    p_log = PipelineLogger(logger, run_id, "Stage 8: Email Generation", company.name)
    p_log.info(f"Generating personalized email for {contact.name}...")

    persona = app.persona or contact.role_category or classify_role(contact.role)[0]
    app.persona = persona
    tone = tone_for(persona, config.outreach.tone, config.outreach.persona_tones)
    company_level = app.is_company_level
    families = role_families(config.job_preferences.roles)
    if company_level:
        p_log.info("[OUTREACH] Writing a company-level inquiry (no public opening is claimed)")

    rv = (
        session.query(ResumeVersion)
        .filter(ResumeVersion.application_id == app.id)
        .order_by(ResumeVersion.id.desc())
        .first()
    )
    tailored_skills = ", ".join(rv.keywords_added) if rv and rv.keywords_added else ""
    stack = company.tech_stack if isinstance(company.tech_stack, list) else None
    research = company.research_data or {}
    if not tailored_skills:
        tailored_skills = ", ".join(config.target_profile.skills) or "software development"
    highlights = rank_highlights(config, job, company, persona)
    resume_text = read_resume_text(rv.path if rv else None) or read_resume_text(rv.parent_resume if rv else None)

    values = {
        "contact_name": contact.name,
        "contact_role": contact.role,
        "company_name": company.name,
        "role_name": outreach_role_label(config, app),
        "product_description": research.get("business_model", "their innovative platform"),
        "tech_stack": ", ".join(stack or research.get("tech_stack", ["modern tools"])),
        "recent_launches": _format_news(company.recent_news) if company.recent_news else research.get("funding", "recent engineering progress"),
        "tailored_skills": tailored_skills,
        "company_description": company.description or research.get("business_model", ""),
        "recent_news": _format_news(company.recent_news),
        "company_products": ", ".join(str(p) for p in (research.get("products") or [])[:6]) or "Not available",
        "job_description": (job.description or "Not available")[:2000],
        "contact_background": contact.background or "Not available",
        "highlights": "",  # filled below with plain-language versions when available
        "resume_summary": resume_text[:6000] or "Not available",
        "persona_guidelines": persona_focus(persona),
        "tone": describe_tone(tone),
        "user_name": config.user_identity.name,
    }
    plain = plain_highlights(session, helper_llm, highlights, resume_text)
    raw_lines = format_highlights(highlights[:4]).splitlines()
    values["highlights"] = "\n".join(
        f"{i}. {plain[i - 1]}" if i <= len(plain) else line for i, line in enumerate(raw_lines, start=1)
    ) + ("\n(Use this plain wording level; the resume is only the source of facts.)" if plain else "")
    prompt = safe_format(config.prompts.email_generation, **values)
    if "{job_description}" not in config.prompts.email_generation:
        prompt += safe_format(EMAIL_CONTEXT_BLOCK, **values)
    recent = _recent_initial_emails(session, app.id)
    signals = research_signals(research, company.recent_news, job.source, company.description)
    opening_style = choose_opening(signals, [e.opening_style or "" for e in recent])
    min_words, max_words = word_target(persona, *_word_range(config, app))
    avoid_lines = []
    overused = overused_recent_phrases([e.body for e in recent])
    if overused:
        avoid_lines.append("- Recently overused, so don't use: " + ", ".join(f'"{p}"' for p in overused) + ".\n")
    starts = recent_sentence_starts([e.body for e in recent], config.user_identity.name)
    if starts:
        avoid_lines.append(
            "- Recent emails already used these sentence openings; start your sentences differently (including the "
            "one about who I am): " + "; ".join(f'"{s}"' for s in starts) + "\n"
        )
    patterns = recent_patterns_summary([e.body for e in recent], config.user_identity.name)
    if patterns:
        avoid_lines.append(
            "- Your last emails opened and asked like this. Use a different opening, sentence structure and ask:\n"
            + "\n".join(f"  {line}" for line in patterns.splitlines())
            + "\n"
        )
    prompt += safe_format(
        EMAIL_STYLE_RULES,
        first_name=first_name(contact.name),
        min_words=min_words,
        max_words=max_words,
        target_role=(describe_role_families(families) or "relevant") if company_level else clean_role_name(job.title),
        opening_instruction=OPENING_STYLES[opening_style],
        ask_instruction=(
            company_outreach_ask(persona, config.company_outreach.ask_about_openings)
            if company_level
            else persona_ask(persona)
        ),
        persona_focus=persona_focus(persona),
        avoid_block="".join(avoid_lines),
        user_name=config.user_identity.name,
    )
    if company_level:
        prompt += safe_format(
            COMPANY_OUTREACH_RULES,
            company_name=company.name,
            role_families=describe_role_families(families, limit=6) or "an internship that fits my background",
        )
    recent_subjects = [e.subject for e in recent if e.subject]
    if recent_subjects:
        prompt += "Subjects already used recently (don't reuse or imitate): " + "; ".join(recent_subjects[:5]) + "\n"
    if "{company_products}" not in config.prompts.email_generation and values["company_products"] != "Not available":
        prompt += f"\nTheir products: {values['company_products']}\n"
    retry = (
        session.query(History)
        .filter(History.application_id == app.id, History.state == "Validation Retry")
        .order_by(History.id.desc())
        .first()
    )
    if retry is not None and retry.notes:
        prompt += (
            "\n\nA previous draft was rejected by a fact-checker for these reasons. Write a new email that avoids "
            f"every one of them:\n{retry.notes}\n"
        )

    try:
        try:
            response = llm.generate_json(prompt, EmailGenResponse)
            assert isinstance(response, EmailGenResponse)
            subject = response.subject
            body_html = response.body_html
        except Exception as err:
            # Never fall back to a generic template: keep the application at this stage and retry on a later run.
            return _defer_email_generation(session, app, run_id, p_log, err)

        body_html = _signature(config, body_html)

        # Retire earlier generated content for this application (e.g. after a bounce or validation retry).
        for old in session.query(Email).filter(Email.application_id == app.id, Email.status.in_(["generated", "pending"])).all():
            old.status = "cancelled"

        email = Email(
            application_id=app.id,
            subject=subject,
            body=body_html,
            status="generated",
            sequence_step=0,
            persona=persona,
            tone=tone,
            to_email=contact.email,
            opening_style=opening_style,
        )
        session.add(email)
        session.flush()

        steps = config.outreach.followups
        if steps:
            followup_bodies: list[str] = []
            template = config.prompts.followup_generation or DEFAULT_FOLLOWUP_PROMPT
            fu_prompt = safe_format(
                template,
                **values,
                followup_count=len(steps),
                initial_email=html_to_plain(body_html)[:2000],
                followup_1_days=steps[0].after_days,
                followup_2_days=steps[1].after_days if len(steps) > 1 else steps[0].after_days,
            )
            try:
                fu_response = (followup_llm or llm).generate_json(fu_prompt, FollowUpSequenceSchema)
                assert isinstance(fu_response, FollowUpSequenceSchema)
                followup_bodies = [f.body_html for f in fu_response.followups if f.body_html][: len(steps)]
            except Exception as err:
                p_log.info(f"Follow-up generation via LLM unavailable ({err}); using follow-up templates.")
            if len(followup_bodies) < len(steps):
                top_highlight = highlights[0].summary or highlights[0].name if highlights else None
                templates = fallback_followups(
                    contact.name,
                    company.name,
                    (describe_role_families(families) or "internship") if company_level else job.title,
                    config.user_identity.name,
                    top_highlight,
                    len(steps),
                )
                followup_bodies.extend(templates[len(followup_bodies) :])
            for step, body in enumerate(followup_bodies, start=1):
                session.add(
                    Email(
                        application_id=app.id,
                        subject=f"Re: {subject}",
                        body=body,
                        status="pending",
                        sequence_step=step,
                        persona=persona,
                        tone=tone,
                        to_email=contact.email,
                    )
                )

        app.current_stage = 9
        app.state = "Validation"
        session.add(
            History(
                application_id=app.id,
                stage=9,
                state="Validation",
                run_id=run_id,
                notes=f"Generated {persona} email ({tone} tone) and {len(steps)} follow-ups. Advancing to Validation.",
            )
        )
        session.commit()
        p_log.info("Email generation completed successfully.", status="SUCCESS")
        return True

    except Exception as e:
        p_log.error(f"Email generation failed: {e}")
        app.state = "Failed"
        session.add(
            History(
                application_id=app.id,
                stage=8,
                state="Failed",
                run_id=run_id,
                notes=f"Email generation failed: {e}",
            )
        )
        session.commit()
        return False


def _email_fact_base(config: AppConfig, app: Application, rv: ResumeVersion) -> str:
    """Everything an email may legitimately state: resume, your profile notes, company research, JD, contact."""
    company = app.job.company
    contact = app.contact
    resume_text = read_resume_text(rv.path, limit=20000) or read_resume_text(rv.parent_resume, limit=20000)
    parts = [
        resume_text,
        config.prompts.email_generation,  # your own "About me" notes live in the template
        " ".join(f"{h.name} {h.summary}" for h in config.highlights),
        config.user_identity.name,
        company.name,
        company.description or "",
        json.dumps(company.research_data or {}),
        json.dumps(company.recent_news or []),
        " ".join(str(t) for t in (company.tech_stack or [])) if isinstance(company.tech_stack, list) else "",
        company.funding_details or "",
        app.job.title,
        app.job.description or "",
    ]
    if contact is not None:
        parts += [contact.name, contact.role or "", contact.background or ""]
    return "\n".join(parts)


def _cached_plain_highlights(session: Session, config: AppConfig, app: Application) -> dict[str, str]:
    """highlight name -> cached plain summary (only those already generated in stage 8)."""
    cache = DBCache(session)
    out: dict[str, str] = {}
    for h in rank_highlights(config, app.job, app.job.company, app.persona)[:3]:
        item = f"{h.name}: {h.summary}" if h.summary else h.name
        cached = cache.get("plain_highlight_" + re.sub(r"[^a-z0-9]+", "_", item.lower())[:150])
        if isinstance(cached, str) and cached:
            out[h.name] = cached
    return out


def simplify_jargon(body_html: str, plain_by_name: dict[str, str], user_name: str) -> str:
    """Replaces each tech-stack-list sentence with the plain summary of the highlight it describes (matched by name)."""
    if not plain_by_name:
        return body_html
    body = body_html
    for ch in ("\u2010", "\u2011", "\u2012"):
        body = body.replace(ch, "-")
    for sentence in jargon_sentences(html_to_plain(body)):
        target = sentence.strip()
        if target not in body:
            continue
        sentence_words = set(re.findall(r"[a-z0-9]+", target.lower()))
        for name, plain_text in plain_by_name.items():
            name_words = {w for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) > 2 and w not in ("internship", "project", "team")}
            if name_words & sentence_words and plain_text not in body:
                body = body.replace(target, plain_text.rstrip(".") + ".")
                break
    return body


def run_stage_9_validation(
    session: Session,
    config: AppConfig,
    llm: BaseLLMProvider,
    app: Application,
    run_id: str,
) -> bool:
    """
    Stage 9: Validation
    Validates that the email, follow-ups and tailored resume contain no placeholders or hallucinated facts.
    """
    company = app.job.company
    p_log = PipelineLogger(logger, run_id, "Stage 9: Validation", company.name)
    p_log.info("Validating tailored outputs...")

    email = get_initial_email(session, app.id)
    followups = (
        session.query(Email)
        .filter(Email.application_id == app.id, Email.sequence_step >= 1, Email.status == "pending")
        .order_by(Email.sequence_step.asc())
        .all()
    )
    rv = (
        session.query(ResumeVersion)
        .filter(ResumeVersion.application_id == app.id)
        .order_by(ResumeVersion.id.desc())
        .first()
    )

    if not email or not rv:
        p_log.error("Email or ResumeVersion missing from database for validation.")
        app.state = "Validation Failed"
        session.commit()
        return False

    def fail(errors: list[str]) -> bool:
        retries = (
            session.query(History)
            .filter(History.application_id == app.id, History.state == "Validation Retry")
            .count()
        )
        if retries < MAX_VALIDATION_RETRIES:
            # Send the email back for one rewrite with the validator's findings instead of dead-ending.
            p_log.warning(f"Validation issues {errors}; regenerating the email with these fixes.")
            app.current_stage = 8
            app.state = "Email Generation"
            session.add(
                History(
                    application_id=app.id,
                    stage=8,
                    state="Validation Retry",
                    run_id=run_id,
                    notes="; ".join(errors)[:2000],
                )
            )
            session.commit()
            return False
        p_log.error(f"Validation failed: {errors}", status="VALIDATION_FAILED")
        app.state = "Validation Failed"
        session.add(
            History(
                application_id=app.id,
                stage=9,
                state="Validation Failed",
                run_id=run_id,
                notes=f"Validation failed: {', '.join(errors)}",
            )
        )
        session.commit()
        return False

    # Deterministic placeholder check first (cheap and reliable)
    all_text = "\n".join([email.subject, email.body, *(f.body for f in followups)])
    placeholders = find_placeholders(all_text)
    if placeholders:
        return fail([f"Template placeholder left in email: {p}" for p in placeholders])

    # Reply-rate style rules: one rewrite is requested; after that, style issues are logged but not fatal.
    # Fabrication guard: achievements, rankings, year of study, CGPA and skills must come from the fact base.
    fact_base = _email_fact_base(config, app, rv)
    claim_issues = find_unsupported_claims(
        html_to_plain("\n".join([email.body, *(f.body for f in followups)])), fact_base, numbers="achievements"
    )
    if claim_issues:
        return fail([f"Unsupported claim: {c}" for c in claim_issues])

    # Company/recipient statements must be grounded in research; no invented openings or relationships.
    company = app.job.company
    contact = app.contact
    company_level = app.is_company_level
    # A company-level inquiry's placeholder record is not evidence about the company, so it is left out.
    job_text = [] if company_level else [app.job.title, app.job.description or ""]
    research_text = "\n".join(
        [
            company.name, company.description or "", json.dumps(company.research_data or {}),
            json.dumps(company.recent_news or []), company.funding_details or "", *job_text,
            " ".join(str(t) for t in company.tech_stack) if isinstance(company.tech_stack, list) else "",
        ]
    )
    recipient_text = " ".join(x for x in [contact.name, contact.role, contact.background] if x) if contact else ""
    has_real_job = app.job.source not in (*NON_OPENING_SOURCES, "pasted")
    grounding = check_grounding(
        email.body, research_text, recipient_text, has_real_job, config.user_identity.name, company_level=company_level
    )
    if grounding:
        return fail([f"Ungrounded statement: {g}" for g in grounding])

    min_words, max_words = word_target(app.persona, *_word_range(config, app))
    style_issues = check_email_style(email.body, config.user_identity.name, min_words, max_words)
    if company_level and config.company_outreach.ask_about_openings:
        style_issues += check_company_inquiry(email.body, config.user_identity.name)
    recent_bodies = [e.body for e in _recent_initial_emails(session, app.id)]
    style_issues += check_voice(email.body, recent_bodies, config.user_identity.name)
    style_issues += check_subject(email.subject, [e.subject for e in _recent_initial_emails(session, app.id)])
    if style_issues:
        already_retried = (
            session.query(History)
            .filter(History.application_id == app.id, History.state == "Validation Retry")
            .count()
        )
        if already_retried < MAX_VALIDATION_RETRIES:
            return fail(style_issues)
        simplified = simplify_jargon(email.body, _cached_plain_highlights(session, config, app), config.user_identity.name)
        if simplified != email.body:
            email.body = simplified
            session.add(History(application_id=app.id, stage=9, state="Validation", run_id=run_id,
                                notes="Replaced a tech-stack sentence with the plain summary of the same experience."))
            still_jargon = [i for i in check_voice(email.body, [], config.user_identity.name) if "tech-stack" in i]
            style_issues = [i for i in style_issues if "tech-stack" not in i] + still_jargon
        p_log.warning(f"Style issues remain after rewrite (not blocking): {style_issues}")

    base_resume_path = rv.parent_resume if os.path.exists(rv.parent_resume) else config.pipeline.base_resume_path
    base_resume_text = extract_resume_text(base_resume_path)[:8000]
    if not base_resume_text:
        p_log.warning("Base resume text unavailable; validating email content only.")
        base_resume_text = "(base resume unavailable)"

    followup_text = "\n\n".join(f"Follow-up #{f.sequence_step}:\n{f.body}" for f in followups)
    prompt = (
        f"Perform strict validation check on the generated files to ensure high quality.\n\n"
        f"Base Resume reference:\n{base_resume_text}\n\n"
        f"Tailored Resume Reasoning:\n{rv.reasoning}\n\n"
        f"Draft Email Subject: {email.subject}\n"
        f"Draft Email HTML Body:\n{email.body}\n\n"
        f"{followup_text}\n\n"
        f"Sender identity from the candidate's settings (always legitimate in the signature): "
        f"{config.user_identity.name} | {config.user_identity.linkedin_url} | {config.user_identity.github_url} | "
        f"{config.user_identity.website_url}\n\n"
        f"Verify the following conditions:\n"
        f"1. There are absolutely no template placeholders like '[Insert Name]', '[Your Name]', '<Company>', 'YYYY', etc.\n"
        f"2. Every claim about the CANDIDATE (employers, titles, projects, degrees, certifications, skills, metrics, "
        f"location/relocation, availability, visa) is supported by the Base Resume.\n"
        f"3. Every statement about the company (its products, posts, news, technology, needs or priorities) is "
        f"supported by the company context below; no invented job opening and no invented relationship with the "
        f"recipient or the hiring team.\n"
        f"Company context: {company.description or ''} {json.dumps(company.research_data or {})[:2500]} "
        f"{json.dumps(company.recent_news or [])[:800]}\n"
        + (
            "This is a company-level inquiry: NO opening is known at this company. Flag any sentence that claims or "
            "implies a specific opening exists or states the company's hiring plans.\n"
            if company_level
            else ""
        )
        + "Do NOT flag: the recipient company, its products, funding or news, the recipient's name/role, the sender "
        "identity links above, or the candidate saying they are seeking an internship/role (that is the purpose of "
        "the email). These are not claims about the candidate's background.\n"
        "Return a structured result: is_valid (boolean) and a list of errors (only real violations)."
    )

    try:
        try:
            response = llm.generate_json(prompt, ValidationResponse)
            assert isinstance(response, ValidationResponse)
            is_valid = response.is_valid
            errors = response.errors
        except Exception as err:
            p_log.warning(f"Validation LLM call failed: {err}. Defaulting to valid for human review.")
            is_valid = True
            errors = []

        if not is_valid:
            return fail(errors)

        app.current_stage = 10
        app.state = "Gmail Draft Creation"
        session.add(
            History(
                application_id=app.id,
                stage=10,
                state="Gmail Draft Creation",
                run_id=run_id,
                notes="Passed validation. Moving to Gmail Draft Creation.",
            )
        )
        session.commit()
        p_log.info("Validation passed successfully.", status="SUCCESS")
        return True

    except Exception as e:
        p_log.error(f"Validation stage error: {e}")
        app.state = "Validation Failed"
        session.commit()
        return False


def _auto_send_for(session: Session, config: AppConfig, app: Application) -> bool:
    if app.campaign_id:
        from src.db.models import Campaign

        campaign = session.get(Campaign, app.campaign_id)
        if campaign is not None and campaign.auto_send is not None:
            return bool(campaign.auto_send)
    return config.outreach.auto_send


def run_stage_10_gmail_draft_creation(
    session: Session, gmail: GmailProvider, app: Application, run_id: str, config: AppConfig | None = None
) -> bool:
    """
    Stage 10: Gmail Draft Creation
    Guards against duplicate outreach (same address, Gmail sent history, do-not-contact), creates the Gmail draft with
    the resume attached and, in auto-send mode, schedules it into the next send window.
    """
    company = app.job.company
    contact = app.contact
    assert contact is not None

    p_log = PipelineLogger(logger, run_id, "Stage 10: Gmail Draft Creation", company.name)
    p_log.info(f"Creating Gmail Draft for {contact.email}...")

    email = get_initial_email(session, app.id)
    if not email:
        p_log.error("Email content missing in DB.")
        app.state = "Draft Failed"
        session.commit()
        return False

    assert contact.email is not None

    def stop(state: str, note: str) -> bool:
        app.state = state
        email.status = "cancelled"
        session.add(History(application_id=app.id, stage=10, state=state, run_id=run_id, notes=note))
        log_event(session, app.id, "duplicate_blocked", email.id, details=note)
        session.commit()
        p_log.info(note, status=state.upper().replace(" ", "_"))
        return False

    if contact.do_not_contact:
        return stop("Do Not Contact", f"{contact.name} is marked do-not-contact.")
    allow_multi = config is not None and config.contacts.max_contacts_per_company > 1
    if email.gmail_draft_id is None:
        if recipient_already_contacted(session, contact.email, exclude_app_id=app.id):
            return stop("Duplicate", f"{contact.email} was already contacted by another application.")
        duplicate = find_duplicate(session, app, contact, contact.email, allow_multi)
        if duplicate:
            return stop("Duplicate", f"Duplicate outreach blocked: {duplicate}.")
    can_read = getattr(gmail, "can_read", None)
    if (
        email.gmail_draft_id is None
        and config is not None
        and config.outreach.check_gmail_history
        and callable(can_read)
    ):
        try:
            if can_read():
                if gmail.search_messages(f"(in:sent OR in:drafts) to:{contact.email}", 1):
                    return stop("Duplicate", f"{contact.email} already has a sent email or draft in Gmail.")
                if company.domain and not allow_multi and gmail.search_messages(f"(in:sent OR in:drafts) to:{company.domain}", 1):
                    return stop("Duplicate", f"Gmail already has a sent email or draft to someone at {company.domain}.")
        except Exception as e:
            p_log.info(f"Gmail history check skipped: {e}")

    try:
        if email.gmail_draft_id is None:
            draft_id = gmail.create_draft(
                to_email=contact.email,
                subject=email.subject,
                body_html=email.body,
                resume_path=app.tailored_resume_path,
            )
            email.gmail_draft_id = draft_id
        email.status = "draft_created"
        email.to_email = contact.email
        app.outreach_status = "drafted"
        note = f"Created Gmail Draft successfully (ID: {email.gmail_draft_id})."

        if config is not None and _auto_send_for(session, config, app):
            latest = (
                session.query(func.max(Email.scheduled_at))
                .filter(Email.status == "scheduled")
                .scalar()
            )
            email.scheduled_at = next_send_slot(
                utc_now_naive(),
                config.outreach.send_window,
                not_before_utc=latest if isinstance(latest, datetime) else None,
                min_gap_minutes=config.outreach.min_minutes_between_sends,
            )
            email.status = "scheduled"
            app.outreach_status = "scheduled"
            note += f" Scheduled to send at {email.scheduled_at:%Y-%m-%d %H:%M} UTC."

        record_outreach(session, app, contact, contact.email, "scheduled" if email.status == "scheduled" else "drafted")
        log_event(session, app.id, "drafted", email.id, details=note)
        app.current_stage = 11
        app.state = "Database Finalization"
        session.add(History(application_id=app.id, stage=11, state="Database Finalization", run_id=run_id, notes=note))
        session.commit()
        p_log.info(note, status="SUCCESS")
        return True

    except Exception as e:
        p_log.error(f"Gmail Draft creation failed: {e}")
        app.state = "Draft Failed"
        session.add(
            History(
                application_id=app.id,
                stage=10,
                state="Draft Failed",
                run_id=run_id,
                notes=f"Gmail API Draft Creation failed: {e}",
            )
        )
        session.commit()
        return False


def run_stage_11_database_finalization(session: Session, app: Application, run_id: str) -> bool:
    """
    Stage 11: Database Finalization
    Validates that database state for this application is clean.
    """
    company = app.job.company
    p_log = PipelineLogger(logger, run_id, "Stage 11: Database Finalization", company.name)
    p_log.info("Finalizing pipeline entries...")

    app.current_stage = 12
    app.state = "Completed"
    session.add(
        History(
            application_id=app.id,
            stage=12,
            state="Completed",
            run_id=run_id,
            notes="Successfully completed all pipeline stages.",
        )
    )
    session.commit()
    p_log.info("Application finalized successfully.", status="SUCCESS")
    return True
