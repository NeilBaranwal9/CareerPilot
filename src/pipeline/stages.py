import contextlib
import html
import json
import logging
import os
import re
import subprocess
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.config import AppConfig
from src.db.models import (
    Application,
    Company,
    Contact,
    Email,
    History,
    Job,
    ResumeVersion,
)
from src.intel.classify import classify_role, classify_sector, normalize_funding_stage, normalize_sector
from src.intel.learning import compute_outcome_stats, estimate_response_probability
from src.intel.scoring import FitResult, compute_company_fit, rule_based_opportunity, title_relevance, weighted_total
from src.outreach.engine import get_initial_email, log_event, recipient_already_contacted
from src.outreach.personas import (
    DEFAULT_FOLLOWUP_PROMPT,
    EMAIL_STYLE_RULES,
    check_email_style,
    describe_tone,
    fallback_followups,
    find_placeholders,
    first_name,
    guideline_for,
    html_to_plain,
    safe_format,
    tone_for,
)
from src.outreach.scheduling import next_send_slot, utc_now_naive
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
    ResumeTailorResponse,
    StructuredResumeSchema,
    ValidationResponse,
)
from src.providers.browser import BrowserProvider
from src.providers.gmail import GmailProvider
from src.providers.llm import BaseLLMProvider
from src.sources.ats import ats_job_to_dict, detect_ats_from_html, filter_relevant_jobs, probe_ats
from src.sources.companies import CompanyCandidate, DiscoverySpec, normalize_company_name, run_discovery
from src.sources.contacts import ContactCandidate, discover_contacts, rank_contacts
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
        f"Extract any open software engineering jobs that match these preferred roles: "
        f"{config.job_preferences.roles}. Look for experience requirements close to: "
        f"up to {config.job_preferences.experience_years_max} years (SDE-1, Entry Level, Graduate)."
    )
    response = llm.generate_json(prompt, JobListResponse)
    assert isinstance(response, JobListResponse)
    return [{**j.model_dump(), "source": "career_page"} for j in response.jobs]


def _discover_jobs_for_company(
    config: AppConfig, llm: BaseLLMProvider, browser: BrowserProvider, company: Company, p_log: PipelineLogger
) -> list[dict[str, Any]]:
    sources = config.discovery.job_sources
    prefs = config.job_preferences
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

    if "linkedin" in sources:
        try:
            location = next((g for g in prefs.geographies if g.lower() != "remote"), "India")
            role = prefs.roles[0] if prefs.roles else "Software Engineer"
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
            jobs.extend(search_wellfound_jobs(browser, company.name))
        except Exception as e:
            p_log.info(f"Wellfound job search skipped for {company.name}: {e}")

    if "indeed" in sources:
        try:
            jobs.extend(search_indeed_jobs(browser, company.name, prefs.roles[0] if prefs.roles else "software engineer"))
        except Exception as e:
            p_log.info(f"Indeed job search skipped for {company.name}: {e}")

    if "web_search" in sources and not any(j.get("source") not in ("wellfound", "indeed") for j in jobs):
        search_results = browser.search_google(f"'{company.name}' software engineer careers jobs", num_results=3)
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
        # The company is hiring even if none of its postings fit you (speculative outreach can still apply).
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


def _speculative_job(config: AppConfig, company: Company) -> dict[str, Any]:
    preferred_role = config.job_preferences.roles[0] if config.job_preferences.roles else "Software Engineer"
    return {
        "title": f"{preferred_role} (Speculative Application)",
        "url": f"speculative://{company.name.lower().replace(' ', '_')}",
        "location": config.job_preferences.geographies[0] if config.job_preferences.geographies else "Remote",
        "salary": None,
        "experience_years": config.job_preferences.experience_years_max,
        "description": f"Speculative outreach for a {preferred_role} position matching the company's tech stack and domain.",
        "source": "speculative",
    }


def run_stage_1_job_discovery(
    session: Session,
    config: AppConfig,
    llm: BaseLLMProvider,
    browser: BrowserProvider,
    companies: list[Company],
    run_id: str,
) -> list[Job]:
    """
    Stage 1: Job Discovery
    Finds open roles at each company from ATS boards (Greenhouse/Lever/Ashby/Workable/SmartRecruiters), LinkedIn,
    the careers page, Wellfound, Indeed and web search. Companies without postings get a speculative job
    (when allow_speculative_outreach is on) so they can still be contacted.
    """
    p_log = PipelineLogger(logger, run_id, "Stage 1: Job Discovery")
    p_log.info(f"Searching jobs for {len(companies)} companies...")

    all_jobs = []
    seen_urls = set()
    for company in companies:
        p_log.company = company.name

        cache = DBCache(session)
        cache_key = f"job_discovery_v2_{company.name.lower()}"
        cached_jobs = cache.get(cache_key)

        jobs_data: list[dict[str, Any]] = []
        if cached_jobs is not None:
            p_log.info(f"Found cached job listings for {company.name}")
            jobs_data = list(cached_jobs)
        else:
            jobs_data = _discover_jobs_for_company(config, llm, browser, company, p_log)
            cache.set(cache_key, jobs_data, config.pipeline.cache_lifetime_seconds)

        real_jobs = [j for j in jobs_data if j.get("source") != "speculative"]
        if real_jobs:
            company.hiring_status = "hiring"
            company.open_roles_count = max(company.open_roles_count or 0, len(real_jobs))
        elif company.hiring_status != "hiring":
            company.hiring_status = "no_public_openings"

        if not jobs_data and config.job_preferences.allow_speculative_outreach:
            speculative = _speculative_job(config, company)
            p_log.info(f"No active job listings found. Creating speculative job: '{speculative['title']}' for {company.name}")
            jobs_data = [speculative]

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

        app = Application(run_id=run_id, job_id=job.id, current_stage=2, state="Filtering", campaign_id=campaign_id)
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

        # 2. Keyword Exclusions in title
        is_keyword_excluded = any(ex_k.lower() in job.title.lower() for ex_k in config.exclusions.keywords)
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
        p_log.info(f"Passed filtering: {job.title} at {company.name}")

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
        cache.set(cache_key, research, config.pipeline.cache_lifetime_seconds)

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
                config.pipeline.cache_lifetime_seconds,
            )

    # Never re-target people who asked not to be contacted or were already emailed.
    blocked = {
        c.name
        for c in session.query(Contact).filter(Contact.company_id == company.id).all()
        if c.do_not_contact or c.last_contacted_at
    }
    candidates = [c for c in candidates if c.name not in blocked]

    stats = compute_outcome_stats(session, config)
    ranked = rank_contacts(candidates, company, config, stats)

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
            notes=f"Discovered email: {clean_email} [{result.status}, confidence {result.confidence:.2f}, via {result.source}]. Moving to Scoring.",
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
            f"Note: If this is a 'Speculative Application' (indicated in the title/description with no active public job listing), "
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
    app.response_probability = estimate_response_probability(
        company, stats, persona=app.persona, email_status=contact.email_status if contact else None
    )
    app.score = weighted_total(final, config)
    app.score_breakdown = {
        **final,
        "reasoning": reasoning,
        "mode": mode,
        "llm": llm_scores,
        "rules": rules,
        "company_fit": company.fit_score,
        "response_probability": app.response_probability,
    }
    session.commit()

    p_log.info(
        f"Weighted Score: {app.score:.2f} (Threshold: {config.scoring.thresholds.minimum_score}); "
        f"reply probability {app.response_probability:.1%}"
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
        role_name=job.title,
        company_name=company.name,
        variant_name=choice.name,
        variant_focus=choice.focus or "general software engineering",
        base_resume_text=original_text[:12000],
        job_description=(job.description or "Not available (speculative outreach)")[:3000],
        company_description=company.description or "",
        company_sector=company.sector or "unknown",
        tech_stack=", ".join(str(t) for t in stack or []),
        highlights=format_highlights(highlights),
    )
    try:
        tailored = llm.generate_json(prompt, StructuredResumeSchema)
        assert isinstance(tailored, StructuredResumeSchema)
    except Exception as e:
        p_log.warning(f"Structured resume tailoring failed ({e}); attaching the original resume.")
        return _record_resume(
            session, app, run_id, p_log, source_path, source_path, choice.name, [],
            f"Tailoring failed ({e}); attached original {choice.name} resume.", highlight_names,
        )

    violations = check_tailored_resume(original_text, tailored)
    if violations:
        p_log.warning(f"Tailored resume rejected by fabrication guard: {violations[:3]}")
        return _record_resume(
            session, app, run_id, p_log, source_path, source_path, choice.name, [],
            f"Tailored version rejected ({'; '.join(violations[:3])}); attached original {choice.name} resume.",
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
            "role_name": job.title,
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
    if link_parts:
        signature_links = "<br>" + " | ".join(link_parts)
        already_has_links = any(url in body_html for url in [linkedin, github] if url)
        if not already_has_links:
            if user_name and user_name in body_html:
                body_html = body_html.replace(user_name, f"{user_name}{signature_links}", 1)
            else:
                body_html += f"<p>{signature_links}</p>"
    return body_html


def run_stage_8_email_generation(
    session: Session,
    config: AppConfig,
    llm: BaseLLMProvider,
    app: Application,
    run_id: str,
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
        "role_name": job.title,
        "product_description": research.get("business_model", "their innovative platform"),
        "tech_stack": ", ".join(stack or research.get("tech_stack", ["modern tools"])),
        "recent_launches": _format_news(company.recent_news) if company.recent_news else research.get("funding", "recent engineering progress"),
        "tailored_skills": tailored_skills,
        "company_description": company.description or research.get("business_model", ""),
        "recent_news": _format_news(company.recent_news),
        "job_description": (job.description or "Not available (speculative outreach)")[:2000],
        "contact_background": contact.background or "Not available",
        "highlights": format_highlights(highlights[:4]),
        "resume_summary": resume_text[:6000] or "Not available",
        "persona_guidelines": guideline_for(persona),
        "tone": describe_tone(tone),
        "user_name": config.user_identity.name,
    }
    prompt = safe_format(config.prompts.email_generation, **values)
    if "{job_description}" not in config.prompts.email_generation:
        prompt += safe_format(EMAIL_CONTEXT_BLOCK, **values)
    prompt += safe_format(EMAIL_STYLE_RULES, first_name=first_name(contact.name))
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
            p_log.warning(f"LLM email generation failed: {err}. Falling back to standard professional email.")
            subject = f"Software Engineering Opportunities - {company.name}"
            user_name = config.user_identity.name
            body_html = (
                f"<p>Hi {html.escape(contact.name)},</p>"
                f"<p>I hope you are doing well.</p>"
                f"<p>I am reaching out because I am very interested in software engineering roles at {html.escape(company.name)}. "
                f"My background includes {html.escape(tailored_skills)}.</p>"
                f"<p>I have attached my resume for your review. I would love to connect and chat about how my background might align with your team's goals.</p>"
                f"<p>Best regards,<br>{html.escape(user_name)}</p>"
            )

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
                fu_response = llm.generate_json(fu_prompt, FollowUpSequenceSchema)
                assert isinstance(fu_response, FollowUpSequenceSchema)
                followup_bodies = [f.body_html for f in fu_response.followups if f.body_html][: len(steps)]
            except Exception as err:
                p_log.info(f"Follow-up generation via LLM unavailable ({err}); using follow-up templates.")
            if len(followup_bodies) < len(steps):
                top_highlight = highlights[0].summary or highlights[0].name if highlights else None
                templates = fallback_followups(
                    contact.name, company.name, job.title, config.user_identity.name, top_highlight, len(steps)
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
    style_issues = check_email_style(email.body, config.user_identity.name)
    if style_issues:
        already_retried = (
            session.query(History)
            .filter(History.application_id == app.id, History.state == "Validation Retry")
            .count()
        )
        if already_retried < MAX_VALIDATION_RETRIES:
            return fail(style_issues)
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
        f"Do NOT flag: the recipient company, its products, funding or news, the recipient's name/role, the sender "
        f"identity links above, or the candidate saying they are seeking an internship/role (that is the purpose of "
        f"the email). These are not claims about the candidate's background.\n"
        f"Return a structured result: is_valid (boolean) and a list of errors (only real violations)."
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
    if email.gmail_draft_id is None and recipient_already_contacted(session, contact.email, exclude_app_id=app.id):
        return stop("Duplicate", f"{contact.email} was already contacted by another application.")
    can_read = getattr(gmail, "can_read", None)
    if (
        email.gmail_draft_id is None
        and config is not None
        and config.outreach.check_gmail_history
        and callable(can_read)
    ):
        try:
            if can_read() and gmail.search_messages(f"in:sent to:{contact.email}", 1):
                return stop("Duplicate", f"Already emailed {contact.email} from Gmail (found in Sent mail).")
        except Exception as e:
            p_log.info(f"Gmail sent-history check skipped: {e}")

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
