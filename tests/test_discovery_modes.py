"""Discovery modes (job_search / company_outreach / hybrid), role-aware job search and company-level inquiries."""

import logging

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import load_config
from src.db.models import Application, Base, Company, Contact, Email, History, Job, ResumeVersion, Run
from src.intel.classify import classify_role, describe_role_families, expand_sectors, role_families, role_search_terms
from src.intel.scoring import title_relevance
from src.outreach.voice import check_company_inquiry, check_grounding
from src.pipeline.campaign import parse_goal, resolve_mode
from src.pipeline.schemas import EmailGenResponse, JobListResponse, ValidationResponse
from src.pipeline.stages import (
    run_opening_check,
    run_stage_1_job_discovery,
    run_stage_2_filtering,
    run_stage_8_email_generation,
    run_stage_9_validation,
)
from src.sources.companies import CompanyCandidate, DiscoverySpec, infer_outreach_mode, matches_spec
from src.sources.contacts import ContactCandidate, from_web_sources, personas_for_families, rank_contacts
from src.sources.job_boards import search_wellfound_jobs

PRODUCT_DATA_ROLES = ["Product Analyst Intern", "Associate Product Manager Intern", "Data Analyst Intern",
                      "Data Analytics Intern"]
ENGINEERING_ROLES = ["Software Engineer Intern", "Backend Engineer Intern"]
MIXED_ROLES = ["Product Analyst Intern", "Data Analyst Intern", "QA Engineer Intern", "SDET Intern",
               "Software Engineer Intern", "Backend Engineer Intern", "AI/ML Intern"]


def para(*paragraphs: str) -> str:
    return "".join(f"<p>{p}</p>" for p in paragraphs) + "<p>Best,<br>Jane Doe</p>"


class FakeBrowser:
    """Records every search query; careers pages exist but contain no ATS board."""

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.fetched: list[str] = []

    def search_google(self, query, num_results=5, include_blocked=False):
        self.queries.append(query)
        return []

    def fetch_page(self, url, use_playwright=False):
        self.fetched.append(url)
        return "<html><body>Careers</body></html>"

    def extract_text(self, html):
        return "Careers page text"


class JobsLLM:
    """Careers-page extraction: OpenCo lists a Product Analyst internship, every other company lists nothing."""

    def generate_json(self, prompt, schema, system_prompt=None):
        assert schema is JobListResponse
        if "OpenCo" in prompt:
            return JobListResponse(jobs=[{
                "title": "Product Analyst Intern", "url": "https://openco.in/careers/pa-intern", "location": "Bengaluru",
                "salary": None, "experience_years": None, "description": "Analyse product funnels with SQL.",
            }])
        return JobListResponse(jobs=[])


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _config(roles=None, mode=None):
    cfg = load_config("config.example.yaml")
    cfg.job_preferences.roles = list(roles or PRODUCT_DATA_ROLES)
    cfg.discovery.job_sources = ["career_page", "wellfound", "indeed", "web_search"]
    if mode:
        cfg.discovery.mode = mode
    return cfg


def _companies(session, *names):
    companies = [Company(name=n, domain=f"{n.lower()}.in", employee_count=300) for n in names]
    session.add_all(companies)
    session.commit()
    return companies


# ---------------------------------------------------------------------------
# Role-aware job-search queries
# ---------------------------------------------------------------------------


def test_product_and_data_roles_do_not_generate_engineer_queries():
    assert role_search_terms(PRODUCT_DATA_ROLES) == ["product analyst", "data analyst"]
    browser = FakeBrowser()
    search_wellfound_jobs(browser, "Weekday", role_terms=role_search_terms(PRODUCT_DATA_ROLES))
    assert browser.queries == ['site:wellfound.com "Weekday" jobs product analyst data analyst']
    assert "engineer" not in browser.queries[0]


def test_engineering_roles_still_generate_engineering_queries():
    assert role_search_terms(ENGINEERING_ROLES) == ["software engineer"]
    assert role_search_terms(["QA Engineer Intern", "SDET Intern"]) == ["qa engineer"]


def test_different_role_configurations_produce_different_short_queries():
    assert role_search_terms(MIXED_ROLES) == ["product analyst", "data analyst"]  # first two families only
    assert role_search_terms(["Business Analyst Intern", "Business Intelligence Intern"]) == [
        "business analyst", "business intelligence",
    ]
    assert role_families(MIXED_ROLES) == ["product", "data", "qa", "software", "ai_ml"]
    assert describe_role_families(["product", "data", "qa"]) == "product, data/analytics or QA/testing"
    # no recognisable role -> no role words at all (never a default "engineer")
    assert role_search_terms(["Summer Intern"]) == [] and role_search_terms([]) == []


@pytest.mark.parametrize(
    ("roles", "expected", "forbidden"),
    [(PRODUCT_DATA_ROLES, "product analyst data analyst", "engineer"), (ENGINEERING_ROLES, "software engineer", "product")],
)
def test_stage1_queries_follow_configured_roles(roles, expected, forbidden):
    session, browser = _db(), FakeBrowser()
    session.add(Run(id="R"))
    run_stage_1_job_discovery(session, _config(roles), JobsLLM(), browser, _companies(session, "QuietCo"), "R")
    assert browser.queries, "wellfound/indeed/web search should run in job_search mode"
    assert all(expected in q for q in browser.queries)
    assert not any(forbidden in q.lower() for q in browser.queries)


def test_title_matching_uses_role_families_not_engineering_by_default():
    # a product/data-only configuration no longer treats engineering postings as matches
    assert title_relevance("Software Engineer Intern", PRODUCT_DATA_ROLES, 1.0) < 0.5
    assert title_relevance("Business Analyst - Intern", ["Business Analyst Intern"], 1.0) >= 0.5
    # engineering configurations behave exactly as before
    assert title_relevance("Platform Engineer Intern", ENGINEERING_ROLES, 1.0) >= 0.5


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def test_job_search_skips_companies_without_a_matching_opening(caplog):
    session = _db()
    session.add(Run(id="R"))
    cfg = _config()
    cfg.job_preferences.allow_speculative_outreach = True  # legacy flag must not bring the fake job back
    with caplog.at_level(logging.INFO):
        jobs = run_stage_1_job_discovery(session, cfg, JobsLLM(), FakeBrowser(), _companies(session, "OpenCo", "QuietCo"), "R")
    assert [(j.company.name, j.title) for j in jobs] == [("OpenCo", "Product Analyst Intern")]
    titles = [j.title for j in session.query(Job).all()]
    assert titles == ["Product Analyst Intern"]
    assert not any("Speculative" in t for t in titles)
    assert "[DISCOVERY] Mode: job_search" in caplog.text
    assert "[OUTREACH] No matching public opening found for QuietCo" in caplog.text
    assert "[OUTREACH] Matching opening found -> job-specific outreach" in caplog.text


def test_company_outreach_needs_no_public_opening(caplog):
    session, browser = _db(), FakeBrowser()
    session.add(Run(id="R"))
    cfg = _config(mode="company_outreach")
    with caplog.at_level(logging.INFO):
        jobs = run_stage_1_job_discovery(session, cfg, JobsLLM(), browser, _companies(session, "QuietCo"), "R")
    assert len(jobs) == 1
    job = jobs[0]
    assert job.source == "company_outreach" and job.title == "Company-level internship inquiry"
    assert "No matching public opening" in (job.description or "")
    assert not any(role.lower() in job.title.lower() for role in PRODUCT_DATA_ROLES)
    assert browser.queries == []  # only the company's own careers page is checked, no job-board searches
    assert "Company-level outreach enabled; public job opening not required." in caplog.text
    assert "[OUTREACH] Creating company-level speculative outreach" in caplog.text

    apps = run_stage_2_filtering(session, cfg, jobs, "R")
    assert len(apps) == 1 and apps[0].outreach_type == "company_speculative" and apps[0].is_company_level
    assert apps[0].state == "Company Research"


def test_company_outreach_checks_openings_only_after_contacts_are_found():
    session, browser = _db(), FakeBrowser()
    session.add(Run(id="R"))
    cfg = _config(mode="company_outreach")
    jobs = run_stage_1_job_discovery(session, cfg, JobsLLM(), browser, _companies(session, "OpenCo", "QuietCo"), "R")
    # company discovery is independent of jobs: nothing is searched or fetched up front
    assert browser.queries == [] and browser.fetched == []
    assert {j.source for j in jobs} == {"company_outreach"}
    apps = {a.job.company.name: a for a in run_stage_2_filtering(session, cfg, jobs, "R")}
    # a second thread at OpenCo (max_contacts_per_company: 2) shares the same company-level target
    sibling = Application(run_id="R", job_id=apps["OpenCo"].job_id, current_stage=6, state="Opportunity Scoring",
                          outreach_type="company_speculative")
    session.add(sibling)
    session.commit()

    assert run_opening_check(session, cfg, JobsLLM(), browser, apps["OpenCo"], "R") is True
    for app in (apps["OpenCo"], sibling):
        assert app.outreach_type == "job" and not app.is_company_level
        assert (app.job.title, app.job.source) == ("Product Analyst Intern", "career_page")
    assert browser.queries == []  # only the careers page / ATS board, no job-board searches

    assert run_opening_check(session, cfg, JobsLLM(), browser, apps["QuietCo"], "R") is False
    assert apps["QuietCo"].is_company_level and apps["QuietCo"].job.title == "Company-level internship inquiry"
    note = session.query(History).filter(History.application_id == apps["QuietCo"].id, History.state == "Opening Check").one()
    assert "company-level internship inquiry" in note.notes


def test_opening_check_can_be_turned_off():
    session, browser = _db(), FakeBrowser()
    session.add(Run(id="R"))
    cfg = _config(mode="company_outreach")
    cfg.company_outreach.check_openings = False
    jobs = run_stage_1_job_discovery(session, cfg, JobsLLM(), browser, _companies(session, "OpenCo"), "R")
    app = run_stage_2_filtering(session, cfg, jobs, "R")[0]
    assert run_opening_check(session, cfg, JobsLLM(), browser, app, "R") is False
    assert app.is_company_level and browser.fetched == []


def test_company_outreach_pipeline_contacts_first_then_uses_a_real_opening(tmp_path):
    from tests.test_campaign import make_runner

    runner, session = make_runner(tmp_path, discovery__mode="company_outreach")
    runner.run()
    app = session.query(Application).one()
    # the mock careers page lists "Software Engineer", which matches the example roles
    assert app.state == "Completed" and app.outreach_type == "job" and app.job.title == "Software Engineer"
    history = session.query(History).filter(History.application_id == app.id).order_by(History.id).all()
    states = [h.state for h in history]
    assert states.index("Professional Email Discovery") < states.index("Opening Check")
    assert app.contact.email and session.query(Email).filter(Email.application_id == app.id).count() >= 1


def test_hybrid_prefers_real_openings_and_falls_back_to_company_level(caplog):
    session = _db()
    session.add(Run(id="R"))
    cfg = _config(mode="hybrid")
    with caplog.at_level(logging.INFO):
        jobs = run_stage_1_job_discovery(session, cfg, JobsLLM(), FakeBrowser(), _companies(session, "OpenCo", "QuietCo"), "R")
    by_company = {j.company.name: j for j in jobs}
    assert by_company["OpenCo"].title == "Product Analyst Intern" and by_company["OpenCo"].source == "career_page"
    assert by_company["QuietCo"].source == "company_outreach"
    assert "[OUTREACH] Hybrid mode -> falling back to company-level outreach" in caplog.text

    apps = {a.job.company.name: a for a in run_stage_2_filtering(session, cfg, jobs, "R")}
    assert apps["OpenCo"].outreach_type == "job" and not apps["OpenCo"].is_company_level
    assert apps["QuietCo"].outreach_type == "company_speculative"
    # re-running does not create a second thread for the same company
    again = run_stage_1_job_discovery(session, cfg, JobsLLM(), FakeBrowser(), [by_company["QuietCo"].company], "R")
    run_stage_2_filtering(session, cfg, again, "R")
    assert session.query(Application).count() == 2


def test_invalid_mode_is_rejected_and_default_is_job_search():
    cfg = load_config("config.example.yaml")
    assert cfg.discovery.mode == "job_search" and not cfg.discovery.mode_is_explicit()
    assert cfg.company_outreach.ask_about_openings is True
    with pytest.raises(ValueError):
        type(cfg.discovery)(mode="speculative")


# ---------------------------------------------------------------------------
# Recipients
# ---------------------------------------------------------------------------


def test_company_level_recipients_follow_target_role_families():
    mid = Company(name="MidCo", employee_count=300)
    assert personas_for_families(["product", "data"], mid)[:4] == ["product_lead", "product_manager", "recruiter", "data_lead"]
    assert personas_for_families(["software"], mid)[:3] == ["engineering_manager", "vp_engineering", "hiring_manager"]
    assert personas_for_families(["qa"], mid)[:3] == ["qa_lead", "engineering_manager", "recruiter"]
    assert personas_for_families(["product"], Company(name="Tiny", employee_count=12))[0] == "founder"
    big_bank = Company(name="BigBank", employee_count=50000)
    assert personas_for_families(["software", "data"], big_bank)[:3] == ["recruiter", "engineering_manager", "vp_engineering"]
    assert personas_for_families([], mid) == ["hiring_manager", "recruiter"]


def test_rank_contacts_prefers_role_family_personas_only_when_asked():
    cfg = load_config("config.example.yaml")
    company = Company(name="MidCo", employee_count=300)

    def candidates():
        return [
            ContactCandidate(name="Eve Manager", title="Engineering Manager", source="team_page", confidence=0.8,
                             role_category="engineering_manager"),
            ContactCandidate(name="Pat Lead", title="Head of Product", source="blog", confidence=0.65,
                             role_category="product_lead"),
        ]

    assert rank_contacts(candidates(), company, cfg)[0].name == "Eve Manager"  # unchanged job_search ranking
    preferred = personas_for_families(["product"], company)
    assert rank_contacts(candidates(), company, cfg, preferred_personas=preferred)[0].name == "Pat Lead"
    cfg.contacts.persona_strategy = "ordered"  # an explicit order is respected as-is
    assert rank_contacts(candidates(), company, cfg, preferred_personas=preferred)[0].name == "Eve Manager"


def test_large_companies_are_searched_for_campus_recruiters():
    for size, expected in ((50000, True), (200, False)):
        browser = FakeBrowser()
        from_web_sources(browser, None, Company(name="BigBank", employee_count=size), ["press"])
        assert any("campus recruiting" in q for q in browser.queries) is expected
    assert classify_role("Head of Early Careers")[0] == "recruiter"


# ---------------------------------------------------------------------------
# Company discovery (independent of jobs)
# ---------------------------------------------------------------------------


def test_fintech_goal_covers_the_whole_financial_space():
    assert expand_sectors(["fintech"]) == ["fintech", "insurtech", "trading"]
    assert expand_sectors(["ai", "fintech"]) == ["ai", "fintech", "insurtech", "trading"]
    spec = DiscoverySpec(query="Find 200 fintech companies in India", sectors=["fintech"], geographies=["India"])
    assert matches_spec(CompanyCandidate(name="Acko", sector="insurtech", description="Digital insurance"), spec)
    assert matches_spec(CompanyCandidate(name="Zerodha", sector="trading", description="Stock broking"), spec)
    assert not matches_spec(CompanyCandidate(name="Practo", sector="healthtech", description="Doctor booking"), spec)


def test_llm_company_discovery_asks_for_startups_and_established_firms_regardless_of_openings():
    from src.pipeline.schemas import CompanyListResponse
    from src.sources.companies import discover_from_llm

    prompts: list[str] = []

    class ListLLM:
        def generate_json(self, prompt, schema, system_prompt=None):
            prompts.append(prompt)
            return CompanyListResponse(companies=[])

    spec = DiscoverySpec(query="Find 200 fintech companies in India", sectors=["fintech"], geographies=["India"])
    discover_from_llm(ListLLM(), spec, [], 40)
    assert "startups and established companies" in prompts[0] and "established banks" in prompts[0]
    assert "whether or not they currently advertise roles" in prompts[0]
    assert "junior engineers" not in prompts[0]


# ---------------------------------------------------------------------------
# Emails
# ---------------------------------------------------------------------------


class CaptureLLM:
    def __init__(self, body: str) -> None:
        self.prompts: list[str] = []
        self.body = body

    def generate_json(self, prompt, schema, system_prompt=None):
        self.prompts.append(prompt)
        if schema is ValidationResponse:
            return ValidationResponse(is_valid=True, errors=[])
        return EmailGenResponse(subject="Refunds and product analytics", body_html=self.body)


def _app(session, tmp_path, company_level: bool) -> Application:
    session.add(Run(id="R"))
    company = Company(name="PayCo", domain="payco.in", description="UPI payment infrastructure for merchants",
                      recent_news=[{"title": "PayCo launches instant refunds"}], research_data={"products": ["Refunds API"]})
    session.add(company)
    session.flush()
    if company_level:
        job = Job(company_id=company.id, title="Company-level internship inquiry", url="company-outreach://payco_in",
                  source="company_outreach", description="No matching public opening was found.")
    else:
        job = Job(company_id=company.id, title="Product Analyst Intern", url="https://payco.in/jobs/1",
                  source="career_page", description="Analyse refund funnels.")
    contact = Contact(company_id=company.id, name="Priya Sharma", role="Head of Product", role_category="product_lead",
                      email="priya@payco.in")
    session.add_all([job, contact])
    session.flush()
    app = Application(run_id="R", job_id=job.id, contact_id=contact.id, current_stage=8, state="Email Generation",
                      persona="product_lead", outreach_type="company_speculative" if company_level else "job")
    session.add(app)
    session.flush()
    resume = tmp_path / "r.typ"
    resume.write_text("= Jane Doe\nBuilt dashboards for payment refunds at Acme.")
    session.add(ResumeVersion(application_id=app.id, parent_resume=str(resume), company="PayCo", role=job.title,
                              reasoning="attach base", path=str(resume)))
    session.commit()
    return app


def test_company_level_email_prompt_asks_about_opportunities_without_a_role(tmp_path):
    session = _db()
    cfg = load_config("config.example.yaml")
    cfg.job_preferences.roles = PRODUCT_DATA_ROLES
    cfg.outreach.followups = []
    llm = CaptureLLM(para("Hi Priya,", "I noticed PayCo launched instant refunds.", "Is the team taking interns?"))
    app = _app(session, tmp_path, company_level=True)
    assert run_stage_8_email_generation(session, cfg, llm, app, "R") is True
    prompt = llm.prompts[0]
    assert "about internship opportunities in product or data/analytics" in prompt
    assert "company-level inquiry" in prompt and "No matching public opening was found at PayCo" in prompt
    assert "current or upcoming internship opportunities" in prompt
    assert "About 100-150 words" in prompt
    assert "Only ONE experience or project".lower() in prompt.lower()
    assert "Product Analyst Intern" not in prompt and "Speculative Application" not in prompt


def test_job_specific_email_prompt_names_the_discovered_role(tmp_path):
    session = _db()
    cfg = load_config("config.example.yaml")
    cfg.outreach.followups = []
    llm = CaptureLLM(para("Hi Priya,", "I saw the Product Analyst Intern opening on your careers page."))
    app = _app(session, tmp_path, company_level=False)
    assert run_stage_8_email_generation(session, cfg, llm, app, "R") is True
    assert "about Product Analyst Intern" in llm.prompts[0] and "company-level inquiry" not in llm.prompts[0]


def test_company_level_grounding_rejects_claimed_openings_and_hiring_plans():
    research = "PayCo builds UPI payment infrastructure. Recent: launched instant refunds."
    claims = para(
        "Hi Priya,",
        "I'm applying for the Product Analyst Intern role at PayCo.",
        "I saw your opening for a data analyst.",
        "I know you're growing the analytics team.",
    )
    issues = " ".join(check_grounding(claims, research, "", False, "Jane Doe", company_level=True))
    assert "applying for the Product Analyst Intern role" in issues
    assert "your opening" in issues and "growing" in issues

    inquiry = para("Hi Priya,", "I noticed PayCo launched instant refunds.",
                   "Do you have any current or upcoming internship opportunities in product or data?")
    assert check_grounding(inquiry, research, "", False, "Jane Doe", company_level=True) == []
    assert check_company_inquiry(inquiry, "Jane Doe") == []
    assert check_company_inquiry(para("Hi Priya,", "I noticed PayCo launched instant refunds."), "Jane Doe")
    # job-specific outreach may mention the actual discovered opening
    posting = para("Hi Priya,", "I saw the Product Analyst Intern opening on your careers page.")
    assert check_grounding(posting, research + " Product Analyst Intern", "", True, "Jane Doe") == []


def test_validation_sends_a_company_level_email_that_claims_an_opening_back_for_rewrite(tmp_path):
    session = _db()
    cfg = load_config("config.example.yaml")
    cfg.outreach.followups = []
    body = para("Hi Priya,", "I noticed PayCo launched instant refunds.", "I'm applying for the Product Analyst role at PayCo.")
    app = _app(session, tmp_path, company_level=True)
    session.add(Email(application_id=app.id, subject="Refunds", body=body, status="generated", sequence_step=0))
    session.commit()
    assert run_stage_9_validation(session, cfg, CaptureLLM(body), app, "R") is False
    retry = session.query(History).filter(History.state == "Validation Retry").one()
    assert "applying for the Product Analyst role" in retry.notes and app.current_stage == 8


def test_company_outreach_run_ends_in_a_gmail_draft_with_zero_openings(tmp_path):
    from tests.test_campaign import make_runner
    from tests.test_pipeline import MockLLMProvider

    class NoOpeningsLLM(MockLLMProvider):
        def generate_json(self, prompt, schema, system_prompt=None):
            if schema is JobListResponse:
                return JobListResponse(jobs=[])
            return super().generate_json(prompt, schema, system_prompt)

    runner, session = make_runner(tmp_path, discovery__mode="company_outreach")
    runner.llm = NoOpeningsLLM()
    runner.run()
    app = session.query(Application).one()
    assert app.is_company_level and app.state == "Completed" and app.job.title == "Company-level internship inquiry"
    email = session.query(Email).filter(Email.application_id == app.id, Email.sequence_step == 0).one()
    # drafts only: auto_send stays false, nothing is sent
    assert email.status == "draft_created" and email.gmail_draft_id == "draft_abc123" and email.sent_at is None


# ---------------------------------------------------------------------------
# Campaign goals
# ---------------------------------------------------------------------------


def test_campaign_goal_selects_company_outreach(tmp_path):
    from src.db.models import Campaign
    from tests.test_campaign import make_runner

    runner, session = make_runner(tmp_path)
    progress = runner.start_campaign("Find 3 SaaS companies in India and ask if they have internship opportunities for me")
    campaign = session.query(Campaign).one()
    assert campaign.spec["mode"] == "company_outreach" and progress["mode"] == "company_outreach"
    assert campaign.spec["sectors"] == ["saas"] and campaign.spec["geographies"] == ["India"]
    assert session.query(Application).count() == 1


def test_goal_wording_infers_the_mode():
    assert infer_outreach_mode("Find fintech startups in India and ask if they have internship opportunities for me.") == "company_outreach"
    assert infer_outreach_mode(
        "Find fintech startups in India, use actual openings when available, otherwise ask companies about internship "
        "opportunities."
    ) == "hybrid"
    assert infer_outreach_mode("Find 200 fintech companies in India, reach engineering managers") == ""
    spec = parse_goal(None, "Find fintech startups in India and ask if they have internship opportunities for me.")
    assert spec.mode == "company_outreach" and spec.search_phrase() == "fintech startups in India"
    assert spec.sectors == ["fintech"] and spec.geographies == ["India"]
    assert DiscoverySpec.from_dict({"query": "old campaign"}).mode == ""  # campaigns saved before modes existed


def test_explicit_configuration_takes_precedence_over_the_goal():
    cfg = load_config("config.example.yaml")
    assert resolve_mode(cfg, "company_outreach") == ("company_outreach", "inferred from the goal")
    assert resolve_mode(cfg, "") == ("job_search", "default")
    cfg.discovery.mode = "hybrid"  # as if set in config.yaml
    mode, reason = resolve_mode(cfg, "company_outreach")
    assert mode == "hybrid" and "config.yaml" in reason and "company_outreach" in reason
    assert resolve_mode(cfg, "company_outreach", "job_search") == ("job_search", "set with --mode")
    with pytest.raises(ValueError):
        resolve_mode(cfg, "", "speculative")
