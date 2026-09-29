import json
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import LLMConfig, load_config
from src.db.models import Application, Base, Company, Contact, Email, History, Job, LLMUsage, OutreachLedger, Run
from src.intel.classify import classify_role
from src.intel.learning import OutcomeStats, explain_reply_probability
from src.outreach.dedupe import backfill_ledger, find_duplicate, record_outreach, role_key
from src.pipeline.runner import PipelineRunner
from src.pipeline.schemas import EmailGenResponse
from src.pipeline.stages import run_stage_8_email_generation
from src.providers.browser import BrowserProvider, FetchError
from src.providers.llm.ollama import OllamaProvider, OllamaUnavailableError
from src.providers.llm.router import BudgetExceededError, LLMRouter, UsageTracker
from src.sources.contacts import (
    _llm_extract,
    discover_site_links,
    name_in_text,
    normalize_personas,
    parse_theorg_result,
)
from src.utils.claims import find_unsupported_claims
from tests.test_pipeline import MockBrowserProvider, MockGmailProvider, MockLLMProvider


class Item(BaseModel):
    sector: str
    stage: str | None = None


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------


def test_ollama_structured_output_and_usage():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"message": {"content": '{"sector": "fintech", "stage": "series_b"}'},
                                         "prompt_eval_count": 120, "eval_count": 18})

    llm = OllamaProvider(model="qwen3:8b", num_ctx=8192, transport=httpx.MockTransport(handler))
    result = llm.generate_json("classify", Item)
    assert result.sector == "fintech"
    assert seen[0]["format"]["properties"]["sector"]["type"] == "string"  # schema-constrained output
    assert seen[0]["think"] is False and seen[0]["options"]["num_ctx"] == 8192 and seen[0]["stream"] is False
    assert llm.last_usage == {"prompt_tokens": 120, "completion_tokens": 18, "model": "qwen3:8b"}


def test_ollama_retries_without_think_and_reports_missing_model():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": "model does not support thinking"})
        return httpx.Response(200, json={"message": {"content": "hello"}})

    llm = OllamaProvider(transport=httpx.MockTransport(handler))
    assert llm.generate_text("hi") == "hello" and "think" not in calls[-1]

    missing = OllamaProvider(transport=httpx.MockTransport(lambda _r: httpx.Response(404, json={"error": "not found"})))
    with pytest.raises(OllamaUnavailableError, match="ollama pull"):
        missing.generate_text("hi")


# ---------------------------------------------------------------------------
# Router & budget
# ---------------------------------------------------------------------------


class FakeProvider(MockLLMProvider):
    def __init__(self, name: str, provider_name: str, fail: Exception | None = None):
        self.model = name
        self.provider_name = provider_name
        self.fail = fail
        self.calls: list[str] = []
        self.last_usage = None

    def generate_json(self, prompt, schema, system_prompt=None):
        self.calls.append(schema.__name__)
        if self.fail:
            raise self.fail
        self.last_usage = {"prompt_tokens": len(prompt) // 4, "completion_tokens": 50, "model": self.model}
        return super().generate_json(prompt, schema, system_prompt)

    def generate_text(self, prompt, system_prompt=None):
        self.last_usage = None  # forces token estimation
        return "ok"


def _router(local=None, budget=190_000, reserve=120_000, fallback="premium_fast"):
    cfg = LLMConfig(provider="groq", model="big", fast_model="fast", local_provider="ollama",
                    groq_daily_token_budget=budget, groq_reserved_for_emails=reserve, local_fallback=fallback)
    tracker = UsageTracker(None)
    premium, fast = FakeProvider("big", "groq"), FakeProvider("fast", "groq")
    local = local or FakeProvider("qwen3:8b", "ollama")
    return LLMRouter(cfg, tracker, premium=premium, premium_fast=fast, local=local), tracker, premium, fast, local


def test_router_sends_only_email_writing_to_groq():
    router, tracker, premium, fast, local = _router()
    router.get("research").generate_json("x", EmailGenResponse)
    router.get("contact_finding").generate_json("x", EmailGenResponse)
    router.get("email_generation").generate_json("x", EmailGenResponse)
    router.get("followup_generation").generate_json("x", EmailGenResponse)
    assert len(local.calls) == 2 and len(premium.calls) == 1 and len(fast.calls) == 1
    tiers = {r["task"]: r["tier"] for r in tracker.pending}
    assert tiers == {"research": "local", "contact_finding": "local", "email_generation": "premium",
                     "followup_generation": "premium_fast"}
    assert router.get("scoring").generate_text("hello") == "ok"
    assert tracker.pending[-1]["estimated"] is True and tracker.pending[-1]["total_tokens"] > 0


def test_local_outage_falls_back_within_non_email_budget():
    down = FakeProvider("qwen3:8b", "ollama", fail=OllamaUnavailableError("down"))
    router, tracker, premium, fast, _ = _router(local=down, budget=1000, reserve=900)
    router.get("research").generate_json("x" * 400, EmailGenResponse)  # falls back to Groq fast (150 tokens)
    assert fast.calls and tracker.pending[-1]["tier"] == "premium_fast"
    # Non-email share (1000 - 900 = 100 tokens) is now used up; research must not eat the email reserve.
    with pytest.raises(BudgetExceededError, match="non-email share"):
        router.get("research").generate_json("x" * 100, EmailGenResponse)
    # Email writing still has its reserve.
    router.get("email_generation").generate_json("x", EmailGenResponse)
    assert premium.calls


def test_local_outage_without_fallback_fails():
    down = FakeProvider("qwen3:8b", "ollama", fail=OllamaUnavailableError("down"))
    router, *_ = _router(local=down, fallback="none")
    with pytest.raises(OllamaUnavailableError):
        router.get("scoring").generate_json("x", EmailGenResponse)


def test_usage_tracker_flushes_and_counts_todays_premium():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    session.add(LLMUsage(task="email_generation", tier="premium", provider="groq", total_tokens=5000,
                         timestamp=datetime.now(UTC).replace(tzinfo=None)))
    session.commit()
    tracker = UsageTracker(factory)
    tracker.record("email_generation", "premium", "groq", "big", 900, 100, False, True, 10)
    tracker.record("research", "local", "ollama", "q", 3000, 200, False, True, 10)
    assert tracker.premium_used_today() == 6000
    assert tracker.flush() == 2 and tracker.pending == []
    assert tracker.premium_used_today() == 6000
    assert session.query(LLMUsage).count() == 3


def test_pipeline_uses_groq_only_for_email_writing(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    cfg = load_config("config.example.yaml")
    cfg.pipeline.db_path = ":memory:"
    resume = tmp_path / "resume.typ"
    resume.write_text("= Example User Resume")
    cfg.pipeline.base_resume_path = str(resume)
    cfg.pipeline.generated_resumes_dir = str(tmp_path / "gen")

    runner = PipelineRunner()
    runner.config = cfg
    runner.SessionLocal = factory
    runner.browser = MockBrowserProvider()
    runner.gmail = MockGmailProvider()
    runner.router, _tracker, premium, fast, local = _router()
    runner.usage = runner.router.tracker
    runner.usage.session_factory = factory
    runner.run()

    session = factory()
    assert session.query(Application).one().state == "Completed"
    rows = session.query(LLMUsage).all()
    premium_tasks = {r.task for r in rows if r.tier in ("premium", "premium_fast")}
    local_tasks = {r.task for r in rows if r.tier == "local"}
    assert premium_tasks <= {"email_generation", "email_regeneration", "followup_generation"}
    assert {"discovery", "research", "contact_finding", "scoring", "validation"} <= local_tasks
    assert all(r.total_tokens > 0 for r in rows)


def test_email_generation_defers_instead_of_generic_template(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    cfg = load_config("config.example.yaml")
    session.add(Run(id="R"))
    company = Company(name="PayCo", domain="payco.in")
    session.add(company)
    session.flush()
    job = Job(company_id=company.id, title="SDE Intern", url="u")
    contact = Contact(company_id=company.id, name="Priya Sharma", role="Engineering Manager", email="priya@payco.in")
    session.add_all([job, contact])
    session.flush()
    app = Application(run_id="R", job_id=job.id, contact_id=contact.id, current_stage=8, state="Email Generation")
    session.add(app)
    session.commit()

    budget_out = FakeProvider("big", "groq", fail=BudgetExceededError("budget exhausted"))
    for _ in range(4):
        assert run_stage_8_email_generation(session, cfg, budget_out, app, "R") is False
    assert app.current_stage == 8 and app.state == "Email Generation"  # transient: never failed
    assert session.query(Email).count() == 0  # no generic fallback email
    assert session.query(History).filter(History.state == "Email Deferred").count() == 4

    broken = FakeProvider("big", "groq", fail=ValueError("schema mismatch"))
    for _ in range(3):
        run_stage_8_email_generation(session, cfg, broken, app, "R")
    assert app.state == "Failed"


# ---------------------------------------------------------------------------
# Contacts without LinkedIn
# ---------------------------------------------------------------------------


def test_llm_contacts_must_appear_in_source_text():
    class ExtractLLM:
        def generate_json(self, prompt, schema, system_prompt=None):
            return schema(contacts=[
                {"name": "Priya Sharma", "role": "Engineering Manager", "email_pattern": None, "linkedin_url": None},
                {"name": "Invented Person", "role": "CTO", "email_pattern": None,
                 "linkedin_url": "https://linkedin.com/in/fake"},
            ])

    chunks = [("https://payco.in/about", "Our team"), ("https://payco.in/team", "Priya Sharma leads engineering at PayCo.")]
    found = _llm_extract(ExtractLLM(), Company(name="PayCo"), chunks, "team_page")
    assert [c.name for c in found] == ["Priya Sharma"]
    assert found[0].source_url == "https://payco.in/team"
    assert name_in_text("José Pérez", "Meet jose perez, CTO") and not name_in_text("Ravi Kumar", "Ravi leads sales")


def test_theorg_parsing_and_site_links():
    cand = parse_theorg_result(
        {"title": "Harshil Mathur - CEO at Razorpay | The Org", "url": "https://theorg.com/org/razorpay/org-chart/harshil",
         "snippet": ""}, "Razorpay")
    assert cand is not None and cand.name == "Harshil Mathur" and cand.title == "CEO" and cand.source == "theorg"

    class HomepageBrowser:
        def fetch_page(self, url, use_playwright=False):
            return ('<a href="/about-us">About</a><a href="/leadership">Leadership</a><a href="/careers">Careers</a>'
                    '<a href="https://blog.payco.in">Blog</a><a href="https://twitter.com/payco">X</a>')

        def extract_text(self, html):
            return ""

    team, blogs = discover_site_links(HomepageBrowser(), Company(name="PayCo", domain="payco.in"))
    assert team == ["https://payco.in/about-us", "https://payco.in/leadership"]
    assert blogs == ["https://blog.payco.in"]


def test_new_personas():
    assert classify_role("Senior Product Manager")[0] == "product_manager"
    assert classify_role("Head of Product")[0] == "product_lead"
    assert classify_role("Director of Engineering")[0] == "vp_engineering"
    assert classify_role("Staff Software Engineer")[0] == "tech_lead"
    assert classify_role("QA Manager")[0] == "qa_lead"
    assert normalize_personas(["director_engineering", "head_of_engineering", "talent_acquisition", "recruiter"]) == [
        "vp_engineering", "recruiter"]


# ---------------------------------------------------------------------------
# Claims guard, reply probability, ledger, browser reliability
# ---------------------------------------------------------------------------


def test_claims_guard():
    resume = ("Software Engineering Intern, Jio Platforms, May 2026. Built telemetry in Rust with Prometheus. "
              "Finalist, Goldman Sachs India Hackathon. B.Tech CSE (AI & ML), CGPA: 8.74")
    assert find_unsupported_claims("I interned at Jio and was a finalist at the Goldman Sachs India Hackathon.", resume) == []
    issues = find_unsupported_claims(
        "As a final-year student I won the hackathon, ranked top 5% and used Kubernetes.", resume)
    joined = " ".join(issues)
    assert "won" in joined and "final-year" in joined and "kubernetes" in joined and "5%" in joined


def test_reply_probability_is_explained():
    stats = OutcomeStats()
    company = Company(name="PayCo", sector="fintech", fit_score=0.85, hiring_status="hiring", employee_count=200)
    strong = explain_reply_probability(company, stats, persona="engineering_manager", email_level="verified",
                                       resume_match=0.9, relevant_job=True)
    weak = explain_reply_probability(company, stats, persona="executive", email_level="guessed", resume_match=0.3)
    assert strong["final"] > weak["final"]
    assert set(strong["components"]) == {"company_fit", "hiring_signal", "contact_seniority", "resume_match", "email_confidence"}
    assert "Final Reply Probability" in strong["explanation"] and "Company Fit: 0.85" in strong["explanation"]


def test_duplicate_ledger_blocks_company_role_contact_and_email():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Run(id="R"))
    company = Company(name="PayCo Pvt Ltd", domain="payco.in")
    session.add(company)
    session.flush()
    job1 = Job(company_id=company.id, title="Software Engineer Intern", url="u1")
    job2 = Job(company_id=company.id, title="Data Analyst Intern", url="u2")
    priya = Contact(company_id=company.id, name="Priya Sharma", role="EM", email="priya@payco.in")
    ravi = Contact(company_id=company.id, name="Ravi Rao", role="Recruiter", email="ravi@payco.in")
    session.add_all([job1, job2, priya, ravi])
    session.flush()
    app1 = Application(run_id="R", job_id=job1.id, contact_id=priya.id, state="Completed")
    app2 = Application(run_id="R", job_id=job2.id, contact_id=ravi.id, state="Gmail Draft Creation")
    session.add_all([app1, app2])
    session.flush()
    session.add(Email(application_id=app1.id, subject="s", body="b", gmail_draft_id="d1", sequence_step=0, to_email="priya@payco.in"))
    session.commit()

    assert backfill_ledger(session) == 1 and backfill_ledger(session) == 0
    assert "already contacted" in find_duplicate(session, app2, ravi, "ravi@payco.in")  # same company
    assert find_duplicate(session, app2, ravi, "ravi@payco.in", allow_multiple_contacts=True) is None  # new role+person
    assert "priya@payco.in" in find_duplicate(session, app2, ravi, "priya@payco.in", allow_multiple_contacts=True)
    job3 = Job(company_id=company.id, title="Software Engineer Intern (Speculative Application)", url="u3")
    session.add(job3)
    session.flush()
    app3 = Application(run_id="R", job_id=job3.id, contact_id=ravi.id, state="Gmail Draft Creation")
    session.add(app3)
    session.flush()
    assert "role" in find_duplicate(session, app3, ravi, "ravi@payco.in", allow_multiple_contacts=True)
    record_outreach(session, app2, ravi, "ravi@payco.in")
    session.commit()
    assert session.query(OutreachLedger).count() == 2
    assert role_key("Software Engineer Intern (Speculative Application)") == "software engineer"


def test_404_does_not_trip_domain_circuit_breaker(monkeypatch):
    browser = BrowserProvider()
    browser._save_domain_stats = lambda: None
    attempts = {"playwright": 0}

    def fake_http(url):
        raise FetchError(url, 404 if "missing" in url else 403, "failed")

    monkeypatch.setattr(browser, "fetch_page_http", fake_http)
    monkeypatch.setattr(browser, "fetch_page_playwright", lambda _url: attempts.__setitem__("playwright", 1))
    for _ in range(5):
        with pytest.raises(FetchError):
            browser.fetch_page("https://payco.in/missing-page")
    assert "payco.in" not in browser.disabled_domains and attempts["playwright"] == 0
    for _ in range(3):
        with pytest.raises(RuntimeError):
            browser.fetch_page("https://payco.in/blocked")
    assert "payco.in" in browser.disabled_domains


def test_playwright_missing_skips_fallback(monkeypatch):
    browser = BrowserProvider()
    browser._save_domain_stats = lambda: None
    browser._playwright_ready, browser.playwright_message = False, "Playwright browser is not installed. Fix: ..."
    monkeypatch.setattr(browser, "fetch_page_http", lambda url: (_ for _ in ()).throw(FetchError(url, None, "net")))
    monkeypatch.setattr(browser, "fetch_page_playwright", lambda _url: pytest.fail("must not launch Playwright"))
    with pytest.raises(RuntimeError, match="Playwright fallback unavailable"):
        browser.fetch_page("https://example.org/x")


def test_linkedin_subdomains_are_blocked_and_ta_is_recruiter():
    browser = BrowserProvider()
    assert browser.is_disabled("in.linkedin.com") and browser.is_disabled("linkedin.com")
    assert not browser.is_disabled("notlinkedin.com")
    assert classify_role("Manager - TA at Cashfree Payments")[0] == "recruiter"
    assert classify_role("Chief Financial Officer")[0] == "executive"


def test_middle_initials_merge_into_one_person():
    from src.sources.contacts import ContactCandidate, merge_contact_candidates

    merged = merge_contact_candidates([
        ContactCandidate(name="Ramkumar M Venkatesan", title="Chief Technology Officer", source="team_page"),
        ContactCandidate(name="Ramkumar Venkatesan", title="CTO", source="crunchbase"),
    ])
    assert len(merged) == 1 and merged[0].sources == ["team_page", "crunchbase"]
