from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import load_config
from src.db.models import Application, Base, Company, Contact, Job, Run
from src.intel.classify import (
    classify_role,
    classify_sector,
    normalize_funding_stage,
    normalize_sector,
    parse_headcount,
    parse_salary_lpa,
)
from src.intel.learning import compute_outcome_stats, estimate_response_probability
from src.intel.scoring import compute_company_fit, rule_based_opportunity, title_relevance


def test_sector_classification():
    assert classify_sector("UPI payments, lending and credit cards for SMBs")[0] == "fintech"
    assert classify_sector("Online courses and exam prep for students")[0] == "edtech"
    assert classify_sector("Telemedicine platform connecting patients with doctors")[0] == "healthtech"
    assert classify_sector("High-frequency trading and market making firm")[0] == "trading"
    assert classify_sector("Generative AI copilots built on LLMs")[0] == "ai"
    # whole-word matching: 'email' must not look like AI
    assert "ai" not in classify_sector("Email marketing automation")[1]
    assert normalize_sector("Financial Services") == "fintech"
    assert normalize_sector("FinTech") == "fintech"


def test_funding_and_numbers():
    assert normalize_funding_stage("Raised a seed round in 2021 and a $40M Series B in 2024") == "series_b"
    assert normalize_funding_stage("Pre-seed funded by angels") == "pre_seed"
    assert normalize_funding_stage("Listed on NSE since 2021") == "public"
    assert normalize_funding_stage("Bootstrapped and profitable") == "bootstrapped"
    assert parse_headcount("51-200 employees") == 125
    assert parse_salary_lpa("10-20 LPA") == (10.0, 20.0)
    assert parse_salary_lpa("₹12,00,000 per annum") == (12.0, 12.0)
    low, high = parse_salary_lpa("$120k - $150k")
    assert low == pytest.approx(102.0) and high == pytest.approx(127.5)


def test_role_classification():
    assert classify_role("Senior Technical Recruiter")[0] == "recruiter"
    assert classify_role("Talent Acquisition Partner")[0] == "recruiter"
    assert classify_role("Engineering Manager, Payments")[0] == "engineering_manager"
    assert classify_role("Co-founder & CTO")[0] == "founder"
    assert classify_role("Head of Engineering")[0] == "vp_engineering"
    assert classify_role("Hiring Manager - Backend")[0] == "hiring_manager"
    assert classify_role("Founding Engineer")[0] == "engineer"


def test_title_relevance_prefers_entry_level():
    roles = ["Software Engineer", "Backend Engineer"]
    assert title_relevance("Software Engineer I", roles, 1.0) >= 0.9
    assert title_relevance("Senior Staff Software Engineer", roles, 1.0) < 0.5
    assert title_relevance("Sales Executive", roles, 1.0) == 0.0


def _config():
    cfg = load_config("config.example.yaml")
    return cfg


def test_company_fit_and_rejection():
    cfg = _config()
    fintech = Company(
        name="PayCo", sector="fintech", funding_stage="series_a", employee_count=300, hiring_status="hiring",
        tech_stack=["Python", "Django", "PostgreSQL"], location="Bengaluru, India",
    )
    generic = Company(name="Widgets", sector="generic", funding_stage="public", employee_count=20000, tech_stack=["COBOL"])
    good = compute_company_fit(fintech, cfg)
    bad = compute_company_fit(generic, cfg)
    assert good.score > 0.8 and not good.rejected
    assert bad.score < good.score and bad.rejected

    cfg.target_profile.allowed_sectors = ["fintech", "trading"]
    edtech = Company(name="LearnCo", sector="edtech", funding_stage="series_a", employee_count=300)
    result = compute_company_fit(edtech, cfg)
    assert result.rejected and "not in allowed sectors" in result.reject_reason


def test_rule_based_opportunity():
    cfg = _config()
    company = Company(name="PayCo", sector="fintech", fit_score=0.9, hiring_status="hiring", funding_stage="series_a", tech_stack=["Python"])
    job = Job(title="Backend Engineer", salary="12-18 LPA", experience_years_required=0.0, description="Python APIs")
    rules = rule_based_opportunity(job, company, cfg)
    assert rules["role_match"] == 1.0 and rules["salary"] == 1.0 and rules["company_quality"] == 0.9


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _add_outcome(session, idx: int, sector: str, persona: str, replied: bool, interview: bool = False):
    company = Company(name=f"{sector}-{idx}", sector=sector, funding_stage="series_a", employee_count=150)
    session.add(company)
    session.flush()
    job = Job(company_id=company.id, title="Software Engineer", url=f"http://x/{sector}/{idx}")
    contact = Contact(company_id=company.id, name=f"Person {idx}", role=persona, role_category=persona)
    session.add_all([job, contact])
    session.flush()
    status = "interview" if interview else ("replied" if replied else "no_response")
    now = datetime.now(UTC).replace(tzinfo=None)
    session.add(
        Application(
            run_id="R", job_id=job.id, contact_id=contact.id, state="Completed", current_stage=12, persona=persona,
            outreach_status=status, sent_at=now, replied_at=now if replied else None,
            interview_at=now if interview else None,
        )
    )


def test_learning_prefers_what_gets_replies():
    session = _session()
    session.add(Run(id="R"))
    for i in range(10):
        _add_outcome(session, i, "fintech", "engineering_manager", replied=i < 5, interview=i < 2)
    for i in range(10, 20):
        _add_outcome(session, i, "generic", "recruiter", replied=False)
    session.commit()
    cfg = _config()
    stats = compute_outcome_stats(session, cfg)
    assert stats.total_sent == 20 and stats.total_replies == 5
    assert stats.multiplier("sector", "fintech") > 1.0 > stats.multiplier("sector", "generic")
    assert stats.multiplier("persona", "engineering_manager") > stats.multiplier("persona", "recruiter")

    fintech = Company(name="New Fintech", sector="fintech", funding_stage="series_a", employee_count=150)
    generic = Company(name="New Generic", sector="generic", funding_stage="series_a", employee_count=150)
    assert estimate_response_probability(fintech, stats) > estimate_response_probability(generic, stats)

    # Learned preference also lifts company fit for the sector that produces replies/interviews
    cfg.target_profile.sector_weights = {"fintech": 0.6, "generic": 0.6}
    fit_fintech = compute_company_fit(fintech, cfg, stats).score
    fit_generic = compute_company_fit(generic, cfg, stats).score
    assert fit_fintech > fit_generic


def test_intern_only_roles_reject_full_time_titles():
    roles = ["Software Engineer Intern", "Data Analyst Intern"]
    assert title_relevance("Site Reliability Engineer", roles, 1.0) < 0.5
    assert title_relevance("Software Engineer Intern - Payments", roles, 1.0) >= 0.9


def test_founder_penalised_at_large_company_with_unknown_headcount():
    from src.intel.classify import effective_headcount
    from src.intel.learning import persona_multiplier

    assert effective_headcount(None, "series_d_plus") == 1500
    big = persona_multiplier("founder", effective_headcount(None, "series_d_plus"), None)
    assert big < persona_multiplier("engineering_manager", 1500, None)
