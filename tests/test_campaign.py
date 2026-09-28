import sqlite3

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from src.analytics.funnel import compute_funnel
from src.config import Highlight, ResumeVariant, load_config
from src.db.models import Application, Base, Campaign, Company, Email, Job
from src.db.session import init_db
from src.pipeline.runner import PipelineRunner
from src.utils.resume import rank_highlights, select_resume_variant, typst_to_text
from tests.test_pipeline import MockBrowserProvider, MockGmailProvider, MockLLMProvider


def make_runner(tmp_path, **overrides):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    cfg = load_config("config.example.yaml")
    cfg.pipeline.db_path = ":memory:"
    resume = tmp_path / "resume.typ"
    resume.write_text("= Example User\n== Experience\nBuilt payment APIs in Python")
    cfg.pipeline.base_resume_path = str(resume)
    cfg.pipeline.generated_resumes_dir = str(tmp_path / "generated")
    for key, value in overrides.items():
        section, attr = key.split("__")
        setattr(getattr(cfg, section), attr, value)

    runner = PipelineRunner()
    runner.config = cfg
    runner.SessionLocal = session_factory
    runner.llm = MockLLMProvider()
    runner.browser = MockBrowserProvider()
    runner.gmail = MockGmailProvider()
    return runner, session_factory()


def test_campaign_end_to_end(tmp_path):
    runner, session = make_runner(tmp_path)
    progress = runner.start_campaign("Find 3 SaaS companies in India, reach engineering managers")

    campaign = session.query(Campaign).one()
    assert campaign.target_companies == 3
    assert campaign.spec["sectors"] == ["saas"] and campaign.spec["geographies"] == ["India"]
    assert campaign.personas == ["engineering_manager"]
    assert progress["companies_found"] == 1 and progress["drafts"] == 1 and progress["status"] == "active"

    company = session.query(Company).one()
    assert company.campaign_id == campaign.id
    assert company.sector == "saas" and company.funding_stage == "series_a" and company.hiring_status == "hiring"
    assert company.status == "target" and company.fit_score is not None and company.response_probability is not None

    app = session.query(Application).one()
    assert app.campaign_id == campaign.id and app.state == "Completed" and app.outreach_status == "drafted"
    assert app.persona == "engineering_manager"

    emails = session.query(Email).filter(Email.application_id == app.id).order_by(Email.sequence_step).all()
    assert [e.sequence_step for e in emails] == [0, 1, 2]
    assert emails[0].status == "draft_created" and all(e.status == "pending" for e in emails[1:])
    assert all(e.subject.startswith("Re: ") for e in emails[1:])

    contact = app.contact
    assert contact.role_category == "engineering_manager" and contact.email_status == "unverified"

    funnel = {row["stage"]: row["count"] for row in compute_funnel(session, campaign.id)["funnel"]}
    assert funnel["Companies Found"] == 1 and funnel["Contacts Found"] == 1
    assert funnel["Emails Found"] == 1 and funnel["Drafts Created"] == 1 and funnel["Emails Sent"] == 0

    # Continuing the campaign does not create duplicate outreach for the same company
    runner.continue_campaign(campaign.id)
    assert session.query(Application).count() == 1


def test_hybrid_scoring_blends_rules_and_llm(tmp_path):
    runner, session = make_runner(tmp_path)
    runner.run()
    app = session.query(Application).one()
    breakdown = app.score_breakdown
    assert breakdown["mode"] == "hybrid" and breakdown["llm"]["role_match"] == 0.9
    assert breakdown["rules"]["role_match"] == 1.0
    assert breakdown["role_match"] == pytest.approx(0.95)
    assert 0.6 < app.score <= 1.0 and app.response_probability is not None


def test_disallowed_sector_is_filtered_at_discovery(tmp_path):
    runner, session = make_runner(tmp_path, target_profile__allowed_sectors=["fintech"])
    runner.run()
    assert session.query(Company).count() == 0  # the SaaS company never enters the target list
    assert session.query(Application).count() == 0


def test_poor_fit_company_is_rejected_after_research(tmp_path):
    runner, session = make_runner(tmp_path, target_profile__min_company_fit=0.99)
    runner.run()
    company = session.query(Company).one()
    app = session.query(Application).one()
    assert company.status == "rejected" and "below minimum" in company.fit_reasoning
    assert app.state == "Poor Fit" and app.contact_id is None


def test_auto_send_schedules_draft(tmp_path):
    runner, session = make_runner(tmp_path, outreach__auto_send=True)
    runner.run()
    email = session.query(Email).filter(Email.sequence_step == 0).one()
    assert email.status == "scheduled" and email.scheduled_at is not None and email.gmail_draft_id == "draft_abc123"
    assert session.query(Application).one().outreach_status == "scheduled"


def test_resume_variant_and_highlight_ranking(tmp_path):
    cfg = load_config("config.example.yaml")
    paths = {}
    for name in ("ai", "backend", "fintech"):
        paths[name] = tmp_path / f"resume_{name}.typ"
        paths[name].write_text(f"= {name} resume")
    cfg.resumes = [
        ResumeVariant(name="ai", path=str(paths["ai"]), tags=["ai", "llm", "ml"], sectors=["ai"]),
        ResumeVariant(name="backend", path=str(paths["backend"]), tags=["backend", "python"], sectors=["saas"]),
        ResumeVariant(name="fintech", path=str(paths["fintech"]), tags=["payments", "upi"], sectors=["fintech"]),
        ResumeVariant(name="missing", path=str(tmp_path / "nope.typ"), tags=["backend"], sectors=["fintech"]),
    ]
    cfg.highlights = [
        Highlight(name="Jio internship", tags=["backend", "fintech", "payments"]),
        Highlight(name="GSIH finalist", tags=["ai", "llm", "hackathon"]),
        Highlight(name="Team Ignition", tags=["systems", "embedded"]),
    ]
    fintech_co = Company(name="PayCo", sector="fintech", tech_stack=["Java", "Kafka"])
    ai_co = Company(name="GenAI Co", sector="ai", tech_stack=["Python", "PyTorch"])
    payments_job = Job(title="Backend Engineer (Payments)", description="Build UPI payment rails")
    ai_job = Job(title="AI Engineer", description="Ship LLM features")

    choice = select_resume_variant(cfg, payments_job, fintech_co)
    assert choice is not None and choice.name == "fintech"
    assert select_resume_variant(cfg, ai_job, ai_co).name == "ai"
    assert [h.name for h in rank_highlights(cfg, payments_job, fintech_co)][0] == "Jio internship"
    assert [h.name for h in rank_highlights(cfg, ai_job, ai_co)][0] == "GSIH finalist"

    text = typst_to_text('#set page(margin: 1cm)\n= Jane\n*Backend* intern at #link("https://jio.com")[Jio]')
    assert "#set" not in text and "Jane" in text and "Jio" in text


def test_auto_migration_adds_new_columns(tmp_path):
    db_file = tmp_path / "old.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE companies (id INTEGER PRIMARY KEY, name VARCHAR UNIQUE, domain VARCHAR, employee_count INTEGER, industry VARCHAR, research_data JSON, created_at DATETIME, updated_at DATETIME)")
    conn.execute("INSERT INTO companies (id, name) VALUES (1, 'Legacy Co')")
    conn.commit()
    conn.close()

    init_db(str(db_file))
    engine = create_engine(f"sqlite:///{db_file}")
    columns = {c["name"] for c in inspect(engine).get_columns("companies")}
    assert {"sector", "funding_stage", "fit_score", "email_pattern", "is_catch_all", "campaign_id"} <= columns
    assert "outreach_events" in inspect(engine).get_table_names()
    session = sessionmaker(bind=engine)()
    assert session.query(Company).one().name == "Legacy Co"


def test_validation_failure_triggers_one_rewrite(tmp_path):
    from src.db.models import History
    from src.pipeline.stages import ValidationResponse

    class FlakyValidatorLLM(MockLLMProvider):
        def __init__(self):
            self.validations = 0
            self.email_prompts = []

        def generate_json(self, prompt, schema, system_prompt=None):
            if "ValidationResponse" in str(schema):
                self.validations += 1
                if self.validations == 1:
                    return ValidationResponse(is_valid=False, errors=["Claims relocation, not in resume"])
                return ValidationResponse(is_valid=True, errors=[])
            if "EmailGenResponse" in str(schema):
                self.email_prompts.append(prompt)
            return super().generate_json(prompt, schema, system_prompt)

    runner, session = make_runner(tmp_path)
    llm = FlakyValidatorLLM()
    runner.llm = llm
    runner.run()
    app = session.query(Application).one()
    assert app.state == "Completed"
    assert len(llm.email_prompts) == 2 and "Claims relocation" in llm.email_prompts[1]
    assert session.query(History).filter(History.state == "Validation Retry").count() == 1
    initial = session.query(Email).filter(Email.sequence_step == 0).order_by(Email.id).all()
    assert [e.status for e in initial] == ["cancelled", "draft_created"]


def test_campaign_sector_is_enforced_after_research(tmp_path):
    runner, session = make_runner(tmp_path)
    # Discovery can't tell the sector (no industry), research then reveals SaaS -> must be rejected for a fintech campaign.
    original = runner.llm.generate_json

    def no_industry(prompt, schema, system_prompt=None):
        result = original(prompt, schema, system_prompt)
        if "CompanyListResponse" in str(schema):
            for c in result.companies:
                c.industry = None
        return result

    runner.llm.generate_json = no_industry
    runner.start_campaign("Find 5 fintech companies in India")
    company = session.query(Company).one()
    assert company.sector == "saas" and company.status == "rejected"
    assert session.query(Application).one().state == "Poor Fit"
    assert runner.config.target_profile.allowed_sectors == []  # restored after the campaign run
