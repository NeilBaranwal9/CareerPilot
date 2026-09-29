import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import load_config
from src.db.models import Application, Base, Email
from src.pipeline.runner import PipelineRunner
from src.pipeline.stages import (
    CompanyListResponse,
    CompanyResearchResponse,
    ContactListResponse,
    EmailDiscoveryResponse,
    EmailGenResponse,
    JobListResponse,
    OpportunityScoreResponse,
    ResumeTailorResponse,
    ValidationResponse,
)


# Mock classes
class MockLLMProvider:
    def __init__(self, *args, **kwargs):
        pass

    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        return "Mock response text"

    def generate_json(self, prompt: str, schema, system_prompt: str = None):
        # Dynamically return matching mock responses based on schema type
        if "CompanyListResponse" in str(schema):
            return CompanyListResponse(
                companies=[
                    {
                        "name": "MockTech",
                        "domain": "mocktech.com",
                        "employee_count": 150,
                        "industry": "SaaS",
                    }
                ]
            )
        elif "JobListResponse" in str(schema):
            if "nojobscorp" in prompt.lower():
                return JobListResponse(jobs=[])
            return JobListResponse(
                jobs=[
                    {
                        "title": "Software Engineer",
                        "url": "http://mocktech.com/jobs/1",
                        "location": "Remote",
                        "salary": "12 LPA",
                        "experience_years": 0.5,
                        "description": "Python developer role.",
                    }
                ]
            )
        elif "TargetedParseResponse" in str(schema):
            if "nojobscorp" in prompt.lower():
                return schema(
                    company_name="NoJobsCorp",
                    domain="nojobscorp.com",
                    contact_email=None,
                    contact_name=None
                )
            return schema(
                company_name="TargetedCorp",
                domain="targetedcorp.com",
                contact_email="hiring@targetedcorp.com",
                contact_name="Hiring Manager"
            )
        elif "JDParseResponse" in str(schema):
            return schema(
                title="Pasted Software Engineer",
                location="Remote",
                salary="15 LPA",
                experience_years=0.5,
                description="This is a parsed description of the pasted job description."
            )
        elif "CompanyResearchResponse" in str(schema):
            return CompanyResearchResponse(
                business_model="B2B SaaS product",
                funding="Series A 5M USD",
                tech_stack=["Python", "FastAPI", "React"],
                culture="Remote friendly, high ownership",
                leadership=["CTO Alice", "CEO Bob"],
            )
        elif "ContactListResponse" in str(schema):
            return ContactListResponse(
                contacts=[
                    {
                        "name": "Alice Developer",
                        "role": "Engineering Manager",
                        "email_pattern": "alice@mocktech.com",
                        "linkedin_url": "https://linkedin.com/alice",
                    }
                ]
            )
        elif "EmailDiscoveryResponse" in str(schema):
            return EmailDiscoveryResponse(email="alice@mocktech.com", pattern_used="explicit")
        elif "OpportunityScoreResponse" in str(schema):
            return OpportunityScoreResponse(
                role_match=0.9,
                tech_stack=0.9,
                salary=0.8,
                company_quality=0.8,
                growth=0.8,
                confidence=0.9,
                reasoning="Excellent match.",
            )
        elif "ResumeTailorResponse" in str(schema):
            return ResumeTailorResponse(
                tailored_typst_content="= Tailored Resume Content",
                keywords_added=["FastAPI", "Playwright"],
                reasoning="Aligned experience.",
            )
        elif "EmailGenResponse" in str(schema):
            return EmailGenResponse(
                subject="Internship interest at MockTech",
                body_html=(
                    "<p>Hi Alice,</p><p>I was reading about MockTech's Series A and the B2B SaaS product you "
                    "sell to finance teams, especially how much of their month-end work it takes off their plate."
                    "</p><p>I'm a computer science student looking for a software engineering internship. Over the "
                    "last year I have built backend services in Python and FastAPI that collect operational data "
                    "from several internal tools, clean it up and make it available to the people who need to act "
                    "on it every morning.</p><p>Most of that work was about keeping the pipeline dependable when "
                    "inputs were messy, which is why a product that finance teams rely on every day interests me "
                    "more than a demo project would.</p><p>If your team is taking interns this year, I'd be glad to "
                    "hear where I should apply, or who on the team I could talk to about it.</p>"
                    "<p>Best,<br>Jane Doe</p>"
                ),
            )
        elif "ValidationResponse" in str(schema):
            return ValidationResponse(is_valid=True, errors=[])
 
        raise ValueError(f"No mock handler for schema {schema}")


class MockBrowserProvider:
    def search_google(self, query, num_results=5):
        return [
            {
                "title": "Mock Link",
                "url": "http://example.com/mock",
                "snippet": "Snippet content",
            }
        ]

    def fetch_page(self, url, use_playwright=False):
        return "<html><body>Mock Careers text</body></html>"

    def extract_text(self, html):
        return "Mock Careers text. Alice Developer, Engineering Manager at MockTech, leads the payments team."


class MockGmailProvider:
    def __init__(self, *args, **kwargs):
        pass

    def authenticate(self, interactive=True):
        return True

    def create_draft(self, to_email, subject, body_html, resume_path=None):
        return "draft_abc123"


def test_pipeline_integration(tmp_path):
    """
    Test the full pipeline stages workflow end-to-end
    using mocked LLM, Browser, and Gmail APIs over an in-memory SQLite DB.
    """
    # 1. Setup in-memory SQLite DB
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    # 2. Setup mock runner config
    config_data = load_config("config.example.yaml")
    # Set paths to temp directories to avoid writing to project root during tests
    config_data.pipeline.db_path = ":memory:"
    # Pure LLM scoring so the weighted score is exactly predictable from the mock (hybrid is tested separately)
    config_data.scoring.mode = "llm"

    # Write a temporary resume file for testing
    temp_resume = tmp_path / "resume.typ"
    temp_resume.write_text("= Example User Resume")
    config_data.pipeline.base_resume_path = str(temp_resume)
    config_data.pipeline.generated_resumes_dir = str(tmp_path / "generated")

    # 3. Create runner with injected mocks
    runner = PipelineRunner()
    runner.config = config_data
    runner.SessionLocal = session_factory
    runner.llm = MockLLMProvider()
    runner.browser = MockBrowserProvider()
    runner.gmail = MockGmailProvider()

    # 4. Run pipeline stages
    _run_id = runner.run()

    # 5. Assert states in database
    db_apps = session.query(Application).all()
    assert len(db_apps) == 1
    app = db_apps[0]

    assert app.state == "Completed"
    assert app.current_stage == 12
    assert app.score == pytest.approx(0.88, abs=0.01)  # Calculated based on mock scores

    # Verify related entities
    assert app.job.title == "Software Engineer"
    assert app.job.company.name == "MockTech"
    assert app.contact.name == "Alice Developer"
    assert app.contact.email == "alice@mocktech.com"

    # Verify resume file generated
    assert app.tailored_resume_path is not None
    assert os.path.exists(app.tailored_resume_path)

    # Verify email and draft ID saved
    email_entry = session.query(Email).filter(Email.application_id == app.id).first()
    assert email_entry is not None
    # Email body should contain the user's identity links from config
    assert email_entry.body is not None
    assert email_entry.status == "draft_created"
    assert email_entry.gmail_draft_id == "draft_abc123"

    session.close()


def test_targeted_outreach_normal(tmp_path):
    # 1. Setup in-memory SQLite DB
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    # 2. Setup mock runner config
    config_data = load_config("config.example.yaml")
    config_data.pipeline.db_path = ":memory:"

    temp_resume = tmp_path / "resume.typ"
    temp_resume.write_text("= Example User Resume")
    config_data.pipeline.base_resume_path = str(temp_resume)
    config_data.pipeline.generated_resumes_dir = str(tmp_path / "generated")

    # 3. Create runner with injected mocks
    runner = PipelineRunner()
    runner.config = config_data
    runner.SessionLocal = session_factory
    runner.llm = MockLLMProvider()
    runner.browser = MockBrowserProvider()
    runner.gmail = MockGmailProvider()

    # 4. Run targeted pipeline
    runner.run_targeted("TargetedCorp")

    # 5. Assert states in database
    db_apps = session.query(Application).all()
    assert len(db_apps) == 1
    app = db_apps[0]

    assert app.state == "Completed"
    assert app.current_stage == 12
    assert app.job.company.name == "TargetedCorp"
    # Verify the provided contact from LLM parse was linked
    assert app.contact.email == "hiring@targetedcorp.com"
    assert app.contact.name == "Hiring Manager"

    session.close()


def test_targeted_outreach_speculative(tmp_path):
    # 1. Setup in-memory SQLite DB
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    # 2. Setup mock runner config
    config_data = load_config("config.example.yaml")
    config_data.pipeline.db_path = ":memory:"
    # Legacy flag: must no longer produce a "<role> (Speculative Application)" job
    config_data.job_preferences.allow_speculative_outreach = True

    temp_resume = tmp_path / "resume.typ"
    temp_resume.write_text("= Example User Resume")
    config_data.pipeline.base_resume_path = str(temp_resume)
    config_data.pipeline.generated_resumes_dir = str(tmp_path / "generated")

    # 3. Create runner with injected mocks
    runner = PipelineRunner()
    runner.config = config_data
    runner.SessionLocal = session_factory
    runner.llm = MockLLMProvider()
    runner.browser = MockBrowserProvider()
    runner.gmail = MockGmailProvider()

    # 4. Run targeted pipeline for a company with no jobs (NoJobsCorp)
    runner.run_targeted("no_jobs at nojobscorp.com")

    # 5. Assert states in database
    db_apps = session.query(Application).all()
    assert len(db_apps) == 1
    app = db_apps[0]

    assert app.state == "Completed"
    assert app.current_stage == 12
    assert app.job.company.name == "NoJobsCorp"
    # No opening exists, so the targeted company gets a company-level inquiry instead of an invented job title
    assert app.outreach_type == "company_speculative" and app.is_company_level
    assert app.job.source == "company_outreach" and app.job.url.startswith("company-outreach://")
    assert app.job.title == "Company-level internship inquiry"
    assert "Speculative Application" not in app.job.title
    assert not any(role.lower() in app.job.title.lower() for role in config_data.job_preferences.roles)

    session.close()


def test_targeted_outreach_with_jd(tmp_path):
    # 1. Setup in-memory SQLite DB
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    # 2. Setup mock runner config
    config_data = load_config("config.example.yaml")
    config_data.pipeline.db_path = ":memory:"

    temp_resume = tmp_path / "resume.typ"
    temp_resume.write_text("= Example User Resume")
    config_data.pipeline.base_resume_path = str(temp_resume)
    config_data.pipeline.generated_resumes_dir = str(tmp_path / "generated")

    # 3. Create runner with injected mocks
    runner = PipelineRunner()
    runner.config = config_data
    runner.SessionLocal = session_factory
    runner.llm = MockLLMProvider()
    runner.browser = MockBrowserProvider()
    runner.gmail = MockGmailProvider()

    # 4. Run targeted pipeline with a pasted JD
    pasted_jd = "We are hiring a Python Software Engineer to build beautiful web apps. Requires 2 years of experience."
    runner.run_targeted("TargetedCorp", jd=pasted_jd)

    # 5. Assert states in database
    db_apps = session.query(Application).all()
    assert len(db_apps) == 1
    app = db_apps[0]

    assert app.state == "Completed"
    assert app.current_stage == 12
    assert app.job.company.name == "TargetedCorp"
    # Verify job is parsed and created from pasted JD
    assert app.job.title == "Pasted Software Engineer"
    assert app.job.url.startswith("pasted://")
    assert app.job.experience_years_required == 0.5
    assert "parsed description" in app.job.description

    session.close()


def test_retry_failed_fk_safety(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    config_data = load_config("config.example.yaml")
    config_data.pipeline.db_path = ":memory:"

    temp_resume = tmp_path / "resume.typ"
    temp_resume.write_text("= Example User Resume")
    config_data.pipeline.base_resume_path = str(temp_resume)
    config_data.pipeline.generated_resumes_dir = str(tmp_path / "generated")

    runner = PipelineRunner()
    runner.config = config_data
    runner.SessionLocal = session_factory
    runner.llm = MockLLMProvider()
    runner.browser = MockBrowserProvider()
    runner.gmail = MockGmailProvider()

    # Run targeted to create a completed app
    runner.run_targeted("TargetedCorp")

    # Manually mark app as failed
    app = session.query(Application).first()
    app.state = "Draft Failed"
    session.commit()

    # Execute retry_failed - must not crash with Foreign Key IntegrityError
    runner.retry_failed()

    session.refresh(app)
    assert app.state == "Completed"
    session.close()


def test_stage_7_resume_selection_and_toggle(tmp_path):
    from src.db.models import Application, Company, Job, ResumeVersion, Run
    from src.pipeline.stages import run_stage_7_resume_tailoring

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    config_data = load_config("config.example.yaml")
    config_data.pipeline.db_path = ":memory:"
    config_data.pipeline.generate_resume = False

    ai_resume = tmp_path / "resume_ai.typ"
    ai_resume.write_text("= AI Resume Content")
    dev_resume = tmp_path / "resume_dev.typ"
    dev_resume.write_text("= Dev Resume Content")

    config_data.pipeline.ai_resume_path = str(ai_resume)
    config_data.pipeline.dev_resume_path = str(dev_resume)
    config_data.pipeline.generated_resumes_dir = str(tmp_path / "generated")

    run = Run(id="test_run")
    session.add(run)
    company = Company(name="TestAI", domain="testai.com")
    session.add(company)
    session.commit()

    job_ai = Job(company_id=company.id, title="AI Engineer", url="http://test.com")
    session.add(job_ai)
    session.commit()

    app_ai = Application(run_id="test_run", job_id=job_ai.id, current_stage=6, state="Resume Tailoring")
    session.add(app_ai)
    session.commit()

    res = run_stage_7_resume_tailoring(session, config_data, MockLLMProvider(), app_ai, "test_run")
    assert res is True
    assert app_ai.current_stage == 8

    rv = session.query(ResumeVersion).filter(ResumeVersion.application_id == app_ai.id).first()
    assert rv is not None
    assert "AI-focused" in rv.reasoning

    session.close()

