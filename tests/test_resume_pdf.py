import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import ResumeVariant, VariantRule, load_config
from src.db.models import Application, Base, Company, Job, ResumeVersion, Run
from src.pipeline.runner import PipelineRunner
from src.pipeline.schemas import ResumeEntry, ResumeSection, StructuredResumeSchema
from src.pipeline.stages import run_stage_7_resume_tailoring
from src.utils.resume import select_resume_variant
from src.utils.resume_pdf import (
    check_tailored_resume,
    clean_extracted_text,
    extract_resume_text,
    render_resume_pdf,
    validate_resume_file,
)
from tests.test_pipeline import MockBrowserProvider, MockGmailProvider, MockLLMProvider


def sample_resume(org: str = "Jio Platforms Limited", skills: str = "Python, Rust, FastAPI") -> StructuredResumeSchema:
    return StructuredResumeSchema(
        name="Neil Test",
        contact_line="neil@example.org | github.com/neiltest",
        sections=[
            ResumeSection(
                heading="Experience",
                entries=[
                    ResumeEntry(
                        title="Software Engineering Intern",
                        organization=org,
                        location="Mumbai",
                        dates="May 2026 - Jun 2026",
                        bullets=["Built a UDP telemetry ingestion daemon in Rust", "Automated rulebook parsing with local LLMs"],
                    )
                ],
            ),
            ResumeSection(heading="Skills", items=[f"Languages: {skills}"]),
        ],
        keywords_added=["Rust", "Kubernetes"],
        reasoning="Led with telemetry work.",
    )


@pytest.fixture
def original_pdf(tmp_path):
    path = tmp_path / "Neil_Resume.pdf"
    render_resume_pdf(sample_resume(), str(path))
    return path


def test_pdf_render_and_extract_roundtrip(original_pdf):
    assert validate_resume_file(str(original_pdf)) == []
    text = extract_resume_text(str(original_pdf))
    assert "Neil Test" in text and "Jio Platforms Limited" in text and "UDP telemetry" in text
    assert clean_extracted_text("V ellore Institute of T echnology") == "Vellore Institute of Technology"


def test_validate_rejects_fake_pdf(tmp_path):
    fake = tmp_path / "resume.pdf"
    fake.write_text("not a pdf")
    assert "file does not look like a PDF" in validate_resume_file(str(fake))
    assert validate_resume_file(str(tmp_path / "missing.pdf"))


def test_fabrication_guard(original_pdf):
    text = extract_resume_text(str(original_pdf))
    assert check_tailored_resume(text, sample_resume()) == []
    errors = check_tailored_resume(text, sample_resume(org="Goldman Sachs", skills="Python, Kubernetes"))
    assert any("Goldman Sachs" in e for e in errors) and any("Kubernetes" in e for e in errors)


class StructuredMockLLM(MockLLMProvider):
    def __init__(self, resume: StructuredResumeSchema):
        self.resume = resume
        self.prompts: dict[str, str] = {}

    def generate_json(self, prompt, schema, system_prompt=None):
        self.prompts[schema.__name__] = prompt
        if schema is StructuredResumeSchema:
            return self.resume
        return super().generate_json(prompt, schema, system_prompt)


def _stage7_setup(tmp_path, pdf_path, generate: bool):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    cfg = load_config("config.example.yaml")
    cfg.pipeline.generate_resume = generate
    cfg.pipeline.generated_resumes_dir = str(tmp_path / "generated")
    cfg.resumes = [ResumeVariant(name="fintech", path=str(pdf_path), default=True, focus="payments first")]
    session.add(Run(id="R"))
    company = Company(name="PayCo", sector="fintech", description="UPI payments", tech_stack=["Rust"])
    session.add(company)
    session.flush()
    job = Job(company_id=company.id, title="Software Engineer Intern", url="u", description="Rust services for payments")
    session.add(job)
    session.flush()
    app = Application(run_id="R", job_id=job.id, current_stage=7, state="Resume Tailoring")
    session.add(app)
    session.commit()
    return session, cfg, app


def test_stage7_attaches_pdf_unchanged_when_generation_off(tmp_path, original_pdf):
    session, cfg, app = _stage7_setup(tmp_path, original_pdf, generate=False)
    assert run_stage_7_resume_tailoring(session, cfg, MockLLMProvider(), app, "R") is True
    assert app.tailored_resume_path == str(original_pdf) and app.current_stage == 8
    rv = session.query(ResumeVersion).one()
    assert rv.variant == "fintech" and "unchanged" in rv.reasoning


def test_stage7_generates_tailored_pdf(tmp_path, original_pdf):
    session, cfg, app = _stage7_setup(tmp_path, original_pdf, generate=True)
    llm = StructuredMockLLM(sample_resume())
    assert run_stage_7_resume_tailoring(session, cfg, llm, app, "R") is True
    out = app.tailored_resume_path
    assert out != str(original_pdf) and out.endswith("Neil_Resume.pdf") and os.path.exists(out)
    assert validate_resume_file(out) == []
    assert "UDP telemetry" in extract_resume_text(out)
    prompt = llm.prompts["StructuredResumeSchema"]
    assert "Jio Platforms Limited" in prompt and "Rust services for payments" in prompt and "payments first" in prompt
    rv = session.query(ResumeVersion).one()
    assert rv.keywords_added == ["Rust"]  # 'Kubernetes' is not in the original resume


def test_stage7_falls_back_to_original_on_fabrication(tmp_path, original_pdf):
    session, cfg, app = _stage7_setup(tmp_path, original_pdf, generate=True)
    llm = StructuredMockLLM(sample_resume(org="Goldman Sachs"))
    assert run_stage_7_resume_tailoring(session, cfg, llm, app, "R") is True
    assert app.tailored_resume_path == str(original_pdf)
    assert "rejected" in session.query(ResumeVersion).one().reasoning


def test_rule_based_variant_selection(tmp_path, original_pdf):
    cfg = load_config("config.example.yaml")
    cfg.resumes = [
        ResumeVariant(name="backend", path=str(original_pdf), default=True),
        ResumeVariant(name="fintech", path=str(original_pdf), priority=20, rules=[VariantRule(sectors=["fintech", "trading"])]),
        ResumeVariant(
            name="aiml", path=str(original_pdf), priority=30,
            rules=[VariantRule(title_any=["ml", "ai", "machine learning"], title_none=["research"])],
        ),
        ResumeVariant(name="research", path=str(original_pdf), priority=40, rules=[VariantRule(title_any=["research"])]),
        ResumeVariant(name="founder_pitch", path=str(original_pdf), priority=50,
                      rules=[VariantRule(personas=["founder"], sectors=["fintech"])]),
        ResumeVariant(name="missing", path=str(tmp_path / "nope.pdf"), priority=99, rules=[VariantRule()]),
    ]
    fintech = Company(name="PayCo", sector="fintech")
    saas = Company(name="Docs", sector="saas")

    assert select_resume_variant(cfg, Job(title="Backend Intern"), fintech).name == "fintech"
    assert select_resume_variant(cfg, Job(title="ML Intern"), fintech).name == "aiml"  # higher priority than fintech
    assert select_resume_variant(cfg, Job(title="ML Research Intern"), saas).name == "research"
    assert select_resume_variant(cfg, Job(title="Backend Intern"), fintech, persona="founder").name == "founder_pitch"
    default = select_resume_variant(cfg, Job(title="Backend Intern"), saas)
    assert default.name == "backend" and "default" in default.reason
    assert "sector=fintech" in select_resume_variant(cfg, Job(title="SDE Intern"), fintech).reason


def test_full_pipeline_with_pdf_resume_passes_everything_to_email(tmp_path, original_pdf):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    cfg = load_config("config.example.yaml")
    cfg.pipeline.db_path = ":memory:"
    cfg.pipeline.generate_resume = False
    cfg.pipeline.base_resume_path = str(original_pdf)
    cfg.resumes = []

    llm = StructuredMockLLM(sample_resume())
    runner = PipelineRunner()
    runner.config = cfg
    runner.SessionLocal = session_factory
    runner.llm = llm
    runner.browser = MockBrowserProvider()
    runner.gmail = MockGmailProvider()
    runner.run()

    session = session_factory()
    app = session.query(Application).one()
    assert app.state == "Completed" and app.tailored_resume_path == str(original_pdf)
    prompt = llm.prompts["EmailGenResponse"]
    assert "UDP telemetry ingestion daemon" in prompt          # resume text
    assert "Python developer role." in prompt                  # job description
    assert "B2B SaaS product" in prompt                        # company research
    assert "Alice Developer" in prompt and "Engineering Manager" in prompt  # contact


def test_non_breaking_hyphens_survive_rendering(tmp_path):
    resume = sample_resume()
    resume.sections[0].entries[0].bullets = ["Built a zero‑allocation, real‑time daemon"]
    out = tmp_path / "hyphen.pdf"
    render_resume_pdf(resume, str(out))
    assert "zero-allocation, real-time" in extract_resume_text(str(out))


def test_sector_rules_use_primary_sector_only(tmp_path, original_pdf):
    cfg = load_config("config.example.yaml")
    cfg.resumes = [
        ResumeVariant(name="fintech", path=str(original_pdf), priority=20, rules=[VariantRule(sectors=["fintech"])]),
        ResumeVariant(name="aiml", path=str(original_pdf), priority=30, rules=[VariantRule(sectors=["ai"])]),
        ResumeVariant(name="backend", path=str(original_pdf), default=True),
    ]
    ai_heavy_fintech = Company(name="PayCo", sector="fintech", sub_sectors=["ai", "devtools"])
    assert select_resume_variant(cfg, Job(title="Product Analyst Intern"), ai_heavy_fintech).name == "fintech"
