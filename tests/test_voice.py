from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import load_config
from src.db.models import Application, Base, Company, Contact, Email, Job, ResumeVersion, Run
from src.outreach.voice import (
    ResearchSignals,
    check_grounding,
    check_voice,
    choose_opening,
    overused_recent_phrases,
    persona_ask,
    recent_patterns_summary,
    research_signals,
    word_target,
)
from src.pipeline.schemas import EmailGenResponse
from src.pipeline.stages import clean_role_name, run_stage_8_email_generation


def para(*paragraphs: str) -> str:
    return "".join(f"<p>{p}</p>" for p in paragraphs) + "<p>Best,<br>Neil Baranwal</p>"


TEMPLATEY = para(
    "Hi Priya,",
    "I was impressed by PayCo's new UPI refunds feature, which aligns with my interest in payments.",
    "I am a pre-final-year student. At Jio, I built telemetry systems, which aligns with PayCo's focus on scale.",
    "I am keen to support PayCo's growth.",
    "I would appreciate any referral or guidance.",
)


def test_stock_phrases_repeated_within_one_email_are_flagged():
    issues = check_voice(TEMPLATEY, [], "Neil Baranwal")
    assert any("'which aligns with' is used 2 times" in i for i in issues)


def test_phrases_overused_across_recent_emails_are_flagged():
    recent = [
        para("Hi A,", "I came across your payments API. It is data-driven work."),
        para("Hi B,", "I came across your lending app. Our data-driven approach."),
    ]
    assert set(overused_recent_phrases(recent)) >= {"I came across", "data-driven"}
    fresh = para("Hi C,", "I came across your card product last week.")
    assert any("'I came across' was used in recent emails" in i for i in check_voice(fresh, recent, "Neil Baranwal"))
    # a strict phrase is avoided even after a single recent use
    assert "which aligns with" in overused_recent_phrases([para("Hi A,", "X, which aligns with Y.")])


def test_copied_opening_and_ask_are_flagged_but_varied_email_passes():
    previous = para(
        "Hi A,",
        "I saw your post on migrating payments to Kafka last month.",
        "I'm a student at VIT Chennai looking for a backend internship.",
        "If your team is taking interns, I'd appreciate a pointer to the right person.",
    )
    copy = para(
        "Hi B,",
        "I saw your post on fraud detection models.",
        "I'm a student at VIT Chennai looking for a data internship.",
        "If your team is taking interns, I'd appreciate a pointer to the right person.",
    )
    issues = check_voice(copy, [previous], "Neil Baranwal")
    assert any("opening" in i for i in issues) and any("closing ask" in i for i in issues)
    varied = para(
        "Hi B,",
        "Your fraud detection write-up about graph features was a good read.",
        "I'm studying CSE (AI & ML) at VIT Chennai and looking for a data science internship.",
        "Is the risk team hiring interns this cycle? If so, where should I apply?",
    )
    assert check_voice(varied, [previous], "Neil Baranwal") == []


def test_grounding_rejects_invented_needs_openings_relationships_and_responsibilities():
    research = "PayCo builds UPI payment infrastructure for merchants. Recent: launched instant refunds."
    body = para(
        "Hi Priya,",
        "PayCo's focus on developer happiness and your team's need for ML engineers stood out.",
        "I saw the backend internship opening on your careers page.",
        "Ravi suggested I reach out. You lead the fraud platform team.",
    )
    issues = " ".join(check_grounding(body, research, "Priya Sharma Engineering Manager", False, "Neil Baranwal"))
    assert "focus on developer happiness" in issues and "need for ML engineers" in issues
    assert "job opening" in issues and "relationship" in issues and "responsibilities" in issues

    grounded = para(
        "Hi Priya,",
        "I noticed PayCo launched instant refunds for UPI merchants.",
        "If your team is hiring interns, I'd be glad to know where to apply.",
    )
    assert check_grounding(grounded, research, "Priya Sharma Engineering Manager", False, "Neil Baranwal") == []
    # a real posting may be referred to
    posting = para("Hi Priya,", "I saw the backend internship opening for the payments team.")
    assert check_grounding(posting, research + " backend internship payments", "", True, "Neil Baranwal") == []


def test_openings_follow_research_and_rotate():
    signals = ResearchSignals(has_post=True, has_news=True, has_real_job=False, has_product=True)
    first = choose_opening(signals, [])
    second = choose_opening(signals, [first])
    third = choose_opening(signals, [second, first])
    assert len({first, second, third}) == 3
    assert choose_opening(ResearchSignals(False, False, False, False), ["news"]) == "plain"
    assert research_signals({"culture": "Engineering blog on Kafka"}, [], "speculative", None).has_post
    assert not research_signals({}, [], "speculative", None).has_real_job


def test_persona_asks_and_lengths():
    assert "Do not ask a recruiter for a referral" in persona_ask("recruiter")
    assert "No referral" in persona_ask("founder")
    assert "referral" in persona_ask("engineering_manager")
    assert word_target("founder", 120, 170) == (120, 155)
    assert word_target("engineering_manager", 120, 170) == (120, 170)
    assert clean_role_name("Product Analyst Intern (Speculative Application)") == "Product Analyst"


def test_stage8_prompt_varies_and_records_opening(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    cfg = load_config("config.example.yaml")
    cfg.outreach.followups = []
    resume = tmp_path / "r.typ"
    resume.write_text("= Neil\nInterned at Jio Platforms on telemetry.")
    session.add(Run(id="R"))

    prompts: list[str] = []

    class CaptureLLM:
        def generate_json(self, prompt, schema, system_prompt=None):
            prompts.append(prompt)
            body = para("Hi X,", "I was reading about your UPI refunds launch.", "I'm keen to join, which aligns with my goals.")
            return EmailGenResponse(subject="Internship", body_html=body)

    apps = []
    for i, persona in enumerate(["recruiter", "engineering_manager"]):
        company = Company(name=f"PayCo{i}", domain=f"payco{i}.in", description="UPI payment infrastructure for merchants",
                          recent_news=[{"title": "PayCo launches instant refunds"}], research_data={"products": ["Refunds API"]})
        session.add(company)
        session.flush()
        job = Job(company_id=company.id, title="Backend Intern (Speculative Application)", url=f"u{i}", source="speculative")
        contact = Contact(company_id=company.id, name=f"Priya Sharma{i}", role=persona, role_category=persona,
                          email=f"p{i}@payco{i}.in")
        session.add_all([job, contact])
        session.flush()
        app = Application(run_id="R", job_id=job.id, contact_id=contact.id, current_stage=8, state="Email Generation",
                          persona=persona)
        session.add(app)
        session.flush()
        session.add(ResumeVersion(application_id=app.id, parent_resume=str(resume), company="c", role="r", reasoning="x",
                                  path=str(resume)))
        session.commit()
        apps.append(app)

    for app in apps:
        assert run_stage_8_email_generation(session, cfg, CaptureLLM(), app, "R") is True
    styles = [e.opening_style for e in session.query(Email).order_by(Email.id).all()]
    assert styles[0] and styles[1] and styles[0] != styles[1]
    assert "Do not ask a recruiter for a referral" in prompts[0]
    assert "a Backend internship" in prompts[0]
    # the second prompt knows how the first email opened and which stock phrases to avoid
    assert "Your last emails opened and asked like this" in prompts[1] and "which aligns with" in prompts[1]
    assert recent_patterns_summary([session.query(Email).first().body], "Neil Baranwal").startswith("1. opened:")


def test_at_most_one_linking_sentence_and_jargon():
    body = para(
        "Hi A,",
        "I saw Relay automates payment retries.",
        "It reminded me of the telemetry work I did at Jio.",
        "That work seems directly useful for the automation Relay provides.",
        "Is the team taking interns?",
    )
    assert any("connect your experience" in i for i in check_voice(body, [], "Neil Baranwal"))
    dump = para("Hi A,", "At Jio I built a zero-allocation Rust UDP telemetry ingestion daemon with Prometheus.", "Hiring?")
    assert any("tech-stack list" in i for i in check_voice(dump, [], "Neil Baranwal"))


def test_who_i_am_sentence_is_not_forced_to_change():
    first = para("Hi A,", "I saw your refunds launch.", "I'm a pre-final-year B.Tech CSE student at VIT Chennai looking for a role.", "Any interns?")
    second = para("Hi B,", "Your fraud write-up was clear.", "I'm a pre-final-year B.Tech CSE student at VIT Chennai looking for a role.", "Hiring interns?")
    assert not any("built like one" in i for i in check_voice(second, [first], "Neil Baranwal"))


def test_plain_highlights_are_generated_once_and_checked(tmp_path):
    from src.config import Highlight
    from src.pipeline.schemas import PlainSummarySchema
    from src.pipeline.stages import plain_highlights

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    calls = []

    class Helper:
        def __init__(self, text):
            self.text = text

        def generate_json(self, prompt, schema, system_prompt=None):
            calls.append(prompt)
            return PlainSummarySchema(summary=self.text)

    jio = Highlight(name="Jio Platforms internship", summary="Built a zero-allocation Rust UDP telemetry ingestion daemon")
    good = "At Jio Platforms I worked on a system that collects network data and flags faults as they happen."
    assert plain_highlights(session, Helper(good), [jio], "resume text") == [good]
    assert plain_highlights(session, Helper("ignored"), [jio], "resume text") == [good]  # cached
    assert len(calls) == 1
    other = Highlight(name="Team Ignition", summary="Telemetry dashboard for rockets")
    assert plain_highlights(session, Helper("Team Ignition: I won first place using Kubernetes."), [other], "resume") == []
    assert plain_highlights(session, None, [jio], "resume") == []


def test_simplify_jargon_swaps_matching_sentence_only():
    from src.pipeline.stages import simplify_jargon

    body = para(
        "Hi A,",
        "I saw your Relay launch.",
        "At Jio Platforms I built a zero‑allocation Rust UDP telemetry ingestion daemon with Prometheus/Grafana observability.",
        "Is the team taking interns?",
    )
    plain = {"Jio Platforms internship": "At Jio Platforms I worked on a system that collects network data and flags faults"}
    fixed = simplify_jargon(body, plain, "Neil Baranwal")
    assert "zero" not in fixed and "flags faults." in fixed and "I saw your Relay launch." in fixed
    assert simplify_jargon(body, {"Team Ignition": "Built rocket dashboards"}, "Neil Baranwal") == body.replace("‑", "-")
