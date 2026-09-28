from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String, default="running")  # running, completed, failed

    applications: Mapped[list["Application"]] = relationship(back_populates="run")


class Campaign(Base):
    """A natural-language outreach goal, e.g. 'Find 200 fintech companies in India'."""

    __tablename__ = "campaigns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)
    goal: Mapped[str] = mapped_column(Text)
    spec: Mapped[Any | None] = mapped_column(JSON, nullable=True)  # parsed CampaignSpec
    target_companies: Mapped[int] = mapped_column(Integer, default=50)
    personas: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    auto_send: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    status: Mapped[str] = mapped_column(String, default="active")  # active, paused, completed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    domain: Mapped[str | None] = mapped_column(String, nullable=True)
    employee_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    industry: Mapped[str | None] = mapped_column(String, nullable=True)
    research_data: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    # --- Intelligence ---
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    sector: Mapped[str | None] = mapped_column(String, nullable=True)  # fintech, healthtech, edtech, ai, ...
    sub_sectors: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    funding_stage: Mapped[str | None] = mapped_column(String, nullable=True)  # seed, series_a, ...
    funding_details: Mapped[str | None] = mapped_column(Text, nullable=True)
    hiring_status: Mapped[str | None] = mapped_column(String, nullable=True)  # hiring, no_public_openings, unknown
    open_roles_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tech_stack: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    recent_news: Mapped[Any | None] = mapped_column(JSON, nullable=True)  # [{title, date, summary, url}]
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    linkedin_url: Mapped[str | None] = mapped_column(String, nullable=True)
    careers_url: Mapped[str | None] = mapped_column(String, nullable=True)
    ats_provider: Mapped[str | None] = mapped_column(String, nullable=True)  # greenhouse, lever, ashby, ...
    ats_token: Mapped[str | None] = mapped_column(String, nullable=True)
    github_org: Mapped[str | None] = mapped_column(String, nullable=True)
    extra_data: Mapped[Any | None] = mapped_column(JSON, nullable=True)

    # --- Discovery provenance ---
    source: Mapped[str | None] = mapped_column(String, nullable=True)  # llm, web_search, yc, linkedin_jobs, ...
    discovery_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    campaign_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("campaigns.id"), nullable=True, index=True)

    # --- Scoring & lifecycle ---
    fit_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    fit_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str | None] = mapped_column(String, nullable=True)  # candidate, target, rejected, contacted
    email_pattern: Mapped[str | None] = mapped_column(String, nullable=True)  # e.g. {first}.{last}
    is_catch_all: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    last_researched_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_contacted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    jobs: Mapped[list["Job"]] = relationship(back_populates="company")
    contacts: Mapped[list["Contact"]] = relationship(back_populates="company")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(Integer, ForeignKey("companies.id"), index=True)
    title: Mapped[str] = mapped_column(String)
    url: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    salary: Mapped[str | None] = mapped_column(String, nullable=True)
    salary_min_lpa: Mapped[float | None] = mapped_column(Float, nullable=True)
    salary_max_lpa: Mapped[float | None] = mapped_column(Float, nullable=True)
    experience_years_required: Mapped[float | None] = mapped_column(Float, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    # greenhouse, lever, ashby, workable, smartrecruiters, linkedin, wellfound, indeed, career_page, speculative, pasted
    source: Mapped[str | None] = mapped_column(String, nullable=True)
    posted_at: Mapped[str | None] = mapped_column(String, nullable=True)

    company: Mapped[Company] = relationship(back_populates="jobs")
    applications: Mapped[list["Application"]] = relationship(back_populates="job")


class Contact(Base):
    __tablename__ = "contacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(Integer, ForeignKey("companies.id"), index=True)
    name: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String)
    email: Mapped[str | None] = mapped_column(String, nullable=True, unique=True)
    linkedin_url: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # recruiter, engineering_manager, hiring_manager, founder, cto, vp_engineering, tech_lead, engineer, executive, generic_inbox, other
    role_category: Mapped[str | None] = mapped_column(String, nullable=True)
    seniority: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str | None] = mapped_column(String, nullable=True)  # linkedin_search, team_page, press, github, hunter, apollo
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    background: Mapped[str | None] = mapped_column(Text, nullable=True)
    github_url: Mapped[str | None] = mapped_column(String, nullable=True)
    # valid, catch_all, unverified, invalid, unknown
    email_status: Mapped[str | None] = mapped_column(String, nullable=True)
    email_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    email_source: Mapped[str | None] = mapped_column(String, nullable=True)
    email_verified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    rejected_emails: Mapped[Any | None] = mapped_column(JSON, nullable=True)  # bounced / invalid addresses
    do_not_contact: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    last_contacted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    company: Mapped[Company] = relationship(back_populates="contacts")
    applications: Mapped[list["Application"]] = relationship(back_populates="contact")


class Application(Base):
    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.id"), index=True)
    job_id: Mapped[int] = mapped_column(Integer, ForeignKey("jobs.id"), index=True)
    contact_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("contacts.id"), nullable=True)
    current_stage: Mapped[int] = mapped_column(Integer, default=0)  # 0 to 12
    state: Mapped[str] = mapped_column(String, default="Company Discovery")  # Terminal states or stages
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_breakdown: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    tailored_resume_path: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    campaign_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("campaigns.id"), nullable=True, index=True)
    persona: Mapped[str | None] = mapped_column(String, nullable=True)
    # drafted, scheduled, sent, followed_up, replied, interview, not_interested, bounced, no_response, offer, rejected
    outreach_status: Mapped[str | None] = mapped_column(String, nullable=True)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    replied_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    interview_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reply_category: Mapped[str | None] = mapped_column(String, nullable=True)
    reply_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_probability: Mapped[float | None] = mapped_column(Float, nullable=True)

    run: Mapped[Run] = relationship(back_populates="applications")
    job: Mapped[Job] = relationship(back_populates="applications")
    contact: Mapped[Contact | None] = relationship(back_populates="applications")
    emails: Mapped[list["Email"]] = relationship(back_populates="application")
    resume_versions: Mapped[list["ResumeVersion"]] = relationship(back_populates="application")
    history_records: Mapped[list["History"]] = relationship(back_populates="application")
    events: Mapped[list["OutreachEvent"]] = relationship(back_populates="application")


class Email(Base):
    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[int] = mapped_column(Integer, ForeignKey("applications.id"), index=True)
    subject: Mapped[str] = mapped_column(String)
    body: Mapped[str] = mapped_column(Text)
    gmail_draft_id: Mapped[str | None] = mapped_column(String, nullable=True)
    # generated, draft_created, scheduled, sent, pending (follow-up waiting), cancelled, replied, bounced
    status: Mapped[str] = mapped_column(String, default="draft_created")
    scheduled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    sequence_step: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 0 = initial, 1..n = follow-ups
    persona: Mapped[str | None] = mapped_column(String, nullable=True)
    tone: Mapped[str | None] = mapped_column(String, nullable=True)
    to_email: Mapped[str | None] = mapped_column(String, nullable=True)
    gmail_message_id: Mapped[str | None] = mapped_column(String, nullable=True)
    gmail_thread_id: Mapped[str | None] = mapped_column(String, nullable=True)
    rfc_message_id: Mapped[str | None] = mapped_column(String, nullable=True)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    application: Mapped[Application] = relationship(back_populates="emails")


class ResumeVersion(Base):
    __tablename__ = "resume_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[int] = mapped_column(Integer, ForeignKey("applications.id"), index=True)
    parent_resume: Mapped[str] = mapped_column(String)
    company: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String)
    keywords_added: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    reasoning: Mapped[str] = mapped_column(Text)
    path: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    variant: Mapped[str | None] = mapped_column(String, nullable=True)
    highlights_order: Mapped[Any | None] = mapped_column(JSON, nullable=True)

    application: Mapped[Application] = relationship(back_populates="resume_versions")


class History(Base):
    __tablename__ = "history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[int] = mapped_column(Integer, ForeignKey("applications.id"), index=True)
    stage: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    application: Mapped[Application] = relationship(back_populates="history_records")


class OutreachEvent(Base):
    """Timeline of outreach events: sent, follow-up sent, reply, bounce, interview, status changes."""

    __tablename__ = "outreach_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[int] = mapped_column(Integer, ForeignKey("applications.id"), index=True)
    email_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("emails.id"), nullable=True)
    event_type: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    gmail_message_id: Mapped[str | None] = mapped_column(String, nullable=True)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)

    application: Mapped[Application] = relationship(back_populates="events")


class CacheEntry(Base):
    __tablename__ = "cache_entries"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
