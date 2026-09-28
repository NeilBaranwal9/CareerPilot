"""
Outcome learning: turns past outreach results (sent -> replied -> interview) into smoothed per-feature
reply-rate multipliers, which feed company fit, contact ranking and response-probability estimates.
"""

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from src.config import AppConfig
from src.db.models import Application, Company, Contact, ResumeVersion
from src.intel.classify import effective_headcount, size_bucket

CONTACTED_STATUSES = {
    "sent", "followed_up", "replied", "interview", "not_interested", "no_response", "offer", "rejected", "bounced",
}
REPLIED_STATUSES = {"replied", "interview", "not_interested", "offer", "rejected"}
POSITIVE_STATUSES = {"replied", "interview", "offer"}

FEATURES = ("sector", "funding_stage", "size_bucket", "persona", "resume_variant", "company_source")

# Relative reply-likelihood priors per persona before there is enough data to learn from.
PERSONA_PRIORS: dict[str, float] = {
    "founder": 1.3,
    "hiring_manager": 1.25,
    "engineering_manager": 1.15,
    "recruiter": 1.15,
    "cto": 0.95,
    "vp_engineering": 0.85,
    "tech_lead": 0.8,
    "engineer": 0.6,
    "executive": 0.7,
    "generic_inbox": 0.35,
    "other": 0.6,
}


@dataclass
class FeatureStat:
    sent: int = 0
    replies: int = 0
    positives: int = 0
    interviews: int = 0


@dataclass
class OutcomeStats:
    total_sent: int = 0
    total_replies: int = 0
    total_positive: int = 0
    total_interviews: int = 0
    prior_rate: float = 0.08
    prior_strength: float = 10.0
    features: dict[str, dict[str, FeatureStat]] = field(default_factory=dict)

    @property
    def global_rate(self) -> float:
        """Smoothed overall reply rate (Beta prior centred on the configured prior)."""
        return (self.total_replies + self.prior_rate * self.prior_strength) / (self.total_sent + self.prior_strength)

    def reply_rate(self, feature: str, value: str | None) -> float:
        stat = self.features.get(feature, {}).get(value or "unknown")
        base = self.global_rate
        if not stat:
            return base
        return (stat.replies + base * self.prior_strength) / (stat.sent + self.prior_strength)

    def multiplier(self, feature: str, value: str | None) -> float:
        """Learned reply-rate multiplier for a feature value relative to the global rate, clipped to [0.4, 2.5]."""
        base = self.global_rate
        if base <= 0:
            return 1.0
        return max(0.4, min(2.5, self.reply_rate(feature, value) / base))

    def positive_multiplier(self, feature: str, value: str | None) -> float:
        """Like `multiplier` but on positive outcomes, where interviews count double."""
        stat = self.features.get(feature, {}).get(value or "unknown")
        total_pos = self.total_positive + self.total_interviews
        base = (total_pos + self.prior_rate * self.prior_strength) / (self.total_sent + self.prior_strength)
        if not stat or base <= 0:
            return 1.0
        rate = (stat.positives + stat.interviews + base * self.prior_strength) / (stat.sent + self.prior_strength)
        return max(0.4, min(2.5, rate / base))


def application_features(session: Session, app: Application) -> dict[str, str]:
    company: Company = app.job.company
    contact: Contact | None = app.contact
    rv = (
        session.query(ResumeVersion)
        .filter(ResumeVersion.application_id == app.id)
        .order_by(ResumeVersion.id.desc())
        .first()
    )
    return {
        "sector": company.sector or "generic",
        "funding_stage": company.funding_stage or "unknown",
        "size_bucket": size_bucket(company.employee_count),
        "persona": app.persona or (contact.role_category if contact else None) or "other",
        "resume_variant": (rv.variant if rv and rv.variant else "unknown"),
        "company_source": company.source or "unknown",
    }


def compute_outcome_stats(session: Session, config: AppConfig) -> OutcomeStats:
    stats = OutcomeStats(prior_rate=config.learning.prior_reply_rate, prior_strength=config.learning.prior_strength)
    if not config.learning.enabled:
        return stats
    apps = (
        session.query(Application)
        .filter(Application.outreach_status.in_(list(CONTACTED_STATUSES)))
        .all()
    )
    for app in apps:
        if app.outreach_status == "bounced":
            continue  # a bounce says nothing about the recipient's interest
        replied = app.outreach_status in REPLIED_STATUSES or app.replied_at is not None
        positive = app.outreach_status in POSITIVE_STATUSES
        interview = app.outreach_status in ("interview", "offer") or app.interview_at is not None
        stats.total_sent += 1
        stats.total_replies += int(replied)
        stats.total_positive += int(positive)
        stats.total_interviews += int(interview)
        for feature, value in application_features(session, app).items():
            bucket = stats.features.setdefault(feature, {}).setdefault(value, FeatureStat())
            bucket.sent += 1
            bucket.replies += int(replied)
            bucket.positives += int(positive)
            bucket.interviews += int(interview)
    return stats


def persona_multiplier(persona: str | None, employee_count: int | None, stats: OutcomeStats | None) -> float:
    """Persona prior adjusted for company size, then scaled by what has actually worked for you."""
    persona = persona or "other"
    prior = PERSONA_PRIORS.get(persona, 0.6)
    if employee_count is not None:
        if persona in ("founder", "cto") and employee_count <= 60:
            prior *= 1.3
        elif persona in ("founder", "cto", "executive") and employee_count > 300:
            prior *= 0.55
        elif persona == "recruiter" and employee_count <= 30:
            prior *= 0.6
    learned = stats.multiplier("persona", persona) if stats else 1.0
    return prior * learned


EMAIL_STATUS_FACTOR = {"valid": 1.0, "catch_all": 0.8, "unverified": 0.7, "unknown": 0.6, "invalid": 0.0}


def estimate_response_probability(
    company: Company,
    stats: OutcomeStats,
    persona: str | None = None,
    email_status: str | None = None,
) -> float:
    """Estimated probability that outreach to this company (and persona) gets a human reply."""
    probability = stats.global_rate
    probability *= stats.multiplier("sector", company.sector or "generic")
    probability *= stats.multiplier("funding_stage", company.funding_stage or "unknown")
    probability *= stats.multiplier("size_bucket", size_bucket(company.employee_count))
    probability *= stats.multiplier("company_source", company.source or "unknown")
    if company.hiring_status == "hiring":
        probability *= 1.2
    elif company.hiring_status == "no_public_openings":
        probability *= 0.85
    if persona:
        probability *= persona_multiplier(
            persona, effective_headcount(company.employee_count, company.funding_stage), stats
        )
    if email_status:
        probability *= EMAIL_STATUS_FACTOR.get(email_status, 0.6)
    return round(max(0.005, min(0.9, probability)), 4)
