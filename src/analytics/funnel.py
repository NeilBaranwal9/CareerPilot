"""Conversion funnel and outreach analytics."""

from typing import Any

from sqlalchemy import or_
from sqlalchemy.orm import Session

from src.db.models import Application, Company, Contact, Email, Job
from src.intel.learning import CONTACTED_STATUSES, REPLIED_STATUSES, application_features

FOUND_EMAIL_STATUSES = ("valid", "catch_all", "unverified")


def _company_ids(session: Session, campaign_id: int | None) -> set[int]:
    query = session.query(Company.id)
    if campaign_id is None:
        return {row[0] for row in query.all()}
    ids = {row[0] for row in query.filter(Company.campaign_id == campaign_id).all()}
    ids |= {
        row[0]
        for row in session.query(Job.company_id)
        .join(Application, Application.job_id == Job.id)
        .filter(Application.campaign_id == campaign_id)
        .all()
    }
    return ids


def _applications(session: Session, campaign_id: int | None) -> list[Application]:
    query = session.query(Application)
    if campaign_id is not None:
        query = query.filter(Application.campaign_id == campaign_id)
    return query.all()


def compute_funnel(session: Session, campaign_id: int | None = None) -> dict[str, Any]:
    company_ids = _company_ids(session, campaign_id)
    apps = _applications(session, campaign_id)

    companies = session.query(Company).filter(Company.id.in_(company_ids)).all() if company_ids else []
    rejected = sum(1 for c in companies if c.status == "rejected")
    qualified = sum(1 for c in companies if c.status in ("target", "contacted"))

    contacts = session.query(Contact).filter(Contact.company_id.in_(company_ids)).all() if company_ids else []
    companies_with_contacts = {c.company_id for c in contacts}
    email_contacts = [c for c in contacts if c.email and (c.email_status in FOUND_EMAIL_STATUSES or c.email_status is None)]
    companies_with_email = {c.company_id for c in email_contacts}
    companies_with_verified = {c.company_id for c in email_contacts if c.email_status == "valid"}

    app_ids = [a.id for a in apps]
    drafted_ids: set[int] = set()
    sent_ids: set[int] = set()
    if app_ids:
        initial_emails = (
            session.query(Email)
            .filter(Email.application_id.in_(app_ids), or_(Email.sequence_step == 0, Email.sequence_step.is_(None)))
            .all()
        )
        drafted_ids = {e.application_id for e in initial_emails if e.gmail_draft_id or e.status == "sent"}
        sent_ids = {e.application_id for e in initial_emails if e.status in ("sent", "replied") or e.sent_at}
    sent_ids |= {a.id for a in apps if a.sent_at or a.outreach_status in CONTACTED_STATUSES - {"bounced"}}
    replied_ids = {a.id for a in apps if a.replied_at or a.outreach_status in REPLIED_STATUSES}
    interview_ids = {a.id for a in apps if a.interview_at or a.outreach_status in ("interview", "offer")}
    bounced = sum(1 for a in apps if a.outreach_status == "bounced")

    stages = [
        ("Companies Found", len(company_ids)),
        ("Contacts Found", len(companies_with_contacts)),
        ("Emails Found", len(companies_with_email)),
        ("Emails Verified", len(companies_with_verified)),
        ("Drafts Created", len(drafted_ids)),
        ("Emails Sent", len(sent_ids)),
        ("Replies", len(replied_ids)),
        ("Interviews", len(interview_ids)),
    ]
    top = stages[0][1] or 1
    funnel = []
    previous: int | None = None
    for name, count in stages:
        funnel.append(
            {
                "stage": name,
                "count": count,
                "pct_of_top": round(100.0 * count / top, 1),
                "pct_of_previous": round(100.0 * count / previous, 1) if previous else None,
            }
        )
        previous = count or None

    return {
        "campaign_id": campaign_id,
        "funnel": funnel,
        "extra": {
            "qualified_companies": qualified,
            "rejected_companies": rejected,
            "total_contacts": len(contacts),
            "bounced": bounced,
            "reply_rate": round(100.0 * len(replied_ids) / len(sent_ids), 1) if sent_ids else 0.0,
            "interview_rate": round(100.0 * len(interview_ids) / len(sent_ids), 1) if sent_ids else 0.0,
        },
    }


def email_verification_stats(session: Session) -> dict[str, Any]:
    rows = session.query(Contact.email_status).filter(Contact.email.isnot(None)).all()
    counts: dict[str, int] = {}
    for (status,) in rows:
        key = status or "unchecked"
        counts[key] = counts.get(key, 0) + 1
    checked = sum(v for k, v in counts.items() if k != "unchecked")
    valid = counts.get("valid", 0)
    likely = valid + counts.get("catch_all", 0)
    return {
        "counts": counts,
        "checked": checked,
        "smtp_verified_pct": round(100.0 * valid / checked, 1) if checked else 0.0,
        "deliverable_likely_pct": round(100.0 * likely / checked, 1) if checked else 0.0,
        "catch_all_domains": session.query(Company).filter(Company.is_catch_all.is_(True)).count(),
    }


def outcome_breakdown(session: Session, feature: str, campaign_id: int | None = None) -> list[dict[str, Any]]:
    """Sent / replies / interviews per value of a feature (sector, persona, resume_variant, funding_stage, ...)."""
    table: dict[str, dict[str, int]] = {}
    for app in _applications(session, campaign_id):
        if not (app.sent_at or app.outreach_status in CONTACTED_STATUSES):
            continue
        if app.outreach_status == "bounced":
            continue
        value = application_features(session, app).get(feature, "unknown")
        row = table.setdefault(value, {"sent": 0, "replies": 0, "interviews": 0})
        row["sent"] += 1
        row["replies"] += int(bool(app.replied_at) or app.outreach_status in REPLIED_STATUSES)
        row["interviews"] += int(bool(app.interview_at) or app.outreach_status in ("interview", "offer"))
    result = [
        {
            "value": value,
            **row,
            "reply_rate": round(100.0 * row["replies"] / row["sent"], 1) if row["sent"] else 0.0,
        }
        for value, row in table.items()
    ]
    return sorted(result, key=lambda r: (r["reply_rate"], r["sent"]), reverse=True)
