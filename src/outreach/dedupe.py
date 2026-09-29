"""
Duplicate-outreach prevention. Every outreach thread that reaches a Gmail draft is written to `outreach_ledger`
(company, role, contact, email). Before a new draft is created the ledger is checked so the same email, person,
company or role is never contacted twice.
"""

import re

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from src.db.models import Application, Contact, Email, OutreachLedger
from src.sources.companies import normalize_company_name

_ROLE_NOISE = re.compile(r"\((?:speculative application|targeted outreach)\)|\b(intern(ship)?|trainee)\b", re.IGNORECASE)


def role_key(title: str | None) -> str:
    """Normalised role family: 'Software Engineer Intern (Speculative Application)' -> 'software engineer'."""
    cleaned = _ROLE_NOISE.sub(" ", title or "")
    return " ".join(re.findall(r"[a-z0-9+#]+", cleaned.lower()))


def contact_key(company_id: int | None, name: str | None) -> str | None:
    if company_id is None or not name:
        return None
    return f"{company_id}:{re.sub(r'[^a-z0-9]', '', name.lower())}"


def find_duplicate(
    session: Session, app: Application, contact: Contact, email: str, allow_multiple_contacts: bool = False
) -> str | None:
    """Reason this outreach would duplicate an earlier thread, or None if it is new."""
    company = app.job.company
    others = session.query(OutreachLedger).filter(
        or_(OutreachLedger.application_id.is_(None), OutreachLedger.application_id != app.id),
        OutreachLedger.status != "cancelled",
    )
    hit = others.filter(func.lower(OutreachLedger.email) == email.lower()).first()
    if hit:
        return f"email {email} was already contacted (application #{hit.application_id})"
    ckey = contact_key(company.id, contact.name)
    if ckey:
        hit = others.filter(OutreachLedger.contact_key == ckey).first()
        if hit:
            return f"{contact.name} was already contacted (application #{hit.application_id})"
    company_filter = OutreachLedger.company_key == normalize_company_name(company.name)
    if company.domain:
        company_filter = or_(company_filter, OutreachLedger.company_domain == company.domain.lower())
    same_company = others.filter(company_filter)
    if not allow_multiple_contacts:
        hit = same_company.first()
        if hit:
            return f"{company.name} was already contacted via {hit.contact_name or hit.email} (application #{hit.application_id})"
    hit = same_company.filter(OutreachLedger.role_key == role_key(app.job.title)).first()
    if hit:
        return f"the {app.job.title} role at {company.name} was already contacted (application #{hit.application_id})"
    return None


def record_outreach(session: Session, app: Application, contact: Contact, email: str, status: str = "drafted") -> None:
    company = app.job.company
    existing = session.query(OutreachLedger).filter(OutreachLedger.application_id == app.id).first()
    if existing:
        existing.status = status
        existing.email = email.lower()
        return
    session.add(
        OutreachLedger(
            application_id=app.id,
            company_id=company.id,
            company_key=normalize_company_name(company.name),
            company_domain=(company.domain or "").lower() or None,
            role_key=role_key(app.job.title),
            contact_key=contact_key(company.id, contact.name),
            contact_name=contact.name,
            email=email.lower(),
            status=status,
        )
    )


def update_ledger_status(session: Session, application_id: int, status: str) -> None:
    for row in session.query(OutreachLedger).filter(OutreachLedger.application_id == application_id).all():
        row.status = status


def backfill_ledger(session: Session) -> int:
    """Adds ledger rows for drafts/sends that pre-date the ledger (idempotent)."""
    added = 0
    rows = (
        session.query(Email)
        .filter(
            or_(Email.sequence_step == 0, Email.sequence_step.is_(None)),
            or_(Email.gmail_draft_id.isnot(None), Email.status.in_(["sent", "replied"])),
        )
        .all()
    )
    for email in rows:
        app = email.application
        contact = app.contact
        address = email.to_email or (contact.email if contact else None)
        if contact is None or not address:
            continue
        if session.query(OutreachLedger).filter(OutreachLedger.application_id == app.id).first():
            continue
        record_outreach(session, app, contact, address, "sent" if email.status in ("sent", "replied") else "drafted")
        added += 1
    if added:
        session.commit()
    return added
