"""
Campaigns: natural-language outreach goals such as
  "Find 200 fintech companies in India, reach engineering managers or recruiters"
parsed into a DiscoverySpec and driven to completion across daily runs.
"""

import logging
from typing import Any

from sqlalchemy import or_
from sqlalchemy.orm import Session

from src.db.models import Application, Campaign, Company, Email
from src.intel.classify import SECTOR_KEYWORDS, normalize_funding_stage, normalize_sector
from src.pipeline.schemas import CampaignSpecSchema
from src.providers.llm import BaseLLMProvider
from src.sources.companies import DiscoverySpec, parse_spec_fallback

logger = logging.getLogger("recruiting-platform.pipeline.campaign")

VALID_PERSONAS = {"engineering_manager", "recruiter", "founder", "cto", "hiring_manager", "tech_lead", "vp_engineering"}


def parse_goal(llm: BaseLLMProvider | None, goal: str, default_count: int = 25) -> DiscoverySpec:
    """Parses a goal with the LLM, filling any gaps with the deterministic parser."""
    fallback = parse_spec_fallback(goal, default_count)
    if llm is None:
        return fallback
    try:
        parsed = llm.generate_json(
            "Extract a structured company-search specification from this job-outreach goal. "
            f"Canonical sectors are: {sorted(SECTOR_KEYWORDS)}. Goal: '{goal}'",
            CampaignSpecSchema,
        )
        assert isinstance(parsed, CampaignSpecSchema)
    except Exception as e:
        logger.info(f"LLM goal parsing unavailable ({e}); using rule-based parse.")
        return fallback

    sectors = [s for s in (normalize_sector(x) for x in parsed.sectors) if s] or fallback.sectors
    stages = [s for s in (normalize_funding_stage(x) for x in parsed.funding_stages) if s] or fallback.funding_stages
    personas = [p for p in parsed.personas if p in VALID_PERSONAS] or fallback.personas
    return DiscoverySpec(
        query=goal.strip(),
        sectors=list(dict.fromkeys(sectors)),
        geographies=parsed.geographies or fallback.geographies,
        funding_stages=list(dict.fromkeys(stages)),
        min_employees=parsed.min_employees or fallback.min_employees,
        max_employees=parsed.max_employees or fallback.max_employees,
        count=parsed.company_count or fallback.count,
        keywords=parsed.keywords,
        role_titles=parsed.role_titles,
        personas=personas,
    )


def create_campaign(
    session: Session,
    goal: str,
    spec: DiscoverySpec,
    target: int | None = None,
    personas: list[str] | None = None,
    auto_send: bool | None = None,
) -> Campaign:
    campaign = Campaign(
        name=goal[:80],
        goal=goal,
        spec=spec.to_dict(),
        target_companies=target or spec.count,
        personas=personas or spec.personas or None,
        auto_send=auto_send,
        status="active",
    )
    session.add(campaign)
    session.commit()
    return campaign


def campaign_spec(campaign: Campaign) -> DiscoverySpec:
    return DiscoverySpec.from_dict(dict(campaign.spec or {"query": campaign.goal}))


def qualified_company_count(session: Session, campaign_id: int) -> int:
    return (
        session.query(Company)
        .filter(Company.campaign_id == campaign_id, or_(Company.status.is_(None), Company.status != "rejected"))
        .count()
    )


def campaign_progress(session: Session, campaign: Campaign) -> dict[str, Any]:
    companies = session.query(Company).filter(Company.campaign_id == campaign.id).all()
    apps = session.query(Application).filter(Application.campaign_id == campaign.id).all()
    app_ids = [a.id for a in apps]
    drafted = 0
    if app_ids:
        drafted = (
            session.query(Email)
            .filter(Email.application_id.in_(app_ids), Email.gmail_draft_id.isnot(None), or_(Email.sequence_step == 0, Email.sequence_step.is_(None)))
            .count()
        )
    return {
        "id": campaign.id,
        "goal": campaign.goal,
        "status": campaign.status,
        "target": campaign.target_companies,
        "companies_found": len(companies),
        "companies_qualified": sum(1 for c in companies if c.status != "rejected"),
        "companies_rejected": sum(1 for c in companies if c.status == "rejected"),
        "applications": len(apps),
        "drafts": drafted,
        "sent": sum(1 for a in apps if a.sent_at),
        "replies": sum(1 for a in apps if a.replied_at),
        "interviews": sum(1 for a in apps if a.interview_at or a.outreach_status in ("interview", "offer")),
    }
