"""Rule-based company fit and opportunity scoring (combined with the LLM score in hybrid mode)."""

import re
from dataclasses import dataclass, field
from typing import Any

from src.config import AppConfig
from src.db.models import Company, Job
from src.intel.classify import keyword_in, parse_salary_lpa
from src.intel.learning import OutcomeStats

SENIOR_TITLE_MARKERS = ["senior", "sr", "staff", "principal", "lead", "manager", "director", "head", "vp", "architect"]
ENTRY_TITLE_MARKERS = ["intern", "graduate", "new grad", "entry", "junior", "jr", "associate", "sde 1", "sde-1", "sde i"]
ENGINEERING_MARKERS = [
    "engineer", "developer", "sde", "software", "backend", "frontend", "full stack", "fullstack", "ml", "ai",
    "data", "platform", "infrastructure", "devops", "sre", "member of technical staff", "programmer", "quant",
]


@dataclass
class FitResult:
    score: float
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    rejected: bool = False
    reject_reason: str = ""


def sector_preference(sector: str | None, sub_sectors: list[str] | None, config: AppConfig) -> float:
    weights = config.target_profile.sector_weights
    default = weights.get("generic", 0.5)
    candidates = [s for s in [sector, *(sub_sectors or [])] if s]
    if not candidates:
        return default
    return max(weights.get(s, default) for s in candidates)


def tech_overlap(stack: list[str] | None, skills: list[str]) -> float | None:
    """Share of the company's stack (top items) covered by your skills; None when either side is unknown."""
    if not stack or not skills:
        return None
    lowered_skills = [s.lower() for s in skills]
    top = [t.lower() for t in stack][:8]
    matches = sum(1 for t in top if any(s in t or t in s for s in lowered_skills))
    return min(1.0, matches / max(1, min(len(top), 5)))


def compute_company_fit(company: Company, config: AppConfig, stats: OutcomeStats | None = None) -> FitResult:
    profile = config.target_profile
    prefs = config.job_preferences
    components: dict[str, float] = {}
    reasons: list[str] = []
    sub_sectors = list(company.sub_sectors or []) if isinstance(company.sub_sectors, list) else []

    # Sector preference, adjusted by what has historically produced replies/interviews.
    sector_pref = sector_preference(company.sector, sub_sectors, config)
    if stats is not None and stats.total_sent >= config.learning.min_samples:
        learned = stats.positive_multiplier("sector", company.sector or "generic")
        sector_pref = min(1.0, sector_pref * learned)
        if abs(learned - 1.0) > 0.1:
            reasons.append(f"learned sector multiplier x{learned:.2f}")
    components["sector"] = round(sector_pref, 3)
    reasons.append(f"sector={company.sector or 'unknown'} pref={sector_pref:.2f}")

    # Funding stage
    if profile.funding_stages:
        if company.funding_stage in profile.funding_stages:
            components["funding"] = 1.0
        elif not company.funding_stage:
            components["funding"] = 0.45
        else:
            components["funding"] = 0.2
            reasons.append(f"funding {company.funding_stage} not in {profile.funding_stages}")
    else:
        components["funding"] = 0.7

    # Size
    count = company.employee_count
    lo, hi = prefs.company_size.min_employees, prefs.company_size.max_employees
    if count is None:
        components["size"] = 0.6
    elif lo <= count <= hi:
        components["size"] = 1.0
    else:
        distance = (lo - count) / max(lo, 1) if count < lo else (count - hi) / max(hi, 1)
        components["size"] = round(max(0.2, 1.0 - distance), 3)

    # Hiring momentum
    components["hiring"] = {"hiring": 1.0, "no_public_openings": 0.6}.get(company.hiring_status or "", 0.7)

    # Tech overlap
    stack = company.tech_stack if isinstance(company.tech_stack, list) else None
    if stack is None and isinstance(company.research_data, dict):
        stack = company.research_data.get("tech_stack")
    overlap = tech_overlap(stack, profile.skills)
    components["tech"] = 0.5 if overlap is None else round(overlap, 3)

    # Geography
    location = (company.location or "").lower()
    geos = [g.lower() for g in prefs.geographies]
    if not location:
        components["geo"] = 0.75
    elif any(g in location for g in geos) or ("remote" in geos and "remote" in location):
        components["geo"] = 1.0
    else:
        components["geo"] = 0.4

    weights = {"sector": 0.35, "tech": 0.2, "funding": 0.15, "size": 0.1, "hiring": 0.1, "geo": 0.1}
    score = round(sum(components[k] * w for k, w in weights.items()), 3)

    rejected = False
    reject_reason = ""
    all_sectors = {s for s in [company.sector, *sub_sectors] if s}
    if profile.allowed_sectors and not (all_sectors & set(profile.allowed_sectors)):
        rejected = True
        reject_reason = f"sector {company.sector or 'unknown'} not in allowed sectors {profile.allowed_sectors}"
    elif profile.reject_below_fit and score < profile.min_company_fit:
        rejected = True
        reject_reason = f"fit {score:.2f} below minimum {profile.min_company_fit:.2f}"
    if any(ex.lower() in (company.name or "").lower() for ex in config.exclusions.companies):
        rejected = True
        reject_reason = "company is in exclusions list"
    if company.domain and any(company.domain.endswith(d.lower()) for d in config.exclusions.domains):
        rejected = True
        reject_reason = "domain is in exclusions list"

    return FitResult(score=score, components=components, reasons=reasons, rejected=rejected, reject_reason=reject_reason)


def title_relevance(title: str, roles: list[str], experience_years_max: float) -> float:
    """0..1 relevance of a job title for the configured roles and experience level."""
    lowered = (title or "").lower()
    if not lowered:
        return 0.0
    score = 0.0
    if any(r.lower() in lowered for r in roles):
        score = 1.0
    elif any(keyword_in(lowered, m) for m in ENGINEERING_MARKERS):
        score = 0.7
    if experience_years_max <= 2:
        if any(keyword_in(lowered, m) for m in SENIOR_TITLE_MARKERS):
            score *= 0.25
        if any(keyword_in(lowered, m) or m in lowered for m in ENTRY_TITLE_MARKERS):
            score = min(1.0, score + 0.2)
    # When every target role is an internship, full-time postings are not a fit.
    if roles and all("intern" in r.lower() for r in roles) and not re.search(r"\bintern(ship)?s?\b|trainee", lowered):
        score *= 0.3
    if "speculative" in lowered or "targeted outreach" in lowered:
        score = max(score, 0.75)
    return round(score, 3)


def rule_based_opportunity(job: Job, company: Company, config: AppConfig) -> dict[str, Any]:
    """Deterministic 0..1 scores on the same six dimensions the LLM scores."""
    prefs = config.job_preferences
    role_match = title_relevance(job.title, prefs.roles, prefs.experience_years_max)

    if job.experience_years_required is None:
        experience = 0.8
    elif job.experience_years_required <= prefs.experience_years_max:
        experience = 1.0
    else:
        experience = max(0.0, 1.0 - (job.experience_years_required - prefs.experience_years_max) / 3.0)
    role_match = round(role_match * 0.7 + experience * 0.3, 3)

    lo, hi = parse_salary_lpa(job.salary)
    if lo is None or hi is None:
        salary = 0.6
    elif hi < prefs.salary_range.min_lpa:
        salary = max(0.0, hi / prefs.salary_range.min_lpa - 0.3)
    else:
        salary = 1.0

    stack = company.tech_stack if isinstance(company.tech_stack, list) else None
    if stack is None and isinstance(company.research_data, dict):
        stack = company.research_data.get("tech_stack")
    overlap = tech_overlap(stack, config.target_profile.skills)
    if overlap is None and stack:
        text = " ".join(stack).lower()
        overlap = 0.8 if any(k in text for k in ("python", "java", "go", "node", "typescript", "react")) else 0.5
    tech = 0.5 if overlap is None else overlap

    company_quality = company.fit_score if company.fit_score is not None else 0.6
    growth = {"hiring": 0.9, "no_public_openings": 0.55}.get(company.hiring_status or "", 0.65)
    if company.funding_stage in ("series_a", "series_b", "series_c"):
        growth = min(1.0, growth + 0.1)

    filled = sum(
        1
        for v in (company.description, company.sector, company.funding_stage, stack, job.description, company.domain)
        if v
    )
    confidence = round(0.3 + 0.7 * filled / 6, 3)

    return {
        "role_match": round(role_match, 3),
        "tech_stack": round(tech, 3),
        "salary": round(salary, 3),
        "company_quality": round(company_quality, 3),
        "growth": round(growth, 3),
        "confidence": confidence,
    }


def weighted_total(components: dict[str, Any], config: AppConfig) -> float:
    w = config.scoring.weights
    return float(
        components["role_match"] * w.role_match
        + components["tech_stack"] * w.tech_stack
        + components["salary"] * w.salary
        + components["company_quality"] * w.company_quality
        + components["growth"] * w.growth
        + components["confidence"] * w.confidence
    )
