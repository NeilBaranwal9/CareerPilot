"""Resume variants (AI / Backend / Fintech / Systems ...), highlight prioritisation and Typst text extraction."""

import logging
import os
import re
from dataclasses import dataclass, field

from src.config import AppConfig, Highlight, ResumeVariant, VariantRule
from src.db.models import Company, Job
from src.intel.classify import keyword_in

AI_KEYWORDS = [
    "ai", "llm", "genai", "generative ai", "machine learning", "ml", "data science", "nlp", "prompt", "rag",
    "vertex", "deep learning",
]
DEFAULT_VARIANT_NAMES = ("base", "default", "general", "backend")

logger = logging.getLogger("recruiting-platform.utils.resume")


@dataclass
class ResumeChoice:
    name: str
    path: str
    score: float
    reason: str
    matched: list[str] = field(default_factory=list)
    focus: str = ""


def _context_parts(job: Job, company: Company) -> tuple[str, str, list[str], set[str]]:
    title = (job.title or "").lower()
    description = (job.description or "").lower()
    stack: list[str] = []
    if isinstance(company.tech_stack, list):
        stack = [str(t).lower() for t in company.tech_stack]
    elif isinstance(company.research_data, dict):
        stack = [str(t).lower() for t in company.research_data.get("tech_stack", [])]
    sub_sectors = [str(s) for s in company.sub_sectors] if isinstance(company.sub_sectors, list) else []
    sectors = {s for s in [company.sector, *sub_sectors] if s}
    return title, description, stack, sectors


def _legacy_variants(config: AppConfig) -> list[ResumeVariant]:
    return [
        ResumeVariant(name="AI-focused", path=config.pipeline.ai_resume_path, tags=AI_KEYWORDS, sectors=["ai"]),
        ResumeVariant(name="Base", path=config.pipeline.base_resume_path, tags=[], sectors=[]),
    ]


def available_variants(config: AppConfig) -> list[ResumeVariant]:
    configured = [v for v in config.resumes if os.path.exists(v.path)]
    missing = [v.name for v in config.resumes if not os.path.exists(v.path)]
    if missing:
        logger.warning(f"Resume variants skipped because their file is missing: {missing}")
    if configured:
        return configured
    return [v for v in _legacy_variants(config) if os.path.exists(v.path)]


def score_variant(variant: ResumeVariant, job: Job, company: Company) -> tuple[float, list[str]]:
    title, description, stack, sectors = _context_parts(job, company)
    score = 0.0
    matched: list[str] = []
    for tag in variant.tags:
        hit = False
        if keyword_in(title, tag):
            score += 3
            hit = True
        if keyword_in(description, tag):
            score += 1
            hit = True
        if any(keyword_in(item, tag) for item in stack):
            score += 1
            hit = True
        if hit:
            matched.append(tag)
    overlap = sectors & set(variant.sectors)
    if overlap:
        score += 3
        matched.extend(f"sector:{s}" for s in overlap)
    return score, matched


def rule_matches(rule: VariantRule, job: Job, company: Company, persona: str | None) -> list[str] | None:
    """Returns the list of satisfied conditions when every set condition holds, else None."""
    title, description, stack, sectors = _context_parts(job, company)
    satisfied: list[str] = []
    if rule.sectors:
        primary = {company.sector} if company.sector else set()
        hit = primary & {s.lower() for s in rule.sectors}
        if not hit:
            return None
        satisfied.append(f"sector={','.join(sorted(hit))}")
    if rule.title_any:
        hits = [k for k in rule.title_any if keyword_in(title, k)]
        if not hits:
            return None
        satisfied.append(f"title~{hits[0]}")
    if rule.description_any:
        hits = [k for k in rule.description_any if keyword_in(description, k)]
        if not hits:
            return None
        satisfied.append(f"description~{hits[0]}")
    if rule.stack_any:
        hits = [k for k in rule.stack_any if any(keyword_in(item, k) for item in stack)]
        if not hits:
            return None
        satisfied.append(f"stack~{hits[0]}")
    if rule.personas:
        if (persona or "") not in rule.personas:
            return None
        satisfied.append(f"persona={persona}")
    if rule.title_none and any(keyword_in(title, k) for k in rule.title_none):
        return None
    return satisfied or ["(unconditional rule)"]


def select_resume_variant(
    config: AppConfig, job: Job, company: Company, persona: str | None = None
) -> ResumeChoice | None:
    """
    Chooses the resume variant for a role.
    Rule mode (any variant defines `rules` or `default`): the highest-priority variant with a matching rule wins,
    ties go to config order, otherwise the `default` variant. Without rules: tag/sector scoring.
    """
    variants = available_variants(config)
    if not variants:
        return None

    if any(v.rules or v.default for v in variants):
        matches = []
        for index, variant in enumerate(variants):
            for rule_index, rule in enumerate(variant.rules, start=1):
                satisfied = rule_matches(rule, job, company, persona)
                if satisfied is not None:
                    matches.append((variant.priority, -index, variant, rule_index, satisfied))
                    break
        if matches:
            matches.sort(key=lambda m: (m[0], m[1]), reverse=True)
            _prio, _idx, best, rule_index, satisfied = matches[0]
            reason = f"rule #{rule_index} of '{best.name}' matched ({'; '.join(satisfied)})"
            return ResumeChoice(best.name, best.path, float(best.priority), reason, satisfied, best.focus)
        default = next((v for v in variants if v.default), variants[0])
        return ResumeChoice(default.name, default.path, 0.0, "no rule matched; using default variant", [], default.focus)

    scored = []
    for index, variant in enumerate(variants):
        score, matched = score_variant(variant, job, company)
        is_default = variant.name.lower() in DEFAULT_VARIANT_NAMES
        scored.append((score, is_default, -index, variant, matched))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best_score, _is_default, _idx, best, matched = scored[0]
    reason = (
        f"matched {', '.join(matched[:6])}" if matched else "no specific signals; using default variant"
    )
    return ResumeChoice(best.name, best.path, best_score, reason, matched, best.focus)


def rank_highlights(config: AppConfig, job: Job, company: Company, persona: str | None = None) -> list[Highlight]:
    """Orders your highlights (e.g. Jio internship, GSIH finalist, Team Ignition) by relevance to this role."""
    if not config.highlights:
        return []
    title, description, stack, sectors = _context_parts(job, company)
    context = " ".join([title, description, " ".join(stack), " ".join(sectors), company.description or ""])
    scored: list[tuple[float, int, Highlight]] = []
    for index, highlight in enumerate(config.highlights):
        score = 0.0
        for tag in highlight.tags:
            if tag in sectors:
                score += 3
            if keyword_in(title, tag):
                score += 2
            if keyword_in(context, tag):
                score += 1
        if persona == "founder" and any(t in ("hackathon", "product", "startup") for t in highlight.tags):
            score += 1
        scored.append((score, -index, highlight))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [h for _s, _i, h in scored]


def format_highlights(highlights: list[Highlight]) -> str:
    if not highlights:
        return "(no highlights configured)"
    return "\n".join(
        f"{i}. {h.name}" + (f" — {h.summary}" if h.summary else "") for i, h in enumerate(highlights, start=1)
    )


def typst_to_text(source: str, limit: int = 3500) -> str:
    """Rough plain-text rendering of a Typst resume for LLM context (markup and settings removed)."""
    lines = []
    for line in (source or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if re.match(r"#(set|show|import|let|include|pagebreak|v\(|h\()", stripped):
            continue
        lines.append(stripped)
    text = "\n".join(lines)
    text = re.sub(r'#link\("([^"]*)"\)\[([^\]]*)\]', r"\2 (\1)", text)
    text = re.sub(r"#[a-zA-Z_][\w.-]*", " ", text)
    text = re.sub(r'[\[\]{}*_#=]|\\', " ", text)
    text = re.sub(r"\(\s*\)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text[:limit]


def read_resume_text(path: str | None, limit: int = 6000) -> str:
    """Plain text of a resume in any supported format (.pdf, .docx, .typ, .md, .txt)."""
    from src.utils.resume_pdf import extract_resume_text

    return extract_resume_text(path)[:limit]
