"""
Professional email discovery and verification.

Evidence (highest confidence first): address supplied by a source (Hunter/Apollo/GitHub) -> address printed on
the company site or in commits -> company pattern inferred from known addresses -> Hunter/Apollo finders ->
LLM reading search results -> common-pattern permutations.
Verification: SMTP RCPT probe with catch-all detection, then Hunter's verifier, then evidence-weighted acceptance.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.config import AppConfig
from src.db.models import Company
from src.pipeline.schemas import EmailDiscoveryResponse
from src.providers.browser import BrowserProvider
from src.providers.llm import BaseLLMProvider
from src.sources.companies import resolve_company_domain
from src.sources.enrichment import ApolloClient, HunterClient
from src.utils.email_verifier import (
    GENERIC_INBOXES,
    apply_pattern,
    infer_email_pattern,
    is_placeholder_name,
    normalize_name_parts,
    rank_email_candidates,
    smtp_probe,
    verify_domain_mx,
    verify_email_syntax,
)

logger = logging.getLogger("recruiting-platform.sources.emails")


@dataclass
class EmailResult:
    email: str | None
    status: str  # valid | catch_all | unverified | invalid | unknown
    confidence: float
    source: str
    detail: str = ""
    catch_all: bool | None = None
    pattern: str | None = None
    rejected: list[str] = field(default_factory=list)


def _llm_guess(
    llm: BaseLLMProvider, browser: BrowserProvider, name: str, company: Company, domain: str
) -> tuple[str | None, str]:
    results = browser.search_google(f"'{name}' email '{domain}'", num_results=2)
    pattern_results = browser.search_google(f"\"{domain}\" email format OR \"email pattern\"", num_results=2)
    scraped = ""
    for r in results + pattern_results:
        if "example.com/search" in r.get("url", ""):
            continue
        scraped += f"\n--- {r.get('title', '')} ({r.get('url', '')}) ---\n{r.get('snippet', '')}\n"
        try:
            page = browser.fetch_page(r["url"], use_playwright=False)
            scraped += browser.extract_text(page)[:1500]
        except Exception as e:
            logger.debug(f"Error scraping {r.get('url')}: {e}")
    prompt = (
        f"Based on this search content and the contact details:\n"
        f"Contact Name: {name}\nCompany: {company.name}\nDomain: {domain}\n\n"
        f"Scraped Web Content:\n{scraped}\n\n"
        f"Determine or predict the professional email address of {name}. "
        f"Use common company email patterns (e.g. first.last@company.com, first@company.com) if explicit not found. "
        f"If it's impossible to deduce, leave the email field null."
    )
    response = llm.generate_json(prompt, EmailDiscoveryResponse)
    assert isinstance(response, EmailDiscoveryResponse)
    return response.email, scraped


def find_contact_email(
    config: AppConfig,
    llm: BaseLLMProvider,
    browser: BrowserProvider,
    company: Company,
    contact_name: str,
    hint_email: str | None = None,
    hint_confidence: float = 0.0,
    hint_source: str = "source",
    observed_emails: list[str] | None = None,
    email_samples: list[tuple[str, str]] | None = None,
    known_pattern: str | None = None,
    accept_all_hint: bool | None = None,
    rejected: list[str] | None = None,
) -> EmailResult:
    verification = config.email_verification
    rejected_set = {e.lower() for e in (rejected or [])} | {e.lower() for e in config.exclusions.emails}

    domain = (company.domain or "").lower().replace("www.", "") or None
    if not domain:
        domain = resolve_company_domain(browser, company.name)
        if domain:
            company.domain = domain
    if not domain:
        domain = f"{company.name.lower().replace(' ', '')}.com"
    if any(domain.endswith(d.lower()) for d in config.exclusions.domains):
        return EmailResult(None, "invalid", 0.0, "excluded", f"domain {domain} is excluded")
    if not verify_domain_mx(domain):
        return EmailResult(None, "invalid", 0.0, "mx", f"Domain '{domain}' has no MX records")

    candidates: dict[str, tuple[float, str]] = {}

    def add(email: str | None, confidence: float, source: str) -> None:
        if not email:
            return
        email = email.strip().lower()
        if not verify_email_syntax(email) or email in rejected_set:
            return
        email_domain = email.split("@")[-1]
        if email_domain != domain and not email_domain.endswith("." + domain):
            return
        if email not in candidates or candidates[email][0] < confidence:
            candidates[email] = (confidence, source)

    placeholder = is_placeholder_name(contact_name)
    observed = [e.lower() for e in (observed_emails or [])]
    samples = list(email_samples or [])

    add(hint_email, hint_confidence or 0.7, hint_source)

    if placeholder:
        for email in observed:
            if email.split("@")[0] in GENERIC_INBOXES:
                add(email, 0.65, "website")
    else:
        parts = normalize_name_parts(contact_name)
        first, last = (parts[0], parts[-1]) if len(parts) > 1 else (parts[0] if parts else "", "")

        hunter_key = config.api_keys.get("hunter")
        if hunter_key and first and last:
            try:
                found, score = HunterClient(browser, hunter_key).email_finder(domain, first, last)
                add(found, max(0.5, score), "hunter")
            except Exception as e:
                logger.info(f"Hunter email finder failed for {contact_name}: {e}")

        apollo_key = config.api_keys.get("apollo")
        if apollo_key and first and last:
            try:
                found, score = ApolloClient(browser, apollo_key).match_person(first, last, domain)
                add(found, score, "apollo")
            except Exception as e:
                logger.info(f"Apollo match failed for {contact_name}: {e}")

        pattern, pattern_conf = known_pattern, 0.8 if known_pattern else 0.0
        if not pattern and samples:
            pattern, pattern_conf = infer_email_pattern(samples, domain)
        if pattern:
            add(apply_pattern(pattern, contact_name, domain), max(0.6, min(0.9, 0.6 + 0.3 * pattern_conf)), "pattern")

        for email, _prior in rank_email_candidates(contact_name, domain):
            if email in observed:
                add(email, 0.9, "website")

        if max((c for c, _s in candidates.values()), default=0.0) < 0.6:
            try:
                guess, scraped = _llm_guess(llm, browser, contact_name, company, domain)
                if guess:
                    add(guess, 0.85 if guess.lower() in scraped.lower() else 0.45, "llm")
            except Exception as e:
                logger.warning(f"LLM failed to deduce email for {contact_name}: {e}")

    for email, prior in rank_email_candidates(contact_name, domain, pattern_confidence=0.0):
        add(email, prior, "permutation")

    ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)
    if not ranked:
        return EmailResult(None, "invalid", 0.0, "none", "no candidate addresses")

    inferred_pattern = known_pattern or (infer_email_pattern(samples, domain)[0] if samples else None)
    newly_rejected: list[str] = []
    catch_all = accept_all_hint

    # 1. SMTP probe (single session: catch-all test + candidates)
    if verification.smtp_check:
        to_probe = [email for email, _ in ranked[: verification.max_smtp_probes]]
        probe = smtp_probe(
            domain,
            to_probe,
            mail_from=verification.smtp_from or config.user_identity.email,
            helo_host=verification.helo_host,
            timeout=verification.smtp_timeout,
        )
        if probe.catch_all is not None:
            catch_all = probe.catch_all
            company.is_catch_all = probe.catch_all
        if probe.catch_all is False:
            for email, (_conf, source) in ranked:
                res = probe.results.get(email)
                if res is None:
                    continue
                if res.status == "valid":
                    if not placeholder:
                        pattern, _ = infer_email_pattern([(contact_name, email)], domain)
                        if pattern and not company.email_pattern:
                            company.email_pattern = pattern
                    return EmailResult(email, "valid", 0.97, source, f"SMTP accepted ({res.code})", False, pattern=inferred_pattern)
                if res.status == "invalid":
                    newly_rejected.append(email)
            ranked = [(e, v) for e, v in ranked if e not in newly_rejected]
            if not ranked:
                return EmailResult(None, "invalid", 0.0, "smtp", "all candidates rejected by mail server", False, rejected=newly_rejected)

    if catch_all:
        email, (conf, source) = ranked[0]
        if verification.allow_catch_all and conf >= verification.min_confidence:
            return EmailResult(
                email, "catch_all", round(min(conf, 0.8), 2), source,
                "domain accepts all addresses; chose strongest evidence", True, inferred_pattern, newly_rejected,
            )
        return EmailResult(None, "catch_all", conf, source, "catch-all domain and catch-all disallowed", True, rejected=newly_rejected)

    # 2. Hunter verifier when SMTP was inconclusive
    hunter_key = config.api_keys.get("hunter")
    if hunter_key:
        for email, (conf, source) in ranked[:2]:
            try:
                status = HunterClient(browser, hunter_key).verify(email)
            except Exception as e:
                logger.info(f"Hunter verification failed for {email}: {e}")
                break
            if status == "valid":
                return EmailResult(email, "valid", 0.95, source, "Hunter verified", catch_all, inferred_pattern, newly_rejected)
            if status == "catch_all":
                return EmailResult(email, "catch_all", round(min(conf, 0.8), 2), source, "Hunter: accept-all", True, inferred_pattern, newly_rejected)
            if status == "invalid":
                newly_rejected.append(email)
        ranked = [(e, v) for e, v in ranked if e not in newly_rejected]

    # 3. Evidence-weighted acceptance (SMTP unavailable/greylisted)
    if ranked and verification.allow_unverified:
        email, (conf, source) = ranked[0]
        if conf >= verification.min_confidence:
            return EmailResult(
                email, "unverified", round(conf, 2), source,
                "mailbox could not be verified; best evidence-backed candidate", catch_all, inferred_pattern, newly_rejected,
            )
    return EmailResult(None, "unknown", 0.0, "none", "no candidate met the confidence threshold", catch_all, rejected=newly_rejected)


def verification_timestamp() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
