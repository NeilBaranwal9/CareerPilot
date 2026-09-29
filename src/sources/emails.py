"""
Professional email discovery and verification.

Evidence (highest confidence first): address supplied by a source (Hunter/Apollo/GitHub) -> address printed on
the company site or in commits -> company pattern inferred from known addresses -> Hunter/Apollo finders ->
LLM reading search results -> common-pattern permutations.
Verification: SMTP RCPT probe with catch-all detection, then Hunter's verifier, then evidence-weighted acceptance.

Every result carries a confidence level and a human-readable evidence string:
  verified         mail server (or Hunter's verifier) confirmed the mailbox on a non-catch-all domain
  high_confidence  address found explicitly (Hunter, Apollo, company pages, public commits, search results)
  pattern_match    company's email pattern learned from other employees' real addresses
  catch_all        domain accepts any address, so the mailbox cannot be confirmed
  guessed          common naming pattern / LLM guess with no supporting evidence
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

CONFIDENCE_LEVELS = ("verified", "high_confidence", "pattern_match", "catch_all", "guessed")
_HIGH_CONFIDENCE_SOURCES = {"hunter", "apollo", "website", "github", "team_page", "provided", "source"}


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
    level: str = "guessed"  # verified | high_confidence | pattern_match | catch_all | guessed
    evidence: str = ""

    def as_evidence(self) -> dict[str, str | None]:
        return {"email": self.email, "confidence": self.level, "evidence": self.evidence}


def confidence_level(status: str, source: str, confidence: float) -> str:
    if status == "valid":
        return "verified"
    if status == "catch_all":
        return "catch_all"
    if source in _HIGH_CONFIDENCE_SOURCES or (source == "llm" and confidence >= 0.85):
        return "high_confidence"
    if source == "pattern":
        return "pattern_match"
    return "guessed"


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

    # email -> (confidence, source, evidence)
    candidates: dict[str, tuple[float, str, str]] = {}

    def add(email: str | None, confidence: float, source: str, evidence: str) -> None:
        if not email:
            return
        email = email.strip().lower()
        if not verify_email_syntax(email) or email in rejected_set:
            return
        email_domain = email.split("@")[-1]
        if email_domain != domain and not email_domain.endswith("." + domain):
            return
        if email not in candidates or candidates[email][0] < confidence:
            candidates[email] = (confidence, source, evidence)

    placeholder = is_placeholder_name(contact_name)
    observed = [e.lower() for e in (observed_emails or [])]
    samples = list(email_samples or [])

    hint_conf = hint_confidence or 0.7
    add(hint_email, hint_conf, hint_source, f"provided by {hint_source} (confidence {hint_conf:.2f})")

    if placeholder:
        for email in observed:
            if email.split("@")[0] in GENERIC_INBOXES:
                add(email, 0.65, "website", "hiring inbox listed on the company website")
    else:
        parts = normalize_name_parts(contact_name)
        first, last = (parts[0], parts[-1]) if len(parts) > 1 else (parts[0] if parts else "", "")

        hunter_key = config.api_keys.get("hunter")
        if hunter_key and first and last:
            try:
                found, score = HunterClient(browser, hunter_key).email_finder(domain, first, last)
                add(found, max(0.5, score), "hunter", f"Hunter email-finder (score {score:.0%})")
            except Exception as e:
                logger.info(f"Hunter email finder failed for {contact_name}: {e}")

        apollo_key = config.api_keys.get("apollo")
        if apollo_key and first and last:
            try:
                found, score = ApolloClient(browser, apollo_key).match_person(first, last, domain)
                add(found, score, "apollo", "Apollo people match")
            except Exception as e:
                logger.info(f"Apollo match failed for {contact_name}: {e}")

        pattern, pattern_conf = known_pattern, 0.8 if known_pattern else 0.0
        pattern_origin = "known company pattern (Hunter or a previously verified address)"
        if not pattern and samples:
            pattern, pattern_conf = infer_email_pattern(samples, domain)
            if pattern:
                matches = sum(1 for name, email in samples if apply_pattern(pattern, name, domain) == email.lower())
                pattern_origin = f"matched from {matches} employee email(s) found in public sources"
        if pattern:
            add(
                apply_pattern(pattern, contact_name, domain),
                max(0.6, min(0.9, 0.6 + 0.3 * pattern_conf)),
                "pattern",
                f"pattern {pattern} {pattern_origin}",
            )

        for email, _prior in rank_email_candidates(contact_name, domain):
            if email in observed:
                add(email, 0.9, "website", "address printed on company pages or public commits")

        if max((c for c, _s, _e in candidates.values()), default=0.0) < 0.6:
            try:
                guess, scraped = _llm_guess(llm, browser, contact_name, company, domain)
                if guess:
                    in_text = guess.lower() in scraped.lower()
                    add(
                        guess,
                        0.85 if in_text else 0.45,
                        "llm",
                        "address appears in search results" if in_text else "LLM guess from name and domain",
                    )
            except Exception as e:
                logger.warning(f"LLM failed to deduce email for {contact_name}: {e}")

    for email, prior in rank_email_candidates(contact_name, domain, pattern_confidence=0.0):
        add(email, prior, "permutation", f"common corporate pattern (prior {prior:.0%})")

    ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)
    if not ranked:
        return EmailResult(None, "invalid", 0.0, "none", "no candidate addresses")

    inferred_pattern = known_pattern or (infer_email_pattern(samples, domain)[0] if samples else None)
    newly_rejected: list[str] = []
    catch_all = accept_all_hint
    smtp_note = "SMTP check disabled"

    def result(
        email: str, status: str, conf: float, source: str, evidence: str, is_catch_all: bool | None
    ) -> EmailResult:
        level = confidence_level(status, source, conf)
        return EmailResult(
            email, status, round(conf, 2), source, evidence, is_catch_all, inferred_pattern, newly_rejected, level,
            evidence,
        )

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
        smtp_note = probe.detail or "SMTP inconclusive"
        if probe.catch_all is not None:
            catch_all = probe.catch_all
            company.is_catch_all = probe.catch_all
        if probe.catch_all is False:
            for email, (_conf, source, evidence) in ranked:
                res = probe.results.get(email)
                if res is None:
                    continue
                if res.status == "valid":
                    if not placeholder:
                        pattern, _ = infer_email_pattern([(contact_name, email)], domain)
                        if pattern and not company.email_pattern:
                            company.email_pattern = pattern
                    return result(
                        email, "valid", 0.97, source,
                        f"SMTP {probe.detail}: mailbox accepted ({res.code}) and a random address was rejected, "
                        f"so the domain is not catch-all. Found via: {evidence}",
                        False,
                    )
                if res.status == "invalid":
                    newly_rejected.append(email)
            ranked = [(e, v) for e, v in ranked if e not in newly_rejected]
            if not ranked:
                return EmailResult(
                    None, "invalid", 0.0, "smtp", "all candidates rejected by mail server", False, rejected=newly_rejected
                )

    if catch_all:
        email, (conf, source, evidence) = ranked[0]
        if verification.allow_catch_all and conf >= verification.min_confidence:
            return result(
                email, "catch_all", min(conf, 0.8), source,
                f"Domain accepts any address (catch-all); chose the strongest evidence: {evidence}", True,
            )
        return EmailResult(
            None, "catch_all", conf, source, "catch-all domain and catch-all disallowed", True, rejected=newly_rejected
        )

    # 2. Hunter verifier when SMTP was inconclusive
    hunter_key = config.api_keys.get("hunter")
    if hunter_key:
        for email, (conf, source, evidence) in ranked[:2]:
            try:
                status = HunterClient(browser, hunter_key).verify(email)
            except Exception as e:
                logger.info(f"Hunter verification failed for {email}: {e}")
                break
            if status == "valid":
                return result(email, "valid", 0.95, source, f"Hunter verifier: deliverable. Found via: {evidence}", catch_all)
            if status == "catch_all":
                return result(
                    email, "catch_all", min(conf, 0.8), source, f"Hunter verifier: accept-all domain. Found via: {evidence}", True
                )
            if status == "invalid":
                newly_rejected.append(email)
        ranked = [(e, v) for e, v in ranked if e not in newly_rejected]

    # 3. Evidence-weighted acceptance (SMTP unavailable/greylisted)
    if ranked and verification.allow_unverified:
        email, (conf, source, evidence) = ranked[0]
        if conf >= verification.min_confidence:
            return result(
                email, "unverified", conf, source, f"Mailbox not verifiable ({smtp_note}); found via: {evidence}", catch_all
            )
    return EmailResult(
        None, "unknown", 0.0, "none", "no candidate met the confidence threshold", catch_all, rejected=newly_rejected
    )


def verification_timestamp() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
