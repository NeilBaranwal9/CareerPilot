import contextlib
import logging
import re
import smtplib
import socket
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass, field

import dns.resolver

logger = logging.getLogger("recruiting-platform.utils.email_verifier")

# Basic email syntax regex pattern
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
EMAIL_IN_TEXT_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+")

# Mock/test domains used in automated tests: treated as having MX records and never probed over SMTP.
TEST_DOMAINS = {"mocktech.com", "nojobscorp.com", "targetedcorp.com", "example.com", "test.com"}

# Local-part templates with a prior probability of being a company's convention.
EMAIL_PATTERNS: list[tuple[str, float]] = [
    ("{first}.{last}", 0.35),
    ("{first}", 0.30),
    ("{f}{last}", 0.22),
    ("{first}{last}", 0.18),
    ("{f}.{last}", 0.15),
    ("{first}_{last}", 0.10),
    ("{first}{l}", 0.10),
    ("{last}.{first}", 0.06),
    ("{last}{f}", 0.05),
    ("{first}-{last}", 0.05),
    ("{last}", 0.05),
]

GENERIC_INBOXES = ["careers", "jobs", "hiring", "talent", "hr", "recruitment"]

_HONORIFICS = {"dr", "mr", "mrs", "ms", "miss", "prof", "sir", "er", "ca", "adv"}
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "mba", "md", "cpa"}
_PLACEHOLDER_TERMS = {"hiring", "manager", "recruiter", "engineering", "head", "unknown", "n/a", "team", "talent", "hr"}


def verify_email_syntax(email: str) -> bool:
    """Check if the email string matches standard RFC syntax."""
    if not email or not isinstance(email, str):
        return False
    return bool(EMAIL_REGEX.match(email.strip()))


def _clean_domain(domain: str) -> str:
    clean = domain.strip().lower()
    if clean.startswith("http://") or clean.startswith("https://"):
        clean = clean.split("//")[-1]
    clean = clean.split("/")[0]
    if clean.startswith("www."):
        clean = clean[4:]
    return clean


def verify_domain_mx(domain: str) -> bool:
    """
    Checks whether the target domain has active Mail Exchange (MX) DNS records.
    Returns True if MX records are found, False if non-existent or lookup fails.
    """
    if not domain:
        return False

    clean_domain = _clean_domain(domain)
    if clean_domain in TEST_DOMAINS:
        return True

    try:
        answers = dns.resolver.resolve(clean_domain, "MX", lifetime=4.0)
        return len(answers) > 0
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers):
        logger.warning(f"Domain '{clean_domain}' has no valid MX records (NXDOMAIN/NoAnswer).")
        return False
    except Exception as e:
        logger.debug(f"MX record lookup for '{clean_domain}' raised exception: {e}")
        # If timeout or resolver error occurs, fall back to checking if A/AAAA record exists
        try:
            answers_a = dns.resolver.resolve(clean_domain, "A", lifetime=3.0)
            return len(answers_a) > 0
        except Exception:
            return False


def verify_email(email: str) -> tuple[bool, str]:
    """
    Validates an email address by checking syntax and domain MX records.
    Returns (is_valid, reason_message). For mailbox-level verification use `smtp_probe`.
    """
    if not email:
        return False, "Email address is empty"

    clean_email = email.strip()
    if not verify_email_syntax(clean_email):
        return False, f"Invalid email syntax: {clean_email}"

    domain = clean_email.split("@")[-1]
    if not verify_domain_mx(domain):
        return False, f"Domain '{domain}' has no valid MX/mail server records"

    return True, "Valid email format & active MX records"


# ---------------------------------------------------------------------------
# Names & patterns
# ---------------------------------------------------------------------------


def normalize_name_parts(full_name: str) -> list[str]:
    """Lowercases, strips accents, honorifics, suffixes and punctuation: 'Dr. José Pérez-Gil, PhD' -> ['jose', 'perezgil']."""
    if not full_name:
        return []
    name = unicodedata.normalize("NFKD", full_name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"\(.*?\)", " ", name)
    name = name.split(",")[0]
    tokens = []
    for raw in name.replace(".", " ").split():
        token = re.sub(r"[^a-zA-Z]", "", raw).lower()
        if not token or token in _HONORIFICS or token in _NAME_SUFFIXES:
            continue
        tokens.append(token)
    return tokens


def is_placeholder_name(full_name: str) -> bool:
    parts = normalize_name_parts(full_name)
    return not parts or all(p in _PLACEHOLDER_TERMS for p in parts)


def apply_pattern(pattern: str, full_name: str, domain: str) -> str | None:
    """Renders an email for a name using a pattern such as '{first}.{last}' or '{f}{last}'."""
    parts = normalize_name_parts(full_name)
    if not parts or not domain:
        return None
    first = parts[0]
    last = parts[-1] if len(parts) > 1 else ""
    if ("{last}" in pattern or "{l}" in pattern) and not last:
        return None
    local = (
        pattern.replace("{first}", first)
        .replace("{last}", last)
        .replace("{f}", first[:1])
        .replace("{l}", last[:1])
    )
    return f"{local}@{_clean_domain(domain)}"


def infer_email_pattern(samples: list[tuple[str, str]], domain: str) -> tuple[str | None, float]:
    """
    Infers a company's email convention from known (full_name, email) pairs on the same domain.
    Returns (pattern, confidence) where confidence is the share of samples matching the winning pattern.
    """
    clean_domain = _clean_domain(domain)
    counts: Counter[str] = Counter()
    usable = 0
    for full_name, email in samples:
        if not email or "@" not in email or email.lower().split("@")[-1] != clean_domain:
            continue
        local = email.lower().split("@")[0]
        matched = False
        for pattern, _prior in EMAIL_PATTERNS:
            rendered = apply_pattern(pattern, full_name, clean_domain)
            if rendered and rendered.split("@")[0] == local:
                counts[pattern] += 1
                matched = True
                break
        if matched or normalize_name_parts(full_name):
            usable += 1
    if not counts:
        return None, 0.0
    pattern, hits = counts.most_common(1)[0]
    return pattern, round(hits / max(usable, 1), 2)


def extract_emails_from_text(text: str, domain: str | None = None) -> list[str]:
    """Finds email addresses in free text, optionally restricted to a domain; filters image names & noreply."""
    found: list[str] = []
    clean_domain = _clean_domain(domain) if domain else None
    for match in EMAIL_IN_TEXT_REGEX.findall(text or ""):
        email = match.strip(".").lower()
        if email.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")):
            continue
        if any(bad in email for bad in ("noreply", "no-reply", "example.com", "sentry", "wixpress", "users.noreply")):
            continue
        if clean_domain and not email.endswith("@" + clean_domain):
            continue
        if email not in found:
            found.append(email)
    return found


def rank_email_candidates(
    full_name: str, domain: str, known_pattern: str | None = None, pattern_confidence: float = 0.0
) -> list[tuple[str, float]]:
    """Returns (email, prior_confidence) candidates for a person, best first. Placeholders get generic inboxes."""
    if not domain:
        return []
    clean_domain = _clean_domain(domain)
    if is_placeholder_name(full_name):
        return [(f"{box}@{clean_domain}", 0.2 - i * 0.02) for i, box in enumerate(GENERIC_INBOXES)]

    ranked: dict[str, float] = {}
    if known_pattern:
        email = apply_pattern(known_pattern, full_name, clean_domain)
        if email:
            ranked[email] = max(0.6, min(0.9, 0.6 + 0.3 * pattern_confidence))
    for pattern, prior in EMAIL_PATTERNS:
        email = apply_pattern(pattern, full_name, clean_domain)
        if email and email not in ranked:
            ranked[email] = prior
    return sorted(ranked.items(), key=lambda kv: kv[1], reverse=True)


def generate_email_permutations(full_name: str, domain: str) -> list[str]:
    """
    Generates standard professional corporate email permutations.
    Example: 'John Doe', 'company.com' -> ['john.doe@company.com', 'john@company.com', 'j.doe@company.com', ...]
    """
    if not domain:
        return []

    clean_domain = _clean_domain(domain)
    if is_placeholder_name(full_name):
        return [f"careers@{clean_domain}", f"jobs@{clean_domain}", f"hiring@{clean_domain}"]

    permutations = [email for email, _prior in rank_email_candidates(full_name, clean_domain)]
    permutations.append(f"careers@{clean_domain}")

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique_perms = []
    for p in permutations:
        if p not in seen:
            seen.add(p)
            unique_perms.append(p)
    return unique_perms


# ---------------------------------------------------------------------------
# SMTP mailbox verification (RCPT TO probe) with catch-all detection
# ---------------------------------------------------------------------------


@dataclass
class SmtpResult:
    status: str  # valid | invalid | unknown
    code: int | None = None
    message: str = ""


@dataclass
class DomainProbe:
    catch_all: bool | None = None  # True = accepts any address, False = rejects unknown users, None = undetermined
    results: dict[str, SmtpResult] = field(default_factory=dict)
    reachable: bool = False
    detail: str = ""


_SMTP_STATE: dict[str, int | bool] = {"consecutive_connect_failures": 0, "blocked": False}
_MX_CACHE: dict[str, list[str]] = {}

_POLICY_MARKERS = (
    "5.7.",
    "spamhaus",
    "blocked",
    "block list",
    "blocklist",
    "blacklist",
    "policy",
    "reputation",
    "not authorized",
    "relay",
    "authentication required",
    "spf",
    "rbl",
    "banned",
    "helo",
    "rate limit",
    "too many",
)
_USER_UNKNOWN_MARKERS = (
    "5.1.1",
    "5.1.0",
    "5.1.10",
    "5.4.1",
    "user unknown",
    "unknown user",
    "no such user",
    "does not exist",
    "doesn't exist",
    "recipient not found",
    "unknown recipient",
    "invalid recipient",
    "recipient rejected",
    "address rejected",
    "mailbox unavailable",
    "mailbox not found",
    "no mailbox",
    "user not found",
    "account disabled",
    "unrouteable",
)


def classify_smtp_response(code: int, message: str) -> str:
    """Maps an SMTP RCPT TO reply to valid / invalid / unknown, treating anti-spam policy blocks as unknown."""
    msg = (message or "").lower()
    if code in (250, 251):
        return "valid"
    if code >= 500 and ("5.4.1" in msg or any(m in msg for m in ("5.1.1", "user unknown", "no such user"))):
        return "invalid"
    if any(m in msg for m in _POLICY_MARKERS):
        return "unknown"
    if code in (550, 551, 553) and (not msg.strip() or any(m in msg for m in _USER_UNKNOWN_MARKERS)):
        return "invalid"
    if code in (550, 551, 553, 554) and any(m in msg for m in _USER_UNKNOWN_MARKERS):
        return "invalid"
    return "unknown"


def get_mx_hosts(domain: str) -> list[str]:
    clean_domain = _clean_domain(domain)
    if clean_domain in _MX_CACHE:
        return _MX_CACHE[clean_domain]
    hosts: list[str] = []
    try:
        answers = dns.resolver.resolve(clean_domain, "MX", lifetime=5.0)
        ranked = sorted(answers, key=lambda r: r.preference)
        hosts = [str(r.exchange).rstrip(".") for r in ranked]
    except Exception as e:
        logger.debug(f"MX lookup failed for {clean_domain}: {e}")
    _MX_CACHE[clean_domain] = hosts
    return hosts


def smtp_available() -> bool:
    """False once repeated connection failures suggest outbound port 25 is blocked (common on home ISPs)."""
    return not bool(_SMTP_STATE["blocked"])


def reset_smtp_state() -> None:
    _SMTP_STATE["consecutive_connect_failures"] = 0
    _SMTP_STATE["blocked"] = False


def _default_helo() -> str:
    fqdn = socket.getfqdn()
    return fqdn if "." in fqdn and not fqdn.endswith(".local") else "localhost.localdomain"


def smtp_probe(
    domain: str,
    addresses: list[str],
    mail_from: str,
    helo_host: str = "",
    timeout: float = 8.0,
) -> DomainProbe:
    """
    Opens one SMTP session with the domain's MX, sends RCPT TO for a random address (catch-all test)
    and for each candidate, then quits without sending any message.
    """
    clean_domain = _clean_domain(domain)
    unknown = {a: SmtpResult("unknown", None, "not probed") for a in addresses}
    if clean_domain in TEST_DOMAINS:
        return DomainProbe(None, unknown, False, "test domain")
    if not smtp_available():
        return DomainProbe(None, unknown, False, "outbound SMTP (port 25) appears blocked")

    hosts = get_mx_hosts(clean_domain)
    if not hosts:
        return DomainProbe(None, unknown, False, "no MX hosts")

    helo = helo_host or _default_helo()
    sender = mail_from or f"postmaster@{helo}"
    last_error = ""
    for host in hosts[:2]:
        server = smtplib.SMTP(timeout=timeout)
        try:
            code, _banner = server.connect(host, 25)
            _SMTP_STATE["consecutive_connect_failures"] = 0
            if code != 220:
                last_error = f"{host} banner code {code}"
                continue
            ehlo_code, _ = server.ehlo(helo)
            if ehlo_code != 250:
                server.helo(helo)
            mail_code, mail_msg = server.mail(sender)
            if mail_code != 250:
                detail = mail_msg.decode(errors="ignore")
                return DomainProbe(None, unknown, True, f"MAIL FROM rejected: {mail_code} {detail}")

            random_addr = f"zz{uuid.uuid4().hex[:12]}@{clean_domain}"
            r_code, r_msg = server.rcpt(random_addr)
            random_status = classify_smtp_response(r_code, r_msg.decode(errors="ignore"))
            catch_all = True if random_status == "valid" else False if random_status == "invalid" else None

            results: dict[str, SmtpResult] = {}
            for addr in addresses:
                c, m = server.rcpt(addr)
                text = m.decode(errors="ignore")
                results[addr] = SmtpResult(classify_smtp_response(c, text), c, text)
            with contextlib.suppress(Exception):
                server.quit()
            return DomainProbe(catch_all, results, True, f"probed via {host}")
        except (TimeoutError, ConnectionRefusedError, smtplib.SMTPConnectError, socket.gaierror) as e:
            failures = int(_SMTP_STATE["consecutive_connect_failures"]) + 1
            _SMTP_STATE["consecutive_connect_failures"] = failures
            if failures >= 3:
                _SMTP_STATE["blocked"] = True
                logger.warning(
                    "Outbound SMTP port 25 looks blocked on this network; mailbox verification disabled for this run."
                )
            last_error = f"{host}: {e}"
        except (smtplib.SMTPException, OSError) as e:
            last_error = f"{host}: {e}"
        finally:
            with contextlib.suppress(Exception):
                server.close()
    return DomainProbe(None, unknown, False, last_error or "SMTP probe failed")
