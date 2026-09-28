import pytest

from src.config import load_config
from src.db.models import Company
from src.sources import emails as emails_module
from src.sources.emails import find_contact_email
from src.utils import email_verifier
from src.utils.email_verifier import (
    DomainProbe,
    SmtpResult,
    apply_pattern,
    classify_smtp_response,
    extract_emails_from_text,
    infer_email_pattern,
    rank_email_candidates,
    smtp_probe,
)


def test_pattern_inference_and_application():
    samples = [
        ("Priya Sharma", "priya.sharma@acme.io"),
        ("Rahul Verma", "rahul.verma@acme.io"),
        ("Anita Rao", "anita@acme.io"),
    ]
    pattern, confidence = infer_email_pattern(samples, "acme.io")
    assert pattern == "{first}.{last}" and confidence == pytest.approx(0.67, abs=0.01)
    assert apply_pattern("{f}{last}", "Dr. José Pérez-Gil", "acme.io") == "jperezgil@acme.io"
    assert apply_pattern("{first}.{last}", "Madonna", "acme.io") is None
    ranked = rank_email_candidates("Jane Doe", "acme.io", known_pattern="{f}{last}", pattern_confidence=1.0)
    assert ranked[0][0] == "jdoe@acme.io"
    assert ranked[0][1] > ranked[1][1]


def test_extract_emails_filters_noise():
    text = "Reach careers@acme.io or jane.doe@acme.io; logo@2x.png; noreply@acme.io; bob@gmail.com"
    assert extract_emails_from_text(text, "acme.io") == ["careers@acme.io", "jane.doe@acme.io"]


@pytest.mark.parametrize(
    ("code", "message", "expected"),
    [
        (250, "2.1.5 OK", "valid"),
        (550, "5.1.1 The email account that you tried to reach does not exist", "invalid"),
        (550, "5.4.1 Recipient address rejected: Access denied", "invalid"),
        (550, "5.7.1 Service unavailable; client host blocked using Spamhaus", "unknown"),
        (554, "5.7.1 Relay access denied", "unknown"),
        (450, "4.2.0 Greylisted, please try again later", "unknown"),
        (553, "sorry, no mailbox here by that name", "invalid"),
    ],
)
def test_smtp_response_classification(code, message, expected):
    assert classify_smtp_response(code, message) == expected


class FakeSMTP:
    """Minimal smtplib.SMTP double: accepts addresses in `valid`, everything if `catch_all`."""

    valid: set[str] = set()
    catch_all = False

    def __init__(self, timeout=None):
        pass

    def connect(self, host, port):
        return 220, b"ready"

    def ehlo(self, name):
        return 250, b"ok"

    def helo(self, name):
        return 250, b"ok"

    def mail(self, sender):
        return 250, b"ok"

    def rcpt(self, address):
        if self.catch_all or address in self.valid:
            return 250, b"2.1.5 OK"
        return 550, b"5.1.1 user unknown"

    def quit(self):
        return 221, b"bye"

    def close(self):
        pass


@pytest.fixture
def fake_smtp(monkeypatch):
    email_verifier.reset_smtp_state()
    monkeypatch.setattr(email_verifier.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(email_verifier, "get_mx_hosts", lambda _domain: ["mx.acme.io"])
    FakeSMTP.valid = set()
    FakeSMTP.catch_all = False
    return FakeSMTP


def test_smtp_probe_valid_invalid_and_catch_all(fake_smtp):
    fake_smtp.valid = {"jane.doe@acme.io"}
    probe = smtp_probe("acme.io", ["jane@acme.io", "jane.doe@acme.io"], mail_from="me@gmail.com")
    assert probe.catch_all is False
    assert probe.results["jane.doe@acme.io"].status == "valid"
    assert probe.results["jane@acme.io"].status == "invalid"

    fake_smtp.catch_all = True
    probe = smtp_probe("acme.io", ["anything@acme.io"], mail_from="me@gmail.com")
    assert probe.catch_all is True


class NoNetworkBrowser:
    def search_google(self, query, num_results=5, include_blocked=False):
        return []

    def fetch_page(self, url, use_playwright=False):
        raise RuntimeError("offline")

    def extract_text(self, html):
        return ""


class NoLLM:
    def generate_json(self, prompt, schema, system_prompt=None):
        raise RuntimeError("no llm in this test")


@pytest.fixture
def cfg():
    config = load_config("config.example.yaml")
    config.api_keys.hunter = ""
    config.api_keys.apollo = ""
    return config


def _patch_probe(monkeypatch, probe_fn):
    monkeypatch.setattr(emails_module, "verify_domain_mx", lambda _domain: True)
    monkeypatch.setattr(emails_module, "smtp_probe", probe_fn)


def test_find_email_smtp_verified(monkeypatch, cfg):
    def probe(domain, addresses, mail_from, helo_host="", timeout=8.0):
        results = {a: SmtpResult("valid" if a == "jane.doe@acme.io" else "invalid", 250) for a in addresses}
        return DomainProbe(False, results, True)

    _patch_probe(monkeypatch, probe)
    company = Company(name="Acme", domain="acme.io")
    result = find_contact_email(cfg, NoLLM(), NoNetworkBrowser(), company, "Jane Doe")
    assert result.email == "jane.doe@acme.io" and result.status == "valid" and result.confidence > 0.9
    assert company.email_pattern == "{first}.{last}"


def test_find_email_prefers_observed_pattern_on_catch_all(monkeypatch, cfg):
    def probe(domain, addresses, mail_from, helo_host="", timeout=8.0):
        return DomainProbe(True, {a: SmtpResult("valid", 250) for a in addresses}, True)

    _patch_probe(monkeypatch, probe)
    company = Company(name="Acme", domain="acme.io")
    result = find_contact_email(
        cfg, NoLLM(), NoNetworkBrowser(), company, "Jane Doe",
        email_samples=[("Rahul Verma", "rverma@acme.io"), ("Priya Sharma", "psharma@acme.io")],
    )
    assert result.status == "catch_all" and result.catch_all is True
    assert result.email == "jdoe@acme.io"  # inferred {f}{last} beats the generic first.last prior
    assert company.is_catch_all is True


def test_find_email_unverified_when_smtp_unavailable_and_skips_rejected(monkeypatch, cfg):
    def probe(domain, addresses, mail_from, helo_host="", timeout=8.0):
        return DomainProbe(None, {a: SmtpResult("unknown") for a in addresses}, False, "port 25 blocked")

    _patch_probe(monkeypatch, probe)
    company = Company(name="Acme", domain="acme.io")
    result = find_contact_email(
        cfg, NoLLM(), NoNetworkBrowser(), company, "Jane Doe", rejected=["jane.doe@acme.io"]
    )
    assert result.status == "unverified"
    assert result.email == "jane@acme.io"

    cfg.email_verification.allow_unverified = False
    assert find_contact_email(cfg, NoLLM(), NoNetworkBrowser(), company, "Jane Doe").email is None


def test_generic_inbox_uses_website_address(monkeypatch, cfg):
    def probe(domain, addresses, mail_from, helo_host="", timeout=8.0):
        return DomainProbe(None, {a: SmtpResult("unknown") for a in addresses}, False)

    _patch_probe(monkeypatch, probe)
    company = Company(name="Acme", domain="acme.io")
    result = find_contact_email(
        cfg, NoLLM(), NoNetworkBrowser(), company, "Hiring Team", observed_emails=["talent@acme.io"]
    )
    assert result.email == "talent@acme.io" and result.source == "website"
