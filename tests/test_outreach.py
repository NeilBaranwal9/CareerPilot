from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.analytics.funnel import compute_funnel, email_verification_stats
from src.config import SendWindow, load_config
from src.db.models import Application, Base, Company, Contact, Email, Job, OutreachEvent, Run
from src.outreach.engine import OutreachEngine, classify_reply_rules, recipient_already_contacted
from src.outreach.scheduling import next_send_slot

NOW = datetime(2026, 9, 28, 5, 0)  # Monday 10:30 IST


class FakeGmail:
    """In-memory Gmail double implementing the subset of GmailProvider used by the engine."""

    def __init__(self):
        self.drafts: dict[str, dict] = {}
        self.sent: list[dict] = []
        self.threads: dict[str, list[dict]] = {}
        self.deleted: list[str] = []
        self._n = 0

    def can_read(self):
        return True

    def get_profile_email(self):
        return "me@gmail.com"

    def create_draft(self, to_email, subject, body_html, resume_path=None, thread_id=None, in_reply_to=None):
        self._n += 1
        draft_id = f"d{self._n}"
        self.drafts[draft_id] = {"to": to_email, "subject": subject, "thread_id": thread_id, "in_reply_to": in_reply_to}
        return draft_id

    def send_draft(self, draft_id):
        draft = self.drafts.pop(draft_id)
        self._n += 1
        thread_id = draft["thread_id"] or f"t{self._n}"
        message = {"id": f"m{self._n}", "threadId": thread_id, "labelIds": ["SENT"]}
        self.sent.append({**draft, **message})
        self.threads.setdefault(thread_id, []).append(
            {"id": message["id"], "labelIds": ["SENT"], "payload": {"headers": [{"name": "From", "value": "me@gmail.com"}]}}
        )
        return message

    def draft_exists(self, draft_id):
        return draft_id in self.drafts

    def delete_draft(self, draft_id):
        self.deleted.append(draft_id)
        self.drafts.pop(draft_id, None)

    def search_messages(self, query, max_results=10):
        return []

    def get_message(self, message_id, fmt="metadata"):
        for msgs in self.threads.values():
            for m in msgs:
                if m["id"] == message_id:
                    return {**m, "payload": {**m.get("payload", {}), "headers": m["payload"]["headers"] + [{"name": "Message-ID", "value": f"<{message_id}@mail.gmail.com>"}]}}
        return {"id": message_id, "payload": {"headers": []}}

    def get_thread(self, thread_id):
        return {"id": thread_id, "messages": self.threads.get(thread_id, [])}

    def add_incoming(self, thread_id, sender, subject, body, headers=None):
        self._n += 1
        msg = {
            "id": f"in{self._n}",
            "labelIds": ["INBOX"],
            "snippet": body,
            "payload": {"headers": [{"name": "From", "value": sender}, {"name": "Subject", "value": subject}, *(headers or [])]},
        }
        self.threads.setdefault(thread_id, []).append(msg)
        return msg

    header = staticmethod(lambda message, name: next(
        (h["value"] for h in message.get("payload", {}).get("headers", []) if h["name"].lower() == name.lower()), ""
    ))

    @staticmethod
    def message_text(message):
        return message.get("snippet", "")

    @staticmethod
    def has_calendar_invite(_message):
        return False


@pytest.fixture
def env():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    cfg = load_config("config.example.yaml")
    cfg.outreach.auto_send = True
    cfg.outreach.min_minutes_between_sends = 0
    session.add(Run(id="R"))
    company = Company(name="PayCo", domain="payco.in", sector="fintech")
    session.add(company)
    session.flush()
    job = Job(company_id=company.id, title="Backend Engineer", url="https://payco.in/jobs/1")
    contact = Contact(company_id=company.id, name="Priya Sharma", role="Engineering Manager", email="priya@payco.in",
                      email_status="valid", role_category="engineering_manager")
    session.add_all([job, contact])
    session.flush()
    app = Application(run_id="R", job_id=job.id, contact_id=contact.id, state="Completed", current_stage=12,
                      outreach_status="scheduled", persona="engineering_manager")
    session.add(app)
    session.flush()
    gmail = FakeGmail()
    draft_id = gmail.create_draft("priya@payco.in", "Backend role at PayCo", "<p>Hi</p>")
    session.add(Email(application_id=app.id, subject="Backend role at PayCo", body="<p>Hi Priya</p>", status="scheduled",
                      sequence_step=0, to_email="priya@payco.in", gmail_draft_id=draft_id, scheduled_at=NOW - timedelta(minutes=1)))
    session.add(Email(application_id=app.id, subject="Re: Backend role at PayCo", body="<p>Bump</p>", status="pending", sequence_step=1, to_email="priya@payco.in"))
    session.add(Email(application_id=app.id, subject="Re: Backend role at PayCo", body="<p>Closing</p>", status="pending", sequence_step=2, to_email="priya@payco.in"))
    session.commit()
    clock = {"now": NOW}
    engine_obj = OutreachEngine(session, cfg, gmail, llm=None, now_fn=lambda: clock["now"])
    return session, cfg, gmail, app, engine_obj, clock


def test_send_window_scheduling():
    window = SendWindow(timezone="Asia/Kolkata", start_hour=9, end_hour=12, weekdays_only=True)
    saturday_noon_utc = datetime(2026, 9, 26, 6, 30)  # Saturday 12:00 IST
    slot = next_send_slot(saturday_noon_utc, window, jitter_minutes=0)
    assert slot == datetime(2026, 9, 28, 3, 30)  # Monday 09:00 IST
    inside = next_send_slot(NOW, window, jitter_minutes=0)
    assert inside == NOW
    gap = next_send_slot(NOW, window, not_before_utc=NOW, min_gap_minutes=10, jitter_minutes=0)
    assert gap == NOW + timedelta(minutes=10)


def test_send_followups_and_stop_on_reply(env):
    session, cfg, gmail, app, engine, clock = env

    assert engine.send_due() == 1
    session.refresh(app)
    initial = session.query(Email).filter(Email.sequence_step == 0).one()
    assert initial.status == "sent" and initial.gmail_thread_id and app.outreach_status == "sent"
    assert initial.rfc_message_id.endswith("@mail.gmail.com>")
    fu1, fu2 = session.query(Email).filter(Email.sequence_step >= 1).order_by(Email.sequence_step).all()
    assert fu1.scheduled_at > NOW + timedelta(days=3) and fu2.scheduled_at > fu1.scheduled_at

    # Follow-up #1 becomes due and goes out as a threaded reply
    clock["now"] = fu1.scheduled_at + timedelta(minutes=1)
    assert engine.process_followups() == 1
    session.refresh(fu1)
    assert fu1.status == "sent" and fu1.subject.startswith("Re: ")
    last_sent = gmail.sent[-1]
    assert last_sent["thread_id"] == initial.gmail_thread_id and last_sent["in_reply_to"] == initial.rfc_message_id
    assert app.outreach_status == "followed_up"

    # Recipient replies asking for a call -> interview, follow-up #2 cancelled
    gmail.add_incoming(initial.gmail_thread_id, "Priya Sharma <priya@payco.in>", "Re: Backend role at PayCo",
                       "Thanks! Could you share your availability for a quick call this week?")
    counts = engine.check_replies()
    assert counts["replies"] == 1
    session.refresh(app)
    session.refresh(fu2)
    assert app.outreach_status == "interview" and app.interview_at is not None and app.replied_at is not None
    assert fu2.status == "cancelled"
    clock["now"] = fu2.scheduled_at + timedelta(days=1)
    assert engine.process_followups() == 0

    # Messages are processed only once
    assert engine.check_replies()["replies"] == 0

    funnel = {row["stage"]: row["count"] for row in compute_funnel(session)["funnel"]}
    assert funnel["Discovered"] == 1 and funnel["Sent"] == 1
    assert funnel["Replied"] == 1 and funnel["Interview"] == 1 and funnel["Rejected"] == 0
    assert email_verification_stats(session)["smtp_verified_pct"] == 100.0


def test_out_of_office_does_not_stop_sequence(env):
    session, cfg, gmail, app, engine, clock = env
    engine.send_due()
    initial = session.query(Email).filter(Email.sequence_step == 0).one()
    gmail.add_incoming(initial.gmail_thread_id, "priya@payco.in", "Automatic reply: Out of office", "I am on leave",
                       headers=[{"name": "Auto-Submitted", "value": "auto-replied"}])
    counts = engine.check_replies()
    assert counts["auto_replies"] == 1 and counts["replies"] == 0
    session.refresh(app)
    assert app.outreach_status == "sent"


def test_bounce_marks_invalid_and_retries_discovery(env):
    session, cfg, gmail, app, engine, clock = env
    engine.send_due()
    initial = session.query(Email).filter(Email.sequence_step == 0).one()
    gmail.add_incoming(initial.gmail_thread_id, "Mail Delivery Subsystem <mailer-daemon@googlemail.com>",
                       "Delivery Status Notification (Failure)", "Address not found")
    assert engine.check_replies()["bounces"] == 1
    session.refresh(app)
    contact = app.contact
    assert app.outreach_status == "bounced" and app.current_stage == 5
    assert contact.email is None and contact.email_status == "invalid" and "priya@payco.in" in contact.rejected_emails
    assert all(e.status == "cancelled" for e in session.query(Email).filter(Email.sequence_step >= 1))
    assert session.query(OutreachEvent).filter(OutreachEvent.event_type == "bounce").count() == 1


def test_drafts_only_mode_does_not_send(env):
    session, cfg, gmail, app, engine, clock = env
    cfg.outreach.auto_send = False
    assert engine.send_due() == 0
    assert gmail.sent == []


def test_daily_send_limit(env):
    session, cfg, gmail, app, engine, clock = env
    cfg.outreach.daily_send_limit = 0
    assert engine.send_due() == 0


def test_duplicate_recipient_guard(env):
    session, cfg, gmail, app, engine, clock = env
    assert recipient_already_contacted(session, "PRIYA@payco.in") is True
    assert recipient_already_contacted(session, "priya@payco.in", exclude_app_id=app.id) is False


def test_reply_rules():
    assert classify_reply_rules("Happy to chat, here's my calendly link")[0] == "interview"
    assert classify_reply_rules("Unfortunately we are not hiring right now")[0] == "not_interested"
    assert classify_reply_rules("I've looped in our recruiter Neha who can help")[0] == "referral"
    assert classify_reply_rules("I am out of office until Monday")[0] == "out_of_office"


def test_email_style_checker():
    from src.outreach.personas import check_email_style, email_body_word_count

    good = (
        "<p>Hi Priya,</p><p>I saw that PayCo launched instant UPI refunds for small merchants last month.</p>"
        "<p>I am an AI and ML student at VIT Chennai and recently interned at Jio Platforms in Mumbai.</p>"
        "<p>There I worked on systems for large-scale telemetry collection and analysis, which is close to the "
        "monitoring a real-time payments product depends on.</p><p>I would love to be considered for relevant "
        "internship opportunities.</p><p>If there is a suitable opening, I would be grateful for a referral or "
        "any guidance on the right person to contact on your engineering team.</p>"
        "<p>Best,<br>Neil Baranwal<br>linkedin.com/in/x</p>"
    )
    assert 80 <= email_body_word_count(good, "Neil Baranwal") <= 180
    assert check_email_style(good, "Neil Baranwal", 90, 160) == []
    bad = "<p>Hi Priya, I am passionate about leveraging cutting-edge tech. Can we schedule a quick call?</p>"
    issues = check_email_style(bad, "Neil Baranwal", 90, 160)
    assert any("words" in i for i in issues) and any("passionate" in i for i in issues)
    assert any("meeting" in i for i in issues)
