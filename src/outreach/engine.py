"""
Outreach lifecycle engine (runs after drafts exist):

  sync_manual_sends  -> detects drafts you sent yourself from Gmail and starts their follow-up clock
  check_replies      -> reads each sent thread: human reply / auto-reply / bounce, classifies replies
  send_due           -> sends scheduled initial emails (auto-send mode) within limits and send windows
  process_followups  -> drafts or sends follow-ups #1/#2 as threaded replies when due (skipped after a reply)
  mark_no_response   -> closes sequences that got no answer
"""

import contextlib
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from src.config import AppConfig
from src.db.models import Application, Campaign, Contact, Email, OutreachEvent
from src.outreach.dedupe import update_ledger_status
from src.outreach.scheduling import followup_due_at, utc_now_naive
from src.pipeline.schemas import ReplyClassificationSchema
from src.providers.gmail import GmailProvider
from src.providers.llm import BaseLLMProvider

logger = logging.getLogger("recruiting-platform.outreach.engine")

ACTIVE_SEQUENCE_STATUSES = ("sent", "followed_up")
FINAL_OUTREACH_STATUSES = ("replied", "interview", "not_interested", "bounced", "no_response", "offer", "rejected")

_BOUNCE_FROM = re.compile(r"mailer-daemon|postmaster|mail delivery", re.IGNORECASE)
_BOUNCE_SUBJECT = re.compile(
    r"delivery status notification|undeliverable|undelivered|mail delivery failed|returned mail|failure notice|"
    r"address not found|delivery has failed",
    re.IGNORECASE,
)
_AUTO_SUBJECT = re.compile(r"automatic reply|auto[- ]?reply|out of (the )?office|autoreply|away from", re.IGNORECASE)

_INTERVIEW_RE = re.compile(
    r"(schedule|set up|book|hop on|jump on|arrange|have)\b.{0,40}\b(call|chat|interview|meeting|conversation)|"
    r"calendly|cal\.com|\binterview\b|next round|assessment|take[- ]home|share (your|some) availability|"
    r"available (for|to) (a )?(call|chat)|when are you (free|available)",
    re.IGNORECASE,
)
_NOT_INTERESTED_RE = re.compile(
    r"not (currently )?hiring|no (open )?(positions|openings|roles|vacancies)|not a (good )?fit|unfortunately|"
    r"not interested|unsubscribe|remove me|do not contact|don't contact|position (has been )?filled",
    re.IGNORECASE,
)
_REFERRAL_RE = re.compile(
    r"(forward(ed)?|loop(ed)? in|cc'?d|reach out to|contact|connect(ed)? you with)\b.{0,50}\b"
    r"(team|manager|recruiter|colleague|hr|talent|lead)",
    re.IGNORECASE,
)
_OOO_RE = re.compile(r"out of (the )?office|on leave|on vacation|limited access to (my )?email", re.IGNORECASE)


def classify_reply_rules(text: str) -> tuple[str, bool]:
    """Keyword fallback for reply classification -> (category, interview_requested)."""
    if _OOO_RE.search(text):
        return "out_of_office", False
    if _INTERVIEW_RE.search(text):
        return "interview", True
    if _NOT_INTERESTED_RE.search(text):
        return "not_interested", False
    if _REFERRAL_RE.search(text):
        return "referral", False
    if re.search(r"thank|interested|sounds good|great|love to|happy to", text, re.IGNORECASE):
        return "positive", False
    return "neutral", False


def initial_email_query(session: Session, app_id: int) -> Any:
    return (
        session.query(Email)
        .filter(
            Email.application_id == app_id,
            or_(Email.sequence_step == 0, Email.sequence_step.is_(None)),
            Email.status.notin_(["bounced", "cancelled"]),
        )
        .order_by(Email.id.desc())
    )


def get_initial_email(session: Session, app_id: int) -> Email | None:
    email: Email | None = initial_email_query(session, app_id).first()
    return email


def log_event(
    session: Session,
    app_id: int,
    event_type: str,
    email_id: int | None = None,
    gmail_message_id: str | None = None,
    details: str | None = None,
) -> None:
    session.add(
        OutreachEvent(
            application_id=app_id,
            email_id=email_id,
            event_type=event_type,
            gmail_message_id=gmail_message_id,
            details=details,
        )
    )


def recipient_already_contacted(session: Session, address: str, exclude_app_id: int | None = None) -> bool:
    """True if any other application already drafted/scheduled/sent an initial email to this address."""
    query = session.query(Email).filter(
        func.lower(Email.to_email) == address.lower(),
        Email.status.in_(["draft_created", "scheduled", "sent", "replied"]),
        or_(Email.sequence_step == 0, Email.sequence_step.is_(None)),
    )
    if exclude_app_id is not None:
        query = query.filter(Email.application_id != exclude_app_id)
    return query.first() is not None


class OutreachEngine:
    def __init__(
        self,
        session: Session,
        config: AppConfig,
        gmail: GmailProvider,
        llm: BaseLLMProvider | None = None,
        now_fn: Callable[[], datetime] = utc_now_naive,
    ):
        self.session = session
        self.config = config
        self.gmail = gmail
        self.llm = llm
        self.now_fn = now_fn
        self._can_read: bool | None = None
        self._my_email: str | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def can_read(self) -> bool:
        if self._can_read is None:
            try:
                self._can_read = bool(self.gmail.can_read())
            except Exception:
                self._can_read = False
        return self._can_read

    def my_email(self) -> str:
        if self._my_email is None:
            try:
                self._my_email = (self.gmail.get_profile_email() or self.config.user_identity.email or "").lower()
            except Exception:
                self._my_email = (self.config.user_identity.email or "").lower()
        return self._my_email

    def _auto_send_enabled(self, app: Application) -> bool:
        if app.campaign_id:
            campaign = self.session.get(Campaign, app.campaign_id)
            if campaign is not None and campaign.auto_send is not None:
                return bool(campaign.auto_send)
        return self.config.outreach.auto_send

    def _sent_today(self) -> int:
        start = self.now_fn().replace(hour=0, minute=0, second=0, microsecond=0)
        return self.session.query(Email).filter(Email.status == "sent", Email.sent_at >= start).count()

    def _last_send_time(self) -> datetime | None:
        value = self.session.query(func.max(Email.sent_at)).filter(Email.status == "sent").scalar()
        return value if isinstance(value, datetime) else None

    def _can_send_now(self) -> bool:
        if self._sent_today() >= self.config.outreach.daily_send_limit:
            logger.info("Daily send limit reached; remaining sends wait for tomorrow.")
            return False
        last = self._last_send_time()
        gap = timedelta(minutes=self.config.outreach.min_minutes_between_sends)
        return last is None or self.now_fn() - last >= gap

    def _followups_for(self, app_id: int) -> list[Email]:
        return (
            self.session.query(Email)
            .filter(Email.application_id == app_id, Email.sequence_step >= 1)
            .order_by(Email.sequence_step.asc(), Email.id.asc())
            .all()
        )

    def cancel_followups(self, app: Application, reason: str) -> int:
        cancelled = 0
        for email in self._followups_for(app.id):
            if email.status in ("pending", "draft_created", "scheduled"):
                if email.gmail_draft_id and email.status == "draft_created":
                    with contextlib.suppress(Exception):
                        self.gmail.delete_draft(email.gmail_draft_id)
                email.status = "cancelled"
                cancelled += 1
        if cancelled:
            log_event(self.session, app.id, "followups_cancelled", details=reason)
        return cancelled

    def _record_sent(self, email: Email, sent_message: dict[str, Any], sent_at: datetime | None = None) -> None:
        app = email.application
        email.status = "sent"
        email.gmail_message_id = str(sent_message.get("id") or email.gmail_message_id or "") or None
        email.gmail_thread_id = str(sent_message.get("threadId") or email.gmail_thread_id or "") or None
        email.sent_at = sent_at or self.now_fn()
        if email.gmail_message_id and self.can_read() and not email.rfc_message_id:
            try:
                msg = self.gmail.get_message(email.gmail_message_id)
                email.rfc_message_id = self.gmail.header(msg, "Message-ID") or None
            except Exception as e:
                logger.debug(f"Could not read Message-ID for {email.gmail_message_id}: {e}")

        step = email.sequence_step or 0
        if step == 0:
            app.outreach_status = "sent"
            app.sent_at = email.sent_at
            contact: Contact | None = app.contact
            if contact is not None:
                contact.last_contacted_at = email.sent_at
            company = app.job.company
            company.last_contacted_at = email.sent_at
            company.status = "contacted"
            window = self.config.outreach.send_window
            steps = self.config.outreach.followups
            for followup in self._followups_for(app.id):
                idx = (followup.sequence_step or 1) - 1
                if followup.status == "pending" and idx < len(steps):
                    followup.scheduled_at = followup_due_at(email.sent_at, steps[idx].after_days, window)
                    followup.gmail_thread_id = email.gmail_thread_id
            update_ledger_status(self.session, app.id, "sent")
            log_event(self.session, app.id, "sent", email.id, email.gmail_message_id, f"to {email.to_email}")
        else:
            app.outreach_status = "followed_up"
            log_event(self.session, app.id, "followup_sent", email.id, email.gmail_message_id, f"step {step}")

    # ------------------------------------------------------------------
    # cycle steps
    # ------------------------------------------------------------------

    def sync_manual_sends(self) -> int:
        """Detects drafts that were sent (or deleted) manually from the Gmail UI."""
        if not self.can_read():
            return 0
        synced = 0
        cutoff = self.now_fn() - timedelta(minutes=5)
        drafts = (
            self.session.query(Email)
            .filter(Email.status.in_(["draft_created", "scheduled"]), Email.gmail_draft_id.isnot(None), Email.created_at <= cutoff)
            .all()
        )
        for email in drafts:
            try:
                if self.gmail.draft_exists(str(email.gmail_draft_id)):
                    continue
                to_addr = email.to_email or (email.application.contact.email if email.application.contact else None)
                subject = re.sub(r'["\\]', " ", email.subject or "")[:80]
                hits = self.gmail.search_messages(f'in:sent to:{to_addr} subject:"{subject}" newer_than:90d', 1) if to_addr else []
                if hits:
                    msg = self.gmail.get_message(hits[0]["id"])
                    sent_at = (
                        datetime.fromtimestamp(int(msg["internalDate"]) / 1000, UTC).replace(tzinfo=None)
                        if msg.get("internalDate")
                        else None
                    )
                    self._record_sent(email, {"id": hits[0]["id"], "threadId": hits[0].get("threadId")}, sent_at)
                    synced += 1
                else:
                    email.status = "cancelled"
                    log_event(self.session, email.application_id, "draft_deleted", email.id)
                    if (email.sequence_step or 0) == 0:
                        self.cancel_followups(email.application, "initial draft deleted without sending")
            except Exception as e:
                logger.warning(f"Could not sync draft {email.gmail_draft_id}: {e}")
        self.session.commit()
        return synced

    def send_due(self) -> int:
        """Sends scheduled initial emails whose time has come (auto-send mode only)."""
        now = self.now_fn()
        due = (
            self.session.query(Email)
            .filter(
                Email.status == "scheduled",
                Email.scheduled_at <= now,
                or_(Email.sequence_step == 0, Email.sequence_step.is_(None)),
            )
            .order_by(Email.scheduled_at.asc())
            .all()
        )
        sent = 0
        for email in due:
            app = email.application
            if not self._auto_send_enabled(app) or not email.gmail_draft_id:
                continue
            if app.outreach_status in FINAL_OUTREACH_STATUSES:
                email.status = "cancelled"
                continue
            to_addr = (email.to_email or "").lower()
            if to_addr and recipient_already_contacted(self.session, to_addr, exclude_app_id=app.id):
                email.status = "cancelled"
                log_event(self.session, app.id, "duplicate_blocked", email.id, details=f"{to_addr} already contacted")
                continue
            if not self._can_send_now():
                break
            try:
                result = self.gmail.send_draft(email.gmail_draft_id)
                self._record_sent(email, result)
                self.session.commit()
                sent += 1
            except Exception as e:
                logger.error(f"Failed to send scheduled email #{email.id}: {e}")
                log_event(self.session, app.id, "send_failed", email.id, details=str(e)[:300])
        self.session.commit()
        return sent

    def process_followups(self) -> int:
        """Creates (and in auto-send mode sends) follow-ups that are due, as replies in the original thread."""
        now = self.now_fn()
        due = (
            self.session.query(Email)
            .filter(Email.sequence_step >= 1, Email.status == "pending", Email.scheduled_at.isnot(None), Email.scheduled_at <= now)
            .order_by(Email.scheduled_at.asc())
            .all()
        )
        handled = 0
        for followup in due:
            app = followup.application
            if app.outreach_status not in ACTIVE_SEQUENCE_STATUSES:
                if app.outreach_status in FINAL_OUTREACH_STATUSES:
                    followup.status = "cancelled"
                continue
            initial = (
                self.session.query(Email)
                .filter(Email.application_id == app.id, or_(Email.sequence_step == 0, Email.sequence_step.is_(None)), Email.status == "sent")
                .order_by(Email.id.desc())
                .first()
            )
            if initial is None:
                continue
            previous_steps = [
                e for e in self._followups_for(app.id) if (e.sequence_step or 0) < (followup.sequence_step or 0)
            ]
            if any(e.status in ("pending", "draft_created", "scheduled") for e in previous_steps):
                continue  # keep the sequence in order
            to_addr = initial.to_email or (app.contact.email if app.contact else None)
            if not to_addr:
                continue
            subject = initial.subject if initial.subject.lower().startswith("re:") else f"Re: {initial.subject}"
            try:
                draft_id = self.gmail.create_draft(
                    to_email=to_addr,
                    subject=subject,
                    body_html=followup.body,
                    thread_id=initial.gmail_thread_id,
                    in_reply_to=initial.rfc_message_id,
                )
                followup.gmail_draft_id = draft_id
                followup.subject = subject
                followup.to_email = to_addr
                followup.gmail_thread_id = initial.gmail_thread_id
                if self._auto_send_enabled(app) and self._can_send_now():
                    result = self.gmail.send_draft(draft_id)
                    self._record_sent(followup, result)
                else:
                    followup.status = "draft_created"
                    log_event(self.session, app.id, "followup_drafted", followup.id, details=f"step {followup.sequence_step}")
                self.session.commit()
                handled += 1
            except Exception as e:
                logger.error(f"Follow-up #{followup.sequence_step} for application #{app.id} failed: {e}")
                log_event(self.session, app.id, "followup_failed", followup.id, details=str(e)[:300])
                self.session.commit()
        return handled

    def _classify(self, text: str, subject: str) -> tuple[str, str, bool]:
        if self.llm is not None and self.config.outreach.classify_replies:
            try:
                prompt = (
                    "Classify this reply to a job-seeker's cold email. Categories: interview (they propose a call, "
                    "interview, assessment or next step), positive, referral (they point to someone else), neutral, "
                    "not_interested, out_of_office, bounce, other.\n\n"
                    f"Subject: {subject}\n\nReply:\n{text[:3000]}"
                )
                result = self.llm.generate_json(prompt, ReplyClassificationSchema)
                assert isinstance(result, ReplyClassificationSchema)
                category = result.category.strip().lower().replace(" ", "_")
                return category, result.summary, bool(result.interview_requested) or category == "interview"
            except Exception as e:
                logger.info(f"LLM reply classification failed ({e}); using keyword rules.")
        category, interview = classify_reply_rules(f"{subject}\n{text}")
        return category, text.strip().replace("\n", " ")[:200], interview

    def _already_processed(self, message_id: str) -> bool:
        return (
            self.session.query(OutreachEvent).filter(OutreachEvent.gmail_message_id == message_id).first() is not None
        )

    def handle_bounce(self, app: Application, email: Email | None, message_id: str | None, detail: str) -> None:
        contact = app.contact
        bad_address = (email.to_email if email else None) or (contact.email if contact else None)
        app.outreach_status = "bounced"
        if email is not None:
            email.status = "bounced"
        self.cancel_followups(app, "bounced")
        log_event(self.session, app.id, "bounce", email.id if email else None, message_id, detail[:300])
        if contact is not None and bad_address:
            rejected = list(contact.rejected_emails or [])
            if bad_address not in rejected:
                rejected.append(bad_address)
            contact.rejected_emails = rejected
            contact.email_status = "invalid"
            if self.config.outreach.retry_on_bounce and contact.email == bad_address:
                contact.email = None
                app.current_stage = 5
                app.state = "Professional Email Discovery"
                log_event(self.session, app.id, "bounce_retry", details=f"retrying email discovery without {bad_address}")

    def handle_reply(self, app: Application, message: dict[str, Any], message_id: str) -> str:
        subject = self.gmail.header(message, "Subject")
        try:
            full = self.gmail.get_message(message_id, fmt="full")
            text = self.gmail.message_text(full)
            has_invite = self.gmail.has_calendar_invite(full)
        except Exception:
            text, has_invite = str(message.get("snippet", "")), False
        category, summary, interview = self._classify(text, subject)
        if has_invite or subject.lower().startswith("invitation:"):
            category, interview = "interview", True

        if category == "out_of_office":
            log_event(self.session, app.id, "auto_reply", gmail_message_id=message_id, details=summary)
            return category
        if category == "bounce":
            self.handle_bounce(app, get_initial_email(self.session, app.id), message_id, summary)
            return category

        now = self.now_fn()
        app.replied_at = app.replied_at or now
        app.reply_category = category
        app.reply_summary = summary
        if interview:
            app.outreach_status = "interview"
            app.interview_at = app.interview_at or now
        elif category == "not_interested":
            app.outreach_status = "not_interested"
            if app.contact is not None:
                app.contact.do_not_contact = True
        else:
            app.outreach_status = "replied"
        update_ledger_status(self.session, app.id, "replied")
        if self.config.outreach.stop_on_reply:
            self.cancel_followups(app, f"reply received ({category})")
        log_event(self.session, app.id, "reply", gmail_message_id=message_id, details=f"{category}: {summary}"[:500])
        logger.info(f"Reply detected for application #{app.id} ({category}).")
        return category

    def check_replies(self) -> dict[str, int]:
        counts = {"replies": 0, "bounces": 0, "auto_replies": 0}
        if not self.can_read():
            logger.info("Gmail read scope not granted; skipping reply detection (run `recruiting-platform auth`).")
            return counts
        me = self.my_email()
        apps = self.session.query(Application).filter(Application.outreach_status.in_(ACTIVE_SEQUENCE_STATUSES)).all()
        for app in apps:
            our_emails = self.session.query(Email).filter(Email.application_id == app.id, Email.status == "sent").all()
            our_ids = {e.gmail_message_id for e in our_emails if e.gmail_message_id}
            threads = {e.gmail_thread_id for e in our_emails if e.gmail_thread_id}
            for thread_id in threads:
                try:
                    thread = self.gmail.get_thread(str(thread_id))
                except Exception as e:
                    logger.debug(f"Could not read thread {thread_id}: {e}")
                    continue
                for msg in thread.get("messages", []):
                    msg_id = str(msg.get("id"))
                    if msg_id in our_ids or "SENT" in (msg.get("labelIds") or []) or self._already_processed(msg_id):
                        continue
                    sender = self.gmail.header(msg, "From").lower()
                    if me and me in sender:
                        continue
                    subject = self.gmail.header(msg, "Subject")
                    if _BOUNCE_FROM.search(sender) or _BOUNCE_SUBJECT.search(subject):
                        self.handle_bounce(app, get_initial_email(self.session, app.id), msg_id, f"{sender}: {subject}")
                        counts["bounces"] += 1
                    elif (
                        self.gmail.header(msg, "Auto-Submitted").lower() not in ("", "no")
                        or self.gmail.header(msg, "X-Autoreply")
                        or _AUTO_SUBJECT.search(subject)
                    ):
                        log_event(self.session, app.id, "auto_reply", gmail_message_id=msg_id, details=subject[:200])
                        counts["auto_replies"] += 1
                    else:
                        category = self.handle_reply(app, msg, msg_id)
                        bucket = {"bounce": "bounces", "out_of_office": "auto_replies"}.get(category, "replies")
                        counts[bucket] += 1
                    self.session.commit()
                    if app.outreach_status not in ACTIVE_SEQUENCE_STATUSES:
                        break
        counts["bounces"] += self._scan_bounce_notifications()
        self.session.commit()
        return counts

    def _scan_bounce_notifications(self) -> int:
        """Catches bounces that Gmail did not attach to the original thread."""
        try:
            hits = self.gmail.search_messages("from:(mailer-daemon OR postmaster) newer_than:30d", 25)
        except Exception:
            return 0
        if not hits:
            return 0
        recent = (
            self.session.query(Email)
            .filter(Email.status == "sent", Email.to_email.isnot(None), or_(Email.sequence_step == 0, Email.sequence_step.is_(None)))
            .all()
        )
        bounced = 0
        for hit in hits:
            msg_id = str(hit.get("id"))
            if self._already_processed(msg_id):
                continue
            try:
                full = self.gmail.get_message(msg_id, fmt="full")
            except Exception:
                continue
            text = self.gmail.message_text(full).lower()
            for email in recent:
                if email.to_email and email.to_email.lower() in text:
                    self.handle_bounce(email.application, email, msg_id, "delivery failure notification")
                    bounced += 1
                    break
        return bounced

    def mark_no_response(self) -> int:
        cutoff = self.now_fn() - timedelta(days=self.config.outreach.no_response_after_days)
        closed = 0
        for app in self.session.query(Application).filter(Application.outreach_status.in_(ACTIVE_SEQUENCE_STATUSES)).all():
            followups = self._followups_for(app.id)
            if any(f.status in ("pending", "draft_created", "scheduled") for f in followups):
                continue
            last_sent = max([e.sent_at for e in app.emails if e.sent_at] or [app.sent_at or self.now_fn()])
            if last_sent <= cutoff:
                app.outreach_status = "no_response"
                log_event(self.session, app.id, "no_response")
                closed += 1
        self.session.commit()
        return closed

    def run_cycle(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        summary["synced_manual_sends"] = self.sync_manual_sends()
        summary.update(self.check_replies())
        summary["sent"] = self.send_due()
        summary["followups"] = self.process_followups()
        summary["closed_no_response"] = self.mark_no_response()
        return summary
