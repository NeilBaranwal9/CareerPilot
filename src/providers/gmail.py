import base64
import json
import logging
import mimetypes
import os
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger("recruiting-platform.providers.gmail")

COMPOSE_SCOPE = "https://www.googleapis.com/auth/gmail.compose"
READ_SCOPES = {
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://mail.google.com/",
}
METADATA_HEADERS = ["From", "To", "Subject", "Date", "Message-ID", "Auto-Submitted", "X-Autoreply", "Content-Type"]


class GmailProvider:
    """
    Gmail API integration using Google API Client and OAuth2.
    Drafts & sending need `gmail.compose`; reply/bounce tracking additionally needs `gmail.readonly`.
    """

    def __init__(self, credentials_path: str, token_path: str, scopes: list[str]):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.scopes = scopes
        self.creds: Credentials | None = None
        self.service: Any = None
        self.granted_scopes: set[str] = set()
        self._profile_email: str | None = None

    def _token_scopes(self) -> set[str]:
        try:
            with open(self.token_path, encoding="utf-8") as f:
                data = json.load(f)
            scopes = data.get("scopes") or data.get("scope") or []
            if isinstance(scopes, str):
                scopes = scopes.split()
            return set(scopes)
        except Exception:
            return set()

    def authenticate(self, interactive: bool = True) -> bool:
        """
        Authenticates with Gmail API. Reuses existing token if valid.
        If the saved token lacks newly required scopes (e.g. gmail.readonly for reply tracking) and we are
        interactive, the consent flow is re-run; otherwise the token is still used for what it can do.
        """
        if os.path.exists(self.token_path):
            token_scopes = self._token_scopes()
            missing = set(self.scopes) - token_scopes if token_scopes else set()
            if missing and interactive and os.path.exists(self.credentials_path):
                logger.warning(f"Gmail token is missing scopes {sorted(missing)}; re-running consent flow.")
            else:
                try:
                    self.creds = Credentials.from_authorized_user_file(  # type: ignore[no-untyped-call]
                        self.token_path, sorted(token_scopes) if token_scopes else self.scopes
                    )
                except Exception as e:
                    logger.warning(f"Failed to load token file: {e}. Re-authenticating.")
                if missing:
                    logger.warning(
                        f"Gmail token lacks scopes {sorted(missing)}. Reply tracking needs them: run "
                        "`recruiting-platform auth` to re-authorize."
                    )

        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())  # type: ignore[no-untyped-call]
                    with open(self.token_path, "w") as token:
                        token.write(self.creds.to_json())  # type: ignore[no-untyped-call]
                    logger.info("Successfully refreshed Gmail token.")
                except Exception as e:
                    logger.error(f"Failed to refresh Gmail token: {e}")
                    self.creds = None

            if not self.creds or not self.creds.valid:
                if not os.path.exists(self.credentials_path):
                    logger.warning(
                        f"Gmail OAuth client secrets file not found at {self.credentials_path}. "
                        "Gmail draft creation will fail until credentials.json is provided. "
                        "Please download it from the Google Cloud Console."
                    )
                    return False

                if not interactive:
                    logger.error("Gmail authorization required but running in non-interactive mode.")
                    return False

                try:
                    flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.scopes)
                    self.creds = flow.run_local_server(port=8080, access_type="offline", prompt="consent")
                    # Save the credentials for the next run
                    with open(self.token_path, "w") as token:
                        token.write(self.creds.to_json())
                    logger.info("Successfully authenticated Gmail and saved credentials.")
                except Exception as e:
                    logger.error(f"Gmail OAuth flow failed: {e}")
                    return False

        if self.creds and self.creds.valid:
            self.service = build("gmail", "v1", credentials=self.creds)
            self.granted_scopes = set(self.creds.scopes or []) or self._token_scopes()
            return True

        return False

    def _ensure_service(self) -> Any:
        if not self.service and not self.authenticate(interactive=False):
            raise RuntimeError("Gmail service not authenticated. Run `recruiting-platform auth`.")
        return self.service

    def can_read(self) -> bool:
        """True when the token allows reading threads (needed for reply/bounce detection)."""
        if not self.service:
            try:
                self._ensure_service()
            except Exception:
                return False
        return bool(self.granted_scopes & READ_SCOPES)

    def get_profile_email(self) -> str | None:
        if self._profile_email:
            return self._profile_email
        try:
            profile = self._ensure_service().users().getProfile(userId="me").execute()
            self._profile_email = str(profile.get("emailAddress", "")).lower() or None
        except Exception as e:
            logger.debug(f"Could not read Gmail profile: {e}")
        return self._profile_email

    def _build_message(
        self,
        to_email: str,
        subject: str,
        body_html: str,
        resume_path: str | None = None,
        in_reply_to: str | None = None,
    ) -> str:
        message = MIMEMultipart()
        message["to"] = to_email
        message["subject"] = subject
        if in_reply_to:
            message["In-Reply-To"] = in_reply_to
            message["References"] = in_reply_to

        # Add HTML body
        message.attach(MIMEText(body_html, "html"))

        # Attach resume if provided and exists
        if resume_path and os.path.exists(resume_path):
            filename = os.path.basename(resume_path)
            content_type, encoding = mimetypes.guess_type(resume_path)
            if content_type is None or encoding is not None:
                content_type = "application/octet-stream"
            main_type, sub_type = content_type.split("/", 1)

            try:
                with open(resume_path, "rb") as fp:
                    attachment = MIMEBase(main_type, sub_type)
                    attachment.set_payload(fp.read())
                encoders.encode_base64(attachment)
                attachment.add_header("Content-Disposition", "attachment", filename=filename)
                message.attach(attachment)
                logger.info(f"Attached resume {filename} to draft message.")
            except Exception as e:
                logger.error(f"Failed to attach resume file {resume_path}: {e}")

        return base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

    def create_draft(
        self,
        to_email: str,
        subject: str,
        body_html: str,
        resume_path: str | None = None,
        thread_id: str | None = None,
        in_reply_to: str | None = None,
    ) -> str:
        """
        Creates a draft email in the user's Gmail account and returns the Draft ID.
        With thread_id/in_reply_to the draft is a threaded reply (used for follow-ups).
        """
        service = self._ensure_service()
        raw_message = self._build_message(to_email, subject, body_html, resume_path, in_reply_to)
        try:
            message_body: dict[str, Any] = {"raw": raw_message}
            if thread_id:
                message_body["threadId"] = thread_id
            draft = service.users().drafts().create(userId="me", body={"message": message_body}).execute()
            logger.info(f"Created draft successfully. Draft ID: {draft['id']}")
            return str(draft["id"])
        except Exception as e:
            logger.error(f"Failed to create Gmail draft: {e}")
            raise RuntimeError(f"Gmail draft creation failed: {e}") from e

    def send_draft(self, draft_id: str) -> dict[str, Any]:
        """Sends an existing Gmail draft and returns the sent message resource ({id, threadId, labelIds})."""
        service = self._ensure_service()
        try:
            sent = service.users().drafts().send(userId="me", body={"id": draft_id}).execute()
            logger.info(f"Sent draft successfully. Draft ID: {draft_id}")
            return dict(sent)
        except Exception as e:
            logger.error(f"Failed to send Gmail draft {draft_id}: {e}")
            raise RuntimeError(f"Gmail draft send failed: {e}") from e

    def draft_exists(self, draft_id: str) -> bool:
        try:
            self._ensure_service().users().drafts().get(userId="me", id=draft_id, format="minimal").execute()
            return True
        except HttpError as e:
            if getattr(e, "status_code", None) == 404 or "404" in str(e):
                return False
            raise

    def delete_draft(self, draft_id: str) -> None:
        try:
            self._ensure_service().users().drafts().delete(userId="me", id=draft_id).execute()
        except Exception as e:
            logger.debug(f"Could not delete draft {draft_id}: {e}")

    def search_messages(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Gmail search (same syntax as the Gmail search box). Returns [{id, threadId}]. Needs read scope."""
        response = self._ensure_service().users().messages().list(userId="me", q=query, maxResults=max_results).execute()
        return list(response.get("messages", []))

    def get_message(self, message_id: str, fmt: str = "metadata") -> dict[str, Any]:
        request = self._ensure_service().users().messages().get(
            userId="me", id=message_id, format=fmt, **({"metadataHeaders": METADATA_HEADERS} if fmt == "metadata" else {})
        )
        return dict(request.execute())

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        response = self._ensure_service().users().threads().get(
            userId="me", id=thread_id, format="metadata", metadataHeaders=METADATA_HEADERS
        ).execute()
        return dict(response)

    @staticmethod
    def header(message: dict[str, Any], name: str) -> str:
        for h in message.get("payload", {}).get("headers", []):
            if str(h.get("name", "")).lower() == name.lower():
                return str(h.get("value", ""))
        return ""

    @staticmethod
    def message_text(message: dict[str, Any]) -> str:
        """Extracts the plain-text body from a message fetched with format='full'."""

        def walk(part: dict[str, Any]) -> str:
            mime = part.get("mimeType", "")
            data = part.get("body", {}).get("data")
            if mime == "text/plain" and data:
                return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
            for sub in part.get("parts", []) or []:
                text = walk(sub)
                if text:
                    return text
            if mime == "text/html" and data:
                import re

                raw = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
                return re.sub(r"<[^>]+>", " ", raw)
            return ""

        return walk(message.get("payload", {})) or str(message.get("snippet", ""))

    @staticmethod
    def has_calendar_invite(message: dict[str, Any]) -> bool:
        def walk(part: dict[str, Any]) -> bool:
            if part.get("mimeType") == "text/calendar" or str(part.get("filename", "")).endswith(".ics"):
                return True
            return any(walk(p) for p in part.get("parts", []) or [])

        return walk(message.get("payload", {}))
