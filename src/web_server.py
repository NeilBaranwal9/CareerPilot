import contextlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

DEFAULT_WIDGET_PORT = 18492

STAGE_NAMES = {
    0: "Company Discovery",
    1: "Job Discovery",
    2: "Filtering & Screening",
    3: "Company Research",
    4: "Contact Research",
    5: "Email Discovery",
    6: "Opportunity Scoring",
    7: "Resume Tailoring",
    8: "Email Generation",
    9: "Validation & QC",
    10: "Gmail Draft Creation",
    11: "DB Finalization",
    12: "Completed",
}

TERMINAL_STATES = (
    "Completed",
    "Skipped",
    "Duplicate",
    "Salary Too Low",
    "Low Score",
    "Poor Fit",
    "Excluded Company",
    "Ghost Job",
    "Failed",
    "No Professional Email",
    "Do Not Contact",
)

# Global status tracking for async actions
_action_lock = threading.Lock()
_action_state = {
    "is_running": False,
    "current_action": None,
    "last_message": "Idle",
    "error": None,
}


def check_runner_active() -> dict[str, Any]:
    """Check if systemd service, CLI run process, or web thread is actively executing."""
    with _action_lock:
        if _action_state["is_running"]:
            return {
                "is_running": True,
                "action": _action_state["current_action"] or "UI Action",
                "message": _action_state["last_message"] or "Executing in background...",
            }

    systemd_active = False
    try:
        res = subprocess.run(
            ["systemctl", "--user", "is-active", "careerpilot.service"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        if res.stdout.strip() == "active":
            systemd_active = True
    except Exception:
        pass

    active_cmd = ""
    my_pid = os.getpid()
    try:
        res = subprocess.run(
            ["ps", "-eo", "pid,cmd"], capture_output=True, text=True, timeout=1.5
        )
        for line in res.stdout.splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                pid_str, cmd = parts
                if pid_str.isdigit() and int(pid_str) != my_pid:
                    if any(
                        kw in cmd
                        for kw in (
                            "recruiting-platform run",
                            "recruiting-platform resume",
                            "recruiting-platform retry",
                            "scripts/run.sh",
                        )
                    ) and "web_server" not in cmd:
                        active_cmd = cmd
                        break
    except Exception:
        pass

    if systemd_active or active_cmd:
        msg = (
            "Systemd Agent Active (careerpilot.service)"
            if systemd_active
            else f"CLI Run Active ({active_cmd[:40]}...)"
        )
        return {
            "is_running": True,
            "action": "system_run",
            "message": msg,
        }

    return {
        "is_running": False,
        "action": None,
        "message": _action_state.get("last_message") or "Idle",
    }


def get_db_status(db_path: str) -> dict[str, Any]:
    """Retrieve job application status summary from SQLite database."""
    runner_status = check_runner_active()

    if not os.path.exists(db_path):
        return {
            "error": f"Database file not found at {db_path}",
            "active_jobs_count": 0,
            "jobs": [],
            "summary": {"total": 0, "completed": 0, "failed": 0, "filtered": 0, "active": 0},
            "action_status": {
                "is_running": runner_status["is_running"],
                "current_action": runner_status["action"],
                "last_message": runner_status["message"],
                "error": _action_state.get("error"),
            },
        }

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Summary counts
        cur.execute("SELECT COUNT(*) as total FROM applications")
        total = cur.fetchone()["total"]

        cur.execute(
            """
            SELECT COUNT(*) as completed FROM applications 
            WHERE state = 'Completed' OR id IN (SELECT application_id FROM emails WHERE gmail_draft_id IS NOT NULL)
            """
        )
        completed = cur.fetchone()["completed"]

        cur.execute(
            """
            SELECT COUNT(*) as failed FROM applications 
            WHERE state LIKE '%Failed%' OR state LIKE '%Error%'
            """
        )
        failed = cur.fetchone()["failed"]

        cur.execute(
            """
            SELECT COUNT(*) as filtered FROM applications
            WHERE state IN ('Excluded Company', 'Salary Too Low', 'Low Score', 'Poor Fit', 'Ghost Job', 'Duplicate',
                            'No Professional Email', 'Skipped', 'Do Not Contact')
            """
        )
        filtered = cur.fetchone()["filtered"]

        # Active running applications
        placeholders = ",".join("?" for _ in TERMINAL_STATES)
        cur.execute(
            f"""
            SELECT a.id, a.run_id, a.current_stage, a.state, a.updated_at, a.score,
                   j.title as job_title, c.name as company_name
            FROM applications a
            JOIN jobs j ON a.job_id = j.id
            JOIN companies c ON j.company_id = c.id
            WHERE a.state NOT IN ({placeholders})
            ORDER BY a.updated_at DESC
            """,
            TERMINAL_STATES,
        )
        active_rows = cur.fetchall()
        jobs = []
        for r in active_rows:
            stage_num = r["current_stage"]
            stage_name = STAGE_NAMES.get(stage_num, r["state"])
            percent = int((stage_num / 12.0) * 100) if stage_num <= 12 else 100

            # Calculate relative updated time
            up_str = str(r["updated_at"])
            up_ago = "Unknown"
            try:
                from datetime import datetime

                clean_dt = up_str.split(".")[0]
                dt = datetime.strptime(clean_dt, "%Y-%m-%d %H:%M:%S")
                diff_sec = int((datetime.now() - dt).total_seconds())
                if diff_sec < 60:
                    up_ago = "Just now"
                elif diff_sec < 3600:
                    up_ago = f"{diff_sec // 60}m ago"
                elif diff_sec < 86400:
                    up_ago = f"{diff_sec // 3600}h ago"
                else:
                    up_ago = f"{diff_sec // 86400}d ago"
            except Exception:
                up_ago = up_str

            jobs.append(
                {
                    "id": r["id"],
                    "run_id": r["run_id"],
                    "company_name": r["company_name"],
                    "job_title": r["job_title"],
                    "current_stage": stage_num,
                    "stage_name": stage_name,
                    "stage_percent": percent,
                    "state": r["state"],
                    "updated_at": up_str,
                    "updated_ago": up_ago,
                    "score": round(r["score"], 2) if r["score"] else None,
                }
            )

        # Active runs count
        cur.execute("SELECT COUNT(*) as active_runs FROM runs WHERE status = 'running'")
        active_runs_res = cur.fetchone()
        active_runs = active_runs_res["active_runs"] if active_runs_res else 0

        conn.close()

        return {
            "active_jobs_count": len(jobs),
            "active_runs_count": active_runs,
            "summary": {
                "total": total,
                "completed": completed,
                "failed": failed,
                "filtered": filtered,
                "active": len(jobs),
            },
            "jobs": jobs,
            "action_status": {
                "is_running": runner_status["is_running"],
                "current_action": runner_status["action"],
                "last_message": runner_status["message"],
                "error": _action_state.get("error"),
            },
        }
    except Exception as e:
        return {
            "error": str(e),
            "active_jobs_count": 0,
            "jobs": [],
            "summary": {"total": 0, "completed": 0, "failed": 0, "filtered": 0, "active": 0},
            "action_status": _action_state,
        }


def get_all_applications(db_path: str) -> list[dict[str, Any]]:
    """Retrieve full list of applications for table view."""
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            """
            SELECT a.id, a.run_id, a.current_stage, a.state, a.score, a.tailored_resume_path, a.updated_at,
                   j.title as job_title, c.name as company_name, ct.email as contact_email
            FROM applications a
            JOIN jobs j ON a.job_id = j.id
            JOIN companies c ON j.company_id = c.id
            LEFT JOIN contacts ct ON a.contact_id = ct.id
            ORDER BY a.updated_at DESC
            """
        )
        rows = cur.fetchall()
        apps = []
        for r in rows:
            stage_num = r["current_stage"]
            res_filename = os.path.basename(r["tailored_resume_path"]) if r["tailored_resume_path"] else None
            apps.append(
                {
                    "id": r["id"],
                    "run_id": r["run_id"],
                    "company_name": r["company_name"],
                    "job_title": r["job_title"],
                    "current_stage": stage_num,
                    "stage_name": STAGE_NAMES.get(stage_num, r["state"]),
                    "state": r["state"],
                    "score": round(r["score"], 2) if r["score"] is not None else None,
                    "contact_email": r["contact_email"] or "N/A",
                    "resume_path": r["tailored_resume_path"] or "N/A",
                    "resume_file": res_filename or "N/A",
                    "updated_at": str(r["updated_at"]),
                }
            )
        conn.close()
        return apps
    except Exception:
        return []


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        with contextlib.suppress(Exception):
            return json.loads(value)
    return value


def get_funnel(db_path: str, campaign_id: int | None = None) -> dict[str, Any]:
    """Conversion funnel + email verification stats for the dashboard."""
    if not os.path.exists(db_path):
        return {"funnel": [], "extra": {}, "verification": {}}
    from src.analytics.funnel import compute_funnel, email_verification_stats
    from src.db.session import get_session_factory

    session = get_session_factory(db_path)()
    try:
        data = compute_funnel(session, campaign_id)
        data["verification"] = email_verification_stats(session)
        return data
    finally:
        session.close()


def get_campaigns(db_path: str) -> list[dict[str, Any]]:
    if not os.path.exists(db_path):
        return []
    from src.db.models import Campaign
    from src.db.session import get_session_factory
    from src.pipeline.campaign import campaign_progress

    session = get_session_factory(db_path)()
    try:
        return [campaign_progress(session, c) for c in session.query(Campaign).order_by(Campaign.id.desc()).all()]
    finally:
        session.close()


def get_application_details(db_path: str, app_id: int) -> dict[str, Any] | None:
    """Retrieve complete application inspection details."""
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            """
            SELECT a.id, a.run_id, a.current_stage, a.state, a.score, a.score_breakdown,
                   a.tailored_resume_path, a.created_at, a.updated_at,
                   a.outreach_status, a.persona, a.response_probability, a.reply_category, a.reply_summary,
                   j.title as job_title, j.location as job_location, j.salary as job_salary, j.description as job_description,
                   j.source as job_source,
                   c.name as company_name, c.domain as company_domain, c.industry as company_industry, c.employee_count as company_employees,
                   c.sector as company_sector, c.funding_stage as company_funding, c.hiring_status as company_hiring,
                   c.fit_score as company_fit, c.recent_news as company_news, c.tech_stack as company_stack,
                   ct.name as contact_name, ct.role as contact_role, ct.email as contact_email,
                   ct.role_category as contact_category, ct.email_status as contact_email_status,
                   ct.email_confidence as contact_email_confidence, ct.source as contact_source,
                   ct.linkedin_url as contact_linkedin
            FROM applications a
            JOIN jobs j ON a.job_id = j.id
            JOIN companies c ON j.company_id = c.id
            LEFT JOIN contacts ct ON a.contact_id = ct.id
            WHERE a.id = ?
            """,
            (app_id,),
        )
        app_row = cur.fetchone()
        if not app_row:
            conn.close()
            return None

        # Parse score breakdown
        breakdown = None
        if app_row["score_breakdown"]:
            try:
                breakdown = json.loads(app_row["score_breakdown"])
            except Exception:
                breakdown = app_row["score_breakdown"]

        # Fetch the email sequence (initial email first, then follow-ups)
        cur.execute(
            """
            SELECT subject, body, gmail_draft_id, status, scheduled_at, created_at, sequence_step, sent_at, tone, persona
            FROM emails WHERE application_id = ?
            ORDER BY COALESCE(sequence_step, 0) ASC, id DESC
            """,
            (app_id,),
        )
        sequence = []
        for email_row in cur.fetchall():
            sequence.append(
                {
                    "step": email_row["sequence_step"] or 0,
                    "subject": email_row["subject"],
                    "body": email_row["body"],
                    "gmail_draft_id": email_row["gmail_draft_id"],
                    "status": email_row["status"],
                    "scheduled_at": str(email_row["scheduled_at"]) if email_row["scheduled_at"] else None,
                    "sent_at": str(email_row["sent_at"]) if email_row["sent_at"] else None,
                    "tone": email_row["tone"],
                    "persona": email_row["persona"],
                    "created_at": str(email_row["created_at"]),
                }
            )
        email_info = sequence[0] if sequence else None

        cur.execute(
            "SELECT event_type, timestamp, details FROM outreach_events WHERE application_id = ? ORDER BY id DESC",
            (app_id,),
        )
        events = [
            {"type": e["event_type"], "timestamp": str(e["timestamp"]), "details": e["details"]} for e in cur.fetchall()
        ]

        # Fetch latest resume version
        cur.execute(
            """
            SELECT parent_resume, company, role, keywords_added, reasoning, path, created_at
            FROM resume_versions WHERE application_id = ? ORDER BY id DESC LIMIT 1
            """,
            (app_id,),
        )
        res_row = cur.fetchone()
        resume_info = None
        if res_row:
            kw_added = res_row["keywords_added"]
            if isinstance(kw_added, str):
                with contextlib.suppress(Exception):
                    kw_added = json.loads(kw_added)
            resume_info = {
                "parent_resume": res_row["parent_resume"],
                "company": res_row["company"],
                "role": res_row["role"],
                "keywords_added": kw_added,
                "reasoning": res_row["reasoning"],
                "path": res_row["path"],
                "created_at": str(res_row["created_at"]),
            }

        # Fetch history records
        cur.execute(
            """
            SELECT stage, state, timestamp, notes FROM history
            WHERE application_id = ? ORDER BY timestamp DESC
            """,
            (app_id,),
        )
        history_rows = cur.fetchall()
        history_info = [
            {
                "stage": h["stage"],
                "state": h["state"],
                "timestamp": str(h["timestamp"]),
                "notes": h["notes"],
            }
            for h in history_rows
        ]

        conn.close()
        return {
            "id": app_row["id"],
            "run_id": app_row["run_id"],
            "current_stage": app_row["current_stage"],
            "stage_name": STAGE_NAMES.get(app_row["current_stage"], app_row["state"]),
            "state": app_row["state"],
            "score": round(app_row["score"], 2) if app_row["score"] is not None else None,
            "score_breakdown": breakdown,
            "tailored_resume_path": app_row["tailored_resume_path"],
            "created_at": str(app_row["created_at"]),
            "updated_at": str(app_row["updated_at"]),
            "job": {
                "title": app_row["job_title"],
                "location": app_row["job_location"] or "N/A",
                "salary": app_row["job_salary"] or "N/A",
                "description": app_row["job_description"] or "",
            },
            "outreach_status": app_row["outreach_status"],
            "persona": app_row["persona"],
            "response_probability": app_row["response_probability"],
            "reply": {"category": app_row["reply_category"], "summary": app_row["reply_summary"]},
            "company": {
                "name": app_row["company_name"],
                "domain": app_row["company_domain"] or "N/A",
                "industry": app_row["company_industry"] or "N/A",
                "employee_count": app_row["company_employees"] or "N/A",
                "sector": app_row["company_sector"],
                "funding_stage": app_row["company_funding"],
                "hiring_status": app_row["company_hiring"],
                "fit_score": app_row["company_fit"],
                "recent_news": _loads(app_row["company_news"]),
                "tech_stack": _loads(app_row["company_stack"]),
            },
            "contact": {
                "name": app_row["contact_name"] or "N/A",
                "role": app_row["contact_role"] or "N/A",
                "email": app_row["contact_email"] or "N/A",
                "role_category": app_row["contact_category"],
                "email_status": app_row["contact_email_status"],
                "email_confidence": app_row["contact_email_confidence"],
                "source": app_row["contact_source"],
                "linkedin_url": app_row["contact_linkedin"],
            },
            "job_source": app_row["job_source"],
            "email": email_info,
            "email_sequence": sequence,
            "events": events,
            "resume": resume_info,
            "history": history_info,
        }
    except Exception:
        return None


def get_companies(db_path: str) -> list[dict[str, Any]]:
    """Retrieve list of companies."""
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, domain, employee_count, industry, created_at, sector, funding_stage, hiring_status,
                   fit_score, response_probability, status, source
            FROM companies
            ORDER BY COALESCE(fit_score, 0) * (0.5 + COALESCE(response_probability, 0.08) * 5) DESC, name ASC
            LIMIT 500
            """
        )
        rows = cur.fetchall()
        res = [
            {
                "id": r["id"],
                "name": r["name"],
                "domain": r["domain"] or "N/A",
                "employee_count": r["employee_count"] or "N/A",
                "industry": r["industry"] or "N/A",
                "created_at": str(r["created_at"]),
                "sector": r["sector"] or "N/A",
                "funding_stage": r["funding_stage"] or "N/A",
                "hiring_status": r["hiring_status"] or "N/A",
                "fit_score": round(r["fit_score"], 2) if r["fit_score"] is not None else None,
                "response_probability": r["response_probability"],
                "status": r["status"] or "N/A",
                "source": r["source"] or "N/A",
            }
            for r in rows
        ]
        conn.close()
        return res
    except Exception:
        return []


def get_contacts(db_path: str) -> list[dict[str, Any]]:
    """Retrieve list of contacts."""
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ct.id, ct.name, ct.role, ct.email, ct.linkedin_url, c.name as company_name,
                   ct.role_category, ct.email_status, ct.source, ct.rank_score
            FROM contacts ct
            JOIN companies c ON ct.company_id = c.id
            ORDER BY ct.id DESC LIMIT 500
            """
        )
        rows = cur.fetchall()
        res = [
            {
                "id": r["id"],
                "company_name": r["company_name"],
                "name": r["name"],
                "role": r["role"],
                "email": r["email"] or "N/A",
                "linkedin_url": r["linkedin_url"] or "N/A",
                "role_category": r["role_category"] or "N/A",
                "email_status": r["email_status"] or "N/A",
                "source": r["source"] or "N/A",
                "rank_score": r["rank_score"],
            }
            for r in rows
        ]
        conn.close()
        return res
    except Exception:
        return []


def get_jobs(db_path: str) -> list[dict[str, Any]]:
    """Retrieve list of jobs."""
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT j.id, j.title, j.location, j.salary, c.name as company_name
            FROM jobs j
            JOIN companies c ON j.company_id = c.id
            ORDER BY j.id DESC LIMIT 200
            """
        )
        rows = cur.fetchall()
        res = [
            {
                "id": r["id"],
                "company_name": r["company_name"],
                "title": r["title"],
                "location": r["location"] or "N/A",
                "salary": r["salary"] or "N/A",
            }
            for r in rows
        ]
        conn.close()
        return res
    except Exception:
        return []


def mark_email_nonexistent(db_path: str, app_id: int) -> dict[str, Any]:
    """
    Marks an application's email as non-existent/bounced.
    Updates application state to 'No Professional Email', clears the contact's email address,
    updates email status, and logs a history record.
    """
    if not os.path.exists(db_path):
        return {"success": False, "error": f"Database file not found at {db_path}"}
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("SELECT contact_id, run_id, current_stage FROM applications WHERE id = ?", (app_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return {"success": False, "error": f"Application #{app_id} not found."}

        contact_id = row[0]
        run_id = row[1] or "UNKNOWN"
        current_stage = row[2] or 0

        # Update application state
        cur.execute(
            "UPDATE applications SET state = 'No Professional Email', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (app_id,),
        )

        cleared_contact = False
        if contact_id:
            cur.execute("UPDATE contacts SET email = NULL WHERE id = ?", (contact_id,))
            cleared_contact = True

        # Update associated email status if any
        cur.execute(
            "UPDATE emails SET status = 'email_does_not_exist' WHERE application_id = ?",
            (app_id,),
        )

        # Ensure run_id exists in runs table before adding history record to prevent FK errors
        cur.execute("SELECT id FROM runs WHERE id = ?", (run_id,))
        if not cur.fetchone():
            cur.execute("INSERT INTO runs (id, status) VALUES (?, 'completed')", (run_id,))

        # Log history audit record
        cur.execute(
            """
            INSERT INTO history (application_id, stage, state, run_id, timestamp, notes)
            VALUES (?, ?, 'No Professional Email', ?, CURRENT_TIMESTAMP, 'Marked email as non-existent (bounced) via UI')
            """,
            (app_id, current_stage, run_id),
        )

        conn.commit()
        conn.close()

        msg = f"Marked Application #{app_id} as 'No Professional Email'"
        if cleared_contact:
            msg += " and cleared contact email address."

        return {"success": True, "message": msg}
    except Exception as e:
        return {"success": False, "error": str(e)}


def execute_pipeline_action_thread(
    config_path: str, db_path: str, action_type: str
) -> None:
    """Background task runner for pipeline execution and utilities."""
    global _action_state
    with _action_lock:
        _action_state["is_running"] = True
        _action_state["current_action"] = action_type
        _action_state["last_message"] = f"Executing {action_type}..."
        _action_state["error"] = None

    try:
        if action_type in ("run", "resume", "retry"):
            from src.pipeline.runner import PipelineRunner

            runner = PipelineRunner(config_path=config_path)
            if action_type == "retry":
                run_id = runner.retry_failed()
                msg = f"Retry completed for {run_id}!"
            else:
                run_id = runner.run(max_stage=12)
                msg = f"Pipeline run completed for {run_id}!"
        elif action_type in ("outreach", "daily"):
            from src.pipeline.runner import PipelineRunner

            runner = PipelineRunner(config_path=config_path)
            if action_type == "outreach":
                summary = runner.run_outreach_cycle()
                msg = f"Outreach cycle done: {summary or 'Gmail not authorized'}"
            else:
                runner.run_daily()
                msg = "Daily run completed!"
        elif action_type == "clean_invalid":
            from src.utils.cleaner import clean_invalid_emails_and_states

            res = clean_invalid_emails_and_states(db_path)
            cleared = res.get("cleared_emails_count", res.get("cleared_emails", 0))
            updated = res.get("updated_applications_count", res.get("updated_applications", 0))
            msg = f"Cleaned! Cleared {cleared} invalid emails & updated {updated} apps."
        elif action_type == "export":
            from src.utils.exporter import export_outreach_data

            res = export_outreach_data(db_path)
            msg = f"Exported {res['exported_count']} applications to CSV/JSON."
        else:
            msg = f"Unknown action {action_type}"

        with _action_lock:
            _action_state["is_running"] = False
            _action_state["last_message"] = msg
    except Exception as e:
        with _action_lock:
            _action_state["is_running"] = False
            _action_state["error"] = str(e)
            _action_state["last_message"] = f"Action {action_type} failed: {e}"


class StatusWidgetRequestHandler(BaseHTTPRequestHandler):
    db_path: str = "data/platform.db"
    config_path: str = "config.yaml"
    static_dir: Path = Path(__file__).parent / "static"

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP request logs for clean output."""
        return

    def _send_json(self, data: Any, status_code: int = 200) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)

        if path == "/api/status":
            status_data = get_db_status(self.db_path)
            self._send_json(status_data)
            return

        if path == "/api/applications":
            apps_data = get_all_applications(self.db_path)
            self._send_json(apps_data)
            return

        if path == "/api/application_details":
            app_id_str = query_params.get("id", [None])[0]
            if not app_id_str or not app_id_str.isdigit():
                self._send_json({"error": "Missing or invalid application ID"}, 400)
                return
            app_details = get_application_details(self.db_path, int(app_id_str))
            if app_details is None:
                self._send_json({"error": "Application not found"}, 404)
            else:
                self._send_json(app_details)
            return

        if path == "/api/companies":
            comps_data = get_companies(self.db_path)
            self._send_json(comps_data)
            return

        if path == "/api/contacts":
            contacts_data = get_contacts(self.db_path)
            self._send_json(contacts_data)
            return

        if path == "/api/jobs":
            jobs_data = get_jobs(self.db_path)
            self._send_json(jobs_data)
            return

        if path == "/api/funnel":
            campaign_str = query_params.get("campaign", [None])[0]
            campaign_id = int(campaign_str) if campaign_str and campaign_str.isdigit() else None
            try:
                self._send_json(get_funnel(self.db_path, campaign_id))
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/api/campaigns":
            try:
                self._send_json(get_campaigns(self.db_path))
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/api/config":
            content = ""
            if os.path.exists(self.config_path):
                with open(self.config_path, encoding="utf-8") as f:
                    content = f.read()
            self._send_json({"config": content})
            return

        if path == "/api/logs":
            log_lines = []
            log_path = "logs/platform.log"
            if os.path.exists(log_path):
                try:
                    with open(log_path, encoding="utf-8") as f:
                        log_lines = [line.strip() for line in f.readlines()[-150:]]
                except Exception as e:
                    log_lines = [f"Error reading log file: {e}"]
            else:
                log_lines = ["No active log file found at logs/platform.log."]
            self._send_json({"logs": log_lines})
            return

        # Serve static widget page
        filepath = self.static_dir / "index.html"
        if filepath.exists():
            html_bytes = filepath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_bytes)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path in (
            "/api/actions/run",
            "/api/actions/resume",
            "/api/actions/retry",
            "/api/actions/clean_invalid",
            "/api/actions/export",
            "/api/actions/outreach",
            "/api/actions/daily",
        ):
            action_type = path.split("/")[-1]
            runner_stat = check_runner_active()
            if runner_stat["is_running"]:
                self._send_json(
                    {
                        "success": False,
                        "message": f"An execution process ('{runner_stat['message']}') is already running.",
                    },
                    400,
                )
                return

            t = threading.Thread(
                target=execute_pipeline_action_thread,
                args=(self.config_path, self.db_path, action_type),
                daemon=True,
            )
            t.start()

            self._send_json(
                {
                    "success": True,
                    "message": f"Action '{action_type}' initiated in background.",
                }
            )
            return

        if path == "/api/actions/update_app_state":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode("utf-8"))
                app_id = int(payload["id"])
                new_state = str(payload["state"])

                conn = sqlite3.connect(self.db_path)
                cur = conn.cursor()
                cur.execute(
                    "UPDATE applications SET state = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (new_state, app_id),
                )
                conn.commit()
                conn.close()
                self._send_json(
                    {
                        "success": True,
                        "message": f"Updated Application #{app_id} to '{new_state}'",
                    }
                )
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, 400)
            return

        if path == "/api/actions/mark_email_nonexistent":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode("utf-8"))
                app_id = int(payload["id"])
                result = mark_email_nonexistent(self.db_path, app_id)
                status_code = 200 if result.get("success") else 400
                self._send_json(result, status_code)
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, 400)
            return

        self._send_json({"error": "Endpoint not found"}, 404)


def is_widget_server_running(port: int = DEFAULT_WIDGET_PORT) -> bool:
    """Check if status widget HTTP server is already running on the given port."""
    try:
        url = f"http://127.0.0.1:{port}/api/status"
        req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return bool(resp.status == 200)
    except Exception:
        return False


def ensure_widget_server_running(
    db_path: str = "data/platform.db",
    config_path: str = "config.yaml",
    port: int = DEFAULT_WIDGET_PORT,
    auto_open: bool = False,
) -> bool:
    """
    Automates widget launching: checks if widget server is active on port 18492.
    If not running, spawns it as a background process and optionally opens it in browser.
    """
    widget_url = f"http://127.0.0.1:{port}"
    if is_widget_server_running(port):
        print(f"★ Job Status Dark Mode Widget already active at {widget_url}")
        return True

    print(f"★ Launching Minimal Dark Mode Job Status Widget on port {port}...")
    try:
        cmd = [sys.executable, "-m", "src.web_server", "--port", str(port), "--db", db_path, "--config", config_path]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if auto_open:
            with contextlib.suppress(Exception):
                webbrowser.open(widget_url)
        return True
    except Exception as e:
        print(f"Could not auto-start widget server: {e}")
        return False


def run_widget_server(
    db_path: str = "data/platform.db",
    config_path: str = "config.yaml",
    port: int = DEFAULT_WIDGET_PORT,
    host: str = "127.0.0.1",
) -> None:
    """Launch HTTP server serving the dark mode job status widget & web dashboard."""
    with contextlib.suppress(Exception):
        from src.db.session import init_db

        init_db(db_path)  # make sure new columns/tables exist before serving
    StatusWidgetRequestHandler.db_path = db_path
    StatusWidgetRequestHandler.config_path = config_path
    server_address = (host, port)
    httpd = HTTPServer(server_address, StatusWidgetRequestHandler)
    print(f"★ Job Status Dark Mode Widget & Web Board running on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Job Status Widget Server...")
        httpd.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Job Status Widget Web Server")
    parser.add_argument("--port", type=int, default=DEFAULT_WIDGET_PORT, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--db", type=str, default="data/platform.db", help="Path to SQLite database")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration")
    args = parser.parse_args()

    run_widget_server(db_path=args.db, config_path=args.config, port=args.port, host=args.host)
