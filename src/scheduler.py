"""
Continuous operation:
- `run_daemon`      : in-process loop (any OS) running the outreach cycle every N minutes and the daily run once a day
- Windows           : Task Scheduler tasks (CareerPilotDaily + CareerPilotOutreach)
- Linux             : systemd user timers (careerpilot.timer + careerpilot-outreach.timer)
"""

import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.timer_failsafe import check_and_manage_timer

logger = logging.getLogger("recruiting-platform.scheduler")

TASK_DAILY = "CareerPilotDaily"
TASK_OUTREACH = "CareerPilotOutreach"


def _python() -> str:
    return sys.executable


def _write_windows_script(project_dir: Path, name: str, command: str) -> Path:
    script_dir = project_dir / "data" / "scheduler"
    script_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "logs").mkdir(exist_ok=True)
    script = script_dir / f"{name}.cmd"
    script.write_text(
        "@echo off\r\n"
        f'cd /d "{project_dir}"\r\n'
        f'"{_python()}" -m src.cli {command} >> "logs\\scheduler.log" 2>&1\r\n',
        encoding="utf-8",
    )
    return script


def install_windows_tasks(project_dir: Path, daily_time: str = "09:00", outreach_every_minutes: int = 30) -> list[str]:
    daily_script = _write_windows_script(project_dir, "careerpilot_daily", "daily")
    outreach_script = _write_windows_script(project_dir, "careerpilot_outreach", "outreach")
    commands = [
        ["schtasks", "/Create", "/F", "/SC", "DAILY", "/ST", daily_time, "/TN", TASK_DAILY, "/TR", f'"{daily_script}"'],
        [
            "schtasks", "/Create", "/F", "/SC", "MINUTE", "/MO", str(outreach_every_minutes),
            "/TN", TASK_OUTREACH, "/TR", f'"{outreach_script}"',
        ],
    ]
    output = []
    for cmd in commands:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output.append((res.stdout or res.stderr).strip())
        if res.returncode != 0:
            raise RuntimeError(f"schtasks failed: {res.stderr or res.stdout}")
    return output


def remove_windows_tasks() -> list[str]:
    output = []
    for task in (TASK_DAILY, TASK_OUTREACH):
        res = subprocess.run(["schtasks", "/Delete", "/F", "/TN", task], capture_output=True, text=True, timeout=30)
        output.append((res.stdout or res.stderr).strip())
    return output


def windows_task_status(task: str) -> str | None:
    """Returns 'Ready'/'Disabled'/'Running' for an existing task, or None if the task does not exist."""
    try:
        res = subprocess.run(
            ["schtasks", "/Query", "/TN", task, "/FO", "LIST"], capture_output=True, text=True, timeout=10
        )
    except Exception:
        return None
    if res.returncode != 0:
        return None
    for line in res.stdout.splitlines():
        if line.strip().lower().startswith("status:"):
            return line.split(":", 1)[1].strip()
    return "Unknown"


def install_systemd_units(project_dir: Path, daily_time: str = "09:00", outreach_every_minutes: int = 30) -> list[str]:
    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    python = _python()
    units = {
        "careerpilot.service": (
            "[Unit]\nDescription=CareerPilot daily discovery & outreach\n\n[Service]\nType=oneshot\n"
            f"WorkingDirectory={project_dir}\nExecStart={python} -m src.cli daily\n"
        ),
        "careerpilot.timer": (
            "[Unit]\nDescription=Run CareerPilot daily\n\n[Timer]\n"
            f"OnCalendar=*-*-* {daily_time}:00\nPersistent=true\n\n[Install]\nWantedBy=timers.target\n"
        ),
        "careerpilot-outreach.service": (
            "[Unit]\nDescription=CareerPilot outreach cycle (send, follow-ups, replies)\n\n[Service]\nType=oneshot\n"
            f"WorkingDirectory={project_dir}\nExecStart={python} -m src.cli outreach\n"
        ),
        "careerpilot-outreach.timer": (
            "[Unit]\nDescription=Run CareerPilot outreach cycle periodically\n\n[Timer]\n"
            f"OnBootSec=5min\nOnUnitActiveSec={outreach_every_minutes}min\n\n[Install]\nWantedBy=timers.target\n"
        ),
    }
    for name, content in units.items():
        (unit_dir / name).write_text(content, encoding="utf-8")
    output = []
    for cmd in (
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", "careerpilot.timer"],
        ["systemctl", "--user", "enable", "--now", "careerpilot-outreach.timer"],
    ):
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output.append((res.stdout or res.stderr).strip() or " ".join(cmd))
    return output


def install_schedule(project_dir: Path, daily_time: str = "09:00", outreach_every_minutes: int = 30) -> list[str]:
    if sys.platform == "win32":
        return install_windows_tasks(project_dir, daily_time, outreach_every_minutes)
    return install_systemd_units(project_dir, daily_time, outreach_every_minutes)


def remove_schedule() -> list[str]:
    if sys.platform == "win32":
        return remove_windows_tasks()
    output = []
    for timer in ("careerpilot.timer", "careerpilot-outreach.timer"):
        res = subprocess.run(["systemctl", "--user", "disable", "--now", timer], capture_output=True, text=True, timeout=30)
        output.append((res.stdout or res.stderr).strip() or f"disabled {timer}")
    return output


def manage_automation(automation_enabled: bool = True) -> bool:
    """
    Cross-platform failsafe honoring `pipeline.automation`:
    enabled -> re-enables a disabled scheduled task/timer; disabled -> disables it.
    """
    if sys.platform != "win32":
        return check_and_manage_timer(automation_enabled)
    try:
        status = windows_task_status(TASK_DAILY)
        if status is None:
            if automation_enabled:
                logger.debug("No scheduled task found. Run `recruiting-platform schedule install` for daily automation.")
            return False
        disabled = status.lower() == "disabled"
        if automation_enabled and disabled:
            for task in (TASK_DAILY, TASK_OUTREACH):
                subprocess.run(["schtasks", "/Change", "/TN", task, "/ENABLE"], capture_output=True, text=True, timeout=10)
            logger.warning("Automation enabled in config: re-enabled CareerPilot scheduled tasks.")
            return True
        if not automation_enabled and not disabled:
            for task in (TASK_DAILY, TASK_OUTREACH):
                subprocess.run(["schtasks", "/Change", "/TN", task, "/DISABLE"], capture_output=True, text=True, timeout=10)
            logger.warning("Automation disabled in config: disabled CareerPilot scheduled tasks.")
            return False
        return automation_enabled
    except Exception as e:
        logger.warning(f"Unable to manage Windows scheduled tasks: {e}")
        return False


def run_daemon(
    runner_factory: Callable[[], Any],
    daily_at: str = "09:00",
    outreach_every_minutes: int = 15,
    max_iterations: int | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = datetime.now,
) -> None:
    """Foreground scheduler loop. Runs the outreach cycle every N minutes and the daily run once per day."""
    hour, minute = (int(x) for x in daily_at.split(":"))
    last_daily_date = None
    last_outreach = 0.0
    iterations = 0
    logger.info(f"Daemon started: daily run at {daily_at}, outreach cycle every {outreach_every_minutes} min.")
    while max_iterations is None or iterations < max_iterations:
        iterations += 1
        now = now_fn()
        try:
            if (now.hour, now.minute) >= (hour, minute) and last_daily_date != now.date():
                last_daily_date = now.date()
                logger.info("Daemon: starting daily run.")
                runner_factory().run_daily()
                last_outreach = time.monotonic()
            elif time.monotonic() - last_outreach >= outreach_every_minutes * 60 or last_outreach == 0.0:
                last_outreach = time.monotonic()
                summary = runner_factory().run_outreach_cycle()
                logger.info(f"Daemon: outreach cycle {summary}")
        except Exception as e:
            logger.error(f"Daemon iteration failed: {e}")
        if max_iterations is None or iterations < max_iterations:
            sleep_fn(60)


def project_root() -> Path:
    return Path(os.getcwd())
