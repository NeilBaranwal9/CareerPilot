"""Send-window scheduling in the recipient-friendly timezone (e.g. weekdays 9:00-12:00 IST)."""

import logging
import random
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

from src.config import SendWindow

logger = logging.getLogger("recruiting-platform.outreach.scheduling")

_FIXED_OFFSETS = {"Asia/Kolkata": timedelta(hours=5, minutes=30), "Asia/Calcutta": timedelta(hours=5, minutes=30)}


def resolve_timezone(name: str) -> tzinfo:
    try:
        return ZoneInfo(name)
    except Exception:
        if name in _FIXED_OFFSETS:
            return timezone(_FIXED_OFFSETS[name])
        logger.warning(f"Unknown timezone '{name}' (install the 'tzdata' package on Windows). Using UTC.")
        return UTC


def utc_now_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def next_send_slot(
    after_utc: datetime,
    window: SendWindow,
    not_before_utc: datetime | None = None,
    min_gap_minutes: int = 0,
    jitter_minutes: int = 15,
) -> datetime:
    """
    Returns the earliest naive-UTC datetime >= after_utc that falls inside the send window, at least
    `min_gap_minutes` after `not_before_utc` (the previous scheduled send), plus a little random jitter
    so sends don't go out on exact minute marks.
    """
    tz = resolve_timezone(window.timezone)
    earliest = after_utc
    if not_before_utc is not None:
        earliest = max(earliest, not_before_utc + timedelta(minutes=min_gap_minutes))
    local = earliest.replace(tzinfo=UTC).astimezone(tz)
    start_hour = max(0, min(23, window.start_hour))
    end_hour = max(start_hour + 1, min(24, window.end_hour))

    for _ in range(15):
        day_start = local.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(hours=end_hour - start_hour)
        if window.weekdays_only and local.weekday() >= 5:
            local = (day_start + timedelta(days=1)).replace(hour=start_hour)
            continue
        if local < day_start:
            local = day_start
        if local >= day_end:
            local = day_start + timedelta(days=1)
            continue
        if jitter_minutes > 0:
            room = int((day_end - local).total_seconds() // 60) - 1
            if room > 0:
                local += timedelta(minutes=random.randint(0, min(jitter_minutes, room)))
        return local.astimezone(UTC).replace(tzinfo=None)
    return earliest


def followup_due_at(sent_at_utc: datetime, after_days: int, window: SendWindow) -> datetime:
    return next_send_slot(sent_at_utc + timedelta(days=after_days), window)
