"""Token-usage reporting from the `llm_usage` table."""

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.db.models import LLMUsage

PREMIUM_TIERS = ("premium", "premium_fast")


def usage_summary(session: Session, days: int = 1, budget: int = 190_000, reserve: int = 120_000) -> dict[str, Any]:
    today = datetime.now(UTC).replace(tzinfo=None).replace(hour=0, minute=0, second=0, microsecond=0)
    since = today - timedelta(days=max(days, 1) - 1)
    grouped = (
        session.query(
            LLMUsage.task,
            LLMUsage.tier,
            LLMUsage.provider,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.prompt_tokens), 0),
            func.coalesce(func.sum(LLMUsage.completion_tokens), 0),
            func.coalesce(func.sum(LLMUsage.total_tokens), 0),
            func.coalesce(func.sum(func.cast(~LLMUsage.success, type_=LLMUsage.id.type)), 0),
            func.avg(LLMUsage.latency_ms),
        )
        .filter(LLMUsage.timestamp >= since)
        .group_by(LLMUsage.task, LLMUsage.tier, LLMUsage.provider)
        .all()
    )
    by_task = [
        {
            "task": task,
            "tier": tier,
            "provider": provider,
            "calls": int(calls),
            "prompt_tokens": int(prompt),
            "completion_tokens": int(completion),
            "total_tokens": int(total),
            "failures": int(failures or 0),
            "avg_latency_ms": int(latency or 0),
        }
        for task, tier, provider, calls, prompt, completion, total, failures, latency in grouped
    ]
    by_task.sort(key=lambda r: r["total_tokens"], reverse=True)

    def tokens_today(tiers: tuple[str, ...]) -> int:
        value = (
            session.query(func.coalesce(func.sum(LLMUsage.total_tokens), 0))
            .filter(LLMUsage.timestamp >= today, LLMUsage.tier.in_(tiers))
            .scalar()
        )
        return int(value or 0)

    premium_today = tokens_today(PREMIUM_TIERS)
    local_today = tokens_today(("local",))
    per_stage: dict[str, int] = {}
    for row in by_task:
        per_stage[row["task"]] = per_stage.get(row["task"], 0) + row["total_tokens"]
    total = premium_today + local_today
    return {
        "since": since.isoformat(),
        "days": days,
        "by_task": by_task,
        "per_stage": dict(sorted(per_stage.items(), key=lambda kv: kv[1], reverse=True)),
        "premium_used_today": premium_today,
        "local_used_today": local_today,
        "premium_budget": budget,
        "premium_reserved_for_emails": reserve,
        "premium_remaining_today": max(0, budget - premium_today),
        "premium_share_today_pct": round(100.0 * premium_today / total, 1) if total else 0.0,
    }
