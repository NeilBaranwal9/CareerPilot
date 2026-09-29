"""
Task-based LLM routing with token accounting.

Every LLM call names a task (discovery, research, contact_finding, email_generation, ...). The router maps the
task to a tier:
  local        -> Ollama (default for extraction, classification, scoring, validation, research)
  premium      -> Groq main model (email writing)
  premium_fast -> Groq fast model (follow-ups; fallback when the local model is unavailable)
and records prompt/completion tokens for every call. Premium usage is capped by a daily budget, and local->premium
fallbacks may only spend (budget - reserve) so email writing always keeps its share.
"""

import logging
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.config import LLMConfig
from src.providers.llm.base import BaseLLMProvider
from src.providers.llm.ollama import OllamaProvider, OllamaUnavailableError

logger = logging.getLogger("recruiting-platform.llm.router")

TASKS = (
    "discovery", "campaign_parsing", "targeted_parsing", "job_extraction", "research", "contact_finding",
    "email_finding", "scoring", "resume_tailoring", "email_generation", "email_regeneration",
    "followup_generation", "validation", "reply_classification", "summarization",
)
PREMIUM_TIERS = ("premium", "premium_fast")


class BudgetExceededError(RuntimeError):
    """The premium provider's daily token budget (or the share available to this task) is used up."""


def _utc_day_start() -> datetime:
    now = datetime.now(UTC).replace(tzinfo=None)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text or "") / 3.6))


class UsageTracker:
    """Buffers token usage in memory and flushes it to the `llm_usage` table at safe points."""

    def __init__(self, session_factory: Callable[[], Session] | None = None):
        self.session_factory = session_factory
        self.pending: list[dict[str, Any]] = []
        self.run_id: str | None = None
        self._premium_db_today: int | None = None
        self._day: datetime | None = None

    def record(
        self,
        task: str,
        tier: str,
        provider: str,
        model: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        estimated: bool,
        success: bool,
        latency_ms: int,
    ) -> None:
        self.pending.append(
            {
                "timestamp": datetime.now(UTC).replace(tzinfo=None),
                "run_id": self.run_id,
                "task": task,
                "tier": tier,
                "provider": provider,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated": estimated,
                "success": success,
                "latency_ms": latency_ms,
            }
        )

    def _load_premium_today(self) -> int:
        day = _utc_day_start()
        if self._day != day or self._premium_db_today is None:
            self._day = day
            self._premium_db_today = 0
            if self.session_factory is not None:
                from src.db.models import LLMUsage

                try:
                    session = self.session_factory()
                    try:
                        total = (
                            session.query(func.coalesce(func.sum(LLMUsage.total_tokens), 0))
                            .filter(LLMUsage.tier.in_(PREMIUM_TIERS), LLMUsage.timestamp >= day)
                            .scalar()
                        )
                        self._premium_db_today = int(total or 0)
                    finally:
                        session.close()
                except Exception as e:
                    logger.debug(f"Could not read premium usage from DB: {e}")
        return int(self._premium_db_today or 0)

    def premium_used_today(self) -> int:
        day = _utc_day_start()
        pending = sum(r["total_tokens"] for r in self.pending if r["tier"] in PREMIUM_TIERS and r["timestamp"] >= day)
        return self._load_premium_today() + int(pending)

    def flush(self) -> int:
        """Writes buffered usage rows; never raises (accounting must not break the pipeline)."""
        if not self.pending or self.session_factory is None:
            return 0
        from src.db.models import LLMUsage

        rows = list(self.pending)
        try:
            session = self.session_factory()
            try:
                session.add_all(LLMUsage(**row) for row in rows)
                session.commit()
            finally:
                session.close()
        except Exception as e:
            logger.warning(f"Could not store LLM usage: {e}")
            return 0
        self.pending = self.pending[len(rows) :]
        day = _utc_day_start()
        flushed_premium = sum(r["total_tokens"] for r in rows if r["tier"] in PREMIUM_TIERS and r["timestamp"] >= day)
        if self._premium_db_today is not None and self._day == day:
            self._premium_db_today += flushed_premium
        return len(rows)

    def summary(self) -> dict[str, dict[str, int]]:
        """Pending (not yet flushed) usage per task: {task: {tokens, calls}}."""
        out: dict[str, dict[str, int]] = {}
        for r in self.pending:
            row = out.setdefault(r["task"], {"tokens": 0, "calls": 0})
            row["tokens"] += r["total_tokens"]
            row["calls"] += 1
        return out


class RoutedLLM(BaseLLMProvider):
    """A provider bound to one task; the router chooses the real provider at call time."""

    provider_name = "router"

    def __init__(self, router: "LLMRouter", task: str):
        super().__init__(api_key="", model=f"routed:{task}")
        self.router = router
        self.task = task

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        result = self.router.call(self.task, prompt, None, system_prompt)
        assert isinstance(result, str)
        return result

    def generate_json(self, prompt: str, schema: type[BaseModel], system_prompt: str | None = None) -> BaseModel:
        result = self.router.call(self.task, prompt, schema, system_prompt)
        assert isinstance(result, BaseModel)
        return result


class LLMRouter:
    def __init__(
        self,
        config: LLMConfig,
        tracker: UsageTracker,
        premium: BaseLLMProvider | None = None,
        premium_fast: BaseLLMProvider | None = None,
        local: BaseLLMProvider | None = None,
    ):
        from src.providers.llm import get_llm_provider

        self.config = config
        self.tracker = tracker
        self.premium = premium or get_llm_provider(config)
        self.premium_fast = premium_fast or (get_llm_provider(config, fast=True) if config.fast_model else self.premium)
        if local is not None:
            self.local: BaseLLMProvider | None = local
        elif config.local_provider.lower() == "ollama":
            self.local = OllamaProvider(
                model=config.local_model,
                api_url=config.local_url,
                max_tokens=config.local_max_tokens,
                num_ctx=config.local_num_ctx,
                timeout_seconds=config.local_timeout_seconds,
            )
        else:
            self.local = None
        self._local_ok: bool | None = None
        self.local_status = "local model disabled" if self.local is None else "not checked"

    # -- routing ---------------------------------------------------------

    def tier_for(self, task: str) -> str:
        tier = self.config.routing.get(task, self.config.routing.get("default", "local"))
        if tier not in ("local", "premium", "premium_fast"):
            tier = "local"
        if tier == "local" and self.local is None:
            return "premium_fast"
        return tier

    def get(self, task: str) -> RoutedLLM:
        return RoutedLLM(self, task)

    def local_available(self) -> bool:
        if self.local is None:
            return False
        if self._local_ok is None:
            health = getattr(self.local, "health", None)
            if callable(health):
                ok, message = health()
                self._local_ok, self.local_status = bool(ok), str(message)
                if not ok:
                    logger.warning(f"Local LLM unavailable: {message}")
            else:
                self._local_ok, self.local_status = True, "local provider ready"
        return bool(self._local_ok)

    def _ensure_budget(self, task: str, fallback: bool) -> None:
        used = self.tracker.premium_used_today()
        limit = self.config.groq_daily_token_budget
        if fallback:
            limit -= self.config.groq_reserved_for_emails
        if used >= limit:
            scope = "the non-email share of " if fallback else ""
            raise BudgetExceededError(
                f"Premium LLM budget exhausted for task '{task}': {used:,} tokens used today, "
                f"{scope}daily budget is {limit:,}. Work resumes tomorrow (or raise llm.groq_daily_token_budget)."
            )

    # -- execution -------------------------------------------------------

    def call(self, task: str, prompt: str, schema: type[BaseModel] | None, system_prompt: str | None) -> Any:
        tier = self.tier_for(task)
        if tier == "local":
            if self.local_available():
                assert self.local is not None
                try:
                    return self._invoke(self.local, task, "local", prompt, schema, system_prompt)
                except OllamaUnavailableError as e:
                    self._local_ok, self.local_status = False, str(e)
                    logger.warning(f"Local LLM failed ({e}); considering fallback.")
            if self.config.local_fallback != "premium_fast":
                raise OllamaUnavailableError(f"Local LLM unavailable for task '{task}': {self.local_status}")
            self._ensure_budget(task, fallback=True)
            return self._invoke(self.premium_fast, task, "premium_fast", prompt, schema, system_prompt)
        provider = self.premium if tier == "premium" else self.premium_fast
        self._ensure_budget(task, fallback=False)
        return self._invoke(provider, task, tier, prompt, schema, system_prompt)

    def _invoke(
        self,
        provider: BaseLLMProvider,
        task: str,
        tier: str,
        prompt: str,
        schema: type[BaseModel] | None,
        system_prompt: str | None,
    ) -> Any:
        start = time.monotonic()
        success = False
        output = ""
        try:
            if schema is None:
                result: Any = provider.generate_text(prompt, system_prompt)
                output = str(result)
            else:
                result = provider.generate_json(prompt, schema, system_prompt)
                output = result.model_dump_json()
            success = True
            return result
        finally:
            usage = provider.last_usage
            estimated = not usage or not (usage.get("prompt_tokens") or usage.get("completion_tokens"))
            if estimated:
                prompt_tokens = estimate_tokens(prompt + (system_prompt or ""))
                completion_tokens = estimate_tokens(output) if success else 0
                model = provider.model
            else:
                assert usage is not None
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                model = str(usage.get("model") or provider.model)
            self.tracker.record(
                task, tier, getattr(provider, "provider_name", "llm"), model, prompt_tokens, completion_tokens,
                estimated, success, int((time.monotonic() - start) * 1000),
            )

    def describe(self) -> list[tuple[str, str, str]]:
        """(task, tier, model) for every known task — used by `doctor` and the dashboard."""
        rows = []
        for task in TASKS:
            tier = self.tier_for(task)
            provider = self.local if tier == "local" else self.premium if tier == "premium" else self.premium_fast
            rows.append((task, tier, getattr(provider, "model", "?") if provider else "?"))
        return rows
