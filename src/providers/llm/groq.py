import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from src.providers.llm.base import BaseLLMProvider

logger = logging.getLogger("recruiting-platform.llm.groq")

GROQ_API_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# Waits longer than this (e.g. a daily token quota reset) are not slept through; we switch model instead.
MAX_RATE_LIMIT_WAIT_SECONDS = 60.0

JSON_SYSTEM_PROMPT = (
    "You are a precise research and writing assistant inside an automated job-outreach pipeline. "
    "Always answer with a single valid JSON object and nothing else. Never invent facts that are not "
    "supported by the provided context; use null or empty lists when information is unknown."
)


class ModelUnavailableError(RuntimeError):
    """The requested model does not exist, was decommissioned, or is not enabled for this API key."""


class RateLimitExhaustedError(RuntimeError):
    """Rate limits persisted after all retries (or the reset window is too long to wait for)."""


def _parse_duration_seconds(value: str | None) -> float | None:
    """Parses Groq reset headers such as '7.66s', '2m59.56s', '1h2m3s' or plain seconds '12'."""
    if not value:
        return None
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    total = 0.0
    matched = False
    for amount, unit in re.findall(r"([\d.]+)(ms|h|m|s)", value):
        matched = True
        num = float(amount)
        total += {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}[unit] * num
    return total if matched else None


def extract_json_block(text: str) -> str:
    """Strips reasoning tags / markdown fences and returns the outermost JSON object in the text."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


class GroqProvider(BaseLLMProvider):
    provider_name = "groq"

    """
    Groq LLM provider (OpenAI-compatible Chat Completions API at api.groq.com).

    Features:
    - JSON mode (`response_format: json_object`) with Pydantic validation and one self-repair round.
    - Automatic retries honouring `retry-after` / `x-ratelimit-reset-*` headers on 429 and 5xx responses.
    - Model fallback chain when a model is rate limited for a long window, decommissioned or unavailable.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GROQ_MODEL,
        api_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
        fallback_models: list[str] | None = None,
        max_retries: int = 4,
        timeout_seconds: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ):
        resolved_model = model if model and model != "default" else DEFAULT_GROQ_MODEL
        super().__init__(
            api_key=api_key,
            model=resolved_model,
            api_url=(api_url or GROQ_API_URL).rstrip("/"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.fallback_models = [m for m in (fallback_models or []) if m and m != resolved_model]
        self.max_retries = max(0, max_retries)
        self.timeout_seconds = timeout_seconds
        self._transport = transport
        self._unavailable_models: set[str] = set()
        self._sleep: Callable[[float], None] = time.sleep

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError(
                "Groq API key is missing. Set llm.api_key in config.yaml or the GROQ_API_KEY environment variable."
            )
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _client(self) -> httpx.Client:
        if self._transport is not None:
            return httpx.Client(timeout=self.timeout_seconds, transport=self._transport)
        return httpx.Client(timeout=self.timeout_seconds)

    def _retry_wait(self, response: httpx.Response, attempt: int) -> float:
        headers = response.headers
        for header in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            parsed = _parse_duration_seconds(headers.get(header))
            if parsed is not None and parsed > 0:
                return parsed
        return float(min(2**attempt, 30))

    def _chat(self, model: str, messages: list[dict[str, str]], json_mode: bool) -> str:
        url = f"{self.api_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error = ""
        with self._client() as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.post(url, headers=self._headers(), json=payload)
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last_error = f"network error: {e}"
                    self._sleep(float(min(2**attempt, 20)))
                    continue

                if response.status_code == 200:
                    data = response.json()
                    self._add_usage(model, data.get("usage") or {})
                    return str(data["choices"][0]["message"].get("content") or "")

                body = response.text[:500]
                last_error = f"HTTP {response.status_code}: {body}"

                if response.status_code == 429 or response.status_code >= 500:
                    wait = self._retry_wait(response, attempt)
                    if wait > MAX_RATE_LIMIT_WAIT_SECONDS:
                        raise RateLimitExhaustedError(f"Groq model {model} rate limited for {wait:.0f}s: {body}")
                    logger.warning(f"Groq {response.status_code} on {model}; retrying in {wait:.1f}s")
                    self._sleep(wait)
                    continue

                lowered = body.lower()
                if response.status_code == 400 and json_mode and "json_validate_failed" in lowered:
                    # The model produced invalid JSON under JSON mode; retry without the constraint and parse leniently.
                    logger.info(f"Groq JSON mode validation failed on {model}; retrying without response_format.")
                    payload.pop("response_format", None)
                    json_mode = False
                    continue

                if response.status_code in (403, 404) or any(
                    marker in lowered for marker in ("model_not_found", "decommissioned", "does not exist", "model_terms")
                ):
                    raise ModelUnavailableError(f"Groq model {model} unavailable: {body}")

                raise RuntimeError(f"Groq API error for model {model}: {last_error}")

        raise RateLimitExhaustedError(f"Groq request failed after {self.max_retries + 1} attempts: {last_error}")

    def _chat_with_fallback(self, messages: list[dict[str, str]], json_mode: bool) -> str:
        candidates = [m for m in [self.model, *self.fallback_models] if m not in self._unavailable_models]
        if not candidates:
            candidates = [self.model]
        errors: list[str] = []
        for model in candidates:
            try:
                return self._chat(model, messages, json_mode)
            except ModelUnavailableError as e:
                self._unavailable_models.add(model)
                errors.append(str(e))
                logger.warning(f"{e}. Trying next fallback model.")
            except RateLimitExhaustedError as e:
                errors.append(str(e))
                logger.warning(f"{e}. Trying next fallback model.")
        raise RuntimeError("All Groq models failed: " + " | ".join(errors))

    def _reset_usage(self) -> None:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "model": self.model}

    def _add_usage(self, model: str, usage: dict[str, Any]) -> None:
        if self.last_usage is None:
            self._reset_usage()
        assert self.last_usage is not None
        self.last_usage["prompt_tokens"] = int(self.last_usage["prompt_tokens"]) + int(usage.get("prompt_tokens") or 0)
        self.last_usage["completion_tokens"] = int(self.last_usage["completion_tokens"]) + int(
            usage.get("completion_tokens") or 0
        )
        self.last_usage["model"] = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        self._reset_usage()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = self._chat_with_fallback(messages, json_mode=False)
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    def generate_json(self, prompt: str, schema: type[BaseModel], system_prompt: str | None = None) -> BaseModel:
        self._reset_usage()
        schema_json = json.dumps(schema.model_json_schema())
        user_prompt = (
            f"{prompt}\n\nReturn ONLY a JSON object that validates against this JSON schema:\n{schema_json}\n"
            "Use the exact field names from the schema. Output JSON only."
        )
        messages = [
            {"role": "system", "content": system_prompt or JSON_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = self._chat_with_fallback(messages, json_mode=True)
        try:
            return schema.model_validate_json(extract_json_block(raw))
        except (ValidationError, ValueError) as first_error:
            logger.info(f"Groq output failed {schema.__name__} validation; requesting a corrected JSON object.")
            repair_messages = [
                *messages,
                {"role": "assistant", "content": raw[:6000]},
                {
                    "role": "user",
                    "content": (
                        f"That output failed validation with this error:\n{str(first_error)[:1500]}\n"
                        "Return the corrected JSON object only."
                    ),
                },
            ]
            repaired = self._chat_with_fallback(repair_messages, json_mode=True)
            try:
                return schema.model_validate_json(extract_json_block(repaired))
            except (ValidationError, ValueError) as e:
                raise ValueError(f"JSON validation failed for schema {schema.__name__}: {e}") from e

    def list_models(self) -> list[str]:
        """Lists model IDs enabled for this Groq API key (used by `recruiting-platform doctor`)."""
        with self._client() as client:
            response = client.get(f"{self.api_url}/models", headers=self._headers())
            response.raise_for_status()
            return sorted(str(m.get("id")) for m in response.json().get("data", []))
