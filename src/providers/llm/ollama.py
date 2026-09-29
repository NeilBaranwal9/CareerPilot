"""Local Ollama provider (native /api/chat with schema-constrained JSON output)."""

import json
import logging
import re
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from src.providers.llm.base import BaseLLMProvider
from src.providers.llm.groq import extract_json_block

logger = logging.getLogger("recruiting-platform.llm.ollama")

DEFAULT_OLLAMA_URL = "http://localhost:11434"

JSON_SYSTEM_PROMPT = (
    "You are a careful data extraction and classification assistant. Answer with one JSON object that matches the "
    "requested schema. Use only facts present in the provided text; use null or empty lists when unknown."
)


class OllamaUnavailableError(RuntimeError):
    """Ollama server is not reachable or the model is not installed."""


class OllamaProvider(BaseLLMProvider):
    """
    Local LLM via Ollama. Uses `format` (JSON schema) for structured output, disables "thinking" for speed,
    sets an explicit context window (Ollama's default is small and silently truncates long prompts) and reports
    real token counts (prompt_eval_count / eval_count).
    """

    provider_name = "ollama"

    def __init__(
        self,
        model: str = "qwen3:8b",
        api_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        num_ctx: int = 12288,
        timeout_seconds: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ):
        super().__init__(
            api_key="", model=model, api_url=(api_url or DEFAULT_OLLAMA_URL).rstrip("/"),
            temperature=temperature, max_tokens=max_tokens,
        )
        self.num_ctx = num_ctx
        self.timeout_seconds = timeout_seconds
        self._transport = transport
        self._supports_think_flag = True

    def _client(self, timeout: float | None = None) -> httpx.Client:
        kwargs: dict[str, Any] = {"timeout": timeout or self.timeout_seconds}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def health(self) -> tuple[bool, str]:
        """(ok, message): server reachable and model installed."""
        try:
            with self._client(timeout=3.0) as client:
                response = client.get(f"{self.api_url}/api/tags")
                response.raise_for_status()
                names = {m.get("name", "") for m in response.json().get("models", [])}
        except Exception as e:
            return False, f"Ollama not reachable at {self.api_url} ({e}). Start Ollama or set llm.local_provider: ''"
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names and self.model not in names:
            return False, f"Ollama model '{self.model}' is not installed. Run: ollama pull {self.model}"
        return True, f"Ollama ready with {self.model}"

    def _chat(self, messages: list[dict[str, str]], fmt: Any = None) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens, "num_ctx": self.num_ctx},
        }
        if fmt is not None:
            body["format"] = fmt
        if self._supports_think_flag:
            body["think"] = False
        try:
            with self._client() as client:
                response = client.post(f"{self.api_url}/api/chat", json=body)
                if response.status_code == 400 and "think" in response.text.lower() and self._supports_think_flag:
                    self._supports_think_flag = False
                    body.pop("think", None)
                    response = client.post(f"{self.api_url}/api/chat", json=body)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise OllamaUnavailableError(f"Ollama not reachable at {self.api_url}: {e}") from e
        if response.status_code == 404:
            raise OllamaUnavailableError(f"Ollama model '{self.model}' not found. Run: ollama pull {self.model}")
        if response.status_code >= 400:
            raise RuntimeError(f"Ollama error {response.status_code}: {response.text[:300]}")
        data = response.json()
        usage_prompt = int(data.get("prompt_eval_count") or 0)
        usage_completion = int(data.get("eval_count") or 0)
        previous = self.last_usage or {"prompt_tokens": 0, "completion_tokens": 0}
        self.last_usage = {
            "prompt_tokens": int(previous.get("prompt_tokens", 0)) + usage_prompt,
            "completion_tokens": int(previous.get("completion_tokens", 0)) + usage_completion,
            "model": self.model,
        }
        content = str((data.get("message") or {}).get("content") or "")
        return re.sub(r"<think>[\s\S]*?</think>", "", content).strip()

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        self.last_usage = None
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": "user", "content": prompt}
        ]
        return self._chat(messages)

    def generate_json(self, prompt: str, schema: type[BaseModel], system_prompt: str | None = None) -> BaseModel:
        self.last_usage = None
        schema_def = schema.model_json_schema()
        messages = [
            {"role": "system", "content": system_prompt or JSON_SYSTEM_PROMPT},
            {"role": "user", "content": f"{prompt}\n\nRespond with JSON matching this schema:\n{json.dumps(schema_def)}"},
        ]
        raw = self._chat(messages, fmt=schema_def)
        try:
            return schema.model_validate_json(extract_json_block(raw))
        except (ValidationError, ValueError) as first_error:
            repair = [
                *messages,
                {"role": "assistant", "content": raw[:4000]},
                {"role": "user", "content": f"That failed validation: {str(first_error)[:800]}. Return corrected JSON only."},
            ]
            fixed = self._chat(repair, fmt=schema_def)
            try:
                return schema.model_validate_json(extract_json_block(fixed))
            except (ValidationError, ValueError) as e:
                raise ValueError(f"JSON validation failed for schema {schema.__name__}: {e}") from e
