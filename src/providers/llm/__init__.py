from src.config import LLMConfig
from src.providers.llm.agy_cli import AGYCLIProvider
from src.providers.llm.anthropic import AnthropicProvider
from src.providers.llm.base import BaseLLMProvider
from src.providers.llm.gemini import GeminiProvider
from src.providers.llm.groq import GroqProvider
from src.providers.llm.local_agy import LocalAGYProvider
from src.providers.llm.openai import OpenAIProvider


def get_llm_provider(config: LLMConfig, fast: bool = False) -> BaseLLMProvider:
    """
    Factory function to retrieve the configured LLM provider.
    With fast=True, the provider is built with `fast_model` (if configured) for cheap extraction calls.
    """
    provider_name = config.provider.lower()
    model = config.fast_model if (fast and config.fast_model) else config.model
    api_key = config.resolved_api_key()

    if provider_name == "groq":
        return GroqProvider(
            api_key=api_key,
            model=model,
            api_url=config.api_url or None,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            fallback_models=config.fallback_models,
            max_retries=config.max_retries,
            timeout_seconds=config.timeout_seconds,
        )
    elif provider_name == "local_agy":
        return LocalAGYProvider(
            api_key=api_key,
            model=model,
            api_url=config.api_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    elif provider_name == "agy_cli":
        return AGYCLIProvider(
            api_key=api_key,
            model=model,
            api_url=config.api_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    elif provider_name == "openai":
        return OpenAIProvider(
            api_key=api_key,
            model=model,
            api_url=config.api_url or None,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    elif provider_name == "anthropic":
        return AnthropicProvider(
            api_key=api_key,
            model=model,
            api_url=config.api_url or None,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    elif provider_name == "gemini":
        return GeminiProvider(
            api_key=api_key,
            model=model,
            api_url=config.api_url or None,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")


__all__ = [
    "BaseLLMProvider",
    "LocalAGYProvider",
    "AGYCLIProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "GroqProvider",
    "get_llm_provider",
]
