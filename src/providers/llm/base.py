from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseLLMProvider(ABC):
    """
    Abstract Base Class for LLM providers.
    Ensures a consistent interface across Groq, Ollama, OpenAI, Anthropic, Gemini, and Local AGY.
    """

    provider_name = "llm"

    def __init__(
        self,
        api_key: str,
        model: str,
        api_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ):
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Token usage of the most recent generate_* call: {"prompt_tokens", "completion_tokens", "model"}.
        # Providers that cannot report usage leave it as None and the router estimates it.
        self.last_usage: dict[str, int | str] | None = None

    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Sends a standard text generation request to the LLM.
        """
        pass

    @abstractmethod
    def generate_json(self, prompt: str, schema: type[BaseModel], system_prompt: str | None = None) -> BaseModel:
        """
        Sends a request to the LLM and parses the response into the specified Pydantic schema.
        """
        pass
