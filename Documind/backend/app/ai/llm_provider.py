"""Abstract Base Class for LLM and Embedding Providers.

Defines the contract for text generation, structured JSON generation,
and vector embedding so alternative providers (OpenAI, Anthropic, or local LLMs)
can be seamlessly swapped without altering business logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseLLMProvider(ABC):
    """Abstract interface for Generative AI and Embedding interactions."""

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.2,
    ) -> str:
        """Generates standard unconstrained text response given a prompt and system instruction."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: str = "",
        temperature: float = 0.1,
    ) -> T:
        """Generates validated structured data adhering to a specific Pydantic schema."""
        pass

    @abstractmethod
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Computes dense vector representations for a batch of text strings."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Checks if the provider is initialized and credentials are configured."""
        pass
