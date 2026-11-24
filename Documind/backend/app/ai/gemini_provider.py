"""Concrete Gemini LLM and Embedding Provider using official google-genai SDK.

Implements text generation, structured JSON generation with Pydantic validation,
and dense text embeddings using Google Gemini 3.8 and Gemini Embedding models.
"""

import logging
from typing import Type, TypeVar
from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from app.ai.llm_provider import BaseLLMProvider
from app.core.config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini GenAI Provider implementing BaseLLMProvider."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key: str = (
            api_key if api_key is not None else settings.GEMINI_API_KEY
        )
        self.model_name: str = settings.GEMINI_MODEL
        self.embedding_model_name: str = settings.GEMINI_EMBEDDING_MODEL
        self._client: genai.Client | None = None

        if self.api_key and self.api_key != "your_api_key_here":
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error("Failed to initialize Google GenAI Client: %s", e)
                self._client = None

    def _ensure_client(self) -> genai.Client:
        """Validates that the Gemini client is initialized."""
        if not self._client:
            raise RuntimeError(
                "Gemini API Client is not configured. "
                "Please configure a valid GEMINI_API_KEY in backend/.env."
            )
        return self._client

    def is_ready(self) -> bool:
        """Returns True if the client is configured with an active key."""
        return (
            self._client is not None
            and bool(self.api_key)
            and self.api_key != "your_api_key_here"
        )

    def generate_text(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.2,
    ) -> str:
        """Generates unconstrained text using Gemini.

        Args:
            prompt: User or task prompt.
            system_instruction: Grounding and role instructions for system prompt.
            temperature: Sampling temperature (0.0 - 1.0).

        Returns:
            The generated string response.
        """
        client = self._ensure_client()
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction or None,
            )
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except errors.APIError as exc:
            logger.error("Gemini API Error during generate_text: %s", exc)
            raise RuntimeError(f"Gemini API Error: {exc.message}") from exc
        except Exception as exc:
            logger.error("Unexpected error during generate_text: %s", exc)
            raise RuntimeError(f"Text generation failed: {str(exc)}") from exc

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: str = "",
        temperature: float = 0.1,
    ) -> T:
        """Generates structured JSON adhering strictly to a Pydantic response_schema.

        Uses Gemini's native structured outputs engine, enforcing zero hallucination
        of schema keys and returning a validated Pydantic model instance.
        """
        client = self._ensure_client()
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction or None,
                response_mime_type="application/json",
                response_schema=response_schema,
            )
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            if not response.text:
                raise ValueError("Gemini returned empty response for structured generation")
            return response_schema.model_validate_json(response.text)
        except errors.APIError as exc:
            logger.error("Gemini API Error during generate_structured: %s", exc)
            raise RuntimeError(f"Gemini API Error: {exc.message}") from exc
        except Exception as exc:
            logger.error("Error during structured generation: %s", exc)
            raise RuntimeError(f"Structured extraction failed: {str(exc)}") from exc

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generates dense vector embeddings for input texts using Gemini Embedding model.

        Batches requests according to EMBEDDING_BATCH_SIZE to respect payload constraints.
        """
        if not texts:
            return []

        client = self._ensure_client()
        embeddings: list[list[float]] = []
        batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                res = client.models.embed_content(
                    model=self.embedding_model_name,
                    contents=batch,
                )
                if not res.embeddings:
                    raise RuntimeError("No embeddings returned by Gemini Embedding API")
                batch_vectors = [list(e.values) for e in res.embeddings]
                embeddings.extend(batch_vectors)
            except errors.APIError as exc:
                logger.error("Gemini API Error during get_embeddings: %s", exc)
                raise RuntimeError(f"Gemini Embedding API Error: {exc.message}") from exc
            except Exception as exc:
                logger.error("Error generating embeddings: %s", exc)
                raise RuntimeError(f"Embedding generation failed: {str(exc)}") from exc

        return embeddings


# Default provider instance
gemini_provider = GeminiProvider()
