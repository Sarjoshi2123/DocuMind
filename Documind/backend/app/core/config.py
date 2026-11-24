"""Application configuration and environment variables."""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Locate .env file in backend/ or root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseModel):
    """Global configuration settings for DocuMind AI Backend."""

    # Gemini GenAI configuration
    GEMINI_API_KEY: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    GEMINI_MODEL: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    )
    GEMINI_EMBEDDING_MODEL: str = Field(
        default_factory=lambda: os.getenv(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
        )
    )

    # Document upload limits and allowed formats
    MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    ALLOWED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt"}

    # Text chunking defaults
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Long document threshold (characters) for hierarchical map-reduce summarization
    LONG_DOCUMENT_CHAR_THRESHOLD: int = 14000

    # Batch size for embedding calls to avoid payload limits
    EMBEDDING_BATCH_SIZE: int = 32


settings = Settings()
