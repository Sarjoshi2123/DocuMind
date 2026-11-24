"""FastAPI Main Application for DocuMind AI.

Configures CORS, lifespan events, global exception handling, and API routes.
"""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.ai.gemini_provider import gemini_provider
from app.api.routes import router as api_router
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("documind")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler verifying GenAI pipeline readiness on startup."""
    logger.info("Initializing DocuMind AI Backend...")
    logger.info("Configured Gemini Model: %s", settings.GEMINI_MODEL)
    logger.info("Configured Embedding Model: %s", settings.GEMINI_EMBEDDING_MODEL)

    if gemini_provider.is_ready():
        logger.info("Google GenAI SDK client successfully configured and ready.")
    else:
        logger.warning(
            "GEMINI_API_KEY is not configured or uses placeholder value. "
            "Please provide a valid key in backend/.env for production GenAI execution."
        )

    yield

    logger.info("DocuMind AI Backend shutdown complete.")


app = FastAPI(
    title="DocuMind AI Backend",
    description=(
        "Enterprise Generative AI Document Intelligence Platform powered by Google GenAI. "
        "Provides Classification, Hierarchical Summarization, Grounded RAG Q&A, "
        "Deterministic Structured Extraction, and Semantic Vector Search."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Standardized HTTP exception response format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all unhandled exception handler preserving API stability."""
    logger.exception("Unhandled server exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": f"Internal Server Error: {str(exc)}",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        },
    )


# Mount API Router under /api
app.include_router(api_router, prefix="/api")


@app.get("/", summary="Root index")
def root_index() -> dict[str, str]:
    """Root health and welcome message."""
    return {
        "app": "DocuMind AI Backend",
        "status": "online",
        "docs": "/docs",
        "api": "/api",
    }
