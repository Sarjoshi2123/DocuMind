"""FastAPI REST API endpoints for DocuMind AI."""

from typing import Annotated, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, UploadFile, status
from app.ai.gemini_provider import gemini_provider
from app.ai.llm_provider import BaseLLMProvider
from app.core.config import settings
from app.core.session_store import session_store
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ClassificationResult,
    ClassifyRequest,
    ClassifyResponse,
    CompareRequest,
    CompareResponse,
    DeleteDocumentResponse,
    DocumentMetadata,
    DocumentsListResponse,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.services.ai_services import ai_service
from app.services.document_service import document_service

router = APIRouter()


def get_session_id(
    session_id: Optional[str] = Query(
        default=None,
        description="Session identifier query parameter",
    ),
    x_session_id: Optional[str] = Header(
        default=None,
        alias="X-Session-ID",
        description="Session identifier HTTP header",
    ),
) -> str:
    """Extracts session_id from query parameter or header, defaulting to 'default_session'."""
    if session_id:
        return session_id
    if x_session_id:
        return x_session_id
    return "default_session"


def get_llm_provider() -> BaseLLMProvider:
    """Dependency provider for LLM service, easily swappable in tests."""
    return gemini_provider


# -----------------------------------------------------------------------------
# 1. HEALTH CHECK
# -----------------------------------------------------------------------------
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check and GenAI readiness status",
)
def health_check(
    provider: Annotated[BaseLLMProvider, Depends(get_llm_provider)],
) -> HealthResponse:
    """Returns the operational status of the backend, active models, and pipeline readiness."""
    is_ready = provider.is_ready()
    return HealthResponse(
        status="healthy",
        genai_ready=is_ready,
        active_model=settings.GEMINI_MODEL,
        active_embedding_model=settings.GEMINI_EMBEDDING_MODEL,
        version="1.0.0",
    )


# -----------------------------------------------------------------------------
# 2. DOCUMENT UPLOAD & INGESTION
# -----------------------------------------------------------------------------
@router.post(
    "/documents/upload",
    response_model=list[DocumentMetadata],
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest documents into active session FAISS index",
)
async def upload_documents(
    files: list[UploadFile],
    session_id: Annotated[str, Depends(get_session_id)],
    provider: Annotated[BaseLLMProvider, Depends(get_llm_provider)],
) -> list[DocumentMetadata]:
    """Uploads one or more documents (PDF, DOCX, TXT).

    Parses page text, normalizes content, chunks with sliding window overlap,
    generates dense Gemini vector embeddings, and registers them in the session index.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files were provided for upload.",
        )

    uploaded_docs: list[DocumentMetadata] = []
    for file in files:
        meta = await document_service.process_and_ingest_file(
            file=file,
            session_id=session_id,
            provider=provider,
        )
        uploaded_docs.append(meta)

    return uploaded_docs


# -----------------------------------------------------------------------------
# 3. LIST DOCUMENTS
# -----------------------------------------------------------------------------
@router.get(
    "/documents",
    response_model=DocumentsListResponse,
    summary="List all documents and statistics for the active session",
)
def list_documents(
    session_id: Annotated[str, Depends(get_session_id)],
) -> DocumentsListResponse:
    """Returns all uploaded document metadata and aggregate page/chunk statistics

    for the designated session.
    """
    docs = session_store.list_documents(session_id)
    stats = session_store.get_session_stats(session_id)
    return DocumentsListResponse(
        session_id=session_id,
        documents=docs,
        stats=stats,
    )


# -----------------------------------------------------------------------------
# 4. DELETE DOCUMENT
# -----------------------------------------------------------------------------
@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document and rebuild the session vector index",
)
def delete_document(
    doc_id: str,
    session_id: Annotated[str, Depends(get_session_id)],
) -> DeleteDocumentResponse:
    """Deletes a document from the session store and cleanly rebuilds the session's

    FAISS vector index without orphan vectors or dangling citations.
    """
    success, rem_docs, rem_chunks = session_store.delete_document(
        session_id=session_id, doc_id=doc_id
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found in session '{session_id}'.",
        )

    return DeleteDocumentResponse(
        doc_id=doc_id,
        message=f"Document '{doc_id}' successfully deleted and vector index rebuilt.",
        remaining_documents=rem_docs,
        remaining_chunks=rem_chunks,
    )


# -----------------------------------------------------------------------------
# 5. DOCUMENT CLASSIFICATION
# -----------------------------------------------------------------------------
@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify document into 8 canonical categories using structured JSON",
)
def classify_document(
    request: ClassifyRequest,
    session_id: Annotated[str, Depends(get_session_id)],
    provider: Annotated[BaseLLMProvider, Depends(get_llm_provider)],
) -> ClassifyResponse:
    """Classifies a document into one of 8 categories using Gemini structured outputs."""
    result: ClassificationResult = ai_service.classify_document(
        session_id=session_id,
        doc_id=request.doc_id,
        provider=provider,
    )
    return ClassifyResponse(
        doc_id=request.doc_id,
        classification=result,
    )


# -----------------------------------------------------------------------------
# 6. SUMMARIZATION
# -----------------------------------------------------------------------------
@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize document via Direct or Hierarchical Map-Reduce strategy",
)
def summarize_document(
    request: SummarizeRequest,
    session_id: Annotated[str, Depends(get_session_id)],
    provider: Annotated[BaseLLMProvider, Depends(get_llm_provider)],
) -> SummarizeResponse:
    """Summarizes a document using either direct LLM generation (<=14k chars) or

    hierarchical map-reduce summarization (>14k chars) to guarantee no truncation.
    """
    summary_text, strategy = ai_service.summarize_document(
        session_id=session_id,
        doc_id=request.doc_id,
        mode=request.mode,
        provider=provider,
    )
    return SummarizeResponse(
        doc_id=request.doc_id,
        mode=request.mode,
        strategy_used=strategy,
        summary=summary_text,
    )


# -----------------------------------------------------------------------------
# 7. GROUNDED RAG CHAT
# -----------------------------------------------------------------------------
@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Grounded conversational RAG Q&A with strict citations",
)
def chat_document(
    request: ChatRequest,
    session_id: Annotated[str, Depends(get_session_id)],
    provider: Annotated[BaseLLMProvider, Depends(get_llm_provider)],
) -> ChatResponse:
    """Queries session vector store using FAISS cosine similarity and generates

    a strictly grounded answer with source citations [Source: filename, Page: X].
    """
    return ai_service.chat_rag(
        session_id=session_id,
        question=request.question,
        doc_ids=request.doc_ids,
        top_k=request.top_k,
        provider=provider,
    )


# -----------------------------------------------------------------------------
# 8. STRUCTURED EXTRACTION
# -----------------------------------------------------------------------------
@router.post(
    "/extract",
    response_model=ExtractResponse,
    summary="Extract structured Pydantic JSON with zero hallucinated keys",
)
def extract_document(
    request: ExtractRequest,
    session_id: Annotated[str, Depends(get_session_id)],
    provider: Annotated[BaseLLMProvider, Depends(get_llm_provider)],
) -> ExtractResponse:
    """Extracts strongly-typed JSON adhering to Invoice, Resume, Research Paper,

    or Contract schemas (or auto-detects from classification).
    """
    schema_used, data = ai_service.extract_structured(
        session_id=session_id,
        doc_id=request.doc_id,
        schema_type=request.schema_type,
        provider=provider,
    )
    return ExtractResponse(
        doc_id=request.doc_id,
        schema_used=schema_used,
        extracted_data=data,
    )


# -----------------------------------------------------------------------------
# 9. SEMANTIC VECTOR SEARCH
# -----------------------------------------------------------------------------
@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Pure mathematical FAISS semantic vector search",
)
def search_documents(
    request: SearchRequest,
    session_id: Annotated[str, Depends(get_session_id)],
    provider: Annotated[BaseLLMProvider, Depends(get_llm_provider)],
) -> SearchResponse:
    """Embeds the search query and performs fast cosine similarity retrieval

    against the session's FAISS index.
    """
    results = ai_service.semantic_search(
        session_id=session_id,
        query=request.query,
        top_k=request.top_k,
        doc_ids=request.doc_ids,
        provider=provider,
    )
    return SearchResponse(
        query=request.query,
        total_results=len(results),
        results=results,
    )


# -----------------------------------------------------------------------------
# 10. MULTI-DOCUMENT COMPARATIVE SYNTHESIS
# -----------------------------------------------------------------------------
@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Comparative multi-document synthesis and contradiction detection",
)
def compare_documents(
    request: CompareRequest,
    session_id: Annotated[str, Depends(get_session_id)],
    provider: Annotated[BaseLLMProvider, Depends(get_llm_provider)],
) -> CompareResponse:
    """Cross-analyzes two or more documents, highlighting similarities, differences,

    conflicts, common themes, and an executive verdict.
    """
    report = ai_service.compare_documents(
        session_id=session_id,
        doc_ids=request.doc_ids,
        provider=provider,
    )
    return CompareResponse(
        doc_ids=request.doc_ids,
        comparison_report=report,
    )
