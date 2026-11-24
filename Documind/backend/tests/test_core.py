"""Comprehensive Pytest test suite for DocuMind AI Backend.

Verifies:
- All 10 REST API endpoints using FastAPI TestClient
- Real in-memory PDF and DOCX parsing with PyMuPDF and python-docx
- Grounded RAG with strict citations and missing-info refusal
- Deterministic structured outputs with Pydantic validation
- FAISS cosine similarity vector search and index rebuilding
"""

import io
from typing import Type, TypeVar

try:
    import fitz  # PyMuPDF
except (ImportError, Exception):
    fitz = None

try:
    import pypdf
except (ImportError, Exception):
    pypdf = None

from docx import Document as DocxDocument
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import BaseModel
import pytest

from app.ai.llm_provider import BaseLLMProvider
from app.api.routes import get_llm_provider
from app.core.session_store import session_store
from app.main import app
from app.models.schemas import (
    ChatResponse,
    ClassificationResult,
    ClassifyResponse,
    ComparisonReport,
    DeleteDocumentResponse,
    DocumentsListResponse,
    ExtractResponse,
    HealthResponse,
    InvoiceLineItem,
    InvoiceSchema,
    ResearchPaperSchema,
    ResumeSchema,
    ContractSchema,
    SearchResponse,
    SummarizeResponse,
)
from app.rag.chunker import TextChunker
from app.rag.text_cleaner import clean_text
from app.rag.vector_store import VectorStore

T = TypeVar("T", bound=BaseModel)


# ============================================================================
# DETERMINISTIC MOCK LLM PROVIDER
# ============================================================================


class MockLLMProvider(BaseLLMProvider):
    """Deterministic in-memory LLM and embedding provider for automated test suite."""

    def __init__(self, embedding_dim: int = 16) -> None:
        self.embedding_dim = embedding_dim

    def is_ready(self) -> bool:
        return True

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generates deterministic pseudo-embeddings based on text length and character hash."""
        embeddings: list[list[float]] = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 50
            vec = [float((seed + i * 5) % 19) + 0.1 for i in range(self.embedding_dim)]
            embeddings.append(vec)
        return embeddings

    def generate_text(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.2,
    ) -> str:
        """Generates deterministic responses conditioned on prompt keywords."""
        if "missing_information_test_query" in prompt:
            return "I cannot find this information in the uploaded documents."

        if "Mode: Quick" in system_instruction:
            return "Quick summary: This document discusses advanced AI architectures and retrieval-augmented generation."

        if "Mode: Bullet Points" in system_instruction:
            return "- Point 1: Neural network scaling\n- Point 2: Grounded document retrieval\n- Point 3: Evaluation metrics"

        if "Executive" in system_instruction or "Executive" in prompt:
            return (
                "### Executive Summary\n"
                "**Purpose:** Establish foundational document intelligence standards.\n"
                "**Findings:** Dense vector search combined with structured outputs yields 99% extraction accuracy.\n"
                "**Recommendations:** Deploy automated verification pipelines across enterprise workflows."
            )

        if "User Question" in prompt:
            return (
                "Based on the provided research paper, transformers utilize multi-head self-attention "
                "to capture contextual relationships across input sequences [Source: test_paper.pdf, Page: 1]."
            )

        return "Synthesized analytical summary of the document."

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: str = "",
        temperature: float = 0.1,
    ) -> T:
        """Returns typed Pydantic instances matching the requested schema."""
        if response_schema is ClassificationResult:
            return ClassificationResult(
                category="Research Paper",
                certainty_level="High",
                explanation="The document features abstract sections, academic methodology, and empirical benchmarks.",
                keywords=["neural networks", "attention", "transformer", "embeddings"],
            )  # type: ignore

        if response_schema is InvoiceSchema:
            return InvoiceSchema(
                vendor_name="Acme Artificial Intelligence Inc.",
                invoice_number="INV-2026-9042",
                invoice_date="2026-03-15",
                due_date="2026-04-15",
                total_amount=2450.75,
                currency="USD",
                line_items=[
                    InvoiceLineItem(
                        description="GPU Cluster Compute Reservation",
                        quantity=10.0,
                        unit_price=245.075,
                        total_price=2450.75,
                    )
                ],
                payment_terms="Net 30",
            )  # type: ignore

        if response_schema is ResumeSchema:
            return ResumeSchema(
                full_name="Alex Mercer",
                email="alex.mercer@example.com",
                phone="+1-555-0199",
                summary="Senior Generative AI Architect specializing in large language model pipelines.",
                skills=["Python", "FastAPI", "PyTorch", "FAISS", "Gemini API"],
            )  # type: ignore

        if response_schema is ResearchPaperSchema:
            return ResearchPaperSchema(
                title="Foundations of Generative Document Intelligence",
                authors=["Dr. Evelyn Reed", "Marcus Vance"],
                abstract="We introduce an in-memory retrieval-augmented architecture for real-time document analysis.",
                methodology="Dense vector embedding indexing with FAISS and constrained grammar decoding.",
                key_findings=[
                    "Cosine similarity on L2-normalized embeddings achieves sub-millisecond retrieval.",
                    "Strict grounding eliminates speculative hallucinations.",
                ],
                conclusions="The architecture scales efficiently across enterprise document formats.",
                references_count=34,
            )  # type: ignore

        if response_schema is ContractSchema:
            return ContractSchema(
                contract_title="Master Software Services Agreement",
                parties_involved=["Alpha Cloud LLC", "Omega Innovations Corp"],
                effective_date="2026-01-01",
                expiration_date="2027-01-01",
                key_obligations=[
                    "Deliver high-availability backend API services",
                    "Maintain 99.9% uptime SLA",
                ],
                termination_clauses="30 days written notice for breach of covenants.",
                governing_law="State of Delaware",
            )  # type: ignore

        if response_schema is ComparisonReport:
            return ComparisonReport(
                similarities=[
                    "Both documents focus on modern AI architectures and document intelligence.",
                    "Both highlight vector search and retrieval-augmented generation as core pillars.",
                ],
                differences=[
                    "Document 1 emphasizes theoretical foundations and academic citations.",
                    "Document 2 emphasizes enterprise production deployment and API endpoints.",
                ],
                conflicting_information=[
                    "Document 1 benchmarks chunk latency at 12ms whereas Document 2 reports 4ms under identical hardware.",
                ],
                common_themes=[
                    "Generative AI",
                    "Dense Vector Search",
                    "Document Ingestion",
                ],
                summary_verdict="The two documents provide complementary theoretical and operational specifications.",
            )  # type: ignore

        # Fallback default instantiation if generic
        return response_schema.model_validate({})


# ============================================================================
# TEST FIXTURES & HELPERS
# ============================================================================


@pytest.fixture(scope="module")
def mock_provider() -> MockLLMProvider:
    return MockLLMProvider(embedding_dim=16)


@pytest.fixture(scope="module")
def client(mock_provider: MockLLMProvider) -> TestClient:
    app.dependency_overrides[get_llm_provider] = lambda: mock_provider
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def create_in_memory_pdf_bytes(title: str, body: str) -> bytes:
    """Generates a valid in-memory PDF using PyMuPDF if available, or pure Python pypdf."""
    if fitz is not None:
        try:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 72), f"{title}\n\n{body}", fontsize=11)
            pdf_bytes = doc.write()
            doc.close()
            return pdf_bytes
        except Exception:
            pass

    # Pure Python valid PDF format compatible with pypdf and all PDF parsers
    text_content = f"{title} {body}".replace("(", "\\(").replace(")", "\\)")
    stream = f"BT\n/F1 12 Tf\n72 712 Td\n({text_content}) Tj\nET".encode("latin-1")
    pdf = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length {len(stream)} >>
stream
{stream.decode('latin-1')}
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000010 00000 n 
0000000060 00000 n 
0000000117 00000 n 
0000000247 00000 n 
0000000340 00000 n 
trailer << /Root 1 0 R /Size 6 >>
startxref
427
%%EOF""".encode("latin-1")
    return pdf


def create_in_memory_docx_bytes(title: str, body: str) -> bytes:
    """Generates a valid in-memory DOCX using python-docx."""
    doc = DocxDocument()
    doc.add_heading(title, level=1)
    doc.add_paragraph(body)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ============================================================================
# UNIT TESTS: TEXT CLEANER, CHUNKER, VECTOR STORE
# ============================================================================


def test_text_cleaner_unit() -> None:
    raw = "This is a demon-\nstration of hyphenated    words and \r\n excessive \n\n\n\n breaks."
    cleaned = clean_text(raw)
    assert "demonstration" in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned


def test_chunker_overlapping_boundaries() -> None:
    chunker = TextChunker(chunk_size=100, chunk_overlap=30)
    from app.models.schemas import DocumentPage

    page = DocumentPage(
        page_number=1,
        text="The quick brown fox jumps over the lazy dog. " * 5,
    )
    chunks = chunker.chunk_page(
        doc_id="test_doc", filename="fox.txt", page=page
    )
    assert len(chunks) > 1
    assert chunks[0].page_number == 1
    assert chunks[0].doc_id == "test_doc"


def test_vector_store_cosine_similarity() -> None:
    store = VectorStore()
    from app.models.schemas import DocumentChunk

    chunks = [
        DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            filename="a.txt",
            page_number=1,
            chunk_index=0,
            text="Vector search in deep learning",
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="d1",
            filename="a.txt",
            page_number=1,
            chunk_index=1,
            text="Cooking recipes for dinner",
        ),
    ]
    # Synthetic vectors
    emb = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    store.add_chunks(chunks=chunks, embeddings=emb)
    assert store.total_chunks() == 2

    # Query matching c1
    results = store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"
    assert abs(results[0][1] - 1.0) < 1e-4  # Cosine similarity equals 1.0


# ============================================================================
# API INTEGRATION TESTS: ALL 10 REST ENDPOINTS
# ============================================================================

TEST_SESSION = "pytest_integration_session"


def test_01_health_endpoint(client: TestClient) -> None:
    """ENDPOINT 1: GET /api/health"""
    response = client.get("/api/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    validated = HealthResponse.model_validate(data)
    assert validated.status == "healthy"
    assert validated.genai_ready is True
    assert validated.active_model != ""


def test_02_upload_documents_endpoint(client: TestClient) -> None:
    """ENDPOINT 2: POST /api/documents/upload with real in-memory PDF & DOCX."""
    pdf_bytes = create_in_memory_pdf_bytes(
        title="Foundations of Generative Document Intelligence",
        body=(
            "Retrieval-Augmented Generation (RAG) grounds language models with authoritative context. "
            "Transformers utilize multi-head self-attention mechanisms to map dependencies. "
            "FAISS provides efficient dense vector similarity search."
        ),
    )
    docx_bytes = create_in_memory_docx_bytes(
        title="Master Enterprise Cloud Agreement",
        body=(
            "This Software Services Agreement is entered into between Alpha Cloud LLC and Omega Innovations. "
            "The vendor agrees to provide 99.9% uptime SLA across all generative AI endpoints."
        ),
    )

    files = [
        ("files", ("test_paper.pdf", pdf_bytes, "application/pdf")),
        ("files", ("test_contract.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
    ]

    response = client.post(
        f"/api/documents/upload?session_id={TEST_SESSION}",
        files=files,
    )
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

    # Verify metadata fields
    for doc in data:
        assert "doc_id" in doc
        assert doc["chunk_count"] > 0
        assert doc["page_count"] > 0
        assert doc["preview_snippet"] != ""


def test_03_list_documents_endpoint(client: TestClient) -> None:
    """ENDPOINT 3: GET /api/documents"""
    response = client.get(f"/api/documents?session_id={TEST_SESSION}")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    validated = DocumentsListResponse.model_validate(data)
    assert validated.session_id == TEST_SESSION
    assert len(validated.documents) == 2
    assert validated.stats.total_documents == 2
    assert validated.stats.total_chunks >= 2
    assert validated.stats.total_pages >= 2


def test_04_classify_endpoint(client: TestClient) -> None:
    """ENDPOINT 5: POST /api/classify"""
    # Get doc_id of uploaded document
    docs_resp = client.get(f"/api/documents?session_id={TEST_SESSION}").json()
    doc_id = docs_resp["documents"][0]["doc_id"]

    response = client.post(
        f"/api/classify?session_id={TEST_SESSION}",
        json={"doc_id": doc_id},
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    validated = ClassifyResponse.model_validate(data)
    assert validated.doc_id == doc_id
    assert validated.classification.category == "Research Paper"
    assert validated.classification.certainty_level in ("High", "Medium", "Low")
    assert len(validated.classification.keywords) > 0


def test_05_summarize_endpoint(client: TestClient) -> None:
    """ENDPOINT 6: POST /api/summarize"""
    docs_resp = client.get(f"/api/documents?session_id={TEST_SESSION}").json()
    doc_id = docs_resp["documents"][0]["doc_id"]

    for mode in ["Quick", "Bullet Points", "Executive"]:
        response = client.post(
            f"/api/summarize?session_id={TEST_SESSION}",
            json={"doc_id": doc_id, "mode": mode},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        validated = SummarizeResponse.model_validate(data)
        assert validated.doc_id == doc_id
        assert validated.mode == mode
        assert len(validated.summary) > 0


def test_06_chat_rag_endpoint(client: TestClient) -> None:
    """ENDPOINT 7: POST /api/chat with citations and refusal check."""
    # 1. Successful grounded question
    response = client.post(
        f"/api/chat?session_id={TEST_SESSION}",
        json={
            "question": "How do transformers handle contextual dependencies?",
            "top_k": 3,
        },
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    validated = ChatResponse.model_validate(data)
    assert len(validated.answer) > 0
    assert len(validated.citations) > 0
    assert validated.genai_pipeline_used.chunks_retrieved > 0

    # 2. Refusal check for missing info
    refusal_resp = client.post(
        f"/api/chat?session_id={TEST_SESSION}",
        json={
            "question": "What is the missing_information_test_query regarding Martian agriculture?",
            "top_k": 3,
        },
    )
    assert refusal_resp.status_code == status.HTTP_200_OK
    refusal_data = refusal_resp.json()
    refusal_validated = ChatResponse.model_validate(refusal_data)
    assert "I cannot find this information in the uploaded documents." in refusal_validated.answer


def test_07_extract_endpoint(client: TestClient) -> None:
    """ENDPOINT 8: POST /api/extract"""
    docs_resp = client.get(f"/api/documents?session_id={TEST_SESSION}").json()
    doc_id = docs_resp["documents"][0]["doc_id"]

    response = client.post(
        f"/api/extract?session_id={TEST_SESSION}",
        json={"doc_id": doc_id, "schema_type": "Research Paper"},
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    validated = ExtractResponse.model_validate(data)
    assert validated.doc_id == doc_id
    assert validated.schema_used == "Research Paper"
    assert "title" in validated.extracted_data
    assert "authors" in validated.extracted_data


def test_08_search_endpoint(client: TestClient) -> None:
    """ENDPOINT 9: POST /api/search"""
    response = client.post(
        f"/api/search?session_id={TEST_SESSION}",
        json={"query": "dense vector search and attention", "top_k": 3},
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    validated = SearchResponse.model_validate(data)
    assert validated.total_results > 0
    assert len(validated.results) > 0
    assert validated.results[0].cosine_similarity_score is not None
    assert validated.results[0].filename != ""


def test_09_compare_endpoint(client: TestClient) -> None:
    """ENDPOINT 10: POST /api/compare"""
    docs_resp = client.get(f"/api/documents?session_id={TEST_SESSION}").json()
    doc_ids = [d["doc_id"] for d in docs_resp["documents"][:2]]

    response = client.post(
        f"/api/compare?session_id={TEST_SESSION}",
        json={"doc_ids": doc_ids},
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "comparison_report" in data
    report = data["comparison_report"]
    assert len(report["similarities"]) > 0
    assert len(report["differences"]) > 0
    assert len(report["common_themes"]) > 0
    assert report["summary_verdict"] != ""


def test_10_delete_document_endpoint(client: TestClient) -> None:
    """ENDPOINT 4: DELETE /api/documents/{doc_id}"""
    docs_resp = client.get(f"/api/documents?session_id={TEST_SESSION}").json()
    doc_to_delete = docs_resp["documents"][0]["doc_id"]

    del_resp = client.delete(
        f"/api/documents/{doc_to_delete}?session_id={TEST_SESSION}"
    )
    assert del_resp.status_code == status.HTTP_200_OK
    del_data = del_resp.json()
    validated = DeleteDocumentResponse.model_validate(del_data)
    assert validated.doc_id == doc_to_delete
    assert validated.remaining_documents == 1

    # Verify session now only has 1 document remaining
    remaining_docs = client.get(f"/api/documents?session_id={TEST_SESSION}").json()
    assert len(remaining_docs["documents"]) == 1
    assert remaining_docs["documents"][0]["doc_id"] != doc_to_delete
