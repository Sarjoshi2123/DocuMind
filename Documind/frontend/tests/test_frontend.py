"""Frontend verification test suite.

Tests:
1. Clean imports of all views and components
2. SVG diagram generation and Verso HTML helpers
3. API Client endpoint wrappers and error handling with mocked responses
"""

from unittest.mock import MagicMock, patch
import pytest
from components.diagrams import (
    get_fig01_vector_cabinet_svg,
    get_fig02_embedding_conveyor_svg,
    render_empty_state,
    render_hero_header,
    render_metric_card,
)
from services.api_client import DocuMindAPIError, DocuMindClient


# ============================================================================
# 1. TEST ISOMETRIC SVG & HTML HELPERS
# ============================================================================


def test_fig01_vector_cabinet_svg() -> None:
    svg_html = get_fig01_vector_cabinet_svg(doc_count=3, chunk_count=18, status_label="ONLINE")
    assert "<svg" in svg_html
    assert "FIG 01" in svg_html
    assert "3 FILES" in svg_html
    assert "18 VECTORS" in svg_html
    assert "ONLINE" in svg_html


def test_fig02_embedding_conveyor_svg() -> None:
    svg_html = get_fig02_embedding_conveyor_svg(
        model_name="gemini-embedding-001", dimension=3072, similarity_metric="COSINE"
    )
    assert "<svg" in svg_html
    assert "FIG 02" in svg_html
    assert "gemini-embedding-001" in svg_html
    assert "d=3072" in svg_html
    assert "COSINE" in svg_html


def test_verso_html_helpers() -> None:
    hero = render_hero_header("White Title", "Muted Title", "Sub")
    assert "White Title" in hero
    assert "Muted Title" in hero
    assert "Sub" in hero

    metric = render_metric_card("ACTIVE FILES", 12, "Hint")
    assert "ACTIVE FILES" in metric
    assert "12" in metric
    assert "Hint" in metric

    empty = render_empty_state("No Docs", "Desc")
    assert "No Docs" in empty
    assert "Desc" in empty


# ============================================================================
# 2. TEST VIEW MODULE IMPORTS
# ============================================================================


def test_view_module_imports() -> None:
    from views.chat import render_chat
    from views.compare import render_compare
    from views.dashboard import render_dashboard
    from views.extract import render_extract
    from views.search import render_search
    from views.summarize import render_summarize
    from views.upload import render_upload

    assert callable(render_dashboard)
    assert callable(render_upload)
    assert callable(render_chat)
    assert callable(render_summarize)
    assert callable(render_extract)
    assert callable(render_search)
    assert callable(render_compare)


# ============================================================================
# 3. TEST API CLIENT ENDPOINTS & ERROR HANDLING
# ============================================================================


@pytest.fixture
def client() -> DocuMindClient:
    return DocuMindClient(base_url="http://testbackend:8000")


@patch("requests.get")
def test_client_health(mock_get: MagicMock, client: DocuMindClient) -> None:
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "status": "healthy",
        "genai_ready": True,
        "active_model": "gemini-3.8-flash",
    }
    res = client.get_health()
    assert res["status"] == "healthy"
    assert res["genai_ready"] is True


@patch("requests.get")
def test_client_get_documents(mock_get: MagicMock, client: DocuMindClient) -> None:
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "documents": [
            {
                "doc_id": "d1",
                "filename": "test.pdf",
                "file_type": ".pdf",
                "file_size_bytes": 1024,
                "page_count": 1,
                "chunk_count": 2,
                "preview_snippet": "Sample",
                "uploaded_at": "2026-11-23T00:00:00",
            }
        ],
        "stats": {"total_documents": 1, "total_chunks": 2, "total_pages": 1},
    }
    res = client.get_documents(session_id="s1")
    assert len(res["documents"]) == 1
    assert res["stats"]["total_documents"] == 1


@patch("requests.post")
def test_client_upload_documents(mock_post: MagicMock, client: DocuMindClient) -> None:
    mock_post.return_value.status_code = 201
    mock_post.return_value.json.return_value = [
        {"doc_id": "d1", "filename": "sample.pdf", "chunk_count": 3}
    ]
    files = [("sample.pdf", b"pdf content", "application/pdf")]
    res = client.upload_documents(files=files, session_id="s1")
    assert len(res) == 1
    assert res[0]["doc_id"] == "d1"


@patch("requests.delete")
def test_client_delete_document(mock_del: MagicMock, client: DocuMindClient) -> None:
    mock_del.return_value.status_code = 200
    mock_del.return_value.json.return_value = {
        "doc_id": "d1",
        "message": "Deleted",
        "remaining_documents": 0,
        "remaining_chunks": 0,
    }
    res = client.delete_document(doc_id="d1", session_id="s1")
    assert res["doc_id"] == "d1"


@patch("requests.post")
def test_client_classify(mock_post: MagicMock, client: DocuMindClient) -> None:
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "doc_id": "d1",
        "classification": {
            "category": "Research Paper",
            "certainty_level": "High",
            "explanation": "Valid paper",
            "keywords": ["ai", "transformers"],
        },
    }
    res = client.classify_document(doc_id="d1", session_id="s1")
    assert res["classification"]["category"] == "Research Paper"


@patch("requests.post")
def test_client_summarize(mock_post: MagicMock, client: DocuMindClient) -> None:
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "doc_id": "d1",
        "mode": "Executive",
        "strategy_used": "Direct LLM Summarization",
        "summary": "Executive summary content.",
    }
    res = client.summarize_document(doc_id="d1", mode="Executive", session_id="s1")
    assert res["strategy_used"] == "Direct LLM Summarization"
    assert "Executive" in res["summary"]


@patch("requests.post")
def test_client_chat_rag(mock_post: MagicMock, client: DocuMindClient) -> None:
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "answer": "Grounded response [Source: test.pdf, Page: 1].",
        "citations": [
            {
                "doc_id": "d1",
                "filename": "test.pdf",
                "page_number": 1,
                "excerpt": "Excerpt text",
                "similarity_score": 0.89,
            }
        ],
        "genai_pipeline_used": {
            "model": "gemini-3.8-flash",
            "chunks_retrieved": 1,
            "prompt_strategy": "Grounded RAG",
        },
    }
    res = client.chat_rag(question="What is this?", session_id="s1", top_k=3)
    assert len(res["citations"]) == 1
    assert "Source: test.pdf" in res["answer"]


@patch("requests.post")
def test_client_extract(mock_post: MagicMock, client: DocuMindClient) -> None:
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "doc_id": "d1",
        "schema_used": "Invoice",
        "extracted_data": {
            "vendor_name": "Acme Corp",
            "total_amount": 1500.0,
            "currency": "USD",
        },
    }
    res = client.extract_structured(doc_id="d1", schema_type="Invoice", session_id="s1")
    assert res["schema_used"] == "Invoice"
    assert res["extracted_data"]["vendor_name"] == "Acme Corp"


@patch("requests.post")
def test_client_search(mock_post: MagicMock, client: DocuMindClient) -> None:
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "query": "transformers",
        "total_results": 1,
        "results": [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "filename": "test.pdf",
                "page_number": 1,
                "excerpt": "Attention mechanisms",
                "cosine_similarity_score": 0.92,
            }
        ],
    }
    res = client.search_semantic(query="transformers", session_id="s1", top_k=5)
    assert res["total_results"] == 1
    assert res["results"][0]["cosine_similarity_score"] == 0.92


@patch("requests.post")
def test_client_compare(mock_post: MagicMock, client: DocuMindClient) -> None:
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "doc_ids": ["d1", "d2"],
        "comparison_report": {
            "similarities": ["Both study AI"],
            "differences": ["Doc 1 theoretical, Doc 2 applied"],
            "conflicting_information": [],
            "common_themes": ["LLMs"],
            "summary_verdict": "Complementary findings",
        },
    }
    res = client.compare_documents(doc_ids=["d1", "d2"], session_id="s1")
    assert "comparison_report" in res
    assert len(res["comparison_report"]["similarities"]) == 1


@patch("requests.post")
def test_client_error_handling(mock_post: MagicMock, client: DocuMindClient) -> None:
    mock_post.return_value.status_code = 400
    mock_post.return_value.json.return_value = {"detail": "Invalid document ID"}
    with pytest.raises(DocuMindAPIError) as exc_info:
        client.classify_document(doc_id="invalid_id", session_id="s1")
    assert "Invalid document ID" in str(exc_info.value)
