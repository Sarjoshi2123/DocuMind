"""FastAPI Backend Client for DocuMind AI Frontend.

Wraps all 10 REST endpoints with clean error handling, X-Session-ID tracking,
and sensible network timeouts.
"""

import os
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
import requests

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class DocuMindAPIError(Exception):
    """Custom exception raised when backend requests fail."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class DocuMindClient:
    """HTTP client interacting with the DocuMind FastAPI backend."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url: str = (
            base_url
            or os.getenv("BACKEND_URL", "http://localhost:8000")
        ).rstrip("/")

    def _headers(self, session_id: str) -> dict[str, str]:
        return {
            "X-Session-ID": session_id,
            "Accept": "application/json",
        }

    # 1. Health Check
    def get_health(self) -> dict[str, Any]:
        """Checks backend health and model readiness."""
        try:
            resp = requests.get(f"{self.base_url}/api/health", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {"status": "unhealthy", "genai_ready": False}
        except Exception:
            return {"status": "offline", "genai_ready": False}

    # 2. Upload Documents
    def upload_documents(
        self,
        files: list[tuple[str, bytes, str]],  # (filename, bytes, content_type)
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Uploads files to the session FAISS index."""
        url = f"{self.base_url}/api/documents/upload"
        multipart_files = [
            ("files", (fname, fbytes, mtype))
            for fname, fbytes, mtype in files
        ]
        try:
            resp = requests.post(
                url,
                files=multipart_files,
                params={"session_id": session_id},
                headers=self._headers(session_id),
                timeout=90,
            )
            if resp.status_code == 201:
                return resp.json()
            err_msg = resp.json().get("detail", resp.text)
            raise DocuMindAPIError(f"Upload failed: {err_msg}", resp.status_code)
        except requests.RequestException as e:
            raise DocuMindAPIError(f"Network error during upload: {str(e)}") from e

    # 3. List Documents & Stats
    def get_documents(self, session_id: str) -> dict[str, Any]:
        """Fetches all documents and statistics for a session."""
        url = f"{self.base_url}/api/documents"
        try:
            resp = requests.get(
                url,
                params={"session_id": session_id},
                headers=self._headers(session_id),
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            return {"documents": [], "stats": {"total_documents": 0, "total_chunks": 0, "total_pages": 0}}
        except requests.RequestException:
            return {"documents": [], "stats": {"total_documents": 0, "total_chunks": 0, "total_pages": 0}}

    # 4. Delete Document
    def delete_document(self, doc_id: str, session_id: str) -> dict[str, Any]:
        """Deletes a document and rebuilds the session vector index."""
        url = f"{self.base_url}/api/documents/{doc_id}"
        try:
            resp = requests.delete(
                url,
                params={"session_id": session_id},
                headers=self._headers(session_id),
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            raise DocuMindAPIError(resp.json().get("detail", resp.text), resp.status_code)
        except requests.RequestException as e:
            raise DocuMindAPIError(f"Failed to delete document: {str(e)}") from e

    # 5. Classify Document
    def classify_document(self, doc_id: str, session_id: str) -> dict[str, Any]:
        """Classifies a document into one of 8 canonical categories."""
        url = f"{self.base_url}/api/classify"
        try:
            resp = requests.post(
                url,
                json={"doc_id": doc_id},
                params={"session_id": session_id},
                headers=self._headers(session_id),
                timeout=45,
            )
            if resp.status_code == 200:
                return resp.json()
            raise DocuMindAPIError(resp.json().get("detail", resp.text), resp.status_code)
        except requests.RequestException as e:
            raise DocuMindAPIError(f"Classification failed: {str(e)}") from e

    # 6. Summarize Document
    def summarize_document(
        self, doc_id: str, mode: str, session_id: str
    ) -> dict[str, Any]:
        """Generates a Direct or Hierarchical Map-Reduce summary."""
        url = f"{self.base_url}/api/summarize"
        try:
            resp = requests.post(
                url,
                json={"doc_id": doc_id, "mode": mode},
                params={"session_id": session_id},
                headers=self._headers(session_id),
                timeout=90,
            )
            if resp.status_code == 200:
                return resp.json()
            raise DocuMindAPIError(resp.json().get("detail", resp.text), resp.status_code)
        except requests.RequestException as e:
            raise DocuMindAPIError(f"Summarization failed: {str(e)}") from e

    # 7. Grounded RAG Chat
    def chat_rag(
        self,
        question: str,
        session_id: str,
        doc_ids: Optional[list[str]] = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Executes grounded Q&A with source citations."""
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {"question": question, "top_k": top_k}
        if doc_ids:
            payload["doc_ids"] = doc_ids

        try:
            resp = requests.post(
                url,
                json=payload,
                params={"session_id": session_id},
                headers=self._headers(session_id),
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json()
            raise DocuMindAPIError(resp.json().get("detail", resp.text), resp.status_code)
        except requests.RequestException as e:
            raise DocuMindAPIError(f"Chat request failed: {str(e)}") from e

    # 8. Structured Extraction
    def extract_structured(
        self, doc_id: str, schema_type: str, session_id: str
    ) -> dict[str, Any]:
        """Extracts structured JSON adhering to Pydantic schemas."""
        url = f"{self.base_url}/api/extract"
        try:
            resp = requests.post(
                url,
                json={"doc_id": doc_id, "schema_type": schema_type},
                params={"session_id": session_id},
                headers=self._headers(session_id),
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json()
            raise DocuMindAPIError(resp.json().get("detail", resp.text), resp.status_code)
        except requests.RequestException as e:
            raise DocuMindAPIError(f"Extraction failed: {str(e)}") from e

    # 9. Semantic Vector Search
    def search_semantic(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        doc_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Performs pure cosine similarity search against the session FAISS index."""
        url = f"{self.base_url}/api/search"
        payload: dict[str, Any] = {"query": query, "top_k": top_k}
        if doc_ids:
            payload["doc_ids"] = doc_ids

        try:
            resp = requests.post(
                url,
                json=payload,
                params={"session_id": session_id},
                headers=self._headers(session_id),
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()
            raise DocuMindAPIError(resp.json().get("detail", resp.text), resp.status_code)
        except requests.RequestException as e:
            raise DocuMindAPIError(f"Search request failed: {str(e)}") from e

    # 10. Multi-Document Comparison
    def compare_documents(
        self, doc_ids: list[str], session_id: str
    ) -> dict[str, Any]:
        """Cross-analyzes 2+ documents for similarities, differences, and conflicts."""
        url = f"{self.base_url}/api/compare"
        try:
            resp = requests.post(
                url,
                json={"doc_ids": doc_ids},
                params={"session_id": session_id},
                headers=self._headers(session_id),
                timeout=90,
            )
            if resp.status_code == 200:
                return resp.json()
            raise DocuMindAPIError(resp.json().get("detail", resp.text), resp.status_code)
        except requests.RequestException as e:
            raise DocuMindAPIError(f"Comparison failed: {str(e)}") from e


api_client = DocuMindClient()
