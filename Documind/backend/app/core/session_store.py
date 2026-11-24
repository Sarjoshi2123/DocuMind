"""Thread-safe in-memory session store for DocuMind AI.

Manages multi-tenant or multi-session documents, page texts, chunks,
and each session's FAISS VectorStore without requiring an external database.
"""

import threading
from typing import Optional
from app.models.schemas import (
    DocumentChunk,
    DocumentMetadata,
    DocumentPage,
    SessionStats,
)
from app.rag.vector_store import VectorStore


class SessionData:
    """Encapsulates all document data and vector indices for a single session."""

    def __init__(self, session_id: str) -> None:
        self.session_id: str = session_id
        self.documents: dict[str, DocumentMetadata] = {}
        self.pages: dict[str, list[DocumentPage]] = {}
        self.chunks: dict[str, list[DocumentChunk]] = {}
        self.vector_store: VectorStore = VectorStore()
        self.lock: threading.RLock = threading.RLock()


class SessionStore:
    """Thread-safe manager for all active user sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionData] = {}
        self._global_lock: threading.RLock = threading.RLock()

    def get_or_create_session(self, session_id: str) -> SessionData:
        """Retrieves or initializes a session atomically."""
        with self._global_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionData(session_id=session_id)
            return self._sessions[session_id]

    def add_document(
        self,
        session_id: str,
        metadata: DocumentMetadata,
        pages: list[DocumentPage],
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Stores a document, its extracted pages, chunks, and indexes vectors."""
        session = self.get_or_create_session(session_id)
        with session.lock:
            session.documents[metadata.doc_id] = metadata
            session.pages[metadata.doc_id] = pages
            session.chunks[metadata.doc_id] = chunks
            if chunks and embeddings:
                session.vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

    def get_document(
        self, session_id: str, doc_id: str
    ) -> Optional[DocumentMetadata]:
        """Returns document metadata if found in session."""
        session = self.get_or_create_session(session_id)
        with session.lock:
            return session.documents.get(doc_id)

    def get_document_pages(
        self, session_id: str, doc_id: str
    ) -> list[DocumentPage]:
        """Returns extracted pages for a document."""
        session = self.get_or_create_session(session_id)
        with session.lock:
            return session.pages.get(doc_id, [])

    def get_document_full_text(self, session_id: str, doc_id: str) -> str:
        """Combines all page texts for a given document."""
        session = self.get_or_create_session(session_id)
        with session.lock:
            pages = session.pages.get(doc_id, [])
            return "\n\n".join(p.text for p in pages if p.text.strip())

    def list_documents(self, session_id: str) -> list[DocumentMetadata]:
        """Lists all document metadata in the session."""
        session = self.get_or_create_session(session_id)
        with session.lock:
            return list(session.documents.values())

    def get_session_stats(self, session_id: str) -> SessionStats:
        """Computes aggregate document, chunk, and page statistics for a session."""
        session = self.get_or_create_session(session_id)
        with session.lock:
            total_docs = len(session.documents)
            total_chunks = sum(len(c) for c in session.chunks.values())
            total_pages = sum(len(p) for p in session.pages.values())
            return SessionStats(
                total_documents=total_docs,
                total_chunks=total_chunks,
                total_pages=total_pages,
            )

    def delete_document(
        self, session_id: str, doc_id: str
    ) -> tuple[bool, int, int]:
        """Deletes a document from the session and triggers index rebuild.

        Returns (success, remaining_docs, remaining_chunks).
        """
        session = self.get_or_create_session(session_id)
        with session.lock:
            if doc_id not in session.documents:
                remaining_docs = len(session.documents)
                remaining_chunks = sum(len(c) for c in session.chunks.values())
                return False, remaining_docs, remaining_chunks

            # Remove doc from maps
            del session.documents[doc_id]
            session.pages.pop(doc_id, None)
            session.chunks.pop(doc_id, None)

            # Rebuild vector store with remaining chunks
            session.vector_store.remove_document_and_rebuild(doc_id=doc_id)

            remaining_docs = len(session.documents)
            remaining_chunks = sum(len(c) for c in session.chunks.values())
            return True, remaining_docs, remaining_chunks

    def get_vector_store(self, session_id: str) -> VectorStore:
        """Returns the FAISS VectorStore for the session."""
        session = self.get_or_create_session(session_id)
        return session.vector_store

    def clear_session(self, session_id: str) -> None:
        """Resets a session completely."""
        with self._global_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]


# Global singleton session store
session_store = SessionStore()
