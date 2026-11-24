"""Document parsing, ingestion, chunking, and embedding orchestration service.

Handles PDF (PyMuPDF with robust pypdf fallback), DOCX (python-docx), and TXT files,
extracting page-level content, cleaning textual artifacts, generating overlapping chunks,
batching embeddings, and indexing them in the active session's vector store.
"""

from datetime import datetime, timezone
import io
import os
import uuid

# PDF engines: prefer PyMuPDF if available and not blocked by Smart App Control; fallback to pure Python pypdf
try:
    import fitz  # PyMuPDF
except (ImportError, Exception):
    fitz = None

try:
    import pypdf
except (ImportError, Exception):
    pypdf = None

from docx import Document as DocxDocument
from fastapi import HTTPException, UploadFile, status
from app.ai.gemini_provider import gemini_provider
from app.ai.llm_provider import BaseLLMProvider
from app.core.config import settings
from app.core.session_store import session_store
from app.models.schemas import DocumentMetadata, DocumentPage
from app.rag.chunker import text_chunker
from app.rag.text_cleaner import clean_text


class DocumentService:
    """Service orchestrating document ingestion and pipeline processing."""

    def __init__(self, llm_provider: BaseLLMProvider | None = None) -> None:
        self.llm_provider: BaseLLMProvider = (
            llm_provider if llm_provider is not None else gemini_provider
        )

    def validate_file(self, filename: str, file_size: int) -> str:
        """Validates file extension and size constraints.

        Returns:
            Normalized lowercase file extension.
        """
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported file format '{ext}'. Allowed extensions are: "
                    f"{', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
                ),
            )

        if file_size > settings.MAX_FILE_SIZE_BYTES:
            max_mb = settings.MAX_FILE_SIZE_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {max_mb} MB.",
            )

        return ext

    def parse_pdf(self, file_bytes: bytes) -> list[DocumentPage]:
        """Parses PDF bytes extracting text with exact 1-indexed page numbers.

        Uses PyMuPDF (fitz) if accessible; seamlessly falls back to pure-Python pypdf
        to avoid Windows Smart App Control binary blocks.
        """
        pages: list[DocumentPage] = []
        parsed = False

        if fitz is not None:
            try:
                with fitz.open(stream=file_bytes, filetype="pdf") as pdf_doc:
                    for page_idx in range(len(pdf_doc)):
                        page = pdf_doc[page_idx]
                        page_text = clean_text(page.get_text("text"))
                        pages.append(
                            DocumentPage(
                                page_number=page_idx + 1,
                                text=page_text,
                            )
                        )
                parsed = True
            except Exception:
                pages = []
                parsed = False

        if not parsed and pypdf is not None:
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                for page_idx, page in enumerate(reader.pages):
                    page_text = clean_text(page.extract_text() or "")
                    pages.append(
                        DocumentPage(
                            page_number=page_idx + 1,
                            text=page_text,
                        )
                    )
                parsed = True
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Failed to parse PDF document: {str(exc)}",
                ) from exc

        if not parsed:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No available PDF engine found. Please ensure pypdf is installed.",
            )

        return pages

    def parse_docx(self, file_bytes: bytes) -> list[DocumentPage]:
        """Parses DOCX document paragraphs, packaging them into structured pages."""
        try:
            doc = DocxDocument(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            cleaned_text = clean_text("\n\n".join(paragraphs))

            # DOCX lacks native fixed page boundaries; treat as page 1 or split if large
            pages: list[DocumentPage] = [
                DocumentPage(
                    page_number=1,
                    text=cleaned_text,
                )
            ]
            return pages
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse DOCX document: {str(exc)}",
            ) from exc

    def parse_txt(self, file_bytes: bytes) -> list[DocumentPage]:
        """Parses plain text bytes with robust encoding fallback."""
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1", errors="replace")

        cleaned = clean_text(text)
        return [DocumentPage(page_number=1, text=cleaned)]

    def extract_pages(self, filename: str, file_bytes: bytes) -> list[DocumentPage]:
        """Routes file to appropriate parser based on extension."""
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext == ".pdf":
            return self.parse_pdf(file_bytes)
        elif ext == ".docx":
            return self.parse_docx(file_bytes)
        elif ext == ".txt":
            return self.parse_txt(file_bytes)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {ext}",
            )

    async def process_and_ingest_file(
        self,
        file: UploadFile,
        session_id: str,
        provider: BaseLLMProvider | None = None,
    ) -> DocumentMetadata:
        """Processes an uploaded file through validation, parsing, chunking,

        embedding, and session vector indexing.
        """
        active_provider = provider or self.llm_provider
        content = await file.read()
        file_size = len(content)
        filename = file.filename or "unknown_document.txt"

        # Validate
        file_ext = self.validate_file(filename=filename, file_size=file_size)

        # Parse text into pages
        pages = self.extract_pages(filename=filename, file_bytes=content)

        # Check for non-empty content
        total_text_len = sum(len(p.text.strip()) for p in pages)
        if total_text_len == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document '{filename}' contains no readable or extractable text.",
            )

        doc_id = str(uuid.uuid4())

        # Chunk pages into overlapping chunks
        chunks = text_chunker.chunk_document(
            doc_id=doc_id,
            filename=filename,
            pages=pages,
        )

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to generate chunks for document '{filename}'.",
            )

        # Compute embeddings for all chunks in batch
        chunk_texts = [c.text for c in chunks]
        embeddings = active_provider.get_embeddings(chunk_texts)

        # Generate snippet preview
        first_page_text = pages[0].text if pages else ""
        preview_snippet = (
            first_page_text[:300].strip() + "..."
            if len(first_page_text) > 300
            else first_page_text[:300].strip()
        )

        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_type=file_ext,
            file_size_bytes=file_size,
            page_count=len(pages),
            chunk_count=len(chunks),
            preview_snippet=preview_snippet,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
        )

        # Store in session and vector index
        session_store.add_document(
            session_id=session_id,
            metadata=metadata,
            pages=pages,
            chunks=chunks,
            embeddings=embeddings,
        )

        return metadata


document_service = DocumentService()
