"""Overlapping, boundary-aware text chunker with rich provenance metadata.

EDUCATIONAL ARCHITECTURE NOTES (VIVA):
--------------------------------------
1. The Semantic Seam Problem in RAG:
   If a text is partitioned into hard, non-overlapping windows (e.g. exactly 1,000 characters),
   critical statements often get bifurcated across the seam. For example, a condition might
   end up in Chunk A ("The contract terminates immediately if...") while the governing exception
   lands in Chunk B ("...the lessee notifies the lessor within 30 days.").
   A vector search query matching Chunk A will fail to retrieve Chunk B, causing the LLM
   to hallucinate or present false conclusions.

2. Overlap as Contextual Stitching:
   By configuring a sliding window overlap (e.g. chunk_size=1000, chunk_overlap=200),
   the trailing 200 characters of Chunk A are duplicated as the prefix of Chunk B.
   This guarantees that semantic units spanning boundary seams remain intact within
   at least one chunk's receptive field.

3. Structural Boundary Awareness:
   Splitting along paragraph breaks (`\\n\\n`) and sentence terminals (`. `) rather
   than arbitrary character offsets maintains grammatical integrity, improving both
   dense vector embedding accuracy and LLM comprehension.
"""

from app.core.config import settings
from app.models.schemas import DocumentChunk, DocumentPage


class TextChunker:
    """Splits page-level text into sliding overlapping chunks with provenance metadata."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size: int = (
            chunk_size if chunk_size is not None else settings.DEFAULT_CHUNK_SIZE
        )
        self.chunk_overlap: int = (
            chunk_overlap
            if chunk_overlap is not None
            else settings.DEFAULT_CHUNK_OVERLAP
        )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly smaller "
                f"than chunk_size ({self.chunk_size})"
            )

    def chunk_page(
        self,
        doc_id: str,
        filename: str,
        page: DocumentPage,
        starting_chunk_index: int = 0,
    ) -> list[DocumentChunk]:
        """Splits an individual page's text into overlapping chunks.

        Args:
            doc_id: Unique document identifier.
            filename: Original file name for citation tracking.
            page: DocumentPage instance containing page number and text.
            starting_chunk_index: Offset for chunk indexing across multiple pages.

        Returns:
            List of DocumentChunk objects with unique IDs and metadata.
        """
        text = page.text.strip()
        if not text:
            return []

        chunks: list[DocumentChunk] = []
        step = self.chunk_size - self.chunk_overlap
        current_idx = starting_chunk_index
        text_len = len(text)
        start = 0

        while start < text_len:
            end = start + self.chunk_size

            if end >= text_len:
                chunk_slice = text[start:]
                chunk_text = chunk_slice.strip()
                if chunk_text:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{doc_id}_p{page.page_number}_c{current_idx}",
                            doc_id=doc_id,
                            filename=filename,
                            page_number=page.page_number,
                            chunk_index=current_idx,
                            text=chunk_text,
                        )
                    )
                break

            # Try to snap boundary to natural paragraph or sentence break
            boundary = self._find_natural_boundary(text, start, end)
            chunk_slice = text[start:boundary].strip()

            if chunk_slice:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}_p{page.page_number}_c{current_idx}",
                        doc_id=doc_id,
                        filename=filename,
                        page_number=page.page_number,
                        chunk_index=current_idx,
                        text=chunk_slice,
                    )
                )
                current_idx += 1

            # Advance window by step, adjusting from boundary
            start = max(boundary - self.chunk_overlap, start + step)

        return chunks

    def chunk_document(
        self,
        doc_id: str,
        filename: str,
        pages: list[DocumentPage],
    ) -> list[DocumentChunk]:
        """Chunks all pages in a document while maintaining sequential chunk indices.

        Args:
            doc_id: Unique document ID.
            filename: Original file name.
            pages: List of DocumentPage objects.

        Returns:
            Flat list of all DocumentChunk objects created across all pages.
        """
        all_chunks: list[DocumentChunk] = []
        current_chunk_idx = 0

        for page in pages:
            page_chunks = self.chunk_page(
                doc_id=doc_id,
                filename=filename,
                page=page,
                starting_chunk_index=current_chunk_idx,
            )
            all_chunks.extend(page_chunks)
            current_chunk_idx += len(page_chunks)

        return all_chunks

    def _find_natural_boundary(self, text: str, start: int, proposed_end: int) -> int:
        """Finds the nearest paragraph or sentence delimiter near proposed_end.

        If a natural delimiter exists in the trailing 25% of the window,
        snaps to it to avoid slicing words or mid-sentence structures.
        """
        search_window_start = max(start, proposed_end - int(self.chunk_size * 0.25))
        sub = text[search_window_start:proposed_end]

        # 1. Check for paragraph break
        para_idx = sub.rfind("\n\n")
        if para_idx != -1:
            return search_window_start + para_idx + 2

        # 2. Check for sentence terminal followed by whitespace
        for punct in (". ", "! ", "? ", ":\n"):
            p_idx = sub.rfind(punct)
            if p_idx != -1:
                return search_window_start + p_idx + len(punct)

        # 3. Check for single newline
        nl_idx = sub.rfind("\n")
        if nl_idx != -1:
            return search_window_start + nl_idx + 1

        # 4. Check for space to avoid breaking a word
        space_idx = sub.rfind(" ")
        if space_idx != -1:
            return search_window_start + space_idx + 1

        # Fallback to exact proposed boundary
        return proposed_end


text_chunker = TextChunker()
