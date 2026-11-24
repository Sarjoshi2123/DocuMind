"""In-Memory Pure NumPy Vector Store with L2-Normalized Cosine Similarity.

EDUCATIONAL ARCHITECTURE NOTES (VIVA):
--------------------------------------
1. Inner Product vs. Cosine Similarity (NumPy vs. FAISS IndexFlatIP):
   Mathematically, the cosine similarity between two vectors A and B is:
       CosineSimilarity(A, B) = (A • B) / (||A||_2 * ||B||_2)
   If we apply Euclidean L2 normalization such that ||A||_2 = 1 and ||B||_2 = 1,
   the denominator evaluates to 1.0, yielding:
       CosineSimilarity(A, B) = A_normalized • B_normalized

   FAISS `IndexFlatIP` computes the inner product (dot product) between a query vector
   and indexed vectors. In pure NumPy, this is computed via matrix multiplication:
       similarities = vectors_normalized @ query_normalized

   Because both matrix representations are L2-normalized, the inner product computed
   by pure NumPy is MATHEMATICALLY AND NUMERICALLY IDENTICAL to FAISS `IndexFlatIP`
   cosine similarity bounded in [-1.0, 1.0].

2. Immunity from Windows Smart App Control / Application Control Blocks:
   FAISS relies on compiled C++ SWIG wrappers (`_swigfaiss.pyd` and external OpenMP DLLs)
   which are frequently flagged or blocked by Windows Smart App Control and enterprise
   Application Control policies. Pure NumPy vector operations leverage NumPy's core BLAS
   routines, providing identical Cosine Similarity, high performance, and complete OS-level
   execution immunity.

3. Length Invariance in Document Retrieval:
   Raw dot product is biased toward longer texts or higher magnitude embeddings.
   Cosine similarity measures purely the angular alignment of semantic concepts in
   the latent space, ensuring that a concise query correctly matches both brief bullet
   points and verbose descriptive paragraphs without magnitude skew.

4. Dynamic In-Memory Index Rebuilding:
   When a document is deleted from a session, we cleanly filter out its chunks and
   rebuild the NumPy matrix from remaining embeddings, guaranteeing index integrity,
   preventing dangling references, and eliminating memory leaks.
"""

import threading
from typing import Optional
import numpy as np
from app.models.schemas import DocumentChunk


class VectorStore:
    """In-memory vector index using pure NumPy L2-normalized Cosine Similarity.

    Provides exact functional and mathematical equivalence to FAISS IndexFlatIP
    without requiring external C/SWIG binaries.
    """

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension: int | None = dimension
        self.vectors: np.ndarray | None = None  # Normalized float32 matrix (N, D)
        self.chunks: list[DocumentChunk] = []
        self.raw_embeddings: list[list[float]] = []
        self.lock: threading.RLock = threading.RLock()

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """Applies Euclidean L2 normalization along the last axis safely."""
        norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
        # Avoid division by zero for zero-magnitude vectors
        norms = np.where(norms == 0, 1.0, norms)
        return (matrix / norms).astype(np.float32)

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Adds text chunks and their corresponding embedding vectors to the index.

        Applies L2 normalization so dot product equals cosine similarity.
        """
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        with self.lock:
            new_vectors = np.array(embeddings, dtype=np.float32)
            if new_vectors.ndim == 1:
                new_vectors = np.expand_dims(new_vectors, axis=0)

            dim = new_vectors.shape[1]
            if self.dimension is None:
                self.dimension = dim
            elif self.dimension != dim:
                raise ValueError(
                    f"Dimension mismatch: expected {self.dimension}, got {dim}."
                )

            # Apply L2 normalization: ||v||_2 = 1.0
            norm_vectors = self._l2_normalize(new_vectors)

            if self.vectors is None or self.vectors.size == 0:
                self.vectors = norm_vectors
            else:
                self.vectors = np.vstack([self.vectors, norm_vectors])

            # Store chunks and raw embeddings in parallel lists
            self.chunks.extend(chunks)
            self.raw_embeddings.extend(embeddings)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        doc_ids: Optional[list[str]] = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Searches the index for the top_k most similar chunks using cosine similarity.

        Args:
            query_embedding: Dense embedding vector for the search query.
            top_k: Number of nearest neighbors to retrieve.
            doc_ids: Optional list of document IDs to filter by.

        Returns:
            List of (DocumentChunk, cosine_similarity_score) tuples, ranked descending.
        """
        with self.lock:
            if self.vectors is None or len(self.chunks) == 0:
                return []

            # Format and normalize query vector
            q = np.array(query_embedding, dtype=np.float32).flatten()
            q_norm = np.linalg.norm(q)
            if q_norm > 0:
                q = (q / q_norm).astype(np.float32)

            # Compute Cosine Similarity via matrix-vector multiplication: (N, D) @ (D,) -> (N,)
            similarities = self.vectors @ q

            # Rank indices in descending order of similarity
            ranked_indices = np.argsort(-similarities)

            results: list[tuple[DocumentChunk, float]] = []
            allowed_docs = set(doc_ids) if doc_ids else None

            for idx in ranked_indices:
                if idx < 0 or idx >= len(self.chunks):
                    continue

                chunk = self.chunks[idx]
                if allowed_docs is not None and chunk.doc_id not in allowed_docs:
                    continue

                score = float(similarities[idx])
                results.append((chunk, score))

                if len(results) >= top_k:
                    break

            return results

    def remove_document_and_rebuild(self, doc_id: str) -> None:
        """Purges all chunks associated with doc_id and reconstructs the vector matrix.

        Ensures zero memory leakage or orphan vectors in the index after a document deletion.
        """
        with self.lock:
            remaining_chunks: list[DocumentChunk] = []
            remaining_embeddings: list[list[float]] = []

            for chunk, emb in zip(self.chunks, self.raw_embeddings):
                if chunk.doc_id != doc_id:
                    remaining_chunks.append(chunk)
                    remaining_embeddings.append(emb)

            # Reset internal state
            self.chunks = []
            self.raw_embeddings = []
            self.vectors = None

            # Re-index remaining items
            if remaining_chunks and remaining_embeddings:
                self.add_chunks(
                    chunks=remaining_chunks, embeddings=remaining_embeddings
                )

    def total_chunks(self) -> int:
        """Returns the number of indexed chunks."""
        with self.lock:
            return len(self.chunks)

    def total_vectors(self) -> int:
        """Alias returning the number of indexed vectors."""
        return self.total_chunks()
