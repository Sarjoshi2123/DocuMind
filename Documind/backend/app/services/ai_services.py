"""GenAI Business Logic orchestrating DocuMind's 6 Core AI Capabilities.

EDUCATIONAL ARCHITECTURE NOTES (VIVA):
--------------------------------------
1. RAG (Retrieval-Augmented Generation):
   Standard LLMs suffer from knowledge cutoffs and hallucinations. RAG grounds
   generation in proprietary, deterministic external knowledge retrieved from vector
   indices. The LLM acts solely as a reasoning and synthesis engine over retrieved
   citations rather than relying on its internal parametric memory.

2. Context Window Management (Hierarchical Map-Reduce):
   While modern Gemini models offer million-token context windows, stuffing raw
   hundred-page documents into a single prompt suffers from "Lost in the Middle"
   attention decay and high latency/cost. Hierarchical Map-Reduce summarizes discrete
   sections in parallel (Map) before synthesizing cross-cutting themes (Reduce),
   preserving dense factual recall without degradation.

3. Structured Outputs with Guaranteed JSON Schemas:
   Using Pydantic models with Gemini's `response_schema` leverages constrained grammar
   decoding. The model's token logits are masked at each generation step so that only
   syntactically valid tokens conforming to the JSON schema can be emitted, eliminating
   regex parsing failures and schema drift.
"""

from typing import Any, Optional, Type
from fastapi import HTTPException, status
from pydantic import BaseModel
from app.ai.gemini_provider import gemini_provider
from app.ai.llm_provider import BaseLLMProvider
from app.ai.prompt_manager import prompt_manager
from app.core.config import settings
from app.core.session_store import session_store
from app.models.schemas import (
    ChatPipelineMetadata,
    ChatResponse,
    Citation,
    ClassificationResult,
    ComparisonReport,
    ContractSchema,
    ExtractionSchemaType,
    InvoiceSchema,
    ResearchPaperSchema,
    ResumeSchema,
    SearchResultItem,
    SummarizationMode,
)


class AIService:
    """Orchestrates all Generative AI tasks across documents and sessions."""

    def __init__(self, llm_provider: BaseLLMProvider | None = None) -> None:
        self.llm_provider: BaseLLMProvider = (
            llm_provider if llm_provider is not None else gemini_provider
        )

    # =========================================================================
    # 1. DOCUMENT CLASSIFICATION
    # =========================================================================
    def classify_document(
        self,
        session_id: str,
        doc_id: str,
        provider: BaseLLMProvider | None = None,
    ) -> ClassificationResult:
        """Classifies a document into one of 8 standardized domain categories.

        Uses the document's header, first pages, and key structural cues with
        Gemini structured output to guarantee deterministic categorization.
        """
        active_provider = provider or self.llm_provider
        doc_meta = session_store.get_document(session_id, doc_id)
        if not doc_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{doc_id}' not found in active session.",
            )

        full_text = session_store.get_document_full_text(session_id, doc_id)
        # Use first ~8000 characters (sufficient for title, abstract, headers, layout)
        sample_text = full_text[:8000].strip()

        sys_prompt = prompt_manager.get_classification_system_prompt()
        user_prompt = (
            f"Filename: {doc_meta.filename}\n\n"
            f"<<<UNTRUSTED_DOCUMENT_CONTENT>>>\n{sample_text}\n<<<END_UNTRUSTED_DOCUMENT_CONTENT>>>\n\n"
            "Analyze the above excerpt and produce the classification JSON."
        )

        return active_provider.generate_structured(
            prompt=user_prompt,
            response_schema=ClassificationResult,
            system_instruction=sys_prompt,
            temperature=0.1,
        )

    # =========================================================================
    # 2. SUMMARIZATION (DIRECT VS HIERARCHICAL MAP-REDUCE)
    # =========================================================================
    def summarize_document(
        self,
        session_id: str,
        doc_id: str,
        mode: SummarizationMode = "Executive",
        provider: BaseLLMProvider | None = None,
    ) -> tuple[str, str]:
        """Summarizes a document based on character volume.

        - Documents <= 14,000 chars: Direct single-pass LLM summarization.
        - Documents > 14,000 chars: Hierarchical Map-Reduce summarization
          (splits into sections, generates intermediate summaries, and reduces).

        Returns:
            Tuple of (summary_text, strategy_used).
        """
        active_provider = provider or self.llm_provider
        doc_meta = session_store.get_document(session_id, doc_id)
        if not doc_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{doc_id}' not found in active session.",
            )

        full_text = session_store.get_document_full_text(session_id, doc_id)
        if not full_text:
            return "Document contains no extractable text to summarize.", "Direct LLM"

        # Case A: Short to medium document -> Direct Summarization
        if len(full_text) <= settings.LONG_DOCUMENT_CHAR_THRESHOLD:
            sys_prompt = prompt_manager.get_summarization_system_prompt(mode)
            user_prompt = (
                f"Document Name: {doc_meta.filename}\n\n"
                f"<<<UNTRUSTED_DOCUMENT_CONTENT>>>\n{full_text}\n<<<END_UNTRUSTED_DOCUMENT_CONTENT>>>\n\n"
                f"Generate a {mode} summary based strictly on the above document."
            )
            summary = active_provider.generate_text(
                prompt=user_prompt,
                system_instruction=sys_prompt,
                temperature=0.2,
            )
            return summary, "Direct LLM Summarization"

        # Case B: Long document -> Hierarchical Map-Reduce
        strategy_used = "Hierarchical Map-Reduce Summarization"
        section_size = 10000
        section_overlap = 500
        step = section_size - section_overlap

        sections: list[str] = []
        for i in range(0, len(full_text), step):
            sec = full_text[i : i + section_size].strip()
            if sec:
                sections.append(sec)

        # Map phase: generate concise summary for each section
        section_summaries: list[str] = []
        total_secs = len(sections)

        for idx, sec in enumerate(sections, 1):
            map_sys_prompt = prompt_manager.get_hierarchical_map_prompt(
                section_idx=idx, total_sections=total_secs
            )
            map_prompt = (
                f"Document: {doc_meta.filename} (Section {idx}/{total_secs})\n\n"
                f"<<<UNTRUSTED_DOCUMENT_CONTENT>>>\n{sec}\n<<<END_UNTRUSTED_DOCUMENT_CONTENT>>>\n\n"
                "Synthesize the key points and findings from this section."
            )
            sec_summary = active_provider.generate_text(
                prompt=map_prompt,
                system_instruction=map_sys_prompt,
                temperature=0.2,
            )
            section_summaries.append(f"### Section {idx} Summary:\n{sec_summary}")

        # Reduce phase: synthesize all intermediate section summaries into final target mode
        combined_intermediate = "\n\n".join(section_summaries)
        reduce_sys_prompt = prompt_manager.get_hierarchical_reduce_prompt(mode)
        reduce_prompt = (
            f"Document: {doc_meta.filename}\n\n"
            f"Intermediate Section Summaries:\n{combined_intermediate}\n\n"
            f"Synthesize the above intermediate section summaries into a final coherent {mode} summary."
        )

        final_summary = active_provider.generate_text(
            prompt=reduce_prompt,
            system_instruction=reduce_sys_prompt,
            temperature=0.2,
        )

        return final_summary, strategy_used

    # =========================================================================
    # 3. GROUNDED RAG CHAT
    # =========================================================================
    def chat_rag(
        self,
        session_id: str,
        question: str,
        doc_ids: Optional[list[str]] = None,
        top_k: int = 5,
        provider: BaseLLMProvider | None = None,
    ) -> ChatResponse:
        """Executes full Grounded Retrieval-Augmented Generation Q&A.

        Steps:
        1. Embeds question with Gemini Embedding model.
        2. Retrieves top_k most similar chunks from FAISS vector store.
        3. Formulates context block with [Source: filename, Page: X] citation anchors.
        4. Evaluates if context is available; prompts LLM under strict hallucination defense.
        """
        active_provider = provider or self.llm_provider
        vector_store = session_store.get_vector_store(session_id)

        # Check if any documents exist in session
        if vector_store.total_chunks() == 0:
            return ChatResponse(
                answer="No documents have been uploaded to this session yet. Please upload documents first.",
                citations=[],
                genai_pipeline_used=ChatPipelineMetadata(
                    model=settings.GEMINI_MODEL,
                    chunks_retrieved=0,
                    prompt_strategy="Zero-Document Direct Refusal",
                ),
            )

        # 1. Embed query
        q_embeddings = active_provider.get_embeddings([question])
        if not q_embeddings:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate embedding for query.",
            )
        q_vec = q_embeddings[0]

        # 2. Retrieve top-k chunks
        search_matches = vector_store.search(
            query_embedding=q_vec,
            top_k=top_k,
            doc_ids=doc_ids,
        )

        if not search_matches:
            return ChatResponse(
                answer="I cannot find this information in the uploaded documents.",
                citations=[],
                genai_pipeline_used=ChatPipelineMetadata(
                    model=settings.GEMINI_MODEL,
                    chunks_retrieved=0,
                    prompt_strategy="Grounded RAG with Prompt-Injection Defense",
                ),
            )

        # 3. Construct grounded context block with citations
        citations: list[Citation] = []
        context_blocks: list[str] = []

        for chunk, score in search_matches:
            citations.append(
                Citation(
                    doc_id=chunk.doc_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    excerpt=chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
                    similarity_score=round(score, 4),
                )
            )
            block = (
                f"[Source: {chunk.filename}, Page: {chunk.page_number}]\n"
                f"{chunk.text}"
            )
            context_blocks.append(block)

        context_str = "\n\n---\n\n".join(context_blocks)

        sys_prompt = prompt_manager.get_rag_chat_system_prompt()
        user_prompt = (
            f"User Question:\n{question}\n\n"
            f"<<<UNTRUSTED_DOCUMENT_CONTENT>>>\n{context_str}\n<<<END_UNTRUSTED_DOCUMENT_CONTENT>>>\n\n"
            "Provide a factual answer grounded exclusively in the context above. "
            "Cite each source chunk using '[Source: <filename>, Page: <page_number>]'. "
            "If the answer cannot be confirmed directly from the text, state: "
            "'I cannot find this information in the uploaded documents.'"
        )

        answer = active_provider.generate_text(
            prompt=user_prompt,
            system_instruction=sys_prompt,
            temperature=0.1,
        )

        return ChatResponse(
            answer=answer,
            citations=citations,
            genai_pipeline_used=ChatPipelineMetadata(
                model=settings.GEMINI_MODEL,
                chunks_retrieved=len(citations),
                prompt_strategy="Grounded RAG with Prompt-Injection Defense",
            ),
        )

    # =========================================================================
    # 4. STRUCTURED DATA EXTRACTION
    # =========================================================================
    def extract_structured(
        self,
        session_id: str,
        doc_id: str,
        schema_type: ExtractionSchemaType = "Auto-Detect",
        provider: BaseLLMProvider | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Extracts structured JSON adhering to a specific or auto-detected schema.

        Enforces null values for absent fields and eliminates fabricated data.
        """
        active_provider = provider or self.llm_provider
        doc_meta = session_store.get_document(session_id, doc_id)
        if not doc_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{doc_id}' not found in active session.",
            )

        full_text = session_store.get_document_full_text(session_id, doc_id)
        sample_text = full_text[:14000].strip()

        # Schema registry mapping
        schema_mapping: dict[str, Type[BaseModel]] = {
            "Invoice": InvoiceSchema,
            "Resume": ResumeSchema,
            "Research Paper": ResearchPaperSchema,
            "Contract": ContractSchema,
        }

        chosen_schema_name = schema_type

        # Auto-Detect schema type if requested
        if schema_type == "Auto-Detect":
            classification = self.classify_document(
                session_id=session_id, doc_id=doc_id, provider=active_provider
            )
            cat = classification.category
            if cat == "Invoice":
                chosen_schema_name = "Invoice"
            elif cat == "Resume":
                chosen_schema_name = "Resume"
            elif cat == "Research Paper":
                chosen_schema_name = "Research Paper"
            elif cat in ("Legal Contract", "Financial Statement", "Business Report"):
                chosen_schema_name = "Contract"
            else:
                chosen_schema_name = "Research Paper"

        target_model = schema_mapping.get(chosen_schema_name, ResearchPaperSchema)
        sys_prompt = prompt_manager.get_extraction_system_prompt(chosen_schema_name)
        user_prompt = (
            f"Target Document: {doc_meta.filename}\n\n"
            f"<<<UNTRUSTED_DOCUMENT_CONTENT>>>\n{sample_text}\n<<<END_UNTRUSTED_DOCUMENT_CONTENT>>>\n\n"
            f"Extract all fields adhering to the '{chosen_schema_name}' schema. "
            "Set any unmentioned or missing field to null."
        )

        extracted_model = active_provider.generate_structured(
            prompt=user_prompt,
            response_schema=target_model,
            system_instruction=sys_prompt,
            temperature=0.1,
        )

        return chosen_schema_name, extracted_model.model_dump()

    # =========================================================================
    # 5. SEMANTIC VECTOR SEARCH
    # =========================================================================
    def semantic_search(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[list[str]] = None,
        provider: BaseLLMProvider | None = None,
    ) -> list[SearchResultItem]:
        """Performs pure mathematical cosine similarity search against the session FAISS index.

        No LLM generation is performed.
        """
        active_provider = provider or self.llm_provider
        vector_store = session_store.get_vector_store(session_id)

        if vector_store.total_chunks() == 0:
            return []

        q_embeddings = active_provider.get_embeddings([query])
        if not q_embeddings:
            return []

        matches = vector_store.search(
            query_embedding=q_embeddings[0],
            top_k=top_k,
            doc_ids=doc_ids,
        )

        results: list[SearchResultItem] = []
        for chunk, score in matches:
            results.append(
                SearchResultItem(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    excerpt=chunk.text,
                    cosine_similarity_score=round(score, 4),
                )
            )

        return results

    # =========================================================================
    # 6. MULTI-DOCUMENT COMPARATIVE SYNTHESIS
    # =========================================================================
    def compare_documents(
        self,
        session_id: str,
        doc_ids: list[str],
        provider: BaseLLMProvider | None = None,
    ) -> ComparisonReport:
        """Synthesizes an objective comparative report across multiple documents.

        Extracts representative sections from each document to fit within context budgets.
        """
        active_provider = provider or self.llm_provider
        if len(doc_ids) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least two doc_ids are required for comparative analysis.",
            )

        doc_excerpts: list[str] = []
        for d_id in doc_ids:
            doc_meta = session_store.get_document(session_id, d_id)
            if not doc_meta:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document '{d_id}' not found in active session.",
                )
            full_text = session_store.get_document_full_text(session_id, d_id)
            # Budget ~5,000 chars per document to allow comparing several documents smoothly
            snippet = full_text[:5000].strip()
            doc_excerpts.append(
                f"### Document: {doc_meta.filename} (ID: {d_id})\n{snippet}"
            )

        combined_docs_text = "\n\n====================\n\n".join(doc_excerpts)
        sys_prompt = prompt_manager.get_comparison_system_prompt()
        user_prompt = (
            "Examine and contrast the following documents:\n\n"
            f"<<<UNTRUSTED_DOCUMENT_CONTENT>>>\n{combined_docs_text}\n<<<END_UNTRUSTED_DOCUMENT_CONTENT>>>\n\n"
            "Generate the comprehensive comparative JSON report according to schema."
        )

        return active_provider.generate_structured(
            prompt=user_prompt,
            response_schema=ComparisonReport,
            system_instruction=sys_prompt,
            temperature=0.1,
        )


ai_service = AIService()
