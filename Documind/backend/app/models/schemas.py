"""Pydantic v2 schemas and models for DocuMind AI.

Contains request/response models and structured extraction schemas.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ============================================================================
# STRUCTURED EXTRACTION SCHEMAS (Used with GenAI response_schema)
# ============================================================================


class InvoiceLineItem(BaseModel):
    """Line item in an invoice."""

    description: Optional[str] = Field(
        default=None, description="Description of the product or service"
    )
    quantity: Optional[float] = Field(
        default=None, description="Quantity billed"
    )
    unit_price: Optional[float] = Field(
        default=None, description="Price per unit"
    )
    total_price: Optional[float] = Field(
        default=None, description="Total cost for this item"
    )


class InvoiceSchema(BaseModel):
    """Structured extraction schema for Invoices and Billing Receipts."""

    vendor_name: Optional[str] = Field(
        default=None, description="Name of the company or vendor issuing the invoice"
    )
    invoice_number: Optional[str] = Field(
        default=None, description="Unique invoice identifier or code"
    )
    invoice_date: Optional[str] = Field(
        default=None, description="Date invoice was issued (YYYY-MM-DD or textual format)"
    )
    due_date: Optional[str] = Field(
        default=None, description="Payment due date"
    )
    total_amount: Optional[float] = Field(
        default=None, description="Final total amount due including taxes and discounts"
    )
    currency: Optional[str] = Field(
        default=None, description="Currency code or symbol (e.g. USD, EUR, INR)"
    )
    line_items: Optional[list[InvoiceLineItem]] = Field(
        default=None, description="List of itemized products or services"
    )
    payment_terms: Optional[str] = Field(
        default=None, description="Net terms or payment instructions"
    )


class ResumeExperience(BaseModel):
    """Professional work experience entry."""

    role: Optional[str] = Field(default=None, description="Job title or role")
    company: Optional[str] = Field(
        default=None, description="Organization or company name"
    )
    duration: Optional[str] = Field(
        default=None, description="Dates or timeframe of employment"
    )
    responsibilities: Optional[list[str]] = Field(
        default=None, description="Key duties or achievements"
    )


class ResumeEducation(BaseModel):
    """Educational qualification entry."""

    degree: Optional[str] = Field(
        default=None, description="Degree, major or qualification"
    )
    institution: Optional[str] = Field(
        default=None, description="School, university or college"
    )
    year: Optional[str] = Field(
        default=None, description="Graduation year or dates attended"
    )


class ResumeSchema(BaseModel):
    """Structured extraction schema for CVs and Resumes."""

    full_name: Optional[str] = Field(
        default=None, description="Candidate's full name"
    )
    email: Optional[str] = Field(
        default=None, description="Contact email address"
    )
    phone: Optional[str] = Field(
        default=None, description="Contact telephone number"
    )
    summary: Optional[str] = Field(
        default=None, description="Professional summary or bio"
    )
    skills: Optional[list[str]] = Field(
        default=None, description="Technical, domain, or soft skills"
    )
    experience: Optional[list[ResumeExperience]] = Field(
        default=None, description="Work history"
    )
    education: Optional[list[ResumeEducation]] = Field(
        default=None, description="Academic background"
    )
    certifications: Optional[list[str]] = Field(
        default=None, description="Professional licenses or certifications"
    )


class ResearchPaperSchema(BaseModel):
    """Structured extraction schema for Scientific / Academic Papers."""

    title: Optional[str] = Field(
        default=None, description="Full paper title"
    )
    authors: Optional[list[str]] = Field(
        default=None, description="List of authors and affiliations"
    )
    abstract: Optional[str] = Field(
        default=None, description="Summary abstract of the paper"
    )
    methodology: Optional[str] = Field(
        default=None, description="Research methodology or experimental setup"
    )
    key_findings: Optional[list[str]] = Field(
        default=None, description="Primary empirical or theoretical findings"
    )
    conclusions: Optional[str] = Field(
        default=None, description="Final conclusions or future work"
    )
    references_count: Optional[int] = Field(
        default=None, description="Approximate number of citations/references"
    )


class ContractSchema(BaseModel):
    """Structured extraction schema for Legal Contracts & Agreements."""

    contract_title: Optional[str] = Field(
        default=None, description="Title or nature of the agreement"
    )
    parties_involved: Optional[list[str]] = Field(
        default=None, description="Legal entities or individuals entering the contract"
    )
    effective_date: Optional[str] = Field(
        default=None, description="Start date of agreement validity"
    )
    expiration_date: Optional[str] = Field(
        default=None, description="Termination or expiration date"
    )
    key_obligations: Optional[list[str]] = Field(
        default=None, description="Core deliverables or promises"
    )
    termination_clauses: Optional[str] = Field(
        default=None, description="Conditions for termination or breach"
    )
    governing_law: Optional[str] = Field(
        default=None, description="Jurisdiction or governing state/country law"
    )


DocumentCategory = Literal[
    "Research Paper",
    "Business Report",
    "Invoice",
    "Resume",
    "Legal Contract",
    "Financial Statement",
    "Technical Documentation",
    "General Document",
]

CertaintyLevel = Literal["High", "Medium", "Low"]


class ClassificationResult(BaseModel):
    """Document classification output schema."""

    category: DocumentCategory = Field(
        description="Identified category of document"
    )
    certainty_level: CertaintyLevel = Field(
        description="Qualitative confidence rating (High, Medium, or Low)"
    )
    explanation: str = Field(
        description="Brief reasoning highlighting structural and lexical clues"
    )
    keywords: list[str] = Field(
        description="Salient domain keywords identified in the document"
    )


class ComparisonReport(BaseModel):
    """Structured multi-document comparative analysis."""

    similarities: list[str] = Field(
        description="Shared themes, common claims, or overlapping topics"
    )
    differences: list[str] = Field(
        description="Diverging perspectives, unique sections, or disparate scopes"
    )
    conflicting_information: list[str] = Field(
        description="Direct contradictions or numerical discrepancies between documents"
    )
    common_themes: list[str] = Field(
        description="Underlying core subjects tying the documents together"
    )
    summary_verdict: str = Field(
        description="Executive synthesis comparing the overall value, focus, or authority"
    )


# ============================================================================
# INTERNAL DOCUMENT & CHUNK MODELS
# ============================================================================


class DocumentPage(BaseModel):
    """Represents text extracted from a specific page."""

    page_number: int
    text: str


class DocumentChunk(BaseModel):
    """Text chunk enriched with provenance metadata for RAG."""

    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str


class DocumentMetadata(BaseModel):
    """Client-facing metadata for an uploaded document."""

    doc_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    preview_snippet: str
    uploaded_at: str


class SessionStats(BaseModel):
    """Aggregate statistics for an active user session."""

    total_documents: int
    total_chunks: int
    total_pages: int


class DocumentsListResponse(BaseModel):
    """Response returned when fetching documents in a session."""

    session_id: str
    documents: list[DocumentMetadata]
    stats: SessionStats


# ============================================================================
# API ENDPOINT REQUEST & RESPONSE SCHEMAS
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    genai_ready: bool
    active_model: str
    active_embedding_model: str
    version: str = "1.0.0"


class ClassifyRequest(BaseModel):
    """Request model for document classification."""

    doc_id: str


class ClassifyResponse(BaseModel):
    """Response model for document classification."""

    doc_id: str
    classification: ClassificationResult


SummarizationMode = Literal["Quick", "Bullet Points", "Executive", "Detailed"]


class SummarizeRequest(BaseModel):
    """Request model for document summarization."""

    doc_id: str
    mode: SummarizationMode = "Executive"


class SummarizeResponse(BaseModel):
    """Response model for document summarization."""

    doc_id: str
    mode: SummarizationMode
    strategy_used: str = Field(
        description="Direct LLM call or Hierarchical Map-Reduce"
    )
    summary: str


class ChatRequest(BaseModel):
    """Request model for grounded RAG Q&A."""

    question: str
    doc_ids: Optional[list[str]] = Field(
        default=None, description="Optional filter: restrict Q&A to specific documents"
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of context chunks to retrieve"
    )


class Citation(BaseModel):
    """Provenance citation backing the generated RAG answer."""

    doc_id: str
    filename: str
    page_number: int
    excerpt: str
    similarity_score: float


class ChatPipelineMetadata(BaseModel):
    """GenAI pipeline telemetry for grounded Q&A."""

    model: str
    chunks_retrieved: int
    prompt_strategy: str


class ChatResponse(BaseModel):
    """Response model for grounded RAG Q&A."""

    answer: str
    citations: list[Citation]
    genai_pipeline_used: ChatPipelineMetadata


ExtractionSchemaType = Literal[
    "Auto-Detect", "Invoice", "Resume", "Research Paper", "Contract"
]


class ExtractRequest(BaseModel):
    """Request model for structured schema extraction."""

    doc_id: str
    schema_type: ExtractionSchemaType = "Auto-Detect"


class ExtractResponse(BaseModel):
    """Response model for structured schema extraction."""

    doc_id: str
    schema_used: str
    extracted_data: dict[str, Any]


class SearchRequest(BaseModel):
    """Request model for semantic vector search."""

    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    doc_ids: Optional[list[str]] = None


class SearchResultItem(BaseModel):
    """Individual chunk match from vector search."""

    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    excerpt: str
    cosine_similarity_score: float


class SearchResponse(BaseModel):
    """Response model for semantic vector search."""

    query: str
    total_results: int
    results: list[SearchResultItem]


class CompareRequest(BaseModel):
    """Request model for multi-document comparative synthesis."""

    doc_ids: list[str] = Field(
        min_length=2, description="At least two doc_ids must be provided"
    )


class CompareResponse(BaseModel):
    """Response model for multi-document comparative synthesis."""

    doc_ids: list[str]
    comparison_report: ComparisonReport


class DeleteDocumentResponse(BaseModel):
    """Response model after deleting a document."""

    doc_id: str
    message: str
    remaining_documents: int
    remaining_chunks: int
