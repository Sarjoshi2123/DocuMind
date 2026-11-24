"""DocuMind AI — Grounded RAG Terminal View."""

import streamlit as st
from components.diagrams import (
    clean_html,
    render_hero_header,
    render_empty_state,
)
from services.api_client import api_client, DocuMindAPIError


def render_chat(session_id: str) -> None:
    """Renders the Grounded RAG Q&A conversational interface."""
    st.markdown(
        render_hero_header(
            title_white="Grounded RAG Terminal,",
            title_muted="cited down to the page.",
            subtitle="Conversational question-answering with strict prompt-injection defense and hallucination refusal. Every claim references verified source citations.",
        ),
        unsafe_allow_html=True,
    )

    docs_data = api_client.get_documents(session_id)
    documents = docs_data.get("documents", [])

    if not documents:
        st.markdown(
            render_empty_state(
                title="Vector Index Empty",
                description="Upload documents first to start interrogating them through Grounded RAG.",
            ),
            unsafe_allow_html=True,
        )
        if st.button("Go to Upload Bench →", type="primary"):
            st.session_state["active_view"] = "Upload"
            st.rerun()
        return

    # Top Controls Bar: Scope, Top-K Slider, & Telemetry
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1.2, 1])

    with ctrl1:
        doc_options = {"All Documents in Session": None}
        for d in documents:
            doc_options[f"{d['filename']} ({d['page_count']}p)"] = d["doc_id"]

        selected_scope_label = st.selectbox(
            "Document Scope",
            options=list(doc_options.keys()),
            index=0,
            help="Filter retrieval to a specific document or query all active files.",
        )
        selected_doc_id = doc_options[selected_scope_label]
        target_doc_ids = [selected_doc_id] if selected_doc_id else None

    with ctrl2:
        top_k = st.slider(
            "Retrieval Depth (Top-K)",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of nearest-neighbor chunks retrieved from FAISS cosine index.",
        )

    with ctrl3:
        st.markdown(
            clean_html("""
<div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #777; margin-top: 14px; text-transform: uppercase;">
GROUNDING DEFENSE<br>
<span style="color: #22C55E;">● ACTIVE DELIMITER</span>
</div>
"""),
            unsafe_allow_html=True,
        )

    # Initialize chat history in session_state
    if "rag_chat_messages" not in st.session_state:
        st.session_state["rag_chat_messages"] = [
            {
                "role": "assistant",
                "content": "DocuMind terminal initialized. Ask any factual question about your uploaded documents.",
                "citations": [],
                "telemetry": None,
            }
        ]

    # Quick Starter Prompt Pills
    st.markdown(
        clean_html("""
<div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #666; text-transform: uppercase; margin: 12px 0 6px 0;">
SUGGESTED INSTRUMENT PROMPTS:
</div>
"""),
        unsafe_allow_html=True,
    )
    s1, s2, s3, s4 = st.columns(4)
    starter_query = None

    with s1:
        if st.button("Core purpose & methodology", use_container_width=True):
            starter_query = "What is the primary objective and methodology described in the document?"
    with s2:
        if st.button("Key quantitative metrics", use_container_width=True):
            starter_query = "List all specific numbers, performance benchmarks, and quantitative metrics mentioned."
    with s3:
        if st.button("Obligations & conditions", use_container_width=True):
            starter_query = "What are the core obligations, terms, conditions, or governing rules specified?"
    with s4:
        if st.button("Authoritative conclusions", use_container_width=True):
            starter_query = "What final conclusions or strategic recommendations are documented?"

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Render Chat History
    for msg in st.session_state["rag_chat_messages"]:
        if msg["role"] == "user":
            st.markdown(
                clean_html(f"""
<div style="display: flex; justify-content: flex-end; margin-bottom: 14px;">
<div style="background-color: #171717; border: 1px solid #2B2B2B; border-radius: 14px 14px 2px 14px; padding: 12px 18px; max-width: 78%; color: #FFFFFF; font-size: 14px; line-height: 1.5;">
{msg["content"]}
</div>
</div>
"""),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                clean_html(f"""
<div style="display: flex; justify-content: flex-start; margin-bottom: 14px;">
<div style="background-color: #0E0E0E; border: 1px solid #1F1F1F; border-radius: 14px 14px 14px 2px; padding: 16px 20px; max-width: 85%; color: #E5E5E5; font-size: 14px; line-height: 1.6;">
<div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #8A8A8A; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;">
DOCUMIND INTELLIGENCE &bull; GROUNDED SYNTHESIS
</div>
{msg["content"]}
</div>
</div>
"""),
                unsafe_allow_html=True,
            )

            # Render expandable citations if present
            if msg.get("citations"):
                with st.expander(f"Verified Source Citations ({len(msg['citations'])} chunks retrieved)"):
                    for c in msg["citations"]:
                        sim_pct = round(c["similarity_score"] * 100, 1)
                        st.markdown(
                            clean_html(f"""
<div class="citation-box">
<div class="citation-header">
<span>[SOURCE: {c['filename']} &middot; PAGE {c['page_number']}]</span>
<span class="citation-score">COSINE {c['similarity_score']} ({sim_pct}%)</span>
</div>
<div style="color: #CCCCCC; font-style: italic;">&ldquo;{c['excerpt']}&rdquo;</div>
</div>
"""),
                            unsafe_allow_html=True,
                        )

            # Render telemetry badge
            if msg.get("telemetry"):
                t = msg["telemetry"]
                st.caption(f"Engine: `{t.get('model', 'gemini')}` · Chunks: `{t.get('chunks_retrieved', 0)}` · Strategy: `{t.get('prompt_strategy', 'RAG')}`")

    # Chat Input Box
    user_input = st.chat_input("Enter factual query regarding indexed documents...") or starter_query

    if user_input:
        st.session_state["rag_chat_messages"].append(
            {"role": "user", "content": user_input, "citations": [], "telemetry": None}
        )

        with st.spinner("Embedding query, traversing FAISS cosine index & generating grounded answer..."):
            try:
                res = api_client.chat_rag(
                    question=user_input,
                    session_id=session_id,
                    doc_ids=target_doc_ids,
                    top_k=top_k,
                )
                st.session_state["rag_chat_messages"].append(
                    {
                        "role": "assistant",
                        "content": res.get("answer", ""),
                        "citations": res.get("citations", []),
                        "telemetry": res.get("genai_pipeline_used", {}),
                    }
                )
            except DocuMindAPIError as exc:
                st.session_state["rag_chat_messages"].append(
                    {
                        "role": "assistant",
                        "content": f"Query Error: {str(exc)}",
                        "citations": [],
                        "telemetry": None,
                    }
                )

        st.rerun()
