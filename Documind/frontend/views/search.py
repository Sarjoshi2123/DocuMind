"""DocuMind AI — Semantic Vector Search View."""

import streamlit as st
from components.diagrams import (
    clean_html,
    get_fig02_embedding_conveyor_svg,
    render_hero_header,
    render_empty_state,
)
from services.api_client import api_client, DocuMindAPIError


def render_search(session_id: str) -> None:
    """Renders the pure mathematical FAISS vector search workbench."""
    st.markdown(
        render_hero_header(
            title_white="Semantic Vector Search,",
            title_muted="pure angular cosine matching.",
            subtitle="Queries are embedded into the 3072-dimensional space and matched against FAISS IndexFlatIP without generative hallucination.",
        ),
        unsafe_allow_html=True,
    )

    docs_data = api_client.get_documents(session_id)
    documents = docs_data.get("documents", [])

    if not documents:
        st.markdown(
            render_empty_state(
                title="Vector Index Empty",
                description="Upload documents to construct the dense FAISS cosine index.",
            ),
            unsafe_allow_html=True,
        )
        if st.button("Upload Documents →", key="sea_goto_up", type="primary"):
            st.session_state["active_view"] = "Upload"
            st.rerun()
        return

    # Blueprint Banner
    st.markdown(
        get_fig02_embedding_conveyor_svg(
            model_name="gemini-embedding-001",
            dimension=3072,
            similarity_metric="COSINE (L2-IP)",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Search Controls
    col_q, col_k = st.columns([3, 1])

    with col_q:
        query = st.text_input(
            "Natural Language Search Query",
            placeholder="e.g. 'indemnification clauses', 'multi-head attention benchmarks', 'GPU cloud costs'",
            label_visibility="collapsed",
        )

    with col_k:
        top_k = st.slider("Results (Top-K)", min_value=1, max_value=15, value=5)

    if st.button("Execute Vector Search →", type="primary", use_container_width=True) or query:
        if not query.strip():
            st.warning("Please enter a query string to search.")
            return

        with st.spinner("Embedding query & searching FAISS index..."):
            try:
                res = api_client.search_semantic(
                    query=query,
                    session_id=session_id,
                    top_k=top_k,
                )
                results = res.get("results", [])
                st.session_state["search_results"] = results
            except DocuMindAPIError as exc:
                st.error(f"Search failed: {str(exc)}")

    # Display Ranked Search Results
    if "search_results" in st.session_state:
        results = st.session_state["search_results"]

        st.markdown(
            clean_html(f"""
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.12em; color: #8A8A8A; text-transform: uppercase; margin: 24px 0 14px 0;">
RETRIEVED {len(results)} RANKED VECTOR MATCHES
</div>
"""),
            unsafe_allow_html=True,
        )

        if not results:
            st.info("No matching chunks found above threshold.")
            return

        for idx, item in enumerate(results, 1):
            score = item.get("cosine_similarity_score", 0.0)
            pct = max(0, min(100, int(score * 100)))

            st.markdown(
                clean_html(f"""
<div class="verso-card" style="margin-bottom: 14px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
<div class="mono-tag" style="color: #FFF;">
#{idx} &bull; {item['filename']} &bull; PAGE {item['page_number']}
</div>
<div class="mono-tag" style="color: #22C55E;">
COSINE: {score:.4f} ({pct}%)
</div>
</div>
<div style="background: #1A1A1A; height: 3px; border-radius: 999px; width: 100%; margin-bottom: 12px; overflow: hidden;">
<div style="background: #FFFFFF; height: 100%; width: {pct}%;"></div>
</div>
<div style="color: #D6D6D6; font-size: 13.5px; line-height: 1.6;">
{item['excerpt']}
</div>
<div class="mono-tag" style="color: #555; margin-top: 10px;">
CHUNK ID: {item['chunk_id']}
</div>
</div>
"""),
                unsafe_allow_html=True,
            )
