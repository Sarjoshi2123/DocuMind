"""DocuMind AI — Workspace Overview Dashboard View."""

import streamlit as st
import plotly.graph_objects as go
from components.diagrams import (
    clean_html,
    get_fig01_vector_cabinet_svg,
    get_fig02_embedding_conveyor_svg,
    render_hero_header,
    render_metric_card,
    render_empty_state,
)
from services.api_client import api_client


def render_dashboard(session_id: str) -> None:
    """Renders the central workspace instrument bench."""
    # Two-tone Verso Hero Header
    st.markdown(
        render_hero_header(
            title_white="Every document,",
            title_muted="on a bench of its own.",
            subtitle="Real-time in-memory vector intelligence, grounded multi-hop RAG, and deterministic structured extraction.",
        ),
        unsafe_allow_html=True,
    )

    # Fetch document registry & stats from backend
    data = api_client.get_documents(session_id=session_id)
    documents = data.get("documents", [])
    stats = data.get("stats", {})
    total_docs = stats.get("total_documents", len(documents))
    total_chunks = stats.get("total_chunks", 0)
    total_pages = stats.get("total_pages", 0)

    # Top Row: 4 Metric Instrument Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            render_metric_card("ACTIVE FILES", total_docs, "PDF · DOCX · TXT"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            render_metric_card("INDEXED CHUNKS", total_chunks, "1000ch / 200ov windows"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            render_metric_card("TOTAL PAGES", total_pages, "Exact page boundaries"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            render_metric_card("FAISS ENGINE", "INDEX FLAT IP", "Cosine metric (L2 norm)"),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Middle Row: Isometric Vector Cabinet (FIG 01) & Chunk Conveyor (FIG 02)
    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        st.markdown(
            get_fig01_vector_cabinet_svg(
                doc_count=total_docs,
                chunk_count=total_chunks,
                status_label="ONLINE" if total_docs > 0 else "IDLE",
            ),
            unsafe_allow_html=True,
        )
    with diag_col2:
        st.markdown(
            get_fig02_embedding_conveyor_svg(
                model_name="gemini-embedding-001",
                dimension=3072,
                similarity_metric="COSINE (L2-IP)",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    if total_docs == 0:
        st.markdown(
            render_empty_state(
                title="No Documents Registered in Session",
                description="Upload research papers, corporate contracts, invoices, or resumes to unlock grounded Q&A and analytics.",
            ),
            unsafe_allow_html=True,
        )
        if st.button("Upload Documents →", key="dash_goto_upload", type="primary"):
            st.session_state["active_view"] = "Upload"
            st.rerun()
        return

    # Visual Analytics & Specification Table
    col_chart, col_table = st.columns([1, 1.2])

    with col_chart:
        st.markdown(
            clean_html("""
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.12em; color: #8A8A8A; text-transform: uppercase; margin-bottom: 12px;">
FIGURE 03 &mdash; CHUNK DENSITY PER SPECIFICATION
</div>
"""),
            unsafe_allow_html=True,
        )
        # Monochrome Plotly Bar Chart
        file_names = [d["filename"][:18] + ("…" if len(d["filename"]) > 18 else "") for d in documents]
        chunk_counts = [d["chunk_count"] for d in documents]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=file_names,
                    y=chunk_counts,
                    marker=dict(
                        color="#E5E5E5",
                        line=dict(color="#FFFFFF", width=1),
                    ),
                    hoverinfo="x+y",
                )
            ]
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=10, b=30),
            height=240,
            xaxis=dict(
                tickfont=dict(family="JetBrains Mono", size=10, color="#888888"),
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                tickfont=dict(family="JetBrains Mono", size=10, color="#888888"),
                showgrid=True,
                gridcolor="#1F1F1F",
                zeroline=False,
            ),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_table:
        st.markdown(
            clean_html("""
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.12em; color: #8A8A8A; text-transform: uppercase; margin-bottom: 12px;">
ACTIVE DOCUMENT REGISTRY
</div>
"""),
            unsafe_allow_html=True,
        )

        for doc in documents:
            size_kb = round(doc["file_size_bytes"] / 1024, 1)
            with st.expander(f"📄 {doc['filename']}  ({size_kb} KB · {doc['page_count']}p · {doc['chunk_count']} chunks)"):
                st.markdown(f"**Preview Snippet:**\n> {doc['preview_snippet']}")
                st.caption(f"Doc ID: `{doc['doc_id']}` · Uploaded: `{doc['uploaded_at']}`")
                
                c_del, _ = st.columns([1, 4])
                with c_del:
                    if st.button("Delete File", key=f"del_dash_{doc['doc_id']}", help="Removes document and rebuilds FAISS index"):
                        with st.spinner("Deleting & rebuilding vector index..."):
                            api_client.delete_document(doc["doc_id"], session_id)
                            st.success("Document deleted.")
                            st.rerun()
