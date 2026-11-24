"""DocuMind AI — Adaptive Summarization View."""

import streamlit as st
from components.diagrams import (
    clean_html,
    render_hero_header,
    render_empty_state,
)
from services.api_client import api_client, DocuMindAPIError


def render_summarize(session_id: str) -> None:
    """Renders the document summarization laboratory."""
    st.markdown(
        render_hero_header(
            title_white="Adaptive Summarization,",
            title_muted="hierarchical & zero-truncation.",
            subtitle="Documents under 14k characters synthesize in a direct pass; longer documents trigger Hierarchical Map-Reduce to eliminate context decay and preserve dense recall.",
        ),
        unsafe_allow_html=True,
    )

    docs_data = api_client.get_documents(session_id)
    documents = docs_data.get("documents", [])

    if not documents:
        st.markdown(
            render_empty_state(
                title="No Documents to Summarize",
                description="Upload papers, reports, or contracts to generate structured summaries.",
            ),
            unsafe_allow_html=True,
        )
        if st.button("Upload Documents →", key="sum_goto_up", type="primary"):
            st.session_state["active_view"] = "Upload"
            st.rerun()
        return

    # Document Selection & Mode Configuration
    col_doc, col_mode = st.columns([1.5, 1])

    with col_doc:
        doc_options = {d["filename"]: d["doc_id"] for d in documents}
        selected_fname = st.selectbox("Target Document", options=list(doc_options.keys()))
        selected_doc_id = doc_options[selected_fname]

    with col_mode:
        mode = st.selectbox(
            "Synthesis Mode",
            options=["Executive", "Bullet Points", "Quick", "Detailed"],
            index=0,
            help="Select editorial format and level of granularity.",
        )

    if st.button("Generate Summary →", type="primary", use_container_width=True):
        with st.spinner("Analyzing text volume and executing summarization pipeline..."):
            try:
                res = api_client.summarize_document(
                    doc_id=selected_doc_id,
                    mode=mode,
                    session_id=session_id,
                )
                st.session_state["last_summary"] = {
                    "filename": selected_fname,
                    "mode": res.get("mode", mode),
                    "strategy": res.get("strategy_used", "Direct LLM"),
                    "text": res.get("summary", ""),
                }
            except DocuMindAPIError as exc:
                st.error(f"Summarization failed: {str(exc)}")

    # Display Generated Summary if Available
    if "last_summary" in st.session_state:
        sdata = st.session_state["last_summary"]
        st.markdown("<hr style='border-color: #1A1A1A; margin: 28px 0;'>", unsafe_allow_html=True)

        st.markdown(
            clean_html(f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.12em; color: #8A8A8A; text-transform: uppercase;">
DOCUMENT: {sdata['filename']} &bull; MODE: {sdata['mode']}
</div>
<div class="mono-tag" style="background: #121212; border: 1px solid #242424; padding: 3px 8px; border-radius: 999px; color: #22C55E;">
STRATEGY: {sdata['strategy'].upper()}
</div>
</div>
"""),
            unsafe_allow_html=True,
        )

        st.markdown(
            clean_html(f"""
<div style="background-color: #0B0B0B; border: 1px solid #222222; border-radius: 14px; padding: 28px; line-height: 1.7; color: #EDEDED; font-size: 14px;">
{sdata['text']}
</div>
"""),
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        md_content = f"# Summary: {sdata['filename']}\n**Mode:** {sdata['mode']}  \n**Strategy:** {sdata['strategy']}\n\n---\n\n{sdata['text']}\n"
        st.download_button(
            label="Download Markdown (.md)",
            data=md_content,
            file_name=f"summary_{sdata['filename'].replace(' ', '_')}.md",
            mime="text/markdown",
        )
