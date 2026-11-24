"""DocuMind AI — Multi-Document Comparison View."""

import streamlit as st
from components.diagrams import (
    clean_html,
    render_hero_header,
    render_empty_state,
)
from services.api_client import api_client, DocuMindAPIError


def render_compare(session_id: str) -> None:
    """Renders the multi-document comparative synthesis view."""
    st.markdown(
        render_hero_header(
            title_white="Multi-Doc Comparative Synthesis,",
            title_muted="cross-cutting intelligence.",
            subtitle="Select two or more documents to contrast methodologies, discover conflicting metrics, and synthesize unified executive verdicts.",
        ),
        unsafe_allow_html=True,
    )

    docs_data = api_client.get_documents(session_id)
    documents = docs_data.get("documents", [])

    if len(documents) < 2:
        st.markdown(
            render_empty_state(
                title="Minimum 2 Documents Required",
                description="Comparative synthesis requires at least two distinct documents indexed in your session.",
            ),
            unsafe_allow_html=True,
        )
        if st.button("Upload Additional Documents →", key="cmp_goto_up", type="primary"):
            st.session_state["active_view"] = "Upload"
            st.rerun()
        return

    # Multi-Document Selector
    doc_options = {f"{d['filename']} ({d['page_count']}p)": d["doc_id"] for d in documents}
    default_selection = list(doc_options.keys())[:2]

    selected_labels = st.multiselect(
        "Select Documents for Comparison (Minimum 2)",
        options=list(doc_options.keys()),
        default=default_selection,
    )

    selected_doc_ids = [doc_options[lbl] for lbl in selected_labels]

    if st.button("Synthesize Comparative Analysis →", type="primary", use_container_width=True):
        if len(selected_doc_ids) < 2:
            st.warning("Please select at least 2 documents to compare.")
            return

        with st.spinner("Extracting representative passages & synthesizing comparative matrices..."):
            try:
                res = api_client.compare_documents(
                    doc_ids=selected_doc_ids,
                    session_id=session_id,
                )
                st.session_state["comparison_report"] = res.get("comparison_report", {})
                st.session_state["comparison_doc_count"] = len(selected_doc_ids)
            except DocuMindAPIError as exc:
                st.error(f"Comparison failed: {str(exc)}")

    # Display Comparative Report Grid
    if "comparison_report" in st.session_state:
        report = st.session_state["comparison_report"]
        kdocs = st.session_state.get("comparison_doc_count", 2)

        st.markdown("<hr style='border-color: #1A1A1A; margin: 28px 0;'>", unsafe_allow_html=True)
        st.markdown(
            clean_html(f"""
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.12em; color: #8A8A8A; text-transform: uppercase; margin-bottom: 14px;">
SYNTHESIS REPORT &bull; {kdocs} DOCUMENTS CROSS-ANALYZED
</div>
"""),
            unsafe_allow_html=True,
        )

        # 1. Executive Verdict Card
        verdict = report.get("summary_verdict", "No verdict generated.")
        st.markdown(
            clean_html(f"""
<div class="verso-card" style="border-left: 3px solid #FFFFFF;">
<div class="verso-card-title" style="color: #FFFFFF; font-weight: 700; margin-bottom: 8px;">
EXECUTIVE VERDICT // SYNTHESIS
</div>
<div style="color: #EDEDED; font-size: 14px; line-height: 1.6;">
{verdict}
</div>
</div>
"""),
            unsafe_allow_html=True,
        )

        # 2. Similarities & Differences Columns
        col_sim, col_diff = st.columns(2)

        with col_sim:
            sims = report.get("similarities", [])
            sim_items = "".join(f"<li style='margin-bottom: 8px; color: #CCCCCC;'>{s}</li>" for s in sims)
            st.markdown(
                clean_html(f"""
<div class="verso-card" style="height: 100%;">
<div class="verso-card-title" style="color: #22C55E;">
&bull; SIMILARITIES &amp; CONVERGENCES
</div>
<ul style="padding-left: 18px; margin-top: 10px; font-size: 13.5px; line-height: 1.5;">
{sim_items}
</ul>
</div>
"""),
                unsafe_allow_html=True,
            )

        with col_diff:
            diffs = report.get("differences", [])
            diff_items = "".join(f"<li style='margin-bottom: 8px; color: #CCCCCC;'>{d}</li>" for d in diffs)
            st.markdown(
                clean_html(f"""
<div class="verso-card" style="height: 100%;">
<div class="verso-card-title" style="color: #EAB308;">
&bull; DIVERGING METHODOLOGIES &amp; SCOPES
</div>
<ul style="padding-left: 18px; margin-top: 10px; font-size: 13.5px; line-height: 1.5;">
{diff_items}
</ul>
</div>
"""),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # 3. Conflicting Information & Common Themes Columns
        col_conf, col_themes = st.columns(2)

        with col_conf:
            conflicts = report.get("conflicting_information", [])
            if conflicts:
                conf_items = "".join(f"<li style='margin-bottom: 8px; color: #FFAAAA;'>{c}</li>" for c in conflicts)
            else:
                conf_items = "<li style='color: #666; font-style: italic;'>No explicit factual contradictions detected between documents.</li>"
            st.markdown(
                clean_html(f"""
<div class="verso-card" style="height: 100%;">
<div class="verso-card-title" style="color: #EF4444;">
&bull; CONFLICTING INFORMATION &amp; DISCREPANCIES
</div>
<ul style="padding-left: 18px; margin-top: 10px; font-size: 13.5px; line-height: 1.5;">
{conf_items}
</ul>
</div>
"""),
                unsafe_allow_html=True,
            )

        with col_themes:
            themes = report.get("common_themes", [])
            theme_items = "".join(f"<li style='margin-bottom: 8px; color: #CCCCCC;'>{t}</li>" for t in themes)
            st.markdown(
                clean_html(f"""
<div class="verso-card" style="height: 100%;">
<div class="verso-card-title" style="color: #38BDF8;">
&bull; CORE UNDERLYING THEMES
</div>
<ul style="padding-left: 18px; margin-top: 10px; font-size: 13.5px; line-height: 1.5;">
{theme_items}
</ul>
</div>
"""),
                unsafe_allow_html=True,
            )
