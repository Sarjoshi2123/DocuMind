"""DocuMind AI — Structured Schema Extraction View."""

import json
import streamlit as st
from components.diagrams import (
    clean_html,
    render_hero_header,
    render_empty_state,
)
from services.api_client import api_client, DocuMindAPIError


def render_extract(session_id: str) -> None:
    """Renders the deterministic structured data extraction view."""
    st.markdown(
        render_hero_header(
            title_white="Structured Extraction,",
            title_muted="governed by Pydantic schemas.",
            subtitle="Constrained decoding strictly enforces schema adherence. Missing fields evaluate to null without hallucinated placeholder values.",
        ),
        unsafe_allow_html=True,
    )

    docs_data = api_client.get_documents(session_id)
    documents = docs_data.get("documents", [])

    if not documents:
        st.markdown(
            render_empty_state(
                title="No Documents Available for Extraction",
                description="Upload an invoice, resume, paper, or legal contract to parse structured schemas.",
            ),
            unsafe_allow_html=True,
        )
        if st.button("Upload Documents →", key="ext_goto_up", type="primary"):
            st.session_state["active_view"] = "Upload"
            st.rerun()
        return

    # Document & Schema Selection
    col_doc, col_schema = st.columns([1.5, 1])

    with col_doc:
        doc_options = {d["filename"]: d["doc_id"] for d in documents}
        selected_fname = st.selectbox("Target Document", options=list(doc_options.keys()), key="sel_ext_doc")
        selected_doc_id = doc_options[selected_fname]

    with col_schema:
        schema_type = st.selectbox(
            "Target Schema",
            options=["Auto-Detect", "Invoice", "Resume", "Research Paper", "Contract"],
            index=0,
            help="Select an explicit Pydantic schema or allow DocuMind to auto-classify the document first.",
        )

    if st.button("Extract Structured JSON →", type="primary", use_container_width=True):
        with st.spinner(f"Extracting fields adhering to '{schema_type}' schema..."):
            try:
                res = api_client.extract_structured(
                    doc_id=selected_doc_id,
                    schema_type=schema_type,
                    session_id=session_id,
                )
                st.session_state["last_extraction"] = {
                    "filename": selected_fname,
                    "schema_used": res.get("schema_used", schema_type),
                    "data": res.get("extracted_data", {}),
                }
            except DocuMindAPIError as exc:
                st.error(f"Extraction failed: {str(exc)}")

    # Display Extracted Spec Table
    if "last_extraction" in st.session_state:
        edata = st.session_state["last_extraction"]
        extracted_dict = edata["data"]
        schema_used = edata["schema_used"]

        st.markdown("<hr style='border-color: #1A1A1A; margin: 28px 0;'>", unsafe_allow_html=True)
        st.markdown(
            clean_html(f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.12em; color: #8A8A8A; text-transform: uppercase;">
DOCUMENT: {edata['filename']} &bull; PARSED SCHEMA: <span style="color: #FFF; font-weight: 600;">{schema_used}</span>
</div>
<div class="mono-tag" style="background: #121212; border: 1px solid #242424; padding: 3px 8px; border-radius: 999px; color: #22C55E;">
DETERMINISTIC · ZERO DRIFT
</div>
</div>
"""),
            unsafe_allow_html=True,
        )

        rows_html = []
        for k, v in extracted_dict.items():
            key_label = k.replace("_", " ").upper()
            if v is None or v == "":
                val_repr = '<span style="color: #555555; font-style: italic; font-family: monospace;">NULL</span>'
            elif isinstance(v, list):
                if not v:
                    val_repr = '<span style="color: #555555; font-style: italic; font-family: monospace;">EMPTY ARRAY []</span>'
                elif isinstance(v[0], dict):
                    sub_items = []
                    for item in v:
                        formatted_sub = ", ".join(f"{sk}: {sv}" for sk, sv in item.items() if sv is not None)
                        sub_items.append(f"&bull; {formatted_sub}")
                    val_repr = "<br>".join(sub_items)
                else:
                    val_repr = ", ".join(str(item) for item in v)
            elif isinstance(v, dict):
                val_repr = "<br>".join(f"&bull; <b>{dk}</b>: {dv}" for dk, dv in v.items() if dv is not None)
            else:
                val_repr = str(v)

            rows_html.append(
                f"""<tr style="border-bottom: 1px solid #1A1A1A;"><td style="padding: 12px 14px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #888888; width: 30%; vertical-align: top;">{key_label}</td><td style="padding: 12px 14px; font-size: 13px; color: #EDEDED; vertical-align: top;">{val_repr}</td></tr>"""
            )

        table_html = clean_html(f"""
<div style="background-color: #0A0A0A; border: 1px solid #222222; border-radius: 12px; overflow: hidden; margin-bottom: 20px;">
<table style="width: 100%; border-collapse: collapse;">
<thead>
<tr style="background: #111111; border-bottom: 1px solid #262626;">
<th style="padding: 10px 14px; text-align: left; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #666; letter-spacing: 0.12em;">SCHEMA FIELD</th>
<th style="padding: 10px 14px; text-align: left; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #666; letter-spacing: 0.12em;">EXTRACTED VALUE</th>
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</div>
""")
        st.markdown(table_html, unsafe_allow_html=True)

        with st.expander("View Raw JSON Schema Tree"):
            st.json(extracted_dict)

        json_str = json.dumps(extracted_dict, indent=2)
        st.download_button(
            label="Download JSON (.json)",
            data=json_str,
            file_name=f"extracted_{edata['filename'].replace(' ', '_')}.json",
            mime="application/json",
        )
