"""DocuMind AI — Document Ingestion Bench View."""

import streamlit as st
from components.diagrams import (
    clean_html,
    get_fig01_vector_cabinet_svg,
    render_hero_header,
)
from services.api_client import api_client, DocuMindAPIError


def render_upload(session_id: str) -> None:
    """Renders the document upload and parsing workbench."""
    st.markdown(
        render_hero_header(
            title_white="Document Ingestion Bench,",
            title_muted="streamed to vector memory.",
            subtitle="Drop PDF, DOCX, or TXT documents. In-memory PyMuPDF and python-docx parse exact page boundaries, slice overlapping chunks, and embed via Gemini.",
        ),
        unsafe_allow_html=True,
    )

    col_upload, col_diagram = st.columns([1.2, 1])

    with col_diagram:
        docs_data = api_client.get_documents(session_id)
        current_docs = docs_data.get("documents", [])
        st.markdown(
            get_fig01_vector_cabinet_svg(
                doc_count=len(current_docs),
                chunk_count=docs_data.get("stats", {}).get("total_chunks", 0),
                status_label="ACTIVE BUS",
            ),
            unsafe_allow_html=True,
        )

    with col_upload:
        st.markdown(
            clean_html("""
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.12em; color: #8A8A8A; text-transform: uppercase; margin-bottom: 8px;">
SLOT INGRESS &mdash; UP TO 15MB PER PAYLOAD
</div>
"""),
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Select or drop documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            if st.button("Process & Ingest Payloads →", type="primary", use_container_width=True):
                files_payload: list[tuple[str, bytes, str]] = []
                for uf in uploaded_files:
                    fbytes = uf.read()
                    mtype = uf.type or "application/octet-stream"
                    files_payload.append((uf.name, fbytes, mtype))

                with st.spinner("Parsing text, computing dense embeddings & updating FAISS index..."):
                    try:
                        res = api_client.upload_documents(files_payload, session_id=session_id)
                        st.success(f"Successfully indexed {len(res)} document(s) into session FAISS index.")
                        st.rerun()
                    except DocuMindAPIError as exc:
                        st.error(f"Ingestion Error: {str(exc)}")

    st.markdown("<hr style='border-color: #1A1A1A; margin: 32px 0;'>", unsafe_allow_html=True)

    # Ingested Files List with Classification Badges
    st.markdown(
        clean_html("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
<div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 0.14em; color: #8A8A8A; text-transform: uppercase;">
REGISTERED SESSION DOCUMENTS &amp; LIVE CLASSIFICATION
</div>
</div>
"""),
        unsafe_allow_html=True,
    )

    docs_data = api_client.get_documents(session_id)
    documents = docs_data.get("documents", [])

    if not documents:
        st.info("No documents uploaded yet in this active session.")
        return

    for doc in documents:
        doc_id = doc["doc_id"]
        fname = doc["filename"]
        size_kb = round(doc["file_size_bytes"] / 1024, 1)

        with st.container():
            st.markdown(
                clean_html(f"""
<div class="verso-card" style="padding: 16px 20px; margin-bottom: 12px;">
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
<div>
<div style="font-weight: 600; font-size: 15px; color: #FFF;">{fname}</div>
<div class="mono-tag" style="margin-top: 4px;">{doc['file_type'].upper()} · {size_kb} KB · {doc['page_count']} PAGES · {doc['chunk_count']} CHUNKS</div>
</div>
</div>
</div>
"""),
                unsafe_allow_html=True,
            )

            col_act1, col_act2, col_act3 = st.columns([1.5, 2.5, 1])

            with col_act1:
                if st.button("Run AI Classification", key=f"btn_classify_{doc_id}"):
                    with st.spinner("Analyzing document structure & lexicons..."):
                        try:
                            cls_res = api_client.classify_document(doc_id, session_id)
                            st.session_state[f"class_{doc_id}"] = cls_res["classification"]
                        except Exception as e:
                            st.error(f"Classification failed: {e}")

            with col_act2:
                if f"class_{doc_id}" in st.session_state:
                    cinfo = st.session_state[f"class_{doc_id}"]
                    cat = cinfo.get("category", "General Document")
                    cert = cinfo.get("certainty_level", "Medium")
                    keywords = ", ".join(cinfo.get("keywords", []))
                    st.markdown(
                        clean_html(f"""
<div style="background: #141414; border: 1px solid #292929; border-radius: 8px; padding: 6px 12px; font-size: 12px;">
<span style="color: #FFF; font-weight: 600;">{cat}</span> 
<span class="mono-tag" style="color: #22C55E; margin-left: 6px;">● {cert} CERTAINTY</span>
<div style="color: #777; font-size: 11px; margin-top: 2px;">Keywords: {keywords}</div>
</div>
"""),
                        unsafe_allow_html=True,
                    )

            with col_act3:
                if st.button("Delete File", key=f"del_up_{doc_id}"):
                    api_client.delete_document(doc_id, session_id)
                    st.session_state.pop(f"class_{doc_id}", None)
                    st.success(f"Removed {fname}")
                    st.rerun()
