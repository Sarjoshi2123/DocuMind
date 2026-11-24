"""DocuMind AI — Main Streamlit Application Entrypoint.

Verso Industrial Monochrome Pitch-Black Aesthetic.
Connects directly to FastAPI backend at http://localhost:8000.
"""

from pathlib import Path
import uuid
import streamlit as st
from components.diagrams import clean_html
from services.api_client import api_client
from views.chat import render_chat
from views.compare import render_compare
from views.dashboard import render_dashboard
from views.extract import render_extract
from views.search import render_search
from views.summarize import render_summarize
from views.upload import render_upload

# Configure Streamlit Page
st.set_page_config(
    page_title="DocuMind AI — Document Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    """Injects the Verso monochrome design system CSS."""
    css_path = Path(__file__).parent / "styles" / "main.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css()

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "active_view" not in st.session_state:
    st.session_state["active_view"] = "Dashboard"

session_id = st.session_state["session_id"]

# Query Backend Health
health = api_client.get_health()
backend_online = health.get("status") == "healthy"
genai_ready = health.get("genai_ready", False)
active_model = health.get("active_model", "gemini-3.8-flash")
active_emb = health.get("active_embedding_model", "gemini-embedding-001")

# ============================================================================
# FLOATING VERSO TOP PILL NAVBAR
# ============================================================================
status_dot_class = "status-dot-green" if backend_online else "status-dot-red"
status_text = f"LIVE &bull; {active_model.upper()}" if backend_online else "BACKEND OFFLINE"

top_pill_html = clean_html(f"""
<div class="verso-navbar">
<div class="verso-brand">
<span class="verso-brand-glyph">D</span>
<span>documind<span style="color: #666;">.ai</span></span>
</div>
<div class="verso-nav-status">
<span class="{status_dot_class}"></span>
<span>{status_text}</span>
</div>
</div>
""")
st.markdown(top_pill_html, unsafe_allow_html=True)

# ============================================================================
# SLEEK VERSO LEFT SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown(
        clean_html("""
<div style="padding: 10px 0 16px 0;">
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.16em; color: #666; text-transform: uppercase;">
INSTRUMENT PANEL // V2.1
</div>
<div style="font-size: 18px; font-weight: 700; color: #FFF; letter-spacing: -0.02em; margin-top: 2px;">
DocuMind Core
</div>
</div>
"""),
        unsafe_allow_html=True,
    )

    # Navigation Menu
    nav_options = [
        "Dashboard",
        "Upload",
        "Chat",
        "Summarize",
        "Extract",
        "Search",
        "Compare",
    ]

    nav_labels = {
        "Dashboard": "◈  Workspace Overview",
        "Upload": "⇪  Document Ingestion",
        "Chat": "💬  Grounded RAG Terminal",
        "Summarize": "📝  Adaptive Summarization",
        "Extract": "⌗  Structured Extraction",
        "Search": "⌕  Semantic Vector Search",
        "Compare": "⚖  Multi-Doc Comparison",
    }

    current_idx = nav_options.index(st.session_state["active_view"]) if st.session_state["active_view"] in nav_options else 0

    selected_view = st.radio(
        "Navigation",
        options=nav_options,
        format_func=lambda x: nav_labels[x],
        index=current_idx,
        label_visibility="collapsed",
    )

    st.session_state["active_view"] = selected_view

    st.markdown("<hr>", unsafe_allow_html=True)

    # Session Management Card
    short_session = session_id[:8]
    st.markdown(
        clean_html(f"""
<div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #666; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px;">
SESSION ISOLATION
</div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #AAA; background: #111; padding: 6px 10px; border-radius: 8px; border: 1px solid #222; margin-bottom: 10px;">
ID: {short_session}...
</div>
"""),
        unsafe_allow_html=True,
    )

    if st.button("New Session ↺", use_container_width=True, help="Resets session and vector memory"):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state.pop("rag_chat_messages", None)
        st.session_state.pop("last_summary", None)
        st.session_state.pop("last_extraction", None)
        st.session_state.pop("search_results", None)
        st.session_state.pop("comparison_report", None)
        st.success("Session reset.")
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # System Status Monitor
    st.markdown(
        clean_html(f"""
<div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #666; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px;">
TELEMETRY &amp; HARDWARE
</div>
<div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #888; line-height: 1.6;">
&bull; BACKEND: <span style="color: {'#22C55E' if backend_online else '#EF4444'};">{'ONLINE' if backend_online else 'OFFLINE'}</span><br>
&bull; GENAI: <span style="color: {'#22C55E' if genai_ready else '#EF4444'};">{'READY' if genai_ready else 'NO KEY'}</span><br>
&bull; MODEL: <span style="color: #FFF;">{active_model}</span><br>
&bull; EMBED: <span style="color: #FFF;">{active_emb}</span><br>
&bull; INDEX: <span style="color: #FFF;">FAISS FlatIP</span>
</div>
"""),
        unsafe_allow_html=True,
    )

# ============================================================================
# VIEW ROUTER
# ============================================================================
if selected_view == "Dashboard":
    render_dashboard(session_id=session_id)
elif selected_view == "Upload":
    render_upload(session_id=session_id)
elif selected_view == "Chat":
    render_chat(session_id=session_id)
elif selected_view == "Summarize":
    render_summarize(session_id=session_id)
elif selected_view == "Extract":
    render_extract(session_id=session_id)
elif selected_view == "Search":
    render_search(session_id=session_id)
elif selected_view == "Compare":
    render_compare(session_id=session_id)
