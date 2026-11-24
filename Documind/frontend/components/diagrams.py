"""Verso Industrial Monochrome Isometric SVG Technical Blueprints & UI Helpers.

Provides:
- FIG 01: Vector Storage Cabinet Schematic SVG
- FIG 02: Chunk & Embedding Pipeline Schematic SVG
- Reusable HTML generators for cards, headers, and instrument badges
- clean_html helper ensuring Streamlit never renders indented tags as code blocks
"""


def clean_html(html_str: str) -> str:
    """Strips leading and trailing whitespace from every line.

    Prevents Markdown parsers from treating lines with 4+ spaces of indentation
    as indented code blocks.
    """
    return "\n".join(line.strip() for line in html_str.splitlines() if line.strip())


def get_fig01_vector_cabinet_svg(
    doc_count: int = 0,
    chunk_count: int = 0,
    status_label: str = "INDEXED",
) -> str:
    """Returns an architectural isometric 3-drawer vector storage cabinet SVG."""
    raw = f"""
<div class="diagram-card">
<div class="diagram-label">FIG 01 &mdash; VECTOR STORAGE CABINET // SESSION REGISTRY</div>
<svg viewBox="0 0 540 280" width="100%" height="220" xmlns="http://www.w3.org/2000/svg" style="display:block; margin: 10px auto 0 auto; max-width: 100%;">
<defs>
<linearGradient id="gradTop" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#1A1A1A"/>
<stop offset="100%" stop-color="#0E0E0E"/>
</linearGradient>
<linearGradient id="gradLeft" x1="0%" y1="0%" x2="0%" y2="100%">
<stop offset="0%" stop-color="#141414"/>
<stop offset="100%" stop-color="#080808"/>
</linearGradient>
<linearGradient id="gradRight" x1="0%" y1="0%" x2="0%" y2="100%">
<stop offset="0%" stop-color="#1E1E1E"/>
<stop offset="100%" stop-color="#0C0C0C"/>
</linearGradient>
</defs>
<g opacity="0.15" stroke="#444" stroke-width="0.5">
<line x1="20" y1="40" x2="520" y2="40" />
<line x1="20" y1="140" x2="520" y2="140" />
<line x1="20" y1="240" x2="520" y2="240" />
<line x1="140" y1="20" x2="140" y2="260" />
<line x1="380" y1="20" x2="380" y2="260" />
</g>
<polygon points="270,30 430,95 270,160 110,95" fill="url(#gradTop)" stroke="#4A4A4A" stroke-width="1.2"/>
<polygon points="110,95 270,160 270,250 110,185" fill="url(#gradLeft)" stroke="#383838" stroke-width="1.2"/>
<polygon points="270,160 430,95 430,185 270,250" fill="url(#gradRight)" stroke="#383838" stroke-width="1.2"/>
<polygon points="270,160 415,102 415,127 270,185" fill="#141414" stroke="#666" stroke-width="0.8"/>
<line x1="335" y1="138" x2="360" y2="128" stroke="#FFF" stroke-width="1.8" stroke-linecap="round"/>
<circle cx="347" cy="133" r="1.5" fill="#FFF"/>
<polygon points="270,192 415,134 415,159 270,217" fill="#111111" stroke="#666" stroke-width="0.8"/>
<line x1="335" y1="170" x2="360" y2="160" stroke="#FFF" stroke-width="1.8" stroke-linecap="round"/>
<circle cx="347" cy="165" r="1.5" fill="#FFF"/>
<polygon points="270,224 415,166 415,191 270,249" fill="#0E0E0E" stroke="#666" stroke-width="0.8"/>
<line x1="335" y1="202" x2="360" y2="192" stroke="#FFF" stroke-width="1.8" stroke-linecap="round"/>
<circle cx="347" cy="197" r="1.5" fill="#FFF"/>
<polygon points="180,65 240,40 290,60 230,85" fill="none" stroke="#A3A3A3" stroke-dasharray="3,2" stroke-width="1"/>
<line x1="230" y1="85" x2="230" y2="120" stroke="#FFF" stroke-width="1.2"/>
<text x="40" y="70" fill="#777" font-family="'JetBrains Mono', monospace" font-size="10" letter-spacing="1">REGISTERED DOCS</text>
<text x="40" y="88" fill="#FFF" font-family="'Inter', sans-serif" font-weight="700" font-size="16">{doc_count} FILES</text>
<text x="40" y="145" fill="#777" font-family="'JetBrains Mono', monospace" font-size="10" letter-spacing="1">INDEXED CHUNKS</text>
<text x="40" y="163" fill="#FFF" font-family="'Inter', sans-serif" font-weight="700" font-size="16">{chunk_count} VECTORS</text>
<text x="445" y="125" fill="#555" font-family="'JetBrains Mono', monospace" font-size="9" letter-spacing="1.5">ISOLATION</text>
<text x="445" y="139" fill="#888" font-family="'JetBrains Mono', monospace" font-size="10">THREAD-SAFE</text>
<text x="445" y="175" fill="#555" font-family="'JetBrains Mono', monospace" font-size="9" letter-spacing="1.5">STORE ENGINE</text>
<text x="445" y="189" fill="#22C55E" font-family="'JetBrains Mono', monospace" font-size="10">● {status_label}</text>
</svg>
</div>
"""
    return clean_html(raw)


def get_fig02_embedding_conveyor_svg(
    model_name: str = "gemini-embedding-001",
    dimension: int = 3072,
    similarity_metric: str = "COSINE (L2-IP)",
) -> str:
    """Returns an architectural isometric conveyor & embedding pipeline schematic SVG."""
    raw = f"""
<div class="diagram-card">
<div class="diagram-label">FIG 02 &mdash; CHUNK & EMBEDDING LINE // RAG INGESTION BUS</div>
<svg viewBox="0 0 540 280" width="100%" height="220" xmlns="http://www.w3.org/2000/svg" style="display:block; margin: 10px auto 0 auto; max-width: 100%;">
<g opacity="0.12" stroke="#444" stroke-width="0.5">
<line x1="20" y1="50" x2="520" y2="50" />
<line x1="20" y1="150" x2="520" y2="150" />
<line x1="20" y1="230" x2="520" y2="230" />
</g>
<polygon points="60,160 480,160 420,210 0,210" fill="#0C0C0C" stroke="#333333" stroke-width="1"/>
<line x1="60" y1="160" x2="60" y2="230" stroke="#262626" stroke-width="1.5"/>
<line x1="200" y1="160" x2="200" y2="230" stroke="#262626" stroke-width="1.5"/>
<line x1="340" y1="160" x2="340" y2="230" stroke="#262626" stroke-width="1.5"/>
<line x1="480" y1="160" x2="480" y2="230" stroke="#262626" stroke-width="1.5"/>
<polygon points="80,130 130,110 170,130 120,150" fill="#141414" stroke="#555" stroke-width="1"/>
<polygon points="80,130 120,150 120,175 80,155" fill="#0F0F0F" stroke="#444" stroke-width="1"/>
<polygon points="120,150 170,130 170,155 120,175" fill="#1A1A1A" stroke="#444" stroke-width="1"/>
<text x="125" y="105" text-anchor="middle" fill="#888" font-family="'JetBrains Mono', monospace" font-size="9">PAGE TEXT</text>
<path d="M 180,140 L 220,140" stroke="#666" stroke-width="1" stroke-dasharray="2,2"/>
<polygon points="230,125 275,107 315,125 270,143" fill="#181818" stroke="#777" stroke-width="1"/>
<polygon points="230,125 270,143 270,165 230,147" fill="#121212" stroke="#555" stroke-width="1"/>
<polygon points="270,143 315,125 315,147 270,165" fill="#202020" stroke="#555" stroke-width="1"/>
<line x1="250" y1="134" x2="250" y2="156" stroke="#FFF" stroke-width="0.8" opacity="0.6"/>
<line x1="290" y1="134" x2="290" y2="156" stroke="#FFF" stroke-width="0.8" opacity="0.6"/>
<text x="272" y="98" text-anchor="middle" fill="#AAA" font-family="'JetBrains Mono', monospace" font-size="9">1000ch / 200ov</text>
<path d="M 325,140 L 365,140" stroke="#666" stroke-width="1" stroke-dasharray="2,2"/>
<polygon points="375,120 425,100 465,120 415,140" fill="#222222" stroke="#FFF" stroke-width="1.2"/>
<polygon points="375,120 415,140 415,165 375,145" fill="#161616" stroke="#999" stroke-width="1"/>
<polygon points="415,140 465,120 465,145 415,165" fill="#2A2A2A" stroke="#999" stroke-width="1"/>
<line x1="420" y1="100" x2="420" y2="45" stroke="#FFFFFF" stroke-width="1.5" stroke-dasharray="4,2"/>
<circle cx="420" cy="45" r="3" fill="#FFFFFF"/>
<text x="420" y="32" text-anchor="middle" fill="#FFFFFF" font-family="'JetBrains Mono', monospace" font-size="10" font-weight="700">d={dimension}</text>
<text x="60" y="250" fill="#777" font-family="'JetBrains Mono', monospace" font-size="10">EMBEDDER: <tspan fill="#FFF">{model_name}</tspan></text>
<text x="320" y="250" fill="#777" font-family="'JetBrains Mono', monospace" font-size="10">METRIC: <tspan fill="#22C55E">{similarity_metric}</tspan></text>
</svg>
</div>
"""
    return clean_html(raw)


def render_hero_header(
    title_white: str,
    title_muted: str,
    subtitle: str = "",
) -> str:
    """Renders a Verso-styled two-tone headline."""
    sub_html = f'<div class="hero-subhead">{subtitle}</div>' if subtitle else ""
    raw = f"""
<div style="margin-bottom: 24px;">
<h1 class="hero-headline">
<span class="hero-white">{title_white}</span> 
<span class="hero-muted">{title_muted}</span>
</h1>
{sub_html}
</div>
"""
    return clean_html(raw)


def render_metric_card(
    label: str,
    value: str | int,
    hint: str = "",
) -> str:
    """Renders a dark monochrome instrument KPI card."""
    hint_html = f'<div class="verso-card-hint">{hint}</div>' if hint else ""
    raw = f"""
<div class="verso-card">
<div class="verso-card-header">
<span class="verso-card-title">{label}</span>
<span class="mono-tag" style="color: #444;">●</span>
</div>
<div class="verso-card-value">{value}</div>
{hint_html}
</div>
"""
    return clean_html(raw)


def render_empty_state(
    title: str = "No Documents Indexed",
    description: str = "Upload PDF, DOCX, or TXT documents to initiate RAG indexing, classification, and vector search.",
) -> str:
    """Renders a sleek Verso empty-state card."""
    raw = f"""
<div class="empty-state-card">
<div style="font-size: 32px; margin-bottom: 8px;">◈</div>
<div class="empty-state-title">{title}</div>
<div class="empty-state-desc">{description}</div>
</div>
"""
    return clean_html(raw)
