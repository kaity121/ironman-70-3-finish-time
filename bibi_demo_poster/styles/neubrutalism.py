"""
Neubrutalism + Sports Analytics Editorial Design System for IRONMAN 70.3 Demo.

Exact Color Palette:
- Main background: #FFFDF5 (Warm Cream)
- Primary text: #000000
- Borders: #000000 (2-3px solid black)
- C0 — Before Race: #FFD23F (Yellow)
- C1 — After Swim: #74B9FF (Sky Blue)
- C2 — After Bike: #88D498 (Mint Green)
- Secondary accent: #FF6B6B (Coral Red)

Hard Shadows:
- Standard card: 5px 5px 0px #000000
- Main prediction card: 7px 7px 0px #000000
- Small elements: 3px 3px 0px #000000
- Zero shadow blur, sharp square corners (border-radius: 0px)
"""

NEUBRUTAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:ital,wght@0,400..800;1,400..800&family=Space+Grotesk:wght@600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-cream: #FFFDF5;
    --text-black: #000000;
    --border-black: #000000;
    --c0-yellow: #FFD23F;
    --c1-blue: #74B9FF;
    --c2-mint: #88D498;
    --accent-coral: #FF6B6B;
    --surface-white: #FFFFFF;
    --font-heading: 'Be Vietnam Pro', 'Space Grotesk', 'Segoe UI', 'Arial', sans-serif;
    --font-body: 'Be Vietnam Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-mono: 'Space Mono', 'Be Vietnam Pro', 'Consolas', 'Courier New', monospace;
}

/* Global App Background & Typography */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: var(--bg-cream) !important;
    color: var(--text-black) !important;
    font-family: var(--font-body) !important;
}

[data-testid="stHeader"] {
    background-color: var(--bg-cream) !important;
    border-bottom: 3px solid #000000 !important;
}

.block-container {
    max-width: 1360px !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}

h1, h2, h3, h4, .nb-heading {
    font-family: var(--font-heading) !important;
    color: #000000 !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

/* Hide default Streamlit chrome clutter */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Neubrutalist Cards */
.nb-card {
    background-color: #FFFFFF;
    border: 3px solid #000000;
    box-shadow: 5px 5px 0px #000000;
    border-radius: 0px !important;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    color: #000000;
    box-sizing: border-box !important;
    max-width: 100% !important;
    overflow: hidden;
}

.nb-card-cream {
    background-color: #FFFDF5;
    border: 3px solid #000000;
    box-shadow: 5px 5px 0px #000000;
    border-radius: 0px !important;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    color: #000000;
}

.nb-hero-card {
    border: 3px solid #000000;
    box-shadow: 7px 7px 0px #000000;
    border-radius: 0px !important;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    color: #000000;
    position: relative;
}

.nb-hero-C0 {
    background-color: var(--c0-yellow);
}
.nb-hero-C1 {
    background-color: var(--c1-blue);
}
.nb-hero-C2 {
    background-color: var(--c2-mint);
}

/* Badges & Technical Metadata Pills (Square Edges) */
.nb-badge {
    display: inline-block;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    padding: 0.25rem 0.65rem;
    border: 2px solid #000000;
    box-shadow: 3px 3px 0px #000000;
    border-radius: 0px !important;
    background-color: #FFFFFF;
    color: #000000;
    margin-right: 0.5rem;
    margin-bottom: 0.4rem;
}

.nb-badge-c0 { background-color: var(--c0-yellow); }
.nb-badge-c1 { background-color: var(--c1-blue); }
.nb-badge-c2 { background-color: var(--c2-mint); }
.nb-badge-accent { background-color: var(--accent-coral); color: #000000; }
.nb-badge-black { background-color: #000000; color: #FFFDF5; box-shadow: 3px 3px 0px #FFD23F; }

/* Compact Technical Note Box for Tree Split-Border Saturation */
.nb-saturation-note {
    background-color: var(--c0-yellow);
    border: 2px solid #000000;
    box-shadow: 3px 3px 0px #000000;
    border-radius: 0px !important;
    padding: 0.45rem 0.7rem;
    margin-top: 0.25rem;
    margin-bottom: 0.55rem;
    font-family: var(--font-mono) !important;
    font-size: 0.76rem;
    line-height: 1.38;
    color: #000000;
}

/* Stale / Pending Calculation Status Banner */
.nb-stale-banner {
    background-color: var(--accent-coral);
    border: 3px solid #000000;
    box-shadow: 5px 5px 0px #000000;
    border-radius: 0px !important;
    padding: 0.85rem 1.2rem;
    margin-bottom: 1.15rem;
    font-family: var(--font-mono) !important;
    font-size: 0.90rem;
    font-weight: 700;
    color: #000000;
}

.nb-pending-box {
    background-color: #FFFDF5;
    border: 3px dashed #000000;
    box-shadow: 5px 5px 0px #000000;
    border-radius: 0px !important;
    padding: 1.75rem 1.5rem;
    margin-bottom: 1.25rem;
    text-align: center;
    color: #000000;
}

/* Header Banner */
.nb-header-box {
    background-color: #FFFFFF;
    border: 3px solid #000000;
    box-shadow: 7px 7px 0px #000000;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.5rem;
}

.nb-header-top-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.25rem;
    flex-wrap: wrap;
    border-bottom: 3px solid #000000;
    padding-bottom: 0.9rem;
    margin-bottom: 0.9rem;
}

.nb-logo-brand-wrap {
    display: flex;
    align-items: center;
    gap: 1.1rem;
    flex-wrap: wrap;
}

.nb-official-logo-box {
    background-color: #FFFDF5;
    border: 2px solid #000000;
    box-shadow: 3px 3px 0px #000000;
    padding: 0.45rem 0.85rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

.nb-official-logo-box svg,
.nb-official-logo-box img {
    height: 36px;
    width: auto;
    display: block;
}

.nb-title-main {
    font-family: var(--font-heading) !important;
    font-size: 2.15rem;
    font-weight: 800;
    line-height: 1.15;
    text-transform: uppercase;
    margin: 0.25rem 0 0.5rem 0;
    color: #000000;
}

.nb-subtitle {
    font-family: var(--font-body) !important;
    font-size: 1.05rem;
    font-weight: 500;
    color: #000000;
    margin: 0;
}

/* Checkpoint Stepper Strip */
.nb-stepper-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.25rem;
}

.nb-step-box {
    border: 3px solid #000000;
    padding: 0.9rem 1.1rem;
    background-color: #FFFFFF;
    box-shadow: 3px 3px 0px #000000;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}

.nb-step-active-C0 {
    background-color: var(--c0-yellow);
    box-shadow: 6px 6px 0px #000000;
    transform: translate(-2px, -2px);
}
.nb-step-active-C1 {
    background-color: var(--c1-blue);
    box-shadow: 6px 6px 0px #000000;
    transform: translate(-2px, -2px);
}
.nb-step-active-C2 {
    background-color: var(--c2-mint);
    box-shadow: 6px 6px 0px #000000;
    transform: translate(-2px, -2px);
}

.nb-step-code {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
}

.nb-step-title {
    font-family: var(--font-heading) !important;
    font-size: 1.2rem;
    font-weight: 700;
    margin: 0.15rem 0;
}

.nb-step-desc {
    font-family: var(--font-mono) !important;
    font-size: 0.76rem;
}

/* Giant Prediction Time Readout */
.nb-pred-label {
    font-family: var(--font-mono) !important;
    font-size: 0.88rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 2px solid #000000;
    padding-bottom: 0.35rem;
    margin-bottom: 0.65rem;
}

.nb-pred-time {
    font-family: var(--font-mono) !important;
    font-size: 3.8rem;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.03em;
    color: #000000;
    margin: 0.25rem 0 0.5rem 0;
}

.nb-pred-seconds {
    font-family: var(--font-mono) !important;
    font-size: 0.92rem;
    font-weight: 700;
    background-color: #FFFFFF;
    border: 2px solid #000000;
    padding: 0.2rem 0.6rem;
    display: inline-block;
    box-shadow: 3px 3px 0px #000000;
}

/* Race Stage Progress Track */
.nb-stage-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.85rem;
    margin-top: 0.75rem;
}

.nb-stage-item {
    border: 2px solid #000000;
    padding: 0.85rem 1rem;
    background-color: #FFFFFF;
    box-shadow: 3px 3px 0px #000000;
}

.nb-stage-done-swim {
    background-color: var(--c1-blue);
}

.nb-stage-done-bike {
    background-color: var(--c2-mint);
}

.nb-stage-pending {
    background-color: #FFFDF5;
    border-style: dashed;
}

/* Custom Neubrutalist HTML Tables (Override Streamlit Base Table styles) */
.nb-table-wrap {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: auto !important;
    box-sizing: border-box !important;
}

div[data-testid="stMarkdownContainer"] table.nb-table,
table.nb-table {
    display: table !important;
    width: 100% !important;
    max-width: 100% !important;
    border-collapse: collapse !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    background-color: #FFFFFF !important;
    border: 3px solid #000000 !important;
    margin-bottom: 0.85rem !important;
    table-layout: fixed !important;
    box-sizing: border-box !important;
}

div[data-testid="stMarkdownContainer"] table.nb-table th,
table.nb-table th {
    background-color: #000000 !important;
    color: #FFFDF5 !important;
    text-align: left !important;
    padding: 0.45rem 0.5rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    border: 2px solid #000000 !important;
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow-wrap: anywhere !important;
}

div[data-testid="stMarkdownContainer"] table.nb-table td,
table.nb-table td {
    padding: 0.45rem 0.5rem !important;
    border: 2px solid #000000 !important;
    color: #000000 !important;
    font-weight: 600 !important;
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow-wrap: anywhere !important;
}

.nb-table tr.nb-row-highlight-C0 td { background-color: var(--c0-yellow); font-weight: 700; }
.nb-table tr.nb-row-highlight-C1 td { background-color: var(--c1-blue); font-weight: 700; }
.nb-table tr.nb-row-highlight-C2 td { background-color: var(--c2-mint); font-weight: 700; }
.nb-table tr.nb-row-actual td { background-color: var(--accent-coral); font-weight: 700; }

/* Streamlit Widget Overrides: Buttons, Inputs, Selectboxes */
/* Ensure buttons and containers never fade out during Streamlit Cloud stale/rerun cycles */
[data-stale="true"],
[data-testid="stButton"],
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"] {
    opacity: 1 !important;
    visibility: visible !important;
}

div.stButton > button,
div[data-testid="stButton"] > button,
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-secondary"] {
    font-family: var(--font-mono) !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    border: 3px solid #000000 !important;
    border-radius: 0px !important;
    background-color: #FFFFFF !important;
    color: #000000 !important;
    box-shadow: 4px 4px 0px #000000 !important;
    padding: 0.55rem 1rem !important;
    transition: transform 0.1s ease, box-shadow 0.1s ease !important;
    width: 100%;
    opacity: 1 !important;
    visibility: visible !important;
}

div.stButton > button:hover,
div[data-testid="stButton"] > button:hover,
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover {
    transform: translate(-2px, -2px) !important;
    box-shadow: 6px 6px 0px #000000 !important;
    background-color: var(--c0-yellow) !important;
    color: #000000 !important;
    opacity: 1 !important;
}

div.stButton > button:active,
div[data-testid="stButton"] > button:active,
button[data-testid="stBaseButton-primary"]:active,
button[data-testid="stBaseButton-secondary"]:active {
    transform: translate(3px, 3px) !important;
    box-shadow: 0px 0px 0px #000000 !important;
    opacity: 1 !important;
}

div.stButton > button[kind="primary"],
div[data-testid="stButton"] > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background-color: var(--accent-coral) !important;
    color: #000000 !important;
    opacity: 1 !important;
}

div.stButton > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background-color: var(--c0-yellow) !important;
    color: #000000 !important;
    opacity: 1 !important;
}

/* Input boxes & select boxes */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stNumberInputContainer"] {
    border: 2px solid #000000 !important;
    border-radius: 0px !important;
    background-color: #FFFFFF !important;
    box-shadow: 3px 3px 0px #000000 !important;
    color: #000000 !important;
}

label, [data-testid="stWidgetLabel"] p {
    font-family: var(--font-mono) !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    color: #000000 !important;
}

/* Section Headers */
.nb-section-header {
    font-family: var(--font-heading) !important;
    font-size: 1.35rem;
    font-weight: 800;
    text-transform: uppercase;
    border-bottom: 3px solid #000000;
    padding-bottom: 0.4rem;
    margin-top: 0.25rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
</style>
"""
