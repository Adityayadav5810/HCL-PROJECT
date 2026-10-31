"""
CineMatch Cinematic Design System
---------------------------------
Dark, modern, premium UI/UX styling for Streamlit.
Provides typography, responsive movie card grids, hero spotlight banners,
rating badges, genre pills, and interactive widget enhancements.
"""

CINEMATIC_CSS = """
<style>
/* ==========================================================
   1. GLOBAL THEME VARIABLES & BASE RESETS
   ========================================================== */
:root {
    --bg-primary: #090b10;
    --bg-secondary: #111520;
    --bg-card: #151a27;
    --bg-card-hover: #1c2233;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(56, 189, 248, 0.4);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-cyan: #38bdf8;
    --accent-gold: #f59e0b;
    --accent-red: #f43f5e;
    --accent-green: #10b981;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
    --radius-xl: 20px;
}

/* Background & Main App Container */
.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

[data-testid="stMain"] {
    background: radial-gradient(circle at 85% 10%, rgba(56, 189, 248, 0.05) 0%, transparent 45%),
                radial-gradient(circle at 15% 40%, rgba(244, 63, 94, 0.03) 0%, transparent 40%),
                var(--bg-primary);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1600px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle) !important;
    margin: 1.2rem 0 !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

p, span, label {
    color: var(--text-secondary);
}

/* ==========================================================
   2. HERO SPOTLIGHT BANNER
   ========================================================== */
.hero-spotlight {
    position: relative;
    border-radius: var(--radius-lg);
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid var(--border-subtle);
    padding: clamp(24px, 4vw, 44px);
    margin: 0.5rem 0 2rem 0;
    overflow: hidden;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
}

.hero-spotlight::before {
    content: "";
    position: absolute;
    top: -50%;
    right: -20%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, transparent 65%);
    pointer-events: none;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: var(--accent-cyan);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 4px 10px;
    border-radius: 9999px;
    margin-bottom: 12px;
}

.hero-title {
    font-size: clamp(28px, 3.5vw, 44px);
    font-weight: 800;
    color: #ffffff !important;
    line-height: 1.15;
    margin-bottom: 12px;
    letter-spacing: -0.03em;
}

.hero-desc {
    color: var(--text-secondary);
    font-size: clamp(14px, 1.3vw, 17px);
    line-height: 1.6;
    max-width: 820px;
    margin-bottom: 20px;
}

/* ==========================================================
   3. SECTION HEADINGS & BADGES
   ========================================================== */
.section-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin: 28px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-subtitle {
    font-size: 13px;
    color: var(--text-muted);
}

/* Genre Pills */
.genre-pill {
    display: inline-block;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid var(--border-subtle);
    color: #cbd5e1;
    font-size: 11px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 6px;
    margin-right: 4px;
    margin-bottom: 4px;
}

/* Match Score Badges */
.match-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34d399;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 9999px;
}

.match-badge.high {
    background: rgba(16, 185, 129, 0.18);
    border-color: rgba(16, 185, 129, 0.45);
    color: #34d399;
}

.match-badge.medium {
    background: rgba(56, 189, 248, 0.18);
    border-color: rgba(56, 189, 248, 0.45);
    color: #38bdf8;
}

/* ==========================================================
   4. MOVIE CARD CONTAINERS
   ========================================================== */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 10px !important;
    transition: transform 0.22s cubic-bezier(0.4, 0, 0.2, 1),
                border-color 0.22s ease,
                box-shadow 0.22s ease !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-4px);
    border-color: var(--border-hover) !important;
    box-shadow: 0 14px 28px -10px rgba(0, 0, 0, 0.7) !important;
}

.card-title {
    font-size: 14px;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.3;
    min-height: 2.6em;
    max-height: 2.6em;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    margin: 8px 0 4px 0;
}

.card-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 8px;
}

.rating-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-weight: 700;
    color: var(--accent-gold);
}

/* ==========================================================
   5. STREAMLIT BUTTON & WIDGET ENHANCEMENTS
   ========================================================== */
div.stButton > button {
    border-radius: var(--radius-sm);
    font-weight: 600;
    font-size: 12px;
    padding: 0.35rem 0.65rem;
    transition: all 0.18s ease;
    border: 1px solid var(--border-subtle);
    background-color: rgba(255, 255, 255, 0.04);
    color: #cbd5e1;
}

div.stButton > button:hover {
    border-color: var(--accent-cyan);
    color: #ffffff;
    background-color: rgba(56, 189, 248, 0.12);
}

/* Primary Buttons */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(56, 189, 248, 0.5) !important;
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35) !important;
}

/* Input boxes & Select boxes */
div[data-baseweb="input"] {
    border-radius: var(--radius-md) !important;
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
}

div[data-baseweb="select"] > div {
    border-radius: var(--radius-md) !important;
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
}

/* Movie Poster Image */
[data-testid="stImage"] img {
    border-radius: var(--radius-sm);
    object-fit: cover;
    aspect-ratio: 2 / 3;
    width: 100%;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 14px 18px;
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 800 !important;
    font-size: 26px !important;
}

/* Movie Details Backdrop Banner */
.details-backdrop {
    width: 100%;
    height: 380px;
    border-radius: var(--radius-lg);
    background-size: cover;
    background-position: center top;
    position: relative;
    border: 1px solid var(--border-subtle);
    margin-bottom: 1.5rem;
}

.details-backdrop::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: var(--radius-lg);
    background: linear-gradient(180deg, rgba(9, 11, 16, 0.2) 0%, rgba(9, 11, 16, 0.85) 75%, var(--bg-primary) 100%);
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
    background: #262f44;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #3b4763;
}
</style>
"""


def apply_cinematic_styles():
    """Inject the cinematic stylesheet into Streamlit."""
    import streamlit as st
    st.markdown(CINEMATIC_CSS, unsafe_allow_html=True)

