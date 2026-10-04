"""全局 UI 样式。"""

import streamlit as st


def inject_global_styles() -> None:
    st.markdown(
        """
<style>
    .block-container { padding-top: 1.2rem; max-width: 1400px; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stButton button {
        background: #334155; border: 1px solid #475569; border-radius: 8px;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 55%, #8b5cf6 100%);
        padding: 1.4rem 1.8rem; border-radius: 16px; color: white;
        margin-bottom: 1rem; box-shadow: 0 8px 24px rgba(99,102,241,.25);
    }
    .hero-banner h1 { margin: 0; font-size: 1.75rem; font-weight: 700; }
    .hero-banner p.data-asof {
        margin-top: .55rem; opacity: 1; font-size: .92rem;
        background: rgba(255,255,255,.18); display: inline-block;
        padding: .2rem .7rem; border-radius: 999px; font-weight: 600;
    }

    .verdict-card {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1rem 1.1rem; margin-bottom: .75rem;
        box-shadow: 0 2px 8px rgba(15,23,42,.06);
    }
    .verdict-card strong { color: #0f172a; }
    .tag {
        display: inline-block; padding: .15rem .55rem; border-radius: 999px;
        font-size: .78rem; font-weight: 600; margin-right: .35rem;
    }
    .tag-bull { background: #fee2e2; color: #b91c1c; }
    .tag-bear { background: #dcfce7; color: #15803d; }
    .tag-neutral { background: #f1f5f9; color: #475569; }
    .tag-warn { background: #fef3c7; color: #b45309; }
    div[data-testid="stMetric"] {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: .5rem .75rem;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", asof: str | None = None) -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    asof_html = (
        f'<p class="data-asof">数据截止日期：{asof}（未收盘不含当天）</p>' if asof else ""
    )
    st.markdown(
        f'<div class="hero-banner"><h1>{title}</h1>{sub}{asof_html}</div>',
        unsafe_allow_html=True,
    )
