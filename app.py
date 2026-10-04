"""StreamETF 入口：核心双页 + 折叠更多工具。"""

from __future__ import annotations

import os
import re

import streamlit as st

from akshare_patch import install_akshare_patch

install_akshare_patch()

st.set_page_config(
    page_title="ETF持仓分析 · 策略回测",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _legacy_title(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = re.sub(r"^\d+_", "", base)
    base = re.sub(r"^99_", "", base)
    return base.replace("_", " ")


def _discover_legacy_pages() -> list[st.Page]:
    root = os.path.join(os.path.dirname(__file__), "legacy_pages")
    if not os.path.isdir(root):
        return []
    files = sorted(f for f in os.listdir(root) if f.endswith(".py"))
    pages: list[st.Page] = []
    for fname in files:
        path = os.path.join("legacy_pages", fname)
        title = _legacy_title(fname)
        icon = "📁"
        if "持仓" in title:
            icon = "💼"
        elif "动量" in title or "轮动" in title:
            icon = "🔄"
        elif "AI" in title:
            icon = "🤖"
        pages.append(st.Page(path, title=title, icon=icon))
    return pages


core_pages = [
    st.Page("views/holding_analysis.py", title="ETF持仓分析", icon="📊"),
    st.Page("views/smart_money_confidence.py", title="聪明资金信心", icon="🧠"),
    st.Page("views/market_intel.py", title="市场资讯", icon="📰"),
    st.Page("views/strategy_backtest.py", title="ETF策略回测", icon="📈"),
]

legacy_pages = _discover_legacy_pages()
nav_sections = {"核心功能": core_pages}
if legacy_pages:
    nav_sections["更多工具（展开选用）"] = legacy_pages

pg = st.navigation(nav_sections, position="sidebar")
pg.run()
