"""共享侧边栏：持仓池 + ETFirst + 数据源检测。"""

from __future__ import annotations

import streamlit as st

from etf_data_fetcher import diagnose_connectivity, test_akshare_sources
from etfirst_client import get_config, login
from holdings_utils import (
    add_holding,
    get_holdings_pool,
    is_tradable_etf_symbol,
    remove_from_pool,
)


def render_holdings_sidebar() -> dict[str, str]:
    pool = get_holdings_pool(include_watchlist=True)
    symbols = list(pool.keys())

    with st.sidebar:
        st.markdown("### 💼 持仓池")
        new_code = st.text_input("添加 ETF", placeholder="如 515880", label_visibility="collapsed")
        target = st.radio("添加到", ["持仓", "观察池"], horizontal=True, label_visibility="collapsed")
        if st.button("➕ 添加", use_container_width=True):
            if new_code.strip():
                ok, msg = is_tradable_etf_symbol(new_code)
                if not ok:
                    st.error(msg)
                else:
                    success, result = add_holding(new_code, to_watchlist=(target == "观察池"))
                    st.success(result) if success else st.warning(result)
                    st.rerun()

        if symbols:
            rm = st.selectbox(
                "移除标的",
                [""] + symbols,
                format_func=lambda x: "选择移除..." if x == "" else f"{x} {pool.get(x, '')}",
            )
            if st.button("🗑️ 移除", use_container_width=True) and rm:
                remove_from_pool(rm)
                st.rerun()
        else:
            st.info("池子为空，请先添加 ETF")

        st.divider()
        st.markdown("### 🏦 ETFirst")
        try:
            etf_cfg = get_config()
            logged = "✅" if etf_cfg.get("logged_in") else "❌"
            key_ok = "✅" if etf_cfg.get("api_key_saved") else "❌"
            st.caption(f"登录 {logged} · Key {key_ok}")
        except Exception as exc:
            st.caption(f"ETFirst 未就绪: {exc}")
        api_key = st.text_input("API Key", type="password", key="etfirst_key_sidebar")
        if st.button("ETFirst 登录", use_container_width=True):
            if api_key.strip():
                try:
                    msg = login(api_key.strip())
                    st.success(msg)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            else:
                st.warning("请输入 API Key（首趋E指 → 我的 → ETFirst Skill）")

        st.divider()
        if st.button("🔄 检测 AkShare", use_container_width=True):
            st.session_state["api_test"] = test_akshare_sources()
            st.session_state["net_diag"] = diagnose_connectivity()

    return pool
