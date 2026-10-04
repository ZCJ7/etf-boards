import streamlit as st

from holdings_utils import (
    add_holding,
    get_enabled_holdings,
    get_watchlist,
    is_tradable_etf_symbol,
    load_holdings_config,
    refresh_all_names,
    remove_holding,
    resolve_etf_name,
    set_holding_enabled,
)

st.set_page_config(page_title="持仓管理", page_icon="💼", layout="wide")
st.title("💼 我的 ETF 持仓管理")
st.caption("在此增删持仓与观察池标的，动量策略、轮动回测等页面会自动同步。")

config = load_holdings_config()
notes = config.get("notes", {})
if notes:
    with st.expander("说明", expanded=True):
        for code, text in notes.items():
            st.info(f"**{code}**：{text}")

col_add, col_refresh = st.columns([3, 1])
with col_add:
    st.subheader("添加标的")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        new_symbol = st.text_input("ETF 代码", placeholder="例如 515880", key="add_symbol")
    with c2:
        new_name = st.text_input("名称（可选，留空自动查询）", placeholder="自动从 AkShare 获取", key="add_name")
    with c3:
        add_target = st.selectbox("添加到", ["持仓", "观察池"], key="add_target")

    if st.button("➕ 添加", type="primary", use_container_width=True):
        if not new_symbol.strip():
            st.warning("请输入 ETF 代码")
        else:
            ok, msg = is_tradable_etf_symbol(new_symbol)
            if not ok:
                st.error(msg)
            else:
                success, result_msg = add_holding(
                    new_symbol,
                    name=new_name.strip() or None,
                    to_watchlist=(add_target == "观察池"),
                )
                if success:
                    st.success(result_msg)
                    st.rerun()
                else:
                    st.warning(result_msg)

with col_refresh:
    st.subheader("维护")
    if st.button("🔄 刷新全部名称", use_container_width=True):
        count = refresh_all_names()
        st.success(f"已更新 {count} 条名称")
        st.rerun()

st.markdown("---")

holdings = get_enabled_holdings()
all_holdings = load_holdings_config().get("holdings", [])
watchlist = get_watchlist()

left, right = st.columns(2)

with left:
    st.subheader(f"当前持仓（{len(all_holdings)}）")
    if not all_holdings:
        st.info("暂无持仓，请在上方添加。")
    for item in all_holdings:
        symbol = item["symbol"]
        name = item.get("name") or resolve_etf_name(symbol)
        enabled = item.get("enabled", True)
        box_col, btn_col1, btn_col2 = st.columns([4, 1, 1])
        with box_col:
            status = "✅" if enabled else "⏸️"
            st.markdown(f"{status} **{symbol}** — {name}")
        with btn_col1:
            if enabled:
                if st.button("暂停", key=f"pause_{symbol}"):
                    set_holding_enabled(symbol, False)
                    st.rerun()
            else:
                if st.button("启用", key=f"enable_{symbol}"):
                    set_holding_enabled(symbol, True)
                    st.rerun()
        with btn_col2:
            if st.button("删除", key=f"del_h_{symbol}"):
                remove_holding(symbol, from_watchlist=False)
                st.rerun()

with right:
    st.subheader(f"观察池（{len(watchlist)}）")
    st.caption("观察池标的会参与动量/轮动分析，但不代表实际持仓。")
    if not watchlist:
        st.info("观察池为空，可添加候选 ETF 用于轮动对比。")
    for item in watchlist:
        symbol = item["symbol"]
        name = item.get("name") or resolve_etf_name(symbol)
        box_col, btn_col = st.columns([4, 1])
        with box_col:
            st.markdown(f"👁️ **{symbol}** — {name}")
        with btn_col:
            if st.button("删除", key=f"del_w_{symbol}"):
                remove_holding(symbol, from_watchlist=True)
                st.rerun()

st.markdown("---")
st.subheader("AkShare 数据源检测")
st.caption("点击检测当前网络下各 AkShare 接口是否可用。历史 K 线若东财失败，系统会自动用新浪。")

if st.button("🔄 重新检测 AkShare 接口", type="primary"):
    from etf_data_fetcher import test_akshare_sources

    with st.spinner("正在测试 AkShare 接口..."):
        results = test_akshare_sources()
    st.dataframe(results, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("预拉取持仓数据")
if st.button("📥 立即拉取全部持仓行情"):
    from etf_data_fetcher import fetch_etf_daily
    from holdings_utils import get_enabled_holdings

    holdings = get_enabled_holdings()
    if not holdings:
        st.warning("暂无持仓")
    else:
        progress = st.progress(0)
        logs = []
        for i, item in enumerate(holdings):
            sym = item["symbol"]
            try:
                df, src = fetch_etf_daily(sym)
                logs.append(f"✅ {sym} — {len(df)} 天 — 来源: {src}")
            except Exception as e:
                logs.append(f"❌ {sym} — {str(e)[:80]}")
            progress.progress((i + 1) / len(holdings))
        st.markdown("\n".join(f"- {x}" for x in logs))

st.markdown("---")
st.subheader("快速跳转")
st.markdown(
    "- **动量策略**：分析持仓高低位、BIAS、轮动推荐\n"
    "- **轮动回测**：验证历史换仓表现\n"
    "- **组合回测**：固定权重持有回测\n"
    "- **ETF对比分析**：横向对比收益与风险"
)
