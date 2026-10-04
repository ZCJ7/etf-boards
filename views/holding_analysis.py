"""ETF 持仓分析页。"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from akshare_patch import install_akshare_patch

install_akshare_patch()

from analysis.conclusion import ConclusionConfig, build_pool_verdicts
from analysis.trade_records import (
    TradeRecord,
    dataframe_to_trades,
    get_trades_for_symbol,
    load_trades,
    save_trades,
    trades_to_dataframe,
)
from analysis.trade_screenshot import dedupe_trades, parse_screenshot, parse_trades_from_text
from analysis.indicators import (
    calc_bias,
    calc_macd,
    calc_ma,
    calc_ratio_line,
    ensure_close,
    latest_snapshot,
)
from analysis.screener import (
    _load_full_spot,
    build_ratio_sector_options,
    get_market_sector_map,
    get_momentum_top30,
)
from analysis.board_store import load_boards_snapshot, snapshot_caption
from analysis.low_position_screener import get_low_position_top30
from analysis.reversal_screener import get_reversal_top30
from analysis.trade_review import FORWARD_DAYS, evaluate_trade_points, reviews_to_dataframe
from etf_data_fetcher import data_asof_str, fetch_etf_daily_cached, fetch_pool_daily
from etfirst_client import etf_detail, is_logged_in
from holdings_utils import get_enabled_holdings
from ui.chart_utils import apply_chart_style
from ui.etfirst_display import render_etfirst_summary
from ui.sidebar import render_holdings_sidebar
from ui.styles import inject_global_styles, page_header
from ui.trade_chart import annotate_all_trade_rows

inject_global_styles()
pool = render_holdings_sidebar()
symbols = list(pool.keys())
holdings = get_enabled_holdings()

page_header(
    "📊 ETF 持仓分析",
    "核心结论 · 技术指标 · 比价线 · 数据源 AkShare / ETFirst",
    asof=data_asof_str(),
)

_asof_now = data_asof_str()
_prev_asof = st.session_state.get("boards_asof")
_has_boards = any(
    k in st.session_state
    for k in ("low_pos_top30_df", "reversal_top30_df", "ratio_top30_df", "core_verdicts")
)
if _prev_asof != _asof_now and (_prev_asof or _has_boards):
    for _k in (
        "low_pos_top30_df",
        "reversal_top30_df",
        "ratio_top30_df",
        "ratio_top30_key",
        "core_verdicts",
        "detail_result",
        "ratio_result",
        "ratio_rev_expand",
        "ratio_low_expand",
        "ratio_top30_expand",
        "boards_snapshot_hydrated",
        "boards_snapshot_meta",
        "boards_live_override",
    ):
        st.session_state.pop(_k, None)
    st.session_state["boards_asof_switched"] = (_prev_asof or "旧缓存", _asof_now)
st.session_state["boards_asof"] = _asof_now

if not st.session_state.get("boards_snapshot_hydrated"):
    st.session_state["boards_snapshot_hydrated"] = True
    _snap = load_boards_snapshot()
    if _snap:
        _meta = _snap.get("meta") or {}
        st.session_state["boards_snapshot_meta"] = _meta
        if "low_pos_top30_df" not in st.session_state and not _snap["low_position"].empty:
            st.session_state["low_pos_top30_df"] = _snap["low_position"]
        if "reversal_top30_df" not in st.session_state and not _snap["reversal"].empty:
            st.session_state["reversal_top30_df"] = _snap["reversal"]
        if "ratio_top30_df" not in st.session_state and not _snap["momentum"].empty:
            st.session_state["ratio_top30_df"] = _snap["momentum"]
            st.session_state["ratio_top30_key"] = (
                str(_meta.get("momentum_denominator") or "510300"),
                int(_meta.get("momentum_period") or 20),
            )

if len(symbols) < 1:
    st.warning("请先在侧边栏添加至少 1 只 ETF")
    st.stop()

if st.session_state.get("boards_asof_switched"):
    _old, _new = st.session_state.pop("boards_asof_switched")
    st.info(f"数据截止日期已从 **{_old}** 切换到 **{_new}**。请重新点击加载榜单 / 扫描结论。")

if "user_trades" not in st.session_state:
    st.session_state["user_trades"] = load_trades()

# st.tabs 在 st.rerun() 后总会回到第一项；用 segmented_control + session_state 记住分区
TAB_CORE = "🎯 核心结论"
TAB_DETAIL = "📈 指标详情"
TAB_TRADES = "📷 交易截图"
TAB_RATIO = "⚖️ 比价线"
TAB_ETFIRST = "🏦 ETFirst"
TAB_OPTIONS = [TAB_CORE, TAB_DETAIL, TAB_TRADES, TAB_RATIO, TAB_ETFIRST]

# 加载榜单过程中 / 刚点加载后，强制留在比价线（须在控件创建前写入）
if (
    st.session_state.get("ratio_top30_pending")
    or st.session_state.get("reversal_top30_pending")
    or st.session_state.get("low_pos_top30_pending")
    or st.session_state.get("ratio_market_pending")
    or st.session_state.get("ratio_pending")
    or st.session_state.get("pin_ratio_tab")
):
    st.session_state["holding_main_tab"] = TAB_RATIO
    st.session_state.pop("pin_ratio_tab", None)

if "holding_main_tab" not in st.session_state:
    st.session_state["holding_main_tab"] = TAB_CORE

active_tab = st.segmented_control(
    "页面分区",
    options=TAB_OPTIONS,
    key="holding_main_tab",
    label_visibility="collapsed",
)
# segmented_control 偶发返回 None；勿误落到核心结论，否则 pending 加载永不执行
if active_tab is None or active_tab not in TAB_OPTIONS:
    active_tab = st.session_state.get("holding_main_tab") or TAB_CORE
    if active_tab not in TAB_OPTIONS:
        active_tab = TAB_CORE

# 榜单加载放在分区内容之前：只要 pending=True 就会执行（不依赖当前 UI 分支）
_ratio_den = st.session_state.get("ratio_den") or ["510300"]
_rank_den = _ratio_den[0] if isinstance(_ratio_den, list) and _ratio_den else "510300"
_ratio_period_int = int(st.session_state.get("ratio_period") or 20)

if st.session_state.get("ratio_market_pending"):
    status = st.status("正在拉取全市场现货（超时约45秒，失败会提示）…", expanded=True)
    try:
        m_map, m_names = get_market_sector_map()
        st.session_state["ratio_market_map"] = m_map
        st.session_state["ratio_market_names"] = m_names
        if not m_map:
            st.session_state["ratio_market_err"] = "全市场现货超时或为空，请稍后重试"
            status.update(label="全市场板块加载失败", state="error")
        else:
            status.update(label=f"全市场板块已加载（{len(m_map)} 个）", state="complete")
    except Exception as exc:
        st.session_state["ratio_market_err"] = str(exc)
        status.update(label=f"失败：{exc}", state="error")
    st.session_state["ratio_market_pending"] = False

if st.session_state.get("reversal_top30_pending"):
    status = st.status("正在扫描周线反转 TOP30（约 40~90 秒）…", expanded=True)
    try:
        status.write("拉取成交额候选池与日线…")
        rdf = get_reversal_top30()
        st.session_state["reversal_top30_df"] = rdf
        if rdf is None or (isinstance(rdf, pd.DataFrame) and rdf.empty):
            st.session_state["reversal_top30_err"] = "结果为空"
            status.update(label="反转 TOP30 为空", state="error")
        else:
            st.session_state.pop("reversal_top30_err", None)
            status.update(label=f"反转 TOP30 已加载（{len(rdf)} 行）", state="complete")
            st.session_state["ratio_rev_expand"] = True
    except Exception as exc:
        st.session_state["reversal_top30_df"] = pd.DataFrame()
        st.session_state["reversal_top30_err"] = str(exc)
        status.update(label=f"失败：{exc}", state="error")
    st.session_state["reversal_top30_pending"] = False

if st.session_state.get("low_pos_top30_pending"):
    status = st.status("正在扫描低位 TOP30（约 40~90 秒）…", expanded=True)
    try:
        status.write("拉取成交额候选池与日线…")
        ldf = get_low_position_top30()
        st.session_state["low_pos_top30_df"] = ldf
        if ldf is None or (isinstance(ldf, pd.DataFrame) and ldf.empty):
            st.session_state["low_pos_top30_err"] = "结果为空"
            status.update(label="低位 TOP30 为空", state="error")
        else:
            st.session_state.pop("low_pos_top30_err", None)
            status.update(label=f"低位 TOP30 已加载（{len(ldf)} 行）", state="complete")
            st.session_state["ratio_low_expand"] = True
    except Exception as exc:
        st.session_state["low_pos_top30_df"] = pd.DataFrame()
        st.session_state["low_pos_top30_err"] = str(exc)
        status.update(label=f"失败：{exc}", state="error")
    st.session_state["low_pos_top30_pending"] = False

if st.session_state.get("ratio_top30_pending"):
    status = st.status(
        f"正在加载板块 TOP30（近{_ratio_period_int}日，约15~40秒）…",
        expanded=True,
    )
    try:
        df30 = get_momentum_top30(_rank_den, _ratio_period_int)
        st.session_state["ratio_top30_df"] = df30
        st.session_state["ratio_top30_key"] = (_rank_den, _ratio_period_int)
        if df30 is None or (isinstance(df30, pd.DataFrame) and df30.empty):
            st.session_state["ratio_top30_err"] = "榜单为空"
            status.update(label="板块 TOP30 为空", state="error")
        else:
            st.session_state.pop("ratio_top30_err", None)
            status.update(label=f"板块 TOP30 已加载（{len(df30)} 行）", state="complete")
            st.session_state["ratio_top30_expand"] = True
    except Exception as exc:
        st.session_state["ratio_top30_df"] = pd.DataFrame()
        st.session_state["ratio_top30_err"] = str(exc)
        status.update(label=f"失败：{exc}", state="error")
    st.session_state["ratio_top30_pending"] = False

# ── 核心结论 ─────────────────────────────────────────────
if active_tab == TAB_CORE:
    st.markdown(
        f"数据截止日期：**{data_asof_str()}**。一键扫描持仓池，输出每支 ETF 的**价均相对位置 / 均线走势 / BIAS / MACD / 近N日涨跌幅**精简结论。"
        " **「N日涨跌幅%」= 近 N 个交易日的累计涨跌幅**（不是评分、也不是仓位比例）。"
    )

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        core_ma = st.number_input("均线周期", 5, 120, 20, key="core_ma")
    with p2:
        core_bias = st.number_input("BIAS周期", 6, 60, 24, key="core_bias")
    with p3:
        core_mf = st.number_input("MACD快", 5, 30, 12, key="core_mf")
        core_ms = st.number_input("MACD慢", 10, 60, 26, key="core_ms")
        core_msg = st.number_input("MACD信号", 5, 20, 9, key="core_msg")
    with p4:
        core_momentum = st.number_input(
            "动量天数",
            5,
            60,
            20,
            key="core_mom",
            help="用于计算「N日涨跌幅%」，默认 20 个交易日",
        )

    core_cfg = ConclusionConfig(
        ma_period=int(core_ma),
        bias_period=int(core_bias),
        macd_fast=int(core_mf),
        macd_slow=int(core_ms),
        macd_signal=int(core_msg),
        momentum_days=int(core_momentum),
    )
    core_cfg_key = (
        core_cfg.ma_period,
        core_cfg.bias_period,
        core_cfg.macd_fast,
        core_cfg.macd_slow,
        core_cfg.macd_signal,
        core_cfg.momentum_days,
    )

    if st.session_state.get("core_cfg_key") != core_cfg_key:
        st.session_state["core_cfg_key"] = core_cfg_key
        st.session_state.pop("core_verdicts", None)

    c1, c2 = st.columns([1, 3])
    with c1:
        run_core = st.button("扫描持仓结论", type="primary", key="btn_core")
    with c2:
        st.caption(
            f"参数：MA{core_ma} · BIAS{core_bias} · MACD({core_mf}/{core_ms}/{core_msg}) · "
            f"{core_momentum}日涨跌幅。红色偏强、绿色偏弱（A股习惯）。"
        )

    if run_core:
        st.session_state["core_pending"] = True
        st.session_state.pop("core_verdicts", None)

    if st.session_state.get("core_pending") and "core_verdicts" not in st.session_state:
        target_syms = [h["symbol"] for h in holdings] if holdings else symbols[:5]
        need_bars = max(core_cfg.ma_period, core_cfg.bias_period, core_cfg.macd_slow, core_cfg.momentum_days) + 15
        prog = st.progress(0, text="并行拉取行情...")
        price_map = fetch_pool_daily(target_syms, max_workers=3, tail_days=need_bars, prefer_sina=False)
        prog.progress(100, text="生成结论...")
        st.session_state["core_verdicts"] = build_pool_verdicts(price_map, pool, core_cfg)
        st.session_state["core_pending"] = False
        prog.empty()

    if "core_verdicts" in st.session_state:
        verdicts = st.session_state["core_verdicts"]
        if not verdicts:
            st.warning("未能生成结论，请检查行情拉取。")
        else:
            md = verdicts[0].momentum_days
            labels = [f"{v.symbol}\n{v.name[:6]}" for v in verdicts]
            scores = [v.score for v in verdicts]
            colors = [v.color for v in verdicts]

            fig = go.Figure(
                go.Bar(
                    x=labels,
                    y=scores,
                    marker_color=colors,
                    text=[v.verdict for v in verdicts],
                    textposition="outside",
                )
            )
            apply_chart_style(fig, height=360, title="持仓综合评分（0-100）")
            fig.update_yaxes(range=[0, 105], title="评分")
            st.plotly_chart(fig, use_container_width=True)

            cols = st.columns(min(3, len(verdicts)))
            for i, v in enumerate(verdicts):
                with cols[i % len(cols)]:
                    st.markdown(
                        f"""<div class="verdict-card">
                        <strong>{v.symbol} {v.name}</strong><br/>
                        <span class="tag tag-bull">{v.verdict}</span>
                        <span class="tag tag-neutral">评分 {v.score}</span><br/>
                        <small>收盘 {v.close:.3f} · {v.momentum_days}日涨跌幅 {v.ret_momentum:+.1f}%</small><br/>
                        <small>价均：{v.price_ma_position} ({v.price_ma_dev_pct:+.1f}%) · {v.ma_trend} ({v.ma_slope_pct:+.1f}%)</small><br/>
                        <small>{v.bias_text} · MACD {v.macd_desc}</small><br/>
                        <small><strong>操作：{v.position_action}</strong></small>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            table = [
                {
                    "代码": v.symbol,
                    "名称": v.name,
                    "综合": v.verdict,
                    "评分": v.score,
                    f"{md}日涨跌幅%": v.ret_momentum,
                    "价均位置": v.price_ma_position,
                    "偏离均线%": v.price_ma_dev_pct,
                    "均线走势": v.ma_trend,
                    f"BIAS{v.bias_period}": v.bias_value,
                    "BIAS状态": v.bias_zone,
                    "MACD": v.macd_desc,
                    "趋势研判": v.trend,
                    "操作建议": v.position_action,
                }
                for v in verdicts
            ]
            bias_col = f"BIAS{verdicts[0].bias_period}"
            st.dataframe(
                table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "代码": st.column_config.TextColumn("代码", pinned=True, width=90),
                    "名称": st.column_config.TextColumn("名称", pinned=True, width=160),
                    "评分": st.column_config.NumberColumn("评分", format="%.0f"),
                    f"{md}日涨跌幅%": st.column_config.NumberColumn(f"{md}日涨跌幅%", format="%.2f"),
                    "偏离均线%": st.column_config.NumberColumn("偏离均线%", format="%.2f"),
                    bias_col: st.column_config.NumberColumn(bias_col, format="%.2f"),
                    "MACD": st.column_config.TextColumn("MACD", width="medium"),
                    "操作建议": st.column_config.TextColumn("操作建议", width="large"),
                },
            )

# ── 指标详情 ─────────────────────────────────────────────
elif active_tab == TAB_DETAIL:
    st.caption(f"数据截止日期：**{data_asof_str()}**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        focus = st.selectbox("分析标的", symbols, format_func=lambda x: f"{x} {pool.get(x, x)}", key="detail_sym")
    with c2:
        ma_periods = st.multiselect("均线周期", [5, 10, 20, 28, 60], default=[20, 28, 60], key="detail_ma")
    with c3:
        bias_periods = st.multiselect("BIAS周期", [6, 12, 24], default=[6, 12, 24], key="detail_bias")
    with c4:
        macd_fast = st.number_input("MACD快", 5, 30, 12, key="detail_mf")
        macd_slow = st.number_input("MACD慢", 10, 60, 26, key="detail_ms")
        macd_signal = st.number_input("MACD信号", 5, 20, 9, key="detail_msg")
    lookback_days = st.slider("分析区间（交易日约）", 60, 500, 90, key="detail_lb")

    if st.button("运行指标分析", type="primary", key="btn_detail"):
        st.session_state["detail_pending"] = True
        st.session_state.pop("detail_result", None)

    if st.session_state.get("detail_pending") and "detail_result" not in st.session_state:
        with st.spinner("拉取行情..."):
            fetch_days = lookback_days + FORWARD_DAYS + 10
            df, source = fetch_etf_daily_cached(focus, tail_days=fetch_days)
            close_full = ensure_close(df)
            close = close_full.tail(lookback_days)
            ma_df = calc_ma(close, ma_periods)
            bias_df = calc_bias(close, bias_periods)
            macd_df = calc_macd(close, int(macd_fast), int(macd_slow), int(macd_signal))
            ma_full = calc_ma(close_full, ma_periods)
            bias_full = calc_bias(close_full, bias_periods)
            macd_full = calc_macd(close_full, int(macd_fast), int(macd_slow), int(macd_signal))
            snap = latest_snapshot(close, ma_periods, bias_periods, (int(macd_fast), int(macd_slow), int(macd_signal)))
            st.session_state["detail_result"] = {
                "focus": focus,
                "source": source,
                "close": close,
                "close_full": close_full,
                "ma_df": ma_df,
                "bias_df": bias_df,
                "macd_df": macd_df,
                "ma_full": ma_full,
                "bias_full": bias_full,
                "macd_full": macd_full,
                "ma_periods": ma_periods,
                "snap": snap,
            }
            st.session_state["detail_pending"] = False

    if "detail_result" in st.session_state:
        r = st.session_state["detail_result"]
        snap = r["snap"]
        st.success(f"{r['focus']} {pool.get(r['focus'], '')} · 数据来源: {r['source']}")
        if "前复权" in str(r.get("source", "")):
            st.caption(
                "行情来自新浪并已做**份额拆分前复权**（如通信ETF 515880 在 2026-02、2026-07 的 1:3 / 1:2 拆分），"
                "避免未复权数据在图上出现「假暴跌」。"
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("收盘价", f"{snap.get('收盘价', 0):.3f}")
        m2.metric("MACD", snap.get("MACD信号", "—"))
        bias12 = snap.get("BIAS12")
        m3.metric("BIAS12", f"{bias12:.2f}%" if bias12 is not None else "—")
        m4.metric("日期", snap.get("日期", "—"))

        bias_cols = [c for c in snap if c.startswith("BIAS") and not c.endswith("状态")]
        if bias_cols:
            st.markdown("**BIAS 解读**")
            for c in bias_cols:
                val = snap.get(c)
                if val is not None:
                    st.write(f"- {c}: {val:.2f}% → {snap.get(c + '状态', '')}")

        close, ma_df, bias_df, macd_df = r["close"], r["ma_df"], r["bias_df"], r["macd_df"]
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.52, 0.24, 0.24],
            vertical_spacing=0.06,
            subplot_titles=("价格", "BIAS", "MACD"),
        )
        fig.add_trace(go.Scatter(x=close.index, y=close, name="收盘价", line=dict(color="#0ea5e9", width=2)), row=1, col=1)
        for col in ma_df.columns:
            fig.add_trace(go.Scatter(x=ma_df.index, y=ma_df[col], name=col, line=dict(dash="dot", width=1)), row=1, col=1)
        for col in bias_df.columns:
            fig.add_trace(go.Scatter(x=bias_df.index, y=bias_df[col], name=col, line=dict(width=1.5)), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1, row=2, col=1)
        fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["DIF"], name="DIF", line=dict(color="#6366f1", width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["DEA"], name="DEA", line=dict(color="#f59e0b", width=1.5)), row=3, col=1)
        fig.add_trace(go.Bar(x=macd_df.index, y=macd_df["MACD"], name="MACD柱", marker_color="#94a3b8", opacity=0.35), row=3, col=1)

        focus_trades = get_trades_for_symbol(st.session_state.get("user_trades", []), r["focus"])
        if focus_trades:
            annotate_all_trade_rows(fig, focus_trades, close, bias_df, macd_df)
            st.caption(
                f"已标注 {len(focus_trades)} 笔交易：价格图红▲绿▼；BIAS/MACD 为细竖线+小圆点（悬停看详情）"
            )

        apply_chart_style(fig, height=820, title=f"{r['focus']} 技术指标")
        fig.update_annotations(font_size=11, font_color="#64748b")
        st.plotly_chart(fig, use_container_width=True)

        if focus_trades:
            primary_ma = "MA20" if "MA20" in r.get("ma_periods", ma_periods) else (
                f"MA{r['ma_periods'][0]}" if r.get("ma_periods") else "MA20"
            )
            reviews = evaluate_trade_points(
                focus_trades,
                r.get("close_full", close),
                r.get("bias_full", bias_df),
                r.get("macd_full", macd_df),
                r.get("ma_full", ma_df),
                primary_ma=primary_ma,
            )
            st.markdown(f"### 买卖点复盘（后 {FORWARD_DAYS} 个交易日假设收益）")
            st.caption(
                "基于当日 BIAS / MACD / 均线位置评价买卖点；"
                f"「后{FORWARD_DAYS}日收益%」= 从成交当日至其后第 {FORWARD_DAYS} 个交易日的涨跌幅。"
            )
            review_df = reviews_to_dataframe(reviews)
            if review_df.empty:
                st.info("暂无有效买卖点可复盘（日期可能不在行情区间内）。")
            else:
                st.dataframe(
                    review_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "日期": st.column_config.TextColumn("日期", pinned=True),
                        "方向": st.column_config.TextColumn("方向", width=60),
                        "收盘价": st.column_config.NumberColumn("收盘价", format="%.3f"),
                        "BIAS12": st.column_config.NumberColumn("BIAS12", format="%.2f"),
                        "偏离均线%": st.column_config.NumberColumn("偏离均线%", format="%.2f"),
                        f"后{FORWARD_DAYS}日收益%": st.column_config.NumberColumn(
                            f"后{FORWARD_DAYS}日收益%", format="%+.2f"
                        ),
                        "综合评价": st.column_config.TextColumn("综合评价", width=100),
                        "指标分析": st.column_config.TextColumn("指标分析", width="large"),
                        "假设情景": st.column_config.TextColumn("假设情景", width="large"),
                    },
                )
                for rv in reviews:
                    with st.expander(f"{rv.date} {rv.action} · {rv.evaluation}", expanded=False):
                        st.write(rv.analysis)
                        st.write(f"**假设情景**：{rv.counterfactual}")
                        if rv.forward_end_date:
                            st.caption(f"后{FORWARD_DAYS}日统计区间：{rv.date} → {rv.forward_end_date}")
        else:
            st.caption("在「交易截图」页保存买卖记录后，此处将自动复盘。")

# ── 交易截图 ─────────────────────────────────────────────
elif active_tab == TAB_TRADES:
    st.markdown(
        "上传券商/App **交易成交截图**（支持长截图），自动识别 ETF 代码、买卖、日期；"
        "识别结果可校对后保存，并标注到 **指标详情** 价格图。"
    )
    st.info(
        "默认识别：**本地 OCR**（无需 API Key）。截图多为**联接基金/指数基金**，"
        "会自动按**板块+名称**映射到你侧边栏的**持仓池**代码（联接C ≠ 场内ETF，但跟踪同一指数）。\n"
        "支持格式：`名称(代码)买入成交`、`买入 基金名称` + 日期。\n"
        "**截图要求**：完整屏幕宽度；记录太长可多张同时上传（自动合并去重）。"
        " 可选配置 AI Key 作为补充。"
    )

    up_col, paste_col = st.columns([1, 1])
    with up_col:
        uploads = st.file_uploader(
            "上传交易截图（可多选）",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="trade_screenshots",
        )
        if st.button("识别截图", type="primary", key="btn_parse_shot") and uploads:
            parsed_all: list[TradeRecord] = []
            msgs: list[str] = []
            ocr_debug: list[str] = []
            with st.spinner(f"识别中（共 {len(uploads)} 张，本地 OCR）..."):
                for f in uploads:
                    trades, msg, ocr_snip = parse_screenshot(f.getvalue())
                    parsed_all.extend(trades)
                    msgs.append(f"{f.name}: {msg}")
                    if ocr_snip:
                        ocr_debug.append(f"=== {f.name} ===\n{ocr_snip}")
            parsed_all = dedupe_trades(parsed_all)
            st.session_state["trade_parse_preview"] = trades_to_dataframe(parsed_all)
            st.session_state["trade_parse_msgs"] = msgs
            st.session_state["trade_editor_df"] = st.session_state["trade_parse_preview"]
            if ocr_debug:
                st.session_state["trade_ocr_debug"] = ocr_debug

    with paste_col:
        ocr_text = st.text_area("或粘贴截图 OCR 文字 / 成交记录", height=160, key="trade_ocr_text")
        if st.button("从文字解析", key="btn_parse_text") and ocr_text.strip():
            parsed = parse_trades_from_text(ocr_text)
            st.session_state["trade_parse_preview"] = trades_to_dataframe(parsed)
            st.session_state["trade_parse_msgs"] = [f"规则解析 {len(parsed)} 条"]
            st.session_state["trade_editor_df"] = st.session_state["trade_parse_preview"]

    if st.session_state.get("trade_parse_msgs"):
        for m in st.session_state["trade_parse_msgs"]:
            st.write(f"- {m}")

    if st.session_state.get("trade_ocr_debug"):
        with st.expander("OCR 原文（识别失败时可复制到右侧文字框再解析）", expanded=False):
            st.text("\n\n".join(st.session_state["trade_ocr_debug"])[:8000])

    st.subheader("识别结果（可编辑后保存）")
    editor_df = st.session_state.get("trade_editor_df", trades_to_dataframe(st.session_state["user_trades"]))
    edited = st.data_editor(
        editor_df,
        num_rows="dynamic",
        use_container_width=True,
        key="trade_editor",
        column_config={
            "代码": st.column_config.TextColumn("代码", width=90),
            "方向": st.column_config.SelectboxColumn("方向", options=["买入", "卖出"]),
            "日期": st.column_config.TextColumn("日期", help="YYYY-MM-DD"),
            "价格": st.column_config.NumberColumn("价格", format="%.4f"),
            "数量": st.column_config.NumberColumn("数量"),
        },
    )

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("💾 保存到交易记录", type="primary", key="btn_save_trades"):
            saved = dataframe_to_trades(edited)
            st.session_state["user_trades"] = saved
            save_trades(saved)
            st.session_state["trade_editor_df"] = trades_to_dataframe(saved)
            st.session_state.pop("trade_parse_preview", None)
            st.success(f"已保存 {len(saved)} 笔交易")
    with b2:
        if st.button("🔄 重新加载本地记录", key="btn_reload_trades"):
            st.session_state["user_trades"] = load_trades()
            st.session_state["trade_editor_df"] = trades_to_dataframe(st.session_state["user_trades"])
            st.session_state.pop("trade_parse_preview", None)
            st.rerun()
    with b3:
        if st.button("🗑️ 清空全部记录", key="btn_clear_trades"):
            st.session_state["user_trades"] = []
            st.session_state["trade_editor_df"] = trades_to_dataframe([])
            save_trades([])
            st.session_state.pop("trade_parse_preview", None)
            st.rerun()

    st.markdown("**已保存记录**")
    st.dataframe(trades_to_dataframe(st.session_state["user_trades"]), hide_index=True, use_container_width=True)

# ── 比价线 ───────────────────────────────────────────────
elif active_tab == TAB_RATIO:
    _asof = data_asof_str()
    st.caption(
        f"数据截止日期：**{_asof}**。比价线用于发现主线。默认只用**持仓板块**，不自动拉全市场（避免卡顿）。"
        "需要时再点按钮加载 TOP30 / 反转榜 / 低位榜 / 全市场板块。"
    )

    # 全市场板块：仅按需加载，禁止进 tab 就请求东财现货（会卡死页面）
    if "ratio_market_map" not in st.session_state:
        st.session_state["ratio_market_map"] = {}
        st.session_state["ratio_market_names"] = {}

    rank_den = "510300"
    ratio_period_int = 20
    ratio_ma = 20
    den_pool: list[str] = ["510300"]

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        den_default = ["510300"]
        den_pool = st.multiselect(
            "分母（可多选，默认沪深300）",
            ["510300", "510500", "159915", "588000"] + symbols,
            default=den_default,
            format_func=lambda x: f"{x} {'沪深300ETF' if x == '510300' else pool.get(x, '')}",
            key="ratio_den",
        )
    with ctrl2:
        ratio_ma = st.number_input("比价均线周期", 5, 120, 20, key="ratio_ma_p")
    with ctrl3:
        ratio_period = st.selectbox(
            "榜单统计周期",
            [5, 10, 20, 60],
            index=2,
            format_func=lambda x: f"近{x}日",
            key="ratio_period",
        )
    with ctrl4:
        st.write("")  # 占位对齐

    rank_den = den_pool[0] if den_pool else "510300"
    ratio_period_int = int(ratio_period)
    cache_key = (rank_den, ratio_period_int)

    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        load_top30 = st.button("加载板块 TOP30", key="btn_load_ratio_top30")
    with btn_col2:
        load_rev = st.button("加载反转 TOP30", key="btn_load_reversal_top30")
    with btn_col3:
        load_low = st.button("加载低位 TOP30", key="btn_load_low_pos_top30")
    with btn_col4:
        load_market = st.button("加载全市场板块", key="btn_load_market_sectors")

    # 不可在 segmented_control(key=holding_main_tab) 创建后再写该 key；
    # 用 pin_ratio_tab，在下次 run 顶部（控件创建前）切回比价线。
    if load_top30:
        st.session_state["pin_ratio_tab"] = True
        st.session_state["boards_live_override"] = True
        st.session_state["ratio_top30_pending"] = True
        st.session_state.pop("ratio_top30_df", None)
        st.session_state.pop("ratio_top30_err", None)
        try:
            get_momentum_top30.clear()
            _load_full_spot.clear()
        except Exception:
            pass
        st.rerun()

    if load_rev:
        st.session_state["pin_ratio_tab"] = True
        st.session_state["boards_live_override"] = True
        st.session_state["reversal_top30_pending"] = True
        st.session_state.pop("reversal_top30_df", None)
        st.session_state.pop("reversal_top30_err", None)
        try:
            get_reversal_top30.clear()
            _load_full_spot.clear()
        except Exception:
            pass
        st.rerun()

    if load_low:
        st.session_state["pin_ratio_tab"] = True
        st.session_state["boards_live_override"] = True
        st.session_state["low_pos_top30_pending"] = True
        st.session_state.pop("low_pos_top30_df", None)
        st.session_state.pop("low_pos_top30_err", None)
        try:
            get_low_position_top30.clear()
            _load_full_spot.clear()
        except Exception:
            pass
        st.rerun()

    if load_market:
        st.session_state["pin_ratio_tab"] = True
        st.session_state["ratio_market_pending"] = True
        st.session_state.pop("ratio_market_err", None)
        try:
            get_market_sector_map.clear()
            _load_full_spot.clear()
        except Exception:
            pass
        st.rerun()

    _snap_note = ""
    if not st.session_state.get("boards_live_override"):
        _snap_note = snapshot_caption(st.session_state.get("boards_snapshot_meta"))
        _snap_asof = (st.session_state.get("boards_snapshot_meta") or {}).get("asof")
        if _snap_note and _snap_asof and str(_snap_asof) != str(_asof_now):
            st.info(
                f"{_snap_note}。当前截止日是 **{_asof_now}**，点上方加载可按最新完整交易日重算。"
            )
        elif _snap_note:
            st.caption(_snap_note)

    top30_df = pd.DataFrame()
    if st.session_state.get("ratio_top30_key") == cache_key:
        top30_df = st.session_state.get("ratio_top30_df", pd.DataFrame())
    if not isinstance(top30_df, pd.DataFrame):
        top30_df = pd.DataFrame()

    rev_df = st.session_state.get("reversal_top30_df", pd.DataFrame())
    if not isinstance(rev_df, pd.DataFrame):
        rev_df = pd.DataFrame()

    low_df = st.session_state.get("low_pos_top30_df", pd.DataFrame())
    if not isinstance(low_df, pd.DataFrame):
        low_df = pd.DataFrame()

    market_map = st.session_state.get("ratio_market_map") or {}
    market_names = st.session_state.get("ratio_market_names") or {}
    if st.session_state.get("ratio_market_err"):
        st.caption(f"全市场板块：{st.session_state.pop('ratio_market_err')[:160]}")

    sector_list, sector_map = build_ratio_sector_options(
        symbols,
        top30_df if not top30_df.empty else None,
        market_map if market_map else None,
        holdings_names=pool,
    )
    momentum_names = {row["代码"]: row["名称"] for _, row in top30_df.iterrows()} if not top30_df.empty else {}
    momentum_names.update(market_names)
    for extra_df in (rev_df, low_df):
        if not extra_df.empty and "代码" in extra_df.columns:
            for _, row in extra_df.iterrows():
                momentum_names[str(row["代码"])] = str(row.get("名称", ""))

    def _ratio_label(code: str) -> str:
        name = pool.get(code) or momentum_names.get(code, "")
        return f"{code} {name}".strip()

    pick1, pick2 = st.columns(2)
    with pick1:
        ratio_sector = st.selectbox("板块", sector_list, key="ratio_sector")
        sector_syms = sector_map.get(ratio_sector, symbols)
        for extra_df in (rev_df, low_df):
            if not extra_df.empty and "代码" in extra_df.columns:
                extra_codes = [str(c) for c in extra_df["代码"].tolist()]
                sector_syms = list(dict.fromkeys(list(sector_syms) + extra_codes))
    with pick2:
        num_sym = st.selectbox(
            "分子 ETF",
            sector_syms if sector_syms else symbols,
            format_func=_ratio_label,
            key="ratio_num",
        )
    with st.expander(
        "板块 TOP30 · 相对强度榜单",
        expanded=bool(st.session_state.get("ratio_top30_expand")) and not top30_df.empty,
    ):
        st.markdown(
            f"**排名逻辑**：各板块按 **相对强度龙头** 排序（正向跑赢优先）；"
            f"**代码/名称/成交龙头涨幅** = 板块内成交额最高的那只；"
            f"**相对强度 / 强度龙头涨幅** = 板块内相对强度最高的那只（决定排名）。"
            f" 相对强度 = 强度龙头涨幅% − 分母涨幅%。"
            f" 当前分母：**{rank_den}**"
            + (
                f"（近{ratio_period}日 {top30_df['分母近N日涨幅%'].iloc[0]:+.2f}%）"
                if not top30_df.empty and "分母近N日涨幅%" in top30_df.columns
                else ""
            )
            + f" 数据截止日期：**{_asof}**。"
        )
        if top30_df.empty:
            err = st.session_state.get("ratio_top30_err")
            st.info(
                "TOP30 榜单未加载。默认可用**持仓板块**选分子并计算比价线；"
                "需要全市场主线时再点 **「加载板块 TOP30」** 或 **「加载全市场板块」**。"
            )
            if err:
                st.warning(f"上次加载失败: {str(err)[:240]}")
        else:
            st.caption(
                f"数据截止日期：{_asof} · 共 {len(top30_df)} 个板块 · 近 {ratio_period} 日 · "
                "注意：表中「代码」是成交龙头，排名看「强度龙头涨幅/相对强度」，两者可能不是同一只。"
                + (f" · {_snap_note}" if _snap_note else "")
            )
            st.dataframe(top30_df, hide_index=True, use_container_width=True)

    with st.expander(
        "反转 TOP30 · 周线反转榜单",
        expanded=bool(st.session_state.get("ratio_rev_expand")) and not rev_df.empty,
    ):
        st.markdown(
            "**入选条件**（须同时满足）：\n"
            "1. **周线收盘价在 MA20 上方**（最新一周收盘 > 周线 MA20）\n"
            "2. **周线 MACD**：绿柱缩短 **或** 红柱变长 **或** 金叉（满足其一）\n\n"
            "**板块规则**（对齐比价 TOP30）：**每个板块只放 1 只 ETF**"
            "（该板块内近20日均量最高、且满足上述条件的标的）；"
            "再按该代表的近20日均量对**板块**降序，取前 30。"
            f" 数据截止日期：**{_asof}**。"
        )
        if rev_df.empty:
            err = st.session_state.get("reversal_top30_err")
            st.info("反转榜未加载。点击 **「加载反转 TOP30」** 扫描候选池（约 40~90 秒，请勿中途切页）。")
            if err:
                st.warning(f"上次加载失败: {str(err)[:240]}")
        else:
            st.caption(
                f"数据截止日期：{_asof} · 共 {len(rev_df)} 个板块 · 每板块 1 只 · 按近20日均量排板块；分子下拉已追加本榜代码"
                + (f" · {_snap_note}" if _snap_note else "")
            )
            st.dataframe(rev_df, hide_index=True, use_container_width=True)

    with st.expander(
        "低位 TOP30 · 周线深跌 + 日线 MACD 转强",
        expanded=bool(st.session_state.get("ratio_low_expand")) and not low_df.empty,
    ):
        st.markdown(
            "**入选条件**（须同时满足）：\n"
            "1. **周线收盘 < MA20**，且距周线 MA20 **超过 5%**（`(收盘−MA20)/MA20 ≤ −5%`）\n"
            "2. **周线 MA60 不能向下**（本周 MA60 ≥ 上周，走平或向上）\n"
            "3. **日线 MACD**：绿柱缩短 **或** 红柱变长 **或** 金叉\n\n"
            "表中 **离周MA20%** = `(收盘−周线MA20)/周线MA20×100`，负值表示在均线下方。"
            "**排名规则**：**不做板块去重**；按近 20 日均量降序取前 30。"
            f" 数据截止日期：**{_asof}**（未收盘不含当天）。"
        )
        if low_df.empty:
            err = st.session_state.get("low_pos_top30_err")
            st.info("低位榜未加载。点击 **「加载低位 TOP30」** 扫描候选池（约 40~90 秒，请勿中途切页）。")
            if err:
                st.warning(f"上次加载失败: {str(err)[:240]}")
        else:
            st.caption(
                f"数据截止日期：{_asof} · 共 {len(low_df)} 只 · 未按板块去重 · 按近20日均量排序；分子下拉已追加本榜代码"
                + (f" · {_snap_note}" if _snap_note else "")
            )
            st.dataframe(low_df, hide_index=True, use_container_width=True)

    if st.button("计算比价线", type="primary", key="btn_ratio"):
        st.session_state["ratio_pending"] = True
        st.session_state.pop("ratio_result", None)

    if st.session_state.get("ratio_pending") and "ratio_result" not in st.session_state:
        if not den_pool:
            st.warning("请至少选择一个分母")
            st.session_state["ratio_pending"] = False
        else:
            with st.spinner("拉取分子/分母行情..."):
                try:
                    df_num, src1 = fetch_etf_daily_cached(num_sym, tail_days=400, prefer_sina=False)
                    close_num = ensure_close(df_num)
                except Exception as exc:
                    st.error(f"分子 {num_sym} 行情获取失败: {exc}")
                    st.session_state["ratio_pending"] = False
                    close_num = None
                if close_num is not None:
                    series_map = {}
                    for den_code in den_pool:
                        try:
                            df_den, _ = fetch_etf_daily_cached(den_code, tail_days=400, prefer_sina=False)
                            close_den = ensure_close(df_den)
                            ratio_df = calc_ratio_line(close_num, close_den, ma_period=int(ratio_ma))
                            series_map[den_code] = ratio_df
                        except Exception as exc:
                            st.warning(f"分母 {den_code} 获取失败: {exc}")
                    if series_map:
                        st.session_state["ratio_result"] = {
                            "num_sym": num_sym,
                            "den_pool": den_pool,
                            "src1": src1,
                            "series_map": series_map,
                            "ratio_ma": int(ratio_ma),
                        }
                st.session_state["ratio_pending"] = False

    if "ratio_result" in st.session_state:
        rr = st.session_state["ratio_result"]
        st.success(f"分子 {rr['num_sym']} 数据来源 {rr['src1']}")
        for den_code, ratio_df in rr["series_map"].items():
            last = ratio_df.iloc[-1]
            den_label = "沪深300" if den_code == "510300" else den_code
            st.markdown(f"**vs {den_label}**")
            c1, c2 = st.columns(2)
            c1.metric("当前比价", f"{last['比价']:.4f}")
            c2.metric("比价BIAS", f"{last['比价BIAS']:.2f}%")

            ma_col = f"比价MA{rr['ratio_ma']}"
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35])
            fig.add_trace(
                go.Scatter(x=ratio_df.index, y=ratio_df["比价"], name=f"{rr['num_sym']}/{den_code}"),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=ratio_df.index, y=ratio_df[ma_col], name=ma_col, line=dict(dash="dash")),
                row=1,
                col=1,
            )
            fig.add_trace(go.Scatter(x=ratio_df.index, y=ratio_df["比价BIAS"], name="比价BIAS"), row=2, col=1)
            fig.add_hline(y=0, line_dash="dash", row=2, col=1)
            apply_chart_style(fig, height=480, title=f"比价线: {rr['num_sym']} ÷ {den_code}")
            st.plotly_chart(fig, use_container_width=True, key=f"ratio_chart_{den_code}")

        st.info("比价线 = 分子收盘价 / 分母收盘价。比价BIAS 衡量相对强弱，可用于判断行业ETF相对大盘的位置。")

# ── ETFirst ──────────────────────────────────────────────
elif active_tab == TAB_ETFIRST:
    st.caption("数据来源：首趋E指。需先在侧边栏登录 API Key。")
    if not is_logged_in():
        st.warning("请先在侧边栏完成 ETFirst 登录。")
    else:
        ef_code = st.selectbox("查询 ETF", symbols, format_func=lambda x: f"{x} {pool.get(x, x)}", key="etfirst_code")
        if st.button("查询官方详情", type="primary", key="btn_etfirst"):
            st.session_state["etfirst_pending"] = True
            st.session_state.pop("etfirst_result", None)

        if st.session_state.get("etfirst_pending") and "etfirst_result" not in st.session_state:
            with st.spinner("调用 ETFirst..."):
                try:
                    data = etf_detail(ef_code)
                    st.session_state["etfirst_result"] = {"code": ef_code, "data": data, "error": None}
                except Exception as exc:
                    st.session_state["etfirst_result"] = {"code": ef_code, "data": None, "error": str(exc)}
                st.session_state["etfirst_pending"] = False

        if "etfirst_result" in st.session_state:
            er = st.session_state["etfirst_result"]
            if er.get("error"):
                st.error(er["error"])
            elif er.get("data"):
                render_etfirst_summary(er["data"], er["code"], pool.get(er["code"], ""))
