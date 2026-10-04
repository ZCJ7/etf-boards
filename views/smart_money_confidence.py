"""中国版聪明资金信心指数（独立页）。"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from akshare_patch import install_akshare_patch

install_akshare_patch()

from analysis.smart_money import (
    CHART_DAYS,
    DEFAULT_BENCH,
    FACTOR_META,
    ROLL_OPTIONS,
    WEIGHTS,
    compute_csmci,
    factor_explain,
    fetch_benchmark_closes,
    normalize_to_100,
)
from etf_data_fetcher import data_asof_str
from ui.chart_utils import apply_chart_style
from ui.sidebar import render_holdings_sidebar
from ui.styles import inject_global_styles, page_header
from utils import clean_etf_symbol

inject_global_styles()
pool = render_holdings_sidebar()

page_header(
    "🧠 聪明资金信心",
    "中国版 CSMCI · 多因子日度加权 · 可与自选宽基同图对比",
    asof=data_asof_str(),
)

st.markdown(
    """
**指数含义**：0–100 分，刻画机构/聪明资金对 A 股的相对信心。  
**读数**：**>70** 偏强；**30–70** 中性；**<30** 偏谨慎。  
**权重**：IF前20会员净持仓 30% · QVIX（逆向）20% · 北向 20% · 融资（逆向）15% · 宽基ETF流向 15%。
"""
)

preset = ["510300", "510500", "159915", "588000", "512100", "510050"]
extra = [c for c in pool.keys() if c not in preset]
bench_options = preset + extra

b1, b2, b3 = st.columns([2, 1, 1])
with b1:
    bench_codes = st.multiselect(
        "对比宽基/ETF（可多选，走势归一到100）",
        options=bench_options,
        default=[c for c in DEFAULT_BENCH if c in bench_options],
        format_func=lambda x: f"{x} {pool.get(x, '')}".strip(),
        key="csmci_bench_codes",
    )
with b2:
    custom = st.text_input("追加代码（逗号分隔）", placeholder="如 159941,512890", key="csmci_custom_codes")
with b3:
    roll_days = st.selectbox(
        "近 N 日窗口",
        options=list(ROLL_OPTIONS),
        index=0,
        help="IF前20净持仓、融资变化率、北向/ETF 流向等滚动窗口",
        key="csmci_roll_days",
    )

if custom.strip():
    for part in custom.replace("，", ",").split(","):
        code = clean_etf_symbol(part)
        if code and code not in bench_codes:
            bench_codes.append(code)

st.caption(
    f"图表展示近半年（约 {CHART_DAYS} 个交易日）。"
    f"当前因子窗口：**近 {roll_days} 日**。点计算后单因子超时会跳过。"
)

c_btn, c_clear = st.columns([2, 1])
with c_btn:
    run = st.button("计算聪明资金信心指数", type="primary", key="btn_csmci")
with c_clear:
    if st.button("清除缓存重算", key="btn_csmci_clear"):
        compute_csmci.clear()
        fetch_benchmark_closes.clear()
        st.session_state.pop("csmci_result", None)
        st.session_state.pop("csmci_bench_df", None)
        st.success("缓存已清，请再点计算")

if run:
    st.session_state["csmci_pending"] = True
    st.session_state["csmci_run_roll"] = int(roll_days)
    st.session_state.pop("csmci_result", None)
    st.session_state.pop("csmci_bench_df", None)

if st.session_state.get("csmci_pending"):
    status = st.status("正在计算（某源失败会自动跳过，不会整页卡死）...", expanded=True)
    try:
        n = int(st.session_state.get("csmci_run_roll", roll_days))
        status.write(f"拉取北向 / 融资 / QVIX / IF前20会员净持仓 / 宽基流向（近 {n} 日）…")
        st.session_state["csmci_result"] = compute_csmci(roll_days=n)
        codes_tuple = tuple(clean_etf_symbol(c) for c in bench_codes if clean_etf_symbol(c))
        if codes_tuple:
            status.write(f"拉取对比标的：{', '.join(codes_tuple)}")
            st.session_state["csmci_bench_df"] = fetch_benchmark_closes(codes_tuple, tail_days=max(200, CHART_DAYS + 40))
        else:
            st.session_state["csmci_bench_df"] = pd.DataFrame()
        status.update(label="计算完成", state="complete")
    except Exception as exc:
        st.session_state["csmci_result"] = {
            "history": pd.DataFrame(),
            "latest": {},
            "factors": {},
            "error": str(exc),
        }
        status.update(label=f"失败：{exc}", state="error")
    st.session_state["csmci_pending"] = False

result = st.session_state.get("csmci_result")
if not result:
    st.info("尚未计算。选择对比宽基后，点击 **「计算聪明资金信心指数」**。")
    st.stop()

if result.get("error") and (result.get("history") is None or getattr(result.get("history"), "empty", True)):
    st.error(f"计算失败：{result['error']}")
    st.stop()

latest = result.get("latest") or {}
history = result.get("history")
if not isinstance(history, pd.DataFrame):
    history = pd.DataFrame()
factors = result.get("factors") or {}
bench_df = st.session_state.get("csmci_bench_df")
if not isinstance(bench_df, pd.DataFrame):
    bench_df = pd.DataFrame()
used_n = int(result.get("roll_days") or latest.get("roll_days") or roll_days)

if latest:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CSMCI", f"{latest.get('CSMCI', '—')}")
    c2.metric("区间", latest.get("区间", "—"))
    c3.metric("日期", latest.get("日期", "—"))
    hs = latest.get("沪深300")
    c4.metric("沪深300", f"{hs:.2f}" if hs is not None else "—")
else:
    st.warning("暂无有效最新分数（可能部分数据源超时），可点「清除缓存重算」。")

st.markdown("### 子指标得分（0–100）")
rows = []
asof = latest.get("日期")
for key, meta in FACTOR_META.items():
    fac = factors.get(key)
    ok = bool(fac and getattr(fac, "ok", False))
    score_s = fac.score.dropna() if fac is not None else pd.Series(dtype=float)
    last_score_date = ""
    if ok and not score_s.empty:
        last_score_date = pd.Timestamp(score_s.index.max()).strftime("%Y-%m-%d")
    day_score = latest.get(key, "—")
    if ok and last_score_date and asof and last_score_date < asof:
        status = f"可用（数据止于{last_score_date}）"
    elif ok:
        status = "可用"
    else:
        status = "缺失/停更（已降权）"
    note = getattr(fac, "note", "") if fac else ""
    if ok and day_score == 0.0:
        note = (note + "；0分=近窗口极值，不是没数据").strip("；")
    rows.append(
        {
            "因子": meta["name"],
            "方向": meta["dir"],
            "权重": f"{meta['weight']*100:.0f}%",
            "当日得分": day_score,
            "状态": status,
            "说明": note,
        }
    )
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption("说明：0 分表示该因子处于近 60 日最弱（逆向因子则最强），不是「未算出」。北向净买入自 2024-08 官方停更后不再计入权重。")

with st.expander("各因子含义与读法", expanded=True):
    st.caption(f"当前计算窗口：近 **{used_n}** 日")
    for key, meta in FACTOR_META.items():
        st.markdown(f"**{meta['name']}**（{meta['dir']} · 权重 {meta['weight']*100:.0f}%）  \n{factor_explain(key, used_n)}")
    margin_score = latest.get("margin")
    if margin_score is not None and float(margin_score) >= 95:
        st.info(
            f"当前融资得分 **{margin_score}**：近 {used_n} 日融资余额变化率处于近 60 日**最强收缩**一端。"
            "因该因子为**逆向**（散户去杠杆视为过热降温），映射为接近满分，"
            "并不代表「融资余额本身创历史新高」。"
        )

if not history.empty and "CSMCI" in history.columns:
    plot_df = history.dropna(subset=["CSMCI"]).tail(CHART_DAYS).copy()
    plot_df["日期"] = pd.to_datetime(plot_df["日期"])

    # 对齐宽基并归一
    norm_bench = pd.DataFrame()
    if not bench_df.empty:
        aligned = bench_df.copy()
        aligned.index = pd.to_datetime(aligned.index).normalize()
        # 与 CSMCI 日期对齐
        date_index = pd.to_datetime(plot_df["日期"]).dt.normalize()
        aligned = aligned.reindex(date_index.values).ffill()
        aligned.index = plot_df["日期"].values
        # 用 CSMCI 窗口内有效段归一
        sub = aligned.dropna(how="all")
        if not sub.empty:
            norm_bench = normalize_to_100(sub)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=plot_df["日期"],
            y=plot_df["CSMCI"],
            name="CSMCI",
            line=dict(color="#0ea5e9", width=2.5),
        ),
        secondary_y=False,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="#16a34a", line_width=1, secondary_y=False)
    fig.add_hline(y=30, line_dash="dot", line_color="#dc2626", line_width=1, secondary_y=False)

    palette = ["#f59e0b", "#8b5cf6", "#10b981", "#ef4444", "#64748b", "#ec4899"]
    if not norm_bench.empty:
        for i, col in enumerate(norm_bench.columns):
            fig.add_trace(
                go.Scatter(
                    x=norm_bench.index,
                    y=norm_bench[col],
                    name=f"{col}(归一)",
                    line=dict(color=palette[i % len(palette)], width=1.6, dash="solid"),
                    opacity=0.9,
                ),
                secondary_y=True,
            )
    elif "沪深300" in plot_df.columns:
        hs = plot_df[["日期", "沪深300"]].dropna()
        if not hs.empty:
            base = float(hs["沪深300"].iloc[0])
            if base:
                fig.add_trace(
                    go.Scatter(
                        x=hs["日期"],
                        y=hs["沪深300"] / base * 100,
                        name="沪深300(归一)",
                        line=dict(color="#64748b", width=1.6),
                    ),
                    secondary_y=True,
                )

    apply_chart_style(fig, height=520, title=f"近半年 CSMCI（左轴，近{used_n}日窗口） vs 宽基归一100（右轴）")
    fig.update_yaxes(title_text="CSMCI 信心分", range=[0, 100], secondary_y=False)
    fig.update_yaxes(title_text="价格归一（起点=100）", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # 简单相关
    if not norm_bench.empty and len(plot_df) > 10:
        corr_rows = []
        csmci_s = plot_df.set_index("日期")["CSMCI"]
        for col in norm_bench.columns:
            joined = pd.concat([csmci_s, norm_bench[col].rename(col)], axis=1).dropna()
            if len(joined) < 10:
                continue
            corr_rows.append({"标的": col, "与CSMCI相关系数": round(float(joined.corr().iloc[0, 1]), 3), "样本天数": len(joined)})
        if corr_rows:
            st.markdown("### 相关性（同区间）")
            st.dataframe(pd.DataFrame(corr_rows), hide_index=True, use_container_width=True)

    st.caption(
        "左轴 CSMCI，右轴宽基归一走势（便于看背离/同向）。"
        "指数跌而 CSMCI 升 → 机构偏买；指数涨而 CSMCI 降 → 机构偏撤。"
    )

    with st.expander("CSMCI 明细（近半年）", expanded=False):
        st.markdown(
            f"""
**列含义**：`futures_hold` IF前20净持仓 · `qvix` 波动率（逆向）· `northbound` 北向 · `margin` 融资（逆向）· `etf_flow` 宽基流向。  
分数均为近 60 日滚动百分位（0–100）；因子原始量用**近 {used_n} 日**窗口；逆向因子：原始指标越高 → 信心分越低。
"""
        )
        show_cols = ["日期", "CSMCI"] + [k for k in WEIGHTS if k in plot_df.columns]
        if "沪深300" in plot_df.columns:
            show_cols.append("沪深300")
        st.dataframe(plot_df[show_cols], hide_index=True, use_container_width=True)
        for key, meta in FACTOR_META.items():
            st.caption(f"{key}（{meta['name']}）：{factor_explain(key, used_n)}")
