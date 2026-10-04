"""市场资讯页：大盘复盘 + 多源资讯（对齐 daily-stock-analysis）+ AI 提炼。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from akshare_patch import install_akshare_patch

install_akshare_patch()

from ai_utils import ai_chat, get_api_key
from analysis.market_news import (
    AI_SYSTEM,
    CASE_NOTE,
    SOURCES,
    build_ai_digest_prompt,
    fetch_news_bundle,
    filter_signal_related,
    sources_table,
)
from analysis.market_review import fetch_cn_market_review
from analysis.stocksight_intel import fetch_hard_info_today, merge_hard_and_market
from etf_data_fetcher import data_asof_str
from ui.sidebar import render_holdings_sidebar
from ui.styles import inject_global_styles, page_header

inject_global_styles()
render_holdings_sidebar()

page_header("📰 市场资讯", "大盘复盘 · 多源资讯 · AI 提炼（对齐 daily-stock-analysis）", asof=data_asof_str())

with st.expander("能力说明", expanded=False):
    st.markdown(
        """
本页对齐 **daily-stock-analysis** 的大盘复盘，并整合 **StockSight** 情报纪律到 AI：

1. **大盘复盘**：指数 / 涨跌家数 / 板块  
2. **硬信息优先**：今日公司大事 + 高价值公告（排雷/催化）  
3. **市场快讯**：多源今日资讯（次级证据）  
4. **AI 提炼**：硬信息 > 快讯；公告≠自动买卖信号  

不包含：定时日报推送、买卖点决策卡。
"""
    )
    st.markdown(CASE_NOTE)
    st.dataframe(sources_table(), hide_index=True, use_container_width=True)

# —— 控制区 ——
default_keys = [k for k, s in SOURCES.items() if s.default_on]
selected = st.multiselect(
    "资讯源",
    options=list(SOURCES.keys()),
    default=default_keys,
    format_func=lambda k: f"{SOURCES[k].name}（及时性:{SOURCES[k].timeliness}）",
    key="intel_sources",
)
c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
with c1:
    time_window = st.selectbox(
        "时效",
        options=["今日", "不限"],
        index=0,
        help="今日=只保留当天发布的资讯（推荐）；不限=保留接口返回的全部近期快讯",
        key="intel_time_window",
    )
with c2:
    filter_mode = st.selectbox(
        "资讯筛选",
        options=["全部", "信号", "主升", "风险", "观察"],
        index=0,
        help="默认「全部」看今日全貌；「信号」仅 A股风险/主升",
        key="intel_filter_mode",
    )
with c3:
    st.caption("默认「今日 + 全部」。先刷新，再 AI 提炼。")

today_only = time_window == "今日"

b_fetch, b_ai, b_clear = st.columns([2, 2, 1])
with b_fetch:
    do_fetch = st.button("刷新大盘与资讯", type="primary", key="btn_intel_fetch")
with b_ai:
    do_ai = st.button("AI 提炼分析", key="btn_intel_ai")
with b_clear:
    if st.button("清缓存", key="btn_intel_clear"):
        try:
            fetch_news_bundle.clear()
            fetch_cn_market_review.clear()
            fetch_hard_info_today.clear()
        except Exception:
            pass
        st.session_state.pop("intel_bundle", None)
        st.session_state.pop("intel_review", None)
        st.session_state.pop("intel_hard", None)
        st.session_state.pop("intel_ai", None)
        st.success("已清除")

if do_fetch:
    with st.status("正在拉取大盘复盘与资讯…", expanded=True) as status:
        status.write("大盘复盘（指数 / 广度 / 板块）…")
        try:
            review = fetch_cn_market_review()
            st.session_state["intel_review"] = review
            if review.get("error"):
                status.write(f"大盘：{review['error']}")
            else:
                status.write(f"大盘完成 · 情绪 {review.get('mood','—')}")
        except Exception as exc:
            st.session_state["intel_review"] = {"error": str(exc), "markdown": "", "status": {}}
            status.write(f"大盘失败：{exc}")

        if selected:
            status.write("市场快讯…")
            for k in selected:
                status.write(f"  · {SOURCES[k].name}")
            bundle = fetch_news_bundle(
                tuple(selected),
                limit_per_source=500,
                today_only=today_only,
            )
        else:
            bundle = {
                "news": pd.DataFrame(),
                "status": {},
                "error": "未选择资讯源",
                "today_only": today_only,
            }
            status.write("未选择资讯源")

        status.write("StockSight 硬信息（公司大事/高价值公告）…")
        try:
            hard = fetch_hard_info_today()
            st.session_state["intel_hard"] = hard
            for k, v in (hard.get("status") or {}).items():
                status.write(f"  · {k}: {v}")
        except Exception as exc:
            hard = {"news": pd.DataFrame(), "status": {}, "error": str(exc)}
            st.session_state["intel_hard"] = hard
            status.write(f"硬信息失败：{exc}")

        market_news = bundle.get("news") if isinstance(bundle.get("news"), pd.DataFrame) else pd.DataFrame()
        hard_news = hard.get("news") if isinstance(hard.get("news"), pd.DataFrame) else pd.DataFrame()
        merged = merge_hard_and_market(market_news, hard_news)
        bundle = dict(bundle)
        bundle["news"] = merged
        bundle["hard_status"] = hard.get("status") or {}
        st.session_state["intel_bundle"] = bundle

        news_df = bundle.get("news")
        n = len(news_df) if isinstance(news_df, pd.DataFrame) else 0
        n_hard = int((news_df["信息层级"] == "硬信息").sum()) if isinstance(news_df, pd.DataFrame) and "信息层级" in news_df.columns else 0
        day = bundle.get("as_of_day") or ""
        scope = "今日" if bundle.get("today_only") else "不限"
        status.write(f"合并完成（共 {n} 条 · 硬信息 {n_hard} · {scope} {day}）")

        st.session_state.pop("intel_ai", None)
        status.update(label="刷新完成", state="complete")

review = st.session_state.get("intel_review")
bundle = st.session_state.get("intel_bundle")

if not review and not bundle:
    st.info("尚未拉取。点击 **「刷新大盘与资讯」** 开始。")
    st.stop()

# —— 1. 大盘复盘 ——
st.markdown("### 🎯 大盘复盘")
if not review:
    st.warning("尚无大盘数据，请刷新。")
else:
    st_map = review.get("status") or {}
    if st_map:
        st.caption(" · ".join(f"{k}:{v}" for k, v in st_map.items()))
    if review.get("error") and not review.get("markdown"):
        st.error(review["error"])
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("市场情绪", review.get("mood") or "—")
        m2.metric("上涨", review.get("up_count") if review.get("up_count") is not None else "—")
        m3.metric("下跌", review.get("down_count") if review.get("down_count") is not None else "—")
        m4.metric("涨停", review.get("limit_up") if review.get("limit_up") is not None else "—")
        m5.metric("跌停", review.get("limit_down") if review.get("limit_down") is not None else "—")
        if review.get("as_of"):
            st.caption(f"统计时点：{review['as_of']} · 活跃度 {review.get('activity_pct') or '—'}")

        idx_df = review.get("indices")
        if isinstance(idx_df, pd.DataFrame) and not idx_df.empty:
            st.markdown("**主要指数**")
            show_idx = idx_df.copy()
            if "涨跌幅%" in show_idx.columns:
                show_idx["涨跌幅%"] = pd.to_numeric(show_idx["涨跌幅%"], errors="coerce").map(
                    lambda x: f"{x:+.2f}" if pd.notna(x) else "—"
                )
            st.dataframe(show_idx, hide_index=True, use_container_width=True)

        sec_cols = st.columns(2)
        with sec_cols[0]:
            st.markdown("**领涨板块 Top5**")
            tops = review.get("top_sectors") or []
            if tops:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "排名": i,
                                "板块": s.get("name"),
                                "涨跌幅%": s.get("change_pct"),
                                "领涨股": s.get("leader", ""),
                            }
                            for i, s in enumerate(tops, 1)
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("暂无")
        with sec_cols[1]:
            st.markdown("**领跌板块 Top5**")
            bots = review.get("bottom_sectors") or []
            if bots:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "排名": i,
                                "板块": s.get("name"),
                                "涨跌幅%": s.get("change_pct"),
                            }
                            for i, s in enumerate(bots, 1)
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("暂无")

        with st.expander("复盘 Markdown（供核对 / AI 输入）", expanded=False):
            st.markdown(review.get("markdown") or "_空_")

# —— 2. 市场资讯 ——
st.markdown("### 📰 市场资讯")
if not bundle:
    st.warning("尚无资讯，请刷新。")
else:
    status_map = bundle.get("status") or {}
    if status_map:
        st.dataframe(
            pd.DataFrame(
                [{"源": SOURCES[k].name if k in SOURCES else k, "状态": v} for k, v in status_map.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )

    news: pd.DataFrame = bundle.get("news") if isinstance(bundle.get("news"), pd.DataFrame) else pd.DataFrame()
    if news.empty:
        st.warning(bundle.get("error") or "暂无资讯")
    else:
        view = filter_signal_related(news, mode=filter_mode)
        if view.empty and filter_mode != "全部":
            st.warning(f"当前没有命中「{filter_mode}」的条目，已展示全部。")
            view = news

        n_risk = int(news["信号类型"].isin(["风险", "风险+主升"]).sum()) if "信号类型" in news.columns else 0
        n_bull = int(news["信号类型"].isin(["主升", "风险+主升"]).sum()) if "信号类型" in news.columns else 0
        n_hard = int((news["信息层级"] == "硬信息").sum()) if "信息层级" in news.columns else 0
        n_watch = int(news["信号类型"].isin(["外围观察", "背景"]).sum()) if "信号类型" in news.columns else 0
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("拉取条数", len(news))
        a2.metric("硬信息", n_hard)
        a3.metric("A股风险", n_risk)
        a4.metric("A股主升", n_bull)
        a5.metric("观察/背景", n_watch)
        day = bundle.get("as_of_day") or ""
        if bundle.get("today_only"):
            st.caption(
                f"时效：**今日**（{day}）· StockSight：**硬信息优先**（公告/大事排在前，供 AI 排雷）"
            )
        else:
            st.caption("时效：不限 · StockSight 硬信息优先")

        # 展示时段覆盖
        if "时间" in news.columns and not news.empty:
            try:
                from analysis.market_news import _parse_news_time

                ts = _parse_news_time(news["时间"]).dropna()
                if not ts.empty:
                    st.caption(f"时间覆盖：{ts.min()} → {ts.max()}")
            except Exception:
                pass

        show_cols = [
            c
            for c in (
                "时间",
                "信息层级",
                "StockSight类别",
                "来源分级",
                "可信度",
                "来源",
                "领域",
                "投资因子",
                "信号类型",
                "标题",
                "内容",
                "风险标记",
                "主升标记",
                "链接",
            )
            if c in view.columns
        ]
        st.dataframe(view[show_cols], hide_index=True, use_container_width=True, height=400)

# —— 3. AI ——
news_for_ai = pd.DataFrame()
if bundle and isinstance(bundle.get("news"), pd.DataFrame):
    news_for_ai = filter_signal_related(bundle["news"], mode=filter_mode)
    if news_for_ai.empty:
        news_for_ai = bundle["news"]

market_md = (review or {}).get("markdown") or ""

if do_ai:
    api_key = get_api_key()
    if not api_key:
        st.warning("未检测到 API Key。请到「更多工具 → AI助手与API密钥配置」设置 DashScope Key。")
    elif news_for_ai.empty and not market_md.strip():
        st.warning("请先刷新大盘或资讯，再进行 AI 提炼。")
    else:
        with st.spinner("AI 正在按 StockSight 纪律提炼（硬信息优先）…"):
            prompt = build_ai_digest_prompt(news_for_ai, market_markdown=market_md)
            result = ai_chat(prompt, api_key=api_key, system_prompt=AI_SYSTEM)
            st.session_state["intel_ai"] = result

ai_text = st.session_state.get("intel_ai")
st.markdown("### 🤖 AI 提炼分析（StockSight）")
if ai_text:
    st.markdown(ai_text)
    st.caption(
        "已启用 StockSight：硬信息（公告/大事）优先于快讯；公告用于排雷/催化，不自动等于买卖信号。"
        "结论不构成投资建议。"
    )
else:
    st.info("数据就绪后，点击 **「AI 提炼分析」**（硬信息优先进入模型）。")
