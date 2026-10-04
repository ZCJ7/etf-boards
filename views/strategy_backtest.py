"""ETF 策略回测页。"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from akshare_patch import install_akshare_patch

install_akshare_patch()

from analysis.backtest import BacktestConfig, run_pool_backtest
from etf_data_fetcher import fetch_pool_daily
from ui.chart_utils import apply_chart_style
from ui.sidebar import render_holdings_sidebar
from ui.styles import inject_global_styles, page_header

inject_global_styles()
pool = render_holdings_sidebar()
symbols = list(pool.keys())

page_header("📈 ETF 策略回测", "多指标过滤 + 动量轮动 · 支持自定义参数与调仓频率")

BACKTEST_GUIDE = """
### 回测在做什么？
在**同一持仓池**内，按规则定期选出得分最高的 ETF 等权持有，模拟历史收益与回撤。

### 指标与周期含义
| 参数 | 含义 | 调大/调小的影响 |
|------|------|----------------|
| **同时持有** | 每个调仓日持有得分前 N 只 | N=1 集中度高；N>1 更分散 |
| **调仓频率** | 每周 / 每月换仓 | 周频更灵敏、换手更高；月频更稳 |
| **趋势均线** | 收盘价需高于该均线才保留得分 | 周期越长越偏中长期趋势 |
| **BIAS周期/上限** | 乖离率过热时降权 | 上限越低越保守、少追高 |
| **MACD过滤** | DIF>DEA（金叉）才保留得分 | 关闭后更激进 |
| **佣金率** | 每次换仓按权重变化扣费 | 越高越贴近真实交易成本 |

### 得分逻辑（简化）
1. 计算各 ETF **20日涨跌幅**作为动量基础分  
2. 可选过滤：站上均线 / BIAS未过热 / MACD金叉  
3. 每个调仓日从池中选得分最高的 N 只**等权**持有  
4. 用各 ETF **公共上市重叠区间**的收盘价回测（避免新上市品种拉短样本）

### 结果指标
- **总收益率**：整个区间净值涨跌  
- **年化收益率**：按自然日折算的年化  
- **最大回撤**：净值从高点回落的最大幅度  
- **夏普比率**：收益波动比（越高越好，仅供参考）  
- **调仓次数**：实际发生换仓的次数  

### 常见问题
- **收益为 0**：多为过滤过严导致无法调仓，或池内 ETF 公共数据太短 → 放宽过滤、换更长历史的品种  
- **与实盘差异**：未含滑点、冲击成本；回测仅为历史模拟  
"""

if len(symbols) < 2:
    st.warning("回测至少需要 2 只 ETF，请先在侧边栏添加持仓。")
    st.stop()

with st.expander("📖 回测说明与教程（指标含义、参数选择）", expanded=False):
    st.markdown(BACKTEST_GUIDE)

st.subheader("参数设置")
selected = st.multiselect(
    "回测池（至少2只）",
    symbols,
    default=symbols[: min(5, len(symbols))],
    format_func=lambda x: f"{x} {pool.get(x, x)}",
    key="bt_pool",
)
b1, b2, b3, b4 = st.columns(4)
with b1:
    hold_count = st.number_input("同时持有", 1, 5, 1, key="bt_hold")
    rebalance = st.selectbox("调仓频率", ["W", "M"], format_func=lambda x: "每周" if x == "W" else "每月", key="bt_freq")
with b2:
    ma_p = st.number_input("趋势均线", 5, 120, 28, key="bt_ma")
    bias_p = st.number_input("BIAS周期", 5, 60, 12, key="bt_bias_p")
    bias_max = st.number_input("BIAS上限(%)", 1.0, 20.0, 8.0, key="bt_bias_max")
with b3:
    use_ma = st.checkbox("均线过滤", True, key="bt_use_ma")
    use_bias = st.checkbox("BIAS过滤", True, key="bt_use_bias")
    use_macd = st.checkbox("MACD过滤", True, key="bt_use_macd")
with b4:
    init_cash = st.number_input("初始资金", 10000, 5_000_000, 100_000, step=10000, key="bt_cash")
    commission = st.number_input("佣金率", 0.0001, 0.003, 0.0002, format="%.4f", key="bt_comm")

if st.button("运行回测", type="primary", key="btn_backtest"):
    st.session_state["bt_pending"] = True
    st.session_state.pop("bt_result", None)

if st.session_state.get("bt_pending") and "bt_result" not in st.session_state:
    if len(selected) < 2:
        st.warning("请至少选择 2 只 ETF")
        st.session_state["bt_pending"] = False
    else:
        with st.spinner("拉取池内数据并回测（首次可能较慢）..."):
            try:
                price_map = fetch_pool_daily(selected, pause_sec=0.25)
                cfg = BacktestConfig(
                    hold_count=int(hold_count),
                    rebalance_freq=rebalance,
                    ma_period=int(ma_p),
                    bias_period=int(bias_p),
                    bias_max=float(bias_max),
                    use_ma_filter=use_ma,
                    use_bias_filter=use_bias,
                    use_macd_filter=use_macd,
                    initial_cash=float(init_cash),
                    commission=float(commission),
                )
                result = run_pool_backtest(price_map, cfg)
                st.session_state["bt_result"] = {"ok": True, "result": result, "error": None, "selected": selected}
            except Exception as exc:
                st.session_state["bt_result"] = {"ok": False, "result": None, "error": str(exc), "selected": selected}
            st.session_state["bt_pending"] = False

if "bt_result" in st.session_state:
    br = st.session_state["bt_result"]
    if not br.get("ok"):
        st.error(f"回测失败：{br.get('error', '未知错误')}")
        st.info("可尝试：关闭部分过滤、减少池内新上市 ETF、或延长数据区间。")
    else:
        result = br["result"]
        m = result["metrics"]
        st.success(f"回测完成 · 样本 {m.get('回测天数', '—')} 天 · 自 {m.get('公共起始日', '—')}")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("总收益率", f"{m['总收益率']}%")
        c2.metric("年化收益率", f"{m['年化收益率']}%")
        c3.metric("最大回撤", f"{m['最大回撤']}%")
        c4.metric("夏普比率", m["夏普比率"])
        c5.metric("调仓次数", m["调仓次数"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result["nav"].index, y=result["nav"], name="策略净值", line=dict(color="#6366f1", width=2)))
        apply_chart_style(fig, height=420, title="回测净值曲线")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**调仓记录（最近20条）**")
        st.dataframe(result["trade_log"].tail(20), hide_index=True, use_container_width=True)
