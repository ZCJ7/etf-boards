import streamlit as st
import pandas as pd
import numpy as np
import datetime
from data import get_etf_list, fetch_etf_data_with_retry

st.set_page_config(page_title="ETF轮动回测", layout="wide")
st.title("🔄 ETF轮动回测（纯pandas版）")

# 参数设置
etf_list = get_etf_list()
if etf_list.empty:
    st.error("无法获取ETF列表，请检查网络连接或数据源是否可用")
    st.stop()

from utils import get_etf_options_with_favorites
from holdings_utils import get_all_symbols

user_defaults = [s for s in get_all_symbols(include_watchlist=True) if s in etf_list['symbol'].unique()]
if len(user_defaults) >= 2:
    default_etfs = user_defaults
else:
    default_etfs = [s for s in ['159915', '159941'] if s in etf_list['symbol'].unique()]
    if not default_etfs:
        default_etfs = etf_list['symbol'].unique()[:2]

etf_options = get_etf_options_with_favorites(etf_list)
selected_etfs = st.multiselect(
    "选择ETF (至少2只)",
    options=etf_options,
    format_func=lambda x: f"{x} - {etf_list[etf_list['symbol'] == x]['name'].values[0]}",
    default=default_etfs
)
start_date = st.date_input("开始日期", value=datetime.date(2020, 1, 1), min_value=datetime.date(2010, 1, 1))
end_date = st.date_input("结束日期", value=datetime.date.today(), min_value=start_date)
mom_window = st.number_input("动量窗口（天）", min_value=5, max_value=120, value=20, step=1)
hold_num = st.number_input("持仓ETF数量", min_value=1, max_value=len(selected_etfs) if selected_etfs else 2, value=2, step=1)
init_cash = st.number_input("初始资金 (元)", min_value=1000, value=10000, step=1000)
rebalance_freq = st.selectbox("调仓频率", options=[('M', '每月'), ('W', '每周'), ('Q', '每季度')], format_func=lambda x: x[1])
run_btn = st.button("运行轮动回测")

if run_btn:
    if len(selected_etfs) < 2:
        st.warning("请至少选择2只ETF")
        st.stop()
    st.info(f"开始获取 {len(selected_etfs)} 只ETF的数据...")
    price_df = pd.DataFrame()
    for symbol in selected_etfs:
        df = fetch_etf_data_with_retry(symbol, start_date, end_date, etf_list)
        if df.empty:
            st.warning(f"{symbol} 数据为空，已跳过")
            continue
        df = df.copy()
        df.columns = [symbol]
        price_df = pd.concat([price_df, df], axis=1)
    price_df = price_df.dropna(how='all')
    price_df = price_df.fillna(method='ffill').dropna(axis=0, how='any')
    if price_df.shape[1] < 2:
        st.error("有效ETF数量不足2，无法回测")
        st.stop()
    st.success(f"成功获取 {price_df.shape[1]} 只ETF的有效数据，{price_df.shape[0]} 个交易日")

    # 计算动量
    momentum = price_df.pct_change(int(mom_window))
    # 生成调仓日期
    freq_code = rebalance_freq[0]  # 'M'/'W'/'Q'
    rebalance_dates = price_df.resample(freq_code).last().index
    positions = pd.DataFrame(0, index=price_df.index, columns=price_df.columns)
    for date in rebalance_dates:
        if date not in momentum.index:
            continue
        top_etfs = momentum.loc[date].nlargest(int(hold_num)).index
        positions.loc[date, top_etfs] = 1 / int(hold_num)
    # 持仓信号前向填充
    positions = positions.shift(1).reindex(price_df.index).fillna(method='ffill').fillna(0)
    # 计算组合净值
    returns = price_df.pct_change().fillna(0)
    portfolio_returns = (positions * returns).sum(axis=1)
    nav = (1 + portfolio_returns).cumprod() * init_cash

    # 主要回测指标
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    days = (nav.index[-1] - nav.index[0]).days
    annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = (annual_return) / annual_vol if annual_vol > 0 else np.nan

    # 展示净值曲线
    st.subheader("📈 轮动策略净值曲线")
    import plotly.graph_objs as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav, mode='lines', name='轮动净值'))
    fig.update_layout(title="ETF轮动策略净值曲线", xaxis_title="日期", yaxis_title="资产净值 (元)")
    st.plotly_chart(fig, use_container_width=True)

    # 指标卡
    st.subheader("📊 主要回测指标")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总收益率", f"{total_return*100:.2f}%")
    with col2:
        st.metric("年化收益率", f"{annual_return*100:.2f}%")
    with col3:
        st.metric("最大回撤", f"{max_drawdown*100:.2f}%")
    with col4:
        st.metric("夏普比率", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")

    st.caption("本轮动回测为等权分配，不含手续费和分红，仅供参考。调仓频率可选：每月、每周、每季度。")