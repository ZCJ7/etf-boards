"""ETF 多指标轮动回测引擎。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.indicators import calc_bias, calc_macd, calc_ma, ensure_close


@dataclass
class BacktestConfig:
    hold_count: int = 1
    rebalance_freq: str = "W"  # W/M
    ma_period: int = 28
    bias_period: int = 12
    bias_max: float = 8.0
    use_macd_filter: bool = True
    use_ma_filter: bool = True
    use_bias_filter: bool = True
    initial_cash: float = 100_000.0
    commission: float = 0.0002


def _score_symbol(close: pd.Series, cfg: BacktestConfig) -> pd.Series:
    momentum = close.pct_change(20)
    ma = close.rolling(cfg.ma_period).mean()
    bias = (close - close.rolling(cfg.bias_period).mean()) / close.rolling(cfg.bias_period).mean() * 100
    macd = calc_macd(close)
    score = momentum.rank(pct=True) * 0.5
    if cfg.use_ma_filter:
        score = score.where(close > ma, score * 0.2)
    if cfg.use_bias_filter:
        score = score.where(bias < cfg.bias_max, score * 0.3)
    if cfg.use_macd_filter:
        score = score.where(macd["DIF"] > macd["DEA"], score * 0.4)
    return score


def _rebalance_trading_dates(index: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    """按实际交易日取调仓日，避免 resample 产生周日等不在行情索引中的日期。"""
    if len(index) == 0:
        return []
    series = pd.Series(1, index=index)
    grouper = "W" if freq == "W" else "ME"
    dates: list[pd.Timestamp] = []
    for _, grp in series.groupby(pd.Grouper(freq=grouper)):
        if len(grp):
            dates.append(grp.index[-1])
    return dates


def _align_price_panel(price_map: dict[str, pd.DataFrame], min_bars: int = 60) -> pd.DataFrame:
    closes: dict[str, pd.Series] = {}
    for sym, df in price_map.items():
        s = ensure_close(df)
        if not s.empty:
            closes[sym] = s
    if len(closes) < 2:
        raise ValueError("有效 ETF 少于 2 只")

    raw = pd.DataFrame(closes).sort_index().ffill()
    common_start = max(s.first_valid_index() for s in closes.values())
    panel = raw.loc[common_start:].dropna(how="any")
    if panel.shape[1] < 2:
        raise ValueError("公共重叠区间不足 2 只 ETF")
    if len(panel) < min_bars:
        raise ValueError(f"公共交易日数据不足（需至少 {min_bars} 天，当前 {len(panel)} 天）")
    return panel


def run_pool_backtest(price_map: dict[str, pd.DataFrame], cfg: BacktestConfig) -> dict:
    warmup = max(cfg.ma_period, cfg.bias_period, 26) + 5
    min_bars = max(40, warmup + 10)
    price_df = _align_price_panel(price_map, min_bars=min_bars)

    scores = pd.DataFrame({sym: _score_symbol(price_df[sym], cfg) for sym in price_df.columns})
    rebalance_dates = _rebalance_trading_dates(price_df.index, cfg.rebalance_freq)
    positions = pd.DataFrame(0.0, index=price_df.index, columns=price_df.columns)

    trade_log = []
    for date in rebalance_dates:
        if date not in scores.index or date < price_df.index[warmup]:
            continue
        day_scores = scores.loc[date].dropna().sort_values(ascending=False)
        picks = day_scores.head(cfg.hold_count).index.tolist()
        if not picks:
            continue
        weight = 1.0 / len(picks)
        positions.loc[date, picks] = weight
        trade_log.append({"日期": date.strftime("%Y-%m-%d"), "持仓": ",".join(picks), "权重": weight})

    if not trade_log:
        raise ValueError("调仓记录为空：请放宽过滤条件或延长回测区间")

    positions = positions.shift(1).ffill().fillna(0)
    returns = price_df.pct_change().fillna(0)
    gross = (positions * returns).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    net = gross - turnover * cfg.commission
    nav = (1 + net).cumprod() * cfg.initial_cash

    total_return = nav.iloc[-1] / cfg.initial_cash - 1
    days = max((nav.index[-1] - nav.index[0]).days, 1)
    annual_return = (1 + total_return) ** (365 / days) - 1
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    sharpe = net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else 0

    return {
        "nav": nav,
        "positions": positions,
        "trade_log": pd.DataFrame(trade_log),
        "metrics": {
            "总收益率": round(float(total_return * 100), 2),
            "年化收益率": round(float(annual_return * 100), 2),
            "最大回撤": round(float(max_drawdown * 100), 2),
            "夏普比率": round(float(sharpe), 2),
            "调仓次数": len(trade_log),
            "回测天数": len(price_df),
            "公共起始日": price_df.index[0].strftime("%Y-%m-%d"),
        },
    }
