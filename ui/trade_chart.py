"""在指标图上标注买卖点。"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from analysis.trade_records import TradeRecord, align_trade_date

_BUY_COLOR = "rgba(220, 38, 38, 0.85)"
_SELL_COLOR = "rgba(22, 163, 74, 0.85)"
_VLINE_BUY = "rgba(220, 38, 38, 0.28)"
_VLINE_SELL = "rgba(22, 163, 74, 0.28)"


def _trade_label(t: TradeRecord, aligned: pd.Timestamp) -> str:
    label = f"{t.action} · {aligned.strftime('%Y-%m-%d')}"
    if t.note:
        label += f" · {t.note[:24]}"
    if t.price is not None:
        label += f" @{t.price:.3f}"
    return label


def _trade_text_by_date(trades: list[TradeRecord], index: pd.DatetimeIndex) -> dict[pd.Timestamp, str]:
    out: dict[pd.Timestamp, str] = {}
    for t in trades:
        aligned = align_trade_date(t.date, index)
        if aligned is None or aligned not in index:
            continue
        label = _trade_label(t, aligned)
        if aligned in out:
            out[aligned] = f"{out[aligned]}<br>{label}"
        else:
            out[aligned] = label
    return out


def _collect_aligned_trades(
    trades: list[TradeRecord],
    index: pd.DatetimeIndex,
) -> tuple[list[tuple[pd.Timestamp, str, str]], list[tuple[pd.Timestamp, str, str]]]:
    buys: list[tuple[pd.Timestamp, str, str]] = []
    sells: list[tuple[pd.Timestamp, str, str]] = []
    for t in trades:
        aligned = align_trade_date(t.date, index)
        if aligned is None or aligned not in index:
            continue
        label = _trade_label(t, aligned)
        if t.action == "买入":
            buys.append((aligned, label, t.note or ""))
        elif t.action == "卖出":
            sells.append((aligned, label, t.note or ""))
    return buys, sells


def _apply_trade_hover_to_trace(fig: go.Figure, trace_name: str, index: pd.DatetimeIndex, trade_text: dict) -> None:
    """把交易信息写入已有连续曲线 trace 的 customdata，仅在交易日有内容。"""
    if not trade_text:
        return

    extras = [f"<br>{trade_text[d]}" if d in trade_text else "" for d in index]
    if not any(extras):
        return

    for trace in fig.data:
        if trace.name != trace_name:
            continue
        trace.customdata = extras
        trace.hovertemplate = f"{trace_name}: %{{y:.4f}}%{{customdata}}<extra></extra>"
        break


def add_trade_markers(
    fig: go.Figure,
    trades: list[TradeRecord],
    close: pd.Series,
    row: int = 1,
    col: int = 1,
    *,
    show_legend: bool = True,
) -> go.Figure:
    """价格图：小三角标记；悬停信息合并进「收盘价」曲线。"""
    if not trades or close.empty:
        return fig

    buys, sells = _collect_aligned_trades(trades, close.index)
    buy_x, buy_y = [], []
    sell_x, sell_y = [], []

    for aligned, _, _ in buys:
        buy_x.append(aligned)
        buy_y.append(float(close.loc[aligned]))
    for aligned, _, _ in sells:
        sell_x.append(aligned)
        sell_y.append(float(close.loc[aligned]))

    if buy_x:
        fig.add_trace(
            go.Scatter(
                x=buy_x,
                y=buy_y,
                mode="markers",
                name="买入",
                legendgroup="trade_buy",
                showlegend=show_legend,
                marker=dict(symbol="triangle-up", size=9, color=_BUY_COLOR, line=dict(width=0.5, color="#fff")),
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
    if sell_x:
        fig.add_trace(
            go.Scatter(
                x=sell_x,
                y=sell_y,
                mode="markers",
                name="卖出",
                legendgroup="trade_sell",
                showlegend=show_legend,
                marker=dict(symbol="triangle-down", size=9, color=_SELL_COLOR, line=dict(width=0.5, color="#fff")),
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    _apply_trade_hover_to_trace(fig, "收盘价", close.index, _trade_text_by_date(trades, close.index))
    return fig


def add_trade_vlines(
    fig: go.Figure,
    trades: list[TradeRecord],
    index: pd.DatetimeIndex,
    row: int,
    col: int = 1,
) -> go.Figure:
    """BIAS / MACD 子图：细竖线标交易日期，不压指标曲线。"""
    if not trades or index.empty:
        return fig

    buys, sells = _collect_aligned_trades(trades, index)
    for aligned, _, _ in buys:
        fig.add_vline(
            x=aligned,
            line_width=1,
            line_dash="dot",
            line_color=_VLINE_BUY,
            row=row,
            col=col,
        )
    for aligned, _, _ in sells:
        fig.add_vline(
            x=aligned,
            line_width=1,
            line_dash="dash",
            line_color=_VLINE_SELL,
            row=row,
            col=col,
        )
    return fig


def add_trade_value_markers(
    fig: go.Figure,
    trades: list[TradeRecord],
    series: pd.Series,
    row: int,
    col: int = 1,
    *,
    hover_trace_name: str | None = None,
) -> go.Figure:
    """在指标序列上用小圆点标买卖时点；悬停合并进主指标曲线。"""
    if not trades or series.empty:
        return fig

    buys, sells = _collect_aligned_trades(trades, series.index)
    if buys:
        fig.add_trace(
            go.Scatter(
                x=[a for a, _, _ in buys],
                y=[float(series.loc[a]) for a, _, _ in buys],
                mode="markers",
                name="买入",
                legendgroup="trade_buy",
                showlegend=False,
                marker=dict(symbol="circle", size=5, color=_BUY_COLOR, line=dict(width=0)),
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
    if sells:
        fig.add_trace(
            go.Scatter(
                x=[a for a, _, _ in sells],
                y=[float(series.loc[a]) for a, _, _ in sells],
                mode="markers",
                name="卖出",
                legendgroup="trade_sell",
                showlegend=False,
                marker=dict(symbol="circle-open", size=6, color=_SELL_COLOR, line=dict(width=1.5)),
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    if hover_trace_name:
        _apply_trade_hover_to_trace(
            fig,
            hover_trace_name,
            series.index,
            _trade_text_by_date(trades, series.index),
        )
    return fig


def annotate_all_trade_rows(
    fig: go.Figure,
    trades: list[TradeRecord],
    close: pd.Series,
    bias_df: pd.DataFrame | None = None,
    macd_df: pd.DataFrame | None = None,
) -> go.Figure:
    """在三联图的价格 / BIAS / MACD 上统一标注交易。"""
    if not trades:
        return fig

    add_trade_markers(fig, trades, close, row=1, col=1, show_legend=True)

    if bias_df is not None and not bias_df.empty:
        bias_col = "BIAS12" if "BIAS12" in bias_df.columns else bias_df.columns[0]
        add_trade_vlines(fig, trades, bias_df.index, row=2, col=1)
        add_trade_value_markers(
            fig, trades, bias_df[bias_col], row=2, col=1, hover_trace_name=bias_col
        )

    if macd_df is not None and not macd_df.empty:
        add_trade_vlines(fig, trades, macd_df.index, row=3, col=1)
        macd_col = "DIF" if "DIF" in macd_df.columns else macd_df.columns[0]
        add_trade_value_markers(
            fig, trades, macd_df[macd_col], row=3, col=1, hover_trace_name=macd_col
        )

    return fig
