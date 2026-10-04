"""Plotly 图表工具：主题与日期轴。"""

from __future__ import annotations

import plotly.graph_objects as go


CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, Microsoft YaHei, sans-serif", size=12, color="#334155"),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)


def apply_chart_style(fig: go.Figure, height: int = 520, title: str = "") -> go.Figure:
    fig.update_layout(**CHART_LAYOUT, height=height, title=dict(text=title, x=0))
    apply_numeric_date_axis(fig)
    return fig


def apply_numeric_date_axis(fig: go.Figure) -> go.Figure:
    """X 轴刻度到月；悬停显示完整年月日。"""
    fig.update_xaxes(
        tickformat="%Y/%m",
        hoverformat="%Y-%m-%d",
        dtick="M1",
        tickangle=0,
        showgrid=True,
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", linecolor="#cbd5e1")
    return fig
