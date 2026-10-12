"""手机页 Top5 动量模块：表格 + SVG 净值/回撤 + 精简分析。"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.momentum_top5 import run_all

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "momentum_top5"


def _pct(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:.1f}%"


def _num(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.2f}"


def _polyline(series: pd.Series, color: str, w=720, h=260, left=40, top=16, right=12, bottom=28) -> str:
    s = series.dropna()
    if len(s) < 2:
        return ""
    vals = s.to_numpy(dtype=float)
    ymin, ymax = float(vals.min()), float(vals.max())
    if ymin == ymax:
        ymin -= 1
        ymax += 1
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad
    pw, ph = w - left - right, h - top - bottom
    pts = []
    for i, v in enumerate(vals):
        x = left + i / (len(vals) - 1) * pw
        y = top + (1 - (v - ymin) / (ymax - ymin)) * ph
        pts.append(f"{x:.1f},{y:.1f}")
    y0 = top + (1 - (0 - ymin) / (ymax - ymin)) * ph if ymin <= 0 <= ymax else None
    zero = (
        f"<line x1='{left}' y1='{y0:.1f}' x2='{left + pw}' y2='{y0:.1f}' class='guide'/>"
        if y0 is not None
        else ""
    )
    label = (
        f"<text x='{left}' y='{h - 8}' class='axis'>{html.escape(str(s.index[0].date()))}</text>"
        f"<text x='{left + pw}' y='{h - 8}' text-anchor='end' class='axis'>{html.escape(str(s.index[-1].date()))}</text>"
        f"<text x='4' y='{top + 10}' class='axis'>{ymax:.2f}</text>"
        f"<text x='4' y='{top + ph}' class='axis'>{ymin:.2f}</text>"
    )
    return (
        f"<div class='chartbox'><svg viewBox='0 0 {w} {h}' class='chart' role='img'>"
        f"<rect x='{left}' y='{top}' width='{pw}' height='{ph}' class='plot'/>"
        f"{zero}"
        f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(pts)}'/>"
        f"{label}</svg></div>"
    )


def _multi_nav_chart(navs: dict[str, pd.Series], w=720, h=280) -> str:
    colors = {
        "Top5动量": "#dc2626",
        "Top1动量": "#b45309",
        "观察池等权": "#2563eb",
        "沪深300持有": "#64748b",
    }
    left, top, right, bottom = 40, 16, 12, 28
    pw, ph = w - left - right, h - top - bottom
    series_list = [(k, v.dropna()) for k, v in navs.items() if v is not None and len(v.dropna()) > 2]
    if not series_list:
        return ""
    # align roughly by index union length using values only for scale
    all_vals = np.concatenate([s.to_numpy(dtype=float) for _, s in series_list])
    ymin, ymax = float(all_vals.min()), float(all_vals.max())
    if ymin == ymax:
        ymin -= 0.1
        ymax += 0.1
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad
    polylines = []
    legend = []
    for name, s in series_list:
        vals = s.to_numpy(dtype=float)
        pts = []
        for i, v in enumerate(vals):
            x = left + i / (len(vals) - 1) * pw
            y = top + (1 - (v - ymin) / (ymax - ymin)) * ph
            pts.append(f"{x:.1f},{y:.1f}")
        color = colors.get(name, "#0f172a")
        polylines.append(
            f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(pts)}'/>"
        )
        legend.append(f"<span><i style='background:{color}'></i>{html.escape(name)}</span>")
    label = (
        f"<text x='{left}' y='{h - 8}' class='axis'>{html.escape(str(series_list[0][1].index[0].date()))}</text>"
        f"<text x='{left + pw}' y='{h - 8}' text-anchor='end' class='axis'>{html.escape(str(series_list[0][1].index[-1].date()))}</text>"
    )
    return (
        f"<div class='legend'>{''.join(legend)}</div>"
        f"<div class='chartbox'><svg viewBox='0 0 {w} {h}' class='chart' role='img'>"
        f"<rect x='{left}' y='{top}' width='{pw}' height='{ph}' class='plot'/>"
        f"{''.join(polylines)}{label}</svg></div>"
    )


def build_momentum_html(result: dict | None = None) -> tuple[str, dict]:
    result = result or run_all()
    top = result["top"]
    bt = result["backtest"]
    payload = {
        "top": top,
        "compare": bt.get("compare").to_dict(orient="records") if isinstance(bt.get("compare"), pd.DataFrame) else [],
        "start": bt.get("start"),
        "end": bt.get("end"),
        "analysis": bt.get("analysis"),
        "issues": result.get("issues") or bt.get("issues") or [],
    }
    if bt.get("error"):
        body = f"<section id='momentum'><h2>Top5 动量</h2><p class='note'>回测失败：{html.escape(str(bt['error']))}</p></section>"
        return body, payload

    rows_html = "".join(
        "<tr>"
        f"<td>{r['排名']}</td><td>{html.escape(r['ETF名称'])}</td><td>{r['代码']}</td>"
        f"<td>{r['动量评分']}</td><td>{r['近20日收益率'] if r['近20日收益率'] is not None else '—'}%</td>"
        "</tr>"
        for r in top.get("rows") or []
    )
    if not rows_html:
        rows_html = "<tr><td colspan='5'>当前没有符合条件的 ETF，视为现金</td></tr>"

    compare_rows = []
    for r in payload["compare"]:
        compare_rows.append(
            "<tr>"
            f"<td>{html.escape(r['策略'])}</td>"
            f"<td>{_pct(r['累计收益率'])}</td>"
            f"<td>{_pct(r['年化收益率CAGR'])}</td>"
            f"<td>{_pct(r['最大回撤'])}</td>"
            f"<td>{_pct(r['年化波动率'])}</td>"
            f"<td>{_num(r['夏普比率'])}</td>"
            f"<td>{_num(r['卡玛比率'])}</td>"
            f"<td>{int(r['调仓次数'] or 0)}</td>"
            f"<td>{float(r['交易成本'] or 0):,.0f}</td>"
            "</tr>"
        )

    annual = bt["top5"]["metrics"].get("年度收益率")
    annual_html = ""
    if isinstance(annual, pd.Series) and len(annual):
        cells = "".join(
            f"<tr><td>{html.escape(str(y))}</td><td>{_pct(float(v))}</td></tr>" for y, v in annual.items()
        )
        annual_html = (
            "<h3>Top5 年度收益</h3><div class='wrap'><table><thead><tr><th>年份</th><th>收益率</th></tr></thead>"
            f"<tbody>{cells}</tbody></table></div>"
        )

    analysis_html = "".join(
        f"<p class='note'>{html.escape(line)}</p>"
        for line in (bt.get("analysis") or "").splitlines()
        if line.strip()
    )
    issues = payload["issues"]
    issue_html = ""
    if issues:
        items = "".join(f"<li>{html.escape(x)}</li>" for x in issues[:8])
        issue_html = f"<details><summary>数据说明</summary><ul>{items}</ul></details>"

    nav_chart = _multi_nav_chart(bt.get("navs") or {})
    dd = bt["top5"]["metrics"].get("回撤序列")
    dd_chart = _polyline(dd, "#b45309") if isinstance(dd, pd.Series) else ""

    body = f"""
<section id='momentum'>
  <h2>Top5 动量</h2>
  <p class='note'>近25日加权回归动量：Score = 年化斜率收益 × R²。筛选：斜率&gt;0、收盘&gt;MA20、近5日均成交额≥2000万。月末收盘出信号，下一交易日开盘调仓；不足5只时剩余为现金。单边成本 0.05%。缺成交额时用成交量×收盘×100 近似。</p>
  <p class='note'>信号日 {html.escape(top.get('signal_date') or '')} · 行情截止 {html.escape(top.get('asof') or '')} · 计划调仓 {html.escape(str(top.get('plan_rebalance') or ''))} · 符合条件 {top.get('eligible_count', 0)} 只</p>
  <h3>当前清单</h3>
  <div class='wrap'><table><thead><tr><th>排名</th><th>ETF 名称</th><th>代码</th><th>动量评分</th><th>近20日收益率</th></tr></thead><tbody>{rows_html}</tbody></table></div>
  <h3>回测对比 {html.escape(bt.get('start') or '')} ~ {html.escape(bt.get('end') or '')}</h3>
  <div class='wrap'><table><thead><tr><th>策略</th><th>累计收益</th><th>年化</th><th>最大回撤</th><th>年化波动</th><th>夏普</th><th>卡玛</th><th>调仓次数</th><th>交易成本</th></tr></thead><tbody>{''.join(compare_rows)}</tbody></table></div>
  <h3>累计净值</h3>
  {nav_chart}
  <h3>Top5 回撤</h3>
  {dd_chart}
  {annual_html}
  <h3>精简分析</h3>
  {analysis_html}
  {issue_html}
</section>
"""
    return body, payload


def save_momentum_artifacts(result: dict | None = None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_body, payload = build_momentum_html(result)
    (OUT_DIR / "momentum.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "fragment.html").write_text(html_body, encoding="utf-8")
    return {"html": html_body, "payload": payload}
