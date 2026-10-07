"""拉取更新时刻可用行情，重算低位 / 反转 / 动量榜并写入 data/boards。

本地：
    python scripts/refresh_boards.py

GitHub Actions：手动 Run workflow 调用本脚本，发布 GitHub Pages。
交易时段用当天已走出的未收盘日K（例如 14:30 的价格）；收盘后即为收盘价。
东财不通则整次刷新改走新浪。
"""

from __future__ import annotations

import html
import json
import os
import sys
import traceback
from pathlib import Path

os.environ["ETF_USE_LIVE_BAR"] = "1"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akshare_patch import install_akshare_patch

install_akshare_patch()

from analysis.board_store import BOARD_DIR, load_boards_snapshot, now_cst_iso, save_boards_snapshot
from analysis.low_position_screener import get_low_position_top30
from analysis.reversal_screener import get_reversal_top30
from analysis.screener import get_momentum_top30
from etf_data_fetcher import active_hist_source, data_asof_str, hist_prefer_sina, live_bar_note


def _run_one(label: str, fn):
    print(f"==> {label}", flush=True)
    try:
        df = fn()
        n = 0 if df is None else len(df)
        print(f"    ok  {n} 行", flush=True)
        return df, None
    except Exception as exc:
        print(f"    fail  {exc}", flush=True)
        traceback.print_exc()
        return None, str(exc)


def _sample_series(series, keep: int = 240):
    import pandas as pd

    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return s
    step = max(1, len(s) // keep)
    sampled = s.iloc[::step]
    if sampled.index[-1] != s.index[-1]:
        sampled = pd.concat([sampled, s.iloc[[-1]]])
    return sampled


def _polyline(series, width: float, height: float, ymin: float, ymax: float, left: float, top: float) -> str:
    sampled = _sample_series(series)
    if sampled.empty:
        return ""
    span = ymax - ymin or 1.0
    n = len(sampled)
    pts = []
    for i, value in enumerate(sampled.to_numpy(dtype=float)):
        x = left + (i / (n - 1) if n > 1 else 0.0) * width
        y = top + (1 - (float(value) - ymin) / span) * height
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def _chart_svg(lines: list[dict], y_from: float | None = None, y_to: float | None = None, guides: list[float] | None = None) -> str:
    import pandas as pd

    usable = [line for line in lines if line["series"] is not None and not pd.to_numeric(line["series"], errors="coerce").dropna().empty]
    if not usable:
        return "<p class='empty'>暂无图形</p>"
    width, height = 720.0, 280.0
    left, right, top, bottom = 46.0, 12.0, 16.0, 28.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = []
    for line in usable:
        values.extend(float(v) for v in pd.to_numeric(line["series"], errors="coerce").dropna().tolist())
    ymin = min(values) if y_from is None else y_from
    ymax = max(values) if y_to is None else y_to
    if ymin == ymax:
        ymin -= 1
        ymax += 1
    pad = (ymax - ymin) * 0.06
    if y_from is None:
        ymin -= pad
    if y_to is None:
        ymax += pad
    parts = [
        f"<svg viewBox='0 0 {width:.0f} {height:.0f}' role='img' class='chart'>",
        f"<rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' class='plot'/>",
    ]
    for guide in guides or []:
        if ymin <= guide <= ymax:
            gy = top + (1 - (guide - ymin) / (ymax - ymin)) * plot_h
            parts.append(f"<line x1='{left}' y1='{gy:.1f}' x2='{left + plot_w}' y2='{gy:.1f}' class='guide'/>")
    for line in usable:
        pts = _polyline(line["series"], plot_w, plot_h, ymin, ymax, left, top)
        dash = " stroke-dasharray='6 4'" if line.get("dash") else ""
        parts.append(f"<polyline fill='none' stroke='{line['color']}' stroke-width='2'{dash} points='{pts}'/>")
    first = _sample_series(usable[0]["series"])
    labels = [first.index[0], first.index[len(first) // 2], first.index[-1]]
    for i, ts in enumerate(labels):
        x = left + (i / 2) * plot_w
        anchor = ("start", "middle", "end")[i]
        parts.append(f"<text x='{x:.1f}' y='{height - 8}' text-anchor='{anchor}' class='axis'>{ts.strftime('%Y-%m')}</text>")
    for i, tick in enumerate((ymax, (ymax + ymin) / 2, ymin)):
        y = top + i * plot_h / 2
        span = ymax - ymin
        fmt = ".0f" if span >= 20 else (".2f" if span >= 1 else ".3f")
        parts.append(
            f"<text x='{left - 6}' y='{y + 4:.1f}' text-anchor='end' class='axis'>{format(tick, fmt)}</text>"
        )
    parts.append("</svg>")
    legend = "".join(
        f"<span><i style='background:{html.escape(line['color'])}'></i>{html.escape(line['name'])}</span>"
        for line in usable
    )
    return f"<div class='chartbox'>{''.join(parts)}<p class='legend'>{legend}</p></div>"


def _pct_text(value) -> str:
    return "—" if value is None else f"{float(value):.1f}%"


def _style_section_html() -> tuple[str, str | None]:
    from concurrent.futures import ThreadPoolExecutor

    from analysis.style_ratio import WINDOWS, build_style_ratio, load_chinext_dividend, read_price_volume

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pair = pool.submit(load_chinext_dividend).result(timeout=60)
    except Exception as exc:
        return (
            "<section><h2>创业板 ÷ 中证红利低波</h2>"
            f"<p class='empty'>这次没拉到比价：{html.escape(str(exc)[:180])}</p></section>",
            str(exc),
        )

    views = {name: build_style_ratio(pair, window, ma_period=20) for name, window in WINDOWS.items()}
    panels = []
    for name, view in views.items():
        last = view["ratio"].iloc[-1]
        ratio_now = float(last["比价"])
        ma_col = f"比价MA{view['ma_period']}"
        cards = (
            f"<div class='metric'><span>{html.escape(pair['chinext_name'])}历史分位</span><b>{_pct_text(view['chinext_pct'])}</b></div>"
            f"<div class='metric'><span>{html.escape(pair['div_name'])}历史分位</span><b>{_pct_text(view['div_pct'])}</b></div>"
            f"<div class='metric'><span>当前比价</span><b>{ratio_now:.4f}</b></div>"
            f"<div class='metric'><span>比价历史分位</span><b>{_pct_text(view['ratio_pct'])}</b></div>"
        )
        ratio_svg = _chart_svg(
            [
                {"name": "创业板指/红利低波", "series": view["ratio"]["比价"], "color": "#dc2626"},
                {"name": ma_col, "series": view["ratio"][ma_col], "color": "#2563eb", "dash": True},
            ]
        )
        pct_svg = _chart_svg(
            [
                {"name": "创业板指历史分位", "series": view["chinext_pct_path"], "color": "#dc2626"},
                {"name": "红利低波历史分位", "series": view["div_pct_path"], "color": "#d97706"},
                {"name": "比价历史分位", "series": view["ratio_pct_path"], "color": "#64748b", "dash": True},
            ],
            y_from=0,
            y_to=100,
            guides=[20, 80],
        )
        hidden = "" if name == "近5年" else " hidden"
        panels.append(
            f"<div class='style-panel' data-win='{html.escape(name)}'{hidden}>"
            f"<div class='metrics'>{cards}</div>"
            f"<h3>比价趋势</h3>{ratio_svg}"
            f"<h3>历史分位</h3>{pct_svg}"
            "</div>"
        )
    options = "".join(
        f"<option value='{html.escape(name)}'{' selected' if name == '近5年' else ''}>{html.escape(name)}</option>"
        for name in views
    )
    pv = read_price_volume(pair["chinext"], pair.get("chinext_volume"))
    pv_lines = [
        {"name": "创业板指", "series": pv["chart"]["收盘"], "color": "#dc2626"},
        {"name": "MA20", "series": pv["chart"]["MA20"], "color": "#2563eb", "dash": True},
        {"name": "MA60", "series": pv["chart"]["MA60"], "color": "#7c3aed", "dash": True},
    ]
    note = pair.get("note") or ""
    vol_bits = []
    if pv["vol_ratio"] is not None:
        vol_bits.append(f"当日量能 / 20日均量 = {pv['vol_ratio']:.2f}")
    if pv["down_vs_up"] is not None:
        vol_bits.append(f"下跌日均量 / 上涨日均量 = {pv['down_vs_up']:.2f}")
    section = (
        "<section id='style'>"
        "<h2>创业板 ÷ 中证红利低波</h2>"
        "<p class='note'>比价 = 创业板指（399006）÷ 中证红利低波（930955）。"
        "历史分位是所选窗口里，价格不高于当日的交易日占比（100 为这段里的最高）。</p>"
        f"<p class='note'>行情截止 {html.escape(pair['asof'])}"
        + (f" · {html.escape(note)}" if note else "")
        + "</p>"
        "<label class='win'>分位窗口 <select id='styleWin'>"
        f"{options}</select></label>"
        + "".join(panels)
        + f"<h3>创业板指量价：{html.escape(pv['label'])}</h3>"
        f"<p class='note'>{html.escape(pv['why'])}</p>"
        "<p class='note'>均线向下且下跌日量更大，是下跌趋势；还在均线下但量能萎缩，只是缩量止跌；"
        "放量上涨并站上20日均线、均线不再向下，才记为反转尝试。"
        + (html.escape(" ".join(vol_bits)) if vol_bits else "")
        + "</p>"
        + _chart_svg(pv_lines)
        + "</section>"
        "<script>"
        "document.getElementById('styleWin').addEventListener('change',function(e){"
        "document.querySelectorAll('.style-panel').forEach(function(p){p.hidden=p.dataset.win!==e.target.value;});"
        "});"
        "</script>"
    )
    return section, None


def _table_html(title: str, df) -> str:
    import pandas as pd

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return f"<section><h2>{html.escape(title)}</h2><p class='empty'>暂无数据</p></section>"
    rows = []
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    for rec in df.itertuples(index=False):
        cells = "".join(f"<td>{html.escape('' if v is None else str(v))}</td>" for v in rec)
        rows.append(f"<tr>{cells}</tr>")
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f"<div class='wrap'><table><thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def write_pages_html(
    asof: str,
    generated_at: str,
    boards: dict,
    errors: dict[str, str],
    source: str,
    bar_note: str = "",
    style_html: str = "",
    support_html: str = "",
    resist_html: str = "",
    levels_script: str = "",
) -> None:
    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    err_html = ""
    if errors:
        items = "".join(f"<li><b>{html.escape(k)}</b>：{html.escape(v)}</li>" for k, v in errors.items())
        err_html = f"<aside class='err'><p>部分榜单失败</p><ul>{items}</ul></aside>"
    body = (
        _table_html("低位 TOP30", boards.get("low_position"))
        + _table_html("反转 TOP30", boards.get("reversal"))
        + _table_html("动量 TOP30", boards.get("momentum"))
    )
    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>ETF 榜单 · {html.escape(asof)}</title>
    <style>
    :root {{ color-scheme: light dark; }}
    body {{ font-family: "Segoe UI", "PingFang SC", sans-serif; margin: 24px auto; max-width: 1100px; line-height: 1.45; }}
    h1 {{ font-size: 1.4rem; margin: 0 0 8px; }}
    .meta {{ color: #666; margin-bottom: 20px; }}
    section {{ margin: 28px 0; }}
    .wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; white-space: nowrap; }}
    th {{ background: #f4f4f4; }}
    .empty {{ color: #888; }}
    .err {{ background: #fff3f0; border: 1px solid #ffccc7; padding: 10px 14px; }}
    .note {{ color: #555; font-size: 0.92rem; }}
    .metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 12px 0; }}
    .metric {{ background: #f4f7fb; border-radius: 10px; padding: 10px 12px; }}
    .metric span {{ display: block; color: #64748b; font-size: 0.82rem; }}
    .metric b {{ font-size: 1.25rem; }}
    .win {{ display: block; margin: 8px 0 4px; }}
    .chartbox {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 8px 8px 4px; margin: 8px 0 16px; }}
    .chart {{ width: 100%; height: auto; display: block; }}
    .plot {{ fill: #fafafa; stroke: #e5e7eb; }}
    .guide {{ stroke: #cbd5e1; stroke-dasharray: 4 4; }}
    .axis {{ fill: #64748b; font-size: 12px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 10px; color: #334155; font-size: 0.82rem; margin: 4px 4px 8px; }}
    .legend i {{ display: inline-block; width: 12px; height: 3px; margin-right: 4px; vertical-align: middle; }}
    h3 {{ font-size: 1rem; margin: 14px 0 4px; }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 4px; }}
    .tab {{ border: 1px solid #cbd5e1; background: #fff; border-radius: 999px; padding: 8px 14px; font-size: 0.95rem; }}
    .tab.on {{ background: #0f172a; color: #fff; border-color: #0f172a; }}
    .search {{ width: 100%; box-sizing: border-box; font-size: 1rem; padding: 12px 14px; border-radius: 12px; border: 1px solid #cbd5e1; margin: 8px 0; }}
    .hit {{ display: block; width: 100%; text-align: left; margin: 6px 0; padding: 10px 12px; border-radius: 10px; border: 1px solid #e2e8f0; background: #fff; font-size: 0.95rem; }}
    @media (min-width: 720px) {{ .metrics {{ grid-template-columns: repeat(4, 1fr); }} }}
    @media (max-width: 640px) {{ body {{ margin: 12px auto; }} }}
  </style>
</head>
<body>
  <h1>ETF 榜单</h1>
  <p class="meta">更新时刻（上海）：{html.escape(generated_at)} · 行情截止：{html.escape(asof)}{(" · " + html.escape(bar_note)) if bar_note else ""} · 数据源：{html.escape(source)}（东财失败自动切新浪）</p>
  {err_html}
  <nav class="tabs">
    <button type="button" class="tab on" data-tab="support">支撑位</button>
    <button type="button" class="tab" data-tab="resist">压力位</button>
    <button type="button" class="tab" data-tab="style">比价</button>
    <button type="button" class="tab" data-tab="boards">榜单</button>
  </nav>
  <div id="panel-support" class="panel">{support_html}</div>
  <div id="panel-resist" class="panel" hidden>{resist_html}</div>
  <div id="panel-style" class="panel" hidden>{style_html}</div>
  <div id="panel-boards" class="panel" hidden>{body}</div>
  {levels_script}
</body>
</html>
"""
    (docs / "index.html").write_text(page, encoding="utf-8")
    (BOARD_DIR / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    asof = data_asof_str()
    bar_note = live_bar_note()
    generated_at = now_cst_iso()
    source = "sina" if hist_prefer_sina() else "eastmoney"
    print(f"refresh at {generated_at} CST · asof {asof} · {bar_note or '收盘K'} · source {source}", flush=True)
    print(f"active hist source: {active_hist_source()}", flush=True)

    errors: dict[str, str] = {}
    low, err = _run_one("低位 TOP30", get_low_position_top30)
    if err:
        errors["low_position"] = err
    rev, err = _run_one("反转 TOP30", get_reversal_top30)
    if err:
        errors["reversal"] = err
    mom, err = _run_one("动量 TOP30", lambda: get_momentum_top30("510300", 20))
    if err:
        errors["momentum"] = err

    boards_map = {"low_position": low, "reversal": rev, "momentum": mom}
    print("==> 支撑位 / 压力位", flush=True)
    try:
        from scripts.level_page import level_catalog, level_sections

        level_items = level_catalog(boards_map, extra=40)
        support_html, resist_html, levels_script = level_sections(level_items)
        print(f"    ok  {len(level_items)} 只", flush=True)
    except Exception as exc:
        from scripts.level_page import level_sections

        support_html, resist_html, levels_script = level_sections([])
        errors["levels"] = str(exc)
        print(f"    fail  {exc}", flush=True)
        traceback.print_exc()

    print("==> 创业板 ÷ 中证红利低波", flush=True)
    style_html, style_err = _style_section_html()
    if style_err:
        errors["style"] = style_err
        print(f"    fail  {style_err}", flush=True)
    else:
        print("    ok", flush=True)

    save_boards_snapshot(
        asof=asof,
        generated_at=generated_at,
        low_position=low,
        reversal=rev,
        momentum=mom,
        momentum_key=("510300", 20),
        source=source,
        errors=errors,
    )
    write_pages_html(
        asof,
        generated_at,
        {"low_position": low, "reversal": rev, "momentum": mom},
        errors,
        source,
        bar_note,
        style_html,
        support_html,
        resist_html,
        levels_script,
    )
    snap = load_boards_snapshot()
    rows = (snap or {}).get("meta", {}).get("rows", {})
    print(f"wrote {BOARD_DIR} rows={rows}", flush=True)
    if errors and not any(int(v or 0) > 0 for v in rows.values()):
        print("all boards failed", flush=True)
        return 1
    if errors:
        print("partial success", flush=True)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
