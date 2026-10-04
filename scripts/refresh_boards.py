"""拉取更新时刻可用行情，重算低位 / 反转 / 动量榜并写入 data/boards。

本地：
    python scripts/refresh_boards.py

GitHub Actions：手动 Run workflow 调用本脚本，发布 GitHub Pages。
15:05 前按上一完整交易日收盘；15:05 后用当天已收盘价。
东财不通则整次刷新改走新浪。
"""

from __future__ import annotations

import html
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akshare_patch import install_akshare_patch

install_akshare_patch()

from analysis.board_store import BOARD_DIR, load_boards_snapshot, now_cst_iso, save_boards_snapshot
from analysis.low_position_screener import get_low_position_top30
from analysis.reversal_screener import get_reversal_top30
from analysis.screener import get_momentum_top30
from etf_data_fetcher import active_hist_source, data_asof_str, hist_prefer_sina


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
  </style>
</head>
<body>
  <h1>ETF 榜单</h1>
  <p class="meta">更新时刻（上海）：{html.escape(generated_at)} · 行情截止：{html.escape(asof)} · 数据源：{html.escape(source)}（东财失败自动切新浪）</p>
  {err_html}
  {body}
</body>
</html>
"""
    (docs / "index.html").write_text(page, encoding="utf-8")
    (BOARD_DIR / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    asof = data_asof_str()
    generated_at = now_cst_iso()
    source = "sina" if hist_prefer_sina() else "eastmoney"
    print(f"refresh at {generated_at} CST · asof {asof} · source {source}", flush=True)
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
