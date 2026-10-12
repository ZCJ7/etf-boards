"""把空仓/题材判断写进手机页。"""

from __future__ import annotations

import html
from pathlib import Path

from analysis.cash_regime import run_cash_regime

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.html"


def _rows(items: list[dict]) -> str:
    if not items:
        return "<tr><td colspan='7'>无</td></tr>"
    body = []
    for r in items:
        body.append(
            "<tr>"
            f"<td>{html.escape(r['名称'])}</td><td>{r['代码']}</td><td>{html.escape(r['分组'])}</td>"
            f"<td>{r['站上MA20']}</td><td>{r['近20日%'] if r['近20日%'] is not None else '—'}</td>"
            f"<td>{r['价格分位%'] if r['价格分位%'] is not None else '—'}</td>"
            f"<td>{html.escape(r.get('原因') or '')}</td>"
            "</tr>"
        )
    return "".join(body)


def _flow_rows(flow: dict) -> str:
    if not flow.get("ok"):
        return "<tr><td colspan='4'>行业资金流没取到</td></tr>"
    left = flow.get("outflow") or []
    right = flow.get("inflow") or []
    rows = []
    for i in range(max(len(left), len(right))):
        a = left[i] if i < len(left) else {"行业": "", "净额": ""}
        b = right[i] if i < len(right) else {"行业": "", "净额": ""}
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(a['行业']))}</td><td>{a['净额']}</td>"
            f"<td>{html.escape(str(b['行业']))}</td><td>{b['净额']}</td>"
            "</tr>"
        )
    return "".join(rows) or "<tr><td colspan='4'>无</td></tr>"


def _reduce_rows(items: list[dict]) -> str:
    if not items:
        return "<tr><td colspan='4'>无</td></tr>"
    return "".join(
        "<tr>"
        f"<td>{html.escape(r['名称'])}</td><td>{html.escape(r['股东'])}</td>"
        f"<td>{r['金额亿']}</td><td>{html.escape(r['公告日'])}</td>"
        "</tr>"
        for r in items
    )


def render_cash_section(result: dict | None = None) -> str:
    result = result or run_cash_regime()
    layers = "".join(
        f"<tr><td>{html.escape(x['层'])}</td><td>{html.escape(x['状态'])}</td><td>{html.escape(x['说明'])}</td></tr>"
        for x in result["layers"]
    )
    pe_rows = "".join(
        f"<tr><td>{html.escape(r['name'])}</td><td>{r.get('pe')}</td><td>{r.get('percentile')}</td><td>{html.escape(r.get('asof') or '')}</td></tr>"
        for r in result["pe"]
        if r.get("ok")
    )
    return f"""
<section id='cash'>
  <h2>仓位与题材</h2>
  <p class='note'>空仓不是清光全市场。估值、资金、技术、外部至少两层同时报警，才降低仓位或从成长迁走。取不到的层标为缺失，不参与打分。</p>
  <div class='metrics'><div class='metric'><span>当前判断</span><b>{html.escape(result['regime'])}</b></div></div>
  <p>{html.escape(result['summary'])}</p>
  <div class='wrap'><table><thead><tr><th>层</th><th>状态</th><th>说明</th></tr></thead><tbody>{layers}</tbody></table></div>
  <h3>当日行业资金（亿元）</h3>
  <p class='note'>同花顺口径。左边是净流出最多的行业，右边是净流入最多的行业。单日流向只看钱往哪边走。</p>
  <div class='wrap'><table><thead><tr><th>净流出</th><th>净额</th><th>净流入</th><th>净额</th></tr></thead><tbody>{_flow_rows(result.get('flow') or {})}</tbody></table></div>
  <h3>近30日减持金额靠前</h3>
  <div class='wrap'><table><thead><tr><th>名称</th><th>股东</th><th>估算金额（亿元）</th><th>公告日</th></tr></thead><tbody>{_reduce_rows(result.get('reduction_top') or [])}</tbody></table></div>
  <h3>指数滚动市盈率分位</h3>
  <div class='wrap'><table><thead><tr><th>指数</th><th>市盈率</th><th>历史分位%</th><th>截止</th></tr></thead><tbody>{pe_rows}</tbody></table></div>
  <h3>现在可以看</h3>
  <div class='wrap'><table><thead><tr><th>名称</th><th>代码</th><th>分组</th><th>站上MA20</th><th>近20日%</th><th>价格分位%</th><th>原因</th></tr></thead><tbody>{_rows(result['prefer'])}</tbody></table></div>
  <h3>均线上但价格分位很高</h3>
  <p class='note'>还没跌破均线，但近3年价格已经很靠上，适合拿着看，不适合追。</p>
  <div class='wrap'><table><thead><tr><th>名称</th><th>代码</th><th>分组</th><th>站上MA20</th><th>近20日%</th><th>价格分位%</th><th>原因</th></tr></thead><tbody>{_rows(result.get('extended') or [])}</tbody></table></div>
  <h3>先避开</h3>
  <div class='wrap'><table><thead><tr><th>名称</th><th>代码</th><th>分组</th><th>站上MA20</th><th>近20日%</th><th>价格分位%</th><th>原因</th></tr></thead><tbody>{_rows(result['avoid'])}</tbody></table></div>
  <p class='note'>{html.escape(result['missing'])}</p>
</section>
"""


def splice_index(section: str, index_path: Path = INDEX) -> None:
    text = index_path.read_text(encoding="utf-8")
    button = '<button type="button" class="tab" data-tab="cash">仓位题材</button>'
    if 'data-tab="cash"' not in text:
        text = text.replace(
            '<button type="button" class="tab" data-tab="momentum">动量Top5</button>',
            '<button type="button" class="tab" data-tab="momentum">动量Top5</button>\n    ' + button,
        )
    panel = f'<div id="panel-cash" class="panel" hidden>{section}</div>'
    start = text.find('<div id="panel-cash"')
    if start >= 0:
        end = text.find('<div id="panel-style"', start)
        text = text[:start] + panel + "\n  " + text[end:]
    else:
        text = text.replace(
            '<div id="panel-style"',
            panel + '\n  <div id="panel-style"',
        )
    index_path.write_text(text, encoding="utf-8")
    board = ROOT / "data" / "boards" / "index.html"
    if board.exists():
        board.write_text(text, encoding="utf-8")
