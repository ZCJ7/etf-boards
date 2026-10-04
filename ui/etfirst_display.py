"""ETFirst 详情格式化展示（不暴露原始 JSON）。"""

from __future__ import annotations

from typing import Any

import streamlit as st


def _dig(data: dict, *keys: str, default: str = "—") -> str:
    cur: Any = data
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if cur is None or cur == "":
        return default
    if isinstance(cur, float):
        return f"{cur:.4f}" if abs(cur) < 100 else f"{cur:.2f}"
    return str(cur)


def render_etfirst_summary(data: dict[str, Any], symbol: str, name: str) -> None:
    root = data.get("data", data) if isinstance(data, dict) else {}
    if isinstance(root, dict) and "data" in root and isinstance(root["data"], dict):
        root = root["data"]

    st.markdown(f"#### {symbol} {name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新净值", _dig(root, "nav", default=_dig(root, "unit_nav")))
    c2.metric("溢价率", _dig(root, "premium_rate", default=_dig(root, "iopv_premium")))
    c3.metric("规模(亿)", _dig(root, "fund_size", default=_dig(root, "scale")))
    c4.metric("跟踪指数", _dig(root, "index_name", default=_dig(root, "benchmark")))

    st.markdown("**要点摘要**")
    bullets = []
    for key, label in [
        ("risk_level", "风险等级"),
        ("fund_manager", "基金经理"),
        ("establish_date", "成立日期"),
        ("expense_ratio", "管理费率"),
        ("tracking_error", "跟踪误差"),
    ]:
        val = _dig(root, key, default="")
        if val and val != "—":
            bullets.append(f"- **{label}**：{val}")

    holdings = root.get("top_holdings") or root.get("holdings")
    if isinstance(holdings, list) and holdings:
        top3 = holdings[:3]
        names = []
        for h in top3:
            if isinstance(h, dict):
                names.append(h.get("name") or h.get("stock_name") or str(h))
            else:
                names.append(str(h))
        bullets.append(f"- **前三大持仓**：{' / '.join(names)}")

    if not bullets:
        bullets.append("- 官方数据已获取，可在首趋E指小程序查看更完整持仓与风险指标。")
    st.markdown("\n".join(bullets))
