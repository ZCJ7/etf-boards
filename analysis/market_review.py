"""A 股大盘复盘（对齐 daily-stock-analysis 能力边界，本地 AkShare 实现）。

覆盖：主要指数、涨跌家数/涨跌停、行业领涨领跌。
不依赖 DSA 运行时；单源失败 fail-open。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field

import akshare as ak
import pandas as pd
import streamlit as st

from akshare_patch import install_akshare_patch

install_akshare_patch()

_INDEX_NAMES = ("上证指数", "深证成指", "创业板指", "科创50", "沪深300", "上证50")


def _call_timeout(fn, timeout: float = 20):
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout)
    except FutTimeout:
        return None
    except Exception:
        return None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _col(df: pd.DataFrame, *names: str) -> str | None:
    cols = list(df.columns)
    for n in names:
        if n in cols:
            return n
    for n in names:
        for c in cols:
            if n in str(c):
                return str(c)
    return None


@dataclass
class MarketOverview:
    as_of: str = ""
    indices: pd.DataFrame = field(default_factory=pd.DataFrame)
    up_count: int | None = None
    down_count: int | None = None
    flat_count: int | None = None
    limit_up: int | None = None
    limit_down: int | None = None
    activity_pct: str = ""
    top_sectors: list[dict] = field(default_factory=list)
    bottom_sectors: list[dict] = field(default_factory=list)
    status: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def up_ratio(self) -> float | None:
        if self.up_count is None or self.down_count is None:
            return None
        den = self.up_count + self.down_count
        if den <= 0:
            return None
        return self.up_count / den

    def mood_label(self) -> str:
        r = self.up_ratio
        if r is None:
            return "数据不足"
        if r >= 0.65:
            return "偏强/偏暖"
        if r >= 0.45:
            return "震荡分化"
        return "偏弱"


def _fetch_indices() -> tuple[pd.DataFrame, str]:
    # 优先新浪（本机更稳）；东财作备选
    for label, fn in (
        ("sina", ak.stock_zh_index_spot_sina),
        ("em", ak.stock_zh_index_spot_em),
    ):
        raw = _call_timeout(fn, timeout=25)
        if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
            continue
        name_c = _col(raw, "名称", "name")
        code_c = _col(raw, "代码", "code")
        price_c = _col(raw, "最新价", "trade")
        pct_c = _col(raw, "涨跌幅", "changepercent", "涨跌幅")
        amt_c = _col(raw, "成交额", "amount")
        if not name_c or not pct_c:
            continue
        work = raw.copy()
        work = work[work[name_c].astype(str).isin(_INDEX_NAMES)].copy()
        if work.empty:
            # 新浪可能用略不同名称
            work = raw[raw[name_c].astype(str).str.contains("上证指数|深证成指|创业板指|科创50|沪深300|上证50", na=False)].copy()
        if work.empty:
            continue
        out = pd.DataFrame(
            {
                "代码": work[code_c].astype(str) if code_c else "",
                "名称": work[name_c].astype(str),
                "最新价": pd.to_numeric(work[price_c], errors="coerce") if price_c else None,
                "涨跌幅%": pd.to_numeric(work[pct_c], errors="coerce"),
                "成交额": pd.to_numeric(work[amt_c], errors="coerce") if amt_c else None,
            }
        )
        # 去重保序：按目标名单排序
        order = {n: i for i, n in enumerate(_INDEX_NAMES)}
        out["_ord"] = out["名称"].map(lambda x: order.get(x, 99))
        out = out.sort_values("_ord").drop(columns=["_ord"]).drop_duplicates("名称").reset_index(drop=True)
        return out, f"OK({label},{len(out)})"
    return pd.DataFrame(columns=["代码", "名称", "最新价", "涨跌幅%", "成交额"]), "失败"


def _fetch_breadth() -> tuple[dict, str]:
    raw = _call_timeout(ak.stock_market_activity_legu, timeout=15)
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return {}, "失败"
    item_c = _col(raw, "item", "指标", "名称") or raw.columns[0]
    val_c = _col(raw, "value", "数值", "值") or raw.columns[1]
    mapping = {str(k).strip(): v for k, v in zip(raw[item_c].astype(str), raw[val_c])}
    as_of = str(mapping.get("统计日期", "") or "")

    def _int(key: str) -> int | None:
        v = mapping.get(key)
        if v is None:
            return None
        try:
            return int(float(str(v).replace("%", "").replace(",", "")))
        except Exception:
            return None

    return (
        {
            "as_of": as_of,
            "up_count": _int("上涨"),
            "down_count": _int("下跌"),
            "flat_count": _int("平盘"),
            "limit_up": _int("涨停"),
            "limit_down": _int("跌停"),
            "activity_pct": str(mapping.get("活跃度", "") or ""),
        },
        f"OK({as_of or 'legu'})",
    )


def _sector_rows(df: pd.DataFrame, top_n: int = 5) -> tuple[list[dict], list[dict], str]:
    name_c = _col(df, "名称", "板块", "行业", "label")
    pct_c = _col(df, "涨跌幅", "涨跌幅%", "change")
    if not name_c or not pct_c:
        return [], [], "列缺失"
    work = df.copy()
    work["_pct"] = pd.to_numeric(work[pct_c], errors="coerce")
    work = work.dropna(subset=["_pct"])
    if work.empty:
        return [], [], "空"
    up_c = _col(work, "上涨家数", "上涨")
    down_c = _col(work, "下跌家数", "下跌")
    lead_c = _col(work, "领涨股", "领涨股票")

    def _pack(row) -> dict:
        d = {"name": str(row[name_c]), "change_pct": round(float(row["_pct"]), 2)}
        if up_c:
            d["up"] = row.get(up_c)
        if down_c:
            d["down"] = row.get(down_c)
        if lead_c:
            d["leader"] = str(row.get(lead_c, "") or "")
        return d

    top = [_pack(r) for _, r in work.sort_values("_pct", ascending=False).head(top_n).iterrows()]
    bottom = [_pack(r) for _, r in work.sort_values("_pct", ascending=True).head(top_n).iterrows()]
    return top, bottom, f"OK({len(work)})"


def _fetch_sectors(top_n: int = 5) -> tuple[list[dict], list[dict], str]:
    for label, fn in (
        ("ths", ak.stock_board_industry_summary_ths),
        ("em", ak.stock_board_industry_name_em),
        ("spot", lambda: ak.stock_sector_spot(indicator="行业")),
    ):
        raw = _call_timeout(fn, timeout=25)
        if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
            continue
        top, bottom, st_msg = _sector_rows(raw, top_n=top_n)
        if top or bottom:
            return top, bottom, f"{st_msg}/{label}"
    return [], [], "失败"


def overview_to_markdown(ov: MarketOverview) -> str:
    lines = ["# 🎯 A股大盘复盘", ""]
    if ov.as_of:
        lines.append(f"> 统计时点：{ov.as_of} · 情绪：**{ov.mood_label()}**")
        lines.append("")
    lines.append("## 主要指数")
    if ov.indices is not None and not ov.indices.empty:
        lines.append("| 名称 | 最新价 | 涨跌幅% |")
        lines.append("|------|--------|--------|")
        for _, r in ov.indices.iterrows():
            pct = r.get("涨跌幅%")
            pct_s = f"{float(pct):+.2f}" if pd.notna(pct) else "—"
            price = r.get("最新价")
            price_s = f"{float(price):.2f}" if pd.notna(price) else "—"
            lines.append(f"| {r.get('名称','')} | {price_s} | {pct_s} |")
    else:
        lines.append("- 暂无指数数据")
    lines.append("")
    lines.append("## 市场广度")
    if ov.up_count is not None:
        ratio = ov.up_ratio
        ratio_s = f"{ratio:.0%}" if ratio is not None else "—"
        lines.append(
            f"- 上涨/下跌/平盘：{ov.up_count} / {ov.down_count} / {ov.flat_count}（上涨占比 {ratio_s}）"
        )
        lines.append(f"- 涨停/跌停：{ov.limit_up} / {ov.limit_down}")
        if ov.activity_pct:
            lines.append(f"- 活跃度：{ov.activity_pct}")
    else:
        lines.append("- 暂无涨跌家数")
    lines.append("")
    lines.append("## 板块主线")
    if ov.top_sectors:
        lines.extend(["#### 领涨板块 Top5", "| 排名 | 板块 | 涨跌幅 |", "|------|------|--------|"])
        for i, s in enumerate(ov.top_sectors, 1):
            lines.append(f"| {i} | {s.get('name','')} | {s.get('change_pct',0):+.2f}% |")
    if ov.bottom_sectors:
        lines.extend(["", "#### 领跌板块 Top5", "| 排名 | 板块 | 涨跌幅 |", "|------|------|--------|"])
        for i, s in enumerate(ov.bottom_sectors, 1):
            lines.append(f"| {i} | {s.get('name','')} | {s.get('change_pct',0):+.2f}% |")
    if not ov.top_sectors and not ov.bottom_sectors:
        lines.append("- 暂无板块榜")
    return "\n".join(lines)


def overview_to_prompt_block(ov: MarketOverview) -> str:
    return overview_to_markdown(ov)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_cn_market_review(top_n: int = 5) -> dict:
    """拉取 A 股大盘复盘；返回可序列化 dict（便于 session_state）。"""
    status: dict[str, str] = {}
    indices, st_idx = _fetch_indices()
    status["指数"] = st_idx
    breadth, st_br = _fetch_breadth()
    status["广度"] = st_br
    top, bottom, st_sec = _fetch_sectors(top_n=top_n)
    status["板块"] = st_sec

    ov = MarketOverview(
        as_of=str(breadth.get("as_of") or ""),
        indices=indices,
        up_count=breadth.get("up_count"),
        down_count=breadth.get("down_count"),
        flat_count=breadth.get("flat_count"),
        limit_up=breadth.get("limit_up"),
        limit_down=breadth.get("limit_down"),
        activity_pct=str(breadth.get("activity_pct") or ""),
        top_sectors=top,
        bottom_sectors=bottom,
        status=status,
    )
    if indices.empty and ov.up_count is None and not top:
        ov.error = "大盘数据全部失败，请稍后重试"
    return {
        "as_of": ov.as_of,
        "indices": ov.indices,
        "up_count": ov.up_count,
        "down_count": ov.down_count,
        "flat_count": ov.flat_count,
        "limit_up": ov.limit_up,
        "limit_down": ov.limit_down,
        "activity_pct": ov.activity_pct,
        "top_sectors": ov.top_sectors,
        "bottom_sectors": ov.bottom_sectors,
        "status": ov.status,
        "error": ov.error,
        "mood": ov.mood_label(),
        "markdown": overview_to_markdown(ov),
    }


def dict_to_overview(d: dict) -> MarketOverview:
    return MarketOverview(
        as_of=str(d.get("as_of") or ""),
        indices=d.get("indices") if isinstance(d.get("indices"), pd.DataFrame) else pd.DataFrame(),
        up_count=d.get("up_count"),
        down_count=d.get("down_count"),
        flat_count=d.get("flat_count"),
        limit_up=d.get("limit_up"),
        limit_down=d.get("limit_down"),
        activity_pct=str(d.get("activity_pct") or ""),
        top_sectors=list(d.get("top_sectors") or []),
        bottom_sectors=list(d.get("bottom_sectors") or []),
        status=dict(d.get("status") or {}),
        error=d.get("error"),
    )
