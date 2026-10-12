"""空仓时机与板块迁移。

规则来自公开讨论的四层框架，这里只落地能取到的两层，避免编造：
- 估值层：乐咕指数滚动市盈率的历史分位（月频）
- 技术层：对应 ETF 是否跌破 MA20

至少两层同时报警才给出空仓/减仓结论。单一指标只提示。
资金流向、两融、产业减持、美债目前接口不稳定或未接入，页面上标为缺失，不参与打分。

板块选择：空仓信号不是清光全市场，而是从贵且走弱的方向，迁到便宜或还站上均线的方向。
"""

from __future__ import annotations

import concurrent.futures as cf

import akshare as ak
import numpy as np
import pandas as pd

from analysis.indicators import ensure_close
from etf_data_fetcher import fetch_pool_daily

# 估值锚（乐咕有月频市盈率的指数）
PE_ANCHORS = {
    "创业板50": "成长",
    "沪深300": "宽基",
    "上证红利": "红利",
    "中证500": "中盘",
    "中证1000": "小盘",
}

# 技术确认用的场内 ETF
TECH_ETFS = {
    "159949": "创业板50ETF",
    "159915": "创业板ETF",
    "588000": "科创50ETF",
    "510300": "沪深300ETF",
    "510880": "红利ETF",
}

# 题材 ETF：非空仓时从这里选。成长标签用于结构性减仓时避开。
THEME_ETFS = {
    "588000": ("科创50ETF", "成长"),
    "159915": ("创业板ETF", "成长"),
    "512480": ("半导体ETF", "成长"),
    "512760": ("芯片ETF", "成长"),
    "159819": ("人工智能ETF", "成长"),
    "515050": ("5GETF", "成长"),
    "512010": ("医药ETF", "医药"),
    "512400": ("有色金属ETF", "周期"),
    "515220": ("煤炭ETF", "红利周期"),
    "512800": ("银行ETF", "红利"),
    "510880": ("红利ETF", "红利"),
    "512890": ("红利低波ETF", "红利"),
    "518880": ("黄金ETF", "避险"),
    "159981": ("能源化工ETF", "周期"),
    "512690": ("酒ETF", "消费"),
    "513100": ("纳指ETF", "跨境"),
    "510300": ("沪深300ETF", "宽基"),
}

GROWTH_TAGS = {"成长"}
HOT_PCT = 85.0
CHEAP_PCT = 50.0
BROAD_HOT_PCT = 90.0


def _percentile(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 36:
        return None
    last = float(s.iloc[-1])
    return round(float((s <= last).mean() * 100), 1)


def _call(fn, timeout: float = 12):
    ex = cf.ThreadPoolExecutor(1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def load_pe_anchor(symbol: str) -> dict:
    try:
        raw = _call(lambda: ak.stock_index_pe_lg(symbol=symbol), timeout=15)
    except Exception as exc:
        return {"name": symbol, "ok": False, "error": str(exc)[:160]}
    if raw is None or raw.empty or "滚动市盈率" not in raw.columns:
        return {"name": symbol, "ok": False, "error": "无滚动市盈率"}
    pe = pd.to_numeric(raw["滚动市盈率"], errors="coerce").dropna()
    date = str(raw["日期"].iloc[-1])[:10] if "日期" in raw.columns else ""
    pct = _percentile(pe)
    return {
        "name": symbol,
        "ok": pct is not None,
        "pe": round(float(pe.iloc[-1]), 2) if len(pe) else None,
        "percentile": pct,
        "asof": date,
        "error": "" if pct is not None else "样本不足",
    }


def _ma_state(close: pd.Series) -> dict:
    s = pd.to_numeric(close, errors="coerce").dropna()
    if len(s) < 25:
        return {"ok": False}
    ma20 = s.rolling(20).mean()
    last = float(s.iloc[-1])
    ma = float(ma20.iloc[-1])
    ma_prev = float(ma20.iloc[-6]) if len(ma20.dropna()) >= 6 else ma
    ret20 = float(s.iloc[-1] / s.iloc[-21] - 1) if len(s) >= 21 else None
    return {
        "ok": True,
        "close": round(last, 3),
        "ma20": round(ma, 3),
        "above_ma20": last > ma,
        "ma_falling": ma < ma_prev,
        "ret20": None if ret20 is None else round(ret20 * 100, 2),
        "price_pct": _percentile(s.tail(756) if len(s) > 756 else s),
        "asof": s.index[-1].strftime("%Y-%m-%d"),
    }


def load_etf_states(codes: list[str]) -> dict[str, dict]:
    daily = fetch_pool_daily(codes, tail_days=900, max_workers=8, pause_sec=0, prefer_sina=None)
    out = {}
    for code in codes:
        frame = daily.get(code)
        if frame is None or frame.empty:
            out[code] = {"ok": False}
            continue
        out[code] = _ma_state(ensure_close(frame))
    return out


def judge(pe_rows: list[dict], tech: dict[str, dict]) -> dict:
    growth_pe = next((r for r in pe_rows if r["name"] == "创业板50" and r.get("ok")), None)
    broad_pe = next((r for r in pe_rows if r["name"] == "沪深300" and r.get("ok")), None)
    div_pe = next((r for r in pe_rows if r["name"] == "上证红利" and r.get("ok")), None)

    growth_hot = bool(growth_pe and growth_pe["percentile"] >= HOT_PCT)
    broad_hot = bool(broad_pe and broad_pe["percentile"] >= BROAD_HOT_PCT)

    cyb = tech.get("159915") or {}
    star = tech.get("588000") or {}
    hs300 = tech.get("510300") or {}
    growth_break = bool(cyb.get("ok") and (not cyb["above_ma20"]) and cyb.get("ma_falling"))
    star_break = bool(star.get("ok") and (not star["above_ma20"]) and star.get("ma_falling"))
    broad_break = bool(hs300.get("ok") and (not hs300["above_ma20"]) and hs300.get("ma_falling"))
    growth_tech_alarm = growth_break or star_break

    val_alarm = growth_hot or broad_hot
    tech_alarm = growth_tech_alarm or broad_break

    if broad_hot and broad_break:
        regime = "降低仓位"
        summary = "宽基估值和趋势同时转弱，先少做、多留现金，不要在这个位置加成长。"
    elif growth_hot and growth_tech_alarm:
        regime = "结构性减仓"
        summary = "贵的是成长方向，而且已经跌破均线。不是把账户清成现金，而是把仓位从成长迁走。"
    elif val_alarm or tech_alarm:
        regime = "观察"
        summary = "只有一层在报警，按讨论里的规则还不够构成空仓。先看，不因为单一指标砍仓。"
    else:
        regime = "可以配置"
        summary = "估值层和技术层都没有同时报警。可以拿着仓位，但只在还站上均线、分位不算极端的题材里选。"

    layers = [
        {
            "层": "估值",
            "状态": "报警" if val_alarm else ("正常" if growth_pe and broad_pe else "缺失"),
            "说明": _pe_text(growth_pe, broad_pe, div_pe),
        },
        {
            "层": "技术",
            "状态": "报警" if tech_alarm else ("正常" if cyb.get("ok") else "缺失"),
            "说明": _tech_text(cyb, star, hs300),
        },
        {
            "层": "资金",
            "状态": "缺失",
            "说明": "板块资金流接口这次没有稳定返回，不参与判断。",
        },
    ]
    return {
        "regime": regime,
        "summary": summary,
        "layers": layers,
        "growth_hot": growth_hot,
        "growth_tech_alarm": growth_tech_alarm,
        "broad_hot": broad_hot,
        "broad_break": broad_break,
    }


def _pe_text(growth, broad, div) -> str:
    parts = []
    for row, label in ((growth, "创业板50"), (broad, "沪深300"), (div, "上证红利")):
        if not row or not row.get("ok"):
            parts.append(f"{label}市盈率缺失")
            continue
        parts.append(f"{label}滚动市盈率 {row['pe']} 倍，历史分位 {row['percentile']}%（{row['asof']}）")
    return "；".join(parts)


def _tech_text(cyb, star, hs300) -> str:
    def one(name, row):
        if not row or not row.get("ok"):
            return f"{name}缺失"
        side = "站上" if row["above_ma20"] else "跌破"
        slope = "向下" if row["ma_falling"] else "没有向下"
        return f"{name} {row['close']}，{side} MA20（{row['ma20']}），均线{slope}，截止 {row['asof']}"

    return "；".join([one("创业板ETF", cyb), one("科创50ETF", star), one("沪深300ETF", hs300)])


def pick_themes(tech: dict[str, dict], regime: str) -> tuple[list[dict], list[dict], list[dict]]:
    """返回 (避开, 可看, 均线上但价格分位很高)。

    估值贵不贵看市盈率分位；这里的价格分位只说明这段涨多了。
    跌破 MA20 的放进避开。还站上均线、价格分位没到 90 的放进可看。
    """
    avoid, prefer, extended = [], [], []
    structural = regime in ("结构性减仓", "降低仓位")
    for code, (name, tag) in THEME_ETFS.items():
        row = tech.get(code) or {}
        if not row.get("ok"):
            continue
        pct = row.get("price_pct")
        item = {
            "代码": code,
            "名称": name,
            "分组": tag,
            "收盘": row["close"],
            "站上MA20": "是" if row["above_ma20"] else "否",
            "近20日%": row["ret20"],
            "价格分位%": pct,
            "截止": row["asof"],
        }
        if not row["above_ma20"]:
            item["原因"] = "已跌破 MA20"
            avoid.append(item)
            continue
        if structural and tag in GROWTH_TAGS:
            item["原因"] = "成长方向在减仓名单里"
            avoid.append(item)
            continue
        if pct is not None and pct >= 90:
            item["原因"] = "还站上均线，但近3年价格分位已经很高"
            extended.append(item)
            continue
        item["原因"] = "站上均线，价格分位不算极端"
        prefer.append(item)
    prefer.sort(key=lambda x: -(x["近20日%"] or -999))
    extended.sort(key=lambda x: -(x["近20日%"] or -999))
    avoid.sort(key=lambda x: (x["近20日%"] is None, x["近20日%"] or 0))
    return avoid[:8], prefer[:5], extended[:5]


def run_cash_regime() -> dict:
    pe_rows = []
    for name, _group in PE_ANCHORS.items():
        pe_rows.append(load_pe_anchor(name))
    codes = list(dict.fromkeys(list(TECH_ETFS) + list(THEME_ETFS)))
    tech = load_etf_states(codes)
    decision = judge(pe_rows, tech)
    avoid, prefer, extended = pick_themes(tech, decision["regime"])
    if decision["regime"] == "降低仓位":
        prefer = [p for p in prefer if p["分组"] in ("红利", "避险", "红利周期")][:5]
    return {
        "regime": decision["regime"],
        "summary": decision["summary"],
        "layers": decision["layers"],
        "pe": pe_rows,
        "avoid": avoid,
        "prefer": prefer,
        "extended": extended,
        "missing": "资金流向、两融、产业资本减持、美债收益率这次没有接入，所以结论只来自估值和技术两层。",
    }
