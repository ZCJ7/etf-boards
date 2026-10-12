"""空仓时机与板块迁移。

四层里至少两层同时报警，才降低仓位或从成长迁走。单一指标只提示。
- 估值：乐咕指数滚动市盈率的历史分位（月频）
- 资金：两融融资余额近 5 日变化；近 30 日已公告股东减持估算；同花顺行业资金流只作迁仓参考
- 技术：对应 ETF 是否跌破向下的 MA20
- 外部：美国 10 年期国债收益率

取不到或过期的层标为缺失，不参与打分。减持金额是已公告变动股数乘收盘价，不是减持计划上限。
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
US_YIELD_ALARM = 4.5
US_YIELD_SEVERE = 5.0
REDUCTION_ALARM_YI = 1000.0
MARGIN_DOWN_DAYS = 4
STALE_DAYS = 8
GROWTH_INDUSTRIES = (
    "半导体",
    "元件",
    "光学光电子",
    "消费电子",
    "通信设备",
    "计算机设备",
    "软件开发",
    "互联网服务",
    "军工电子",
)


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


def _stale(date_value) -> bool:
    ts = pd.Timestamp(date_value).normalize()
    return (pd.Timestamp.today().normalize() - ts).days > STALE_DAYS


def load_margin() -> dict:
    """东方财富两融账户统计，融资余额单位为亿元。"""
    try:
        raw = _call(ak.stock_margin_account_info, timeout=25)
    except Exception as exc:
        return {"ok": False, "alarm": False, "error": str(exc)[:160]}
    if raw is None or raw.empty or "融资余额" not in raw.columns:
        return {"ok": False, "alarm": False, "error": "无融资余额"}
    work = raw.copy()
    work["日期"] = pd.to_datetime(work["日期"], errors="coerce")
    work["融资余额"] = pd.to_numeric(work["融资余额"], errors="coerce")
    work = work.dropna(subset=["日期", "融资余额"]).sort_values("日期")
    if len(work) < 8:
        return {"ok": False, "alarm": False, "error": "融资余额样本不足"}
    last_dt = work["日期"].iloc[-1]
    if _stale(last_dt):
        return {"ok": False, "alarm": False, "error": f"融资余额停在 {last_dt.strftime('%Y-%m-%d')}"}
    bal = work["融资余额"]
    last = float(bal.iloc[-1])
    prev = float(bal.iloc[-6])
    daily = bal.diff().iloc[-5:]
    down_days = int((daily < 0).sum())
    chg = last - prev
    alarm = chg < 0 and down_days >= MARGIN_DOWN_DAYS
    return {
        "ok": True,
        "alarm": alarm,
        "balance": round(last, 2),
        "chg5": round(chg, 2),
        "pct5": round((last / prev - 1) * 100, 2),
        "down_days": down_days,
        "asof": last_dt.strftime("%Y-%m-%d"),
        "error": "",
    }


def load_us_yield() -> dict:
    try:
        raw = _call(lambda: ak.bond_zh_us_rate(start_date="20240101"), timeout=25)
    except Exception as exc:
        return {"ok": False, "alarm": False, "error": str(exc)[:160]}
    if raw is None or raw.empty or "美国国债收益率10年" not in raw.columns:
        return {"ok": False, "alarm": False, "error": "无美国10年期收益率"}
    work = raw.copy()
    work["日期"] = pd.to_datetime(work["日期"], errors="coerce")
    work["收益率"] = pd.to_numeric(work["美国国债收益率10年"], errors="coerce")
    work = work.dropna(subset=["日期", "收益率"]).sort_values("日期")
    if work.empty:
        return {"ok": False, "alarm": False, "error": "美国10年期收益率为空"}
    last_dt = work["日期"].iloc[-1]
    if _stale(last_dt):
        return {"ok": False, "alarm": False, "error": f"美债收益率停在 {last_dt.strftime('%Y-%m-%d')}"}
    value = float(work["收益率"].iloc[-1])
    return {
        "ok": True,
        "alarm": value >= US_YIELD_ALARM,
        "severe": value >= US_YIELD_SEVERE,
        "y10": round(value, 2),
        "asof": last_dt.strftime("%Y-%m-%d"),
        "error": "",
    }


def _reduction_page(page: int) -> list[dict]:
    import requests

    resp = requests.get(
        "https://datacenter-web.eastmoney.com/api/data/v1/get",
        params={
            "sortColumns": "END_DATE,SECURITY_CODE,EITIME",
            "sortTypes": "-1,-1,-1",
            "pageSize": "500",
            "pageNumber": str(page),
            "reportName": "RPT_SHARE_HOLDER_INCREASE",
            "quoteColumns": "f2~01~SECURITY_CODE~NEWEST_PRICE",
            "quoteType": "0",
            "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,HOLDER_NAME,DIRECTION,CHANGE_NUM,END_DATE,NOTICE_DATE,CLOSE_PRICE,NEWEST_PRICE",
            "source": "WEB",
            "client": "WEB",
            "filter": '(DIRECTION="减持")',
        },
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json().get("result") or {}
    return payload.get("data") or []


def load_reductions() -> dict:
    """近 30 日已公告减持。变动数量按万股，金额 = 万股 × 价格 × 10000。"""
    try:
        rows: list[dict] = []
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=30)
        covered = False
        for page in range(1, 4):
            batch = _reduction_page(page)
            if not batch:
                break
            rows.extend(batch)
            oldest = pd.to_datetime([item.get("NOTICE_DATE") for item in batch], errors="coerce").min()
            if pd.notna(oldest) and oldest <= cutoff:
                covered = True
                break
    except Exception as exc:
        return {"ok": False, "alarm": False, "error": str(exc)[:160]}
    if not rows:
        return {"ok": False, "alarm": False, "error": "股东减持公告为空"}
    work = pd.DataFrame(rows)
    work["公告日"] = pd.to_datetime(work["NOTICE_DATE"], errors="coerce")
    window = work[work["公告日"] >= cutoff].copy()
    if window.empty:
        return {"ok": True, "alarm": False, "amount_yi": 0.0, "count": 0, "top": [], "covered": covered, "error": ""}
    price = pd.to_numeric(window["CLOSE_PRICE"], errors="coerce")
    price = price.fillna(pd.to_numeric(window.get("NEWEST_PRICE"), errors="coerce"))
    qty = pd.to_numeric(window["CHANGE_NUM"], errors="coerce").abs()
    amount = qty * price * 10000 / 1e8
    window = window.assign(金额=amount)
    total = float(amount.dropna().sum())
    top = []
    for _, item in window.dropna(subset=["金额"]).nlargest(5, "金额").iterrows():
        top.append(
            {
                "名称": str(item.get("SECURITY_NAME_ABBR") or ""),
                "股东": str(item.get("HOLDER_NAME") or ""),
                "金额亿": round(float(item["金额"]), 2),
                "公告日": item["公告日"].strftime("%Y-%m-%d"),
            }
        )
    alarm = bool(covered and total >= REDUCTION_ALARM_YI)
    return {
        "ok": True,
        "alarm": alarm,
        "amount_yi": round(total, 2),
        "count": int(amount.notna().sum()),
        "top": top,
        "covered": covered,
        "error": "" if covered else "最近公告没有覆盖满30日，金额可能偏低，不据此报警",
    }


def load_industry_flow() -> dict:
    """同花顺行业资金流，净额单位为亿元。单日流出不单独作为空仓层。"""
    try:
        raw = _call(lambda: ak.stock_fund_flow_industry(symbol="即时"), timeout=20)
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:160]}
    if raw is None or raw.empty or "净额" not in raw.columns:
        return {"ok": False, "error": "无行业资金流"}
    work = raw.copy()
    work["净额"] = pd.to_numeric(work["净额"], errors="coerce")
    work = work.dropna(subset=["净额"])
    if work.empty:
        return {"ok": False, "error": "行业资金流为空"}

    def _side(frame: pd.DataFrame) -> list[dict]:
        rows = []
        for _, item in frame.iterrows():
            rows.append({"行业": str(item["行业"]), "净额": round(float(item["净额"]), 2)})
        return rows

    growth = work[work["行业"].isin(GROWTH_INDUSTRIES)]
    bank = work[work["行业"] == "银行"]
    liquor = work[work["行业"] == "白酒"]
    return {
        "ok": True,
        "outflow": _side(work.nsmallest(5, "净额")),
        "inflow": _side(work.nlargest(5, "净额")),
        "growth_net": round(float(growth["净额"].sum()), 2) if len(growth) else None,
        "bank_net": round(float(bank["净额"].iloc[0]), 2) if len(bank) else None,
        "liquor_net": round(float(liquor["净额"].iloc[0]), 2) if len(liquor) else None,
        "error": "",
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


def _margin_text(margin: dict) -> str:
    if not margin.get("ok"):
        return f"两融缺失：{margin.get('error') or '无数据'}"
    direction = "减少" if margin["chg5"] < 0 else "增加"
    flag = "，杠杆资金在撤" if margin["alarm"] else ""
    return (
        f"融资余额 {margin['balance']} 亿元（{margin['asof']}），"
        f"近5个交易日{direction} {abs(margin['chg5'])} 亿元（{margin['pct5']}%），"
        f"这5日里有 {margin['down_days']} 日下降{flag}"
    )


def _reduction_text(reduction: dict) -> str:
    if not reduction.get("ok"):
        return f"股东减持缺失：{reduction.get('error') or '无数据'}"
    note = "已到单月1000亿参考线" if reduction["alarm"] else "未到单月1000亿参考线"
    extra = f"。{reduction['error']}" if reduction.get("error") else ""
    return f"近30日已公告股东减持估算 {reduction['amount_yi']} 亿元、{reduction['count']} 笔，{note}{extra}"


def _flow_text(flow: dict) -> str:
    if not flow.get("ok"):
        return f"行业资金流缺失：{flow.get('error') or '无数据'}"
    parts = []
    if flow.get("growth_net") is not None:
        parts.append(f"成长相关行业合计净额 {flow['growth_net']} 亿元")
    if flow.get("bank_net") is not None:
        parts.append(f"银行 {flow['bank_net']} 亿元")
    if flow.get("liquor_net") is not None:
        parts.append(f"白酒 {flow['liquor_net']} 亿元")
    lead = "当日同花顺行业资金（亿元）：" + "，".join(parts) if parts else "当日同花顺行业资金已取到"
    return lead + "。单日行业流向只说明钱往哪边走，不单独当成空仓"


def _us_text(us: dict) -> str:
    if not us.get("ok"):
        return f"美债缺失：{us.get('error') or '无数据'}"
    if us.get("severe"):
        tone = "已到5%，高估值成长的外部约束偏紧"
    elif us["alarm"]:
        tone = "已过4.5%，全球风险资产估值承压"
    else:
        tone = "低于4.5%"
    return f"美国10年期国债收益率 {us['y10']}%（{us['asof']}），{tone}"


def judge(
    pe_rows: list[dict],
    tech: dict[str, dict],
    margin: dict,
    reduction: dict,
    flow: dict,
    us: dict,
) -> dict:
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
    capital_alarm = bool(margin.get("alarm") or reduction.get("alarm"))
    external_alarm = bool(us.get("alarm"))
    alarm_count = sum([val_alarm, capital_alarm, tech_alarm, external_alarm])

    if broad_hot and broad_break and alarm_count >= 2:
        regime = "降低仓位"
        summary = "宽基估值和趋势同时转弱，先少做、多留现金。"
    elif alarm_count >= 2 and growth_tech_alarm and (growth_hot or capital_alarm or external_alarm):
        regime = "结构性减仓"
        summary = "至少两层一起转弱，而且成长方向已经跌破均线。先把仓位从这些方向迁走，红利和还站上均线的方向可以留着看。"
    elif alarm_count >= 2 and growth_hot and capital_alarm:
        regime = "结构性减仓"
        summary = "成长估值和资金层一起报警。先少做成长，往还站上均线的方向看。"
    elif alarm_count == 1:
        regime = "观察"
        summary = "只有一层在报警，按讨论里的规则还不够降仓。先看，不因为单一指标砍仓。"
    elif alarm_count >= 2:
        regime = "观察"
        summary = "有两层在报警，但成长和宽基的趋势还没有一起坏掉。先看结构和外部约束，不把仓位降成现金。"
    else:
        regime = "可以配置"
        summary = "四层都没有报警。可以拿着仓位，只在还站上均线、分位不算极端的题材里选。"

    capital_ok = margin.get("ok") or reduction.get("ok")
    layers = [
        {
            "层": "估值",
            "状态": "报警" if val_alarm else ("正常" if growth_pe and broad_pe else "缺失"),
            "说明": _pe_text(growth_pe, broad_pe, div_pe),
        },
        {
            "层": "资金",
            "状态": "报警" if capital_alarm else ("正常" if capital_ok else "缺失"),
            "说明": "；".join([_margin_text(margin), _reduction_text(reduction), _flow_text(flow)]),
        },
        {
            "层": "技术",
            "状态": "报警" if tech_alarm else ("正常" if cyb.get("ok") else "缺失"),
            "说明": _tech_text(cyb, star, hs300),
        },
        {
            "层": "外部",
            "状态": "报警" if external_alarm else ("正常" if us.get("ok") else "缺失"),
            "说明": _us_text(us),
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
    margin = load_margin()
    us = load_us_yield()
    reduction = load_reductions()
    flow = load_industry_flow()
    pe_rows = [load_pe_anchor(name) for name in PE_ANCHORS]
    codes = list(dict.fromkeys(list(TECH_ETFS) + list(THEME_ETFS)))
    tech = load_etf_states(codes)
    decision = judge(pe_rows, tech, margin, reduction, flow, us)
    avoid, prefer, extended = pick_themes(tech, decision["regime"])
    if decision["regime"] == "降低仓位":
        prefer = [p for p in prefer if p["分组"] in ("红利", "避险", "红利周期")][:5]
    gaps = []
    if not margin.get("ok"):
        gaps.append("两融")
    if not reduction.get("ok"):
        gaps.append("股东减持")
    if not flow.get("ok"):
        gaps.append("行业资金流")
    if not us.get("ok"):
        gaps.append("美债")
    if gaps:
        missing = "这次没取到：" + "、".join(gaps) + "。缺失的层不参与打分。"
    else:
        missing = "两融、股东减持、行业资金流和美债都已取到。减持金额是已公告变动股数乘价格的估算，不是减持计划上限。行业资金是当日净额，只用来看迁仓方向。"
    return {
        "regime": decision["regime"],
        "summary": decision["summary"],
        "layers": decision["layers"],
        "pe": pe_rows,
        "avoid": avoid,
        "prefer": prefer,
        "extended": extended,
        "flow": flow,
        "reduction_top": reduction.get("top") or [],
        "missing": missing,
    }
