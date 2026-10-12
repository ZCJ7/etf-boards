"""先判断全市场，再选大方向。

空仓只在两件事同时出现时成立：
1. 沪深300、中证500、中证1000、创业板、科创50 里，至少 4 个收盘跌破 MA60
2. 再有一层确认：两融在撤、美债 10 年期达到 5%、宽基市盈率分位极高，或近 30 日减持估算达到 1000 亿

没到这一步，就不把账户当成空仓。方向分（满分大约 100）：
- 趋势 45：该方向代表 ETF 里站上 MA20 的比例
- 动量 25：近 20 日收益在七个方向里的位置
- 资金 20：对应行业当日净流入的位置；没有行业映射的记中性
- 研报 10：近 14 日行业研报篇数的位置

至少一半代表 ETF 还站上 MA20，并且分数达到 40，才进入优先方向，最多两个。
研报和资金不能把已经跌破的方向抬回来。
"""

from __future__ import annotations

import concurrent.futures as cf

import akshare as ak
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

# 全市场是否进入下行期，只看这五个宽基
MARKET = (
    ("510300", "沪深300"),
    ("510500", "中证500"),
    ("512100", "中证1000"),
    ("159915", "创业板"),
    ("588000", "科创50"),
)

# 大方向。flow_keys 用来加总行业资金，report_keys 用来数行业研报。
# 列表顺序也是研报归类的优先级，避免「化学制药」被算进周期。
DIRECTIONS = (
    {
        "name": "医药",
        "etfs": (("512010", "医药ETF"),),
        "flow_keys": ("制药", "生物制品", "中药", "医疗"),
        "report_keys": ("医药", "制药", "创新药", "医疗", "中药", "生物制品"),
    },
    {
        "name": "科技",
        "etfs": (
            ("588000", "科创50ETF"),
            ("512480", "半导体ETF"),
            ("512760", "芯片ETF"),
            ("159819", "人工智能ETF"),
            ("515050", "5GETF"),
        ),
        "flow_keys": ("半导体", "元件", "光学", "消费电子", "通信", "计算机", "软件", "互联网", "军工电子"),
        "report_keys": ("半导体", "芯片", "通信", "计算机", "软件", "电子", "人工智能", "光模块", "科创", "算力"),
    },
    {
        "name": "金融红利",
        "etfs": (("512800", "银行ETF"), ("510880", "红利ETF"), ("512890", "红利低波ETF")),
        "flow_keys": ("银行", "证券", "保险"),
        "report_keys": ("银行", "非银", "证券", "保险", "红利"),
    },
    {
        "name": "消费",
        "etfs": (("159928", "消费ETF"), ("512690", "酒ETF")),
        "flow_keys": ("白酒", "食品", "饮料", "家电", "旅游", "零售", "酒店"),
        "report_keys": ("食品", "白酒", "饮料", "家电", "旅游", "零售", "酒店", "消费", "社服"),
    },
    {
        "name": "周期",
        "etfs": (("512400", "有色金属ETF"), ("515220", "煤炭ETF"), ("159981", "能源化工ETF")),
        "flow_keys": ("工业金属", "贵金属", "小金属", "煤炭", "化学制品", "化学原料", "石油", "钢铁"),
        "report_keys": ("有色", "煤炭", "化工", "石油", "钢铁", "金属", "锂电", "电池"),
    },
    {
        "name": "跨境",
        "etfs": (("513100", "纳指ETF"), ("513500", "标普500ETF")),
        "flow_keys": (),
        "report_keys": ("港股", "美股", "纳指", "恒生", "海外"),
    },
    {
        "name": "避险",
        "etfs": (("518880", "黄金ETF"),),
        "flow_keys": (),
        "report_keys": ("黄金",),
    },
)

DEFENSIVE = {"金融红利", "避险"}
HOT_PCT = 85.0
BROAD_HOT_PCT = 90.0
PRICE_EXTENDED = 90.0
MIN_ABOVE_RATIO = 0.5
MIN_DIRECTION_SCORE = 40.0
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
    ma60 = None
    above_ma60 = None
    if len(s) >= 60:
        ma60_s = s.rolling(60).mean()
        if pd.notna(ma60_s.iloc[-1]):
            ma60 = round(float(ma60_s.iloc[-1]), 3)
            above_ma60 = last > ma60
    return {
        "ok": True,
        "close": round(last, 3),
        "ma20": round(ma, 3),
        "ma60": ma60,
        "above_ma20": last > ma,
        "above_ma60": above_ma60,
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
        "by_name": {str(item["行业"]): round(float(item["净额"]), 2) for _, item in work.iterrows()},
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


def _rank_points(values: list[float | None], cap: float, neutral: float) -> list[float]:
    """横向位置换算成分数。能比较的样本少于 2 个时，全部用中性分。"""
    known = [(i, v) for i, v in enumerate(values) if v is not None]
    out = [neutral] * len(values)
    if len(known) < 2:
        return out
    for i, value in known:
        pct = sum(1 for _, other in known if other <= value) / len(known)
        out[i] = round(pct * cap, 1)
    return out


def _match_report(text: str) -> str | None:
    for item in DIRECTIONS:
        if any(key in text for key in item["report_keys"]):
            return item["name"]
    return None


def load_reports() -> dict:
    """近 14 日东方财富行业研报，按标题和行业名归到大方向。策略研报只摘录，不打分。"""
    import datetime
    import requests

    end = datetime.date.today()
    begin = end - datetime.timedelta(days=14)
    url = "https://reportapi.eastmoney.com/report/list"

    def fetch(qtype: str, pages: int, size: int) -> list[dict]:
        rows: list[dict] = []
        for page in range(1, pages + 1):
            resp = requests.get(
                url,
                params={
                    "industryCode": "*",
                    "pageSize": str(size),
                    "industry": "*",
                    "rating": "*",
                    "ratingChange": "*",
                    "beginTime": begin.isoformat(),
                    "endTime": end.isoformat(),
                    "pageNo": str(page),
                    "qType": qtype,
                    "orgCode": "",
                    "code": "",
                    "p": str(page),
                    "pageNum": str(page),
                    "pageNumber": str(page),
                },
                timeout=20,
            )
            resp.raise_for_status()
            payload = resp.json()
            batch = payload.get("data") or []
            if not batch:
                break
            rows.extend(batch)
            oldest = pd.to_datetime(batch[-1].get("publishDate"), errors="coerce")
            if pd.isna(oldest) or oldest.normalize() < pd.Timestamp(begin):
                break
            if page * size >= int(payload.get("hits") or 0):
                break
        return rows

    try:
        industry_rows = fetch("1", 4, 80)
        strategy_rows = fetch("2", 1, 8)
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:160], "counts": {}, "samples": [], "strategy": []}

    counts = {item["name"]: 0 for item in DIRECTIONS}
    samples: dict[str, list[str]] = {item["name"]: [] for item in DIRECTIONS}
    cutoff = pd.Timestamp(begin)
    used = 0
    for row in industry_rows:
        published = pd.to_datetime(row.get("publishDate"), errors="coerce")
        if pd.isna(published) or published.normalize() < cutoff:
            continue
        text = f"{row.get('industryName') or ''}{row.get('indvInduName') or ''}{row.get('title') or ''}"
        name = _match_report(text)
        used += 1
        if not name:
            continue
        counts[name] += 1
        title = str(row.get("title") or "").strip()
        if title and len(samples[name]) < 2:
            samples[name].append(title[:42])
    strategy = []
    for row in strategy_rows:
        title = str(row.get("title") or "").strip()
        org = str(row.get("orgSName") or "").strip()
        if title:
            strategy.append(f"{org}：{title}" if org else title)
    return {
        "ok": True,
        "counts": counts,
        "samples": samples,
        "strategy": strategy[:5],
        "used": used,
        "error": "",
    }


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


def _flow_nets(flow: dict) -> tuple[dict[str, float | None], dict[str, bool]]:
    by_name = flow.get("by_name") or {}
    nets: dict[str, float | None] = {}
    matched: dict[str, bool] = {}
    claimed: set[str] = set()
    for item in DIRECTIONS:
        if not item["flow_keys"] or not flow.get("ok"):
            nets[item["name"]] = None
            matched[item["name"]] = False
            continue
        total = 0.0
        hit = False
        for name, value in by_name.items():
            if name in claimed:
                continue
            if any(key in name for key in item["flow_keys"]):
                total += float(value)
                hit = True
                claimed.add(name)
        nets[item["name"]] = round(total, 2) if hit else None
        matched[item["name"]] = hit
    return nets, matched


def _etf_item(code: str, name: str, direction: str, row: dict, reason: str) -> dict:
    return {
        "代码": code,
        "名称": name,
        "分组": direction,
        "收盘": row.get("close"),
        "站上MA20": "是" if row.get("above_ma20") else "否",
        "站上MA60": "是" if row.get("above_ma60") else ("缺失" if row.get("above_ma60") is None else "否"),
        "近20日%": row.get("ret20"),
        "价格分位%": row.get("price_pct"),
        "原因": reason,
    }


def _score_directions(tech: dict[str, dict], flow: dict, reports: dict) -> list[dict]:
    packs = []
    for item in DIRECTIONS:
        etfs = []
        for code, name in item["etfs"]:
            row = tech.get(code) or {}
            if row.get("ok"):
                etfs.append((code, name, row))
        ok_n = len(etfs)
        above = [one for one in etfs if one[2].get("above_ma20")]
        ratio = (len(above) / ok_n) if ok_n else 0.0
        rets = [one[2]["ret20"] for one in etfs if one[2].get("ret20") is not None]
        mom = round(sum(rets) / len(rets), 2) if rets else None
        pcts = [one[2]["price_pct"] for one in above if one[2].get("price_pct") is not None]
        extended = bool(pcts) and (sorted(pcts)[len(pcts) // 2] >= PRICE_EXTENDED)
        packs.append(
            {
                "name": item["name"],
                "etfs": etfs,
                "above": above,
                "ratio": ratio,
                "mom": mom,
                "extended": extended,
                "eligible": ok_n > 0 and ratio >= MIN_ABOVE_RATIO,
            }
        )
    nets, _matched = _flow_nets(flow)
    counts = reports.get("counts") or {}
    mom_pts = _rank_points([p["mom"] if p["mom"] is not None else None for p in packs], 25, 12.5)
    flow_pts = _rank_points([nets[p["name"]] for p in packs], 20, 10)
    report_pts = _rank_points(
        [float(counts.get(p["name"], 0)) for p in packs] if reports.get("ok") else [None] * len(packs),
        10,
        5,
    )
    rows = []
    for i, pack in enumerate(packs):
        trend_pts = round(45 * pack["ratio"], 1)
        penalty = 8 if pack["extended"] else 0
        score = round(max(0, trend_pts + mom_pts[i] + flow_pts[i] + report_pts[i] - penalty), 1)
        net = nets[pack["name"]]
        count = counts.get(pack["name"]) if reports.get("ok") else None
        if not pack["etfs"]:
            verdict = "数据缺失"
        elif not pack["eligible"]:
            verdict = "先不碰"
        elif score < MIN_DIRECTION_SCORE:
            verdict = "分数不够"
        elif pack["extended"]:
            verdict = "线上但价格已高"
        else:
            verdict = "可比较"
        bits = [f"站上MA20的比例 {round(pack['ratio'] * 100)}%"]
        if pack["mom"] is not None:
            bits.append(f"近20日平均 {pack['mom']}%")
        if net is None:
            bits.append("行业资金中性")
        else:
            bits.append(f"行业资金 {net} 亿元")
        if count is None:
            bits.append("研报缺失")
        else:
            bits.append(f"近14日行业研报 {count} 篇")
        if pack["extended"]:
            bits.append("站上均线的ETF价格分位已高")
        rows.append(
            {
                "方向": pack["name"],
                "分数": score,
                "趋势": trend_pts,
                "动量": mom_pts[i],
                "资金": flow_pts[i],
                "研报": report_pts[i],
                "状态": verdict,
                "说明": "，".join(bits),
                "eligible": pack["eligible"],
                "extended": pack["extended"],
                "etfs": pack["etfs"],
                "above": pack["above"],
            }
        )
    return rows


def _market_rows(tech: dict[str, dict]) -> list[dict]:
    rows = []
    for code, name in MARKET:
        row = tech.get(code) or {}
        if not row.get("ok"):
            rows.append({"代码": code, "指数": name, "状态": "缺失", "above_ma60": None})
            continue
        if row.get("above_ma60") is False:
            state = "跌破MA60"
        elif row.get("above_ma20"):
            state = "站上MA60和MA20"
        else:
            state = "站上MA60，短线在MA20下"
        rows.append(
            {
                "代码": code,
                "指数": name,
                "收盘": row["close"],
                "MA20": row["ma20"],
                "MA60": row.get("ma60"),
                "近20日%": row.get("ret20"),
                "状态": state,
                "截止": row["asof"],
                "above_ma60": row.get("above_ma60"),
            }
        )
    return rows


def judge(
    pe_rows: list[dict],
    tech: dict[str, dict],
    margin: dict,
    reduction: dict,
    flow: dict,
    us: dict,
    reports: dict,
) -> dict:
    broad = next((r for r in pe_rows if r["name"] == "沪深300" and r.get("ok")), None)
    mid = next((r for r in pe_rows if r["name"] == "中证500" and r.get("ok")), None)
    growth = next((r for r in pe_rows if r["name"] == "创业板50" and r.get("ok")), None)
    div = next((r for r in pe_rows if r["name"] == "上证红利" and r.get("ok")), None)
    val_hot = bool(broad and broad["percentile"] >= BROAD_HOT_PCT) or bool(
        broad and mid and broad["percentile"] >= HOT_PCT and mid["percentile"] >= HOT_PCT
    )
    market = _market_rows(tech)
    valid = [r for r in market if r.get("above_ma60") is not None]
    below = [r for r in valid if r["above_ma60"] is False]
    downtrend = len(valid) >= 4 and len(below) >= 4
    confirms = []
    if margin.get("alarm"):
        confirms.append("两融余额在下降")
    if us.get("severe"):
        confirms.append("美债10年期收益率到了5%以上")
    if val_hot:
        confirms.append("宽基市盈率分位已经很高")
    if reduction.get("alarm"):
        confirms.append("股东减持估算到了1000亿")

    directions = _score_directions(tech, flow, reports)
    ranked = sorted(directions, key=lambda x: -x["分数"])
    picks = [r for r in ranked if r["eligible"] and r["分数"] >= MIN_DIRECTION_SCORE]
    if downtrend and confirms:
        picks = [r for r in picks if r["方向"] in DEFENSIVE]
        regime = "适合降低仓位"
    elif downtrend:
        regime = "市场偏弱"
    elif len(below) >= 2:
        regime = "分化配置"
    else:
        regime = "可以配置"
    picks = picks[:2]
    picked_names = {r["方向"] for r in picks}
    for row in directions:
        chosen = row["方向"] in picked_names
        if chosen and row["extended"]:
            row["状态"] = "优先但价格已高"
        elif chosen:
            row["状态"] = "优先"
        elif row["状态"] == "数据缺失":
            continue
        elif not row["eligible"]:
            row["状态"] = "先不碰"
        elif regime == "适合降低仓位":
            row["状态"] = "趋势还在，先降仓"
        elif row["分数"] < MIN_DIRECTION_SCORE:
            row["状态"] = "分数不够"
        else:
            row["状态"] = "备选"

    weak_names = "、".join(r["指数"] for r in below) or "没有"
    hold_names = "、".join(r["指数"] for r in valid if r["above_ma60"]) or "没有"
    if regime == "适合降低仓位":
        summary = (
            f"五个宽基里有 {len(below)} 个收盘在 MA60 下方（{weak_names}），"
            f"同时{'、'.join(confirms)}。这是全市场下行期，适合降低仓位。"
        )
    elif regime == "市场偏弱":
        summary = (
            f"五个宽基里有 {len(below)} 个收盘在 MA60 下方（{weak_names}）。"
            "中期已经偏弱，但两融、美债、估值、减持还没有一起确认，仓位要轻。"
        )
        if picks:
            summary += "还站得住的方向是" + "、".join(r["方向"] for r in picks) + "。"
        else:
            summary += "七个方向里也没有一半以上的代表 ETF 还站在 MA20 上。"
    elif not picks:
        regime = "没有清晰主线"
        summary = (
            f"还没到全市场空仓。站上 MA60 的是 {hold_names}。"
            "七个方向里没有一个同时满足「至少一半 ETF 站上 MA20」和分数线，先观望。"
        )
    else:
        names = "、".join(r["方向"] for r in picks)
        summary = (
            f"还没到全市场空仓。五个宽基里跌破 MA60 的有 {len(below)} 个，站上的是 {hold_names}。"
            f"大方向上优先看{names}。"
        )
    if picks and any(r["extended"] for r in picks):
        summary += "其中价格已经很高的，只适合拿着，不适合追。"
    if regime == "适合降低仓位" and picks:
        summary += "如果还要留一点仓，只看" + "、".join(r["方向"] for r in picks) + "。"
    elif regime == "适合降低仓位":
        summary += "防守方向自己也没站稳，留现金就行。"

    layers = [
        {
            "层": "市场趋势",
            "状态": "下行" if downtrend else ("正常" if valid else "缺失"),
            "说明": f"{len(valid)} 个宽基有 MA60，其中 {len(below)} 个在均线下方。",
        },
        {
            "层": "估值",
            "状态": "报警" if val_hot else ("正常" if broad and mid else "缺失"),
            "说明": _pe_text(growth, broad, mid, div),
        },
        {
            "层": "资金",
            "状态": "报警" if margin.get("alarm") or reduction.get("alarm") else ("正常" if margin.get("ok") else "缺失"),
            "说明": "；".join([_margin_text(margin), _reduction_text(reduction), _flow_text(flow)]),
        },
        {
            "层": "外部",
            "状态": "报警" if us.get("severe") else ("承压" if us.get("alarm") else ("正常" if us.get("ok") else "缺失")),
            "说明": _us_text(us),
        },
    ]
    return {
        "regime": regime,
        "summary": summary,
        "layers": layers,
        "market": market,
        "directions": directions,
        "picks": picks,
        "downtrend": downtrend,
        "confirms": confirms,
    }


def _pe_text(growth, broad, mid, div) -> str:
    parts = []
    for row, label in ((broad, "沪深300"), (mid, "中证500"), (growth, "创业板50"), (div, "上证红利")):
        if not row or not row.get("ok"):
            parts.append(f"{label}市盈率缺失")
            continue
        parts.append(f"{label} {row['pe']} 倍，分位 {row['percentile']}%")
    return "；".join(parts)


def _pick_etfs(directions: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    prefer, extended, avoid = [], [], []
    for row in directions:
        if row["状态"] in ("优先", "优先但价格已高"):
            for code, name, etf in row["above"]:
                reason = "方向优先，且站上 MA20"
                item = _etf_item(code, name, row["方向"], etf, reason)
                if etf.get("price_pct") is not None and etf["price_pct"] >= PRICE_EXTENDED:
                    item["原因"] = "方向还在，但这只近3年价格分位已经很高"
                    extended.append(item)
                else:
                    prefer.append(item)
        elif row["状态"] == "先不碰":
            avoid.append(
                {
                    "代码": "—",
                    "名称": row["方向"],
                    "分组": row["方向"],
                    "收盘": "—",
                    "站上MA20": "否",
                    "站上MA60": "—",
                    "近20日%": None,
                    "价格分位%": None,
                    "原因": row["说明"],
                }
            )
    prefer.sort(key=lambda x: -(x["近20日%"] or -999))
    extended.sort(key=lambda x: -(x["近20日%"] or -999))
    return prefer, extended, avoid


def run_cash_regime() -> dict:
    margin = load_margin()
    us = load_us_yield()
    reduction = load_reductions()
    flow = load_industry_flow()
    reports = load_reports()
    pe_rows = [load_pe_anchor(name) for name in PE_ANCHORS]
    codes = [code for code, _name in MARKET]
    for item in DIRECTIONS:
        codes.extend(code for code, _name in item["etfs"])
    tech = load_etf_states(list(dict.fromkeys(codes)))
    decision = judge(pe_rows, tech, margin, reduction, flow, us, reports)
    prefer, extended, avoid = _pick_etfs(decision["directions"])
    gaps = []
    if not margin.get("ok"):
        gaps.append("两融")
    if not reduction.get("ok"):
        gaps.append("股东减持")
    if not flow.get("ok"):
        gaps.append("行业资金流")
    if not us.get("ok"):
        gaps.append("美债")
    if not reports.get("ok"):
        gaps.append("行业研报")
    if gaps:
        missing = "这次没取到：" + "、".join(gaps) + "。缺的那一项记成中性，不单独改变结论。"
    else:
        missing = "两融、减持、行业资金、美债和近14日行业研报都已取到。减持金额是已公告变动的估算。策略研报只摘录，不进分数。"
    public_dirs = []
    for row in decision["directions"]:
        public_dirs.append({k: v for k, v in row.items() if k not in ("etfs", "above", "eligible", "extended")})
    return {
        "regime": decision["regime"],
        "summary": decision["summary"],
        "layers": decision["layers"],
        "market": decision["market"],
        "directions": public_dirs,
        "pe": pe_rows,
        "avoid": avoid,
        "prefer": prefer,
        "extended": extended,
        "flow": {k: v for k, v in flow.items() if k != "by_name"},
        "reports": reports.get("strategy") or [],
        "reduction_top": reduction.get("top") or [],
        "missing": missing,
    }
