"""市场资讯聚合：对齐 daily-stock-analysis 的多源资讯能力（本地 AkShare 实现）。

语料分析遵循投资逻辑（而非裸关键词撞车）：
1) 先定资产域（地产/海外宏观/公司事件/A股交易…）
2) 再映射投资因子（杠杆流动性、拥挤情绪、主线风格、外围传导、政策基本面）
3) 仅在「可映射到 A 股交易」时给出风险/主升方向

大盘复盘见 analysis.market_review；本模块专注资讯拉取与标注。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass
from datetime import date, datetime

import akshare as ak
import pandas as pd
import streamlit as st

# ---------- 资产域词典（优先级：越靠前越先判定） ----------
DOMAIN_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "地产/楼市",
        (
            "房价", "租金", "房地产", "楼市", "地产", "房企", "开发商", "二手房", "新房",
            "住宅", "商品房", "按揭", "房贷", "土地出让", "去库存", "供过于求",
        ),
    ),
    (
        "海外宏观",
        (
            "英央行", "英格兰银行", "英国加息", "美联储", "鲍威尔", "欧央行", "欧洲央行",
            "日银", "日本央行", "加息决定", "降息押注", "非农", "CPI", "PCE",
        ),
    ),
    (
        "大宗/能源",
        ("原油", "黄金", "铜价", "铁矿石", "煤炭价格", "天然气", "布伦特", "WTI"),
    ),
    (
        "公司事件",
        ("回购", "增持", "减持", "财报", "净利", "营收", "分红", "定增", "并购", "重组", "停牌"),
    ),
    (
        "产业政策",
        ("产业政策", "十五五", "国务院", "证监会", "央行", "降准", "再贷款", "财政", "补贴"),
    ),
]

# A 股交易锚点：有它才允许映射「股市风险/主升」
EQUITY_TRADE_ANCHOR = (
    "A股", "上证", "沪指", "深成指", "创业板", "科创", "沪深两市", "两市成交",
    "北向", "陆股通", "沪股通", "深股通", "沪深300", "中证500", "中证1000",
    "融资余额", "两融", "ETF", "个股", "板块轮动", "主力资金", "涨停", "跌停", "股指",
    "科创50", "创业板指",
)

# 投资因子 → (风险向线索, 主升向线索)  —— 均需已通过资产域门禁
FACTOR_LEXICON: dict[str, dict[str, tuple[str, ...]]] = {
    "杠杆流动性": {
        "risk": ("去杠杆", "两融降温", "融资盘", "杠杆出清", "融资余额下降", "担保比例"),
        "bull": ("融资余额创新高", "增量资金", "杠杆回升", "两融活跃"),
    },
    "资金流向": {
        "risk": ("主力资金净流出", "北向净流出", "外资流出A股", "外资抛售A股", "ETF净赎回"),
        "bull": ("北向大幅净流入", "主力资金净流入", "ETF净申购", "宽基流入", "陆股通净买入"),
    },
    "拥挤情绪": {
        "risk": ("拥挤度", "交易拥挤", "杀跌", "恐慌抛售", "瀑布式下跌", "牛市高风险", "高风险阶段", "泡沫"),
        "bull": ("赚钱效应", "普涨", "景气扩散", "情绪修复", "反弹确认"),
    },
    "主线风格": {
        "risk": ("科技股抛售", "高位抱团松动", "估值回调", "风格切换防御"),
        "bull": ("主线确认", "主升浪", "趋势确立", "全面看多", "强势上涨", "突破前高", "放量突破"),
    },
    "外围传导": {
        "risk": ("美股科技", "纳指大跌", "费城半导体", "海外加息", "风险偏好下降"),
        "bull": ("美股科技反弹", "外围风险缓和"),
    },
    "政策基本面": {
        "risk": ("监管降温", "规范融资"),
        "bull": ("政策催化", "盈利驱动", "流动性驱动", "产业政策加码"),
    },
}

BULL_VETO = (
    "高风险", "风险阶段", "见顶", "泡沫", "尾声", "过热", "警惕", "谨慎",
    "警告", "警示", "去杠杆", "杀跌", "暴跌", "回调风险",
)

BULL_CONFIRM = ("启动", "确立", "主升", "行情延续", "赚钱效应", "普涨", "趋势向上")

CASE_NOTE = """
**分析框架（贴合投资逻辑）**

1. **先定资产域**：地产/海外宏观/大宗/公司事件/政策/A股交易——跨域词（外资、暴跌、净流出）不直接当股市信号。  
2. **再映射投资因子**：杠杆流动性 · 资金流向 · 拥挤情绪 · 主线风格 · 外围传导 · 政策基本面。  
3. **最后给方向**：仅「可映射到 A 股交易」的条目才标风险/主升；其余标领域供背景，不进双雷达。

**历史锚点**  
- 急跌：2026-07 瀑布（去杠杆、高位科技杀跌）  
- 主升：对标 2025-12 酝酿、2026-04 启动（赚钱效应、增量资金、主线确认）
"""

JULY_CASE_NOTE = CASE_NOTE


@dataclass(frozen=True)
class NewsSource:
    key: str
    name: str
    authority: str  # 权威性
    cadence: str  # 更新节奏
    timeliness: str  # 及时性
    why: str
    default_on: bool = True
    timeout: float = 12


SOURCES: dict[str, NewsSource] = {
    "em": NewsSource(
        key="em",
        name="东方财富·快讯",
        authority="高（主流行情终端资讯）",
        cadence="持续滚动",
        timeliness="很高",
        why="覆盖广、接口稳定，适合做风险雷达主源",
        default_on=True,
        timeout=12,
    ),
    "ths": NewsSource(
        key="ths",
        name="同花顺·快讯",
        authority="较高（A股零售端主流）",
        cadence="持续滚动",
        timeliness="很高",
        why="偏 A 股交易语境，补充东财未覆盖的短讯",
        default_on=True,
        timeout=12,
    ),
    "sina": NewsSource(
        key="sina",
        name="新浪财经·全球快讯",
        authority="较高（门户聚合）",
        cadence="持续滚动",
        timeliness="高",
        why="常聚合机构观点与外围扰动，便于对照内外联动",
        default_on=True,
        timeout=12,
    ),
    "cx": NewsSource(
        key="cx",
        name="财新·要点",
        authority="很高（严肃财经媒体）",
        cadence="日更为主",
        timeliness="中高（偏深度）",
        why="权威叙事与宏观/产业背景，补快讯深度不足",
        default_on=True,
        timeout=12,
    ),
    "futu": NewsSource(
        key="futu",
        name="富途·快讯",
        authority="中高",
        cadence="持续滚动",
        timeliness="高",
        why="公司公告与市场事件补充",
        default_on=False,
        timeout=12,
    ),
    "cls": NewsSource(
        key="cls",
        name="财联社·电报",
        authority="很高（机构盘中电报标配）",
        cadence="盘中高频",
        timeliness="极高",
        why="盘中主源；易超时卡顿，默认关闭，需要时再勾选",
        default_on=False,
        timeout=15,
    ),
    "cctv": NewsSource(
        key="cctv",
        name="新闻联播·文字稿",
        authority="很高（官方日更）",
        cadence="每日",
        timeliness="日级（覆盖全天叙事）",
        why="补快讯只覆盖近几小时的盲区，提供当日宏观/政策主线",
        default_on=True,
        timeout=25,
    ),
}


def _call_timeout(fn, timeout: float = 12):
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


def _norm_frame(df: pd.DataFrame, source_key: str, title_col: str, content_col: str, time_col: str, url_col: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["来源", "标题", "内容", "时间", "链接", "风险标记"])
    work = df.copy()
    title = work[title_col].astype(str) if title_col in work.columns else pd.Series([""] * len(work))
    content = work[content_col].astype(str) if content_col in work.columns else title
    # 若标题空则用内容前段
    title = title.where(title.str.strip().ne("") & title.str.lower().ne("nan"), content.str.slice(0, 48))
    times = work[time_col] if time_col in work.columns else pd.Series([""] * len(work))
    urls = work[url_col].astype(str) if url_col and url_col in work.columns else pd.Series([""] * len(work))
    src_name = SOURCES[source_key].name
    out = pd.DataFrame(
        {
            "来源": src_name,
            "来源键": source_key,
            "标题": title.fillna("").astype(str).str.strip(),
            "内容": content.fillna("").astype(str).str.strip(),
            "时间": times.astype(str),
            "链接": urls.fillna("").astype(str),
        }
    )
    out = out[out["标题"].ne("") | out["内容"].ne("")]
    return out.reset_index(drop=True)


def _fetch_em() -> pd.DataFrame:
    df = ak.stock_info_global_em()
    return _norm_frame(df, "em", "标题", "摘要", "发布时间", "链接")


def _fetch_ths() -> pd.DataFrame:
    df = ak.stock_info_global_ths()
    return _norm_frame(df, "ths", "标题", "内容", "发布时间", "链接")


def _fetch_sina() -> pd.DataFrame:
    df = ak.stock_info_global_sina()
    # 新浪只有时间+内容
    if df is None or df.empty:
        return _norm_frame(pd.DataFrame(), "sina", "标题", "内容", "时间", None)
    work = df.copy()
    work["标题"] = work["内容"].astype(str).str.slice(0, 40)
    return _norm_frame(work, "sina", "标题", "内容", "时间", None)


def _fetch_cx() -> pd.DataFrame:
    df = ak.stock_news_main_cx()
    if df is None or df.empty:
        return _norm_frame(pd.DataFrame(), "cx", "标题", "内容", "时间", "链接")
    work = df.copy()
    # tag / summary / url
    work["标题"] = work.get("tag", pd.Series([""] * len(work))).astype(str)
    work["内容"] = work.get("summary", pd.Series([""] * len(work))).astype(str)
    work["时间"] = ""
    work["链接"] = work.get("url", pd.Series([""] * len(work))).astype(str)
    return _norm_frame(work, "cx", "标题", "内容", "时间", "链接")


def _fetch_futu() -> pd.DataFrame:
    df = ak.stock_info_global_futu()
    return _norm_frame(df, "futu", "标题", "内容", "发布时间", "链接")


def _fetch_cls() -> pd.DataFrame:
    df = ak.stock_info_global_cls(symbol="全部")
    if df is None or df.empty:
        return _norm_frame(pd.DataFrame(), "cls", "标题", "内容", "时间", "链接")
    # 列名随版本可能变化，做宽松映射
    cols = {str(c): c for c in df.columns}
    title_c = next((cols[k] for k in cols if "标题" in k or "title" in k.lower()), df.columns[0])
    content_c = next((cols[k] for k in cols if "内容" in k or "正文" in k or "content" in k.lower()), title_c)
    time_c = next((cols[k] for k in cols if "时间" in k or "time" in k.lower() or "发布" in k), df.columns[-1])
    url_c = next((cols[k] for k in cols if "链接" in k or "url" in k.lower()), None)
    return _norm_frame(df, "cls", str(title_c), str(content_c), str(time_c), str(url_c) if url_c else None)


def _fetch_cctv() -> pd.DataFrame:
    """央视新闻联播文字稿：按自然日拉取，补快讯「只覆盖近几小时」的缺口。"""
    today = date.today()
    for d in (today,):
        raw = None
        try:
            raw = ak.news_cctv(date=d.strftime("%Y%m%d"))
        except Exception:
            raw = None
        if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
            continue
        work = raw.copy()
        title_c = "title" if "title" in work.columns else work.columns[0]
        content_c = "content" if "content" in work.columns else title_c
        work["标题"] = work[title_c].astype(str)
        work["内容"] = work[content_c].astype(str)
        work["时间"] = f"{d.isoformat()} 19:00:00"
        work["链接"] = ""
        return _norm_frame(work, "cctv", "标题", "内容", "时间", "链接")
    return _norm_frame(pd.DataFrame(), "cctv", "标题", "内容", "时间", "链接")


_FETCHERS = {
    "em": _fetch_em,
    "ths": _fetch_ths,
    "sina": _fetch_sina,
    "cx": _fetch_cx,
    "futu": _fetch_futu,
    "cls": _fetch_cls,
    "cctv": _fetch_cctv,
}


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def _collect_hits(text: str, words: tuple[str, ...], limit: int = 4) -> list[str]:
    return [w for w in words if w in text][:limit]


def classify_domain(text: str) -> str:
    """先定资产域：投资逻辑的第一步。"""
    if _has_any(text, EQUITY_TRADE_ANCHOR):
        # 即便夹杂地产/宏观词，只要明确 A 股交易锚点，仍以 A 股交易为主域
        return "A股交易"
    for name, keys in DOMAIN_RULES:
        if _has_any(text, keys):
            return name
    if "牛市" in text and _has_any(text, ("高风险", "风险阶段", "见顶", "泡沫", "尾声", "过热")):
        return "股市叙事"
    if "牛市" in text and _has_any(text, BULL_CONFIRM):
        return "股市叙事"
    return "其他"


def map_investment_factors(text: str, domain: str) -> tuple[list[str], list[str], list[str]]:
    """
    映射投资因子与方向线索。
    返回 (因子列表, 风险线索, 主升线索)。
    非 A 股可映射域：最多保留「外围观察」，不产出主升。
    """
    factors: list[str] = []
    risk: list[str] = []
    bull: list[str] = []

    # 地产/公司事件等：不映射股市双雷达
    if domain in {"地产/楼市", "公司事件", "大宗/能源", "其他"}:
        return [], [], []

    equity_ok = domain in {"A股交易", "股市叙事"} or _has_any(text, EQUITY_TRADE_ANCHOR)

    for factor, sides in FACTOR_LEXICON.items():
        r_hits = _collect_hits(text, sides["risk"])
        b_hits = _collect_hits(text, sides["bull"])
        if not r_hits and not b_hits:
            continue

        if factor == "外围传导":
            # 海外宏观本身只做观察；有 A 股锚点才升为风险线索
            factors.append(factor)
            if equity_ok and r_hits:
                risk.extend(r_hits)
            elif domain == "海外宏观" and r_hits:
                risk.append("外围观察:" + r_hits[0])
            if equity_ok and b_hits and not _has_any(text, BULL_VETO):
                bull.extend(b_hits)
            continue

        if factor == "政策基本面" and domain == "产业政策" and not equity_ok:
            factors.append(factor)
            # 政策可作背景观察，不直接标主升/急跌
            continue

        if not equity_ok:
            continue

        factors.append(factor)
        risk.extend(r_hits)
        if not _has_any(text, BULL_VETO):
            bull.extend(b_hits)

    # 显式北向/陆股通
    if equity_ok and _has_any(text, ("北向资金", "陆股通", "沪股通", "深股通")):
        if "资金流向" not in factors:
            factors.append("资金流向")
        if _has_any(text, ("净流出", "流出", "抛售")) and "北向/陆股通" not in risk:
            risk.append("北向/陆股通")
        if _has_any(text, ("净流入", "流入", "净买入")) and "北向/陆股通流入" not in bull:
            if not _has_any(text, BULL_VETO):
                bull.append("北向/陆股通流入")

    # 牛市 + 高风险 → 只留风险
    if "牛市" in text and _has_any(text, ("高风险", "风险阶段", "见顶", "泡沫", "尾声", "过热")):
        if "拥挤情绪" not in factors:
            factors.append("拥挤情绪")
        if "牛市高风险" not in risk:
            risk.append("牛市高风险")
        bull = []

    # 裸「牛市」偏多确认
    if equity_ok and "牛市" in text and _has_any(text, BULL_CONFIRM) and not _has_any(text, BULL_VETO):
        if "拥挤情绪" not in factors:
            factors.append("拥挤情绪")
        if "牛市" not in bull:
            bull.append("牛市确认")

    # 海外宏观无 A 股锚点：清空主升，风险最多外围观察
    if domain == "海外宏观" and not equity_ok:
        bull = []
        risk = [x for x in risk if x.startswith("外围观察") or x == "海外加息"]
        if _has_any(text, ("加息",)) and not any("加息" in x for x in risk):
            risk.append("外围观察:海外加息")

    # 去重保序
    def _uniq(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out[:6]

    return _uniq(factors), _uniq(risk), _uniq(bull)


def analyze_news_item(text: str) -> dict:
    """单条资讯的投资逻辑标注。"""
    domain = classify_domain(text)
    factors, risk, bull = map_investment_factors(text, domain)

    # 信号类型
    if risk and bull:
        kind = "风险+主升"
    elif risk and any(not x.startswith("外围观察") for x in risk):
        kind = "风险"
    elif risk and all(x.startswith("外围观察") for x in risk):
        kind = "外围观察"
    elif bull:
        kind = "主升"
    elif factors:
        kind = "背景"
    else:
        kind = ""

    note = ""
    if domain == "地产/楼市":
        note = "地产叙事；外资/暴跌不映射股市北向"
    elif domain == "海外宏观" and kind in {"外围观察", ""}:
        note = "海外宏观；需 A 股交易锚点才进双雷达"
    elif domain == "公司事件":
        note = "公司事件；不直接当指数风险/主升"

    return {
        "领域": domain,
        "投资因子": "、".join(factors),
        "风险标记": "、".join(risk),
        "主升标记": "、".join(bull),
        "信号类型": kind,
        "研判备注": note,
    }


def mark_signal_hits(df: pd.DataFrame) -> pd.DataFrame:
    """对资讯表做投资逻辑标注。"""
    if df is None or df.empty:
        return df
    out = df.copy()
    text = (out["标题"].fillna("") + " " + out["内容"].fillna("")).astype(str)
    rows = [analyze_news_item(t) for t in text]
    ann = pd.DataFrame(rows)
    for col in ("领域", "投资因子", "风险标记", "主升标记", "信号类型", "研判备注"):
        out[col] = ann[col].values
    return out


# 兼容旧调用名
def mark_risk_hits(df: pd.DataFrame) -> pd.DataFrame:
    return mark_signal_hits(df)


def filter_signal_related(df: pd.DataFrame, mode: str = "信号") -> pd.DataFrame:
    """mode: 全部 / 信号 / 风险 / 主升 / 观察"""
    if df is None or df.empty:
        return df
    if "信号类型" not in df.columns:
        df = mark_signal_hits(df)
    mode = mode or "信号"
    if mode == "全部":
        return df.reset_index(drop=True)
    if mode == "风险":
        return df[df["信号类型"].isin(["风险", "风险+主升"])].reset_index(drop=True)
    if mode == "主升":
        return df[df["信号类型"].isin(["主升", "风险+主升"])].reset_index(drop=True)
    if mode == "观察":
        return df[df["信号类型"].isin(["外围观察", "背景"])].reset_index(drop=True)
    # 信号 = 风险或主升（不含纯外围观察）
    return df[df["信号类型"].isin(["风险", "主升", "风险+主升"])].reset_index(drop=True)


def filter_risk_related(df: pd.DataFrame, only_risk: bool = True) -> pd.DataFrame:
    """兼容旧接口。"""
    return filter_signal_related(df, mode="风险" if only_risk else "全部")


def _parse_news_time(series: pd.Series) -> pd.Series:
    """解析资讯时间；失败则为 NaT。"""
    raw = series.fillna("").astype(str).str.strip()
    # 常见：2026-07-21 22:21:34 / 2026/07/21 22:21:34 / 07-21 22:21
    parsed = pd.to_datetime(raw, errors="coerce")
    # 仅有月日时，补当前年
    need = parsed.isna() & raw.ne("") & raw.ne("nan")
    if need.any():
        year = date.today().year
        patched = raw[need].map(lambda s: f"{year}-{s}" if len(s) <= 14 and s[0].isdigit() else s)
        parsed.loc[need] = pd.to_datetime(patched, errors="coerce")
    return parsed


def filter_today_news(df: pd.DataFrame, day: date | None = None) -> pd.DataFrame:
    """只保留「今天」发布的资讯；无时间戳的条目默认保留（如财新要点）。"""
    if df is None or df.empty:
        return df
    day = day or date.today()
    out = df.copy()
    if "时间" not in out.columns:
        return out.reset_index(drop=True)
    ts = _parse_news_time(out["时间"])
    out["_ts"] = ts
    keep = ts.isna() | (ts.dt.date == day)
    return out.loc[keep].drop(columns=["_ts"], errors="ignore").reset_index(drop=True)


def _dedupe_news(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    key = out["标题"].fillna("").astype(str).str.strip().str.slice(0, 40)
    out = out.assign(_k=key)
    out = out[out["_k"].ne("")].drop_duplicates(subset=["_k"], keep="first")
    return out.drop(columns=["_k"]).reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_news_bundle(
    source_keys: tuple[str, ...],
    limit_per_source: int = 500,
    today_only: bool = True,
) -> dict:
    """拉取多源资讯；默认保留「今日」全量，不再截成几十条近闻。"""
    frames: list[pd.DataFrame] = []
    status: dict[str, str] = {}
    today = date.today()
    for key in source_keys:
        meta = SOURCES.get(key)
        if not meta or key not in _FETCHERS:
            status[key] = "未知源"
            continue
        raw = _call_timeout(_FETCHERS[key], timeout=meta.timeout)
        if raw is None:
            status[key] = "超时/失败"
            continue
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            status[key] = "空数据"
            continue

        chunk = raw
        if today_only:
            before = len(chunk)
            chunk = filter_today_news(chunk, day=today)
            # 接口本身只返回近几小时时：今日过滤后仍可能=全量
            status[key] = f"OK(原始{before}/今日{len(chunk)})"
        else:
            status[key] = f"OK({len(chunk)})"

        if limit_per_source and len(chunk) > limit_per_source:
            # 保留时间更完整：按时间排序后取，而不是盲目 head 最近
            if "时间" in chunk.columns:
                ts = _parse_news_time(chunk["时间"])
                chunk = chunk.assign(_ts=ts).sort_values("_ts", ascending=False, na_position="last")
                chunk = chunk.head(limit_per_source).drop(columns=["_ts"], errors="ignore")
            else:
                chunk = chunk.head(limit_per_source)

        if not chunk.empty:
            frames.append(chunk.reset_index(drop=True))

    if not frames:
        return {
            "news": pd.DataFrame(),
            "status": status,
            "error": "全部信息源失败或今日无数据，请稍后重试/放宽时效",
            "today_only": today_only,
            "as_of_day": today.isoformat(),
        }

    news = pd.concat(frames, ignore_index=True)
    news = _dedupe_news(news)
    news = mark_signal_hits(news)

    # 今日全量：按时间倒序为主，信号强度仅作次级排序
    if "时间" in news.columns:
        news["_ts"] = _parse_news_time(news["时间"])
    else:
        news["_ts"] = pd.NaT
    rank_map = {"风险+主升": 4, "风险": 3, "主升": 3, "外围观察": 2, "背景": 1, "": 0}
    news["_sig_rank"] = news["信号类型"].map(lambda x: rank_map.get(str(x), 0)) if "信号类型" in news.columns else 0
    news = news.sort_values(
        ["_ts", "_sig_rank"], ascending=[False, False], na_position="last"
    ).drop(columns=["_ts", "_sig_rank"], errors="ignore")

    return {
        "news": news.reset_index(drop=True),
        "status": status,
        "error": None,
        "today_only": today_only,
        "as_of_day": today.isoformat(),
    }


def sources_table() -> pd.DataFrame:
    rows = []
    for s in SOURCES.values():
        rows.append(
            {
                "源": s.name,
                "权威性": s.authority,
                "更新": s.cadence,
                "及时性": s.timeliness,
                "入选理由": s.why,
                "默认开启": "是" if s.default_on else "否",
            }
        )
    return pd.DataFrame(rows)


def build_ai_digest_prompt(
    news: pd.DataFrame,
    max_items: int = 45,
    market_markdown: str | None = None,
    max_hard: int = 22,
    max_soft: int = 18,
) -> str:
    if (news is None or news.empty) and not (market_markdown and market_markdown.strip()):
        return "当前无可用资讯与大盘数据，请说明无法分析。"

    work = news
    if news is not None and not news.empty:
        try:
            from analysis.stocksight_intel import rank_for_ai, tag_news_dataframe

            if "信息层级" not in news.columns:
                work = tag_news_dataframe(news)
            work = rank_for_ai(work)
        except Exception:
            work = news

    hard_lines: list[str] = []
    soft_lines: list[str] = []
    if work is not None and not work.empty:
        # 限额抽样，避免 DashScope 读超时
        if "信息层级" in work.columns:
            hard_df = work[work["信息层级"].astype(str) == "硬信息"]
            soft_df = work[work["信息层级"].astype(str) != "硬信息"]
        else:
            hard_df = work.iloc[0:0]
            soft_df = work

        def _line(row) -> str:
            cat = row.get("StockSight类别") or row.get("信号类型") or ""
            conf = row.get("可信度") or ""
            tag = f"[{cat}/{conf}]" if cat or conf else ""
            return (
                f"- {tag} {str(row.get('时间',''))[:16]} "
                f"{str(row.get('标题',''))[:56]}｜{str(row.get('内容',''))[:72]}"
            )

        for _, row in hard_df.head(max_hard).iterrows():
            hard_lines.append(_line(row))
        for _, row in soft_df.head(max_soft).iterrows():
            soft_lines.append(_line(row))
        # 若无层级列，退回统一截断
        if not hard_lines and not soft_lines:
            for _, row in work.head(max_items).iterrows():
                soft_lines.append(_line(row))

    hard_body = "\n".join(hard_lines) if hard_lines else "（本次无硬信息条目）"
    soft_body = "\n".join(soft_lines) if soft_lines else "（本次无市场快讯条目）"
    market_block = (market_markdown or "").strip() or "（本次无大盘复盘数据）"
    if len(market_block) > 2500:
        market_block = market_block[:2500] + "\n…(复盘已截断)"

    return f"""结合大盘复盘与 StockSight 情报做精炼分析（硬信息优先；公告≠买卖信号）。

输出（简洁）：
1. 大盘温度（强势/偏暖/震荡/偏弱+一句依据）
2. 硬信息排雷/催化（按类别，最多5点）
3. 风险雷达 / 主升雷达（各最多4点，因子→证据）
4. 仓位含义（进攻/防御/观望）
5. 1-5日观察清单（3条）
6. 一句话结论 + 置信度（低/中/高）

===== 大盘复盘 =====
{market_block}

===== 硬信息 =====
{hard_body}

===== 市场快讯 =====
{soft_body}
"""


AI_SYSTEM = (
    "你是专业的 A 股组合与策略分析助手。"
    "采用 StockSight 情报纪律：硬信息（公告/财报/风险提示等）优先于市场快讯；"
    "公告是上下文与排雷，不是自动买卖信号。"
    "同时使用大盘复盘（指数/广度/板块）；先资产域、再因子、后方向；"
    "不编造未提供的数据。回答用中文，条理清晰，强调可验证观察点。"
)


