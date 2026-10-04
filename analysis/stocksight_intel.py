"""StockSight 风格情报标注：硬信息优先、来源分级、类别标签。

移植自 GearVoid/StockSight-Skill 的 news/hard_info 思路，
用于市场资讯 → AI 提炼链路（不依赖其完整运行时）。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from datetime import date

import akshare as ak
import pandas as pd
import streamlit as st

from akshare_patch import install_akshare_patch

install_akshare_patch()

# 类别关键词（对齐 StockSight CATEGORY_KEYWORDS）
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "风险提示": ("风险提示", "退市", "监管函", "问询函", "处罚", "停牌", "复牌", "警示", "异常波动"),
    "业绩预告": ("业绩预告", "业绩快报", "盈利预告", "预亏", "预增", "预减"),
    "财报": ("年度报告", "半年报", "季报", "一季报", "三季报", "年报", "财报", "净利润"),
    "重大事项": ("重大事项", "资产重组", "收购", "出售资产", "对外投资", "诉讼", "公司大事"),
    "持股变动": ("增持", "减持", "持股变动", "股东权益变动", "股权质押"),
    "互动问答": ("互动易", "投资者问答", "投资者关系"),
    "公告": ("公告", "披露", "董事会决议"),
}

HARD_INFO_CATEGORIES = frozenset(CATEGORY_KEYWORDS.keys())

# 来源可信度：关键词 → (展示名, 分)
SOURCE_CONFIDENCE: dict[str, tuple[str, int]] = {
    "交易所": ("交易所", 5),
    "上交所": ("交易所", 5),
    "深交所": ("交易所", 5),
    "北交所": ("交易所", 5),
    "巨潮": ("巨潮资讯", 5),
    "cninfo": ("巨潮资讯", 5),
    "东方财富公告": ("东方财富公告", 4),
    "公司大事": ("东方财富大事", 4),
    "新闻联播": ("官方日更", 5),
    "财联社": ("财联社", 4),
    "财新": ("严肃财经", 4),
    "东方财富": ("东方财富快讯", 3),
    "同花顺": ("同花顺快讯", 3),
    "新浪": ("财经媒体", 2),
    "富途": ("财经媒体", 2),
}

# 公告类型白名单（东财 notice 过载时只保留高价值）
_NOTICE_TYPE_KEEP = (
    "风险",
    "业绩",
    "预告",
    "快报",
    "年报",
    "半年",
    "季报",
    "停牌",
    "复牌",
    "问询",
    "处罚",
    "减持",
    "增持",
    "异常波动",
    "重大",
    "重组",
    "股权",
)


def _call_timeout(fn, timeout: float = 20):
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout)
    except (FutTimeout, Exception):
        return None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def classify_category(text: str) -> str:
    t = (text or "").lower()
    # 中文关键词不需要 lower，但混合 URL 时保留原串再查
    raw = text or ""
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw.lower() in t or kw in raw for kw in keywords):
            return category
    return "新闻"


def classify_source(source: str, url: str = "") -> tuple[str, int]:
    hay = f"{source} {url}".lower()
    raw = f"{source} {url}"
    for keyword, (label, score) in SOURCE_CONFIDENCE.items():
        if keyword.lower() in hay or keyword in raw:
            return label, score
    return (source or "未知来源"), 1


def is_hard_info(category: str) -> bool:
    return category in HARD_INFO_CATEGORIES


def confidence_label(score: int) -> str:
    if score >= 8:
        return "高"
    if score >= 5:
        return "中"
    return "低"


def tag_news_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """为资讯表打上 StockSight 风格标签列。"""
    if df is None or df.empty:
        return df
    out = df.copy()
    cats: list[str] = []
    tiers: list[str] = []
    src_labels: list[str] = []
    scores: list[int] = []
    confs: list[str] = []

    for _, row in out.iterrows():
        text = f"{row.get('标题', '')} {row.get('内容', '')} {row.get('来源', '')}"
        cat = classify_category(text)
        src_label, src_score = classify_source(str(row.get("来源", "")), str(row.get("链接", "")))
        score = src_score + (3 if is_hard_info(cat) else 0)
        cats.append(cat)
        tiers.append("硬信息" if is_hard_info(cat) else "市场资讯")
        src_labels.append(src_label)
        scores.append(score)
        confs.append(confidence_label(score))

    out["信息层级"] = tiers
    out["StockSight类别"] = cats
    out["来源分级"] = src_labels
    out["可信度"] = confs
    out["_ss_score"] = scores
    return out


def rank_for_ai(df: pd.DataFrame) -> pd.DataFrame:
    """硬信息优先，再按可信度与时间。"""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "信息层级" not in out.columns:
        out = tag_news_dataframe(out)
    out["_hard"] = (out["信息层级"] == "硬信息").astype(int)
    if "时间" in out.columns:
        from analysis.market_news import _parse_news_time

        out["_ts"] = _parse_news_time(out["时间"])
    else:
        out["_ts"] = pd.NaT
    out = out.sort_values(
        ["_hard", "_ss_score", "_ts"],
        ascending=[False, False, False],
        na_position="last",
    )
    return out.drop(columns=["_hard", "_ts", "_ss_score"], errors="ignore").reset_index(drop=True)


def _norm_hard_row(
    *,
    source: str,
    title: str,
    content: str,
    when: str,
    url: str = "",
    category_hint: str = "",
) -> dict:
    text = f"{category_hint} {title} {content}"
    cat = classify_category(text) if not category_hint else (
        category_hint if category_hint in HARD_INFO_CATEGORIES else classify_category(text)
    )
    if category_hint and category_hint in HARD_INFO_CATEGORIES:
        cat = category_hint
    src_label, src_score = classify_source(source, url)
    score = src_score + 3
    return {
        "来源": source,
        "来源键": "hard",
        "标题": str(title).strip()[:120],
        "内容": str(content).strip()[:400],
        "时间": when,
        "链接": url or "",
        "领域": "A股交易",
        "投资因子": "政策基本面" if cat in {"公告", "财报", "业绩预告", "重大事项"} else "",
        "风险标记": "、".join(
            [w for w in ("风险提示", "问询", "处罚", "停牌", "减持") if w in f"{title}{content}"]
        )[:40],
        "主升标记": "",
        "信号类型": "风险" if cat == "风险提示" else ("背景" if cat != "新闻" else ""),
        "研判备注": "StockSight硬信息",
        "信息层级": "硬信息",
        "StockSight类别": cat if cat != "新闻" else (category_hint or "公告"),
        "来源分级": src_label,
        "可信度": confidence_label(score),
        "_ss_score": score,
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_hard_info_today(max_notices: int = 60) -> dict:
    """拉取今日公司大事 + 高价值公告（硬信息层）。"""
    today = date.today()
    day_s = today.strftime("%Y%m%d")
    rows: list[dict] = []
    status: dict[str, str] = {}

    # 1) 东方财富·公司大事
    gsdt = _call_timeout(lambda: ak.stock_gsrl_gsdt_em(date=day_s), timeout=20)
    if isinstance(gsdt, pd.DataFrame) and not gsdt.empty:
        code_c = next((c for c in gsdt.columns if "代码" in str(c)), None)
        name_c = next((c for c in gsdt.columns if str(c) in ("简称", "名称") or "简称" in str(c)), None)
        evt_c = next((c for c in gsdt.columns if "事件" in str(c) or "类型" in str(c)), None)
        cont_c = next((c for c in gsdt.columns if "具体" in str(c) or "内容" in str(c)), None)
        date_c = next((c for c in gsdt.columns if "日期" in str(c)), None)
        for _, r in gsdt.iterrows():
            title = f"{r.get(name_c, '')}({r.get(code_c, '')}) {r.get(evt_c, '')}".strip()
            content = str(r.get(cont_c, "") or "")
            when = str(r.get(date_c, today.isoformat()))
            if hasattr(r.get(date_c), "isoformat"):
                when = r.get(date_c).isoformat()
            rows.append(
                _norm_hard_row(
                    source="东方财富·公司大事",
                    title=title,
                    content=content,
                    when=f"{when} 15:00:00" if len(when) <= 10 else when,
                    category_hint="重大事项",
                )
            )
        status["公司大事"] = f"OK({len(gsdt)})"
    else:
        status["公司大事"] = "失败/空"

    # 2) 东财公告（过滤高价值类型）
    notice = _call_timeout(lambda: ak.stock_notice_report(symbol="全部", date=day_s), timeout=45)
    if isinstance(notice, pd.DataFrame) and not notice.empty:
        type_c = next((c for c in notice.columns if "类型" in str(c)), None)
        title_c = next((c for c in notice.columns if "标题" in str(c)), None)
        name_c = next((c for c in notice.columns if str(c) in ("名称", "简称") or "名称" in str(c)), None)
        code_c = next((c for c in notice.columns if "代码" in str(c)), None)
        url_c = next((c for c in notice.columns if "网址" in str(c) or "地址" in str(c) or "url" in str(c).lower()), None)
        date_c = next((c for c in notice.columns if "日期" in str(c)), None)

        kept = notice
        if type_c:
            mask = kept[type_c].astype(str).apply(lambda x: any(k in x for k in _NOTICE_TYPE_KEEP))
            kept = kept.loc[mask]
        kept = kept.head(max_notices)
        for _, r in kept.iterrows():
            ntype = str(r.get(type_c, "公告") if type_c else "公告")
            title = str(r.get(title_c, "") if title_c else "")
            if name_c:
                title = f"{r.get(name_c, '')}({r.get(code_c, '')}): {title}"
            when = str(r.get(date_c, today.isoformat()) if date_c else today.isoformat())
            if hasattr(r.get(date_c) if date_c else None, "isoformat"):
                when = r.get(date_c).isoformat()
            cat_hint = classify_category(f"{ntype} {title}")
            if cat_hint == "新闻":
                cat_hint = "公告"
            rows.append(
                _norm_hard_row(
                    source="东方财富公告",
                    title=title,
                    content=ntype,
                    when=f"{when} 16:00:00" if len(str(when)) <= 10 else str(when),
                    url=str(r.get(url_c, "") if url_c else ""),
                    category_hint=cat_hint,
                )
            )
        status["公告"] = f"OK(筛后{len(kept)}/原始{len(notice)})"
    else:
        status["公告"] = "失败/空"

    if not rows:
        return {
            "news": pd.DataFrame(),
            "status": status,
            "error": "今日硬信息为空",
            "as_of_day": today.isoformat(),
        }

    df = pd.DataFrame(rows)
    # 去重
    df["_k"] = df["标题"].astype(str).str.slice(0, 40)
    df = df.drop_duplicates("_k").drop(columns=["_k"]).reset_index(drop=True)
    return {
        "news": df,
        "status": status,
        "error": None,
        "as_of_day": today.isoformat(),
    }


def merge_hard_and_market(market_news: pd.DataFrame, hard_news: pd.DataFrame) -> pd.DataFrame:
    """合并硬信息与市场快讯，并统一 StockSight 标注与排序。"""
    frames: list[pd.DataFrame] = []
    if isinstance(hard_news, pd.DataFrame) and not hard_news.empty:
        frames.append(hard_news)
    if isinstance(market_news, pd.DataFrame) and not market_news.empty:
        tagged = tag_news_dataframe(market_news)
        frames.append(tagged)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    # 标题去重：硬信息优先保留
    out["_hard"] = (out.get("信息层级", pd.Series([""] * len(out))) == "硬信息").astype(int)
    out["_k"] = out["标题"].fillna("").astype(str).str.strip().str.slice(0, 40)
    out = out.sort_values("_hard", ascending=False).drop_duplicates("_k", keep="first")
    out = out.drop(columns=["_hard", "_k"], errors="ignore")
    return rank_for_ai(out)
