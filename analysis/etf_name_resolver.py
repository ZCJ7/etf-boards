"""ETF 名称 ↔ 代码解析：优先映射到用户持仓（板块/主题模糊匹配）。"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from analysis.sectors import infer_sector
from data import get_etf_list
from utils import clean_etf_symbol

_ON_EXCHANGE_CODE = re.compile(r"^(1[56]\d{4}|5[18]\d{4}|58\d{4})$")
_SUFFIX_NOISE = re.compile(
    r"(ETF|LOF|基金|联接|链接|增强|指数型|指数|主题|产业|行业|分级|后收费|前收费|"
    r"发起式|A类|C类|A|C|\(QDII\)|QDII)$",
    re.I,
)
_COMPANY_PREFIX = re.compile(
    r"^(华夏|国泰|鹏华|易方达|汇添富|天弘|华宝|广发|南方|嘉实|博时|工银|招商|"
    r"富国|建信|银华|华泰柏瑞|景顺长城|中欧|兴全|平安|东财|国联安|华安|大成|"
    r"永赢|中银|农银|民生加银|前海开源|诺安|融通|万家|国投瑞银|摩根|申万菱信|"
    r"中信建投|国寿安保|新华|海富通|交银|光大|浦银|上投|长盛|泰达宏利|宝盈|"
    r"金鹰|东吴|红土创新|浙商|汇丰晋信|金元顺安|英大|创金合信|九泰|泓德|"
    r"中融|中邮|安信|财通|德邦|方正富邦|富荣|格林|国新国证|恒越|华宸未来|"
    r"华润元大|惠升|江信|金信|凯石|联博|路博迈|明亚|南华|鹏扬|平安大华|"
    r"泉果|睿远|山证|尚正|施罗德|苏新|太平|泰康|泰信|同泰|西部利得|湘财|"
    r"鑫元|兴证全球|易米|益民|圆信永丰|长城|长信|朱雀|中庚|中科沃土|"
    r"中银证券|中银基金|中银保诚)",
    re.I,
)


@dataclass
class ResolveResult:
    symbol: str | None
    reason: str = ""
    sector: str = ""


@st.cache_data(ttl=86400, show_spinner=False)
def _etf_name_code_table() -> pd.DataFrame:
    try:
        df = get_etf_list(force_refresh=False)
        if df is None or df.empty:
            return pd.DataFrame(columns=["symbol", "name"])
        sym_col = "symbol" if "symbol" in df.columns else "代码"
        name_col = "name" if "name" in df.columns else "名称"
        out = pd.DataFrame(
            {
                "symbol": df[sym_col].astype(str).map(clean_etf_symbol),
                "name": df[name_col].astype(str),
            }
        )
        return out.drop_duplicates(subset=["symbol"])
    except Exception:
        return pd.DataFrame(columns=["symbol", "name"])


def _normalize_name(name: str) -> str:
    text = re.sub(r"\s+", "", str(name or ""))
    text = re.sub(r"[（）()]", "", text)
    while True:
        new = _COMPANY_PREFIX.sub("", text)
        new = _SUFFIX_NOISE.sub("", new)
        if new == text:
            break
        text = new
    return text.lower()


def _is_on_exchange_code(code: str) -> bool:
    return bool(_ON_EXCHANGE_CODE.match(clean_etf_symbol(code)))


def _extract_on_exchange_code(raw: str) -> str | None:
    for m in re.finditer(r"[\(（](\d{6})[\)）]", raw):
        code = clean_etf_symbol(m.group(1))
        if _is_on_exchange_code(code):
            return code
    m = re.search(r"\b(1[56]\d{4}|5[18]\d{4}|58\d{4})\b", raw)
    if m:
        return clean_etf_symbol(m.group(1))
    return None


def _overlap_score(query: str, candidate: str) -> float:
    if not query or not candidate:
        return 0.0
    if query == candidate:
        return 1.0
    if query in candidate or candidate in query:
        return 0.85
    sa, sb = set(query), set(candidate)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / max(len(sa), len(sb))


def _score_candidate(
    query_raw: str,
    query_norm: str,
    query_sector: str,
    symbol: str,
    etf_name: str,
) -> float:
    cand_norm = _normalize_name(etf_name)
    cand_sector = infer_sector(etf_name)
    score = _overlap_score(query_norm, cand_norm) * 60.0
    if query_sector and query_sector == cand_sector and query_sector not in {"综合", "其他"}:
        score += 40.0
    if query_raw and query_raw in etf_name:
        score += 10.0
    # 核心主题词必须有一定重合（避免「华夏中证」误配）
    if len(query_norm) >= 4 and len(cand_norm) >= 2:
        overlap = _overlap_score(query_norm, cand_norm)
        if overlap < 0.22 and query_sector != cand_sector:
            score *= 0.35
    return score


def _pick_best_holding(
    raw: str,
    query_norm: str,
    query_sector: str,
    holdings: dict[str, str],
    *,
    require_sector: bool,
    min_score: float,
) -> tuple[str | None, float, str]:
    best_sym = None
    best_score = 0.0
    best_sector = ""
    for sym, hname in holdings.items():
        h_sector = infer_sector(hname)
        if require_sector and query_sector not in {"综合", "其他"}:
            if h_sector != query_sector:
                continue
        s = _score_candidate(raw, query_norm, query_sector, sym, hname)
        if s > best_score:
            best_score = s
            best_sym = sym
            best_sector = h_sector
    if best_sym and best_score >= min_score:
        return best_sym, best_score, best_sector
    return None, 0.0, ""


def _holdings_pool() -> dict[str, str]:
    try:
        from holdings_utils import get_holdings_pool

        return get_holdings_pool(include_watchlist=True)
    except Exception:
        return {}


def resolve_code_for_trade(
    name: str,
    explicit_code: str | None = None,
    *,
    holdings_only: bool = True,
) -> ResolveResult:
    """
    将截图中的基金名称/代码映射为场内 ETF。
    优先匹配用户持仓池（按板块 + 名称相似度），再回退全场内 ETF 列表。
    """
    raw = str(name or "").strip()
    if not raw and not explicit_code:
        return ResolveResult(None, "无名称")

    query_sector = infer_sector(raw)
    query_norm = _normalize_name(raw)
    holdings = _holdings_pool()

    code_from_name = _extract_on_exchange_code(raw)
    code_candidate = None
    if explicit_code and _is_on_exchange_code(explicit_code):
        code_candidate = clean_etf_symbol(explicit_code)
    elif code_from_name:
        code_candidate = code_from_name

    # 0) 括号内为场外基金代码时，不用该代码，改走名称/板块匹配
    if explicit_code and not _is_on_exchange_code(explicit_code):
        code_candidate = code_from_name if code_from_name else None

    # 1) 持仓池：板块必须一致 + 名称相似度达标
    if holdings and raw:
        best_sym, best_score, best_h_sector = _pick_best_holding(
            raw, query_norm, query_sector, holdings, require_sector=True, min_score=35.0
        )
        if best_sym:
            return ResolveResult(
                best_sym,
                f"持仓匹配·{best_h_sector}",
                query_sector,
            )

    # 1b) 板块明确但持仓无同板块：不再乱配
    if holdings and query_sector not in {"综合", "其他"}:
        sector_hits = [s for s, n in holdings.items() if infer_sector(n) == query_sector]
        if not sector_hits:
            return ResolveResult(None, f"未匹配持仓·{query_sector}", query_sector)

    # 2) OCR 给了场内代码：校验板块一致性，不一致则信名称
    if code_candidate and holdings:
        table = _etf_name_code_table()
        row = table[table["symbol"] == code_candidate]
        code_name = str(row.iloc[0]["name"]) if not row.empty else ""
        code_sector = infer_sector(code_name) if code_name else infer_sector(code_candidate)
        if (
            query_sector not in {"综合", "其他"}
            and code_sector not in {"综合", "其他"}
            and query_sector != code_sector
        ):
            # 板块冲突：回退到持仓匹配（上面未命中时尝试纯板块）
            for sym, hname in holdings.items():
                if infer_sector(hname) == query_sector:
                    return ResolveResult(sym, f"持仓板块·{query_sector}", query_sector)
        if code_candidate in holdings:
            return ResolveResult(code_candidate, "截图代码·持仓内", query_sector)
        if code_name:
            return ResolveResult(code_candidate, f"截图代码·{code_sector}", query_sector)

    if code_candidate:
        return ResolveResult(code_candidate, "截图场内代码", query_sector)

    # 3) 纯板块匹配持仓（仅当板块一致且唯一）
    if holdings and query_sector not in {"综合", "其他"}:
        sector_hits = [(s, n) for s, n in holdings.items() if infer_sector(n) == query_sector]
        if len(sector_hits) == 1:
            sym, hname = sector_hits[0]
            return ResolveResult(sym, f"持仓板块·{query_sector}", query_sector)
        if len(sector_hits) > 1:
            best_sym, best_score, _ = _pick_best_holding(
                raw, query_norm, query_sector, dict(sector_hits), require_sector=False, min_score=30.0
            )
            if best_sym:
                return ResolveResult(best_sym, f"持仓板块·{query_sector}", query_sector)

    # 4) 未命中持仓：默认不强行匹配其他场内 ETF（避免标注到错误标的）
    if holdings_only:
        sector_hint = query_sector if query_sector not in {"综合", "其他"} else "主题不一致"
        return ResolveResult(None, f"未匹配持仓·{sector_hint}", query_sector)

    # 5) 可选：全场内 ETF 表兜底
    table = _etf_name_code_table()
    if table.empty or not query_norm:
        return ResolveResult(None, "未匹配", query_sector)

    best_sym = None
    best_score = 0.0
    for _, row in table.iterrows():
        sym = str(row["symbol"])
        etf_name = str(row["name"])
        s = _score_candidate(raw, query_norm, query_sector, sym, etf_name)
        if s > best_score:
            best_score = s
            best_sym = sym

    if best_sym and best_score >= 35.0:
        row = table[table["symbol"] == best_sym]
        etf_name = str(row.iloc[0]["name"]) if not row.empty else best_sym
        return ResolveResult(best_sym, f"全局匹配·{infer_sector(etf_name)}", query_sector)

    return ResolveResult(None, "未匹配", query_sector)


def resolve_code_by_name(name: str) -> str | None:
    """兼容旧接口。"""
    return resolve_code_for_trade(name).symbol


def lookup_name_by_code(symbol: str) -> str | None:
    code = clean_etf_symbol(symbol)
    if not code:
        return None
    holdings = _holdings_pool()
    if code in holdings:
        return holdings[code]
    table = _etf_name_code_table()
    row = table[table["symbol"] == code]
    if not row.empty:
        return str(row.iloc[0]["name"])
    return None
