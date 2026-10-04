"""近N日动量主线 ETF 筛选（比价线榜单）。"""

from __future__ import annotations

import re

from pathlib import Path

import akshare as ak
import pandas as pd
import streamlit as st

from analysis.indicators import ensure_close
from analysis.sectors import infer_sector
from etf_data_fetcher import (
    data_asof_str,
    fetch_etf_daily_cached,
    fetch_pool_daily,
    hist_prefer_sina,
    last_complete_session_date,
)
from utils import clean_etf_symbol

SECTOR_TOP_N = 30
PERIOD_OPTIONS = (5, 10, 20, 60)
POOL_SIZE = 150

_TURNOVER_CACHE = Path(__file__).resolve().parent.parent / "etf_cache" / "last_turnover.csv"
_SPOT_SNAPSHOT = Path(__file__).resolve().parent.parent / "etf_cache" / "last_spot.csv"


def _load_turnover_cache() -> pd.Series:
    if not _TURNOVER_CACHE.exists():
        return pd.Series(dtype=float)
    try:
        cached = pd.read_csv(_TURNOVER_CACHE, dtype={"代码": str})
        cached["代码"] = cached["代码"].map(clean_etf_symbol)
        cached["成交额"] = pd.to_numeric(cached["成交额"], errors="coerce")
        cached = cached.dropna(subset=["代码", "成交额"])
        return cached.set_index("代码")["成交额"]
    except Exception:
        return pd.Series(dtype=float)


def _save_turnover_cache(codes: pd.Series, amounts: pd.Series) -> None:
    try:
        _TURNOVER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({"代码": codes.map(clean_etf_symbol), "成交额": amounts})
        frame = frame.dropna()
        if frame.empty:
            return
        frame["asof"] = last_complete_session_date().strftime("%Y-%m-%d")
        frame.to_csv(_TURNOVER_CACHE, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def _save_spot_snapshot(out: pd.DataFrame) -> None:
    try:
        if out is None or out.empty:
            return
        _SPOT_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        snap = out.copy()
        snap["asof"] = last_complete_session_date().strftime("%Y-%m-%d")
        snap.to_csv(_SPOT_SNAPSHOT, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def _load_spot_snapshot() -> pd.DataFrame:
    if not _SPOT_SNAPSHOT.exists():
        return pd.DataFrame()
    try:
        snap = pd.read_csv(_SPOT_SNAPSHOT, dtype={"代码": str})
        snap["代码"] = snap["代码"].map(clean_etf_symbol)
        if "成交额" in snap.columns:
            snap["成交额"] = pd.to_numeric(snap["成交额"], errors="coerce")
        need = {"代码", "名称", "成交额", "板块"}
        if not need.issubset(set(snap.columns)):
            return pd.DataFrame()
        return snap.dropna(subset=["代码"]).drop_duplicates(subset=["代码"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _call_spot(fn, timeout: float) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        raw = fut.result(timeout=timeout)
        return raw if isinstance(raw, pd.DataFrame) else pd.DataFrame()
    except (FuturesTimeout, Exception):
        return pd.DataFrame()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _fetch_spot_eastmoney() -> pd.DataFrame:
    return ak.fund_etf_spot_em()


def _fetch_spot_sina() -> pd.DataFrame:
    raw = ak.fund_etf_category_sina(symbol="ETF基金")
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = raw.copy()
    code_col = "代码" if "代码" in out.columns else out.columns[0]
    name_col = "名称" if "名称" in out.columns else ("基金名称" if "基金名称" in out.columns else None)
    out["代码"] = out[code_col].astype(str).str.replace(r"^(sh|sz|SH|SZ)", "", regex=True).map(clean_etf_symbol)
    out["名称"] = out[name_col].astype(str) if name_col else ""
    if "成交额" in out.columns:
        out["成交额"] = pd.to_numeric(out["成交额"], errors="coerce")
    else:
        out["成交额"] = pd.NA
    return out


def _fetch_spot_ths() -> pd.DataFrame:
    raw = ak.fund_etf_spot_ths()
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = raw.copy()
    code_col = "基金代码" if "基金代码" in out.columns else ("代码" if "代码" in out.columns else None)
    name_col = (
        "基金名称"
        if "基金名称" in out.columns
        else ("基金简称" if "基金简称" in out.columns else ("名称" if "名称" in out.columns else None))
    )
    if not code_col:
        return pd.DataFrame()
    out["代码"] = out[code_col].astype(str).map(clean_etf_symbol)
    out["名称"] = out[name_col].astype(str) if name_col else ""
    out["成交额"] = pd.NA
    return out

_EXCLUDE_NAME = re.compile(
    r"货币|国债|政金|地方债|可转债|信用债|短融|同业存单|"
    r"沪深300|中证500|中证1000|上证50|创业板|科创50|"
    r"恒生|纳指|标普|道琼斯|日经|德国|法国|"
    r"黄金|原油|豆粕|白银|商品|REIT|"
    r"红利低波|价值|成长|全市场|宽基",
    re.I,
)

_PERIOD_RET_COL = {
    5: ("近1周",),
    10: ("近1周",),  # 排行表无10日字段，用近1周近似
    20: ("近1月",),
    60: ("近3月",),
}


def _exact_col(columns: list[str], name: str) -> str | None:
    if name in columns:
        return name
    for col in columns:
        if str(col) == name:
            return col
    return None


def _col_match(columns: list[str], *keywords: str) -> str | None:
    for col in columns:
        text = str(col)
        if all(kw in text for kw in keywords):
            return col
    for col in columns:
        text = str(col)
        if any(kw in text for kw in keywords):
            return col
    return None


def _relative_strength(num_ret: float, den_ret: float) -> float:
    """
    相对强度 = 分子近N日涨幅% − 分母近N日涨幅%（超额收益）。

    不用相除：分母为负时除法会把「跌得更多」排到前面。
    超额收益无论分母涨跌，正值=跑赢分母，越大越强，正向优先。
    """
    return num_ret - den_ret


def _format_amount(amount: float | None) -> str:
    """成交额单位：元人民币。显示为 X.XX亿（1亿=1e8元）。"""
    if amount is None or pd.isna(amount):
        return "—"
    val = float(amount)
    if val >= 1e8:
        return f"{val / 1e8:.2f}亿"
    if val >= 1e4:
        return f"{val / 1e4:.0f}万"
    return f"{val:.0f}"


def _daily_amount_series(df: pd.DataFrame, spot_ref: float | None = None) -> pd.Series:
    """优先 K 线「成交额」列（元）；不用成交量估算除非与现货量级一致。"""
    for col in ("成交额", "amount", "Amount"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                return s
    if "成交量" in df.columns and "收盘" in df.columns:
        vol = pd.to_numeric(df["成交量"], errors="coerce")
        close = pd.to_numeric(df["收盘"], errors="coerce")
        for mult in (100, 1):
            est = vol * mult * close
            if spot_ref is None or est.dropna().empty:
                return est
            med = float(est.dropna().median())
            if spot_ref > 0 and 0.2 <= med / spot_ref <= 5:
                return est
    if "Volume" in df.columns and "Close" in df.columns:
        return pd.to_numeric(df["Volume"], errors="coerce") * pd.to_numeric(df["Close"], errors="coerce")
    return pd.Series(dtype=float)


def _avg_daily_amount(df: pd.DataFrame, n: int, spot_ref: float | None = None) -> float | None:
    amt = _daily_amount_series(df, spot_ref=spot_ref)
    if amt.empty:
        return None
    tail = amt.dropna().tail(n)
    if tail.empty:
        return None
    return float(tail.mean())


def _n_day_return(close: pd.Series, n: int) -> float | None:
    if close is None or len(close) < n + 1:
        return None
    base = float(close.iloc[-(n + 1)])
    last = float(close.iloc[-1])
    if base == 0:
        return None
    return (last / base - 1) * 100


@st.cache_data(ttl=1800, show_spinner=False)
def _load_full_spot_asof(asof: str) -> pd.DataFrame:
    """全市场 ETF 名单（成交额用于排序）。东财失败则新浪/同花顺/本地快照。"""
    sources: list[tuple[str, callable, float]] = [
        ("eastmoney", _fetch_spot_eastmoney, 70),
        ("sina", _fetch_spot_sina, 20),
        ("ths", _fetch_spot_ths, 15),
    ]
    if hist_prefer_sina():
        sources = [
            ("sina", _fetch_spot_sina, 20),
            ("ths", _fetch_spot_ths, 15),
            ("eastmoney", _fetch_spot_eastmoney, 25),
        ]
    last_err = "现货接口均失败"
    for _name, fn, timeout in sources:
        spot = _call_spot(fn, timeout)
        try:
            out = _spot_to_pool(spot)
        except Exception as exc:
            last_err = str(exc)
            continue
        if out is not None and not out.empty:
            _save_spot_snapshot(out)
            return out

    snap = _load_spot_snapshot()
    if snap is not None and not snap.empty:
        if "板块" not in snap.columns:
            snap["板块"] = snap["名称"].map(infer_sector)
        return snap[["代码", "名称", "成交额", "板块"]].copy()

    cached_amt = _load_turnover_cache()
    if not cached_amt.empty:
        frame = cached_amt.rename("成交额").reset_index()
        if "代码" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "代码"})
        frame["名称"] = ""
        frame["板块"] = "其他"
        return frame.drop_duplicates(subset=["代码"]).reset_index(drop=True)

    raise RuntimeError(f"ETF 现货拉取失败（{last_err}），请稍后重试")


def _spot_to_pool(spot: pd.DataFrame) -> pd.DataFrame:
    if spot is None or spot.empty:
        raise RuntimeError("现货为空")

    code_col = _exact_col(list(spot.columns), "代码") or "代码"
    name_col = _exact_col(list(spot.columns), "名称") or _col_match(list(spot.columns), "简称") or _col_match(
        list(spot.columns), "名称"
    )
    amt_col = _exact_col(list(spot.columns), "成交额") or _col_match(list(spot.columns), "成交", "额")
    if code_col not in spot.columns:
        raise RuntimeError("现货缺代码字段")

    work = spot.copy()
    work[code_col] = work[code_col].astype(str).map(clean_etf_symbol)
    if amt_col and amt_col in work.columns:
        rank = pd.to_numeric(work[amt_col], errors="coerce")
    else:
        rank = pd.Series(index=work.index, dtype=float)
    live_amt = rank.where(rank > 0)
    if live_amt.notna().sum() == 0:
        cached = _load_turnover_cache()
        if not cached.empty:
            rank = work[code_col].map(cached)
        if pd.to_numeric(rank, errors="coerce").notna().sum() == 0:
            for key in ("流通市值", "总市值", "成交量"):
                fb = _exact_col(list(work.columns), key) or _col_match(list(work.columns), key)
                if fb and fb in work.columns:
                    rank = pd.to_numeric(work[fb], errors="coerce")
                    if rank.notna().sum() > 0:
                        break
    else:
        _save_turnover_cache(work[code_col], live_amt)
        rank = live_amt
    work["_rank"] = pd.to_numeric(rank, errors="coerce")
    if name_col and name_col in work.columns:
        work = work[~work[name_col].astype(str).str.contains(_EXCLUDE_NAME, na=False)]
    work = work.dropna(subset=["_rank"])
    if work.empty:
        raise RuntimeError("现货无可用成交额，且没有本地成交额缓存")

    out = pd.DataFrame(
        {
            "代码": work[code_col],
            "名称": work[name_col] if name_col in work.columns else "",
            "成交额": work["_rank"],
        }
    )
    out["板块"] = out["名称"].map(infer_sector)
    out = out.drop_duplicates(subset=["代码"]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("现货处理后为空")
    return out


def _load_full_spot() -> pd.DataFrame:
    """按当前数据截止日取现货；截止日变化时自动换缓存。"""
    return _load_full_spot_asof(data_asof_str())


_load_full_spot.clear = _load_full_spot_asof.clear  # type: ignore[method-assign]


def _fetch_spot_pool() -> pd.DataFrame:
    """候选池：全市场中成交额靠前的一批（用于算相对强度）。"""
    full = _load_full_spot()
    if full.empty:
        # 抛错避免 st.cache 把空池锁死一小时
        raise RuntimeError("全市场 ETF 现货为空，请稍后重试")
    return full.sort_values("成交额", ascending=False).head(POOL_SIZE).reset_index(drop=True)


def _merge_rank_returns(pool: pd.DataFrame, period_days: int) -> pd.DataFrame:
    try:
        rank = ak.fund_exchange_rank_em()
    except Exception:
        pool[f"近{period_days}日涨幅%"] = None
        return pool

    code_col = _col_match(list(rank.columns), "基金", "代码") or _col_match(list(rank.columns), "代码")
    if not code_col:
        pool[f"近{period_days}日涨幅%"] = None
        return pool

    ret_col = None
    for key in _PERIOD_RET_COL.get(period_days, ("近1月",)):
        ret_col = _col_match(list(rank.columns), key)
        if ret_col:
            break
    if not ret_col:
        ret_col = _col_match(list(rank.columns), "近1月")

    rank = rank.copy()
    rank[code_col] = rank[code_col].astype(str).map(clean_etf_symbol)
    rank[ret_col] = pd.to_numeric(rank[ret_col], errors="coerce")
    lookup = rank.set_index(code_col)[ret_col].to_dict()

    out = pool.copy()
    out[f"近{period_days}日涨幅%"] = out["代码"].map(lookup)
    return out


def _refine_avg_amounts(codes: list[str], period_days: int, spot_map: dict[str, float]) -> dict[str, float]:
    """对展示代码拉 K 线，用「成交额」列算近 N 日日均；异常则保留现货单日额。"""
    if not codes:
        return {}
    tail = max(period_days + 8, 30)
    daily_map = fetch_pool_daily(
        codes,
        tail_days=tail,
        max_workers=2,
        pause_sec=0.15,
        prefer_sina=None,
    )
    out: dict[str, float] = {}
    for sym in codes:
        spot_ref = spot_map.get(sym)
        df = daily_map.get(sym, pd.DataFrame())
        if df is None or df.empty:
            if spot_ref:
                out[sym] = float(spot_ref)
            continue
        avg = _avg_daily_amount(df, period_days, spot_ref=spot_ref)
        if avg is None:
            if spot_ref:
                out[sym] = float(spot_ref)
            continue
        if spot_ref and not (0.25 * spot_ref <= avg <= 4 * spot_ref):
            out[sym] = float(spot_ref)
        else:
            out[sym] = float(avg)
    return out


def _denominator_return(denominator: str, period_days: int) -> float:
    den_code = clean_etf_symbol(denominator)
    try:
        df, _ = fetch_etf_daily_cached(
            den_code,
            tail_days=max(period_days + 10, 40),
            prefer_sina=None,
        )
        close = ensure_close(df)
        ret = _n_day_return(close, period_days)
        if ret is not None:
            return ret
    except Exception:
        pass

    try:
        rank = ak.fund_exchange_rank_em()
        code_col = _col_match(list(rank.columns), "基金", "代码") or _col_match(list(rank.columns), "代码")
        ret_col = _col_match(list(rank.columns), "近1月")
        for key in _PERIOD_RET_COL.get(period_days, ("近1月",)):
            hit = _col_match(list(rank.columns), key)
            if hit:
                ret_col = hit
                break
        if code_col and ret_col:
            row = rank[rank[code_col].astype(str).map(clean_etf_symbol) == den_code]
            if not row.empty:
                return float(pd.to_numeric(row[ret_col].iloc[0], errors="coerce"))
    except Exception:
        pass
    return 0.0


@st.cache_data(ttl=3600, show_spinner=False)
def _get_momentum_top30_asof(denominator: str, period_days: int, asof: str) -> pd.DataFrame:
    if period_days not in PERIOD_OPTIONS:
        period_days = 20

    pool = _fetch_spot_pool()
    work = _merge_rank_returns(pool, period_days)
    ret_col = f"近{period_days}日涨幅%"
    work = work.dropna(subset=[ret_col])
    if work.empty:
        raise RuntimeError("榜单涨幅数据为空（排行接口失败），请稍后重试")

    den_code = clean_etf_symbol(denominator)
    den_ret = _denominator_return(den_code, period_days)
    work["分母代码"] = den_code
    work["分母近N日涨幅%"] = round(den_ret, 2)
    work["相对强度"] = work[ret_col].apply(lambda x: round(_relative_strength(float(x), den_ret), 3))

    full_spot = _merge_rank_returns(_load_full_spot(), period_days)

    sector_rows: list[dict] = []
    for sector, grp in work.groupby("板块", sort=False):
        if grp.empty:
            continue
        rs_leader = grp.sort_values("相对强度", ascending=False).iloc[0]
        sector_full = full_spot[full_spot["板块"] == sector] if not full_spot.empty else pd.DataFrame()
        if not sector_full.empty:
            turnover_leader = sector_full.sort_values("成交额", ascending=False).iloc[0]
        else:
            turnover_leader = grp.sort_values("成交额", ascending=False).iloc[0]
        t_code = str(turnover_leader["代码"])
        t_amt = float(turnover_leader["成交额"])
        if ret_col in turnover_leader.index and pd.notna(turnover_leader[ret_col]):
            t_ret = float(turnover_leader[ret_col])
        else:
            t_ret_rows = grp[grp["代码"] == t_code][ret_col]
            t_ret = float(t_ret_rows.iloc[0]) if not t_ret_rows.empty else float(rs_leader[ret_col])
        sector_rows.append(
            {
                "板块": sector,
                "_sort_rs": float(rs_leader["相对强度"]),
                "相对强度": round(float(rs_leader["相对强度"]), 3),
                "强度龙头代码": rs_leader["代码"],
                "强度龙头涨幅%": round(float(rs_leader[ret_col]), 2),
                "代码": t_code,
                "名称": turnover_leader["名称"],
                "成交龙头涨幅%": round(t_ret, 2),
                ret_col: round(float(rs_leader[ret_col]), 2),  # 兼容旧列名：与排名一致用强度龙头涨幅
                "日均成交额": t_amt,
                "分母代码": den_code,
                "分母近N日涨幅%": round(den_ret, 2),
            }
        )

    if not sector_rows:
        return pd.DataFrame()

    out = pd.DataFrame(sector_rows)
    out = out.sort_values("_sort_rs", ascending=False).head(SECTOR_TOP_N).reset_index(drop=True)
    out.insert(0, "板块排名", range(1, len(out) + 1))
    out = out.drop(columns=["_sort_rs"])

    spot_map = {str(r["代码"]): float(r["日均成交额"]) for _, r in out.iterrows()}
    refined = _refine_avg_amounts(out["代码"].tolist(), period_days, spot_map)
    for idx, row in out.iterrows():
        code = str(row["代码"])
        if code in refined:
            out.at[idx, "日均成交额"] = refined[code]
    out["日均成交额(显示)"] = out["日均成交额"].map(_format_amount)

    cols = [
        "板块排名",
        "板块",
        "代码",
        "名称",
        "成交龙头涨幅%",
        "日均成交额(显示)",
        "相对强度",
        "强度龙头代码",
        "强度龙头涨幅%",
        "分母代码",
        "分母近N日涨幅%",
    ]
    return out[[c for c in cols if c in out.columns]]


def get_momentum_top30(denominator: str = "510300", period_days: int = 20) -> pd.DataFrame:
    return _get_momentum_top30_asof(denominator, period_days, data_asof_str())


get_momentum_top30.clear = _get_momentum_top30_asof.clear  # type: ignore[method-assign]


@st.cache_data(ttl=3600, show_spinner=False)
def _get_market_sector_map_asof(asof: str) -> tuple[dict[str, list[str]], dict[str, str]]:
    """全市场各板块成交额最高的单只 ETF。"""
    full = _load_full_spot()
    if full.empty:
        raise RuntimeError("全市场 ETF 现货为空，请稍后重试")

    sector_map: dict[str, list[str]] = {}
    names: dict[str, str] = {}
    for sector, grp in full.groupby("板块", sort=False):
        top = grp.sort_values("成交额", ascending=False).iloc[0]
        code = str(top["代码"])
        sector_map[sector] = [code]
        names[code] = str(top["名称"])
    return sector_map, names


def get_market_sector_map() -> tuple[dict[str, list[str]], dict[str, str]]:
    return _get_market_sector_map_asof(data_asof_str())


get_market_sector_map.clear = _get_market_sector_map_asof.clear  # type: ignore[method-assign]


def build_ratio_sector_options(
    holdings_symbols: list[str],
    top30_df: pd.DataFrame | None = None,
    market_map: dict[str, list[str]] | None = None,
    holdings_names: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """
    比价线板块选项：
    - 已加载 TOP30：按相对强度排序的主线板块优先
    - 持仓按板块归类（无需拉全市场）
    - 可选全市场板块（成交额龙头，需主动加载）
    - 「我的持仓」放最后，作快捷入口
    """
    sector_map: dict[str, list[str]] = {}
    ordered: list[str] = []

    if top30_df is not None and not top30_df.empty:
        for _, row in top30_df.iterrows():
            sector = str(row["板块"])
            code = str(row["代码"])
            if sector not in sector_map:
                ordered.append(sector)
            sector_map[sector] = [code]

    if holdings_symbols and holdings_names:
        by_sector: dict[str, list[str]] = {}
        for sym in holdings_symbols:
            sector = infer_sector(holdings_names.get(sym, ""))
            by_sector.setdefault(sector, []).append(sym)
        for sector in sorted(by_sector):
            if sector not in sector_map:
                ordered.append(sector)
                sector_map[sector] = by_sector[sector]

    if market_map:
        extras = sorted(s for s in market_map if s not in sector_map)
        for sector in extras:
            sector_map[sector] = market_map[sector]
        ordered.extend(extras)

    if holdings_symbols:
        sector_map["我的持仓"] = list(holdings_symbols)
        ordered.append("我的持仓")

    ordered = [s for s in ordered if sector_map.get(s)]
    return ordered, sector_map
