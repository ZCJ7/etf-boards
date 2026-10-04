"""周线反转 TOP30：按板块去重，每板块 1 只，按近20日成交量排板块。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis.indicators import calc_macd, calc_ma, ensure_close
from analysis.screener import POOL_SIZE, SECTOR_TOP_N, _fetch_spot_pool
from etf_data_fetcher import data_asof_str, fetch_pool_daily
from utils import clean_etf_symbol

VOLUME_RANK_DAYS = 20
WEEKLY_LOOKBACK_DAYS = 280  # 约 56 周，覆盖 MA20 + MACD(26)


def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """日线合成周线（周五收盘为周）。"""
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index)
    work = work.sort_index()

    close = ensure_close(work)
    vol_col = "成交量" if "成交量" in work.columns else ("Volume" if "Volume" in work.columns else None)
    vol = pd.to_numeric(work[vol_col], errors="coerce") if vol_col else pd.Series(0.0, index=work.index)

    frame = pd.DataFrame({"Close": close, "Volume": vol}, index=work.index)
    weekly = frame.resample("W-FRI").agg({"Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
    return weekly


def _weekly_above_ma20(weekly_close: pd.Series) -> bool:
    """周线收盘价在 MA20 上方。"""
    if weekly_close is None or len(weekly_close) < 21:
        return False
    ma20 = calc_ma(weekly_close, [20])["MA20"]
    if pd.isna(ma20.iloc[-1]):
        return False
    return float(weekly_close.iloc[-1]) > float(ma20.iloc[-1])


def _weekly_macd_reversal(weekly_close: pd.Series) -> tuple[bool, str]:
    """
    周线 MACD 反转/转强条件（满足其一即可）：
    - 金叉：DIF 上穿 DEA
    - 绿柱缩短：MACD 柱仍为负，但绝对值缩小（空头动能减弱）
    - 红柱变长：MACD 柱为正且绝对值变大（多头动能增强）
    """
    if weekly_close is None or len(weekly_close) < 35:
        return False, ""
    macd = calc_macd(weekly_close, 12, 26, 9)
    if len(macd) < 2:
        return False, ""

    dif0, dif1 = float(macd["DIF"].iloc[-2]), float(macd["DIF"].iloc[-1])
    dea0, dea1 = float(macd["DEA"].iloc[-2]), float(macd["DEA"].iloc[-1])
    h0, h1 = float(macd["MACD"].iloc[-2]), float(macd["MACD"].iloc[-1])

    golden = dif0 <= dea0 and dif1 > dea1
    green_shorten = h1 < 0 and h0 < 0 and h1 > h0  # -0.5 → -0.3
    red_lengthen = h1 > 0 and h1 > h0  # 0.3 → 0.5（含由绿转红后继续拉长）

    tags: list[str] = []
    if golden:
        tags.append("金叉")
    if green_shorten:
        tags.append("绿柱缩短")
    if red_lengthen:
        tags.append("红柱变长")
    if not tags:
        return False, ""
    return True, "+".join(tags)


def _avg_volume_20d(df: pd.DataFrame, n: int = VOLUME_RANK_DAYS) -> float | None:
    if df is None or df.empty:
        return None
    vol_col = "成交量" if "成交量" in df.columns else ("Volume" if "Volume" in df.columns else None)
    if not vol_col:
        return None
    vol = pd.to_numeric(df[vol_col], errors="coerce").dropna().tail(n)
    if vol.empty:
        return None
    return float(vol.mean())


def _format_volume(vol: float | None) -> str:
    if vol is None or pd.isna(vol):
        return "—"
    val = float(vol)
    if val >= 1e8:
        return f"{val / 1e8:.2f}亿手"
    if val >= 1e4:
        return f"{val / 1e4:.0f}万手"
    return f"{val:.0f}"


def _scan_one(symbol: str, df: pd.DataFrame, name: str, sector: str) -> dict | None:
    if df is None or df.empty or len(df) < 80:
        return None
    try:
        weekly = _to_weekly(df)
        if weekly.empty or len(weekly) < 30:
            return None
        w_close = weekly["Close"]
        if not _weekly_above_ma20(w_close):
            return None
        ok_macd, macd_tag = _weekly_macd_reversal(w_close)
        if not ok_macd:
            return None
        avg_vol = _avg_volume_20d(df)
        if avg_vol is None:
            return None
        last_close = float(ensure_close(df).iloc[-1])
        return {
            "代码": clean_etf_symbol(symbol),
            "名称": name,
            "板块": sector,
            "收盘价": round(last_close, 3),
            "近20日均量": avg_vol,
            "近20日均量(显示)": _format_volume(avg_vol),
            "MACD信号": macd_tag,
            "周线条件": "收盘>MA20",
        }
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def _get_reversal_top30_asof(pool_size: int, asof: str) -> pd.DataFrame:
    """
    TOP30 反转榜（按板块，每板块仅 1 只）：
    1) 周线收盘价在 MA20 上方
    2) 周线 MACD 绿柱缩短 / 红柱变长 / 金叉（满足其一）
    3) 板块内取近 20 日均量最高的一只作为代表
    4) 按该代表均量对板块降序，取前 30 个板块
    """
    pool = _fetch_spot_pool()

    candidates = pool.head(pool_size)
    codes = [clean_etf_symbol(c) for c in candidates["代码"].tolist()]
    name_map = dict(zip(candidates["代码"].astype(str).map(clean_etf_symbol), candidates["名称"]))
    sector_map = dict(zip(candidates["代码"].astype(str).map(clean_etf_symbol), candidates["板块"]))

    daily_map = fetch_pool_daily(
        codes,
        tail_days=WEEKLY_LOOKBACK_DAYS,
        max_workers=5,
        pause_sec=0.02,
        prefer_sina=None,
    )
    # 日线拉取过少时视为失败（勿缓存空榜）
    got = sum(1 for c in codes if c in daily_map and daily_map[c] is not None and not daily_map[c].empty)
    if got < max(15, len(codes) // 6):
        raise RuntimeError(f"日线行情不足（成功 {got}/{len(codes)}），请稍后重试")

    rows: list[dict] = []
    n_above = n_macd = n_vol = 0
    for code in codes:
        df = daily_map.get(code, pd.DataFrame())
        if df is None or df.empty or len(df) < 80:
            continue
        try:
            weekly = _to_weekly(df)
            if weekly.empty or len(weekly) < 30:
                continue
            w_close = weekly["Close"]
            if not _weekly_above_ma20(w_close):
                continue
            n_above += 1
            ok_macd, macd_tag = _weekly_macd_reversal(w_close)
            if not ok_macd:
                continue
            n_macd += 1
            avg_vol = _avg_volume_20d(df)
            if avg_vol is None:
                continue
            n_vol += 1
            last_close = float(ensure_close(df).iloc[-1])
            rows.append(
                {
                    "代码": clean_etf_symbol(code),
                    "名称": name_map.get(code, ""),
                    "板块": sector_map.get(code, ""),
                    "收盘价": round(last_close, 3),
                    "近20日均量": avg_vol,
                    "近20日均量(显示)": _format_volume(avg_vol),
                    "MACD信号": macd_tag,
                    "周线条件": "收盘>MA20",
                }
            )
        except Exception:
            continue

    if not rows:
        raise RuntimeError(
            f"未扫到反转候选（候选{len(codes)} · 站上MA20:{n_above} · MACD转强:{n_macd} · 有均量:{n_vol}）。"
            "若市场普跌，符合条件的标的会很少；可稍后重试。"
        )

    hits = pd.DataFrame(rows)
    # 每板块只保留近20日均量最高的一只
    sector_rows: list[dict] = []
    for sector, grp in hits.groupby("板块", sort=False):
        if grp.empty:
            continue
        leader = grp.sort_values("近20日均量", ascending=False).iloc[0]
        sector_rows.append(leader.to_dict())

    if not sector_rows:
        raise RuntimeError("反转候选按板块汇总后为空，请稍后重试")

    out = pd.DataFrame(sector_rows)
    out = out.sort_values("近20日均量", ascending=False).head(SECTOR_TOP_N).reset_index(drop=True)
    out.insert(0, "板块排名", range(1, len(out) + 1))

    cols = [
        "板块排名",
        "板块",
        "代码",
        "名称",
        "收盘价",
        "近20日均量(显示)",
        "周线条件",
        "MACD信号",
    ]
    return out[[c for c in cols if c in out.columns]]


def get_reversal_top30(pool_size: int = POOL_SIZE) -> pd.DataFrame:
    return _get_reversal_top30_asof(pool_size, data_asof_str())


get_reversal_top30.clear = _get_reversal_top30_asof.clear  # type: ignore[method-assign]
