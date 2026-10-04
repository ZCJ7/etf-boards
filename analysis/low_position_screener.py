"""低位 TOP30：周线深度偏离 + 日线 MACD 转强（不去重板块）。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis.indicators import calc_macd, calc_ma, ensure_close
from analysis.reversal_screener import (
    _avg_volume_20d,
    _format_volume,
    _to_weekly,
)
from analysis.screener import (
    EXCESS_60_COL,
    POOL_SIZE,
    SECTOR_TOP_N,
    _fetch_spot_pool,
    excess_vs_hs300_60,
    load_hs300_close,
)
from etf_data_fetcher import data_asof_str, fetch_pool_daily
from utils import clean_etf_symbol

MA20_DIST_PCT = 5.0  # 距周线 MA20 至少低 5%
# 周线 MA60 + 斜率对比约需 62 周；按交易日预留缓冲
LOW_LOOKBACK_DAYS = 450
WEEKLY_MIN_BARS = 62


def _weekly_deep_below_ma20(weekly_close: pd.Series) -> tuple[bool, float | None]:
    """周线收盘在 MA20 下方，且 (收盘-MA20)/MA20*100 <= -5。"""
    if weekly_close is None or len(weekly_close) < 21:
        return False, None
    ma20 = calc_ma(weekly_close, [20])["MA20"]
    c = float(weekly_close.iloc[-1])
    m = float(ma20.iloc[-1]) if pd.notna(ma20.iloc[-1]) else None
    if m is None or m == 0:
        return False, None
    dist = (c - m) / m * 100
    if c >= m:
        return False, dist
    if dist > -MA20_DIST_PCT:  # 距离不足 5%（如 -3%）
        return False, dist
    return True, dist


def _weekly_ma60_not_down(weekly_close: pd.Series) -> tuple[bool, float | None]:
    """周线 MA60 不能向下：本周 MA60 >= 上周 MA60（走平或向上）。"""
    if weekly_close is None or len(weekly_close) < WEEKLY_MIN_BARS:
        return False, None
    ma60 = calc_ma(weekly_close, [60])["MA60"]
    m0 = float(ma60.iloc[-2]) if pd.notna(ma60.iloc[-2]) else None
    m1 = float(ma60.iloc[-1]) if pd.notna(ma60.iloc[-1]) else None
    if m0 is None or m1 is None or m0 == 0:
        return False, None
    slope_pct = (m1 - m0) / m0 * 100
    return m1 >= m0, slope_pct


def _daily_macd_turn(daily_close: pd.Series) -> tuple[bool, str]:
    """
    日线 MACD（满足其一）：
    - 绿柱缩短：柱仍为负，但绝对值缩小
    - 红柱变长：柱为正且比前一日更长
    - 金叉：DIF 上穿 DEA
    """
    if daily_close is None or len(daily_close) < 40:
        return False, ""
    macd = calc_macd(daily_close, 12, 26, 9)
    if len(macd) < 2:
        return False, ""

    dif0, dif1 = float(macd["DIF"].iloc[-2]), float(macd["DIF"].iloc[-1])
    dea0, dea1 = float(macd["DEA"].iloc[-2]), float(macd["DEA"].iloc[-1])
    h0, h1 = float(macd["MACD"].iloc[-2]), float(macd["MACD"].iloc[-1])

    golden = dif0 <= dea0 and dif1 > dea1
    green_shorten = h1 < 0 and h0 < 0 and h1 > h0
    red_lengthen = h1 > 0 and h1 > h0

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


@st.cache_data(ttl=1800, show_spinner=False)
def _get_low_position_top30_asof(pool_size: int, asof: str) -> pd.DataFrame:
    """
    低位 TOP30（不做板块去重）：
    1) 周线收盘 < MA20，且距 MA20 超过 5%
    2) 周线 MA60 不能向下（本周 MA60 >= 上周，走平或向上）
    3) 日线 MACD：绿柱缩短 / 红柱变长 / 金叉（满足其一）
    4) 按近 20 日均量降序取前 30
    """
    pool = _fetch_spot_pool()
    candidates = pool.head(pool_size)
    codes = [clean_etf_symbol(c) for c in candidates["代码"].tolist()]
    name_map = dict(zip(candidates["代码"].astype(str).map(clean_etf_symbol), candidates["名称"]))
    sector_map = dict(zip(candidates["代码"].astype(str).map(clean_etf_symbol), candidates["板块"]))

    daily_map = fetch_pool_daily(
        codes,
        tail_days=LOW_LOOKBACK_DAYS,
        max_workers=5,
        pause_sec=0.02,
        prefer_sina=None,
    )
    got = sum(1 for c in codes if c in daily_map and daily_map[c] is not None and not daily_map[c].empty)
    if got < max(15, len(codes) // 6):
        raise RuntimeError(f"日线行情不足（成功 {got}/{len(codes)}），请稍后重试")

    try:
        hs300_close = load_hs300_close()
    except Exception:
        hs300_close = None

    rows: list[dict] = []
    n_ma = n_slope = n_macd = 0
    for code in codes:
        df = daily_map.get(code, pd.DataFrame())
        if df is None or df.empty or len(df) < 80:
            continue
        try:
            weekly = _to_weekly(df)
            if weekly.empty or len(weekly) < WEEKLY_MIN_BARS:
                continue
            w_close = weekly["Close"]
            ok_ma, dist = _weekly_deep_below_ma20(w_close)
            if not ok_ma:
                continue
            n_ma += 1
            ok_slope, slope_pct = _weekly_ma60_not_down(w_close)
            if not ok_slope:
                continue
            n_slope += 1
            d_close = ensure_close(df)
            ok_macd, macd_tag = _daily_macd_turn(d_close)
            if not ok_macd:
                continue
            n_macd += 1
            avg_vol = _avg_volume_20d(df)
            if avg_vol is None:
                continue
            last_close = float(d_close.iloc[-1])
            slope_label = "走平" if slope_pct is not None and abs(slope_pct) < 1e-6 else "向上"
            rows.append(
                {
                    "代码": clean_etf_symbol(code),
                    "名称": name_map.get(code, ""),
                    "板块": sector_map.get(code, ""),
                    "收盘价": round(last_close, 3),
                    EXCESS_60_COL: excess_vs_hs300_60(d_close, hs300_close),
                    "离周MA20%": round(float(dist), 2) if dist is not None else None,
                    "周MA60斜率%": round(float(slope_pct), 3) if slope_pct is not None else None,
                    "周MA60方向": slope_label,
                    "近20日均量": avg_vol,
                    "近20日均量(显示)": _format_volume(avg_vol),
                    "日MACD信号": macd_tag,
                    "周线条件": f"收盘<MA20且偏离≥{MA20_DIST_PCT:.0f}%",
                }
            )
        except Exception:
            continue

    if not rows:
        raise RuntimeError(
            f"未扫到低位候选（候选{len(codes)} · 深跌MA20:{n_ma} · MA60非向下:{n_slope} · 日MACD转强:{n_macd}）。"
            "当前若普跌且绿柱仍在拉长，符合条件的标的会很少；可稍后重试。"
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("近20日均量", ascending=False).head(SECTOR_TOP_N).reset_index(drop=True)
    out.insert(0, "排名", range(1, len(out) + 1))

    cols = [
        "排名",
        "板块",
        "代码",
        "名称",
        "收盘价",
        EXCESS_60_COL,
        "离周MA20%",
        "周MA60方向",
        "周MA60斜率%",
        "近20日均量(显示)",
        "周线条件",
        "日MACD信号",
    ]
    return out[[c for c in cols if c in out.columns]]


def get_low_position_top30(pool_size: int = POOL_SIZE) -> pd.DataFrame:
    return _get_low_position_top30_asof(pool_size, data_asof_str())


get_low_position_top30.clear = _get_low_position_top30_asof.clear  # type: ignore[method-assign]
