"""创业板指 / 中证红利低波 比价，以及各自的历史分位。"""

from __future__ import annotations

import requests
import akshare as ak
import pandas as pd

from analysis.indicators import calc_ratio_line
from etf_data_fetcher import bar_end_date, clip_to_last_complete

CHINEXT_CODE = "399006"
CHINEXT_NAME = "创业板指"
DIV_CODE = "930955"
DIV_NAME = "中证红利低波"
DIV_ETF = "512890"
_CSI_URL = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
_CSI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.csindex.com.cn/",
}

WINDOWS = {
    "近3年": 756,
    "近5年": 1260,
    "全部": None,
}


def _as_close(df: pd.DataFrame, date_col: str, close_col: str) -> pd.Series:
    work = df[[date_col, close_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col].astype(str).str.replace("-", "", regex=False), format="%Y%m%d")
    work[close_col] = pd.to_numeric(work[close_col], errors="coerce")
    work = work.dropna().set_index(date_col).sort_index()
    frame = work.rename(columns={close_col: "收盘"})
    frame["Close"] = frame["收盘"]
    clipped = clip_to_last_complete(frame)
    close = pd.to_numeric(clipped["收盘"], errors="coerce").dropna()
    close = close[~close.index.duplicated(keep="last")]
    close.index = pd.to_datetime(close.index).normalize()
    return close.sort_index()


def _fetch_tx_bars(symbol: str, start: str = "20160101") -> pd.DataFrame:
    end = bar_end_date().strftime("%Y%m%d")
    raw = ak.stock_zh_index_daily_tx(symbol=symbol, start_date=start, end_date=end)
    if raw is None or raw.empty:
        raise RuntimeError(f"{symbol} 无行情")
    close = _as_close(raw, "date", "close")
    vol_col = "amount" if "amount" in raw.columns else ("volume" if "volume" in raw.columns else None)
    volume = pd.Series(dtype=float)
    if vol_col:
        volume = _as_close(raw, "date", vol_col).reindex(close.index)
    return pd.DataFrame({"收盘": close, "量能": volume})


def _fetch_tx_close(symbol: str, start: str = "20160101") -> pd.Series:
    return _fetch_tx_bars(symbol, start=start)["收盘"]


def _fetch_csi_close(index_code: str, start: str = "2016-01-01") -> pd.Series:
    end = bar_end_date().strftime("%Y-%m-%d")
    resp = requests.get(
        _CSI_URL,
        params={"indexCode": index_code, "startDate": start, "endDate": end},
        headers=_CSI_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    rows = (resp.json() or {}).get("data") or []
    if not rows:
        raise RuntimeError(f"中证指数 {index_code} 无行情")
    frame = pd.DataFrame(rows)
    return _as_close(frame, "tradeDate", "close")


def _extend_with_proxy(index_close: pd.Series, proxy_close: pd.Series) -> tuple[pd.Series, str]:
    """指数停更之后，用跟踪 ETF 的涨跌把点位接到最新交易日。"""
    index_close = index_close.sort_index()
    proxy_close = proxy_close.sort_index()
    last = index_close.index[-1]
    proxy_base = proxy_close[proxy_close.index <= last]
    proxy_after = proxy_close[proxy_close.index > last]
    if proxy_base.empty or proxy_after.empty:
        return index_close, ""
    base = float(proxy_base.iloc[-1])
    if base == 0:
        return index_close, ""
    extended = float(index_close.iloc[-1]) * (proxy_after / base)
    note = f"{last.strftime('%Y-%m-%d')} 之后用红利低波ETF {DIV_ETF} 涨跌幅衔接"
    return pd.concat([index_close, extended]).sort_index(), note


def load_chinext_dividend() -> dict:
    chinext_bars = _fetch_tx_bars(f"sz{CHINEXT_CODE}")
    chinext = chinext_bars["收盘"]
    chinext_volume = chinext_bars["量能"]
    dividend = _fetch_csi_close(DIV_CODE)
    note = ""
    try:
        proxy = _fetch_tx_close(f"sh{DIV_ETF}", start="20180101")
        dividend, note = _extend_with_proxy(dividend, proxy)
    except Exception:
        note = ""
    if chinext.empty or dividend.empty:
        raise RuntimeError("创业板或红利低波行情为空")
    asof = min(chinext.index[-1], dividend.index[-1]).strftime("%Y-%m-%d")
    return {
        "chinext": chinext,
        "chinext_volume": chinext_volume,
        "dividend": dividend,
        "chinext_name": CHINEXT_NAME,
        "chinext_code": CHINEXT_CODE,
        "div_name": DIV_NAME,
        "div_code": DIV_CODE,
        "note": note,
        "asof": asof,
    }


def current_percentile(close: pd.Series, window: int | None) -> tuple[float | None, int]:
    """窗口内收盘价不高于最新收盘的占比。100 为窗口最高。"""
    series = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if window:
        series = series.tail(int(window))
    if len(series) < 60:
        return None, int(len(series))
    last = float(series.iloc[-1])
    return round(float((series <= last).mean() * 100), 1), int(len(series))


def percentile_path(close: pd.Series, window: int | None) -> pd.Series:
    """每个交易日的历史分位。有窗口时用滚动窗口，否则用截至当日的全部样本。"""
    series = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = [float("nan")] * n
    if n < 60:
        return pd.Series(dtype=float)
    if window and window < n:
        width = int(window)
        for i in range(width - 1, n):
            chunk = values[i - width + 1 : i + 1]
            out[i] = float((chunk <= chunk[-1]).mean() * 100)
    else:
        for i in range(59, n):
            chunk = values[: i + 1]
            out[i] = float((chunk <= chunk[-1]).mean() * 100)
    path = pd.Series(out, index=series.index).dropna()
    if window:
        path = path.tail(int(window))
    return path


def build_style_ratio(pair: dict, window: int | None, ma_period: int = 20) -> dict:
    ch = pair["chinext"]
    dv = pair["dividend"]
    if window:
        ch = ch.tail(int(window) + ma_period)
        dv = dv.tail(int(window) + ma_period)
    ratio = calc_ratio_line(ch, dv, ma_period=ma_period)
    if window and len(ratio) > window:
        ratio = ratio.tail(int(window))
    ch_pct, ch_n = current_percentile(pair["chinext"], window)
    dv_pct, dv_n = current_percentile(pair["dividend"], window)
    ratio_pct, ratio_n = current_percentile(ratio["比价"], window)
    return {
        "ratio": ratio,
        "chinext_pct": ch_pct,
        "div_pct": dv_pct,
        "ratio_pct": ratio_pct,
        "chinext_pct_path": percentile_path(pair["chinext"], window),
        "div_pct_path": percentile_path(pair["dividend"], window),
        "ratio_pct_path": percentile_path(ratio["比价"], window),
        "chinext_n": ch_n,
        "div_n": dv_n,
        "ratio_n": ratio_n,
        "ma_period": ma_period,
    }


def read_price_volume(close: pd.Series, volume: pd.Series | None) -> dict:
    """用价格相对均线、以及涨跌日的量能对比，区分下跌延续和反转尝试。"""
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    m20 = float(ma20.iloc[-1])
    m20_prev = float(ma20.iloc[-6]) if pd.notna(ma20.iloc[-6]) else m20
    below = last < m20
    ma_down = m20 < m20_prev
    up_day = last >= prev

    vol = pd.Series(dtype=float)
    if volume is not None:
        vol = pd.to_numeric(volume, errors="coerce").reindex(close.index).dropna()
    vol_ratio = None
    down_vs_up = None
    if len(vol) >= 20:
        chg = close.diff()
        recent_vol = vol.tail(20)
        recent_chg = chg.reindex(recent_vol.index)
        up_avg = float(recent_vol[recent_chg > 0].mean()) if (recent_chg > 0).any() else None
        down_avg = float(recent_vol[recent_chg < 0].mean()) if (recent_chg < 0).any() else None
        base = float(recent_vol.mean())
        if base:
            vol_ratio = float(vol.iloc[-1]) / base
        if up_avg and down_avg and up_avg > 0:
            down_vs_up = down_avg / up_avg

    if (not below) and (not ma_down) and up_day and vol_ratio is not None and vol_ratio >= 1:
        label = "反转尝试"
        why = "收盘在20日均线上方，均线不再向下，当天上涨且量能不低于近20日均量。"
    elif (not below) and (not ma_down):
        label = "趋势转强"
        why = "收盘已在20日均线上方，均线不再向下。量能没有明显放大时，只算转强，还不是反转确认。"
    elif below and ma_down and down_vs_up is not None and down_vs_up >= 1.05:
        label = "下跌趋势"
        why = "收盘在20日均线下方，均线仍向下，而且近20日下跌日的量能大于上涨日。"
    elif below and vol_ratio is not None and vol_ratio < 0.85 and not (down_vs_up is not None and down_vs_up >= 1.05):
        label = "缩量止跌"
        why = "价格仍在20日均线下方，但量能低于均量，下跌日也没有更放量。卖压在减轻，要等放量站上均线才算反转尝试。"
    elif below and ma_down:
        label = "下跌趋势"
        extra = ""
        if vol_ratio is not None and down_vs_up is not None:
            extra = (
                f"当日量能是20日均量的 {vol_ratio:.2f} 倍，"
                f"下跌日均量是上涨日的 {down_vs_up:.2f} 倍，还没有变成缩量止跌。"
            )
        why = "收盘在20日均线下方，20日均线仍在向下。" + extra
    else:
        label = "方向未明"
        why = "价格和量能没有同时指向下跌延续或反转。"

    chart = pd.DataFrame({"收盘": close, "MA20": ma20, "MA60": ma60}).tail(120)
    if len(vol):
        aligned = vol.reindex(close.index)
        chart["量能"] = aligned.reindex(chart.index)
        chart["量能MA20"] = aligned.rolling(20).mean().reindex(chart.index)
    return {
        "label": label,
        "why": why,
        "below_ma20": below,
        "ma_down": ma_down,
        "vol_ratio": None if vol_ratio is None else round(vol_ratio, 2),
        "down_vs_up": None if down_vs_up is None else round(down_vs_up, 2),
        "chart": chart,
    }
