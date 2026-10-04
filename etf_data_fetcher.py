"""ETF 日线数据服务：优先 AkShare/东财，失败时切换新浪。"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import akshare as ak
import pandas as pd
import requests
import streamlit as st

from akshare_patch import EASTMONEY_HEADERS, install_akshare_patch
from utils import clean_etf_symbol

install_akshare_patch()

# 新浪接口内部用 py_mini_racer(V8)；多线程同时 MiniRacer() 会 FATAL 崩进程
# （partition_address_space Check failed: !IsConfigurablePoolInitialized）
_SINA_LOCK = threading.Lock()

_UT_TOKENS = (
    "fa5fd1943c7b386f172d6893dbfba10b",
    "7eea3edcaed734bea9cbfc24409ed989",
    "bd1d9ddb04089700cf9c27f6f7426281",
)


def last_complete_session_date() -> pd.Timestamp:
    """上一根已收盘日K：15:05 前用昨天（周末往前顺延）。"""
    now = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    cutoff = now.normalize()
    if now.hour < 15 or (now.hour == 15 and now.minute < 5):
        cutoff -= pd.Timedelta(days=1)
    while int(cutoff.weekday()) >= 5:
        cutoff -= pd.Timedelta(days=1)
    return cutoff


def use_live_bar() -> bool:
    """更新榜单时含当天未收盘日K。本地网页默认关闭。"""
    return os.environ.get("ETF_USE_LIVE_BAR", "").strip().lower() in ("1", "true", "yes")


def bar_end_date() -> pd.Timestamp:
    """K 线允许保留到的日期。live 模式含当天（周末顺延到周五）。"""
    if not use_live_bar():
        return last_complete_session_date()
    now = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    cutoff = now.normalize()
    while int(cutoff.weekday()) >= 5:
        cutoff -= pd.Timedelta(days=1)
    return cutoff


def data_asof_str() -> str:
    return bar_end_date().strftime("%Y-%m-%d")


def live_bar_note() -> str:
    if not use_live_bar():
        return ""
    now = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    if int(now.weekday()) >= 5:
        return "休市，用最近交易日收盘"
    if now.hour < 15 or (now.hour == 15 and now.minute < 5):
        return "含当日未收盘价"
    return "当日已收盘"


def clip_to_last_complete(df: pd.DataFrame) -> pd.DataFrame:
    """默认去掉未收盘的当日K。ETF_USE_LIVE_BAR=1 时保留当天已走出的价格。"""
    if df is None or df.empty:
        return df
    cutoff = bar_end_date()
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Shanghai").tz_localize(None)
    out.index = pd.DatetimeIndex(idx).normalize()
    clipped = out.loc[out.index <= cutoff]
    return clipped if not clipped.empty else out



def _market_id(symbol: str) -> int:
    if symbol.startswith(("5", "6")):
        return 1
    return 0


def sina_symbol_candidates(symbol: str) -> list[str]:
    code = clean_etf_symbol(symbol)
    if code.startswith(("15", "16")):
        return [f"sz{code}", f"sh{code}"]
    return [f"sh{code}", f"sz{code}"]


def _normalize_hist_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    rename_map = {
        "date": "日期",
        "open": "开盘",
        "high": "最高",
        "low": "最低",
        "close": "收盘",
        "volume": "成交量",
        "Open": "开盘",
        "High": "最高",
        "Low": "最低",
        "Close": "收盘",
        "Volume": "成交量",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    if "日期" in out.columns:
        out["日期"] = pd.to_datetime(out["日期"])
        out = out.set_index("日期")
    elif not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)

    out = out.sort_index()
    out.index.name = "日期"

    for zh, en in [("收盘", "Close"), ("开盘", "Open"), ("最高", "High"), ("最低", "Low"), ("成交量", "Volume")]:
        if zh in out.columns:
            out[en] = pd.to_numeric(out[zh], errors="coerce")

    if "成交额" not in out.columns and "amount" in out.columns:
        out["成交额"] = pd.to_numeric(out["amount"], errors="coerce")

    return out


_PRICE_COLS = ("收盘", "开盘", "最高", "最低", "Close", "Open", "High", "Low")


def _detect_split_factor(prev: float, curr: float, min_drop: float = 0.28) -> float | None:
    """识别 ETF 份额拆分导致的价格台阶，返回前复权乘数（<1）。"""
    if prev <= 0 or curr <= 0:
        return None
    ratio = curr / prev
    if ratio > (1 - min_drop):
        return None
    inv = prev / curr
    for n in (2, 3, 4, 5):
        if abs(inv - n) / n < 0.1:
            return 1.0 / n
    if 0.15 < ratio < 0.85:
        return ratio
    return None


def forward_adjust_for_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    新浪等未复权数据：对份额拆分造成的价格断崖做前复权，避免图表「假暴跌」。
    东财 qfq 数据不应重复调用。
    """
    if df is None or df.empty:
        return df

    ref_col = "收盘" if "收盘" in df.columns else ("Close" if "Close" in df.columns else None)
    if not ref_col:
        return df

    close = pd.to_numeric(df[ref_col], errors="coerce")
    if close.dropna().empty:
        return df

    out = df.copy()
    working = close.astype(float).copy()
    split_dates: list[pd.Timestamp] = []

    for i in range(1, len(working)):
        prev = float(working.iloc[i - 1])
        curr = float(working.iloc[i])
        factor = _detect_split_factor(prev, curr)
        if factor is None:
            continue
        price_cols = [c for c in _PRICE_COLS if c in out.columns]
        for col in price_cols:
            out.loc[out.index[:i], col] = pd.to_numeric(out.loc[out.index[:i], col], errors="coerce") * factor
        working.iloc[:i] *= factor
        split_dates.append(working.index[i])

    if split_dates:
        out.attrs["split_adjusted"] = True
        out.attrs["split_dates"] = [d.strftime("%Y-%m-%d") for d in split_dates]
    return out


def _is_conn_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        k in text
        for k in (
            "connection aborted",
            "remotedisconnected",
            "connection reset",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "proxyerror",
            "max retries",
            "name or service not known",
            "failed to establish",
            "ssl",
            "403",
            "502",
            "503",
        )
    )


_SOURCE_LOCK = threading.Lock()
_PREFER_SINA: bool | None = None
_SINA_KLINE_URL = (
    "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData"
)
_SINA_HEADERS = {
    "User-Agent": EASTMONEY_HEADERS["User-Agent"],
    "Referer": "https://finance.sina.com.cn/",
    "Accept": "*/*",
}


def hist_prefer_sina() -> bool:
    """探测东财；不通则本进程一律走新浪。可用 ETF_PREFER_SINA=1 强制。"""
    global _PREFER_SINA
    if _PREFER_SINA is not None:
        return _PREFER_SINA
    with _SOURCE_LOCK:
        if _PREFER_SINA is not None:
            return _PREFER_SINA
        forced = os.environ.get("ETF_PREFER_SINA", "").strip().lower()
        if forced in ("1", "true", "yes", "sina"):
            _PREFER_SINA = True
            return True
        if forced in ("0", "false", "no", "eastmoney"):
            _PREFER_SINA = False
            return False
        try:
            df = _fetch_eastmoney_direct("510300")
            _PREFER_SINA = df is None or df.empty
        except Exception:
            _PREFER_SINA = True
        return _PREFER_SINA


def active_hist_source() -> str:
    return "sina" if hist_prefer_sina() else "eastmoney"


def _fetch_eastmoney_direct(symbol: str) -> pd.DataFrame:
    """直连东财 K 线 API（带浏览器头），与 AkShare fund_etf_hist_em 等价。"""
    code = clean_etf_symbol(symbol)
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    errors: list[str] = []

    for market_id in (_market_id(code), 1 - _market_id(code)):
        for ut in _UT_TOKENS[:2]:
            params = {
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
                "ut": ut,
                "klt": "101",
                "fqt": "1",
                "beg": "19700101",
                "end": bar_end_date().strftime("%Y%m%d"),
                "secid": f"{market_id}.{code}",
            }
            for attempt in range(2):
                try:
                    resp = requests.get(
                        url,
                        params=params,
                        headers=EASTMONEY_HEADERS,
                        timeout=15,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    klines = (data.get("data") or {}).get("klines") or []
                    if not klines:
                        break
                    temp = pd.DataFrame([item.split(",") for item in klines])
                    temp.columns = [
                        "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
                        "振幅", "涨跌幅", "涨跌额", "换手率",
                    ]
                    return _normalize_hist_df(temp)
                except Exception as exc:
                    errors.append(str(exc)[:80])
                    if _is_conn_error(exc):
                        raise RuntimeError(f"东财连接中断: {exc}") from exc
                    time.sleep(0.6 * (attempt + 1))

    raise RuntimeError(f"东财直连失败: {errors[-1] if errors else '无数据'}")


def _retry_fetch(fetcher: Callable[[], pd.DataFrame], retries: int = 3) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            df = _normalize_hist_df(fetcher())
            if not df.empty:
                return df
        except Exception as exc:
            last_exc = exc
            time.sleep(0.8 * (attempt + 1) + 0.5)
    if last_exc:
        raise last_exc
    return pd.DataFrame()


def _fetch_from_eastmoney(symbol: str) -> pd.DataFrame:
    try:
        return _fetch_eastmoney_direct(symbol)
    except Exception as direct_exc:
        if _is_conn_error(direct_exc):
            raise direct_exc
        code = clean_etf_symbol(symbol)
        return _retry_fetch(
            lambda: ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                adjust="qfq",
                end_date=bar_end_date().strftime("%Y%m%d"),
            ),
            retries=2,
        )


def _fetch_from_sina_kline_api(symbol: str) -> pd.DataFrame:
    """新浪日K JSON，不走 MiniRacer，适合 GitHub Actions。"""
    last_exc: Exception | None = None
    for sina_sym in sina_symbol_candidates(symbol):
        try:
            resp = requests.get(
                _SINA_KLINE_URL,
                params={"symbol": sina_sym, "scale": "240", "ma": "no", "datalen": "1023"},
                headers=_SINA_HEADERS,
                timeout=20,
            )
            resp.raise_for_status()
            text = (resp.text or "").strip()
            if not text or text in ("null", "[]"):
                continue
            data = resp.json()
            if not data:
                continue
            frame = pd.DataFrame(data)
            if "day" in frame.columns:
                frame = frame.rename(columns={"day": "date"})
            out = _normalize_hist_df(frame)
            if not out.empty:
                return forward_adjust_for_splits(out)
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    return pd.DataFrame()


def _fetch_from_sina(symbol: str) -> pd.DataFrame:
    """先走新浪 HTTP K 线；失败再退回 AkShare（MiniRacer，必须串行）。"""
    try:
        df = _fetch_from_sina_kline_api(symbol)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    last_exc: Exception | None = None
    with _SINA_LOCK:
        for sina_sym in sina_symbol_candidates(symbol):
            try:
                df = _retry_fetch(lambda sym=sina_sym: ak.fund_etf_hist_sina(symbol=sym))
                if not df.empty:
                    return forward_adjust_for_splits(df)
            except Exception as exc:
                last_exc = exc
    if last_exc:
        raise last_exc
    return pd.DataFrame()


def fetch_etf_daily(
    symbol: str,
    tail_days: int | None = None,
    prefer_sina: bool = False,
) -> tuple[pd.DataFrame, str]:
    """优先东财，失败则用新浪。东财探测不通时直接走新浪。"""
    errors: list[str] = []
    use_sina = bool(prefer_sina) or hist_prefer_sina()
    fetchers = (
        [("sina", lambda: _fetch_from_sina(symbol)), ("eastmoney", lambda: _fetch_from_eastmoney(symbol))]
        if use_sina
        else [("eastmoney", lambda: _fetch_from_eastmoney(symbol)), ("sina", lambda: _fetch_from_sina(symbol))]
    )
    for name, fetcher in fetchers:
        try:
            df = fetcher()
            if not df.empty:
                df = clip_to_last_complete(df)
                if df.empty:
                    continue
                if tail_days and len(df) > tail_days:
                    df = df.tail(tail_days)
                label = "akshare-eastmoney" if name == "eastmoney" else "akshare-sina"
                if name == "sina" and df.attrs.get("split_adjusted"):
                    label = "akshare-sina(前复权)"
                return df, label
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError(f"无法获取 {symbol}: {' | '.join(errors)}")


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_etf_daily_cached_asof(
    symbol: str, tail_days: int, prefer_sina: bool, asof: str
) -> tuple[pd.DataFrame, str]:
    """带缓存的单标的拉取，东财失败自动切新浪。"""
    try:
        return fetch_etf_daily(symbol, tail_days=tail_days, prefer_sina=prefer_sina)
    except Exception:
        if not prefer_sina:
            return fetch_etf_daily(symbol, tail_days=tail_days, prefer_sina=True)
        raise


def fetch_etf_daily_cached(symbol: str, tail_days: int = 150, prefer_sina: bool = False) -> tuple[pd.DataFrame, str]:
    return _fetch_etf_daily_cached_asof(symbol, tail_days, prefer_sina, data_asof_str())


fetch_etf_daily_cached.clear = _fetch_etf_daily_cached_asof.clear  # type: ignore[method-assign]


def _fetch_one(symbol: str, tail_days: int | None, prefer_sina: bool) -> tuple[str, pd.DataFrame, str]:
    # 默认东财；勿因 tail_days 短就强行走新浪（会触发 V8 崩溃）
    try:
        df, source = fetch_etf_daily(symbol, tail_days=tail_days or 150, prefer_sina=prefer_sina)
        return symbol, df, source
    except Exception:
        if not prefer_sina:
            try:
                df, source = fetch_etf_daily(symbol, tail_days=tail_days or 150, prefer_sina=True)
                return symbol, df, source
            except Exception:
                pass
        return symbol, pd.DataFrame(), "failed"


def fetch_pool_daily(
    symbols: list[str],
    pause_sec: float = 0.0,
    max_workers: int = 5,
    tail_days: int = 150,
    prefer_sina: bool | None = None,
) -> dict[str, pd.DataFrame]:
    """并行拉取池内行情；默认东财。prefer_sina=True 时强制单线程（V8 不可并发）。"""
    if not symbols:
        return {}

    unique = list(dict.fromkeys(clean_etf_symbol(s) for s in symbols))
    result: dict[str, pd.DataFrame] = {}
    use_sina = hist_prefer_sina() if prefer_sina is None else bool(prefer_sina)

    if len(unique) == 1:
        _, df, _ = _fetch_one(unique[0], tail_days, use_sina)
        result[unique[0]] = df
        return result

    # 新浪路径即使用锁，并行也无收益；单线程更稳
    workers = 1 if use_sina else min(max_workers, len(unique))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, sym, tail_days, use_sina): sym for sym in unique}
        for future in as_completed(futures):
            try:
                sym, df, _ = future.result()
            except Exception:
                sym = futures[future]
                df = pd.DataFrame()
            result[sym] = df
            if pause_sec > 0:
                time.sleep(pause_sec)
    return result


def diagnose_connectivity() -> list[dict]:
    import socket
    import time

    rows: list[dict] = []
    hosts = {
        "push2his.eastmoney.com": "东财历史K线",
        "finance.sina.com.cn": "新浪",
        "basic.10jqka.com.cn": "同花顺",
    }
    for host, label in hosts.items():
        t0 = time.time()
        try:
            ip = socket.gethostbyname(host)
            r = requests.get(f"https://{host}", headers=EASTMONEY_HEADERS, timeout=8)
            rows.append({"目标": label, "DNS": ip, "状态": f"HTTP {r.status_code}", "耗时": f"{time.time()-t0:.1f}s"})
        except Exception as exc:
            rows.append({"目标": label, "DNS": "-", "状态": str(exc)[:80], "耗时": f"{time.time()-t0:.1f}s"})
    return rows


def test_akshare_sources() -> list[dict]:
    import time

    results: list[dict] = []

    def _run(label: str, fn):
        t0 = time.time()
        try:
            df = fn()
            count = len(df) if hasattr(df, "__len__") else 1
            results.append({"接口": label, "状态": "可用", "详情": f"{count} 条", "耗时": f"{time.time() - t0:.1f}s"})
        except Exception as exc:
            results.append({"接口": label, "状态": "不可用", "详情": str(exc)[:100], "耗时": f"{time.time() - t0:.1f}s"})

    rows = diagnose_connectivity()
    _run("AkShare-同花顺列表", lambda: ak.fund_etf_spot_ths())
    _run("AkShare-东财列表", lambda: ak.fund_etf_spot_em())
    _run("AkShare-东财历史(直连)", lambda: _fetch_eastmoney_direct("510300"))
    _run("AkShare-新浪历史", lambda: ak.fund_etf_hist_sina(symbol="sh510300"))
    return rows + results
