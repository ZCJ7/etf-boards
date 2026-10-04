"""中国版聪明资金信心指数（CSMCI）— 对标 SentimenTrader Confidence。

优化：各数据源独立缓存 + 并行短超时，避免拖死 Streamlit。
"""

from __future__ import annotations

from dataclasses import dataclass

import akshare as ak
import pandas as pd
import streamlit as st

from etf_data_fetcher import fetch_etf_daily_cached
from utils import clean_etf_symbol

LOOKBACK = 60
ROLL_DAYS = 5
ROLL_OPTIONS = (5, 10, 20)
# 约半年交易日（用于图表展示）
CHART_DAYS = 126

_ETF_BASKET = ("510300", "510500", "159915")

WEIGHTS = {
    "futures_hold": 0.30,
    "qvix": 0.20,
    "northbound": 0.20,
    "margin": 0.15,
    "etf_flow": 0.15,
}

FACTOR_META = {
    "futures_hold": {"name": "IF前20会员净持仓", "dir": "正向", "weight": 0.30},
    "qvix": {"name": "50ETF期权波动率QVIX", "dir": "逆向", "weight": 0.20},
    "northbound": {"name": "北向资金净流入", "dir": "正向", "weight": 0.20},
    "margin": {"name": "融资余额变化", "dir": "逆向", "weight": 0.15},
    "etf_flow": {"name": "宽基ETF资金流向", "dir": "正向", "weight": 0.15},
}

# 明细/帮助文案：{n} = 近 N 日窗口
_FACTOR_EXPLAIN_TPL = {
    "futures_hold": (
        "中金所沪深300股指期货（IF）品种：多头前20会员持仓 - 空头前20会员持仓，看近{n}日净持仓变化。"
        "净多增加/净空减少 → 机构偏多 → 信心分偏高（正向）。"
        "数据源：中金所会员持仓排名；若拉取失败则回退 IF0 总持仓变化。"
    ),
    "qvix": (
        "50ETF 期权隐含波动率。"
        "波动率飙升=市场恐慌/对冲需求高 → 信心分偏低（逆向：QVIX 越高分越低）。"
    ),
    "northbound": (
        "北向资金净买入（近{n}日合计）。"
        "外资净流入偏多 → 信心分偏高。"
        "注：官方净买入明细 2024-08 后已停更，当前可能不计入权重。"
    ),
    "margin": (
        "两融「融资余额」近{n}日变化率%。"
        "余额快速上升=散户加杠杆追涨，常对应过热 → 信心分偏低（逆向）。"
        "余额收缩=去杠杆降温 → 信心分偏高；若处近60日最强收缩，得分可到100。"
    ),
    "etf_flow": (
        "宽基 ETF（如 510300/510500/159915）成交额×涨跌方向，近{n}日合计。"
        "上涨日放量偏流入、下跌日放量偏流出；净流入偏多 → 信心分偏高（正向）。"
    ),
}


def factor_explain(key: str, n: int = ROLL_DAYS) -> str:
    tpl = _FACTOR_EXPLAIN_TPL.get(key, "")
    return tpl.format(n=n) if tpl else ""


# 兼容旧引用
FACTOR_EXPLAIN = {k: v.format(n=ROLL_DAYS) for k, v in _FACTOR_EXPLAIN_TPL.items()}

DEFAULT_BENCH = ["510300", "510500", "159915"]


@dataclass
class FactorSeries:
    key: str
    raw: pd.Series
    score: pd.Series
    ok: bool
    note: str = ""


def _norm_index(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").dropna()
    if out.empty:
        return out
    out.index = pd.to_datetime(out.index).normalize()
    return out.sort_index()


def _minmax_percentile(series: pd.Series, window: int = LOOKBACK, reverse: bool = False) -> pd.Series:
    """向量化滚动 min-max 百分位。"""
    s = pd.to_numeric(series, errors="coerce")
    roll_min = s.rolling(window, min_periods=max(10, window // 4)).min()
    roll_max = s.rolling(window, min_periods=max(10, window // 4)).max()
    span = (roll_max - roll_min).replace(0, pd.NA)
    score = (s - roll_min) / span * 100
    score = score.fillna(50.0)
    if reverse:
        score = 100 - score
    return score


def _call_timeout(fn, timeout: float = 12):
    """对纯网络请求做超时（fn 内禁止调用 st.cache）。"""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

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


def _raw_hsgt() -> pd.DataFrame:
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _raw_margin() -> pd.DataFrame:
    try:
        df = ak.macro_china_market_margin_sh()
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _raw_qvix() -> pd.DataFrame:
    try:
        df = ak.index_option_50etf_qvix()
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _raw_futures(symbol: str = "IF0") -> pd.DataFrame:
    try:
        df = ak.futures_zh_daily_sina(symbol=symbol)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_hsgt_hist() -> pd.DataFrame:
    df = _call_timeout(_raw_hsgt, 12)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_margin_raw() -> pd.DataFrame:
    df = _call_timeout(_raw_margin, 12)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_qvix_raw() -> pd.DataFrame:
    df = _call_timeout(_raw_qvix, 10)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_futures_hold(symbol: str = "IF0") -> pd.DataFrame:
    df = _call_timeout(lambda: _raw_futures(symbol), 10)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _raw_if_top20_rank(start_day: str, end_day: str) -> pd.DataFrame:
    """中金所 IF 品种前20会员持仓汇总（按日）。"""
    import contextlib
    import io
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # akshare 会 print 每个交易日，静默以免刷屏拖慢前端
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                df = ak.get_rank_sum_daily(start_day=start_day, end_day=end_day, vars_list=["IF"])
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_if_top20_rank(start_day: str, end_day: str) -> pd.DataFrame:
    # 半年区间约 10–15 秒，超时放宽；失败返回空表走 IF0 回退
    df = _call_timeout(lambda: _raw_if_top20_rank(start_day, end_day), 40)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _parse_if_top20_net(df: pd.DataFrame, roll_days: int = ROLL_DAYS) -> tuple[pd.Series, str]:
    """IF 品种：多头前20 - 空头前20，近 N 日净持仓变化。"""
    empty = pd.Series(dtype=float)
    if df is None or df.empty:
        return empty, "IF前20会员持仓接口失败"
    work = df.copy()
    if "symbol" in work.columns:
        work = work[work["symbol"].astype(str).str.upper() == "IF"]
    if work.empty:
        return empty, "IF前20无品种汇总行"
    date_col = "date" if "date" in work.columns else ("日期" if "日期" in work.columns else None)
    if not date_col:
        return empty, "IF前20无日期列"
    long_col = "long_open_interest_top20"
    short_col = "short_open_interest_top20"
    if long_col not in work.columns or short_col not in work.columns:
        return empty, "IF前20缺多空持仓列"
    raw_dates = work[date_col].astype(str)
    work["_dt"] = pd.to_datetime(raw_dates, format="%Y%m%d", errors="coerce")
    if work["_dt"].isna().all():
        work["_dt"] = pd.to_datetime(raw_dates, errors="coerce")
    long_s = pd.to_numeric(work[long_col], errors="coerce")
    short_s = pd.to_numeric(work[short_col], errors="coerce")
    net = _norm_index(pd.Series((long_s - short_s).values, index=work["_dt"]))
    if net.empty:
        return empty, "IF前20净持仓无有效序列"
    return net.diff(roll_days), f"IF品种前20会员净持仓(多-空)近{roll_days}日变化"


def _series_stale(s: pd.Series, max_lag_days: int = 40) -> bool:
    """序列末日期是否过旧（北向净买入 2024-08 后已停更）。"""
    if s is None or s.dropna().empty:
        return True
    last = pd.Timestamp(s.dropna().index.max()).normalize()
    return (pd.Timestamp.today().normalize() - last).days > max_lag_days


def _parse_northbound(df: pd.DataFrame, roll_days: int = ROLL_DAYS) -> tuple[pd.Series, pd.Series, str]:
    """返回 (净流入滚动, 沪深300, note)。净买额停更后改用资金余额变化；仍过旧则空序列。"""
    empty = pd.Series(dtype=float)
    if df is None or df.empty:
        return empty, empty, "北向资金接口失败"
    work = df.copy()
    date_col = "日期" if "日期" in work.columns else work.columns[0]
    work[date_col] = pd.to_datetime(work[date_col])
    work = work.set_index(date_col).sort_index()

    hs = empty
    if "沪深300" in work.columns:
        hs = _norm_index(pd.to_numeric(work["沪深300"], errors="coerce"))

    flow_col = "当日成交净买额" if "当日成交净买额" in work.columns else None
    if flow_col is None:
        for c in work.columns:
            if "净买" in str(c):
                flow_col = c
                break
    flow = pd.to_numeric(work[flow_col], errors="coerce") if flow_col else pd.Series(dtype=float, index=work.index)
    flow = _norm_index(flow)

    bal_col = next((c for c in work.columns if "资金余额" in str(c) or str(c) == "余额"), None)
    bal_chg = empty
    if bal_col:
        bal_chg = _norm_index(pd.to_numeric(work[bal_col], errors="coerce").diff())

    # 优先用仍更新的序列；净买额自 2024-08 起官方停更
    if not _series_stale(flow, 40):
        s, note = flow, f"当日成交净买额近{roll_days}日合计"
    elif not _series_stale(bal_chg, 40):
        s, note = bal_chg, f"北向资金余额日变化近{roll_days}日合计（净买额已停更）"
    elif not flow.empty:
        last = pd.Timestamp(flow.index.max()).date()
        return empty, hs, f"北向净买额已停更（止于{last}），暂不计入权重"
    else:
        return empty, hs, "北向资金无有效序列"

    return s.rolling(roll_days).sum(), hs, note


def _parse_margin(df: pd.DataFrame, roll_days: int = ROLL_DAYS) -> tuple[pd.Series, str]:
    empty = pd.Series(dtype=float)
    if df is None or df.empty:
        return empty, "融资余额接口失败"
    work = df.copy()
    date_col = "日期" if "日期" in work.columns else work.columns[0]
    bal_col = next((c for c in work.columns if "融资余额" in str(c)), None)
    if not bal_col:
        return empty, "未找到融资余额列"
    work[date_col] = pd.to_datetime(work[date_col])
    s = _norm_index(pd.to_numeric(work.set_index(date_col)[bal_col], errors="coerce"))
    return s.pct_change(roll_days) * 100, f"融资余额近{roll_days}日变化率%"


def _parse_qvix(df: pd.DataFrame) -> tuple[pd.Series, str]:
    empty = pd.Series(dtype=float)
    if df is None or df.empty:
        return empty, "50ETF QVIX 失败"
    work = df.copy()
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"])
        work = work.set_index("date")
    col = "close" if "close" in work.columns else work.columns[-1]
    return _norm_index(pd.to_numeric(work[col], errors="coerce")), "50ETF期权波动率QVIX（逆向）"


def _parse_futures(df: pd.DataFrame, roll_days: int = ROLL_DAYS) -> tuple[pd.Series, str]:
    empty = pd.Series(dtype=float)
    if df is None or df.empty or "hold" not in df.columns:
        return empty, "股指期货持仓失败"
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    s = _norm_index(pd.to_numeric(work.set_index("date")["hold"], errors="coerce"))
    return s.diff(roll_days), f"IF0主力连续持仓近{roll_days}日变化（前20回退）"


def _fetch_futures_factor(roll_days: int, hist_keep: int) -> tuple[pd.Series, str]:
    """优先中金所 IF 前20净持仓；失败回退新浪 IF0 总持仓。"""
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(hist_keep * 1.7) + 10)
    start_s, end_s = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    top20_s, top20_note = _parse_if_top20_net(_cached_if_top20_rank(start_s, end_s), roll_days=roll_days)
    if not top20_s.dropna().empty:
        return top20_s, top20_note
    return _parse_futures(_cached_futures_hold("IF0"), roll_days=roll_days)


def _fetch_etf_flow_fast(roll_days: int = ROLL_DAYS, hist_days: int = LOOKBACK + 40) -> tuple[pd.Series, str]:
    flows: list[pd.Series] = []
    for code in _ETF_BASKET:
        try:
            df, _ = fetch_etf_daily_cached(code, tail_days=hist_days + roll_days, prefer_sina=True)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        close = pd.to_numeric(df["Close"] if "Close" in df.columns else df.get("收盘"), errors="coerce")
        amt = None
        for col in ("成交额", "amount", "Amount"):
            if col in df.columns:
                amt = pd.to_numeric(df[col], errors="coerce")
                break
        if amt is None and "Volume" in df.columns:
            amt = pd.to_numeric(df["Volume"], errors="coerce") * close
        if amt is None:
            continue
        ret = close.pct_change()
        signed = amt * ret.map(lambda x: 0.0 if pd.isna(x) or x == 0 else (1.0 if x > 0 else -1.0))
        flows.append(_norm_index(signed))
    if not flows:
        return pd.Series(dtype=float), "宽基ETF行情不足"
    total = pd.concat(flows, axis=1).fillna(0).sum(axis=1)
    return total.rolling(roll_days).sum(), f"宽基{','.join(_ETF_BASKET)}成交额×涨跌方向近{roll_days}日"


def _build_factor(key: str, raw: pd.Series, reverse: bool, note: str) -> FactorSeries:
    if raw is None or raw.dropna().empty:
        return FactorSeries(key=key, raw=pd.Series(dtype=float), score=pd.Series(dtype=float), ok=False, note=note)
    score = _minmax_percentile(raw.dropna(), window=LOOKBACK, reverse=reverse)
    return FactorSeries(key=key, raw=raw, score=score, ok=True, note=note)


def _try_pack(fn, fallback):
    try:
        return fn()
    except Exception as exc:
        if isinstance(fallback, tuple) and fallback and isinstance(fallback[-1], str):
            items = list(fallback)
            items[-1] = f"{items[-1]}: {str(exc)[:48]}"
            return tuple(items)
        return fallback


@st.cache_data(ttl=3600, show_spinner=False)
def compute_csmci(lookback: int = LOOKBACK, roll_days: int = ROLL_DAYS) -> dict:
    """主线程顺序拉取；roll_days 为近 N 日窗口（5/10/20）；失败因子自动降权。"""
    n = int(roll_days) if int(roll_days) in ROLL_OPTIONS else ROLL_DAYS
    # 图表要近半年，历史至少留足 CHART_DAYS + 百分位窗口预热
    hist_keep = max(lookback + 30, CHART_DAYS + n + 5)

    north_pack = _try_pack(
        lambda: _parse_northbound(_cached_hsgt_hist(), roll_days=n),
        (pd.Series(dtype=float), pd.Series(dtype=float), "北向失败"),
    )
    margin_pack = _try_pack(
        lambda: _parse_margin(_cached_margin_raw(), roll_days=n),
        (pd.Series(dtype=float), "融资失败"),
    )
    qvix_pack = _try_pack(
        lambda: _parse_qvix(_cached_qvix_raw()),
        (pd.Series(dtype=float), "QVIX失败"),
    )
    fut_pack = _try_pack(
        lambda: _fetch_futures_factor(roll_days=n, hist_keep=hist_keep),
        (pd.Series(dtype=float), "期货失败"),
    )
    etf_pack = _try_pack(
        lambda: _fetch_etf_flow_fast(roll_days=n, hist_days=hist_keep),
        (pd.Series(dtype=float), "ETF流向失败"),
    )

    north, hs300, n_note = north_pack
    margin, m_note = margin_pack
    qvix, q_note = qvix_pack
    fut, f_note = fut_pack
    etf, e_note = etf_pack

    factors = {
        "futures_hold": _build_factor("futures_hold", fut, False, f_note),
        "qvix": _build_factor("qvix", qvix, True, q_note),
        "northbound": _build_factor("northbound", north, False, n_note),
        "margin": _build_factor("margin", margin, True, m_note),
        "etf_flow": _build_factor("etf_flow", etf, False, e_note),
    }

    score_map = {k: f.score.rename(k) for k, f in factors.items() if f.ok and not f.score.dropna().empty}
    if not score_map:
        return {
            "history": pd.DataFrame(),
            "latest": {},
            "factors": factors,
            "roll_days": n,
            "error": "全部子指标失败，请稍后重试或点清除缓存",
        }

    scores = pd.concat(score_map.values(), axis=1).sort_index()
    # 融资等源常慢 1 日：因子分向前填最多 3 日，避免最新日整列变「—」
    factor_cols = [c for c in scores.columns if c in WEIGHTS]
    scores[factor_cols] = scores[factor_cols].ffill(limit=3)
    scores = scores.loc[scores[factor_cols].notna().sum(axis=1) >= 1].tail(hist_keep)

    csmci = []
    for _, row in scores.iterrows():
        w_sum = v_sum = 0.0
        for key, w in WEIGHTS.items():
            val = row.get(key)
            if pd.notna(val):
                w_sum += w
                v_sum += float(val) * w
        csmci.append(v_sum / w_sum if w_sum > 0 else float("nan"))
    scores["CSMCI"] = csmci

    if hs300 is not None and not hs300.empty:
        scores["沪深300"] = hs300.reindex(scores.index).ffill()

    history = scores.copy()
    history.index.name = "日期"
    history = history.reset_index()
    history["日期"] = pd.to_datetime(history["日期"])

    latest: dict = {}
    valid = history.dropna(subset=["CSMCI"])
    if not valid.empty:
        last = valid.iloc[-1]
        score = float(last["CSMCI"])
        zone = "偏强（机构信心较高）" if score > 70 else ("偏弱（机构谨慎/对冲）" if score < 30 else "中性")
        latest = {
            "日期": last["日期"].strftime("%Y-%m-%d"),
            "CSMCI": round(score, 1),
            "区间": zone,
            "沪深300": float(last["沪深300"]) if "沪深300" in last and pd.notna(last["沪深300"]) else None,
            "roll_days": n,
        }
        for key in WEIGHTS:
            if key in last and pd.notna(last[key]):
                latest[key] = round(float(last[key]), 1)
            else:
                # 展示该因子自身最近有效分，避免「有序列却显示 —」
                fac = factors.get(key)
                if fac and fac.ok and not fac.score.dropna().empty:
                    latest[key] = round(float(fac.score.dropna().iloc[-1]), 1)

    return {"history": history, "latest": latest, "factors": factors, "roll_days": n, "error": None}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_benchmark_closes(codes: tuple[str, ...], tail_days: int = 150) -> pd.DataFrame:
    """拉取自选宽基/ETF 收盘价，列名为代码。"""
    frames: dict[str, pd.Series] = {}
    for code in codes:
        sym = clean_etf_symbol(code)
        if not sym:
            continue
        try:
            df, _ = fetch_etf_daily_cached(sym, tail_days=tail_days, prefer_sina=True)
            if df is None or df.empty:
                continue
            close = pd.to_numeric(df["Close"] if "Close" in df.columns else df.get("收盘"), errors="coerce")
            frames[sym] = _norm_index(close)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    out.columns = [str(c) for c in out.columns]
    return out


def normalize_to_100(df: pd.DataFrame) -> pd.DataFrame:
    """将价格序列归一到起点=100，便于与 CSMCI 同图对比。"""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in out.columns:
        s = pd.to_numeric(out[col], errors="coerce").dropna()
        if s.empty:
            continue
        base = float(s.iloc[0])
        if base == 0:
            continue
        out[col] = out[col] / base * 100
    return out
