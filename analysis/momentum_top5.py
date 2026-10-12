"""中国场内 ETF Top5 动量轮动：评分、筛选、月度调仓回测。

参数集中在文件开头，方便修改。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.indicators import ensure_close
from etf_data_fetcher import fetch_pool_daily
from utils import clean_etf_symbol

# ---------- 可调参数 ----------
LOOKBACK = 25
MA_PERIOD = 20
MIN_AVG_AMOUNT_5D = 20_000_000  # 近5日均成交额，元（2000万）
TRADING_DAYS_YEAR = 250
ONE_WAY_COST = 0.0005  # 单边交易成本 0.05%
BACKTEST_START = "2019-01-01"
BENCHMARK_CODE = "510300"
INITIAL_CASH = 1_000_000.0
LIMIT_MOVE = 0.095  # 开盘相对昨收涨跌幅达到该值视为难成交

# 观察池：宽基 / 行业 / 红利 / 黄金 / 跨境。可自行增删。
ETF_UNIVERSE: dict[str, str] = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "159915": "创业板ETF",
    "588000": "科创50ETF",
    "512100": "中证1000ETF",
    "510050": "上证50ETF",
    "512890": "红利低波ETF",
    "510880": "红利ETF",
    "518880": "黄金ETF",
    "513100": "纳指ETF",
    "513500": "标普500ETF",
    "513050": "中概互联网ETF",
    "513130": "恒生科技ETF",
    "512480": "半导体ETF",
    "512760": "芯片ETF",
    "515050": "5GETF",
    "512690": "酒ETF",
    "512010": "医药ETF",
    "512800": "银行ETF",
    "515220": "煤炭ETF",
    "159819": "人工智能ETF",
    "512660": "军工ETF",
    "515790": "光伏ETF",
    "159981": "能源化工ETF",
    "512400": "有色金属ETF",
}


@dataclass
class MomentumConfig:
    lookback: int = LOOKBACK
    ma_period: int = MA_PERIOD
    min_avg_amount_5d: float = MIN_AVG_AMOUNT_5D
    trading_days_year: int = TRADING_DAYS_YEAR
    one_way_cost: float = ONE_WAY_COST
    backtest_start: str = BACKTEST_START
    benchmark_code: str = BENCHMARK_CODE
    initial_cash: float = INITIAL_CASH
    limit_move: float = LIMIT_MOVE
    top_n: int = 5


def _weights(n: int) -> np.ndarray:
    """近期权重更高：1,2,...,n。"""
    return np.arange(1, n + 1, dtype=float)


def momentum_score(close: pd.Series, cfg: MomentumConfig | None = None) -> dict | None:
    """加权线性回归：ln(收盘) ~ t，Score = 年化收益率 × R²。"""
    cfg = cfg or MomentumConfig()
    series = pd.to_numeric(close, errors="coerce").dropna()
    if len(series) < cfg.lookback:
        return None
    y_raw = series.tail(cfg.lookback).to_numpy(dtype=float)
    if np.any(y_raw <= 0):
        return None
    y = np.log(y_raw)
    n = len(y)
    x = np.arange(n, dtype=float)
    w = _weights(n)
    # 加权最小二乘：y = a + b x
    sw = w.sum()
    swx = (w * x).sum()
    swy = (w * y).sum()
    swxx = (w * x * x).sum()
    swxy = (w * x * y).sum()
    denom = sw * swxx - swx * swx
    if abs(denom) < 1e-18:
        return None
    slope = (sw * swxy - swx * swy) / denom
    intercept = (swy - slope * swx) / sw
    fitted = intercept + slope * x
    resid = y - fitted
    y_mean = swy / sw
    ss_res = float((w * resid * resid).sum())
    ss_tot = float((w * (y - y_mean) ** 2).sum())
    if ss_tot <= 0:
        return None
    r2 = max(0.0, 1.0 - ss_res / ss_tot)
    annualized = float(math.exp(slope * cfg.trading_days_year) - 1.0)
    score = annualized * r2
    return {
        "slope": float(slope),
        "annualized": annualized,
        "r2": float(r2),
        "score": float(score),
    }


def _amount_series(df: pd.DataFrame) -> pd.Series:
    """成交额（元）。新浪等源常缺成交额，用 成交量(手)×收盘×100 近似。"""
    if "成交额" in df.columns:
        amt = pd.to_numeric(df["成交额"], errors="coerce")
        if amt.notna().sum() > 0:
            return amt
    if "amount" in df.columns:
        amt = pd.to_numeric(df["amount"], errors="coerce")
        if amt.notna().sum() > 0:
            return amt
    close = ensure_close(df)
    vol = None
    if "成交量" in df.columns:
        vol = pd.to_numeric(df["成交量"], errors="coerce")
    elif "Volume" in df.columns:
        vol = pd.to_numeric(df["Volume"], errors="coerce")
    if vol is None:
        return pd.Series(index=df.index, dtype=float)
    # A 股 ETF 成交量多为手，1 手 = 100 份
    return vol * close * 100.0


def _open_series(df: pd.DataFrame) -> pd.Series:
    if "开盘" in df.columns:
        return pd.to_numeric(df["开盘"], errors="coerce")
    if "Open" in df.columns:
        return pd.to_numeric(df["Open"], errors="coerce")
    return ensure_close(df)


def load_universe_prices(
    universe: dict[str, str] | None = None,
    cfg: MomentumConfig | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], list[str]]:
    """拉取观察池日K。返回 (行情, 名称, 失败代码说明)。"""
    cfg = cfg or MomentumConfig()
    universe = universe or ETF_UNIVERSE
    codes = [clean_etf_symbol(c) for c in universe]
    # 优先东财以拿到更长历史和成交额；失败时 fetch 内部会切新浪
    daily = fetch_pool_daily(codes, tail_days=2800, max_workers=8, pause_sec=0, prefer_sina=False)
    frames: dict[str, pd.DataFrame] = {}
    names: dict[str, str] = {}
    issues: list[str] = []
    start = pd.Timestamp(cfg.backtest_start)
    for code in codes:
        frame = daily.get(code)
        name = universe.get(code, code)
        if frame is None or frame.empty:
            issues.append(f"{code} {name}：无行情")
            continue
        work = frame.copy()
        work.index = pd.to_datetime(work.index).normalize()
        work = work[~work.index.duplicated(keep="last")].sort_index()
        close = ensure_close(work)
        if close.dropna().empty:
            issues.append(f"{code} {name}：收盘价为空")
            continue
        # 至少需要 lookback + ma 根完整K，且尽量覆盖回测起点后一段
        if len(close.dropna()) < cfg.lookback + cfg.ma_period:
            issues.append(f"{code} {name}：历史过短（{len(close.dropna())} 根）")
            continue
        first = close.dropna().index[0]
        if first > start + pd.Timedelta(days=400):
            issues.append(f"{code} {name}：历史从 {first.date()} 起（短于回测起点），按可得日进入")
        amt = _amount_series(work)
        if amt.notna().sum() == 0:
            issues.append(f"{code} {name}：无成交额，成交额筛选无法通过")
        frames[code] = work
        names[code] = name
    return frames, names, issues


def _panel(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    closes = {c: ensure_close(df) for c, df in frames.items()}
    opens = {c: _open_series(df) for c, df in frames.items()}
    amounts = {c: _amount_series(df) for c, df in frames.items()}
    close_df = pd.DataFrame(closes).sort_index()
    open_df = pd.DataFrame(opens).reindex(close_df.index)
    amount_df = pd.DataFrame(amounts).reindex(close_df.index)
    # 开盘缺失时用收盘替代（仅作成交价回退）
    open_df = open_df.fillna(close_df)
    return close_df, open_df, amount_df


def score_asof(
    close_df: pd.DataFrame,
    amount_df: pd.DataFrame,
    asof: pd.Timestamp,
    cfg: MomentumConfig | None = None,
) -> pd.DataFrame:
    """在 asof 收盘后计算观察池评分与筛选结果。"""
    cfg = cfg or MomentumConfig()
    asof = pd.Timestamp(asof).normalize()
    if asof not in close_df.index:
        prior = close_df.index[close_df.index <= asof]
        if len(prior) == 0:
            return pd.DataFrame()
        asof = prior[-1]

    rows = []
    hist_close = close_df.loc[:asof]
    hist_amt = amount_df.loc[:asof]
    for code in close_df.columns:
        close = hist_close[code].dropna()
        if len(close) < max(cfg.lookback, cfg.ma_period):
            continue
        scored = momentum_score(close, cfg)
        if scored is None:
            continue
        ma20 = float(close.tail(cfg.ma_period).mean())
        last = float(close.iloc[-1])
        amt = hist_amt[code].dropna().tail(5)
        avg_amt = float(amt.mean()) if len(amt) >= 5 else float("nan")
        ret20 = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else float("nan")
        ok = (
            scored["slope"] > 0
            and last > ma20
            and avg_amt >= cfg.min_avg_amount_5d
            and len(close) >= cfg.lookback
        )
        rows.append(
            {
                "代码": code,
                "动量评分": scored["score"],
                "年化斜率收益": scored["annualized"],
                "R2": scored["r2"],
                "斜率": scored["slope"],
                "现价": last,
                "MA20": ma20,
                "近5日均成交额": avg_amt,
                "近20日收益率": ret20,
                "入选": ok,
                "信号日": asof,
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("动量评分", ascending=False).reset_index(drop=True)
    return out


def pick_top(scores: pd.DataFrame, top_n: int = 5) -> list[str]:
    if scores is None or scores.empty:
        return []
    eligible = scores[scores["入选"]].copy()
    if eligible.empty:
        return []
    return eligible.head(int(top_n))["代码"].tolist()


def month_end_signal_dates(index: pd.DatetimeIndex, start: pd.Timestamp) -> list[pd.Timestamp]:
    work = index[index >= start]
    if len(work) == 0:
        return []
    series = pd.Series(1, index=work)
    dates: list[pd.Timestamp] = []
    for _, grp in series.groupby(pd.Grouper(freq="ME")):
        if len(grp):
            dates.append(pd.Timestamp(grp.index[-1]).normalize())
    # 未走完的当月不要当成月末信号（例如 10 月中只有到 10-09）
    dates = [d for d in dates if len(index[index > d]) > 0]
    return dates


def _next_trade_date(index: pd.DatetimeIndex, signal: pd.Timestamp) -> pd.Timestamp | None:
    later = index[index > signal]
    if len(later) == 0:
        return None
    return pd.Timestamp(later[0]).normalize()


def _can_trade(open_px: float, prev_close: float, limit_move: float) -> bool:
    if not np.isfinite(open_px) or not np.isfinite(prev_close) or prev_close <= 0 or open_px <= 0:
        return False
    move = abs(open_px / prev_close - 1.0)
    return move < limit_move


def _metrics(nav: pd.Series, cost_total: float, n_trades: int) -> dict:
    nav = nav.dropna()
    if len(nav) < 2:
        return {}
    rets = nav.pct_change().dropna()
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    days = (nav.index[-1] - nav.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)
    vol = float(rets.std() * math.sqrt(TRADING_DAYS_YEAR)) if len(rets) else float("nan")
    sharpe = float(cagr / vol) if vol and vol > 0 else float("nan")
    drawdown = nav / nav.cummax() - 1
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    if len(nav):
        year_ends = nav.groupby(nav.index.year).last()
        year_starts = nav.groupby(nav.index.year).first()
        annual = (year_ends / year_starts - 1).dropna()
        annual.index = annual.index.astype(str)
    else:
        annual = pd.Series(dtype=float)
    return {
        "累计收益率": total,
        "年化收益率CAGR": cagr,
        "最大回撤": max_dd,
        "年化波动率": vol,
        "夏普比率": sharpe,
        "卡玛比率": calmar,
        "年度收益率": annual,
        "调仓次数": int(n_trades),
        "交易成本": float(cost_total),
        "回撤序列": drawdown,
        "净值": nav,
    }


def _run_rotation(
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    amount_df: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    top_n: int,
    cfg: MomentumConfig,
) -> dict:
    """信号日收盘选股，下一交易日开盘调仓。

    假设：开盘相对昨收涨跌幅达到涨跌停近似阈值时，该标的当日不买卖；
    卖不出的仓位继续持有，剩余现金在可成交的目标 ETF 上等权买入。
    符合条件不足 top_n 时，只买入选标的，其余保持现金。
    """
    cash = cfg.initial_cash
    holdings: dict[str, float] = {}
    cost_total = 0.0
    n_rebalances = 0
    nav_points: dict[pd.Timestamp, float] = {}
    history: list[dict] = []

    pending: dict[pd.Timestamp, list[str]] = {}
    for sig in signal_dates:
        exec_d = _next_trade_date(close_df.index, sig)
        if exec_d is None:
            continue
        scores = score_asof(close_df, amount_df, sig, cfg)
        picks = pick_top(scores, top_n=top_n)
        pending[exec_d] = picks
        history.append({"信号日": sig, "调仓日": exec_d, "持仓": picks})

    def _prev_close(code: str, dt: pd.Timestamp) -> float:
        loc = close_df.index.get_loc(dt)
        if isinstance(loc, slice) or not np.isscalar(loc) or loc <= 0:
            return float("nan")
        return float(close_df.iloc[int(loc) - 1][code])

    for dt in close_df.index:
        dt = pd.Timestamp(dt).normalize()
        if dt in pending:
            n_rebalances += 1
            targets = pending[dt]
            stuck: dict[str, float] = {}
            for code, shares in list(holdings.items()):
                open_px = float(open_df.at[dt, code]) if code in open_df.columns else float("nan")
                if not _can_trade(open_px, _prev_close(code, dt), cfg.limit_move):
                    stuck[code] = shares
                    continue
                proceeds = shares * open_px
                fee = proceeds * cfg.one_way_cost
                cash += proceeds - fee
                cost_total += fee

            tradable_targets = []
            for code in targets:
                if code not in open_df.columns:
                    continue
                open_px = float(open_df.at[dt, code])
                if _can_trade(open_px, _prev_close(code, dt), cfg.limit_move):
                    tradable_targets.append(code)

            new_holdings = dict(stuck)
            n_buy = len(tradable_targets)
            if n_buy > 0 and cash > 0:
                budget = cash / n_buy
                for code in tradable_targets:
                    open_px = float(open_df.at[dt, code])
                    if open_px <= 0:
                        continue
                    shares = (budget / (1.0 + cfg.one_way_cost)) / open_px
                    fee = shares * open_px * cfg.one_way_cost
                    cash -= shares * open_px + fee
                    cost_total += fee
                    new_holdings[code] = new_holdings.get(code, 0.0) + shares
            holdings = new_holdings

        equity = cash
        for code, shares in holdings.items():
            px = float(close_df.at[dt, code]) if code in close_df.columns else float("nan")
            if np.isfinite(px):
                equity += shares * px
        nav_points[dt] = equity

    nav = pd.Series(nav_points).sort_index() / cfg.initial_cash
    return {
        "nav": nav,
        "metrics": _metrics(nav, cost_total, n_rebalances),
        "history": history,
        "cost_total": cost_total,
        "n_trades": n_rebalances,
    }


def _buy_and_hold(close_df: pd.DataFrame, code: str, start: pd.Timestamp, cfg: MomentumConfig) -> dict:
    if code not in close_df.columns:
        return {"nav": pd.Series(dtype=float), "metrics": {}, "history": [], "cost_total": 0.0, "n_trades": 0}
    series = close_df[code].dropna()
    series = series[series.index >= start]
    if len(series) < 2:
        return {"nav": pd.Series(dtype=float), "metrics": {}, "history": [], "cost_total": 0.0, "n_trades": 0}
    # 第一天开盘买（用收盘近似首日），收一次买入费
    nav = series / series.iloc[0]
    nav.iloc[0] = 1.0
    # 买入成本体现在初始净值折扣
    nav = nav / (1 + cfg.one_way_cost)
    cost = cfg.initial_cash * cfg.one_way_cost
    metrics = _metrics(nav, cost, 1)
    return {"nav": nav, "metrics": metrics, "history": [], "cost_total": cost, "n_trades": 1}


def _equal_weight_hold(
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    cfg: MomentumConfig,
) -> dict:
    """观察池等权：每月按当时可得 ETF 再平衡。"""
    cash = cfg.initial_cash
    holdings: dict[str, float] = {}
    cost_total = 0.0
    n_rebalances = 0
    nav_points: dict[pd.Timestamp, float] = {}
    pending: dict[pd.Timestamp, list[str]] = {}
    for sig in signal_dates:
        exec_d = _next_trade_date(close_df.index, sig)
        if exec_d is None:
            continue
        available = [
            c
            for c in close_df.columns
            if close_df.loc[:sig, c].dropna().shape[0] >= cfg.lookback + cfg.ma_period
        ]
        pending[exec_d] = available

    for dt in close_df.index:
        dt = pd.Timestamp(dt).normalize()
        if dt in pending and pending[dt]:
            n_rebalances += 1
            targets = []
            for code in pending[dt]:
                open_px = float(open_df.at[dt, code])
                prev_idx = close_df.index.get_loc(dt)
                prev_close = float(close_df.iloc[prev_idx - 1][code]) if prev_idx > 0 else float("nan")
                if _can_trade(open_px, prev_close, cfg.limit_move):
                    targets.append(code)
            for code, shares in list(holdings.items()):
                open_px = float(open_df.at[dt, code])
                prev_idx = close_df.index.get_loc(dt)
                prev_close = float(close_df.iloc[prev_idx - 1][code]) if prev_idx > 0 else float("nan")
                if not _can_trade(open_px, prev_close, cfg.limit_move):
                    continue
                proceeds = shares * open_px
                fee = proceeds * cfg.one_way_cost
                cash += proceeds - fee
                cost_total += fee
                del holdings[code]
            n_buy = len(targets)
            if n_buy and cash > 0:
                budget = cash / n_buy
                for code in targets:
                    open_px = float(open_df.at[dt, code])
                    shares = (budget / (1 + cfg.one_way_cost)) / open_px
                    fee = shares * open_px * cfg.one_way_cost
                    cash -= shares * open_px + fee
                    cost_total += fee
                    holdings[code] = shares
        equity = cash
        for code, shares in holdings.items():
            px = float(close_df.at[dt, code])
            if np.isfinite(px):
                equity += shares * px
        nav_points[dt] = equity

    nav = pd.Series(nav_points).sort_index() / cfg.initial_cash
    return {
        "nav": nav,
        "metrics": _metrics(nav, cost_total, n_rebalances),
        "history": [],
        "cost_total": cost_total,
        "n_trades": n_rebalances,
    }


def next_rebalance_date(signal_date: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp | None:
    return _next_trade_date(index, signal_date)


def current_top5(
    frames: dict[str, pd.DataFrame] | None = None,
    names: dict[str, str] | None = None,
    cfg: MomentumConfig | None = None,
) -> dict:
    cfg = cfg or MomentumConfig()
    if frames is None:
        frames, names, issues = load_universe_prices(cfg=cfg)
    else:
        issues = []
        names = names or {}
    close_df, open_df, amount_df = _panel(frames)
    if close_df.empty:
        return {"rows": [], "issues": issues, "error": "观察池无可用行情"}
    asof = close_df.index[-1]
    scores = score_asof(close_df, amount_df, asof, cfg)
    picks = pick_top(scores, top_n=cfg.top_n)
    # 信号日：若今日已是月末则用今日，否则用最近一次月末信号日展示「当前排名」
    # 当前清单按最新完整交易日计算（不是等到月末才更新展示）
    rows = []
    for i, code in enumerate(picks, start=1):
        row = scores[scores["代码"] == code].iloc[0]
        rows.append(
            {
                "排名": i,
                "ETF名称": names.get(code, code),
                "代码": code,
                "动量评分": round(float(row["动量评分"]), 4),
                "近20日收益率": round(float(row["近20日收益率"]) * 100, 2) if pd.notna(row["近20日收益率"]) else None,
            }
        )
    # 计划调仓日：下一个月末信号之后的交易日
    sig_dates = month_end_signal_dates(close_df.index, pd.Timestamp(cfg.backtest_start))
    last_sig = sig_dates[-1] if sig_dates else asof
    # 若最新日已超过上月末，计划调仓是「本月末信号后的下一交易日」——未知未来交易日时标注「下月首个交易日」
    if asof > last_sig:
        plan = "下一次：本月最后一个交易日收盘后出信号，下一交易日调仓"
        signal_for_display = asof
    else:
        plan_dt = next_rebalance_date(last_sig, close_df.index)
        plan = plan_dt.strftime("%Y-%m-%d") if plan_dt else "待定"
        signal_for_display = last_sig
    return {
        "rows": rows,
        "scores": scores,
        "asof": asof.strftime("%Y-%m-%d"),
        "signal_date": signal_for_display.strftime("%Y-%m-%d"),
        "plan_rebalance": plan,
        "issues": issues,
        "names": names,
        "eligible_count": int(scores["入选"].sum()) if not scores.empty else 0,
    }


def run_backtests(
    frames: dict[str, pd.DataFrame] | None = None,
    names: dict[str, str] | None = None,
    cfg: MomentumConfig | None = None,
) -> dict:
    cfg = cfg or MomentumConfig()
    if frames is None:
        frames, names, issues = load_universe_prices(cfg=cfg)
    else:
        issues = []
        names = names or {}
    close_df, open_df, amount_df = _panel(frames)
    if close_df.empty or cfg.benchmark_code not in close_df.columns:
        return {"error": "缺少基准或观察池行情", "issues": issues}

    start = pd.Timestamp(cfg.backtest_start)
    # 回测从 max(设定起点, 基准可得, 至少若干只ETF可得) 开始
    bench = close_df[cfg.benchmark_code].dropna()
    start = max(start, bench.index[0] + pd.Timedelta(days=cfg.lookback + cfg.ma_period))
    # 至少 5 只 ETF 有数据
    avail_counts = close_df.notna().sum(axis=1)
    ready = avail_counts[avail_counts >= 5]
    if ready.empty:
        return {"error": "同时具备行情的 ETF 不足 5 只", "issues": issues}
    start = max(start, ready.index[0])

    close_df = close_df.loc[close_df.index >= start - pd.Timedelta(days=120)]
    open_df = open_df.reindex(close_df.index)
    amount_df = amount_df.reindex(close_df.index)

    signal_dates = month_end_signal_dates(close_df.index, start)
    # 丢掉没有下一交易日的最后信号
    signal_dates = [d for d in signal_dates if _next_trade_date(close_df.index, d) is not None]
    if len(signal_dates) < 3:
        return {"error": "可调仓月份过少", "issues": issues}

    top5 = _run_rotation(close_df, open_df, amount_df, signal_dates, top_n=5, cfg=cfg)
    top1 = _run_rotation(close_df, open_df, amount_df, signal_dates, top_n=1, cfg=cfg)
    eq = _equal_weight_hold(close_df, open_df, signal_dates, cfg)
    bh = _buy_and_hold(close_df, cfg.benchmark_code, start, cfg)

    # 对齐净值起点
    navs = {}
    for key, pack in [("Top5动量", top5), ("Top1动量", top1), ("观察池等权", eq), ("沪深300持有", bh)]:
        nav = pack["nav"].dropna()
        nav = nav[nav.index >= start]
        if len(nav):
            nav = nav / nav.iloc[0]
        pack["nav"] = nav
        pack["metrics"] = _metrics(nav, pack.get("cost_total", 0.0), pack.get("n_trades", 0))
        navs[key] = nav

    common_start = max(s.index[0] for s in navs.values() if len(s))
    common_end = min(s.index[-1] for s in navs.values() if len(s))
    for key in list(navs):
        nav = navs[key].loc[common_start:common_end]
        navs[key] = nav / nav.iloc[0]
        if key == "Top5动量":
            top5["metrics"] = _metrics(nav, top5.get("cost_total", 0.0), top5.get("n_trades", 0))
            top5["nav"] = nav
        elif key == "Top1动量":
            top1["metrics"] = _metrics(nav, top1.get("cost_total", 0.0), top1.get("n_trades", 0))
            top1["nav"] = nav
        elif key == "观察池等权":
            eq["metrics"] = _metrics(nav, eq.get("cost_total", 0.0), eq.get("n_trades", 0))
            eq["nav"] = nav
        else:
            bh["metrics"] = _metrics(nav, bh.get("cost_total", 0.0), bh.get("n_trades", 0))
            bh["nav"] = nav

    compare_rows = []
    for label, pack in [
        ("沪深300持有", bh),
        ("观察池等权", eq),
        ("Top1动量", top1),
        ("Top5动量", top5),
    ]:
        m = pack["metrics"]
        compare_rows.append(
            {
                "策略": label,
                "累计收益率": m.get("累计收益率"),
                "年化收益率CAGR": m.get("年化收益率CAGR"),
                "最大回撤": m.get("最大回撤"),
                "年化波动率": m.get("年化波动率"),
                "夏普比率": m.get("夏普比率"),
                "卡玛比率": m.get("卡玛比率"),
                "调仓次数": m.get("调仓次数"),
                "交易成本": m.get("交易成本"),
            }
        )

    analysis = build_analysis(top5, top1, eq, bh, names, issues, start, common_end)
    return {
        "compare": pd.DataFrame(compare_rows),
        "navs": navs,
        "top5": top5,
        "top1": top1,
        "equal_weight": eq,
        "buy_hold": bh,
        "start": common_start.strftime("%Y-%m-%d"),
        "end": common_end.strftime("%Y-%m-%d"),
        "issues": issues,
        "names": names,
        "analysis": analysis,
        "signal_dates": signal_dates,
    }


def build_analysis(top5, top1, eq, bh, names, issues, start, end) -> str:
    m5 = top5["metrics"]
    m1 = top1["metrics"]
    me = eq["metrics"]
    mb = bh["metrics"]

    def pct(x):
        return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:.1f}%"

    def num(x):
        return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"

    beat_bh = m5.get("累计收益率", -1) > mb.get("累计收益率", 0)
    beat_eq = m5.get("累计收益率", -1) > me.get("累计收益率", 0)
    dd5 = m5.get("最大回撤")
    dd1 = m1.get("最大回撤")
    lower_risk = dd5 is not None and dd1 is not None and abs(dd5) < abs(dd1)

    annual = m5.get("年度收益率")
    weak_years = []
    if isinstance(annual, pd.Series) and len(annual):
        weak_years = [f"{y}（{v * 100:.1f}%）" for y, v in annual.items() if v < 0]

    late = [x for x in issues if "上市偏晚" in x]
    data_note = ""
    if pd.Timestamp(start) > pd.Timestamp("2019-06-01"):
        data_note = (
            f"当前行情源未能覆盖到 2019 年（实际从 {pd.Timestamp(start).date()} 起），"
            "因此回测短于原计划，结论只代表这一段。"
        )
    lines = [
        f"回测区间 {pd.Timestamp(start).date()} 至 {pd.Timestamp(end).date()}。"
        f"Top5 累计收益 {pct(m5.get('累计收益率'))}，年化 {pct(m5.get('年化收益率CAGR'))}，"
        f"最大回撤 {pct(dd5)}。{data_note}",
        f"同期沪深300持有累计 {pct(mb.get('累计收益率'))}，观察池等权累计 {pct(me.get('累计收益率'))}。"
        + ("Top5 跑赢了沪深300持有。" if beat_bh else "Top5 没有跑赢沪深300持有。")
        + ("也跑赢了观察池等权。" if beat_eq else "但没有跑赢观察池等权。"),
        f"收益主要来自每月把资金等权放到动量评分最高、且站上 MA20 的少数 ETF；"
        f"期间调仓 {m5.get('调仓次数', 0)} 次，累计交易成本约占初始资金 "
        f"{(m5.get('交易成本') or 0) / INITIAL_CASH * 100:.2f}%。",
        f"相比 Top1（最大回撤 {pct(dd1)}、年化波动 {pct(m1.get('年化波动率'))}），"
        + (
            f"Top5 回撤更小（{pct(dd5)}），分散持有确实降低了单一主题踩踏的风险。"
            if lower_risk
            else f"Top5 回撤并未明显小于 Top1，分散带来的保护有限。"
        ),
        (
            "表现较差的年份：" + "、".join(weak_years) + "。这些年份往往是趋势反复、板块轮动很快，或动量拥挤回撤的阶段。"
            if weak_years
            else "样本里 Top5 没有出现年度收益为负的年份，但这不代表未来不会亏损。"
        ),
        "结果受观察池选择影响很大：池子里行业主题多，牛市里动量会偏进攻；"
        "部分 ETF 上市晚于 2019 年，早期月份实际可交易品种更少。"
        + (f" 例如：{late[0]}。" if late else "")
        + f" 单边成本按 {ONE_WAY_COST * 100:.2f}% 计，成本提高会侵蚀换手收益。",
        "值得继续研究，但先别加仓实盘。下一步最值得验证的是："
        "换一套更均衡的观察池、把调仓改成每周、以及把近5日成交额门槛提高，看结论是否稳定。",
    ]
    return "\n".join(f"{i}. {t}" for i, t in enumerate(lines, start=1))


def run_all(cfg: MomentumConfig | None = None) -> dict:
    """一次跑完：拉数 → 当前 Top5 → 回测 → 分析。"""
    cfg = cfg or MomentumConfig()
    frames, names, issues = load_universe_prices(cfg=cfg)
    top = current_top5(frames, names, cfg)
    bt = run_backtests(frames, names, cfg)
    return {"top": top, "backtest": bt, "issues": issues, "names": names}
