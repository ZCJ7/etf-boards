"""技术指标计算：BIAS、MACD、均线、比价线。"""

from __future__ import annotations

import pandas as pd


def ensure_close(df: pd.DataFrame) -> pd.Series:
    if "Close" in df.columns:
        return pd.to_numeric(df["Close"], errors="coerce")
    if "收盘" in df.columns:
        return pd.to_numeric(df["收盘"], errors="coerce")
    raise ValueError("DataFrame 缺少收盘价列")


def calc_ma(close: pd.Series, periods: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    for p in periods:
        out[f"MA{p}"] = close.rolling(p).mean()
    return out


def calc_bias(close: pd.Series, periods: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    for p in periods:
        ma = close.rolling(p).mean()
        out[f"BIAS{p}"] = (close - ma) / ma * 100
    return out


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return pd.DataFrame({"DIF": dif, "DEA": dea, "MACD": hist}, index=close.index)


def calc_ratio_line(
    numerator: pd.Series,
    denominator: pd.Series,
    ma_period: int = 20,
) -> pd.DataFrame:
    aligned = pd.concat([numerator.rename("num"), denominator.rename("den")], axis=1).dropna()
    ratio = aligned["num"] / aligned["den"]
    ratio_ma = ratio.rolling(ma_period).mean()
    ratio_bias = (ratio - ratio_ma) / ratio_ma * 100
    return pd.DataFrame(
        {
            "比价": ratio,
            f"比价MA{ma_period}": ratio_ma,
            "比价BIAS": ratio_bias,
        },
        index=aligned.index,
    )


def bias_status(bias_value: float, upper: float = 5.0, lower: float = -5.0) -> str:
    if pd.isna(bias_value):
        return "数据不足"
    if bias_value >= upper:
        return "超买"
    if bias_value <= lower:
        return "超卖"
    return "中性"


def latest_snapshot(
    close: pd.Series,
    ma_periods: list[int],
    bias_periods: list[int],
    macd_params: tuple[int, int, int],
) -> dict:
    ma_df = calc_ma(close, ma_periods)
    bias_df = calc_bias(close, bias_periods)
    macd_df = calc_macd(close, *macd_params)
    last = close.index[-1]
    snap = {
        "日期": last.strftime("%Y-%m-%d"),
        "收盘价": float(close.iloc[-1]),
    }
    for col in ma_df.columns:
        snap[col] = float(ma_df[col].iloc[-1]) if pd.notna(ma_df[col].iloc[-1]) else None
    for col in bias_df.columns:
        val = bias_df[col].iloc[-1]
        snap[col] = float(val) if pd.notna(val) else None
        period = int(col.replace("BIAS", ""))
        snap[f"{col}状态"] = bias_status(
            val,
            upper=3.5 if period <= 6 else (5.0 if period <= 12 else 9.0),
            lower=-3.5 if period <= 6 else (-5.0 if period <= 12 else -9.0),
        )
    for col in macd_df.columns:
        snap[col] = float(macd_df[col].iloc[-1]) if pd.notna(macd_df[col].iloc[-1]) else None
    if snap.get("DIF") is not None and snap.get("DEA") is not None:
        snap["MACD信号"] = "金叉" if snap["DIF"] > snap["DEA"] else "死叉"
    return snap
