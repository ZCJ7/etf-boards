"""持仓核心结论：精简概括每支 ETF 状态。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analysis.indicators import calc_bias, calc_macd, calc_ma, ensure_close


@dataclass
class ConclusionConfig:
    ma_period: int = 20
    bias_period: int = 24
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    momentum_days: int = 20
    ma_slope_days: int = 5


@dataclass
class HoldingVerdict:
    symbol: str
    name: str
    close: float
    ret_momentum: float
    momentum_days: int
    trend: str
    price_ma_position: str
    price_ma_dev_pct: float
    ma_trend: str
    ma_slope_pct: float
    bias_value: float
    bias_zone: str
    bias_text: str
    macd_signal: str
    macd_desc: str
    ma_strength: str
    score: int
    verdict: str
    position_action: str
    color: str
    ma_period: int
    bias_period: int


def _bias_thresholds(period: int) -> tuple[float, float]:
    if period <= 6:
        return 3.5, -3.5
    if period <= 12:
        return 5.0, -5.0
    return 9.0, -9.0


def _price_ma_position(close: float, ma: float | None) -> tuple[str, float]:
    if ma is None or pd.isna(ma) or ma == 0:
        return "未知", 0.0
    dev = (close - ma) / ma * 100
    if dev >= 4.0:
        return "远高于均线", dev
    if dev >= 1.2:
        return "高于均线", dev
    if dev <= -4.0:
        return "远低于均线", dev
    if dev <= -1.2:
        return "低于均线", dev
    return "贴近均线", dev


def _ma_trend_label(ma_series: pd.Series, lookback: int = 5) -> tuple[str, float]:
    clean = ma_series.dropna()
    if len(clean) < lookback + 1:
        return "未知", 0.0
    current = float(clean.iloc[-1])
    prev = float(clean.iloc[-1 - lookback])
    if prev == 0:
        return "未知", 0.0
    slope = (current - prev) / prev * 100
    if slope > 0.8:
        return "均线上升", slope
    if slope < -0.8:
        return "均线下降", slope
    return "均线走平", slope


def _trend_label(price_ma_position: str, ma_trend: str) -> str:
    if price_ma_position == "未知" or ma_trend == "未知":
        return "数据不足"
    above = price_ma_position in {"高于均线", "远高于均线"}
    below = price_ma_position in {"低于均线", "远低于均线"}
    if above and ma_trend == "均线上升":
        return "多头趋势"
    if below and ma_trend == "均线下降":
        return "空头趋势"
    if above and ma_trend == "均线下降":
        return "反弹遇阻"
    if below and ma_trend == "均线上升":
        return "回调蓄势"
    if above:
        return "偏强震荡"
    if below:
        return "偏弱震荡"
    return "横盘整理"


def _bias_zone(bias: float | None, period: int) -> str:
    if bias is None or pd.isna(bias):
        return "未知"
    upper, lower = _bias_thresholds(period)
    if bias >= upper + 1:
        return "高位区"
    if bias <= lower - 1:
        return "低位区"
    if bias >= upper * 0.6:
        return "偏高位"
    if bias <= lower * 0.6:
        return "偏低位"
    return "中性区"


def _format_bias_text(bias_val: float | None, period: int, zone: str) -> str:
    if bias_val is None or pd.isna(bias_val):
        return f"BIAS{period} —"
    return f"BIAS{period} {bias_val:+.2f}%（{zone}）"


def _macd_bar_desc(macd_df: pd.DataFrame) -> tuple[str, str]:
    """
    解析 MACD 柱：A股习惯红柱=正值、绿柱=负值；说明柱体变长/变短。
    返回 (金叉/死叉, 完整描述)
    """
    hist = macd_df["MACD"].dropna()
    if len(hist) < 2:
        return "未知", "MACD数据不足"

    cur = float(hist.iloc[-1])
    prev = float(hist.iloc[-2])
    dif = macd_df["DIF"].iloc[-1]
    dea = macd_df["DEA"].iloc[-1]

    if cur > 0:
        bar_color = "红柱"
    elif cur < 0:
        bar_color = "绿柱"
    else:
        bar_color = "零轴"

    abs_cur, abs_prev = abs(cur), abs(prev)
    if abs_prev < 1e-12:
        bar_len = "持平"
    elif abs_cur > abs_prev * 1.08:
        bar_len = "变长"
    elif abs_cur < abs_prev * 0.92:
        bar_len = "变短"
    else:
        bar_len = "持平"

    if pd.isna(dif) or pd.isna(dea):
        cross = "未知"
    else:
        cross = "金叉" if dif > dea else "死叉"

    desc = f"{bar_color}{bar_len}·{cross}"
    return cross, desc


def _build_position_action(
    score: int,
    verdict: str,
    trend: str,
    price_ma_position: str,
    ma_trend: str,
    bias_zone: str,
    bias_val: float,
    macd_desc: str,
    ret_momentum: float,
) -> str:
    """给出具体加减仓操作建议。"""
    red_long = "红柱" in macd_desc and "变长" in macd_desc
    red_short = "红柱" in macd_desc and "变短" in macd_desc
    green_long = "绿柱" in macd_desc and "变长" in macd_desc
    green_short = "绿柱" in macd_desc and "变短" in macd_desc
    golden = "金叉" in macd_desc
    death = "死叉" in macd_desc
    bias_high = bias_zone in {"高位区", "偏高位"}
    bias_low = bias_zone in {"低位区", "偏低位"}

    # 强势 + 趋势配合
    if score >= 72 and trend == "多头趋势":
        if bias_high:
            return "持有为主；已超买不宜追高，不加仓"
        if red_long and golden:
            return "持有并可加仓20%~30%（趋势与MACD共振）"
        return "持有并可小幅加仓10%~20%"

    if score >= 72:
        if bias_high:
            return "持有；BIAS偏高，不加仓"
        return "持有；可小幅加仓10%"

    # 偏强
    if score >= 58:
        if price_ma_position == "远高于均线" or bias_high:
            return "持有不加仓；偏离过大，等回调再考虑加仓"
        if red_short and bias_high:
            return "持有不加仓；动能减弱，勿追高"
        if golden and red_long:
            return "持有；可试探加仓10%"
        return "持有不加不减"

    # 中性
    if score >= 42:
        if green_long and death:
            return "减仓20%~30%；MACD走弱"
        if trend == "回调蓄势" and bias_low:
            return "轻仓持有；可小仓试探加仓10%（回调+BIAS低）"
        if green_short and golden:
            return "持有观望；绿柱缩短有企稳迹象，暂不加仓"
        return "轻仓持有；不加不减，等待方向"

    # 偏弱
    if score >= 28:
        if green_long or death:
            return "减仓30%~50%；趋势与MACD偏弱"
        if bias_high:
            return "减仓30%；BIAS仍高，落袋为安"
        return "减仓20%~30%；降低博弈仓位"

    # 弱势
    if green_long and death:
        return "减仓至轻仓或清仓观望；空头信号明确"
    if trend == "空头趋势":
        return "减仓50%以上或清仓；空头趋势"
    return "减仓至轻仓；暂不加仓"


def _score_verdict(
    price_ma_position: str,
    ma_trend: str,
    bias_zone: str,
    macd_signal: str,
    ret_momentum: float,
) -> tuple[int, str, str, str]:
    score = 50

    price_scores = {
        "远高于均线": 6,
        "高于均线": 12,
        "贴近均线": 0,
        "低于均线": -10,
        "远低于均线": -14,
    }
    score += price_scores.get(price_ma_position, 0)

    ma_trend_scores = {
        "均线上升": 12,
        "均线走平": 0,
        "均线下降": -12,
    }
    score += ma_trend_scores.get(ma_trend, 0)

    if price_ma_position in {"高于均线", "远高于均线"} and ma_trend == "均线上升":
        score += 6
    elif price_ma_position in {"低于均线", "远低于均线"} and ma_trend == "均线下降":
        score -= 6

    if bias_zone == "低位区":
        score += 10
    elif bias_zone == "偏低位":
        score += 5
    elif bias_zone == "高位区":
        score -= 12
    elif bias_zone == "偏高位":
        score -= 6

    if macd_signal == "金叉":
        score += 8
    elif macd_signal == "死叉":
        score -= 8

    if ret_momentum > 8:
        score += 5
    elif ret_momentum < -8:
        score -= 5

    score = max(0, min(100, score))

    if score >= 72:
        return score, "强势", "#dc2626"
    if score >= 58:
        return score, "偏强", "#f97316"
    if score >= 42:
        return score, "中性", "#64748b"
    if score >= 28:
        return score, "偏弱", "#16a34a"
    return score, "弱势", "#15803d"


def build_holding_verdict(
    symbol: str,
    name: str,
    df: pd.DataFrame,
    cfg: ConclusionConfig | None = None,
) -> HoldingVerdict:
    cfg = cfg or ConclusionConfig()
    close = ensure_close(df)
    last = float(close.iloc[-1])

    ma_col = f"MA{cfg.ma_period}"
    ma_df = calc_ma(close, [cfg.ma_period])
    ma_series = ma_df[ma_col]
    bias_col = f"BIAS{cfg.bias_period}"
    bias_df = calc_bias(close, [cfg.bias_period])
    macd_df = calc_macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)

    ma_val = ma_series.iloc[-1]
    bias_val_raw = bias_df[bias_col].iloc[-1]
    bias_val = float(bias_val_raw) if pd.notna(bias_val_raw) else 0.0

    md = cfg.momentum_days
    ret_momentum = float(close.pct_change(md).iloc[-1] * 100) if len(close) > md else 0.0

    slope_days = min(cfg.ma_slope_days, max(cfg.ma_period // 2, 3))
    price_ma_position, price_ma_dev_pct = _price_ma_position(last, ma_val)
    ma_trend, ma_slope_pct = _ma_trend_label(ma_series, lookback=slope_days)
    trend = _trend_label(price_ma_position, ma_trend)
    bias_zone = _bias_zone(bias_val_raw, cfg.bias_period)
    bias_text = _format_bias_text(bias_val_raw, cfg.bias_period, bias_zone)
    macd_signal, macd_desc = _macd_bar_desc(macd_df)

    ma_strength = f"{price_ma_position}({price_ma_dev_pct:+.1f}%)"
    score, verdict, color = _score_verdict(price_ma_position, ma_trend, bias_zone, macd_signal, ret_momentum)
    position_action = _build_position_action(
        score,
        verdict,
        trend,
        price_ma_position,
        ma_trend,
        bias_zone,
        bias_val,
        macd_desc,
        ret_momentum,
    )

    return HoldingVerdict(
        symbol=symbol,
        name=name,
        close=last,
        ret_momentum=round(ret_momentum, 2),
        momentum_days=md,
        trend=trend,
        price_ma_position=price_ma_position,
        price_ma_dev_pct=round(price_ma_dev_pct, 2),
        ma_trend=ma_trend,
        ma_slope_pct=round(ma_slope_pct, 2),
        bias_value=round(bias_val, 2),
        bias_zone=bias_zone,
        bias_text=bias_text,
        macd_signal=macd_signal,
        macd_desc=macd_desc,
        ma_strength=ma_strength,
        score=score,
        verdict=verdict,
        position_action=position_action,
        color=color,
        ma_period=cfg.ma_period,
        bias_period=cfg.bias_period,
    )


def build_pool_verdicts(
    price_map: dict[str, pd.DataFrame],
    pool: dict[str, str],
    cfg: ConclusionConfig | None = None,
) -> list[HoldingVerdict]:
    out: list[HoldingVerdict] = []
    for sym, df in price_map.items():
        if df is None or df.empty:
            continue
        out.append(build_holding_verdict(sym, pool.get(sym, sym), df, cfg))
    return sorted(out, key=lambda x: x.score, reverse=True)
