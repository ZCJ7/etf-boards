"""买卖点复盘：指标评价 + 后 N 日假设收益。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analysis.conclusion import _bias_thresholds, _bias_zone, _price_ma_position
from analysis.indicators import bias_status
from analysis.trade_records import TradeRecord, align_trade_date

FORWARD_DAYS = 7


@dataclass
class TradePointReview:
    date: str
    action: str
    price: float | None
    note: str
    close: float | None
    bias6: float | None
    bias12: float | None
    bias24: float | None
    bias_zone: str
    macd_signal: str
    ma_position: str
    ma_dev_pct: float | None
    evaluation: str
    analysis: str
    forward_return_pct: float | None
    forward_end_date: str | None
    counterfactual: str


def _macd_at(macd_df: pd.DataFrame, aligned: pd.Timestamp) -> str:
    if macd_df.empty or aligned not in macd_df.index:
        return "—"
    dif = macd_df.loc[aligned, "DIF"] if "DIF" in macd_df.columns else None
    dea = macd_df.loc[aligned, "DEA"] if "DEA" in macd_df.columns else None
    if dif is None or dea is None or pd.isna(dif) or pd.isna(dea):
        return "—"
    return "金叉" if float(dif) > float(dea) else "死叉"


def _forward_return(close: pd.Series, aligned: pd.Timestamp, days: int = FORWARD_DAYS) -> tuple[float | None, str | None]:
    if aligned not in close.index:
        return None, None
    pos = close.index.get_loc(aligned)
    if not isinstance(pos, int):
        return None, None
    end_pos = pos + days
    if end_pos >= len(close):
        return None, None
    p0 = float(close.iloc[pos])
    p1 = float(close.iloc[end_pos])
    if p0 == 0:
        return None, None
    end_date = close.index[end_pos].strftime("%Y-%m-%d")
    return (p1 / p0 - 1) * 100, end_date


def _evaluate_buy(bias12: float | None, macd: str, ma_pos: str, ma_dev: float) -> tuple[str, str]:
    score = 0
    parts: list[str] = []

    if bias12 is not None and not pd.isna(bias12):
        zone = _bias_zone(bias12, 12)
        upper, lower = _bias_thresholds(12)
        st = bias_status(bias12, upper=upper, lower=lower)
        if zone in {"低位区", "偏低位"} or st == "超卖":
            score += 2
            parts.append("BIAS 处于偏低/超卖区，具备博弈性价比")
        elif zone in {"高位区", "偏高位"} or st == "超买":
            score -= 2
            parts.append("BIAS 偏高/超买，买入偏追高")
        else:
            parts.append("BIAS 中性，非极端位置")

    if macd == "金叉":
        score += 1
        parts.append("MACD 金叉，动能转强")
    elif macd == "死叉":
        score -= 1
        parts.append("MACD 死叉，动能仍弱")

    if ma_pos in {"低于均线", "远低于均线"}:
        score += 1
        parts.append(f"价格{ma_pos}（偏离 {ma_dev:+.1f}%），回调买入")
    elif ma_pos in {"高于均线", "远高于均线"}:
        score -= 1
        parts.append(f"价格{ma_pos}（偏离 {ma_dev:+.1f}%），追价成分较大")
    else:
        parts.append(f"价格{ma_pos}")

    if score >= 2:
        verdict = "买点较好"
    elif score >= 0:
        verdict = "买点中性"
    else:
        verdict = "买点偏早/追高"
    return verdict, "；".join(parts)


def _evaluate_sell(bias12: float | None, macd: str, ma_pos: str, ma_dev: float) -> tuple[str, str]:
    score = 0
    parts: list[str] = []

    if bias12 is not None and not pd.isna(bias12):
        zone = _bias_zone(bias12, 12)
        upper, lower = _bias_thresholds(12)
        st = bias_status(bias12, upper=upper, lower=lower)
        if zone in {"高位区", "偏高位"} or st == "超买":
            score += 2
            parts.append("BIAS 偏高/超买，落袋合理")
        elif zone in {"低位区", "偏低位"} or st == "超卖":
            score -= 2
            parts.append("BIAS 偏低/超卖，存在杀跌嫌疑")
        else:
            parts.append("BIAS 中性")

    if macd == "死叉":
        score += 1
        parts.append("MACD 死叉，顺势减仓")
    elif macd == "金叉":
        score -= 1
        parts.append("MACD 仍金叉，可能卖在升势")

    if ma_pos in {"高于均线", "远高于均线"}:
        score += 1
        parts.append(f"价格{ma_pos}（偏离 {ma_dev:+.1f}%），高位兑现")
    elif ma_pos in {"低于均线", "远低于均线"}:
        score -= 1
        parts.append(f"价格{ma_pos}（偏离 {ma_dev:+.1f}%），低位卖出")
    else:
        parts.append(f"价格{ma_pos}")

    if score >= 2:
        verdict = "卖点合理"
    elif score >= 0:
        verdict = "卖点中性"
    else:
        verdict = "卖点偏早"
    return verdict, "；".join(parts)


def _counterfactual_text(action: str, ret: float | None) -> str:
    if ret is None:
        return "后7交易日数据不足"
    if action == "买入":
        if ret > 0:
            return f"若该日不买，后7日少赚 {ret:+.2f}%（买入有效）"
        if ret < 0:
            return f"若该日不买，后7日少亏 {abs(ret):.2f}%（买点偏早）"
        return "若该日不买，后7日价格持平"
    if ret > 0:
        return f"若该日不卖，后7日多赚 {ret:+.2f}%（卖早了）"
    if ret < 0:
        return f"若该日不卖，后7日多亏 {abs(ret):.2f}%（卖出规避下跌）"
    return "若该日不卖，后7日价格持平"


def evaluate_trade_points(
    trades: list[TradeRecord],
    close: pd.Series,
    bias_df: pd.DataFrame,
    macd_df: pd.DataFrame,
    ma_df: pd.DataFrame,
    *,
    forward_days: int = FORWARD_DAYS,
    primary_ma: str = "MA20",
) -> list[TradePointReview]:
    if not trades or close.empty:
        return []

    ma_col = primary_ma if primary_ma in ma_df.columns else (ma_df.columns[0] if not ma_df.empty else None)
    out: list[TradePointReview] = []

    for t in sorted(trades, key=lambda x: x.date):
        aligned = align_trade_date(t.date, close.index)
        if aligned is None:
            continue

        px = float(close.loc[aligned]) if aligned in close.index else None
        b6 = float(bias_df.loc[aligned, "BIAS6"]) if "BIAS6" in bias_df.columns and aligned in bias_df.index and pd.notna(bias_df.loc[aligned, "BIAS6"]) else None
        b12 = float(bias_df.loc[aligned, "BIAS12"]) if "BIAS12" in bias_df.columns and aligned in bias_df.index and pd.notna(bias_df.loc[aligned, "BIAS12"]) else None
        b24 = float(bias_df.loc[aligned, "BIAS24"]) if "BIAS24" in bias_df.columns and aligned in bias_df.index and pd.notna(bias_df.loc[aligned, "BIAS24"]) else None

        zone = _bias_zone(b12, 12) if b12 is not None else "—"
        macd = _macd_at(macd_df, aligned)

        ma_val = float(ma_df.loc[aligned, ma_col]) if ma_col and aligned in ma_df.index and pd.notna(ma_df.loc[aligned, ma_col]) else None
        ma_pos, ma_dev = _price_ma_position(px or 0, ma_val) if px is not None else ("—", 0.0)

        if t.action == "买入":
            evaluation, analysis = _evaluate_buy(b12, macd, ma_pos, ma_dev)
        elif t.action == "卖出":
            evaluation, analysis = _evaluate_sell(b12, macd, ma_pos, ma_dev)
        else:
            evaluation, analysis = "—", "未知方向"

        fwd_ret, fwd_end = _forward_return(close, aligned, forward_days)
        counter = _counterfactual_text(t.action, fwd_ret)

        out.append(
            TradePointReview(
                date=aligned.strftime("%Y-%m-%d"),
                action=t.action,
                price=t.price if t.price is not None else px,
                note=t.note or "",
                close=px,
                bias6=b6,
                bias12=b12,
                bias24=b24,
                bias_zone=zone,
                macd_signal=macd,
                ma_position=ma_pos,
                ma_dev_pct=ma_dev if px is not None else None,
                evaluation=evaluation,
                analysis=analysis,
                forward_return_pct=fwd_ret,
                forward_end_date=fwd_end,
                counterfactual=counter,
            )
        )
    return out


def reviews_to_dataframe(reviews: list[TradePointReview]) -> pd.DataFrame:
    if not reviews:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "日期": r.date,
                "方向": r.action,
                "收盘价": r.close,
                "BIAS6": r.bias6,
                "BIAS12": r.bias12,
                "BIAS24": r.bias24,
                "BIAS区域": r.bias_zone,
                "MACD": r.macd_signal,
                "价均位置": r.ma_position,
                "偏离均线%": r.ma_dev_pct,
                "综合评价": r.evaluation,
                "指标分析": r.analysis,
                f"后{FORWARD_DAYS}日收益%": r.forward_return_pct,
                "统计至": r.forward_end_date,
                "假设情景": r.counterfactual,
            }
            for r in reviews
        ]
    )
