"""指定 ETF 的支撑位：均线、前低、缺口、斐波、成交密集、周线。"""

from __future__ import annotations

import pandas as pd

from analysis.indicators import ensure_close

_MERGE_PCT = 0.008


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(index=df.index, dtype=float)


def _swing_lows(low: pd.Series, span: int = 5) -> list[tuple[pd.Timestamp, float]]:
    values = low.to_numpy(dtype=float)
    out: list[tuple[pd.Timestamp, float]] = []
    for i in range(span, len(values) - span):
        window = values[i - span : i + span + 1]
        if values[i] <= window.min() and values[i] < values[i - 1] and values[i] <= values[i + 1]:
            out.append((low.index[i], float(values[i])))
    return out


def _touch_count(low: pd.Series, close: pd.Series, price: float, band: float = 0.006) -> int:
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    touches = 0
    i = 0
    last = len(lows) - 1
    while i < last:
        if price * (1 - band) <= lows[i] <= price * (1 + band):
            held = closes[i] >= price * 0.998
            if not held and i + 1 < len(closes):
                held = closes[i + 1] > price
            if held:
                touches += 1
            i += 4
        else:
            i += 1
    return touches


def _channel(close: pd.Series, high: pd.Series, low: pd.Series, window: int = 60) -> dict | None:
    """近 N 日收盘的回归通道。下轨、上轨是斜线画到最新一根 K 的价格。"""
    import numpy as np

    c = close.tail(window).dropna()
    if len(c) < 30:
        return None
    h = high.reindex(c.index)
    l = low.reindex(c.index)
    x = np.arange(len(c), dtype=float)
    y = c.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    line = intercept + slope * x
    upper_off = float(np.nanmax(h.to_numpy(dtype=float) - line))
    lower_off = float(np.nanmin(l.to_numpy(dtype=float) - line))
    upper_line = pd.Series(line + upper_off, index=c.index)
    lower_line = pd.Series(line + lower_off, index=c.index)
    upper_now = float(upper_line.iloc[-1])
    lower_now = float(lower_line.iloc[-1])
    last = float(y[-1])
    if last <= 0 or upper_now <= lower_now:
        return None
    width = (upper_now - lower_now) / last
    if width > 0.4:
        return None
    slope_20 = float(slope * 20 / last * 100)
    if slope_20 > 0.6:
        direction = "上升"
    elif slope_20 < -0.6:
        direction = "下降"
    else:
        direction = "横盘"
    return {
        "lower": lower_now,
        "upper": upper_now,
        "direction": direction,
        "slope_20": slope_20,
        "upper_line": upper_line,
        "lower_line": lower_line,
    }


def _strategy(status: str, kinds: str, dist: float, touches: int) -> str:
    if "通道下轨" in kinds:
        piled = "这一档还和别的支撑叠在一起。" if " + " in kinds else ""
        if status == "已跌破":
            return "通道下轨已经被收盘跌破，这条斜向支撑失效。反抽到下轨附近看成压力。" + piled
        if "下降" in kinds:
            return "这是下降通道的下轨，斜线还在往下走。守住只是下跌途中的暂歇，收盘跌破会走得更快。" + piled
        if "上升" in kinds:
            return "这是上升通道的下轨。回踩到这里、收盘守住，才是顺着通道的位置；跌破下轨，这段上升通道就坏了。" + piled
        return "这是横盘通道的下轨。收盘守住可以看反弹；跌破就离开这个通道。" + piled
    weekly = "周" in kinds or "年线" in kinds or "MA250" in kinds or "MA120" in kinds
    dense = "成交密集" in kinds
    gap = "缺口" in kinds
    fib = "斐波" in kinds
    tested = touches >= 2

    if status == "已跌破":
        return "这一档已经失守，不再当支撑。反抽到这里看成压力，收盘重新站上才恢复支撑。"
    if status == "正在测试":
        text = "现价就在这一档上。收盘守住可以观察止跌；收盘跌破并且第二天收不回来，先减不加。"
        if tested:
            text += "这里以前接住过，第一次回踩比连续跌破更值得看。"
        if dense:
            text += "成交曾经堆在这里，来回拉锯会多一些，不适合一次买完。"
        return text
    if status == "临近":
        text = f"离现价大约 {dist:.1f}%。还没到，中间位置不追，等靠近这一档再看。"
        if tested:
            text += "历史上多次在这里止跌，靠近时优先看缩量。"
        elif gap:
            text += "这是向上缺口的下沿，回补到这里短线容易有承接，补完这档就消失。"
        elif fib and "0.618" in kinds:
            text += "0.618 回撤比浅回调更值得等，到了再看有没有止跌。"
        else:
            text += "到了看收盘是否守住，而不是盘中刚碰就加。"
        if weekly:
            text += "这是中线级别，适合等确认，不适合当成短线第一买点。"
        return text
    text = f"还在现价下方约 {dist:.1f}%，是更远的一档。"
    if weekly:
        text += "日线支撑都失守之后才轮到它，用来看中线还有没有防守，不用来做眼前的买卖。"
    elif gap:
        text += "价格真跌到缺口下沿，短线才有承接意义。"
    else:
        text += "前面的支撑先破，再把这一档当成下一笔计划。"
    return text


def _priority(score: float, status: str, rank: int, dist: float) -> str:
    if status in ("已跌破", "已突破"):
        return "中" if abs(dist) <= 3.5 else "低"
    if rank == 1 and status in ("正在测试", "临近"):
        return "高"
    if score >= 8:
        return "高"
    if score >= 4.5:
        return "中"
    return "低"


def _position_summary(last: float, zones: list[dict]) -> str:
    valid = [z for z in zones if z["状态"] != "已跌破"]
    broken = [z for z in zones if z["状态"] == "已跌破" and z["距现价%"] > -4]
    if not valid:
        return f"现价 {last:.3f} 下方没有聚出有效支撑，跌起来缺少就近的防守位。"
    first = valid[0]
    second = valid[1] if len(valid) > 1 else None
    parts = [
        f"现价 {last:.3f}。优先看 {first['支撑价']:.3f}（{first['构成']}，{first['状态']}，距现价 {first['距现价%']:.1f}%）。"
    ]
    if first["状态"] == "正在测试":
        parts.append("价格已经贴在第一支撑上，这一档的策略就是看收盘守不守得住。")
    elif first["距现价%"] > 8:
        parts.append("近处没有支撑，中间是真空，跌起来会直接去找这一档，不适合在半路接。")
    elif second and abs(first["支撑价"] - second["支撑价"]) / last <= 0.03:
        parts.append(
            f"下一档 {second['支撑价']:.3f} 离得很近。第一档失守后容易滑到下一档，仓位按两档之间来留，不要只押一档。"
        )
    else:
        parts.append("第一档和第二档拉开了，计划只放在第一档附近，破了再看下一档。")
    if broken:
        top = min(broken, key=lambda item: abs(item["距现价%"]))
        parts.append(f"{top['支撑价']:.3f} 已经在现价上方，角色改成压力，反抽站不回去就继续按下跌看。")
    overhead = [z for z in zones if z["状态"] == "已跌破"]
    ma_valid = [z for z in valid if any(k in z["构成"] for k in ("MA", "年线", "周"))]
    ma_broken = [z for z in overhead if any(k in z["构成"] for k in ("MA", "年线", "周"))]
    if ma_broken and not ma_valid:
        parts.append("日线和周线均线都在现价上方，眼下不能再把均线当支撑。")
    return "".join(parts)


def analyze_supports(df: pd.DataFrame, lookback: int = 250) -> dict:
    """从日线里整理支撑带，并给出优先级和对应策略。"""
    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index)
    work = work.sort_index()
    close = ensure_close(work).dropna()
    work = work.reindex(close.index)
    high = _col(work, "最高", "High").reindex(close.index).fillna(close)
    low = _col(work, "最低", "Low").reindex(close.index).fillna(close)
    volume = _col(work, "成交量", "Volume").reindex(close.index).fillna(0)
    if len(close) < 40:
        raise RuntimeError("日线不足，至少需要约 40 个交易日")

    last = float(close.iloc[-1])
    window = close.tail(lookback)
    high_w = high.reindex(window.index)
    low_w = low.reindex(window.index)
    vol_w = volume.reindex(window.index)
    raw: list[dict] = []

    def add(price: float, kind: str, weight: float) -> None:
        if price is None or pd.isna(price) or price <= 0:
            return
        if price > last * 1.08:
            return
        raw.append({"price": float(price), "kind": kind, "weight": float(weight)})

    for period, weight, name in (
        (20, 2.2, "日MA20"),
        (60, 4.0, "日MA60"),
        (120, 5.2, "日MA120"),
        (250, 6.0, "年线MA250"),
    ):
        if len(close) >= period:
            add(float(close.rolling(period).mean().iloc[-1]), name, weight)

    weekly = pd.DataFrame({"Close": close, "High": high, "Low": low}).resample("W-FRI").agg(
        {"Close": "last", "High": "max", "Low": "min"}
    ).dropna()
    for period, weight, name in ((20, 6.2, "周MA20"), (60, 7.0, "周MA60")):
        if len(weekly) >= period:
            add(float(weekly["Close"].rolling(period).mean().iloc[-1]), name, weight)
    if len(weekly) >= 2:
        prev = weekly.iloc[-2]
        pivot = (float(prev["High"]) + float(prev["Low"]) + float(prev["Close"])) / 3
        add(2 * pivot - float(prev["High"]), "上周枢轴S1", 2.4)
        add(pivot - (float(prev["High"]) - float(prev["Low"])), "上周枢轴S2", 2.8)

    if len(close) >= 20:
        mid = close.rolling(20).mean()
        lower = mid - 2 * close.rolling(20).std()
        add(float(lower.iloc[-1]), "布林下轨", 3.0)

    swings = [item for item in _swing_lows(low_w, span=5) if item[1] <= last * 1.02]
    if swings:
        picked = {item[0]: item for item in sorted(swings, key=lambda item: item[1])[:3]}
        for item in sorted(swings, key=lambda item: item[0])[-4:]:
            picked[item[0]] = item
        for ts, price in picked.values():
            loc = window.index.get_loc(ts)
            age = len(window) - (loc if isinstance(loc, int) else int(loc.start))
            add(price, "前低", 4.4 if age > 20 else 3.2)
        add(min(price for _, price in swings), "区间最低", 4.8)

    hi_idx = high_w.idxmax()
    base_slice = low.loc[:hi_idx].tail(80)
    if not base_slice.empty:
        hi = float(high.loc[hi_idx])
        base = float(base_slice.min())
        span = hi - base
        if span > 0:
            for ratio, name, weight in (
                (0.382, "斐波0.382", 3.0),
                (0.5, "斐波0.5", 3.4),
                (0.618, "斐波0.618", 4.6),
                (0.786, "斐波0.786", 3.6),
            ):
                add(hi - span * ratio, name, weight)
            add(base, "波段起涨低点", 4.2)

    lows = low_w.to_numpy(dtype=float)
    highs = high_w.to_numpy(dtype=float)
    for i in range(1, len(lows)):
        if lows[i] > highs[i - 1] * 1.003:
            floor = float(highs[i - 1])
            later = lows[i + 1 :].min() if i + 1 < len(lows) else floor
            if later >= floor * 0.998:
                add(floor, "向上缺口下沿", 3.6)

    try:
        typical = ((high_w + low_w + window) / 3).to_numpy(dtype=float)
        vols = vol_w.to_numpy(dtype=float)
        if len(typical) >= 30 and float(vols.sum()) > 0:
            bins = pd.cut(typical, bins=24, duplicates="drop")
            grouped = pd.DataFrame({"vol": vols}).groupby(bins, observed=True)["vol"].sum()
            threshold = float(grouped.median()) if not grouped.empty else 0
            for interval, vol in grouped.items():
                mid = float(interval.mid)
                if vol < threshold or mid >= last:
                    continue
                add(mid, "成交密集", 4.5)
    except Exception:
        pass

    channel = _channel(close, high, low)
    if channel:
        add(
            channel["lower"],
            f"{channel['direction']}通道下轨({channel['slope_20']:+.1f}%/20日)",
            4.3,
        )

    if not raw:
        raise RuntimeError("没有算出支撑位")

    raw.sort(key=lambda item: item["price"])
    clusters: list[dict] = []
    for item in raw:
        if clusters and abs(item["price"] - clusters[-1]["price"]) / last <= _MERGE_PCT:
            bucket = clusters[-1]
            bucket["prices"].append(item["price"])
            bucket["kinds"].append(item["kind"])
            bucket["weight"] = max(bucket["weight"], item["weight"]) + 0.7
            bucket["price"] = float(pd.Series(bucket["prices"]).median())
        else:
            clusters.append(
                {
                    "price": item["price"],
                    "prices": [item["price"]],
                    "kinds": [item["kind"]],
                    "weight": item["weight"],
                }
            )

    zones: list[dict] = []
    recent_low = low.tail(120)
    recent_close = close.tail(120)
    for bucket in clusters:
        price = float(bucket["price"])
        dist = (last - price) / last * 100
        if dist < -8:
            continue
        if dist > 20 and bucket["weight"] < 6:
            continue
        if dist <= 1.2 and dist >= -0.4:
            status = "正在测试"
            prox = 1.45
        elif dist < -0.4:
            status = "已跌破"
            prox = 0.4
        elif dist <= 4:
            status = "临近"
            prox = 1.25
        elif dist <= 8:
            status = "下方"
            prox = 1.0
        elif dist <= 15:
            status = "下方"
            prox = 0.72
        else:
            status = "下方"
            prox = 0.45
        touches = _touch_count(recent_low, recent_close, price)
        kinds = list(dict.fromkeys(bucket["kinds"]))
        kinds = [k for k in kinds if "通道" in k] + [k for k in kinds if "通道" not in k]
        score = (bucket["weight"] + min(touches, 4) * 1.15) * prox
        zones.append(
            {
                "支撑价": round(price, 3),
                "距现价%": round(dist, 2),
                "状态": status,
                "构成": " + ".join(kinds[:4]),
                "测试次数": touches,
                "策略": _strategy(status, " + ".join(kinds), dist, touches),
                "_score": score,
            }
        )

    zones.sort(key=lambda item: (item["状态"] == "已跌破", abs(item["距现价%"]), -item["_score"]))
    zones = zones[:8]
    for i, zone in enumerate(zones, start=1):
        zone["顺位"] = i
        zone["优先级"] = _priority(zone["_score"], zone["状态"], i, zone["距现价%"])
        zone.pop("_score", None)

    chart = close.tail(160)
    ma20 = close.rolling(20).mean().reindex(chart.index)
    ma60 = close.rolling(60).mean().reindex(chart.index)
    return {
        "close": last,
        "asof": close.index[-1].strftime("%Y-%m-%d"),
        "summary": _position_summary(last, zones),
        "zones": zones,
        "chart_close": chart,
        "chart_ma20": ma20,
        "chart_ma60": ma60,
        "channel": channel,
    }


def _swing_highs(high: pd.Series, span: int = 5) -> list[tuple[pd.Timestamp, float]]:
    values = high.to_numpy(dtype=float)
    out: list[tuple[pd.Timestamp, float]] = []
    for i in range(span, len(values) - span):
        window = values[i - span : i + span + 1]
        if values[i] >= window.max() and values[i] > values[i - 1] and values[i] >= values[i + 1]:
            out.append((high.index[i], float(values[i])))
    return out


def _reject_count(high: pd.Series, close: pd.Series, price: float, band: float = 0.006) -> int:
    highs = high.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    rejects = 0
    i = 0
    last = len(highs) - 1
    while i < last:
        if price * (1 - band) <= highs[i] <= price * (1 + band):
            rejected = closes[i] <= price * 1.002
            if not rejected and i + 1 < len(closes):
                rejected = closes[i + 1] < price
            if rejected:
                rejects += 1
            i += 4
        else:
            i += 1
    return rejects


def _resist_strategy(status: str, kinds: str, dist: float, touches: int) -> str:
    if "通道上轨" in kinds:
        piled = "这一档还和别的压力叠在一起。" if " + " in kinds else ""
        if status == "已突破":
            return "通道上轨已经被收盘站上，这条斜向压力失效。跌回上轨附近改看支撑。" + piled
        if "下降" in kinds:
            return "这是下降通道的上轨。反弹到这里更容易被压回去，收盘站上才算脱离这段下降通道。" + piled
        if "上升" in kinds:
            return "这是上升通道的上轨。顺着通道可以靠近，但到上轨不追；站上上轨，通道就往上打开了。" + piled
        return "这是横盘通道的上轨。靠近不追，收盘站上才离开这个通道。" + piled
    weekly = "周" in kinds or "年线" in kinds or "MA250" in kinds or "MA120" in kinds
    dense = "成交密集" in kinds
    gap = "缺口" in kinds
    tested = touches >= 2
    if status == "已突破":
        return "收盘已经站上这一档，不再当压力。跌回来时改看支撑，站不回去才重新变成压力。"
    if status == "正在测试":
        text = "现价顶在这一档上。收盘站上才算过压；冲高回落就先减，不追。"
        if tested:
            text += "这里以前压过，第一次靠近比连续冲破更要小心。"
        if dense:
            text += "上方成交堆着，过压往往要来回几次。"
        return text
    if status == "临近":
        text = f"上面大约 {dist:.1f}%。没到之前不追高，靠近再看量。"
        if gap:
            text += "这是向下缺口的上沿，缺口补完之前都当压力。"
        elif tested:
            text += "前高附近容易有套牢盘，放量过不去就当压力有效。"
        else:
            text += "到了看收盘能不能站上，盘中刚碰不算过压。"
        if weekly:
            text += "这是中线压力，短线冲到这里更容易歇一歇。"
        return text
    text = f"还在现价上方约 {dist:.1f}%，是更远的一档。"
    if weekly:
        text += "近处的压力先过，再把这一档当成中线目标。"
    else:
        text += "近处压力没过之前，不用提前按这一档来做。"
    return text


def _resist_summary(last: float, zones: list[dict]) -> str:
    valid = [z for z in zones if z["状态"] != "已突破"]
    broken = [z for z in zones if z["状态"] == "已突破" and z["距现价%"] > -4]
    if not valid:
        return f"现价 {last:.3f} 上方没有聚出有效压力，涨起来近处缺少明显的压档。"
    first = valid[0]
    second = valid[1] if len(valid) > 1 else None
    parts = [
        f"现价 {last:.3f}。优先看 {first['压力价']:.3f}（{first['构成']}，{first['状态']}，距现价 {first['距现价%']:.1f}%）。"
    ]
    if first["状态"] == "正在测试":
        parts.append("价格已经顶在第一压力上，这一档看收盘能不能站上去。")
    elif first["距现价%"] > 8:
        parts.append("近处没有压力，中间空间比较空，真涨起来会直接去找这一档。")
    elif second and abs(first["压力价"] - second["压力价"]) / last <= 0.03:
        parts.append(
            f"下一档 {second['压力价']:.3f} 离得很近。过了第一档也容易卡在下一档，不要把两档之间当成已经畅通。"
        )
    else:
        parts.append("第一档和第二档拉开了，先看最近这一档，过了再看上面。")
    if broken:
        top = min(broken, key=lambda item: abs(item["距现价%"]))
        parts.append(f"{top['压力价']:.3f} 已经在现价下方，角色改成支撑，跌破才失效。")
    return "".join(parts)


def analyze_resistances(df: pd.DataFrame, lookback: int = 250) -> dict:
    """从日线里整理压力带，并给出优先级和对应策略。"""
    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index)
    work = work.sort_index()
    close = ensure_close(work).dropna()
    work = work.reindex(close.index)
    high = _col(work, "最高", "High").reindex(close.index).fillna(close)
    low = _col(work, "最低", "Low").reindex(close.index).fillna(close)
    volume = _col(work, "成交量", "Volume").reindex(close.index).fillna(0)
    if len(close) < 40:
        raise RuntimeError("日线不足，至少需要约 40 个交易日")

    last = float(close.iloc[-1])
    window = close.tail(lookback)
    high_w = high.reindex(window.index)
    low_w = low.reindex(window.index)
    vol_w = volume.reindex(window.index)
    raw: list[dict] = []

    def add(price: float, kind: str, weight: float) -> None:
        if price is None or pd.isna(price) or price <= 0:
            return
        if price < last * 0.92:
            return
        raw.append({"price": float(price), "kind": kind, "weight": float(weight)})

    for period, weight, name in (
        (20, 2.2, "日MA20"),
        (60, 4.0, "日MA60"),
        (120, 5.2, "日MA120"),
        (250, 6.0, "年线MA250"),
    ):
        if len(close) >= period:
            add(float(close.rolling(period).mean().iloc[-1]), name, weight)

    weekly = pd.DataFrame({"Close": close, "High": high, "Low": low}).resample("W-FRI").agg(
        {"Close": "last", "High": "max", "Low": "min"}
    ).dropna()
    for period, weight, name in ((20, 6.2, "周MA20"), (60, 7.0, "周MA60")):
        if len(weekly) >= period:
            add(float(weekly["Close"].rolling(period).mean().iloc[-1]), name, weight)
    if len(weekly) >= 2:
        prev = weekly.iloc[-2]
        pivot = (float(prev["High"]) + float(prev["Low"]) + float(prev["Close"])) / 3
        add(2 * pivot - float(prev["Low"]), "上周枢轴R1", 2.4)
        add(pivot + (float(prev["High"]) - float(prev["Low"])), "上周枢轴R2", 2.8)

    if len(close) >= 20:
        mid = close.rolling(20).mean()
        upper = mid + 2 * close.rolling(20).std()
        add(float(upper.iloc[-1]), "布林上轨", 3.0)

    swings = [item for item in _swing_highs(high_w, span=5) if item[1] >= last * 0.98]
    if swings:
        picked = {item[0]: item for item in sorted(swings, key=lambda item: item[1], reverse=True)[:3]}
        for item in sorted(swings, key=lambda item: item[0])[-4:]:
            picked[item[0]] = item
        for ts, price in picked.values():
            loc = window.index.get_loc(ts)
            age = len(window) - (loc if isinstance(loc, int) else int(loc.start))
            add(price, "前高", 4.4 if age > 20 else 3.2)
        add(max(price for _, price in swings), "区间最高", 4.8)

    lo_idx = low_w.idxmin()
    peak_slice = high.loc[:lo_idx].tail(80)
    if not peak_slice.empty:
        lo = float(low.loc[lo_idx])
        peak = float(peak_slice.max())
        span = peak - lo
        if span > 0:
            for ratio, name, weight in (
                (0.382, "斐波0.382", 3.0),
                (0.5, "斐波0.5", 3.4),
                (0.618, "斐波0.618", 4.6),
                (0.786, "斐波0.786", 3.6),
            ):
                add(lo + span * ratio, name, weight)
            add(peak, "波段高点", 4.2)

    lows = low_w.to_numpy(dtype=float)
    highs = high_w.to_numpy(dtype=float)
    for i in range(1, len(highs)):
        if highs[i] < lows[i - 1] * 0.997:
            ceiling = float(lows[i - 1])
            later = highs[i + 1 :].max() if i + 1 < len(highs) else ceiling
            if later <= ceiling * 1.002:
                add(ceiling, "向下缺口上沿", 3.6)

    try:
        typical = ((high_w + low_w + window) / 3).to_numpy(dtype=float)
        vols = vol_w.to_numpy(dtype=float)
        if len(typical) >= 30 and float(vols.sum()) > 0:
            bins = pd.cut(typical, bins=24, duplicates="drop")
            grouped = pd.DataFrame({"vol": vols}).groupby(bins, observed=True)["vol"].sum()
            threshold = float(grouped.median()) if not grouped.empty else 0
            for interval, vol in grouped.items():
                mid = float(interval.mid)
                if vol < threshold or mid <= last:
                    continue
                add(mid, "成交密集", 4.5)
    except Exception:
        pass

    channel = _channel(close, high, low)
    if channel:
        add(
            channel["upper"],
            f"{channel['direction']}通道上轨({channel['slope_20']:+.1f}%/20日)",
            4.3,
        )

    if not raw:
        raise RuntimeError("没有算出压力位")

    raw.sort(key=lambda item: item["price"])
    clusters: list[dict] = []
    for item in raw:
        if clusters and abs(item["price"] - clusters[-1]["price"]) / last <= _MERGE_PCT:
            bucket = clusters[-1]
            bucket["prices"].append(item["price"])
            bucket["kinds"].append(item["kind"])
            bucket["weight"] = max(bucket["weight"], item["weight"]) + 0.7
            bucket["price"] = float(pd.Series(bucket["prices"]).median())
        else:
            clusters.append(
                {
                    "price": item["price"],
                    "prices": [item["price"]],
                    "kinds": [item["kind"]],
                    "weight": item["weight"],
                }
            )

    zones: list[dict] = []
    recent_high = high.tail(120)
    recent_close = close.tail(120)
    for bucket in clusters:
        price = float(bucket["price"])
        dist = (price - last) / last * 100
        if dist < -8:
            continue
        if dist > 20 and bucket["weight"] < 6:
            continue
        if -0.4 <= dist <= 1.2:
            status = "正在测试"
            prox = 1.45
        elif dist < -0.4:
            status = "已突破"
            prox = 0.4
        elif dist <= 4:
            status = "临近"
            prox = 1.25
        elif dist <= 8:
            status = "上方"
            prox = 1.0
        elif dist <= 15:
            status = "上方"
            prox = 0.72
        else:
            status = "上方"
            prox = 0.45
        touches = _reject_count(recent_high, recent_close, price)
        kinds = list(dict.fromkeys(bucket["kinds"]))
        kinds = [k for k in kinds if "通道" in k] + [k for k in kinds if "通道" not in k]
        score = (bucket["weight"] + min(touches, 4) * 1.15) * prox
        zones.append(
            {
                "压力价": round(price, 3),
                "距现价%": round(dist, 2),
                "状态": status,
                "构成": " + ".join(kinds[:4]),
                "测试次数": touches,
                "策略": _resist_strategy(status, " + ".join(kinds), dist, touches),
                "_score": score,
            }
        )

    zones.sort(key=lambda item: (item["状态"] == "已突破", abs(item["距现价%"]), -item["_score"]))
    zones = zones[:8]
    for i, zone in enumerate(zones, start=1):
        zone["顺位"] = i
        zone["优先级"] = _priority(zone["_score"], zone["状态"], i, zone["距现价%"])
        zone.pop("_score", None)

    chart = close.tail(160)
    return {
        "close": last,
        "asof": close.index[-1].strftime("%Y-%m-%d"),
        "summary": _resist_summary(last, zones),
        "zones": zones,
        "chart_close": chart,
        "chart_ma20": close.rolling(20).mean().reindex(chart.index),
        "chart_ma60": close.rolling(60).mean().reindex(chart.index),
        "channel": channel,
    }
