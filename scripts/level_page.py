"""手机页的支撑位 / 压力位：全部场内 ETF。"""

from __future__ import annotations

import json

import pandas as pd

from analysis.screener import load_etf_universe
from analysis.support_levels import analyze_resistances, analyze_supports
from etf_data_fetcher import fetch_pool_daily
from utils import clean_etf_symbol


def board_codes(boards: dict) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for key in ("low_position", "reversal", "momentum"):
        frame = boards.get(key)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        for col in ("代码", "强度龙头代码"):
            if col not in frame.columns:
                continue
            for raw in frame[col].tolist():
                code = clean_etf_symbol(raw)
                if code and code not in seen and code.isdigit():
                    seen.add(code)
                    found.append(code)
    return found


def level_catalog(boards: dict, extra: int | None = None) -> list[dict]:
    """榜单 ETF 全部保留。extra 为空时再补上全市场其余 ETF。"""
    spot = load_etf_universe()
    names: dict[str, str] = {}
    sectors: dict[str, str] = {}
    ranked: list[str] = []
    if spot is not None and not spot.empty:
        ordered = spot.sort_values("成交额", ascending=False)
        for _, row in ordered.iterrows():
            code = clean_etf_symbol(row["代码"])
            if not code:
                continue
            names[code] = str(row.get("名称") or "")
            sectors[code] = str(row.get("板块") or "")
            ranked.append(code)
        for key in ("low_position", "reversal", "momentum"):
            frame = boards.get(key)
            if not isinstance(frame, pd.DataFrame) or frame.empty or "代码" not in frame.columns:
                continue
            name_col = "名称" if "名称" in frame.columns else None
            sector_col = "板块" if "板块" in frame.columns else None
            for _, row in frame.iterrows():
                code = clean_etf_symbol(row["代码"])
                if code and name_col and not names.get(code):
                    names[code] = str(row.get(name_col) or "")
                if code and sector_col and not sectors.get(code):
                    sectors[code] = str(row.get(sector_col) or "")

    chosen = board_codes(boards)
    on_board = set(chosen)
    seen = set(chosen)
    added = 0
    for code in ranked:
        if code in seen:
            continue
        chosen.append(code)
        seen.add(code)
        added += 1
        if extra is not None and added >= extra:
            break

    if not chosen:
        return []
    print(f"    名单 {len(chosen)} 只", flush=True)
    daily = fetch_pool_daily(chosen, tail_days=520, max_workers=8, pause_sec=0)
    items: list[dict] = []
    for code in chosen:
        frame = daily.get(code)
        if frame is None or frame.empty:
            continue
        try:
            support = analyze_supports(frame, lookback=250)
        except Exception:
            support = None
        try:
            resist = analyze_resistances(frame, lookback=250)
        except Exception:
            resist = None
        if support is None and resist is None:
            continue
        base = support or resist
        spark = base["chart_close"].tail(60)
        channel = _channel_payload(base.get("channel"), spark.index)
        items.append(
            {
                "code": code,
                "name": names.get(code, ""),
                "sector": sectors.get(code, ""),
                "close": base["close"],
                "asof": base["asof"],
                "on_board": code in on_board,
                "support": {"summary": (support or {}).get("summary", ""), "zones": (support or {}).get("zones", [])},
                "resist": {"summary": (resist or {}).get("summary", ""), "zones": (resist or {}).get("zones", [])},
                "spark": [round(float(v), 4) for v in spark.tolist()],
                "channel": channel,
            }
        )
    return items


def _channel_payload(channel: dict | None, index: pd.Index) -> dict | None:
    if not channel:
        return None
    lower = channel["lower_line"].reindex(index)
    upper = channel["upper_line"].reindex(index)
    if int(lower.notna().sum()) < 2 or int(upper.notna().sum()) < 2:
        return None

    def _nums(series: pd.Series) -> list[float | None]:
        return [None if pd.isna(v) else round(float(v), 4) for v in series.tolist()]

    return {
        "direction": channel["direction"],
        "slope_20": round(float(channel["slope_20"]), 1),
        "lower": _nums(lower),
        "upper": _nums(upper),
    }


def levels_payload(items: list[dict]) -> str:
    return json.dumps(items, ensure_ascii=False)


def level_sections(items: list[dict]) -> tuple[str, str, str]:
    note = (
        f"可以搜全部场内 ETF，当前 {len(items)} 只。"
        "图里的斜线是近60日通道，绿线是下轨，橙线是上轨。"
    )
    support = _panel(
        "support",
        "支撑位",
        "按代码、名称或板块搜索。绿线仍是支撑，橙线是已经跌破、改当压力的位置。" + note,
    )
    resist = _panel(
        "resist",
        "压力位",
        "按代码、名称或板块搜索。橙线仍是压力，绿线是已经站上、改当支撑的位置。" + note,
    )
    return support, resist, _SCRIPT


def _panel(kind: str, title: str, note: str) -> str:
    return (
        f"<section id='{kind}'>"
        f"<h2>{title}</h2>"
        f"<p class='note'>{note}</p>"
        f"<input id='{kind}Q' class='search' type='search' placeholder='红利、有色、创业板、510300' enterkeyhint='search' />"
        f"<div id='{kind}Hits'></div>"
        f"<div id='{kind}Detail'></div>"
        "</section>"
    )


_SCRIPT = r"""
<script>
(function() {
  var data = [];
  var sides = {
    support: {box: "support", price: "支撑价", label: "优先支撑", spent: "已跌破", alive: "#0f766e", dead: "#b45309"},
    resist: {box: "resist", price: "压力价", label: "优先压力", spent: "已突破", alive: "#b45309", dead: "#0f766e"}
  };
  function esc(text) {
    return String(text || "").replace(/[&<>"]/g, function(ch) {
      return {"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;"}[ch];
    });
  }
  function metric(label, value) {
    return "<div class='metric'><span>" + esc(label) + "</span><b>" + esc(String(value)) + "</b></div>";
  }
  function yOf(v, min, max, top, ph) {
    return top + (1 - (v - min) / (max - min)) * ph;
  }
  function rail(vals, ys, min, max, left, top, pw, ph, color, name) {
    if (!vals || vals.length !== ys.length) return "";
    var pts = [];
    vals.forEach(function(v, i) {
      if (v == null || isNaN(v)) return;
      var x = left + (i / (ys.length - 1)) * pw;
      pts.push(x.toFixed(1) + "," + yOf(v, min, max, top, ph).toFixed(1));
    });
    if (pts.length < 2) return "";
    var last = pts[pts.length - 1].split(",");
    return "<polyline fill='none' stroke='" + color + "' stroke-width='2.2' points='" + pts.join(" ") + "'/>"
      + "<text x='" + (Number(last[0]) - 6) + "' y='" + (Number(last[1]) - 4) + "' text-anchor='end' font-size='12' fill='" + color + "'>" + name + "</text>";
  }
  function chart(item, side, zones) {
    var ys = item.spark || [];
    if (ys.length < 2) return "";
    var ch = item.channel || {};
    var extra = (ch.lower || []).concat(ch.upper || []);
    var min = Math.min.apply(null, ys), max = Math.max.apply(null, ys);
    extra.forEach(function(v) {
      if (v == null || isNaN(v)) return;
      if (v < min) min = v;
      if (v > max) max = v;
    });
    zones.forEach(function(z) {
      if (String(z["构成"] || "").indexOf("通道") >= 0 && String(z["构成"]).indexOf("+") < 0) return;
      var y = Number(z[side.price]);
      if (y < min) min = y;
      if (y > max) max = y;
    });
    if (min === max) { min -= 1; max += 1; }
    var pad = (max - min) * 0.08;
    min -= pad; max += pad;
    var w = 720, h = 220, left = 8, top = 12, pw = 704, ph = 196;
    var pts = ys.map(function(v, i) {
      var x = left + (i / (ys.length - 1)) * pw;
      return x.toFixed(1) + "," + yOf(v, min, max, top, ph).toFixed(1);
    }).join(" ");
    var lines = zones.map(function(z) {
      if (String(z["构成"] || "").indexOf("通道") >= 0 && String(z["构成"]).indexOf("+") < 0) return "";
      var y = yOf(Number(z[side.price]), min, max, top, ph);
      var alive = z["状态"] !== side.spent;
      var color = alive ? side.alive : side.dead;
      var dash = alive ? "" : " stroke-dasharray='4 3'";
      return "<line x1='" + left + "' y1='" + y.toFixed(1) + "' x2='" + (left + pw) + "' y2='" + y.toFixed(1) + "' stroke='" + color + "' stroke-width='1.2'" + dash + "/>";
    }).join("");
    var rails = rail(ch.lower, ys, min, max, left, top, pw, ph, "#0f766e", (ch.direction || "") + "通道下轨")
      + rail(ch.upper, ys, min, max, left, top, pw, ph, "#b45309", (ch.direction || "") + "通道上轨");
    return "<div class='chartbox'><svg viewBox='0 0 " + w + " " + h + "' class='chart'><rect x='" + left + "' y='" + top + "' width='" + pw + "' height='" + ph + "' class='plot'/>" + lines + rails + "<polyline fill='none' stroke='#0f172a' stroke-width='2' points='" + pts + "'/></svg></div>";
  }
  function render(item, side) {
    var pack = item[side.box] || {summary: "", zones: []};
    var zones = pack.zones || [];
    var active = null;
    zones.some(function(z) { if (z["状态"] !== side.spent) { active = z; return true; } return false; });
    var rows = zones.map(function(z) {
      return "<tr><td>" + z["顺位"] + "</td><td>" + esc(z["优先级"]) + "</td><td>" + esc(z["状态"])
        + "</td><td>" + Number(z[side.price]).toFixed(3) + "</td><td>" + Number(z["距现价%"]).toFixed(2)
        + "%</td><td>" + z["测试次数"] + "</td><td>" + esc(z["构成"]) + "</td><td>" + esc(z["策略"]) + "</td></tr>";
    }).join("");
    document.getElementById(side.box + "Detail").innerHTML = "<h3>" + esc(item.code + " " + item.name) + "</h3>"
      + "<p class='note'>行情截止 " + esc(item.asof) + (item.on_board ? " · 在当前榜单里" : "") + "</p>"
      + "<div class='metrics'>"
      + metric("现价", Number(item.close).toFixed(3))
      + metric(side.label, active ? Number(active[side.price]).toFixed(3) : "—")
      + metric("距现价", active ? Number(active["距现价%"]).toFixed(1) + "%" : "—")
      + metric("优先级", active ? active["优先级"] : "—")
      + "</div><p>" + esc(pack.summary) + "</p>"
      + chart(item, side, zones)
      + "<div class='wrap'><table><thead><tr><th>顺位</th><th>优先级</th><th>状态</th><th>" + esc(side.price) + "</th><th>距现价%</th><th>测试次数</th><th>构成</th><th>策略</th></tr></thead><tbody>"
      + rows + "</tbody></table></div>";
  }
  function bind(kind) {
    var side = sides[kind];
    var input = document.getElementById(kind + "Q");
    var list = document.getElementById(kind + "Hits");
    function show() {
      var q = (input.value || "").trim().toLowerCase();
      list.innerHTML = "";
      document.getElementById(kind + "Detail").innerHTML = "";
      if (!q) {
        list.innerHTML = "<p class='note'>输入代码、名称或板块，例如 红利、有色、510300。名单是全部场内 ETF。</p>";
        return;
      }
      var hits = data.filter(function(item) {
        return (item.code + " " + item.name + " " + item.sector).toLowerCase().indexOf(q) >= 0;
      }).slice(0, 30);
      if (!hits.length) {
        list.innerHTML = "<p class='note'>没有匹配的 ETF。当前名单 " + data.length + " 只。</p>";
        return;
      }
      hits.forEach(function(item) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "hit";
        btn.textContent = item.code + " " + item.name + (item.sector ? " · " + item.sector : "") + (item.on_board ? " · 榜单" : "");
        btn.addEventListener("click", function() { render(item, side); });
        list.appendChild(btn);
      });
    }
    input.addEventListener("input", show);
    show();
  }
  fetch("levels.json").then(function(r) { return r.json(); }).then(function(rows) {
    data = rows || [];
    bind("support");
    bind("resist");
  }).catch(function() {
    ["support", "resist"].forEach(function(kind) {
      var list = document.getElementById(kind + "Hits");
      if (list) list.innerHTML = "<p class='note'>名单加载失败，请刷新页面。</p>";
    });
  });
  document.querySelectorAll(".tab").forEach(function(btn) {
    btn.addEventListener("click", function() {
      document.querySelectorAll(".tab").forEach(function(other) { other.classList.toggle("on", other === btn); });
      document.querySelectorAll(".panel").forEach(function(panel) { panel.hidden = panel.id !== "panel-" + btn.dataset.tab; });
    });
  });
})();
</script>
"""
