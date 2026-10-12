"""命令行运行 Top5 动量轮动：打印清单、回测表，并保存净值/回撤图。

用法（在 streametf 目录）:
  .\\.venv\\Scripts\\python.exe scripts\\run_momentum_top5.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("ETF_USE_LIVE_BAR", "0")

from analysis.momentum_top5 import run_all  # noqa: E402


def _pct(x):
    if x is None:
        return "—"
    try:
        if x != x:
            return "—"
    except Exception:
        return "—"
    return f"{x * 100:.2f}%"


def _num(x):
    if x is None:
        return "—"
    try:
        if x != x:
            return "—"
    except Exception:
        return "—"
    return f"{float(x):.2f}"


def main() -> None:
    print("==> 拉取观察池并计算 Top5 / 回测", flush=True)
    result = run_all()
    top = result["top"]
    bt = result["backtest"]
    if top.get("error"):
        print("Top5 失败:", top["error"])
    if bt.get("error"):
        print("回测失败:", bt["error"])
        for issue in bt.get("issues") or []:
            print(" -", issue)
        raise SystemExit(1)

    print("\n=== 当前 Top5 ===")
    print(f"信号/数据截止: {top['signal_date']} / {top['asof']}")
    print(f"计划调仓: {top['plan_rebalance']}")
    print(f"符合条件: {top['eligible_count']} 只")
    for row in top["rows"]:
        print(
            f"{row['排名']}. {row['代码']} {row['ETF名称']}  "
            f"score={row['动量评分']}  近20日={row['近20日收益率']}%"
        )
    if not top["rows"]:
        print("（当前没有符合筛选条件的 ETF，视为空仓）")

    print(f"\n=== 回测 {bt['start']} ~ {bt['end']} ===")
    compare = bt["compare"].copy()
    for col in ["累计收益率", "年化收益率CAGR", "最大回撤", "年化波动率"]:
        compare[col] = compare[col].map(_pct)
    for col in ["夏普比率", "卡玛比率"]:
        compare[col] = compare[col].map(_num)
    compare["交易成本"] = compare["交易成本"].map(lambda x: f"{float(x):,.0f}")
    print(compare.to_string(index=False))

    print("\n=== 精简分析 ===")
    print(bt["analysis"])

    out_dir = os.path.join(ROOT, "data", "momentum_top5")
    os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        navs = bt["navs"]
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, series in navs.items():
            ax.plot(series.index, series.values, label=name, linewidth=1.6)
        ax.legend()
        ax.set_title("累计净值")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        nav_path = os.path.join(out_dir, "nav.png")
        fig.savefig(nav_path, dpi=140)
        plt.close(fig)

        dd = bt["top5"]["metrics"].get("回撤序列")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.fill_between(dd.index, dd.values, 0, color="#b45309", alpha=0.35)
        ax.plot(dd.index, dd.values, color="#b45309", linewidth=1.2)
        ax.set_title("Top5 回撤")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        dd_path = os.path.join(out_dir, "drawdown.png")
        fig.savefig(dd_path, dpi=140)
        plt.close(fig)
        print(f"\n图已保存: {nav_path}")
        print(f"图已保存: {dd_path}")
    except Exception as exc:
        print("画图失败:", exc)

    compare.to_csv(os.path.join(out_dir, "compare.csv"), index=False, encoding="utf-8-sig")
    print("表已保存:", os.path.join(out_dir, "compare.csv"))


if __name__ == "__main__":
    main()
