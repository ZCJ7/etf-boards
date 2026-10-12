# Top5 动量轮动

## 怎么跑

在 `streametf` 目录：

```powershell
.\.venv\Scripts\python.exe scripts\run_momentum_top5.py
```

输出：
- 当前 Top5 清单
- 四策略回测对比表
- 精简分析
- `data/momentum_top5/nav.png`、`drawdown.png`、`compare.csv`

本机 Streamlit：打开 **Top5动量** 分区，点「计算当前 Top5 并回测」。

手机页：Run workflow 会一并刷新；也可单独跑：

```powershell
.\.venv\Scripts\python.exe scripts\refresh_momentum_page.py
```

## 参数

全部在 `analysis/momentum_top5.py` 文件开头：`LOOKBACK`、`MA_PERIOD`、成交额门槛、单边成本、观察池 `ETF_UNIVERSE`。

## 成交假设

- 信号：每月最后一个**完整**交易日收盘后计算
- 成交：下一交易日开盘价；开盘相对昨收涨跌幅 ≥ 9.5% 视为难成交（不买卖）
- 不足 5 只时只买入选标的，剩余现金
- 禁止用信号日收盘价成交
- 成交额优先用行情字段；缺失时用 `成交量(手) × 收盘 × 100` 近似
- 若东财不可用而落到新浪，历史可能只有约四年，回测会短于 2019 起点，页面分析里会写明
