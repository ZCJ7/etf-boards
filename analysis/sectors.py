"""ETF 板块归类（用于比价线分子去重）。"""

from __future__ import annotations

import re

# 顺序优先：先匹配更具体的板块
SECTOR_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("半导体设备", re.compile(r"半导体设备|半导体材料|半导体产业", re.I)),
    ("半导体", re.compile(r"半导体|芯片|集成电路", re.I)),
    ("光伏", re.compile(r"光伏|太阳能", re.I)),
    ("锂电池", re.compile(r"锂电|电池ETF|电池$", re.I)),
    ("储能", re.compile(r"储能", re.I)),
    ("新能源", re.compile(r"新能源|碳中和|绿色能源", re.I)),
    ("创新药", re.compile(r"创新药|生物科技|生物医药", re.I)),
    ("医疗器械", re.compile(r"医疗器械|医疗设备", re.I)),
    ("医药", re.compile(r"医药|医疗|健康|疫苗", re.I)),
    ("证券", re.compile(r"证券|券商", re.I)),
    ("银行", re.compile(r"银行", re.I)),
    ("保险", re.compile(r"保险", re.I)),
    ("金融", re.compile(r"金融", re.I)),
    ("通信设备", re.compile(r"通信|5G|光模块", re.I)),
    ("卫星", re.compile(r"卫星|航天", re.I)),
    ("电网设备", re.compile(r"电网|电力设备|特高压", re.I)),
    ("电力", re.compile(r"电力|火电|水电", re.I)),
    ("机床制造", re.compile(r"机床|工业母机", re.I)),
    ("智能制造", re.compile(r"智能制造|高端制造|制造ETF", re.I)),
    ("有色金属", re.compile(r"有色|矿业", re.I)),
    ("稀土", re.compile(r"稀土", re.I)),
    ("钢铁", re.compile(r"钢铁", re.I)),
    ("煤炭", re.compile(r"煤炭|焦煤", re.I)),
    ("化工", re.compile(r"化工|化学", re.I)),
    ("军工", re.compile(r"军工|国防|航空", re.I)),
    ("智能驾驶", re.compile(r"智能驾驶|车联网|汽车ETF", re.I)),
    ("汽车", re.compile(r"汽车", re.I)),
    ("人工智能", re.compile(r"人工智能|AIETF|AI主题", re.I)),
    ("机器人", re.compile(r"机器人", re.I)),
    ("软件", re.compile(r"软件|信创", re.I)),
    ("云计算", re.compile(r"云计算|大数据|数据ETF", re.I)),
    ("消费电子", re.compile(r"消费电子|电子ETF", re.I)),
    ("白酒", re.compile(r"白酒|酒ETF", re.I)),
    ("消费", re.compile(r"消费|免税|零售", re.I)),
    ("家电", re.compile(r"家电", re.I)),
    ("旅游", re.compile(r"旅游|酒店", re.I)),
    ("食品", re.compile(r"食品|饮料|养殖|畜牧|猪", re.I)),
    ("农业", re.compile(r"农业|种业", re.I)),
    ("地产", re.compile(r"地产|房地产", re.I)),
    ("基建", re.compile(r"基建|建筑|建材", re.I)),
    ("传媒", re.compile(r"传媒|影视", re.I)),
    ("游戏", re.compile(r"游戏", re.I)),
    ("环保", re.compile(r"环保|低碳", re.I)),
    ("航运", re.compile(r"航运|港口|物流", re.I)),
    ("石油", re.compile(r"石油|油气", re.I)),
    ("央企", re.compile(r"央企|国企", re.I)),
    ("红利", re.compile(r"红利|股息", re.I)),
    ("港股", re.compile(r"恒生|港股|H股", re.I)),
    ("科创", re.compile(r"科创", re.I)),
]

_ETF_SUFFIX = re.compile(r"(ETF|LOF|基金|指数|主题|产业|行业).*$", re.I)


def _name_theme(name: str) -> str:
    """从基金简称提取主题，用于未命中规则时的细分板块。"""
    text = _ETF_SUFFIX.sub("", str(name or "")).strip()
    text = re.sub(r"[A-Za-z0-9]+", "", text).strip()
    return text[:10] if text else "其他"


def infer_sector(name: str) -> str:
    text = str(name or "")
    for sector, pattern in SECTOR_RULES:
        if pattern.search(text):
            return sector
    theme = _name_theme(text)
    return theme if theme != "其他" else "综合"


def sector_label(symbol: str, name: str) -> str:
    return infer_sector(name or symbol)
