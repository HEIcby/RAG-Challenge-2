"""金融/财报关键概念词汇表，用于Multi-Query等提示词增强。"""

from typing import Any, Dict, List

FINANCIAL_CONCEPTS: List[Dict[str, Any]] = [
    {
        "term": "可自由支配资金",
        "definition": "扣除受限、专项用途及监管占用后，企业在账上可以灵活调度的资金余额，反映短期资金安全垫。",
        "aliases": ["可支配资金", "自由资金"],
        "formula": "可自由支配资金 ≈ 货币资金 + 交易性金融资产（或近现金资产） − 受限 / 专项 / 监管占用资金"
    },
    {
        "term": "归母净利润",
        "definition": "归属于上市公司股东的净利润，剔除少数股东损益后衡量股东可享有的盈利水平。",
        "aliases": ["归属于母公司净利润", "归母利润"],
        "formula": "归母净利润 = 净利润 − 少数股东损益"
    },
    {
        "term": "经营活动现金流量净额",
        "definition": "企业主营业务现金流入减现金流出的净额，衡量经营造血能力，常与净利润对比判断盈利质量。",
        "aliases": ["经营现金流", "经营活动现金流"],
        "formula": "经营活动现金流量净额 = 经营活动现金流入合计 − 经营活动现金流出合计"
    },
    {
        "term": "毛利率",
        "definition": "毛利与营业收入之比，反映产品或业务的盈利空间，受产品结构、成本控制影响显著。",
        "aliases": ["综合毛利率"],
        "formula": "毛利率 = (营业收入 − 营业成本) ÷ 营业收入"
    },
    {
        "term": "在手订单",
        "definition": "尚未交付的订单金额，体现未来已锁定的收入规模，可拆解为内销与外销结构。",
        "aliases": ["订单储备", "订单储量"],
        "formula": "在手订单 = 已签约订单总额 − 已确认收入部分"
    },
    {
        "term": "研发投入占比",
        "definition": "研发费用或研发投入金额占营业收入比重，衡量公司技术投入强度，也可关注资本化比例。",
        "aliases": ["研发费用率", "研发投入比例"],
        "formula": "研发投入占比 = 研发投入金额 ÷ 营业收入"
    },
    {
        "term": "现金分红",
        "definition": "公司按每10股派发的现金金额，体现股东回报政策，需要结合年份与是否含税说明。",
        "aliases": ["每10股派息", "分红金额"],
        "formula": "现金分红总额 = 每股派现 × 流通股本；常按“每10股派X元”表述"
    },
    {
        "term": "政府补助",
        "definition": "当期计入损益的各类财政补助，对利润有一次性影响，需关注金额和持续性。",
        "aliases": ["财政补贴", "补助收入"],
        "formula": "当期计入损益的政府补助 = 各项财政补贴入账金额（区分计入营业外收入或冲减成本）"
    },
    {
        "term": "可再生能源/储能业务收入",
        "definition": "储能系列、AI数据中心等新业务的收入规模与增长率，常伴随订单或产能信息。",
        "aliases": ["储能收入", "数据中心收入"],
        "formula": "业务收入 = 该业务确认的营业收入；同比增速 = (本期收入 − 上期收入) ÷ 上期收入"
    },
    {
        "term": "在建或海外基地扩张",
        "definition": "海外工厂（如墨西哥、美国、波兰等）及上下游扩产计划，对产能与交付半径有重要影响。",
        "aliases": ["海外基地", "全球化产能"],
        "formula": "通常结合“设计产能 + 投产进度 + 资本开支”描述，非单一公式，可通过产能(台/套)或投资额衡量"
    }
]


def find_financial_concepts(question: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    匹配问题中出现的金融术语，若无明显匹配则返回前若干常用概念。
    """
    if not question:
        return FINANCIAL_CONCEPTS[:limit]
    
    normalized_question = question.lower()
    matches: List[Dict[str, Any]] = []
    
    for concept in FINANCIAL_CONCEPTS:
        keywords = [concept["term"]] + concept.get("aliases", [])
        if any(keyword.lower() in normalized_question for keyword in keywords):
            matches.append(concept)
            if len(matches) >= limit:
                break
    
    if not matches:
        return FINANCIAL_CONCEPTS[:limit]
    
    return matches


def format_concepts_for_prompt(concepts: List[Dict[str, Any]]) -> str:
    """
    将概念列表格式化为提示词可读的字符串。
    """
    lines = []
    for concept in concepts:
        aliases = concept.get("aliases", [])
        alias_text = f"（别名：{'、'.join(aliases)}）" if aliases else ""
        formula_text = ""
        if concept.get("formula"):
            formula_text = f"\n  · 计算方式：{concept['formula']}"
        lines.append(f"- {concept['term']}{alias_text}：{concept['definition']}{formula_text}")
    return "\n".join(lines)

