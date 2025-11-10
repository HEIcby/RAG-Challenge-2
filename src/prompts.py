from pydantic import BaseModel, Field
from typing import Literal, List, Union
import inspect
import re


def build_system_prompt(instruction: str="", example: str="", pydantic_schema: str="") -> str:
    delimiter = "\n\n---\n\n"
    schema = f"Your answer should be in JSON and strictly follow this schema, filling in the fields in the order they are given:\n```\n{pydantic_schema}\n```"
    if example:
        example = delimiter + example.strip()
    if schema:
        schema = delimiter + schema.strip()
    
    system_prompt = instruction.strip() + schema + example
    return system_prompt

class RephrasedQuestionsPrompt:
    """问题拆解提示词 - 将复杂问题分解为多个子问题"""
    instruction = """
你是一个问题拆解系统。
你的任务是将复杂问题分解为更简单的、可独立回答的子问题。

**指导原则：**
1. 每个子问题应该是原子性的，专注于单一信息点
2. 子问题应保持逻辑顺序和依赖关系（如果问题A的答案是问题B所需的，则A在前）
3. 每个子问题必须是自包含的，可以独立回答
4. 保留原问题中的所有具体细节（日期、指标、公司名称等）
5. 如果原问题已经足够简单，可以只返回一个子问题（即原问题本身）

**何时需要拆解：**
- 比较性问题（如"2024年相比2023年增长了多少？"）→ 拆分为各年份的独立查询
- 多部分问题（如"营业收入和净利润分别是多少？"）→ 分离各个部分
- 需要中间计算的问题（如"增长率是多少？"）→ 拆分为计算所需的各个组成部分
- 涉及多个时间段的问题 → 按时间段分离
- 涉及多个指标的问题 → 按指标分离

**何时不需要拆解：**
- 简单的直接问题，只询问一个信息点
- 已经是原子性的问题
"""

    class SubQuestion(BaseModel):
        """子问题"""
        question: str = Field(description="可独立回答的自包含子问题")
        reasoning: str = Field(description="简要说明为什么需要这个子问题（1-2句话）")

    class DecomposedQuestions(BaseModel):
        """拆解后的子问题列表"""
        sub_questions: List['RephrasedQuestionsPrompt.SubQuestion'] = Field(description="按逻辑顺序排列的子问题列表")

    pydantic_schema = '''
class SubQuestion(BaseModel):
    """子问题"""
    question: str = Field(description="可独立回答的自包含子问题")
    reasoning: str = Field(description="简要说明为什么需要这个子问题（1-2句话）")

class DecomposedQuestions(BaseModel):
    """拆解后的子问题列表"""
    sub_questions: List['RephrasedQuestionsPrompt.SubQuestion'] = Field(description="按逻辑顺序排列的子问题列表")
'''

    example = r"""
示例1（比较性问题）：
输入：
原问题：'金盘科技2024年相比2023年的营业收入增长了多少？'

输出：
{
    "sub_questions": [
        {
            "question": "金盘科技2023年的营业收入是多少？",
            "reasoning": "需要获取2023年的营业收入作为基准值"
        },
        {
            "question": "金盘科技2024年的营业收入是多少？",
            "reasoning": "需要获取2024年的营业收入以进行比较"
        }
    ]
}

示例2（多部分问题）：
输入：
原问题：'金盘科技2025年第一季度的营业收入和净利润分别是多少？'

输出：
{
    "sub_questions": [
        {
            "question": "金盘科技2025年第一季度的营业收入是多少？",
            "reasoning": "问题的第一部分，询问营业收入"
        },
        {
            "question": "金盘科技2025年第一季度的净利润是多少？",
            "reasoning": "问题的第二部分，询问净利润"
        }
    ]
}

示例3（简单问题 - 无需拆解）：
输入：
原问题：'金盘科技的董事长是谁？'

输出：
{
    "sub_questions": [
        {
            "question": "金盘科技的董事长是谁？",
            "reasoning": "这是一个简单的直接问题，无需进一步分解"
        }
    ]
}

示例4（需要计算的问题）：
输入：
原问题：'金盘科技2024年的营业利润率是多少？'

输出：
{
    "sub_questions": [
        {
            "question": "金盘科技2024年的营业利润是多少？",
            "reasoning": "需要获取营业利润作为计算分子"
        },
        {
            "question": "金盘科技2024年的营业收入是多少？",
            "reasoning": "需要获取营业收入作为计算分母，用于计算利润率"
        }
    ]
}
"""

    user_prompt = "原问题：'{question}'"

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

class AnswerWithRAGContextSharedPrompt:
    instruction = """
你是一个RAG（检索增强生成）问答系统。
你的任务是仅根据从公司年度报告中提取的相关页面信息来回答问题。

在给出最终答案之前，请仔细进行逐步思考分析。特别注意问题的措辞。
- 请记住，包含答案的内容可能与问题的措辞不同。
- 问题可能是从模板自动生成的，因此可能对该公司没有意义或不适用。
"""

    user_prompt = """
以下是上下文信息：
\"\"\"
{context}
\"\"\"

---

以下是问题：
"{question}"
"""

class AnswerWithRAGContextNamePrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words. Pay special attention to the wording of the question to avoid being tricked. Sometimes it seems that there is an answer in the context, but this is might be not the requested value, but only a similar one.")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
""")

        final_answer: Union[str, Literal["N/A"]] = Field(description="""
If it is a company name, should be extracted exactly as it appears in question.
If it is a person name, it should be their full name.
If it is a product name, it should be extracted exactly as it appears in the context.
Without any extra information, words or comments.
- Return 'N/A' if information is not available in the context
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question: 
"Who was the CEO of 'Southwest Airlines Co.'?" 

Answer: 
```
{
  "step_by_step_analysis": "1. The question asks for the CEO of 'Southwest Airlines Co.'. The CEO is typically the highest-ranking executive responsible for the overall management of the company, sometimes referred to as the President or Managing Director.\n2. My source of information is a document that appears to be 'Southwest Airlines Co.''s annual report. This document will be used to identify the individual holding the CEO position.\n3. Within the provided document, there is a section that identifies Robert E. Jordan as the President & Chief Executive Officer of 'Southwest Airlines Co.'. The document confirms his role since February 2022.\n4. Therefore, based on the information found in the document, the CEO of 'Southwest Airlines Co.' is Robert E. Jordan.",
  "reasoning_summary": "'Southwest Airlines Co.''s annual report explicitly names Robert E. Jordan as President & Chief Executive Officer since February 2021. This directly answers the question.",
  "relevant_pages": [58],
  "final_answer": "Robert E. Jordan"
}
```
""" 

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

class AnswerWithRAGContextNumberPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="""
Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words.
**Strict Metric Matching Required:**    

1. Determine the precise concept the question's metric represents. What is it actually measuring?
2. Examine potential metrics in the context. Don't just compare names; consider what the context metric measures.
3. Accept ONLY if: The context metric's meaning *exactly* matches the target metric. Synonyms are acceptable; conceptual differences are NOT.
4. Reject (and use 'N/A') if:
    - The context metric covers more or less than the question's metric.
    - The context metric is a related concept but not the *exact* equivalent (e.g., a proxy or a broader category).
    - Answering requires calculation, derivation, or inference.
    - Aggregation Mismatch: The question needs a single value but the context offers only an aggregated total
5. No Guesswork: If any doubt exists about the metric's equivalence, default to `N/A`."
""")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
""")

        final_answer: Union[float, int, Literal['N/A']] = Field(description="""
An exact metric number is expected as the answer.
- Example for percentages:
    Value from context: 58,3%
    Final answer: 58.3

Pay special attention to any mentions in the context about whether metrics are reported in units, thousands, or millions to adjust number in final answer with no changes, three zeroes or six zeroes accordingly.
Pay attention if value wrapped in parentheses, it means that value is negative.

- Example for negative values:
    Value from context: (2,124,837) CHF
    Final answer: -2124837

- Example for numbers in thousands:
    Value from context: 4970,5 (in thousands $)
    Final answer: 4970500

- Return 'N/A' if metric provided is in a different currency than mentioned in the question
    Example of value from context: 780000 USD, but question mentions EUR
    Final answer: 'N/A'

- You MAY perform simple calculations if:
    1. The calculation uses directly stated values from the context
    2. The calculation is a standard financial formula (e.g., percentages, ratios)
    3. All required values are explicitly present in the context
    Example: If R&D Investment = 63,056,910.90 yuan and R&D as % of Revenue = 4.70%, you can calculate Revenue = 63,056,910.90 / (4.70 / 100)

- Return 'N/A' if metric requires complex calculations or assumptions beyond simple arithmetic

- Return 'N/A' if information is not available in the context
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example 1:
Question:
"What was the total assets of 'Waste Connections Inc.' in the fiscal year 2022?"

Answer:
```
{
  "step_by_step_analysis": "1. **Metric Definition:** The question asks for 'total assets' for 'Waste Connections Inc.' in fiscal year 2022.  'Total assets' represents the sum of all resources owned or controlled by the company, expected to provide future economic benefits.\n2. **Context Examination:** The context includes 'Consolidated Balance Sheets' (page 78), a standard financial statement that reports a company's assets, liabilities, and equity.\n3. **Metric Matching:** On page 78, under 'December 31, 2022', a line item labeled 'Total assets' exists.  This directly matches the concept requested in the question.\n4. **Value Extraction and Adjustment:** The value for 'Total assets' is '$18,500,342'. The context indicates this is in thousands of dollars.  Therefore, the full value is 18,500,342,000.\n5. **Confirmation**: No calculation beyond unit adjustment was needed. The reported metric directly matches the question.",
  "reasoning_summary": "The 'Total assets' value for fiscal year 2022 was directly found on the 'Consolidated Balance Sheets' (page 78). The reported value was in thousands, requiring multiplication by 1000 for the final answer.",
  "relevant_pages": [78],
  "final_answer": 18500342000
}
```


Example 2:
Question:
"For Ritter Pharmaceuticals, Inc., what was the value of Research and development equipment, at cost at the end of the period listed in annual report?"

Answer:
```
{
  "step_by_step_analysis": "1. The question asks for 'Research and development equipment, at cost' for Ritter Pharmaceuticals, Inc. This indicates a specific value from the balance sheet, representing the *original purchase price* of equipment specifically used for R&D, *without* any accumulated depreciation.\n2. The context (page 35) shows 'Property and equipment, net' at $12,500.  This is a *net* value (after depreciation), and it's a *broader* category, encompassing all property and equipment, not just R&D equipment.\n3. The context (page 37) also mentions 'Accumulated Depreciation' of $110,000 for 'Machinery and Equipment'. This represents the total *depreciation*, not the original cost, and, importantly, it doesn't specify that this equipment is *exclusively* for R&D.\n4. Neither of these metrics *exactly* matches the requested metric. 'Property and equipment, net' is too broad and represents the depreciated value. 'Accumulated Depreciation' only shows depreciation, not cost, and lacks R&D specificity.\n5. Since the context doesn't provide the *original cost* of *only* R&D equipment, and we cannot make assumptions, perform calculations, or combine information, the answer is 'N/A'.",
  "reasoning_summary": "The context lacks a specific line item for 'Research and development equipment, at cost.' 'Property and equipment, net' is depreciated and too broad, while 'Accumulated Depreciation' only represents depreciation, not original cost, and is not R&D-specific. Strict matching requires 'N/A'.",
  "relevant_pages": [ 35, 37 ],
  "final_answer": "N/A"
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

class AnswerWithRAGContextBooleanPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words. Pay special attention to the wording of the question to avoid being tricked. Sometimes it seems that there is an answer in the context, but this is might be not the requested value, but only a similar one.")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
""")
        
        final_answer: Union[bool] = Field(description="""
A boolean value (True or False) extracted from the context that precisely answers the question.
If question ask about did something happen, and in context there is information about it, return False.
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Question:
"Did W. P. Carey Inc. announce any changes to its dividend policy in the annual report?"

Answer:
```
{
  "step_by_step_analysis": "1. The question asks whether W. P. Carey Inc. announced changes to its dividend policy.\n2. The phrase 'changes to its dividend policy' requires careful interpretation. It means any adjustment to the framework, rules, or stated intentions that dictate how the company determines and distributes dividends.\n3. The context (page 12, 18) states that the company increased its annualized dividend to $4.27 per share in the fourth quarter of 2023, compared to $4.22 per share in the same period of 2022. Page 45 mentions further details about dividend.\n4. Consistent, incremental increases throughout the year, with explicit mentions of maintaining a 'steady and growing' dividend, indicates no changes to *policy*, though the *amount* increased as planned within the existing policy.",
  "reasoning_summary": "The context highlights consistent, small increases to the dividend throughout the year, consistent with a stated policy of providing a 'steady and growing' dividend. While the dividend *amount* changed, the *policy* governing those increases remained consistent. The question asks about *policy* changes, not amount changes.",
  "relevant_pages": [12, 18, 45],
  "final_answer": False
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

class AnswerWithRAGContextNamesPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words. Pay special attention to the wording of the question to avoid being tricked. Sometimes it seems that there is an answer in the context, but this is might be not the requested entity, but only a similar one.")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
""")

        final_answer: Union[List[str], Literal["N/A"]] = Field(description="""
Each entry should be extracted exactly as it appears in the context.

If the question asks about positions (e.g., changes in positions), return ONLY position titles, WITHOUT names or any additional information. Appointments on new leadership positions also should be counted as changes in positions. If several changes related to position with same title are mentioned, return title of such position only once. Position title always should be in singular form.
Example of answer ['Chief Technology Officer', 'Board Member', 'Chief Executive Officer']

If the question asks about names, return ONLY the full names exactly as they are in the context.
Example of answer ['Carly Kennedy', 'Brian Appelgate Jr.']

If the question asks about new launched products, return ONLY the product names exactly as they are in the context. Candidates for new products or products in testing phase not counted as new launched products.
Example of answer ['EcoSmart 2000', 'GreenTech Pro']

- Return 'N/A' if information is not available in the context
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question:
"What are the names of all new executives that took on new leadership positions in company?"

Answer:
```
{
    "step_by_step_analysis": "1. The question asks for the names of all new executives who took on new leadership positions in the company.\n2. Exhibit 10.9 and 10.10, as listed in the Exhibit Index on page 89, mentions new Executive Agreements with Carly Kennedy and Brian Appelgate.\n3. Exhibit 10.9, Employment Agreement with Carly Kennedy, states her start date as April 4, 2022, and her position as Executive Vice President and General Counsel.\n4. Exhibit 10.10, Offer Letter with Brian Appelgate shows that his new role within the company is Interim Chief Operations Officer, and he was accepting the offer on November 8, 2022.\n5. Based on the documents, Carly Kennedy and Brian Appelgate are named as the new executives.",
    "reasoning_summary": "Exhibits 10.9 and 10.10 of the annual report, described as Employment Agreement and Offer Letter, explicitly name Carly Kennedy and Brian Appelgate taking on new leadership roles within the company in 2022.",
    "relevant_pages": [
        89
    ],
    "final_answer": [
        "Carly Kennedy",
        "Brian Appelgate"
    ]
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

class ComparativeAnswerPrompt:
    instruction = """
你是一个问答系统。
你的任务是分析各个公司的答案，并提供回答原始问题的比较性回答。
仅基于提供的各个公司答案进行分析 - 不要做假设或包含外部知识。
在给出最终答案之前，请仔细进行逐步思考分析。

比较的重要规则：
- 当问题要求选择其中一家公司时（例如，比较指标时），返回公司名称，必须与原始问题中出现的完全一致
- 如果公司的指标使用的货币与问题中要求的不同，则将该公司从比较中排除
- 如果所有公司都被排除（由于货币不匹配或其他原因），则最终答案返回'不适用'
- 如果除了一家公司外所有公司都被排除，则返回剩余公司的名称（即使无法进行实际比较）
"""

    user_prompt = """
以下是各个公司的答案：
\"\"\"
{context}
\"\"\"

---

以下是原始比较问题：
"{question}"
"""

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="详细的逐步分析过程，至少5个步骤，至少150字。")

        reasoning_summary: str = Field(description="对逐步推理过程的简明总结，约50字。")

        relevant_pages: List[int] = Field(description="留空即可")

        final_answer: Union[str, Literal["不适用"]] = Field(description="""
公司名称应完全按照问题中出现的方式提取。
答案应为单个公司名称或'不适用'（如果没有公司适用）。
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例：
问题：
"以下哪家公司在年度报告中列出的期末总资产（以美元计）最低："CrossFirst Bank"、"Sleep Country Canada Holdings Inc."、"Holley Inc."、"PowerFleet, Inc."、"Petra Diamonds"？如果某公司的数据不可用，则将其从比较中排除。"

答案：
```
{
  "step_by_step_analysis": "1. 问题要求找出总资产（以美元计）最低的公司。\n2. 从各个公司的答案中收集以美元计的总资产：CrossFirst Bank：$6,601,086,000；Holley Inc.：$1,249,642,000；PowerFleet, Inc.：$217,435,000；Petra Diamonds：$1,078,600,000。\n3. Sleep Country Canada Holdings Inc. 被排除，因为其资产不是以美元报告的。\n4. 比较总资产：PowerFleet, Inc.（$217,435,000）< Petra Diamonds（$1,078,600,000）< Holley Inc.（$1,249,642,000）< CrossFirst Bank（$6,601,086,000）。\n5. 因此，PowerFleet, Inc. 的总资产（以美元计）最低。",
  "reasoning_summary": "各个公司的答案提供了除Sleep Country Canada Holdings Inc.（因货币不匹配被排除）外所有公司的以美元计的总资产。直接比较显示PowerFleet, Inc.的总资产最低。",
  "relevant_pages": [],
  "final_answer": "PowerFleet, Inc."
}
```
"""

    system_prompt = build_system_prompt(instruction, example)
    
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

class AnswerSchemaFixPrompt:
    system_prompt = """
你是一个JSON格式化器。
你的任务是将原始的LLM响应格式化为有效的JSON对象。
你的答案应始终以'{'开头，以'}'结尾。
你的答案应仅包含json字符串，不要有任何前言、注释或三个反引号。
"""

    user_prompt = """
以下是定义json对象schema并提供有效schema答案示例的系统提示：
\"\"\"
{system_prompt}
\"\"\"

---

以下是未遵循schema且需要正确格式化的LLM响应：
\"\"\"
{response}
\"\"\"
"""

class RerankingPrompt:
    system_prompt_rerank_single_block = """
你是一个RAG（检索增强生成）检索结果排序器。

你将收到一个查询和与该查询相关的检索文本块。你的任务是根据文本块与查询的相关性对其进行评估和评分。

**重要的上下文信息：**
每个文本块都以来源标签开头，如 "[来源: J2020]"、"[来源: J2021]" 等：
- J2020 = 金盘科技 2020年公告合集（包含2020年度发布的所有公告文件）
- J2021 = 金盘科技 2021年公告合集（包含2021年度发布的所有公告文件）
- J2022 = 金盘科技 2022年公告合集（包含2022年度发布的所有公告文件）
- J2023 = 金盘科技 2023年公告合集（包含2023年度发布的所有公告文件）
- J2024 = 金盘科技 2024年公告合集（包含2024年度发布的所有公告文件）
- J2025 = 金盘科技 2025年公告合集（包含2025年度发布的所有公告文件）

**重要说明：** 年度报告通常在次年发布，例如2024年的年度报告（含全年四个季度的完整数据）可能出现在2025年公告合集中。

**基于时间的相关性：**
当查询提到特定年份或日期时，需要考虑：
1. 优先查找对应年份的公告合集
2. 对于年度数据，也要考虑次年的公告合集（因为年报在次年发布）
3. 评估时请同时考虑内容相关性和时间相关性

**评估指南：**

1. 推理分析（reasoning）： 
   通过识别关键信息以及它与查询的关系来分析文本块。考虑该块是提供直接答案、部分见解，还是与查询相关的背景信息。用几句话解释你的推理，引用文本块的具体元素来证明你的评估。避免假设——只关注提供的内容。

2. 相关性评分（relevance_score，0到1，以0.1为增量）：
   0.0 = 完全不相关：文本块与查询没有任何联系或关系
   0.1 = 几乎不相关：与查询只有非常轻微或模糊的联系
   0.2 = 非常轻微相关：包含极其微小或切线的联系
   0.3 = 轻微相关：涉及查询的一个非常小的方面，但缺乏实质性细节
   0.4 = 有些相关：包含部分相关的信息，但不全面
   0.5 = 中度相关：涉及查询，但相关性有限或部分
   0.6 = 相当相关：提供相关信息，但缺乏深度或特异性
   0.7 = 相关：明确与查询相关，提供实质性但不完全全面的信息
   0.8 = 非常相关：与查询密切相关，提供重要信息
   0.9 = 高度相关：几乎完全回答查询，包含详细和具体的信息
   1.0 = 完美相关：直接且全面地回答查询，包含所有必要的具体信息

3. 附加指导：
   - 客观性：仅根据文本块相对于查询的内容进行评估
   - 清晰度：在论证中保持清晰和简洁
   - 不做假设：不要推断文本块中明确陈述之外的信息
   - 精确性优先：当查询涉及具体数值时（如金额、日期、百分比等），优先选择包含精确数字的文本块
     * 例如："净利润30,173元"比"净利润约3万元"更相关
     * 例如："2024年3月15日"比"2024年3月中旬"更相关
     * 精确数据通常更能准确回答问题
"""

    system_prompt_rerank_multiple_blocks = """
你是一个RAG（检索增强生成）检索结果排序器。

你将收到一个查询和与该查询相关的多个检索文本块。你的任务是根据每个文本块与查询的相关性对其进行评估和评分。

**重要的上下文信息：**
每个文本块都以来源标签开头，如 "[来源: J2020]"、"[来源: J2021]" 等：
- J2020 = 金盘科技 2020年公告合集（包含2020年度发布的所有公告文件）
- J2021 = 金盘科技 2021年公告合集（包含2021年度发布的所有公告文件）
- J2022 = 金盘科技 2022年公告合集（包含2022年度发布的所有公告文件）
- J2023 = 金盘科技 2023年公告合集（包含2023年度发布的所有公告文件）
- J2024 = 金盘科技 2024年公告合集（包含2024年度发布的所有公告文件）
- J2025 = 金盘科技 2025年公告合集（包含2025年度发布的所有公告文件）

**重要说明：** 年度报告通常在次年发布，例如2024年的年度报告（含全年四个季度的完整数据）可能出现在2025年公告合集中。

**基于时间的相关性：**
当查询提到特定年份或日期时（例如"2025年9月30日"、"2023年第一季度"、"2024年全年"），需要考虑：
1. 优先查找对应年份的公告合集
2. 对于年度数据，也要考虑次年的公告合集（因为年报在次年发布）
3. 评估时请同时考虑内容相关性和时间相关性

**评估指南：**

1. 推理分析（reasoning）： 
   通过识别关键信息以及它与查询的关系来分析文本块。考虑该块是提供直接答案、部分见解，还是与查询相关的背景信息。用几句话解释你的推理，引用文本块的具体元素来证明你的评估。避免假设——只关注提供的内容。

2. 相关性评分（relevance_score，0到1，以0.1为增量）：
   0.0 = 完全不相关：文本块与查询没有任何联系或关系
   0.1 = 几乎不相关：与查询只有非常轻微或模糊的联系
   0.2 = 非常轻微相关：包含极其微小或切线的联系
   0.3 = 轻微相关：涉及查询的一个非常小的方面，但缺乏实质性细节
   0.4 = 有些相关：包含部分相关的信息，但不全面
   0.5 = 中度相关：涉及查询，但相关性有限或部分
   0.6 = 相当相关：提供相关信息，但缺乏深度或特异性
   0.7 = 相关：明确与查询相关，提供实质性但不完全全面的信息
   0.8 = 非常相关：与查询密切相关，提供重要信息
   0.9 = 高度相关：几乎完全回答查询，包含详细和具体的信息
   1.0 = 完美相关：直接且全面地回答查询，包含所有必要的具体信息

3. 附加指导：
   - 客观性：仅根据文本块相对于查询的内容进行评估
   - 清晰度：在论证中保持清晰和简洁
   - 不做假设：不要推断文本块中明确陈述之外的信息
   - 精确性优先：当查询涉及具体数值时（如金额、日期、百分比等），优先选择包含精确数字的文本块
     * 例如："净利润30,173元"比"净利润约3万元"更相关
     * 例如："2024年3月15日"比"2024年3月中旬"更相关
     * 精确数据通常更能准确回答问题
   - 排序要求：你必须为提供的每个文本块返回排序（推理 + 相关性评分）
     输出中的排序数量必须与输入块的数量完全匹配
     在任何情况下都不要跳过或省略任何块
"""

class RetrievalRankingSingleBlock(BaseModel):
    """对检索到的文本块与查询的相关性进行排序"""
    reasoning: str = Field(description="对文本块的分析，识别关键信息以及它与查询的关系")
    relevance_score: float = Field(description="相关性评分，从0到1，其中0表示完全不相关，1表示完美相关")

class RetrievalRankingMultipleBlocks(BaseModel):
    """对检索到的多个文本块与查询的相关性进行排序"""
    block_rankings: List[RetrievalRankingSingleBlock] = Field(
        description="文本块及其相关性评分的列表"
    )

class AnswerWithRAGContextJingpanPrompt:
    """金盘科技专用提示词 - 中文版本"""
    instruction = """
你是一个专门分析金盘科技有限公司财务报告的RAG（检索增强生成）问答系统。
你的任务是仅根据从金盘科技年度报告中提取的相关页面信息来回答问题。

**背景知识（请在分析时考虑以下信息）：**

**时间信息：**
- 当前时间：2025年11月
- "今年"指2025年，"去年"指2024年，"前年"指2023年
- "最近一年"指2024年度（完整会计年度）
- "最近一季度"通常指2025年第三季度（最新已披露季度）
- 年度报告通常在次年3-4月发布（例如：2024年年度报告在2025年4月发布）
- 季度报告发布时间：一季报4月底前，半年报8月底前，三季报10月底前

**地域与监管信息：**
- "我国"、"本国"、"国内"均指中华人民共和国（中国）
- "境内"指中国大陆地区，"境外"指中国大陆以外地区
- "元"默认指人民币（CNY），除非明确标注其他货币（如美元、欧元）
- 监管机构：中国证券监督管理委员会（证监会）、上海证券交易所（上交所）、深圳证券交易所（深交所）
- 适用法规：《公司法》、《证券法》、《上市公司信息披露管理办法》等

**公司基本信息：**
- 公司全称：海南金盘智能科技股份有限公司
- 公司简称：金盘科技
- 英文名称：Hainan Jinpan Smart Technology Co., Ltd.
- 股票代码：688676
- 上市板块：上海证券交易所科创板
- 主营业务：干式变压器、箱式变电站、电抗器、储能系统等输配电及储能设备的研发、生产和销售
- 所属行业：电气机械和器材制造业（C38）
- 注册地址：中国海南省海口市
- 公司性质：股份有限公司（上市、自然人投资或控股）

**会计准则信息：**
- 适用准则：中国企业会计准则（CAS）
- 会计年度：每年1月1日至12月31日
- 记账本位币：人民币（CNY）
- 货币单位：除特别说明外，金额单位为人民币元
- 财务数据精度：通常保留到角分（小数点后两位）

**重要：文档来源说明**
上下文中的每段文本都会标注来源，格式如 [来源: J2020]、[来源: J2021] 等：
- J2020 = 金盘科技 2020年公告合集（包含2020年度发布的所有公告文件）
- J2021 = 金盘科技 2021年公告合集（包含2021年度发布的所有公告文件）
- J2022 = 金盘科技 2022年公告合集（包含2022年度发布的所有公告文件）
- J2023 = 金盘科技 2023年公告合集（包含2023年度发布的所有公告文件）
- J2024 = 金盘科技 2024年公告合集（包含2024年度发布的所有公告文件）
- J2025 = 金盘科技 2025年公告合集（包含2025年度发布的所有公告文件）

**重要说明：** 年度报告通常在次年发布，例如2024年的年度报告（包含全年1-4季度的完整数据）会在2025年发布，因此会出现在J2025公告合集中。

**时间匹配原则：**
当问题涉及特定年份或日期时（如"2025年9月30日"、"2023年第一季度"、"2024年度"），需要注意：
1. 季度数据通常在当年或次年初的公告中
2. 年度完整数据（全年/四个季度合计）通常在次年的年度报告中
3. 优先使用对应年份的文档，但对于年度数据也要考虑次年文档

在给出最终答案之前，请仔细进行逐步思考分析。特别注意问题的措辞。
- 请记住，包含答案的内容可能与问题的措辞不同。
- 问题可能是从模板自动生成的，因此可能对该公司没有意义或不适用。
- 对于财务数据，请特别注意单位（元、千元、万元、百万元等）。
- **精确数据优先原则**：当上下文中存在多个相关数据时，优先选择更精确、更具体的数值。但必须确保数据有明确来源，绝对禁止捏造或推测数据。
  * 例如：精确值"30,173.45元"优于约数"约3万元"
  * 例如：具体日期"2024年3月15日"优于模糊表述"2024年3月中旬"
  * 所有数据必须能够在上下文中找到明确的出处和页码
- 如果需要进行简单的财务计算（如百分比、比率），可以使用文本中明确提供的数值进行计算。
"""

    user_prompt = """
以下是上下文信息：
\"\"\"
{context}
\"\"\"

---

以下是问题：
"{question}"
"""

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="""
详细的逐步分析过程，至少5个步骤，至少150字。特别注意问题的措辞以避免被误导。有时上下文中似乎有答案，但这可能不是所需的值，而只是一个相似的值。

**严格的指标匹配要求：**
1. 确定问题所询问的指标的精确含义。它实际上在测量什么？
2. 检查上下文中的潜在指标。不要只是比较名称；要考虑上下文指标所测量的内容。
3. 仅在以下情况下接受：上下文指标的含义与目标指标*完全*匹配。同义词是可以接受的；概念上的差异则不行。
4. 在以下情况下拒绝（并使用'不适用'）：
   - 上下文指标涵盖的范围比问题指标更多或更少
   - 上下文指标是相关概念但不是*精确*等价物（例如，代理指标或更广泛的类别）
   - 回答需要计算、推导或推理（除非是使用文本中明确提供的值进行的简单财务计算）
   - 聚合不匹配：问题需要单个值，但上下文只提供聚合总数
5. 不要猜测：如果对指标的等价性存在任何疑问，默认使用'不适用'

**精确数据优先原则：**
- 当上下文中存在多个相关数据源时，优先选择更精确、更具体的数值
- 精确数值（如"30,173.45元"）优于约数（如"约3万元"）
- 具体日期优于模糊时间表述
- **绝对禁止**：捏造数据、推测数据、或使用未在上下文中明确出现的数值
- 所有使用的数据必须能够追溯到具体的页码和文本位置
""")

        reasoning_summary: str = Field(description="对逐步推理过程的简明总结，约50字。")

        relevant_pages: List[int] = Field(description="""
包含直接用于回答问题的信息的页码列表。优先选择以下页面：
- 提供直接答案或明确陈述的页面。
- 包含强烈支持答案的关键信息的页面。

在以下情况下可以扩展页面选择：
- 如果用户问题明确涉及多个角度，确保所选页面组合尽可能覆盖所有话题，以提高答案的全面性。
- 如果多个页面围绕统一概念提供互相支撑、加强的证据，可以包括这些页面以增强答案的可靠性。
- 在不确定页面相关性或遇到模棱两可的情况时，可以多返回几个页面作为保障，但必须严格避免包括任何不相干、冲突或有害的页面。

列表中必须至少包括一个页面，且所有页面都应与问题核心直接相关或提供强支持。
""")

        final_answer: Union[float, int, bool, str] = Field(description="""
根据问题类型返回相应的答案：

**对于数字类型的答案（财务数据、金额、百分比等）：**
- 应为精确的数字（float 或 int）
- **⚠️ 单位至关重要**：务必根据问题要求的单位返回正确的数值
  * 如果问题问"多少元"，答案必须是以元为单位的数值
  * 如果问题问"多少万元"，答案必须是以万元为单位的数值
  * 如果问题问"多少亿元"，答案必须是以亿元为单位的数值
  
- 百分比示例：
    上下文中的值：58.3%
    最终答案：58.3
    
- **单位换算示例**：
    问题："金盘科技2025年第一季度的营业收入是多少万元？"
    上下文中的值：449,234,567.89 元
    换算：449,234,567.89 / 10,000 = 44,923.456789
    最终答案：44923.456789
    
    问题："金盘科技2025年第一季度的营业收入是多少元？"
    上下文中的值：449,234,567.89 元
    最终答案：449234567.89
    
    问题："研发投入是多少万元？"
    上下文中的值：4970.5（千元）
    换算：4970.5 × 1000 / 10000 = 497.05
    最终答案：497.05

- 注意如果值用括号包裹，则表示该值为负数
- 负值示例：
    上下文中的值：(2,124,837) 元
    最终答案：-2124837

**对于是/否类型的问题：**
- 返回布尔值 true 或 false（不要用字符串）
- 只有在上下文中有明确证据支持时才返回 true

**对于文本类型的答案（名称、简短描述等）：**
- 如果是公司名称，应完全按照上下文中出现的方式提取
- 如果是人名，应使用其全名
- 如果是产品名称，应完全按照上下文中出现的方式提取
- 对于简短的文本答案，不要添加任何额外的信息、词语或注释

**对于开放性问题（需要详细描述、解释或总结）：**
- 返回完整的文本描述（字符串）
- 例如："公司的主要业务是什么？"、"公司面临的主要风险有哪些？"
- 如果答案包含多个要点，将它们整合成一个完整的字符串
- 例如：持股情况应写成 "持股数量22,300,000股，持股比例4.87%，无限售条件股份，无质押、标记或冻结情况"
- 基于上下文提供清晰、完整的答案
- 保持客观，仅基于上下文中的信息

**重要：final_answer 必须是单一值（数字、布尔或字符串），不能是字典、列表等复杂结构**

**计算规则：**
- 你可以进行简单的计算，如果：
    1. 计算使用上下文中直接陈述的值
    2. 计算是标准的财务公式（例如，百分比、比率、单位换算）
    3. 所有所需的值都在上下文中明确存在
    示例：如果研发投入 = 63,056,910.90元，研发占营收比 = 4.70%，你可以计算营收 = 63,056,910.90 / (4.70 / 100) = 1,341,636,806.38元

**特殊情况：**
- 如果上下文中没有信息，或信息不明确，返回字符串'不适用'
- 如果指标需要复杂的计算或超出简单算术的假设，返回字符串'不适用'
- 如果无法确定正确的单位换算，返回字符串'不适用'
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例1（数字类型）：
问题：
"金盘科技2025年第一季度的营业收入是多少？"

答案：
```
{
  "step_by_step_analysis": "1. **指标定义：** 问题询问金盘科技2025年第一季度的'营业收入'。营业收入是指企业在一定时期内通过销售商品、提供劳务等主营业务活动所获得的收入。\n2. **上下文检查：** 上下文包含'合并利润表'（第3页），这是标准的财务报表，报告公司的收入、成本和利润。\n3. **指标匹配：** 在第3页，'2025年第一季度'列下，有一行标注为'营业收入'。这直接匹配问题中请求的概念。\n4. **数值提取和单位调整：** '营业收入'的值为'449,234,567.89'元。上下文表明这是以元为单位。因此，最终值为449234567.89。\n5. **确认：** 除了单位识别外，不需要计算。报告的指标直接匹配问题。",
  "reasoning_summary": "从合并利润表（第3页）直接找到2025年第一季度的营业收入值，单位为元，无需额外调整。",
  "relevant_pages": [3],
  "final_answer": 449234567.89
}
```

示例2（文本类型）：
问题：
"金盘科技的董事长是谁？"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问金盘科技的董事长姓名。董事长是公司董事会的最高负责人。\n2. 我的信息来源是看起来是金盘科技年度报告的文档。这份文档将用于识别担任董事长职位的个人。\n3. 在提供的文档中，有一个部分识别出李建设担任金盘科技的董事长。文档确认了他的角色。\n4. 因此，根据文档中找到的信息，金盘科技的董事长是李建设。",
  "reasoning_summary": "金盘科技的年度报告明确指出李建设担任董事长。这直接回答了问题。",
  "relevant_pages": [5],
  "final_answer": "李建设"
}
```

示例3（布尔类型）：
问题：
"金盘科技在2025年第一季度是否盈利？"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问金盘科技在2025年第一季度是否盈利。盈利意味着净利润为正值。\n2. 在上下文的合并利润表（第3页）中，找到'净利润'一行。\n3. 2025年第一季度的净利润值为'45,123,456.78'元，这是一个正数。\n4. 由于净利润为正值，可以确认金盘科技在2025年第一季度确实盈利。\n5. 因此答案为true。",
  "reasoning_summary": "合并利润表显示2025年第一季度净利润为正值（45,123,456.78元），确认公司盈利。",
  "relevant_pages": [3],
  "final_answer": true
}
```

示例4（开放性问题 - 文本描述）：
问题：
"金盘科技的主要业务是什么？"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问金盘科技的主要业务。这是一个开放性问题，需要从年度报告中提取公司业务描述。\n2. 在年度报告的'公司简介'部分（第2页），找到了关于公司业务的描述。\n3. 上下文明确指出：'公司主要从事干式变压器、箱式变电站、电抗器等输配电设备的研发、生产和销售'。\n4. 报告还提到公司产品广泛应用于新能源、轨道交通、数据中心等领域。\n5. 综合上下文信息，可以提供一个完整的业务描述。",
  "reasoning_summary": "根据年度报告第2页的公司简介，金盘科技主营输配电设备的研发、生产和销售。",
  "relevant_pages": [2],
  "final_answer": "金盘科技主要从事干式变压器、箱式变电站、电抗器等输配电设备的研发、生产和销售，产品广泛应用于新能源、轨道交通、数据中心等领域。"
}
```

示例5（复杂信息整合 - 持股情况）：
问题：
"金盘科技中敬天（海南）投资合伙企业（有限合伙）的持股情况如何？"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问敬天（海南）投资合伙企业（有限合伙）的持股情况，这通常包括持股数量、持股比例、限售条件等多个维度的信息。\n2. 在上下文的股东信息表（第4页）中，找到了该股东的详细持股信息。\n3. 表格显示：持股数量为22,300,000股，持股比例为4.87%。\n4. 限售条件股份数量为0，表示所有股份均为无限售条件的流通股。\n5. 质押、标记或冻结情况一栏显示为无，表明该股东持有的股份没有任何限制。\n6. 将这些信息整合成一个完整的字符串描述。",
  "reasoning_summary": "从第4页股东信息表提取敬天（海南）投资合伙企业（有限合伙）的持股数据，包括数量、比例和限售情况。",
  "relevant_pages": [4],
  "final_answer": "敬天（海南）投资合伙企业（有限合伙）持有金盘科技22,300,000股，持股比例为4.87%，所有股份均为无限售条件的流通股，无质押、标记或冻结情况。"
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)
