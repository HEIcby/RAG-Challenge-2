# 检索算法更新汇报

**日期**: 2025年12月3日  
**版本**: v2.0

## 📋 概述

本次更新主要实现了以下核心功能：
1. **新增两种图遍历检索算法**：SSG Traversal 和 Triangulation FullDim
2. **实现混合扩展检索链路**：Hybrid Expansion
3. **优化命名规范**：统一相似度字段命名
4. **增强聚合策略**：引入方法多样性奖励机制

---

## 🆕 新增检索算法

### 1. SSG Traversal Algorithm（SSG图遍历算法）

**核心思想**：基于图结构的贪心遍历，从锚点出发，逐步探索最相似的邻居节点。

**算法流程**：
1. **锚点查找**：使用基础向量检索找到与查询最相似的1个chunk作为起始锚点
2. **迭代扩展**：从当前chunk出发，在FAISS索引中查找其`neighbor_k`个最相似的邻居
3. **贪心选择**：选择与查询相似度最高的邻居作为下一跳
4. **去重机制**：维护已访问节点集合，避免重复遍历
5. **早停机制**：当下一跳的相似度低于阈值时停止扩展

**特点**：
- 适合发现跨文档的逻辑链路
- 强调"创造性探索"，可能发现间接相关但语义连贯的内容
- 默认参数：`max_hops=4`, `neighbor_k=30`

**适用场景**：
- 需要深度探索语义关联的场景
- 发现隐含的逻辑关系链
- 作为基础检索的补充扩展

### 2. Triangulation FullDim Algorithm（三角测量算法）

**核心思想**：在嵌入空间中构建几何三角形（查询点、当前chunk、候选chunk），选择质心最接近查询的候选。

**算法流程**：
1. **锚点查找**：与SSG相同，找到最相似的锚点
2. **三角测量**：对每个候选邻居，计算：
   - 查询向量 `q`
   - 当前chunk向量 `c`
   - 候选chunk向量 `t`
   - 三角形质心：`centroid = (q + c + t) / 3`
3. **距离计算**：计算质心到查询的欧氏距离
4. **相似度转换**：`score = 1.0 / (1.0 + distance)`
5. **贪心选择**：选择质心得分最高的候选作为下一跳

**特点**：
- 平衡创造性和相关性
- 通过几何约束确保扩展结果与查询保持语义关联
- 比基础检索更有探索性，比SSG更保守

**适用场景**：
- 需要平衡相关性和探索性的场景
- 作为基础检索的扩展补充
- 独立运行或与SSG组合使用

---

## 🔗 混合扩展检索链路（Hybrid Expansion）

### 设计理念

结合三种检索方法的优势，构建分层次的召回策略：
- **基础检索**：快速、准确，覆盖直接相关的内容
- **SSG扩展**：深度探索，发现逻辑链路
- **Triangulation扩展**：平衡探索，保持相关性

### 实现流程

```
基础召回（Basic Retrieval）
    ↓
Top-K 结果（默认50个chunks）
    ↓
    ├─→ SSG扩展（对Top-10进行深度探索）
    └─→ Triangulation扩展（对Top-20进行平衡扩展）
    ↓
聚合去重
    ↓
LLM Reranking
```

### 关键参数

- **Basic Top-K**: 50（基础检索数量）
- **SSG Top-K**: 10（对前10个结果进行SSG扩展）
- **Triangulation Top-K**: 20（对前20个结果进行Triangulation扩展）

---

## 🔧 技术改进

### 1. 命名规范统一

**问题**：代码中混用 `distance` 和 `similarity`，造成语义混淆。

**解决方案**：
- 将所有表示"相似度"（越大越好）的字段统一命名为 `vector_similarity`
- 保留 `distance` 仅用于真正的距离计算（如欧氏距离，越小越好）
- 更新所有相关文件：
  - `src/retrieval.py`
  - `src/reranking.py`
  - `src/questions_processing.py`
  - `app_jinpan_qa.py`

**影响**：
- 提高代码可读性
- 减少理解成本
- 保持向后兼容（同时支持新旧字段名）

### 2. 数据结构扩展

**新增字段**：
- `retrieval_source`: 标记每个chunk的检索方法来源（"basic", "ssg", "triangulation"）
- `retrieval_sources`: 聚合后的方法来源列表（支持多方法命中）
- `max_original_similarity`: 原始向量相似度最高分（未加权前）
- `vector_similarity`: 最终向量相似度得分（加权后）

**数据结构变化**：
```python
# 旧格式（5元组）
(key, page_id, text, similarity, sha1)

# 新格式（6元组，向后兼容）
(key, page_id, text, vector_similarity, sha1, retrieval_source)
```

### 3. 聚合策略优化

**新策略**：保留最大相似度 + 查询命中数奖励 + 方法多样性奖励

**计算公式**：
```python
base_similarity = max(similarities)  # 保留最大相似度
query_bonus = 1.0 + 0.2 * (count - 1)  # 查询命中数奖励
method_diversity_bonus = 1.0 + 0.1 * (len(unique_methods) - 1)  # 方法多样性奖励（≥2种方法时）
final_similarity = base_similarity × query_bonus × method_diversity_bonus
```

**奖励机制**：
- **查询命中数奖励**：被多个查询（HYDE/Multi-Query）命中时，给予1.2×、1.4×等奖励
- **方法多样性奖励**：被2种以上检索方法命中时，给予额外奖励
  - 2种方法：1.1×
  - 3种方法：1.2×

**优势**：
- 优先信任直接相关的方法（保留最大值）
- 鼓励多查询命中（提高召回质量）
- 鼓励方法多样性（提高结果可靠性）

---

## 🎨 前端增强

### 1. 检索算法选择

新增下拉框，支持选择：
- **基础检索** (Basic Retrieval)
- **SSG图遍历** (SSG Traversal Algorithm)
- **三角测量** (Triangulation FullDim Algorithm)
- **混合扩展** (Hybrid Expansion) - 试验性

### 2. 参数配置

当选择SSG、Triangulation或Hybrid Expansion时，显示：
- **最大跳数** (Max Hops): 默认4
- **邻居数量** (Neighbor K): 默认30

### 3. 初始召回结果展示增强

新增"初始召回结果"Tab，显示详细信息：
- **最大相似度**：原始向量相似度最高分（未加权前）
- **命中次数**：被多个查询命中的次数
- **方法多样性**：被多少种检索方法命中
- **最终得分**：加权后的最终得分
- **检索方法来源**：彩色tag显示（基础检索、SSG扩展、Triangulation扩展）

### 4. UI优化

- 修复字体颜色问题，确保文字在浅色背景上清晰可见
- 检索方法来源以tag样式显示，类似"命中2次"的badge
- 优化得分详情展示，使用4列布局

---

## 📊 性能影响

### 召回数量

- **Basic Retrieval**: 可配置（默认20-50个chunks）
- **SSG Traversal**: 有限（1 + max_hops，默认5个chunks）
- **Triangulation**: 有限（1 + max_hops，默认5个chunks）
- **Hybrid Expansion**: 扩展（基础50 + SSG扩展 + Triangulation扩展）

### 计算开销

- **Basic Retrieval**: 低（单次向量检索）
- **SSG Traversal**: 中（需要多次向量检索和邻居查找）
- **Triangulation**: 中高（需要多次向量检索和几何计算）
- **Hybrid Expansion**: 高（组合三种方法）

### 质量提升

- **方法多样性奖励**：被多种方法命中的chunk更可靠
- **深度探索**：SSG和Triangulation可以发现间接相关的内容
- **平衡策略**：Hybrid Expansion结合多种方法的优势

---

## 🔄 向后兼容性

### 数据结构兼容

- 支持5元组（旧格式）和6元组（新格式）
- 自动检测格式并设置默认值

### API兼容

- `reranking.py` 同时支持 `vector_similarity` 和 `distance` 字段
- `questions_processing.py` 同时支持新旧字段名

### 配置兼容

- 未指定检索方法时，默认使用 `basic`
- 未指定 `retrieval_source` 时，默认标记为 `"basic"`

---

## 📝 代码变更总结

### 核心文件

1. **`src/retrieval.py`**
   - 新增 `_ssg_search()` 方法
   - 新增 `_triangulation_search()` 方法
   - 实现 `hybrid_expansion` 检索方法
   - 更新聚合逻辑，支持 `retrieval_source` 追踪
   - 实现新的聚合策略（方法多样性奖励）

2. **`src/reranking.py`**
   - 更新字段名：`distance` → `vector_similarity`
   - 保持向后兼容

3. **`src/questions_processing.py`**
   - 更新字段名处理
   - 新增 `retrieval_sources` 字段格式化

4. **`app_jinpan_qa.py`**
   - 新增检索算法选择UI
   - 新增参数配置UI（max_hops, neighbor_k）
   - 增强初始召回结果展示
   - 优化UI样式和颜色

### 默认参数更新

- **max_hops**: 2 → 4
- **neighbor_k**: 10 → 30

---

## 🚀 使用建议

### 场景1：快速准确检索
- **推荐**：Basic Retrieval
- **特点**：速度快，结果直接相关

### 场景2：深度探索语义关联
- **推荐**：SSG Traversal
- **特点**：可以发现跨文档的逻辑链路

### 场景3：平衡相关性和探索性
- **推荐**：Triangulation FullDim
- **特点**：比基础检索更有探索性，比SSG更保守

### 场景4：最大化召回质量
- **推荐**：Hybrid Expansion
- **特点**：结合三种方法的优势，适合高质量要求的场景

---

## 🔍 调试与监控

### 调试日志

新增调试日志，追踪：
- 每个chunk的 `retrieval_sources` 原始列表
- 去重后的 `unique_methods`
- 方法多样性数量

### 前端展示

- 初始召回结果Tab显示详细的检索方法来源
- 方法多样性以tag样式展示
- 得分详情包含完整的加权说明

---

## 📚 相关文档

- [移植指南：SSG Traversal 和 Triangulation FullDim](./移植指南_SSG_Traversal_和_Triangulation_FullDim.md)
- [基础检索逻辑分析](./BASIC_RETRIEVAL_LOGIC_ANALYSIS.md)

---

## ✅ 测试建议

1. **功能测试**：
   - 测试四种检索方法的独立运行
   - 测试Hybrid Expansion的组合效果
   - 验证方法多样性奖励机制

2. **性能测试**：
   - 对比不同检索方法的召回数量
   - 测量计算开销
   - 评估质量提升

3. **兼容性测试**：
   - 验证向后兼容性
   - 测试旧格式数据的处理

---

## 🎯 未来优化方向

1. **参数自适应**：根据查询类型自动调整检索方法
2. **动态Top-K**：根据基础检索结果质量动态调整扩展数量
3. **权重优化**：基于评估结果优化奖励权重
4. **并行优化**：进一步优化Hybrid Expansion的并行执行

---

**文档维护**: 请及时更新本文档以反映最新的代码变更和功能改进。

