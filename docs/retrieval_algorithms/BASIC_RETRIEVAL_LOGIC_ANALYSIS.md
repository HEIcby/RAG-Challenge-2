# Basic Retrieval 实现逻辑详细分析

## 整体流程

```
问题 → [HYDE扩展] → [Multi-Query扩展] → [Embedding生成] → [Basic Retrieval] → [结果聚合] → [加权排序] → Reranker → LLM Judging → Generating
```

## 详细步骤分析

### 1. 问题输入
- **输入**: `query` (原始问题)
- **位置**: `VectorRetriever.retrieve_by_company_name()` 第546行

### 2. HYDE 扩展（可选）
- **触发条件**: `use_hyde = True`
- **实现位置**: 第607-659行
- **逻辑**:
  - 使用 LLM (qwen-turbo) 生成假设答案（markdown表格格式）
  - 将假设答案添加到查询列表：`queries.append(fake_answer_str)`
  - 保存生成的文本到 `expansion_texts['hyde_text']`
- **结果**: `queries = [原始问题, HYDE生成的假设答案]` 或 `[原始问题]`

### 3. Multi-Query 扩展（可选）
- **触发条件**: `use_multi_query = True` 且 `multi_query_config` 中有方法启用
- **实现位置**: 第661-749行
- **支持的方法**:
  1. **名词解释 (synonym)**: 为财务专业名词补充详细解释
  2. **指标拆分 (subquestion)**: 将问题拆分为更细的子问题
  3. **情景变体 (variant)**: 生成情景化/变体提问
- **逻辑**:
  - 查找财务概念术语 (`find_financial_concepts`)
  - 为每个启用方法生成扩展查询
  - 提取 `<...>` 包裹的查询
  - 添加到查询列表：`queries.append(扩展查询)`
- **结果**: `queries = [原始问题, HYDE查询?, Multi-Query查询1, Multi-Query查询2, ...]`

### 4. 查询去重
- **实现位置**: 第752-760行
- **逻辑**: 去除重复和空查询
- **结果**: `deduped_queries` (去重后的查询列表)

### 5. Embedding 生成
- **实现位置**: 第767-784行
- **逻辑**:
  - 对每个查询生成 embedding（使用 Qwen embedding API）
  - 存储为: `query_embeddings[q] = np.array(embedding).reshape(1, -1)`
- **结果**: `query_embeddings` 字典: `{查询文本: embedding数组}`

### 6. Basic Retrieval 核心逻辑

#### 6.1 文档定位
- **实现位置**: 第572-590行
- **逻辑**: 使用 `route_reports_by_time()` 根据公司名和时间筛选相关文档
- **结果**: `matching_reports` 列表（匹配的文档）

#### 6.2 并行检索（每个查询 × 每个文档）
- **实现位置**: 第805-836行 (`process_query_for_document` 函数)
- **核心检索代码**:
  ```python
  distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
  ```
  - `vector_db`: FAISS IndexFlatIP（内积索引）
  - `k=actual_top_n`: 每个文档检索 top_n 个 chunks
  - 返回: `distances` (相似度分数，越大越相关), `indices` (chunk索引)

#### 6.3 结果处理
- **实现位置**: 第818-833行
- **逻辑**:
  - 遍历每个检索结果 `(distance, index)`
  - 分数处理: `distance = round(float(distance) * inner_factor, 4)` (inner_factor=1.0)
  - 根据 `return_parent_pages` 决定返回 chunk 还是 page
  - 构建 key: `(sha1, "page"/"chunk", page_id/chunk_index)`
  - 返回: `local_hits = [(key, page_id, text, distance, sha1), ...]`

#### 6.4 并行执行
- **实现位置**: 第842-864行
- **逻辑**:
  - 使用 `ThreadPoolExecutor` 并行处理所有 (查询, 文档) 组合
  - 每个任务调用 `process_query_for_document()`
  - 汇总所有结果到 `aggregated_results`

### 7. 结果聚合
- **实现位置**: 第852-864行
- **逻辑**:
  - 使用 `aggregated_results` 字典聚合相同 key 的结果
  - 相同页面/chunk 被多个查询命中时:
    - 记录所有距离分数: `distances = [distance1, distance2, ...]`
    - 记录命中次数: `count += 1`
- **关键数据结构**:
  ```python
  aggregated_results[key] = {
      "page": page_id,
      "text": text,
      "distances": [distance1, distance2, ...],  # 所有命中的分数
      "count": N,  # 命中次数
      "source_sha1": sha1
  }
  ```

### 8. 加权排序
- **实现位置**: 第868-896行
- **加权规则**: 
  ```python
  weight_factor(count) = 1.0 + 0.2 * (count - 1)
  # 1次命中: ×1.0
  # 2次命中: ×1.2
  # 3次命中: ×1.4
  # ...
  ```
- **逻辑**:
  - 对每个聚合结果:
    - `base_distance = max(info["distances"])`  # 取最大分数
    - `weighted_distance = base_distance * weight_factor(count)`
  - 按 `weighted_distance` **降序**排序（越大越相关）
  - 取前 `top_n` 个结果
- **结果**: `final_results` (排序后的前 top_n 个结果)

### 9. 返回结果
- **实现位置**: 第913-917行
- **返回格式**:
  ```python
  {
      'results': final_results,  # 最终检索结果
      'timing': timing_info,      # 各阶段耗时
      'expansion_texts': expansion_texts  # HYDE和Multi-Query生成的文本
  }
  ```

## 关键设计特点

### 1. 多查询并行检索
- 每个扩展查询（HYDE、Multi-Query）都会独立进行向量检索
- 结果通过聚合和加权机制合并

### 2. 多次命中加权
- 如果一个页面被多个查询命中，会提升其权重
- 使用最大分数 × 加权因子

### 3. 多文档支持
- 支持从多个文档中检索
- 每个文档独立检索 top_n 个结果
- 最终统一排序选择最优结果

### 4. 并行处理
- 所有 (查询, 文档) 组合并行处理
- 使用线程池提高效率

## 重要参数

- `top_n`: 每个文档检索的 chunks 数量
- `inner_factor`: 距离分数缩放因子（当前=1.0）
- `use_hyde`: 是否启用 HYDE 扩展
- `use_multi_query`: 是否启用 Multi-Query 扩展
- `return_parent_pages`: 返回完整页面还是 chunks
- `parallel_workers`: 并行工作线程数

## 后续流程

检索结果会传递到：
1. **Reranker** (`LLMReranker.rerank_documents()`)
   - 使用 LLM 对结果进行重排序
   - 计算 `combined_score = relevance_score * distance`
   
2. **LLM Judging** (评估答案质量)

3. **Generating** (生成最终答案)

## 与改动后版本的对比

**当前版本（main分支）的特点**:
- 简单直接的实现
- 返回元组格式: `(key, page_id, text, distance, sha1)`
- 没有归一化处理
- 没有 retrieval_method 判断分支
- 没有 retrieval_info 元数据

**改动后版本的潜在问题**:
- 添加了 retrieval_method 判断（虽然 basic 分支逻辑相同）
- 返回格式改为字典（添加了额外字段）
- 可能影响性能的地方需要仔细检查

