# 检索算法移植指南

## 概述

本指南帮助您将 `ssg_traversal` 和 `triangulation_geometric_fulldim` 两种检索方法从本项目移植到您的项目中。您的项目目前使用类似 `basic_retrieval` 的方法。

---

## 一、算法对比：从 Basic Retrieval 到高级遍历

### 1.1 Basic Retrieval（您当前的方法）

**核心思路：**
```
1. 查询 → 嵌入向量
2. 计算查询与所有chunk的相似度
3. 按相似度排序，选择top-k
4. 从选中的chunks中提取句子
```

**特点：**
- 无图遍历，直接相似度排序
- 简单快速，但可能遗漏语义相关的间接连接

---

### 1.2 SSG Traversal（方法3）

**核心思路：**
```
1. 找到与查询最相似的anchor chunk
2. 从anchor开始，遍历到与当前chunk最相似的chunk（不是与查询相似）
3. 每次跳转基于 chunk-to-chunk 相似度
4. 防止重复：跳过包含已提取句子的chunk
5. 早停：当下一跳相似度 ≤ 上一跳相似度时停止
```

**关键差异：**
- ✅ 基于图结构遍历，而非全局排序
- ✅ 探索性更强，能发现间接语义连接
- ✅ 防止句子重复提取
- ✅ 探索潜力早停机制

**伪代码流程：**
```
current_chunk = anchor_chunk
extracted_sentences = []
previous_similarity = 0.0

while (未达到最小句子数 and 未超过最大跳数):
    # 获取当前chunk的所有连接chunks
    connected_chunks = get_connected_chunks(current_chunk)
    
    # 过滤：跳过包含已提取句子的chunk
    candidates = filter_chunks_without_overlap(connected_chunks, extracted_sentences)
    
    # 计算chunk-to-chunk相似度
    similarities = [calculate_similarity(current_chunk, chunk) for chunk in candidates]
    
    # 选择最相似的chunk
    best_chunk, best_similarity = max(similarities)
    
    # 早停检查：探索潜力下降
    if best_similarity <= previous_similarity:
        break
    
    # 跳转到新chunk并提取句子
    current_chunk = best_chunk
    new_sentences = extract_sentences(best_chunk)
    extracted_sentences.extend(new_sentences)
    previous_similarity = best_similarity
```

---

### 1.3 Triangulation Geometric FullDim（方法6）

**核心思路：**
```
1. 找到anchor chunk
2. 在完整嵌入空间中构建几何三角形：
   - 顶点1：查询向量
   - 顶点2：当前chunk向量
   - 顶点3：候选节点向量（chunk或sentence）
3. 计算三角形质心到查询的距离
4. 选择质心距离最小的候选节点
5. 如果最佳候选是sentence → 停止（找到最优提取点）
6. 如果最佳候选是chunk → 跳转并继续
```

**关键差异：**
- ✅ 使用几何三角测量，而非直接相似度
- ✅ 在完整嵌入空间（1024D等）中计算，保留所有信息
- ✅ 同时考虑chunk和sentence节点
- ✅ 质心距离作为选择标准

**伪代码流程：**
```
query_emb = embed(query)
current_chunk = anchor_chunk
extracted_sentences = []

while (未达到最小句子数 and 未超过最大跳数):
    current_emb = get_embedding(current_chunk)
    
    # 获取混合连接（chunks + sentences）
    candidates = get_hybrid_connections(current_chunk)
    
    # 对每个候选节点计算几何三角形
    triangle_metrics = []
    for candidate_id, candidate_type in candidates:
        candidate_emb = get_embedding(candidate_id)
        
        # 构建三角形：query, current, candidate
        centroid = (query_emb + current_emb + candidate_emb) / 3.0
        centroid_distance = distance(centroid, query_emb)
        
        triangle_metrics.append({
            'node_id': candidate_id,
            'node_type': candidate_type,
            'centroid_distance': centroid_distance
        })
    
    # 选择质心距离最小的候选
    best_triangle = min(triangle_metrics, key=lambda x: x['centroid_distance'])
    
    # 决策逻辑
    if best_triangle.node_type == "sentence":
        # 最优提取点，停止
        break
    elif best_triangle.node_type == "chunk":
        # 跳转到新chunk
        current_chunk = best_triangle.node_id
        new_sentences = extract_sentences(current_chunk)
        extracted_sentences.extend(new_sentences)
```

---

## 二、核心依赖和基础设施

### 2.1 必需的数据结构

**语义相似度图（Semantic Similarity Graph）：**
- 存储chunks、sentences及其连接关系
- 提供embedding访问接口
- 支持intra-document和inter-document连接查询

**关键接口：**
```
- get_chunk_embedding(chunk_id) → embedding vector
- get_sentence_embedding(sentence_id) → embedding vector
- get_chunk_sentences(chunk_id) → List[sentence]
- chunk.intra_doc_connections → List[chunk_id]
- chunk.inter_doc_connections → List[chunk_id]
```

### 2.2 基础工具类

**BaseRetrievalAlgorithm（抽象基类）：**
- 提供通用方法：句子提取、去重、相似度计算
- 管理配置参数：max_hops, similarity_threshold等
- 共享embedding模型（内存优化）

**关键方法：**
```
- get_chunk_sentences(chunk_id) → List[str]
- calculate_chunk_similarity(chunk1, chunk2) → float
- deduplicate_sentences(new, existing) → List[str]
- get_hybrid_connections(chunk_id) → List[(node_id, type, similarity)]
```

---

## 三、移植步骤

### 3.1 准备阶段

**1. 确认您的图结构：**
- 是否有chunk级别的连接关系？
- 是否有embedding向量存储？
- 是否支持intra/inter-document连接？

**2. 评估现有代码：**
- 找到您当前的basic_retrieval实现
- 识别可复用的部分（embedding计算、相似度计算等）
- 确定需要新增的部分（图遍历逻辑）

### 3.2 实现 SSG Traversal

**步骤1：创建算法类**
```python
class SSGTraversalAlgorithm:
    def __init__(self, graph, config):
        # 初始化：图、配置、embedding模型
        pass
    
    def retrieve(self, query, anchor_chunk):
        # 主检索逻辑
        pass
```

**步骤2：实现核心遍历循环**
- 从anchor chunk开始
- 获取连接的chunks
- 过滤包含已提取句子的chunks
- 计算chunk-to-chunk相似度
- 选择最佳chunk并跳转
- 实现早停机制

**步骤3：集成到现有系统**
- 替换或扩展您的retrieval接口
- 保持返回格式兼容（sentences列表）
- 添加配置参数支持

### 3.3 实现 Triangulation FullDim

**步骤1：创建算法类**
```python
class TriangulationFullDimAlgorithm:
    def __init__(self, graph, config):
        # 初始化：图、配置、embedding模型
        # 自动检测embedding维度
        pass
    
    def retrieve(self, query, anchor_chunk):
        # 主检索逻辑
        pass
```

**步骤2：实现几何三角测量**
- 获取query、current、candidate的embeddings
- 计算三角形质心
- 计算质心到query的距离
- 对chunk和sentence候选都进行计算

**步骤3：实现决策逻辑**
- 如果最佳候选是sentence → 停止
- 如果最佳候选是chunk → 跳转并继续
- 处理已访问chunk的情况（选择次优）

---

## 四、关键配置参数

### 4.1 SSG Traversal 参数

```yaml
ssg_traversal:
  max_hops: 5                    # 最大跳数
  similarity_threshold: 0.3       # 相似度阈值（过滤低质量连接）
  min_sentence_threshold: 10      # 最小句子数
  max_results: 10                 # 最终返回的句子数
  enable_early_stopping: true     # 启用早停机制
```

### 4.2 Triangulation FullDim 参数

```yaml
triangulation_fulldim:
  max_hops: 5                     # 最大跳数
  similarity_threshold: 0.3       # 相似度阈值
  min_sentence_threshold: 10      # 最小句子数
  max_results: 10                # 最终返回的句子数
  # embedding_dimension: 自动检测 # 嵌入维度（自动从图检测）
```

---

## 五、与现有系统集成

### 5.1 接口兼容性

**保持返回格式一致：**
```python
# 您的现有接口可能类似：
def retrieve(query: str) -> List[str]:
    # 返回句子列表
    pass

# 新算法应该也返回相同格式
def ssg_traverse(query: str) -> List[str]:
    result = ssg_algorithm.retrieve(query, anchor_chunk)
    return result.retrieved_content  # List[str]
```

### 5.2 渐进式集成

**方案1：并行运行**
- 保留basic_retrieval作为fallback
- 新增ssg_traversal和triangulation_fulldim作为选项
- 通过配置选择算法

**方案2：替换**
- 直接替换basic_retrieval
- 需要充分测试确保兼容性

**方案3：混合策略**
- 根据查询类型选择算法
- 简单查询用basic_retrieval，复杂查询用遍历算法

---

## 六、性能考虑

### 6.1 计算开销

**SSG Traversal：**
- 每次跳转需要计算chunk-to-chunk相似度
- 需要检查句子重叠（去重）
- 总体开销：中等（比basic_retrieval慢，但比LLM引导快）

**Triangulation FullDim：**
- 需要计算几何三角形（3个向量的质心）
- 需要处理chunk和sentence两种节点类型
- 总体开销：中等偏高（几何计算）

### 6.2 优化建议

1. **缓存相似度：** 缓存chunk-to-chunk相似度，避免重复计算
2. **共享embedding模型：** 多个检索共享同一个embedding模型实例
3. **早停机制：** 充分利用早停，避免不必要的遍历
4. **批量处理：** 如果可能，批量计算相似度

---

## 七、测试和验证

### 7.1 功能测试

1. **基本功能：** 确保能返回句子列表
2. **遍历逻辑：** 验证跳转路径合理
3. **去重机制：** 确保不返回重复句子
4. **早停机制：** 验证在适当时机停止

### 7.2 性能测试

1. **响应时间：** 与basic_retrieval对比
2. **检索质量：** 评估检索到的内容相关性
3. **资源使用：** 内存和CPU占用

### 7.3 对比测试

- 使用相同查询测试三种方法
- 比较返回结果的差异
- 评估哪种方法更适合您的用例

---

## 八、常见问题和解决方案

### 8.1 图结构不完整

**问题：** 您的图可能没有chunk之间的连接关系

**解决方案：**
- 构建连接关系：基于相似度阈值建立连接
- 使用embedding相似度动态计算连接
- 简化实现：只使用intra-document连接

### 8.2 Embedding维度不匹配

**问题：** 您的embedding维度可能与项目不同

**解决方案：**
- Triangulation FullDim会自动检测维度
- 确保embedding向量格式正确（numpy array）
- 检查向量归一化（如果需要）

### 8.3 性能问题

**问题：** 遍历算法比basic_retrieval慢

**解决方案：**
- 这是正常的，遍历算法需要更多计算
- 优化相似度计算（使用向量化操作）
- 减少max_hops参数
- 启用早停机制

---

## 九、总结

### 9.1 核心思路对比

| 方法 | 选择标准 | 遍历方式 | 优势 |
|------|---------|---------|------|
| Basic Retrieval | 查询- chunk相似度 | 无遍历，全局排序 | 快速、简单 |
| SSG Traversal | Chunk- chunk相似度 | 图遍历，探索性 | 发现间接连接 |
| Triangulation FullDim | 几何质心距离 | 图遍历，几何优化 | 更精确的语义定位 |

### 9.2 选择建议

- **SSG Traversal：** 适合需要探索性检索的场景，能发现间接语义连接
- **Triangulation FullDim：** 适合需要精确语义定位的场景，几何方法更严谨

### 9.3 移植优先级

1. **先移植SSG Traversal：** 相对简单，逻辑清晰
2. **再移植Triangulation FullDim：** 需要几何计算，但效果更好
3. **最后优化：** 根据实际使用情况调整参数和性能

---

## 十、参考资源

### 10.1 关键文件位置

- `utils/algos/ssg_traversal.py` - SSG Traversal实现
- `utils/algos/triangulation_geometric_fulldim.py` - Triangulation FullDim实现
- `utils/algos/base_algorithm.py` - 基础算法类
- `utils/algos/basic_retrieval.py` - Basic Retrieval实现（参考）

### 10.2 核心概念

- **Semantic Similarity Graph：** 语义相似度图结构
- **Traversal Path：** 遍历路径记录
- **Hybrid Connections：** 混合连接（chunk + sentence）
- **Early Stopping：** 早停机制

---

**祝移植顺利！如有问题，请参考源码实现细节。**

