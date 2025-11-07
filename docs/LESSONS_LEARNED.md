# 📚 项目经验总结

> **项目**: 金盘科技 RAG 智能问答系统  
> **最后更新**: 2025-11-07  
> **版本**: v2.1

---

## 目录

1. [核心经验](#核心经验)
2. [关键问题与解决](#关键问题与解决)
3. [最佳实践](#最佳实践)
4. [部署经验](#部署经验)

---

## 🎓 核心经验

### 1. 提示词工程

**精确数据优先原则** ⭐
- 在多个位置强化：instruction + Schema描述 + 示例
- 使用"绝对禁止"等强约束词汇
- 要求数据可追溯到具体页码
- 举具体例子而非抽象描述

**示例**:
```python
# ✅ 好的提示词
"""
- **精确数据优先**：优先选择精确数值（如"30,173.45元"）
  而非约数（如"约3万元"）
- **绝对禁止**：捏造数据、推测数据
- 所有数据必须能追溯到具体页码
"""

# ❌ 差的提示词
"""
请尽量准确地回答问题。
"""
```

### 2. 索引标准处理

**1-based vs 0-based 问题** ⭐
- **内部统一**: 全局使用 1-based（人类视角）
- **边界转换**: 只在调用外部库时转换
- **注释说明**: 在转换处添加注释
- **文档精确**: 明确标注参数索引起点

**数据流**:
```
Docling(1-based) → 分块(1-based) → 向量DB(1-based) 
→ 检索(1-based) → 显示(1-based) → PyMuPDF(0-based转换)
```

**代码示例**:
```python
# ✅ 正确做法
page_image = get_pdf_page_image(
    pdf_path, 
    page_num - 1  # 转换：1-based → 0-based for PyMuPDF
)

# 函数文档明确标注
def get_pdf_page_image(pdf_path: str, page_num: int):
    """
    Args:
        page_num: 页码索引（0-based，第1页=0）
    """
```

### 3. 时间智能路由

**年份提取与扩展**:
- 从问题中提取年份: 正则 `r'(20\d{2})年'`
- 扩展时间窗口: `[min_year-1, max_year+1]`
- 使用连续范围而非离散年份

**示例**:
```python
问题: "2024年相比2023年增长了多少？"
提取: [2023, 2024]
扩展: [2022, 2023, 2024, 2025]  # 连续范围
路由: J2022, J2023, J2024, J2025
```

**年度报告发布规则**:
- 2024年度报告（含全年数据）通常在2025年发布
- 出现在 J2025 公告合集中
- 检索时需同时考虑当年和次年

### 4. 页码验证机制

**防止幻觉页码** ⭐
```python
def _validate_page_references(claimed_pages, retrieval_results):
    # 1. 只保留真实存在的页码
    validated = [p for p in claimed_pages 
                 if p in retrieved_pages]
    
    # 2. 警告移除的幻觉页码
    if len(validated) < len(claimed_pages):
        print(f"Removed hallucinated pages: {removed}")
    
    # 3. 确保最少2个，最多8个
    if len(validated) < 2:
        validated += top_retrieved_pages
    if len(validated) > 8:
        validated = validated[:8]
    
    return validated
```

---

## 🔧 关键问题与解决

### 问题 1: PDF 页码索引错位

**现象**: 显示"第524页"，实际显示第525页

**根因**: 
- 内部使用 1-based 页码
- PyMuPDF 期望 0-based 索引
- 调用时未转换

**解决**:
```python
# 在调用处减1
page_image = get_pdf_page_image(pdf_path, page_num - 1)
```

**影响**: 所有参考来源的PDF图片显示

---

### 问题 2: 监听地址导致外部无法访问

**现象**: Streamlit只能本机访问

**根因**:
```python
# ❌ 默认配置
streamlit run app.py
# → 监听 127.0.0.1:8501 (只有本机能访问)

# ✅ 正确配置
streamlit run app.py --server.address 0.0.0.0
# → 监听所有网络接口，局域网可访问
```

**说明**:
- `127.0.0.1` = 只有本机 localhost
- `0.0.0.0` = 监听所有网络接口

---

### 问题 3: 虚拟环境占用 Google Drive 空间

**现象**: venv 占用 1.3GB + 34,483个文件

**解决**:
```bash
# 1. 移动到本地
mv venv_streamlit ~/Documents/Python_Venvs/RAG-Challenge-2/

# 2. 创建软链接
ln -s ~/Documents/Python_Venvs/RAG-Challenge-2/venv_streamlit venv_streamlit

# 3. 验证
ls -lh venv_streamlit  # 应显示 -> 指向本地路径
```

**效果**: 节省 Google Drive 1.3GB空间

---

### 问题 4: 依赖包遗漏

**现象**: ImportError: No module named 'rank_bm25'

**原因**: 选择性安装依赖，遗漏关键包

**解决**:
```bash
# ❌ 错误做法
pip install streamlit pandas  # 只装了部分

# ✅ 正确做法
pip install -r requirements.txt  # 完整安装
```

**经验**: 
- 始终使用 `requirements.txt` 完整安装
- 不要手动选择性安装
- 新环境首次安装必须完整

---

### 问题 5: macOS 隐藏文件污染

**现象**: `.DS_Store` 文件上传到服务器导致错误

**预防**:
```bash
# 上传前清理
find . -name ".DS_Store" -delete

# 或在 .gitignore 中添加
echo ".DS_Store" >> .gitignore
```

---

## ✅ 最佳实践

### 开发工作流

1. **本地测试优先**
   ```bash
   # 本地启动测试
   streamlit run app.py --server.port 8501
   
   # 测试通过后再部署
   scp file.py user@server:/path/
   ```

2. **增量部署**
   - 修改后立即部署，避免积累
   - 保持本地和生产同步
   - 每次部署记录变更

3. **日志管理**
   ```bash
   # 统一日志位置
   nohup streamlit run app.py > /tmp/streamlit.log 2>&1 &
   
   # 定期查看
   tail -f /tmp/streamlit.log
   ```

### 代码质量

1. **注释关键转换**
   ```python
   # ✅ 好的注释
   page_idx = page_num - 1  # 转换: 1-based → 0-based for PyMuPDF
   
   # ❌ 差的注释
   page_idx = page_num - 1  # 减1
   ```

2. **文档精确描述**
   ```python
   def process(page_num: int):
       """
       Args:
           page_num: 页码（1-based，第1页=1）⭐ 明确标注
       """
   ```

3. **错误处理**
   ```python
   try:
       result = api_call()
   except Exception as e:
       logger.error(f"API失败: {e}")
       return default_value  # 降级策略
   ```

### 提示词设计

1. **多层强化**
   - Instruction 层面
   - Schema 描述
   - 具体示例

2. **强约束表达**
   - "绝对禁止"
   - "必须"
   - "不允许"

3. **可追溯性**
   - 要求明确来源
   - 要求页码引用
   - 防止捏造数据

---

## 🚀 部署经验

### 生产环境检查清单

**部署前**:
- [ ] 所有依赖已安装 (`requirements.txt`)
- [ ] API密钥已配置 (`.env` 文件)
- [ ] 数据库文件已准备 (`databases/`)
- [ ] 端口未被占用 (8501)
- [ ] 监听地址正确 (`0.0.0.0`)

**部署后**:
- [ ] 服务启动成功
- [ ] 日志无错误
- [ ] 本机可访问
- [ ] 局域网可访问
- [ ] 功能测试通过

### 快速部署命令

```bash
# 1. 上传文件
scp -r RAG-Challenge-2 user@server:/path/to/

# 2. 安装依赖
cd /path/to/RAG-Challenge-2
pip install -r requirements.txt

# 3. 配置环境
nano .env  # 添加API密钥

# 4. 启动服务
nohup streamlit run app_jinpan_qa.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  > /tmp/streamlit.log 2>&1 &

# 5. 验证
tail -f /tmp/streamlit.log
curl http://localhost:8501  # 应返回HTML
```

### 常见错误速查

| 错误 | 原因 | 解决 |
|------|------|------|
| ModuleNotFoundError | 依赖未装 | `pip install -r requirements.txt` |
| Address already in use | 端口占用 | `pkill -9 -f streamlit` |
| Connection refused | 监听地址错误 | 添加 `--server.address 0.0.0.0` |
| API Key错误 | 环境变量未配置 | 检查 `.env` 文件 |
| 页面空白 | 数据库缺失 | 检查 `data/val_set/databases/` |

---

## 📊 性能优化

### PDF解析优化
- 使用GPU加速（6-8倍提速）
- 并行处理: `--max-workers 3`
- 缓存解析结果

### 检索优化
- 分年份索引（精确路由）
- 混合检索（向量+BM25）
- LLM重排序（提高精度）

### API调用优化
- 使用批处理
- 设置合理超时
- 实现降级策略

---

## 🔮 已知限制

1. **API依赖**: 需要外部LLM API（有成本）
2. **处理时间**: 首次PDF解析需30分钟-1小时
3. **数据更新**: 需要重新解析PDF和建索引
4. **并发限制**: 建议单机<10并发用户

---

## 📌 快速参考

### 关键文件位置
```
src/prompts.py:606-814       # AnswerWithRAGContextJingpanPrompt
src/retrieval.py:17-51       # extract_years_from_question
app_jinpan_qa.py:176-220     # get_pdf_page_image
app_jinpan_qa.py:256-305     # 参考来源显示逻辑
```

### 重要配置
```python
# 时间扩展窗口
expand_window = ±1 year

# 页码验证范围
min_pages = 2
max_pages = 8

# API超时
timeout = 60 seconds

# 并发重排
max_workers = 10
```

---

**维护者**: Ocean Chen  
**项目**: RAG-Challenge-2  
**最后更新**: 2025-11-07
