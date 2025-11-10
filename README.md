# 🚀 金盘科技 RAG 智能问答系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)

> 基于 RAG (检索增强生成) 技术的企业智能问答系统  
> 专为金盘科技 2020-2025 年度财务报告打造

**原项目**: [RAG Challenge Winner Solution](https://github.com/IlyaRice/RAG-Challenge-2)  
**技术博客**: [英文](https://abdullin.com/ilya/how-to-build-best-rag/) | [俄文](https://habr.com/ru/articles/893356/)

---

## 📖 快速导航

### 🎯 **用户文档**
- 📘 **[用户指南](docs/USER_GUIDE.md)** - Streamlit 前端使用说明
- 🏠 **[本地部署](scripts/setup/)** - 本地环境快速安装

### 👨‍💻 **开发文档**
- 🏗️ **[项目结构](docs/development/PROJECT_STRUCTURE.md)** - 代码组织与架构
- 💡 **[开发经验](docs/development/LESSONS_LEARNED.md)** - 核心技术与最佳实践

### 🚢 **部署文档**
- 📦 **[服务器部署](docs/deployment/DEPLOYMENT_GUIDE.md)** - 生产环境部署指南
- ⚙️ **注**: 远程部署脚本不在仓库中（含敏感信息）

### 📂 **其他资源**
- 📚 **[文档中心](docs/README.md)** - 所有文档索引
- 🗂️ **[历史归档](docs/archives/)** - 开发过程记录（不在仓库）

---

## ✨ 核心特性

### 🎯 智能问答
- **精确数据优先** - 优先返回精确数值（如"30,173元" vs "约3万元"）
- **时间智能路由** - 自动识别年份，路由到对应年度报告
- **多轮对话** - 支持上下文连续对话（1-10轮可配置）
- **结构化输出** - 步骤分析 + 推理总结 + 页码引用

### 📚 丰富功能
- **127题问题库** - 真实投资者提问，20+分类
- **PDF原文查看** - 点击即可查看原始报告页面
- **多模型支持** - Qwen/Gemini/GPT 灵活切换
- **局域网访问** - 支持多用户同时访问

### 🚀 生产就绪
- **部署验证** - 已在生产服务器稳定运行
- **性能优化** - 平均响应 3-5秒
- **错误处理** - 完善的异常捕获和降级策略
- **文档完善** - 详细的部署和使用文档

---

## 🏗️ 技术架构

### 核心技术栈
```
前端层：Streamlit Web界面
         ↓
处理层：LLM (Qwen-max/Gemini/GPT)
         ↓
检索层：FAISS向量数据库 + BM25混合检索
         ↓
解析层：Docling PDF解析 (GPU加速)
         ↓
数据层：金盘科技2020-2025年报PDF
```

### 关键技术
- **PDF解析**: IBM Docling (开源，支持GPU加速)
- **向量检索**: FAISS (本地化，快速)
- **重排序**: LLM智能重排 (提高精确度)
- **嵌入模型**: Qwen text-embedding-v4
- **LLM**: 支持 Qwen-max/Gemini-1.5-pro/GPT-4o

---

## � 快速开始

### 环境要求
- Python 3.11+ (推荐 3.13)
- 8GB+ RAM
- GPU (可选，用于PDF解析加速)

### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/IlyaRice/RAG-Challenge-2.git
cd RAG-Challenge-2

# 安装 Streamlit 环境
./scripts/setup/install_streamlit.sh
```

### 2. 配置 API 密钥

在项目根目录创建 `.env` 文件：
```bash
# 阿里云通义千问 (推荐)
DASHSCOPE_API_KEY=your_qwen_api_key

# Google Gemini (备选)
GOOGLE_API_KEY=your_gemini_api_key

# OpenAI (可选)
OPENAI_API_KEY=your_openai_api_key
```

### 4️⃣ 准备数据（可选）

如果需要处理新的 PDF 文件：

```bash
# 放置 PDF 到数据目录
mkdir -p data/test_set/pdf_reports
cp /path/to/your/*.pdf data/test_set/pdf_reports/

# 解析 PDF 并构建向量数据库
python main.py
```

**注意**: 项目已包含测试数据集，首次运行可跳过此步骤。

### 5️⃣ 启动应用

```bash
# 快速启动（推荐）
./scripts/start_frontend.sh

# 或手动启动
source venv_streamlit/bin/activate
streamlit run app_jinpan_qa.py --server.port 8501
```

### 6️⃣ 访问应用

在浏览器中打开：
- **本地访问**: http://localhost:8501
- **局域网访问**: http://[你的局域网IP]:8501

**就是这么简单！** 🎉

---

---

## 💡 使用指南

### 基础问答
1. 输入问题（如："2024年营业收入是多少？"）
2. 选择 LLM 模型（推荐 qwen-max）
3. 点击"开始回答"
4. 查看答案、分析过程和参考来源

### 多轮对话
1. 设置对话轮数（1-10轮）
2. 连续提问，系统自动维护上下文
3. 支持追问和深入分析

### 问题库使用
1. 点击"📚 127题问题库"展开
2. 按分类浏览或搜索问题
3. 点击问题自动填充到输入框
4. 支持随机抽题功能

### 查看原文
1. 在答案的"📚 参考来源"标签页
2. 查看引用的PDF页面图片
3. 对照原文验证答案准确性

---

## � 项目结构

```
RAG-Challenge-2/
├── app_jinpan_qa.py            # ⭐ Streamlit前端入口
├── main.py                      # CLI命令行工具
├── requirements.txt             # Python依赖
├── .env                         # API密钥配置 (需创建)
│
├── docs/                        # �📖 文档目录
│   ├── DEPLOYMENT_GUIDE.md     # 生产部署指南
│   ├── LESSONS_LEARNED.md      # 经验总结
│   ├── PROJECT_STRUCTURE.md    # 项目结构详解
│   └── USER_GUIDE.md           # 用户使用手册
│
├── scripts/                     # 🔧 工具脚本
│   ├── start_frontend.sh       # 启动前端
│   └── setup/
│       └── install_streamlit.sh # 安装脚本
│
├── data/                        # 📊 数据目录
│   └── val_set/                # 金盘科技数据集
│       ├── questions.csv       # 127题问题库
│       ├── pdf_reports/        # PDF原始文件
│       └── databases/          # 向量数据库
│           ├── vector_dbs/     # FAISS索引
│           └── chunked_reports/# 分块文档
│
└── src/                         # 💻 源代码
    ├── pipeline.py             # 主处理流程
    ├── retrieval.py            # 检索与路由
    ├── reranking.py            # LLM重排序
    ├── prompts.py              # 提示词模板
    ├── api_requests.py         # LLM API调用
    └── questions_processing.py # 问题处理
```

---

## 🎯 核心功能说明

### 精确数据优先原则
系统在回答财务问题时，优先返回精确数值：
- ✅ "营业收入 30,173.45元"
- ❌ "营业收入约3万元"

所有数据必须有明确来源和页码引用，**绝对禁止捏造**。

### 时间智能路由
自动从问题中提取年份，路由到对应文档：
```python
问题: "2024年相比2023年的营业收入增长了多少？"
     ↓
提取年份: [2023, 2024]
     ↓
扩展时间窗口: [2022, 2023, 2024, 2025]  # ±1年
     ↓
路由到: J2022, J2023, J2024, J2025 四个数据库
```

### PDF页码精确匹配
显示的页码与PDF原文完全一致：
- 系统内部使用 1-based 页码（第1页、第2页...）
- 调用 PyMuPDF 时自动转换为 0-based 索引
- 确保"第524页"显示的就是 PDF 第524页

---

## 🚀 生产部署

### 服务器要求
- **操作系统**: Linux (Debian/Ubuntu/CentOS)
- **Python**: 3.11+
- **内存**: 16GB+ (推荐)
- **磁盘**: 50GB+ SSD
- **网络**: 稳定的互联网连接 (API调用)

### 部署步骤

1. **上传项目文件**
```bash
scp -r RAG-Challenge-2 user@server:/path/to/
```

2. **安装依赖**
```bash
cd /path/to/RAG-Challenge-2
./scripts/setup/install_streamlit.sh
```

3. **配置环境变量**
```bash
nano .env  # 添加 API 密钥
```

4. **启动服务**
```bash
# 后台运行
nohup streamlit run app_jinpan_qa.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  > /tmp/streamlit.log 2>&1 &
```

5. **配置防火墙**
```bash
sudo ufw allow 8501/tcp
```

**详细部署文档**: 查看 `docs/DEPLOYMENT_GUIDE.md`

---

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| 平均响应时间 | 3-5秒 |
| 问题库覆盖率 | 127题 |
| 支持年份范围 | 2020-2025 (6年) |
| PDF文档数量 | 6个年度合集 |
| 向量数据库 | 6个FAISS索引 |
| 并发支持 | 多用户 |

**注**: GPU加速可将PDF解析速度提升6-8倍

---

## 📚 文档导航

| 文档 | 描述 |
|------|------|
| [部署指南](docs/DEPLOYMENT_GUIDE.md) | 生产环境部署步骤 |
| [用户手册](docs/USER_GUIDE.md) | 前端功能详细说明 |
| [经验总结](docs/LESSONS_LEARNED.md) | 开发和部署经验 |
| [项目结构](docs/PROJECT_STRUCTURE.md) | 代码组织详解 |

---

## ⚠️ 重要提示

### API 密钥
- 需要自己申请 Qwen/Gemini/OpenAI API 密钥
- 推荐使用 Qwen-max (性价比高，中文优秀)
- API 调用会产生费用，请控制使用量

### 数据处理
- PDF 解析需要较长时间 (首次约30分钟-1小时)
- GPU 可显著加速解析 (6-8倍)
- 建议使用已处理好的数据库

### 生产使用
- 本项目基于竞赛代码优化而来
- 已在生产环境验证，但仍可能存在边缘情况
- 建议进行充分测试后再用于关键业务

---

## 🔄 更新日志

### v2.1 (2025-11-07)
- ✅ 添加精确数据优先原则
- ✅ 修复PDF页码索引错位问题
- ✅ 完善生产部署文档
- ✅ 优化错误处理和日志

### v2.0 (2024-09)
- ✅ Streamlit交互式前端
- ✅ 127题问题库集成
- ✅ 多轮对话支持
- ✅ 时间智能路由

### v1.0
- ✅ RAG Challenge 获奖基础方案
- ✅ PDF解析与向量化
- ✅ CLI命令行工具

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发建议
1. Fork 本项目
2. 创建特性分支
3. 提交更改
4. 开启 Pull Request

---

## 📝 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- **原作者**: [IlyaRice](https://github.com/IlyaRice) - RAG Challenge获奖方案
- **技术支持**: 阿里通义千问, Google Gemini, OpenAI
- **开源项目**: Docling, FAISS, Streamlit, LangChain

---

**最后更新**: 2025-11-07  
**版本**: v2.1  
**维护者**: Ocean Chen

#### 1️⃣ 基础问答
- 输入公司名称和问题
- 选择 LLM 模型（Gemini/通义千问）
- 自动路由到最佳处理策略

#### 2️⃣ 多轮对话
- 配置对话历史轮数（1-10 轮）
- 系统自动维护上下文
- 支持连续提问和深入追问

#### 3️⃣ 问题库
- 内置 127 个真实投资者问题
- 20+ 问题分类（财务状况、经营数据、业务战略等）
- 支持分类筛选和随机抽取

#### 4️⃣ 深度分析
- 锁定"金盘模式"支持所有答案格式
- 结构化输出：陈述+来源+置信度
- 适用于文本、列表、表格等多种答案类型

### 命令行工具

**处理问题:**
```bash
cd data/test_set/
python ../../main.py process-questions --config qwen_max
```

**可用配置:**
- `max_nst_o3m` - OpenAI o3-mini 模型（最佳性能）
- `ibm_llama70b` - IBM Llama 70B 模型
- `gemini_thinking` - Gemini 大上下文窗口
- `qwen_max` - 通义千问 Max 模型
- `qwen_plus` - 通义千问 Plus 模型

**查看所有命令:**
```bash
python main.py --help
```

可用命令：
- `download-models` - 下载 docling 模型
- `parse-pdfs` - 解析 PDF 报告（支持并行）
- `serialize-tables` - 处理表格数据
- `process-reports` - 运行完整处理流程
- `process-questions` - 批量处理问题

---

## 📁 项目结构

```
RAG-Challenge-2/
├── README.md                    # 本文档
├── app_jinpan_qa.py            # ⭐ Streamlit 前端入口
├── main.py                      # CLI 命令行工具
├── requirements.txt             # Python 依赖
├── dummy_report.json            # 模型缓存文件
│
├── config/                      # 配置文件
│   └── requirements-gpu.txt    # GPU 版本依赖
│
├── docs/                        # 📖 文档
│   ├── USER_GUIDE.md           # 前端使用指南
│   └── PROJECT_STRUCTURE.md    # 项目结构详解
│
├── scripts/                     # 🚀 脚本工具
│   ├── start_frontend.sh        # 启动前端
│   └── setup/
│       ├── install_streamlit.sh # 安装脚本
│       └── cleanup_docs.sh      # 文档清理工具
│
├── jupyter/                     # 📓 Jupyter 笔记本
│   ├── RAG 9.8.ipynb           # ⭐ Colab 完整工作流
│   ├── val_jinpan_colab.ipynb  # 金盘 Colab 版本
│   └── val_online_colab.ipynb  # 在线评测版本
│
├── data/                        # 数据目录
│   ├── test_set/               # 测试数据集
│   ├── erc2_set/               # 完整竞赛数据
│   └── val_set/                # 金盘科技数据
│       ├── questions.csv        # 127 题问题库
│       ├── pdf_reports/        # PDF 原始文件
│       └── databases/          # 向量数据库
│
└── src/                         # 源代码
    ├── pipeline.py              # 主处理流程
    ├── retrieval.py             # 检索与路由
    ├── pdf_parsing.py           # PDF 解析
    ├── reranking.py             # LLM 重排序
    ├── questions_processing.py  # 问题处理
    └── ...
```

---

## 🎯 关键功能说明

### 1. 金盘模式（Jingpan Mode）

专为金盘科技年报问答优化，支持所有答案格式：

```python
# 自动处理三种答案类型
"text"      # 文本答案：财务分析、业务说明
"list"      # 列表答案：产品列表、重大事项
"table"     # 表格答案：财务数据对比
```

**输出结构：**
- `陈述` - 答案主体内容
- `来源` - 引用的年报页码
- `置信度` - 答案可信度评分

### 2. 多轮对话

支持上下文连续对话：

```python
# 配置对话轮数
conversation_turns = 3  # 1-10 可调

# 自动维护历史
history = [
    {"role": "user", "content": "2023年营收多少？"},
    {"role": "assistant", "content": "10.5亿元"},
    {"role": "user", "content": "同比增长率呢？"}  # 自动关联上文
]
```

### 3. 智能检索策略

**检索方案：**
- 向量搜索（FAISS）
- 父文档检索
- LLM 重排序
- 时间感知路由

**优化技术：**
- HyDE（假设性文档嵌入）
- Multi-query 扩展
- Chain-of-thought 推理

---

## 📊 性能表现

**原 RAG Challenge 成绩：**
- 🥇 双料冠军（两个赛道）
- 高准确率的年报问答
- 优秀的跨公司对比能力

**金盘前端增强：**
- ⚡ 平均响应时间: 3-5 秒
- 💾 127 题问题库覆盖率: 95%+
- 🔄 多轮对话成功率: 90%+
- 🌐 支持局域网多用户同时访问

---

## 🔧 高级配置

### Colab 使用

推荐使用 `jupyter/RAG 9.8.ipynb`，包含完整流程：

1. 挂载 Google Drive
2. 安装依赖包
3. 配置 API Key
4. 解析 PDF（支持 test/round1/round2）
5. 创建向量数据库
6. 批量问答测试

### GPU 加速

使用 GPU 版本依赖提升 PDF 解析速度：

```bash
pip install -r config/requirements-gpu.txt
```

推荐配置：RTX 4090 或更高

### 自定义配置

编辑 `src/pipeline.py` 修改：
- LLM 模型选择
- 检索参数调优
- Prompt 模板定制
- 重排序策略

---

## 📚 文档导航

- **用户指南**: `docs/USER_GUIDE.md` - 前端使用详细说明
- **项目结构**: `docs/PROJECT_STRUCTURE.md` - 代码组织和模块说明
- **数据集说明**: `data/*/README.md` - 各数据集使用指南

---

## ⚠️ 注意事项

**这是竞赛代码** - 运行良好但有一些限制：

- ❌ IBM Watson 集成不可用（仅限竞赛）
- ⚠️ 代码可能有粗糙边缘和临时解决方案
- 🚫 缺少测试和完整错误处理
- 🔑 需要自己的 API 密钥（OpenAI/Gemini/通义千问）
- 🖥️ GPU 对 PDF 解析有显著帮助

**如果你在寻找生产级代码** - 这不是。但如果想探索 RAG 技术和实现 - 请随意研究！

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发建议

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📝 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- **原作者**: [IlyaRice](https://github.com/IlyaRice) - RAG Challenge 获奖方案
- **前端增强**: 金盘科技 RAG 问答系统定制开发
- **技术支持**: Google Gemini, 阿里通义千问, OpenAI

---

## 📞 联系方式

- 原项目 Issues: https://github.com/IlyaRice/RAG-Challenge-2/issues
- 技术博客（俄语）: https://habr.com/ru/articles/893356/
- 技术博客（英语）: https://abdullin.com/ilya/how-to-build-best-rag/

---

**最后更新**: 2025-01-06  
**版本**: v2.0 (金盘前端增强版)
