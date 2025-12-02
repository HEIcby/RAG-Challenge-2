# 🏠 本地部署指南

本指南将帮助您在本地环境快速部署和运行金盘科技 RAG 问答系统。

## 📋 目录

1. [系统要求](#系统要求)
2. [快速开始](#快速开始)
3. [详细步骤](#详细步骤)
4. [常见问题](#常见问题)
5. [进阶配置](#进阶配置)

---

## 💻 系统要求

### 最低配置
- **操作系统**: macOS 10.15+ / Windows 10+ / Ubuntu 20.04+
- **Python**: 3.10 或更高版本（推荐 3.12）
- **内存**: 4GB RAM
- **硬盘**: 5GB 可用空间（包括依赖和数据）

### 推荐配置
- **Python**: 3.12
- **内存**: 8GB+ RAM
- **硬盘**: 10GB+ 可用空间

---

## 🚀 快速开始

### 1️⃣ 克隆项目

```bash
git clone https://github.com/HEIcby/RAG-Challenge-2.git
cd RAG-Challenge-2
```

### 2️⃣ 创建虚拟环境

**macOS/Linux:**
```bash
python3 -m venv venv_streamlit
source venv_streamlit/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv_streamlit
.\venv_streamlit\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv_streamlit
.\venv_streamlit\Scripts\activate.bat
```

### 3️⃣ 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ 配置 API 密钥

创建 `.env` 文件：

```bash
# 复制示例文件
cp .env.example .env

# 编辑配置（使用你喜欢的编辑器）
nano .env
```

在 `.env` 文件中填入你的 API 密钥：

```ini
# Qwen (通义千问) API Key - 主要使用
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# OpenAI API Key - 可选
OPENAI_API_KEY=your_openai_api_key_here
```

**获取 API 密钥：**
- **Qwen (通义千问)**: https://dashscope.console.aliyun.com/
- **OpenAI**: https://platform.openai.com/api-keys

### 5️⃣ 准备数据

```bash
# 确保数据目录存在
mkdir -p data/test_set/databases
mkdir -p data/test_set/pdf_reports

# 放置你的 PDF 报告到 pdf_reports 目录
# cp /path/to/your/reports/*.pdf data/test_set/pdf_reports/
```

### 6️⃣ 构建向量数据库（首次运行）

```bash
python main.py
```

这个过程会：
- 解析 PDF 报告
- 提取文本和表格
- 构建向量数据库
- 大约需要 5-10 分钟（取决于 PDF 数量）

### 7️⃣ 启动应用

```bash
streamlit run app_jinpan_qa.py --server.port 8501
```

### 8️⃣ 访问应用

在浏览器中打开：
```
http://localhost:8501
```

**就是这么简单！** 🎉

---

## 📖 详细步骤

### 步骤 1: 检查 Python 版本

确保你的 Python 版本符合要求：

```bash
python3 --version
# 或
python --version
```

应该显示 `Python 3.10.x` 或更高版本。

**如果版本过低：**

**macOS (使用 Homebrew):**
```bash
brew install python@3.12
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev
```

**Windows:**
从官网下载安装：https://www.python.org/downloads/

### 步骤 2: 克隆或下载项目

**方法 A: 使用 Git (推荐)**
```bash
git clone https://github.com/HEIcby/RAG-Challenge-2.git
cd RAG-Challenge-2
```

**方法 B: 下载 ZIP**
1. 访问 https://github.com/HEIcby/RAG-Challenge-2
2. 点击 "Code" → "Download ZIP"
3. 解压到你的工作目录

### 步骤 3: 创建并激活虚拟环境

虚拟环境可以隔离项目依赖，避免与系统 Python 包冲突。

**macOS/Linux:**
```bash
# 创建虚拟环境
python3 -m venv venv_streamlit

# 激活虚拟环境
source venv_streamlit/bin/activate

# 激活后，你会看到命令行前缀变为 (venv_streamlit)
```

**Windows PowerShell:**
```powershell
# 创建虚拟环境
python -m venv venv_streamlit

# 激活虚拟环境
.\venv_streamlit\Scripts\Activate.ps1

# 如果遇到权限错误，先运行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows CMD:**
```cmd
python -m venv venv_streamlit
.\venv_streamlit\Scripts\activate.bat
```

### 步骤 4: 安装依赖包

```bash
# 升级 pip
pip install --upgrade pip

# 安装所有依赖
pip install -r requirements.txt

# 验证安装
pip list | grep streamlit
```

**预计安装时间**: 3-5 分钟

**主要依赖包**:
- `streamlit` - Web 界面框架
- `dashscope` - 通义千问 API
- `openai` - OpenAI API
- `langchain` - RAG 框架
- `chromadb` - 向量数据库
- `pypdf` - PDF 解析
- `pandas` - 数据处理

### 步骤 5: 配置环境变量

**创建 .env 文件：**

```bash
# 方法 1: 从示例文件复制（如果存在）
cp .env.example .env

# 方法 2: 手动创建
touch .env
```

**编辑 .env 文件：**

```bash
# macOS/Linux
nano .env
# 或
vim .env
# 或使用 VS Code
code .env

# Windows
notepad .env
```

**填入以下内容：**

```ini
# ==================== API 配置 ====================

# Qwen (通义千问) API Key - 必需
# 获取地址: https://dashscope.console.aliyun.com/
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OpenAI API Key - 可选
# 获取地址: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ==================== 模型配置 ====================

# 使用的 LLM 模型
LLM_MODEL=qwen-max

# 温度参数 (0-1, 越高越随机)
TEMPERATURE=0.1

# 最大返回 token 数
MAX_TOKENS=4096

# ==================== RAG 配置 ====================

# 检索的文档数量
TOP_K=5

# 向量数据库类型
VECTOR_DB=chroma

# ==================== Streamlit 配置 ====================

# 服务器端口
STREAMLIT_SERVER_PORT=8501

# 是否启用 CORS
STREAMLIT_SERVER_ENABLE_CORS=false
```

**保存并退出**（nano: Ctrl+X, Y, Enter）

### 步骤 6: 准备数据文件

**数据目录结构：**

```
data/
├── test_set/
│   ├── pdf_reports/         # 放置 PDF 报告
│   │   ├── report_2024.pdf
│   │   └── report_2023.pdf
│   ├── databases/           # 向量数据库（自动生成）
│   ├── questions.json       # 测试问题（已包含）
│   └── subset.csv          # 数据集配置（已包含）
```

**放置 PDF 文件：**

```bash
# 创建目录
mkdir -p data/test_set/pdf_reports

# 复制你的 PDF 文件
cp /path/to/your/reports/*.pdf data/test_set/pdf_reports/

# 或直接拖放文件到该目录
```

**支持的 PDF 格式：**
- ✅ 企业年报
- ✅ 季度报告
- ✅ 财务报表
- ✅ 包含文本和表格的文档

### 步骤 7: 构建向量数据库

**首次运行需要构建数据库：**

```bash
# 确保虚拟环境已激活
source venv_streamlit/bin/activate  # macOS/Linux
# 或
.\venv_streamlit\Scripts\Activate.ps1  # Windows

# 运行数据处理
python main.py
```

**处理过程：**
1. 扫描 `pdf_reports/` 目录
2. 解析 PDF 文档（文本 + 表格）
3. 文本分块和向量化
4. 构建 ChromaDB 向量数据库
5. 保存到 `databases/` 目录

**预计时间：**
- 小数据集（<10 个 PDF）: 3-5 分钟
- 中等数据集（10-50 个 PDF）: 10-20 分钟
- 大数据集（50+ 个 PDF）: 30+ 分钟

**进度提示：**
```
Processing PDFs: 100%|████████████| 10/10 [00:15<00:00, 0.65it/s]
Building vector database...
✅ Database built successfully!
```

### 步骤 8: 启动 Streamlit 应用

```bash
# 确保虚拟环境已激活
source venv_streamlit/bin/activate  # macOS/Linux

# 启动应用
streamlit run app_jinpan_qa.py --server.port 8501
```

**成功启动后会显示：**

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

### 步骤 9: 使用应用

1. **打开浏览器**，访问 `http://localhost:8501`

2. **配置参数**（左侧边栏）：
   - 选择 LLM 模型（qwen-max / qwen-plus）
   - 调整检索文档数量（Top K）
   - 设置温度参数

3. **输入问题**：
   - 在文本框中输入问题
   - 点击"提交问题"或按 Enter

4. **查看结果**：
   - 答案显示在主区域
   - 展开"检索到的上下文"查看引用
   - 检查调试信息（如果启用）

---

## ❓ 常见问题

### 问题 1: Python 版本不符合要求

**错误信息：**
```
Python 3.8.x detected, but 3.10+ is required
```

**解决方案：**

**macOS:**
```bash
brew install python@3.12
# 使用新版本创建虚拟环境
python3.12 -m venv venv_streamlit
```

**Ubuntu:**
```bash
sudo apt install python3.12 python3.12-venv
python3.12 -m venv venv_streamlit
```

**Windows:**
从官网下载并安装最新版本：https://www.python.org/downloads/

### 问题 2: 依赖安装失败

**错误信息：**
```
ERROR: Failed building wheel for xxx
```

**解决方案：**

```bash
# 升级 pip 和 setuptools
pip install --upgrade pip setuptools wheel

# 安装编译工具（如果需要）
# macOS:
xcode-select --install

# Ubuntu:
sudo apt install build-essential python3-dev

# Windows:
# 安装 Visual Studio Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 重新安装
pip install -r requirements.txt
```

### 问题 3: API 密钥错误

**错误信息：**
```
❌ API Key 无效或未配置
```

**解决方案：**

1. **检查 .env 文件是否存在：**
   ```bash
   ls -la .env
   cat .env  # 查看内容（注意不要分享）
   ```

2. **确认 API Key 格式正确：**
   - Qwen: 以 `sk-` 开头
   - OpenAI: 以 `sk-` 开头

3. **重新获取 API Key：**
   - Qwen: https://dashscope.console.aliyun.com/apiKey
   - OpenAI: https://platform.openai.com/api-keys

4. **检查环境变量是否加载：**
   ```bash
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('DASHSCOPE_API_KEY'))"
   ```

### 问题 4: 端口被占用

**错误信息：**
```
OSError: [Errno 48] Address already in use
```

**解决方案：**

**方法 1: 使用其他端口**
```bash
streamlit run app_jinpan_qa.py --server.port 8502
```

**方法 2: 查找并终止占用进程**

**macOS/Linux:**
```bash
# 查找占用 8501 端口的进程
lsof -i :8501

# 终止进程（替换 PID）
kill -9 <PID>
```

**Windows:**
```cmd
# 查找占用进程
netstat -ano | findstr :8501

# 终止进程（替换 PID）
taskkill /PID <PID> /F
```

### 问题 5: PDF 解析失败

**错误信息：**
```
Failed to parse PDF: xxx.pdf
```

**可能原因：**
- PDF 文件损坏
- PDF 是扫描版（纯图片）
- PDF 有密码保护

**解决方案：**

1. **检查 PDF 是否可以正常打开**

2. **确保 PDF 包含可提取的文本**
   ```bash
   # 测试 PDF 文本提取
   python -c "import pypdf; reader = pypdf.PdfReader('path/to/file.pdf'); print(len(reader.pages[0].extract_text()))"
   ```

3. **如果是扫描版 PDF，需要 OCR 处理**
   - 使用 Adobe Acrobat 的 OCR 功能
   - 或使用在线 OCR 工具转换

### 问题 6: 内存不足

**错误信息：**
```
MemoryError: Unable to allocate array
```

**解决方案：**

1. **减少批处理大小**（编辑 `main.py`）
   ```python
   batch_size = 10  # 改为更小的值，如 5
   ```

2. **减少 Top K 检索数量**
   ```python
   TOP_K = 3  # 从 5 减到 3
   ```

3. **关闭其他占用内存的应用**

4. **增加系统交换空间**（Linux）
   ```bash
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### 问题 7: Streamlit 不显示界面

**问题：** 浏览器显示"Site can't be reached"

**解决方案：**

1. **检查服务是否启动成功**
   ```bash
   # 查看进程
   ps aux | grep streamlit
   ```

2. **检查防火墙设置**
   ```bash
   # macOS: 允许 Python
   # Windows: 添加防火墙规则
   ```

3. **尝试使用 0.0.0.0 地址**
   ```bash
   streamlit run app_jinpan_qa.py --server.address 0.0.0.0 --server.port 8501
   ```

4. **清除 Streamlit 缓存**
   ```bash
   rm -rf ~/.streamlit
   ```

---

## ⚙️ 进阶配置

### 自定义 Streamlit 配置

创建 `~/.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 200

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### 使用不同的向量数据库

默认使用 ChromaDB，也可以配置使用 FAISS：

```python
# 在 src/ingestion.py 中修改
VECTOR_DB_TYPE = "faiss"  # 或 "chroma"
```

### 自定义 Prompt

编辑 `src/prompts.py` 中的 prompt 模板：

```python
class AnswerWithRAGContextJingpanPrompt:
    instruction = """
    你的任务是... （自定义指令）
    """
```

### 批处理模式

如果需要批量处理问题：

```bash
# 准备问题文件 questions.json
python -c "
from src.pipeline import run_batch_questions
results = run_batch_questions('data/test_set/questions.json')
print(f'Processed {len(results)} questions')
"
```

### 性能优化

**1. 启用缓存**

在 `app_jinpan_qa.py` 中确保使用了缓存装饰器：

```python
@st.cache_resource
def load_database():
    # 数据库加载逻辑
    pass
```

**2. 预加载模型**

```python
@st.cache_resource
def load_model():
    # 在应用启动时加载模型
    pass
```

**3. 并行处理**

修改 `main.py` 启用多进程：

```python
from multiprocessing import Pool

with Pool(4) as pool:  # 使用 4 个进程
    results = pool.map(process_pdf, pdf_files)
```

---

## 📊 目录结构说明

```
RAG-Challenge-2/
├── app_jinpan_qa.py           # Streamlit 主应用
├── main.py                     # 数据处理入口
├── requirements.txt            # Python 依赖
├── .env                        # 环境变量（需创建）
├── .gitignore                  # Git 忽略规则
├── README.md                   # 项目说明
│
├── src/                        # 源代码目录
│   ├── ingestion.py           # 数据摄取
│   ├── retrieval.py           # 检索逻辑
│   ├── prompts.py             # Prompt 模板
│   ├── api_requests.py        # API 调用
│   ├── pdf_parsing.py         # PDF 解析
│   └── ...
│
├── data/                       # 数据目录
│   └── test_set/
│       ├── pdf_reports/       # PDF 文件（需添加）
│       ├── databases/         # 向量数据库（自动生成）
│       ├── questions.json     # 测试问题
│       └── subset.csv         # 数据集配置
│
└── docs/                       # 文档目录
    ├── LOCAL_DEPLOYMENT.md    # 本地部署指南（本文档）
    ├── deployment/            # 远程部署指南
    └── development/           # 开发文档
```

---

## 🔄 日常使用流程

### 启动应用

```bash
# 1. 进入项目目录
cd RAG-Challenge-2

# 2. 激活虚拟环境
source venv_streamlit/bin/activate  # macOS/Linux
# 或
.\venv_streamlit\Scripts\Activate.ps1  # Windows

# 3. 启动应用
streamlit run app_jinpan_qa.py
```

### 停止应用

- **方法 1**: 在终端按 `Ctrl+C`
- **方法 2**: 关闭终端窗口
- **方法 3**: 杀死进程
  ```bash
  # macOS/Linux
  pkill -f "streamlit run"
  
  # Windows
  taskkill /F /IM python.exe
  ```

### 更新代码

```bash
# 停止应用
# 拉取最新代码
git pull origin main

# 更新依赖（如果有变化）
pip install -r requirements.txt

# 重启应用
streamlit run app_jinpan_qa.py
```

### 重建数据库

```bash
# 删除旧数据库
rm -rf data/test_set/databases/*

# 重新构建
python main.py
```

---

## 📞 获取帮助

遇到问题？试试这些方法：

1. **查看日志**
   ```bash
   # Streamlit 日志
   ~/.streamlit/logs/
   
   # 应用日志（如果配置了）
   tail -f app.log
   ```

2. **检查常见问题部分**
   - 上面的"常见问题"章节可能已经有答案

3. **查看 GitHub Issues**
   - https://github.com/HEIcby/RAG-Challenge-2/issues

4. **提交新 Issue**
   - 包含错误信息
   - 包含运行环境信息
   - 包含复现步骤

---

## 🎉 开始使用

现在你已经完成了本地部署！

**快速检查清单：**

- [x] Python 3.10+ 已安装
- [x] 虚拟环境已创建并激活
- [x] 依赖包已安装
- [x] .env 文件已配置
- [x] PDF 文件已放置
- [x] 向量数据库已构建
- [x] Streamlit 应用已启动
- [x] 浏览器可以访问 http://localhost:8501

**享受你的 RAG 问答系统！** 🚀✨

---

**最后更新**: 2025-11-10  
**文档版本**: 1.0
