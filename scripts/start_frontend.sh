#!/bin/bash

# 金盘科技 RAG 问答系统 - 快速启动echo "🚀 启动前端..."
echo "💡 访问地址将在下方显示"
echo "   - 本地访问: http://localhost:8501"
echo "   - 局域网访问: http://Network-URL:8501"
echo ""
echo "📚 使用帮助: docs/USER_GUIDE.md"
echo "================================================"
echo ""

source venv_streamlit/bin/activate

# 监听所有网络接口 (0.0.0.0) 以支持外部访问
# 这样可以从局域网或公网访问应用
streamlit run app_jinpan_qa.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true 金盘科技 RAG 问答系统"
echo "================================================"
echo ""

# 检查虚拟环境
if [ ! -d "venv_streamlit" ]; then
    echo "❌ 错误: 虚拟环境不存在！"
    echo ""
    echo "请先运行安装脚本:"
    echo "  ./install_streamlit.sh"
    echo ""
    exit 1
fi

# 检查数据库是否存在
if [ ! -d "data/val_set/databases/vector_dbs" ] || [ ! -d "data/val_set/databases/chunked_reports" ]; then
    echo "❌ 错误: 数据库不存在！"
    echo ""
    echo "请先运行以下命令创建数据库:"
    echo "  python main.py parse-pdfs"
    echo "  python main.py process-reports"
    echo ""
    echo "📖 详细步骤请查看: docs/USER_GUIDE.md"
    echo ""
    exit 1
fi

echo "✅ 数据库检查通过"
echo ""

# 激活虚拟环境并启动
echo "🚀 启动前端..."
echo "💡 访问地址将在下方显示"
echo "   - 本地访问: http://localhost:8502"
echo "   - 局域网访问: http://Network-URL:8502"
echo ""
echo "� 使用帮助: docs/USER_GUIDE.md"
echo "================================================"
echo ""

source venv_streamlit/bin/activate
streamlit run app_jinpan_qa.py
