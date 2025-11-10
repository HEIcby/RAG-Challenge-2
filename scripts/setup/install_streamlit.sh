#!/bin/bash

# 金盘科技 RAG 问答系统 - Streamlit 前端安装脚本

echo "🚀 金盘科技 RAG 问答系统 - 安装 Streamlit"
echo "================================================"
echo ""

# 检查是否已经安装
if [ -d "venv_streamlit" ]; then
    echo "✅ 虚拟环境 venv_streamlit 已存在"
    echo ""
    read -p "是否重新安装？(y/N): " reinstall
    if [[ ! $reinstall =~ ^[Yy]$ ]]; then
        echo "取消安装"
        exit 0
    fi
    echo "🗑️  删除现有环境..."
    rm -rf venv_streamlit
fi

# 检测 Python 环境
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ 错误: 未找到 Python"
    echo "请先安装 Python 3.8+"
    exit 1
fi

echo "✅ 找到 Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# 创建虚拟环境
echo "📦 创建虚拟环境 venv_streamlit..."
$PYTHON_CMD -m venv venv_streamlit

# 激活虚拟环境
echo "激活虚拟环境..."
source venv_streamlit/bin/activate

# 安装依赖
echo "安装依赖包..."
pip install --upgrade pip
pip install streamlit

# 安装 API 客户端
echo "安装 API 客户端..."
pip install google-generativeai dashscope

# 安装其他必要依赖
echo "安装其他依赖..."
pip install pandas

echo ""
echo "✅ 安装完成！"
echo ""
echo "🚀 快速启动:"
echo "   ./start_frontend.sh"
echo ""
echo "📖 手动启动:"
echo "   1. 激活环境: source venv_streamlit/bin/activate"
echo "   2. 运行前端: streamlit run app_jinpan_qa.py"
echo "   3. 退出环境: deactivate"

echo ""
echo "================================================"
echo "📖 更多帮助请查看: docs/USER_GUIDE.md"
echo "================================================"
