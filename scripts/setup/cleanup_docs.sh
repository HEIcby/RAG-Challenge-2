#!/bin/bash
# 项目文档优化清理脚本
# 用途：整理和清理项目文档结构

set -e  # 遇到错误立即退出

echo "========================================="
echo "  RAG-Challenge-2 项目文档优化工具"
echo "========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "📁 项目根目录: $PROJECT_ROOT"
echo ""

# 1. 创建备份
echo "📦 步骤 1/4: 创建文档备份..."
BACKUP_DIR="backup_docs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 备份要删除的文档
if [ -f "FRONTEND_QUICKSTART.md" ]; then
    cp "FRONTEND_QUICKSTART.md" "$BACKUP_DIR/"
    echo "  ✓ 已备份: FRONTEND_QUICKSTART.md"
fi

if [ -f "多轮对话使用说明.md" ]; then
    cp "多轮对话使用说明.md" "$BACKUP_DIR/"
    echo "  ✓ 已备份: 多轮对话使用说明.md"
fi

if [ -f "问题库功能说明.md" ]; then
    cp "问题库功能说明.md" "$BACKUP_DIR/"
    echo "  ✓ 已备份: 问题库功能说明.md"
fi

if [ -f "金盘项目测试前端介绍.md" ]; then
    cp "金盘项目测试前端介绍.md" "$BACKUP_DIR/"
    echo "  ✓ 已备份: 金盘项目测试前端介绍.md"
fi

echo -e "${GREEN}✓ 备份完成: $BACKUP_DIR${NC}"
echo ""

# 2. 移动旧文档到备份目录（不删除）
echo "📂 步骤 2/4: 整理旧文档..."

if [ -f "FRONTEND_QUICKSTART.md" ]; then
    mv "FRONTEND_QUICKSTART.md" "$BACKUP_DIR/"
    echo "  ✓ 已移动: FRONTEND_QUICKSTART.md"
fi

if [ -f "多轮对话使用说明.md" ]; then
    mv "多轮对话使用说明.md" "$BACKUP_DIR/"
    echo "  ✓ 已移动: 多轮对话使用说明.md"
fi

if [ -f "问题库功能说明.md" ]; then
    mv "问题库功能说明.md" "$BACKUP_DIR/"
    echo "  ✓ 已移动: 问题库功能说明.md"
fi

if [ -f "金盘项目测试前端介绍.md" ]; then
    mv "金盘项目测试前端介绍.md" "$BACKUP_DIR/"
    echo "  ✓ 已移动: 金盘项目测试前端介绍.md"
fi

echo -e "${GREEN}✓ 文档整理完成${NC}"
echo ""

# 3. 清理临时文件
echo "🗑️  步骤 3/4: 清理临时文件..."

# 清理临时脚本目录
if [ -d "scripts/temp" ]; then
    rm -rf "scripts/temp"
    echo "  ✓ 已删除: scripts/temp/"
fi

# 清理虚拟环境标记文件
if [ -f "=1.24.0" ]; then
    rm -f "=1.24.0"
    echo "  ✓ 已删除: =1.24.0"
fi

if [ -f "=2.0.0" ]; then
    rm -f "=2.0.0"
    echo "  ✓ 已删除: =2.0.0"
fi

# 注意: dummy_report.json 保留（预下载缓存时需要）

# 清理 Python 缓存
echo "  🧹 清理 Python 缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Python 缓存已清理"

# 清理 macOS 文件
if [ "$(uname)" = "Darwin" ]; then
    find . -name ".DS_Store" -delete 2>/dev/null || true
    echo "  ✓ .DS_Store 文件已清理"
fi

echo -e "${GREEN}✓ 临时文件清理完成${NC}"
echo ""

# 4. 显示优化后的结构
echo "📊 步骤 4/4: 项目结构总结..."
echo ""
echo "核心文件:"
echo "  ✓ app_jinpan_qa.py          - Streamlit 前端"
echo "  ✓ main.py                    - 命令行工具"
echo "  ✓ requirements.txt           - 依赖列表"
echo ""
echo "文档（已优化）:"
echo "  ✓ docs/USER_GUIDE.md        - 用户指南（合并版）"
echo "  ✓ docs/PROJECT_STRUCTURE.md - 项目结构说明"
echo "  ✓ README.md                  - 项目介绍"
echo ""
echo "脚本:"
echo "  ✓ start_frontend.sh          - 启动脚本"
echo "  ✓ install_streamlit.sh       - 安装脚本"
echo ""
echo "数据:"
echo "  ✓ data/val_set/              - 金盘科技数据"
echo "  ✓ data/val_set/questions.csv - 127个问题库"
echo ""
echo "源代码:"
echo "  ✓ src/                       - 所有源代码"
echo ""
echo "Jupyter Notebooks:"
echo "  ✓ jupyter/RAG 9.8.ipynb      - Colab 工作流程"
echo "  ✓ jupyter/val_*.ipynb        - 测试笔记本"
echo ""

# 计算文件大小
if command -v du &> /dev/null; then
    BACKUP_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)
    echo "💾 备份大小: $BACKUP_SIZE"
fi

echo ""
echo "========================================="
echo -e "${GREEN}✅ 项目优化完成！${NC}"
echo "========================================="
echo ""
echo "📌 重要提示:"
echo "  • 旧文档已备份到: $BACKUP_DIR"
echo "  • 新的统一文档: docs/USER_GUIDE.md"
echo "  • 如需恢复，请从备份目录复制"
echo ""
echo "🚀 快速启动:"
echo "  ./start_frontend.sh"
echo ""
echo "📖 查看文档:"
echo "  cat docs/USER_GUIDE.md"
echo ""
