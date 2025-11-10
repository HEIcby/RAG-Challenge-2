# 📚 文档导航

> 金盘科技 RAG 智能问答系统 - 完整文档

---

## 📖 主要文档

### 1. [用户使用手册](USER_GUIDE.md)
**适用对象**: 终端用户  
**内容**: 
- Streamlit前端使用说明
- 问题库功能
- 多轮对话配置
- 常见问题FAQ

### 2. [部署指南](deployment/DEPLOYMENT_GUIDE.md)
**适用对象**: 运维人员、开发者  
**内容**:
- 🏠 本地部署 - 使用 `scripts/setup/install_streamlit.sh`
- 🌐 服务器部署 - 生产环境配置
- 系统要求与依赖
- 故障排查指南

### 3. [项目经验总结](development/LESSONS_LEARNED.md) ⭐
**适用对象**: 开发者、维护者  
**内容**:
- 核心技术经验
- 关键问题与解决方案
- 最佳实践
- 快速参考

### 4. [项目结构说明](development/PROJECT_STRUCTURE.md)
**适用对象**: 开发者  
**内容**:
- 目录组织
- 模块说明
- 代码架构

---

## 🚀 快速开始

**新用户**: 阅读 [主 README](../README.md) → 运行 `./scripts/setup/install_streamlit.sh` → [USER_GUIDE.md](USER_GUIDE.md)  
**开发者**: 阅读 [PROJECT_STRUCTURE](development/PROJECT_STRUCTURE.md) → [LESSONS_LEARNED](development/LESSONS_LEARNED.md)  
**运维人员**: 阅读 [DEPLOYMENT_GUIDE](deployment/DEPLOYMENT_GUIDE.md)

---

## 📋 文档清单

| 文档 | 位置 | 最后更新 | 描述 |
|------|------|----------|------|
| USER_GUIDE.md | docs/ | 2024-11-06 | 用户使用手册 |
| DEPLOYMENT_GUIDE.md | docs/deployment/ | 2025-11-10 | 部署指南 ⭐ |
| **DEPLOYMENT_10.222.4.30.md** | **docs/** | **2025-11-10** | **生产服务器部署记录** 🆕 |
| LESSONS_LEARNED.md | docs/development/ | 2025-11-07 | 经验总结 |
| PROJECT_STRUCTURE.md | docs/development/ | 2024-11-06 | 项目结构 |

### 🆕 新增文档

- **[DEPLOYMENT_10.222.4.30.md](DEPLOYMENT_10.222.4.30.md)** - 详细记录了实际服务器（10.222.4.30）的完整部署过程，包括：
  - 遇到的所有问题和解决方案
  - 性能数据和资源使用情况
  - 维护命令和调试技巧
  - 138分钟完整部署时间线

---

## ⚡ 快速参考

### 本地部署快速命令

```bash
# 1. 克隆项目
git clone https://github.com/HEIcby/RAG-Challenge-2.git
cd RAG-Challenge-2

# 2. 安装依赖
./scripts/setup/install_streamlit.sh

# 3. 配置 API
cp .env.example .env
nano .env  # 填入你的 API keys

# 4. 启动服务
./scripts/start_frontend.sh
```

### 关键概念
- **1-based vs 0-based**: 索引标准处理
- **时间智能路由**: 年份提取与文档路由
- **精确数据优先**: 提示词约束原则
- **页码验证**: 防止幻觉机制

### 关键文件
```
src/prompts.py:606-814       # 金盘答案生成提示词
src/retrieval.py:17-51       # 时间路由逻辑
app_jinpan_qa.py:176-220     # PDF页面图片提取
app_jinpan_qa.py:256-305     # 参考来源显示
```

### 常用命令
```bash
# 启动服务（本地）
./scripts/start_frontend.sh

# 重新安装依赖
./scripts/setup/install_streamlit.sh

# 停止服务
pkill -f streamlit
```

---

## 📞 获取帮助

- **用户问题**: 查看 [USER_GUIDE.md](USER_GUIDE.md)
- **本地部署**: 运行 `./scripts/setup/install_streamlit.sh`
- **技术问题**: 查看 [LESSONS_LEARNED.md](development/LESSONS_LEARNED.md)
- **服务器部署**: 查看 [DEPLOYMENT_GUIDE.md](deployment/DEPLOYMENT_GUIDE.md)
- **提交Issue**: [GitHub Issues](https://github.com/HEIcby/RAG-Challenge-2/issues)

---

**最后更新**: 2025-11-10  
**维护者**: Ocean Chen
