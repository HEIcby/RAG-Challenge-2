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

### 2. [项目经验总结](LESSONS_LEARNED.md) ⭐
**适用对象**: 开发者、维护者  
**内容**:
- 核心技术经验
- 关键问题与解决方案
- 最佳实践
- 快速参考

### 3. [部署指南](DEPLOYMENT_GUIDE.md)
**适用对象**: 运维人员  
**内容**:
- 生产环境部署步骤
- 系统要求
- 配置说明
- 故障排查

### 4. [项目结构说明](PROJECT_STRUCTURE.md)
**适用对象**: 开发者  
**内容**:
- 目录组织
- 模块说明
- 代码架构

---

## 🚀 快速开始

**新用户**: 阅读 [README.md](../README.md) → [USER_GUIDE.md](USER_GUIDE.md)  
**开发者**: 阅读 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) → [LESSONS_LEARNED.md](LESSONS_LEARNED.md)  
**运维人员**: 阅读 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---

## 📋 文档清单

| 文档 | 大小 | 最后更新 | 描述 |
|------|------|----------|------|
| USER_GUIDE.md | 10KB | 2024-11-06 | 用户使用手册 |
| LESSONS_LEARNED.md | 8KB | 2025-11-07 | 经验总结（精简版）⭐ |
| DEPLOYMENT_GUIDE.md | 9KB | 2024-11-06 | 部署指南 |
| DEPLOYMENT_10.222.4.30.md | 8KB | 2024-11-06 | 特定服务器部署记录 |
| PROJECT_STRUCTURE.md | 5KB | 2024-11-06 | 项目结构 |

---

## ⚡ 快速参考

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
# 启动服务
./scripts/start_frontend.sh

# 查看日志
tail -f /tmp/streamlit.log

# 重启服务
pkill -9 -f streamlit && ./scripts/start_frontend.sh
```

---

## 📞 获取帮助

- **用户问题**: 查看 [USER_GUIDE.md](USER_GUIDE.md)
- **技术问题**: 查看 [LESSONS_LEARNED.md](LESSONS_LEARNED.md)
- **部署问题**: 查看 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **提交Issue**: [GitHub Issues](https://github.com/IlyaRice/RAG-Challenge-2/issues)

---

**最后更新**: 2025-11-07  
**维护者**: Ocean Chen
