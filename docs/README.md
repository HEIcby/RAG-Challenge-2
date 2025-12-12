# 文档目录

本目录包含项目的所有技术文档和使用指南。

## 📁 目录结构

```
docs/
├── README.md                          # 本文档
├── USER_GUIDE.md                      # 用户使用指南
├── LOCAL_DEPLOYMENT.md                # 本地部署指南
├── DEPLOYMENT_10.222.4.30.md          # 服务器部署指南
│
├── deployment/                         # 部署相关文档
│   ├── DEPLOYMENT_GUIDE.md
│   ├── DEPLOYMENT.md
│   └── DEPLOYMENT_QUICK.md
│
├── development/                        # 开发相关文档
│   ├── PROJECT_STRUCTURE.md           # 项目结构说明
│   ├── LESSONS_LEARNED.md             # 开发经验总结
│   └── LESSONS_LEARNED_v2.md
│
├── retrieval_algorithms/               # 检索算法相关文档
│   ├── retrieval_algorithms_update_20251203.md  # 检索算法更新汇报（最新）
│   ├── 移植指南_SSG_Traversal_和_Triangulation_FullDim.md  # SSG和Triangulation移植指南
│   └── BASIC_RETRIEVAL_LOGIC_ANALYSIS.md  # 基础检索逻辑分析
│
└── archives/                           # 历史文档归档
    ├── DEPLOYMENT_10.222.4.30.md
    ├── DOCUMENTATION_CLEANUP_2025-11-07.md
    ├── DOCUMENTATION_SUMMARY.md
    └── ...
```

## 📚 文档分类

### 🚀 快速开始

- **[USER_GUIDE.md](./USER_GUIDE.md)**: 用户使用指南，包含基本操作和功能介绍
- **[LOCAL_DEPLOYMENT.md](./LOCAL_DEPLOYMENT.md)**: 本地部署指南，快速搭建开发环境

### 🔧 部署文档

- **[deployment/DEPLOYMENT_GUIDE.md](./deployment/DEPLOYMENT_GUIDE.md)**: 详细部署指南
- **[deployment/DEPLOYMENT_QUICK.md](./deployment/DEPLOYMENT_QUICK.md)**: 快速部署指南
- **[DEPLOYMENT_10.222.4.30.md](./DEPLOYMENT_10.222.4.30.md)**: 特定服务器部署指南

### 🧠 检索算法

- **[retrieval_algorithms/retrieval_algorithms_update_20251203.md](./retrieval_algorithms/retrieval_algorithms_update_20251203.md)**: 
  - 最新检索算法更新汇报
  - 包含SSG Traversal、Triangulation FullDim和Hybrid Expansion的详细介绍
  - 技术改进和性能分析

- **[retrieval_algorithms/移植指南_SSG_Traversal_和_Triangulation_FullDim.md](./retrieval_algorithms/移植指南_SSG_Traversal_和_Triangulation_FullDim.md)**: 
  - SSG和Triangulation算法的移植指南
  - 算法原理和实现细节

- **[retrieval_algorithms/BASIC_RETRIEVAL_LOGIC_ANALYSIS.md](./retrieval_algorithms/BASIC_RETRIEVAL_LOGIC_ANALYSIS.md)**: 
  - 基础检索逻辑的详细分析
  - 代码流程和关键逻辑

### 💻 开发文档

- **[development/PROJECT_STRUCTURE.md](./development/PROJECT_STRUCTURE.md)**: 项目结构说明
- **[development/LESSONS_LEARNED.md](./development/LESSONS_LEARNED.md)**: 开发经验总结

## 🔍 快速查找

### 按主题查找

- **检索算法**: 查看 `retrieval_algorithms/` 目录
- **部署相关**: 查看 `deployment/` 目录
- **开发指南**: 查看 `development/` 目录
- **历史文档**: 查看 `archives/` 目录

### 按日期查找

- **2025-12-03**: 检索算法更新汇报
- **2025-11-07**: 历史文档归档

## 📝 文档维护

- 新增文档请按照分类放入对应目录
- 重要更新请在文档标题中标注日期
- 过时文档请移至 `archives/` 目录

## 🤝 贡献指南

1. 新增文档前，请先检查是否已有相关内容
2. 文档命名使用清晰的描述性名称
3. 重要技术文档请包含：
   - 概述
   - 详细说明
   - 使用示例
   - 注意事项

---

**最后更新**: 2025-12-03
