# QQ 群 RAG Agent

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

基于 Flask + LanceDB + 本地 ONNX 嵌入模型的商用级 RAG（检索增强生成）系统，为 QQ 群聊提供带历史上下文感知的 AI 回复能力。

> 本项目采用 **GPL-3.0-or-later** 开源许可证发布。

## 核心特性

- **商用级 RAG 链路**：查询重写 → 混合检索（LanceDB 历史 + 缓存消息）→ Cross-encoder 重排序 → 上下文组装 → LLM 生成
- **按群按天向量存储**：每个 `group_{群号}_{日期}` 对应一张 LanceDB 向量表，自动增量同步、自动清理旧数据
- **本地 ONNX 模型**：使用 `bge-large-zh-v1.5`（int8 量化）做嵌入，`bge-reranker-base`（int8 量化）做重排序，无需 GPU、零运维
- **多轮工具调用 Agent**：支持 ReAct 风格的 rag_search 工具调用，流式输出思考与回复过程
- **模型自动下载**：首次启动时自动从远程 CDN 检测并下载模型，带 tqdm 进度条
- **CLI 交互调试**：内置命令行客户端，支持实时查看配置、修改 `.env`、流式对话

## 技术栈

- **Web 框架**：Flask
- **向量数据库**：LanceDB（嵌入式，零外部依赖）
- **嵌入模型**：BGE-large-zh-v1.5（ONNX Runtime int8）
- **重排序模型**：bge-reranker-base（ONNX Runtime int8）
- **LLM**：DeepSeek API（兼容 OpenAI 格式，可配置）
- **数据库**：SQLite（WAL 模式）

## 应用场景示例：SCPSL 服务器群组

本项目最初为 **SCP: Secret Laboratory 服务器 QQ 群** 设计，典型场景包括：

- 新玩家问"怎么进服？"→ Agent 自动检索群公告/历史回答并回复
- 玩家问"昨晚服务器发生了什么事？"→ RAG 检索昨晚群聊上下文
- 管理员远程查询近期投诉记录 → 通过历史检索快速定位相关内容

> 你可以把 Agent 配置成任何角色（服务器向导、吐槽担当、资讯助手等）。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制或编辑 `.env`：

```bash
BOT_API_KEY=your_deepseek_api_key
BOT_API_BASE=https://api.deepseek.com/v1/chat/completions
BOT_API_MODEL=deepseek-chat
```

其他可选配置项：
- `API_HOST` / `API_PORT`：服务监听地址（默认 `0.0.0.0:5000`）
- `DB_PATH`：SQLite 数据库路径
- `VECTOR_DB_PATH`：LanceDB 存储目录
- `EMBEDDING_MODEL_DIR` / `RERANKER_MODEL_DIR`：本地模型目录
- `MODEL_URL_EMBEDDING` / `MODEL_URL_RERANKER`：自定义模型下载地址

### 3. 启动 API 服务

```bash
python app.py
```

首次启动时，如果 `models/` 目录缺失 ONNX 模型，程序会自动从远程下载并解压（显示进度条）。

### 4. 启动 CLI 调试

```bash
python cli.py
```

CLI 内置命令：
- `/config`：查看当前配置
- `/config set KEY=VALUE`：修改配置并持久化到 `.env`
- `/config reload`：重新加载 `.env` 并重建 Agent
- `/help`：显示帮助
- `/quit`：退出

---

## 部署与集成

### 架构说明

本服务提供一组通用的 HTTP REST API，**不绑定任何特定的 QQ 机器人框架**。只要你的机器人框架（或插件）能发送 HTTP 请求，即可对接。

典型的对接模式如下：

```
┌─────────────────┐
│  QQ 机器人框架   │  ← 任何支持 HTTP 的框架：NoneBot、Mirai、go-cqhttp 等
│  (你的插件/脚本) │
└────────┬────────┘
         │
         │ ① POST /api/qq/message      写入群消息到 SQLite
         │ ② POST /api/sync/vectorstore 批量同步到向量库
         │ ③ POST /api/llm/rag          获取 AI 回复
         ▼
┌─────────────────────┐
│   QQ 群 RAG Agent   │
│     (本服务)        │
│  Flask + SQLite     │
│  + LanceDB + ONNX   │
└─────────────────────┘
```

### 集成示例参考

- **[MiraiCQ](https://github.com/super1207/MiraiCQ)**（OneBot v11 兼容）：如果你仍在使用 CoolQ 时代的老插件，MiraiCQ 可以低成本复用原有 CQP 插件。本项目作者早期即通过 MiraiCQ + 易语言插件的方式接入本服务。
- **NoneBot2** / **go-cqhttp**：现代主流方案，可直接用 Python/Node 编写转发插件。
- **自定义脚本**：任何能轮询或 Webhook 接收群消息并调用 HTTP API 的程序均可。

### 服务器部署示例

```bash
# 1. 克隆或上传源码到服务器
cd /opt/qqgroup-agent

# 2. 创建 Python 虚拟环境（推荐）
python -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 编辑环境变量
vim .env

# 5. 启动服务（可用 systemd/supervisor 托管后台运行）
python app.py
```

### 模型下载说明

首次启动时，服务会检测 `models/` 下是否存在 `model_int8.onnx`。如果不存在，会从 `MODEL_URL_EMBEDDING` 和 `MODEL_URL_RERANKER` 自动下载 `.tar.gz` 并解压（显示 tqdm 进度条）。

你可以：
- **预上传模型**：把本地已下载的 `models/` 目录（约 600MB）一起传到服务器，跳过下载。
- **使用 CDN**：把 `.tar.gz` 放到自己的虚拟主机/对象存储上，通过环境变量修改下载地址。

---

## 主要 API 端点

### 健康检查
```http
GET /api/health
```

### 接收消息
```http
POST /api/qq/message
Content-Type: application/json

{
  "group_id": "123456789",
  "user_id": "987654321",
  "user_name": "TestUser",
  "content": "Hello World!",
  "time": "20260415123000"
}
```

### RAG 查询（推荐）
```http
POST /api/llm/rag
Content-Type: application/json

{
  "group_id": "123456789",
  "user_id": "987654321",
  "user_name": "TestUser",
  "content": "群里昨天在聊什么？",
  "cache": [
    {"user_id": "111", "user_name": "A", "time": "20260415120000", "content": "刚刚说到..."}
  ]
}
```

返回：
```json
{
  "status": "success",
  "response": "昨天群里在讨论新服务器的白名单申请流程...",
  "sources": [
    {
      "user_name": "Admin",
      "user_id": "12345",
      "time": "2026-04-14T22:15:00",
      "content": "白名单申请格式是...",
      "score": 0.8934,
      "source": "history"
    }
  ],
  "rewritten_query": "昨天 群聊 讨论内容",
  "tool_calls": [...]
}
```

### 向量库同步
```http
POST /api/sync/vectorstore
Content-Type: application/json

{
  "123456789": [
    {"QQ号": "123", "昵称": "User", "时间": "20260415120000", "内容": "msg"}
  ]
}
```

### 其他端点
- `GET /api/qq/history?group_id=xxx`：群聊历史记录
- `GET /api/qq/groups`：活跃群列表
- `GET /api/qq/stats`：数据库统计
- `POST /api/admin/cleanup`：手动清理旧向量表

---

## 项目结构

```
.
├── app.py              # Flask API 主入口
├── cli.py              # 交互式命令行客户端
├── agent.py            # ReAct 多轮 Agent
├── rag_engine.py       # RAG 引擎（检索、重排、上下文组装）
├── vector_store.py     # LanceDB 向量存储管理
├── embedding_client.py # 本地 ONNX 嵌入模型客户端
├── reranker.py         # 本地 ONNX 重排序器
├── llm_client.py       # DeepSeek / OpenAI 兼容 LLM 客户端
├── database.py         # SQLite 数据库管理器
├── config.py           # 配置类
├── model_manager.py    # 模型自动下载管理器
├── requirements.txt    # Python 依赖
├── pyproject.toml      # 打包配置
├── LICENSE             # GPL-3.0 许可证
└── README.md           # 本文件
```

---

## 开源与发布

### 推送到 GitHub

```bash
# 1. 初始化 Git
git init

# 2. 添加文件（.gitignore 已配置，会自动排除敏感文件和运行时数据）
git add .

# 3. 提交
git commit -m "feat: initial release of QQ Group RAG Agent"

# 4. 关联远程仓库（请先登录 GitHub 创建空仓库）
git branch -M main
git remote add origin https://github.com/你的用户名/qqgroup-agent.git

# 5. 推送
git push -u origin main
```

### 发布到 PyPI（可选）

如果你希望别人能用 `pipx install qqgroup-agent` 一键安装 CLI：

```bash
pip install build twine
python -m build
python -m twine upload dist/*
```

安装方：
```bash
pipx install qqgroup-agent
qqgroup-agent    # 启动 CLI
qqgroup-api      # 启动 API 服务
```

> 注意：`pipx` 安装不会携带 `models/` 目录，用户首次运行仍需自动下载或手动放置模型。

---

## 许可证

本项目采用 [GNU General Public License v3.0 or later](LICENSE) 开源。

> Copyright (C) 2026
>
> This program is free software: you can redistribute it and/or modify
> it under the terms of the GNU General Public License as published by
> the Free Software Foundation, either version 3 of the License, or
> (at your option) any later version.
