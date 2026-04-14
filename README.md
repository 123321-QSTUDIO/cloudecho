# 云忆 CloudEcho

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

> 为群聊注入长期记忆的 RAG Agent。

**云忆 CloudEcho** 是一个轻量、高性能的本地 RAG（检索增强生成）服务。它专为即时通讯群组设计，通过向量化存储聊天记录，赋予 AI 助手精准的历史上下文感知能力。

目前我主要把它接在 QQ 群里用，但 API 是通用的，Discord、企业微信等任何能发 HTTP 请求的框架都能对接。

## 核心特性

- **全流程 RAG 引擎**：内置"查询重写 → 混合检索 → Cross-encoder 重排序 → 上下文组装 → LLM 生成"的完整链路。
- **按群按天动态分表**：按 `群号_日期` 建立 LanceDB 向量表，支持自动增量同步与过期数据清理。
- **纯本地轻量部署**：嵌入模型与重排序模型均采用 ONNX Runtime (int8 量化) 运行，**无需 GPU**，普通轻量云服务器即可跑满。
- **ReAct 工具扩展**：原生支持多轮 Agent 架构，不仅能查记忆，也方便后续扩展各类工具。
- **冷启动自动下载模型**：首次运行自动从 CDN 拉取所需模型，内置 CLI 命令行工具方便调试。

## 典型应用场景

- **游戏公会/社区群**："昨晚打团爆了什么装备？"→ AI 检索昨晚群聊并总结。
- **技术交流群**："之前群主发过的解决跨域报错的代码是什么？"→ 穿透几个月的聊天记录精准召回。
- **学习社群**："帮我总结一下今天下午大家讨论的微积分重点。"→ 调用 RAG 生成高质量摘要。

## 架构设计

CloudEcho 提供一组通用的 HTTP REST API，**不绑定任何特定的机器人框架**。NoneBot2、Koishi、Mirai，或者你自己写的脚本，只要能发 HTTP 请求就能接入。

```text
┌─────────────────┐
│  IM 机器人框架  │  ← NoneBot / Koishi / 自研脚本 / Mirai 等
│ (业务逻辑接入层)│
└────────┬────────┘
         │
         │ ① POST /api/qq/message       写入消息至 SQLite
         │ ② POST /api/sync/vectorstore 定时同步至本地向量库
         │ ③ POST /api/llm/rag          发起带历史追溯的 RAG 对话
         ▼
┌─────────────────────┐
│    云忆 CloudEcho   │
├─────────────────────┤
│ • Flask 网关服务    │
│ • ReAct Agent 调度  │
│ • LanceDB 向量检索  │
│ • ONNX 离线推理     │
└─────────────────────┘
```

## 快速开始

### 1. 环境准备

建议使用 Python 3.10+：

```bash
git clone https://github.com/123321-QSTUDIO/cloudecho.git
cd cloudecho
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的 LLM API 信息（默认兼容 DeepSeek / OpenAI 格式）：

```ini
BOT_API_KEY=your_api_key_here
BOT_API_BASE=https://api.deepseek.com/v1
BOT_API_MODEL=deepseek-chat
API_HOST=0.0.0.0
API_PORT=5000
```

### 3. 一键启动

启动 API 服务：
```bash
python app.py
```
*(注：首次启动会自动检测并下载 `bge-large-zh-v1.5` 与 `bge-reranker-base` 的 int8 量化模型到 `models/` 目录，请保持网络畅通。)*

启动 CLI 调试控制台：
```bash
python cli.py
```
> 在 CLI 中输入 `/help` 可查看快捷配置命令，输入文本即可模拟群聊 RAG 对话。

## API 接口文档

### 发起 RAG 查询 (核心接口)

```http
POST /api/llm/rag
Content-Type: application/json

{
  "group_id": "123456789",
  "user_id": "987654321",
  "user_name": "TestUser",
  "content": "群里昨天在聊什么？",
  "cache": [
    {"user_id": "111", "user_name": "Alice", "time": "20260415120000", "content": "刚刚说到..."}
  ]
}
```

<details>
<summary>点击查看响应示例</summary>

```json
{
  "status": "success",
  "response": "昨天群里主要在讨论新的服务器扩容计划...",
  "sources": [
    {
      "user_name": "Admin",
      "content": "明天记得把服务器配置升级一下。",
      "score": 0.8934,
      "source": "history"
    }
  ],
  "rewritten_query": "昨天 群聊 讨论内容 服务器",
  "tool_calls": []
}
```
</details>

### 同步历史消息至向量库

建议机器人在业务层每隔一定时间（或达到一定条数）批量调用此接口。

```http
POST /api/sync/vectorstore
Content-Type: application/json

{
  "123456789": [
    {"QQ号": "123", "昵称": "User", "时间": "20260415120000", "内容": "这是需要被记忆的内容"}
  ]
}
```

*(其他如健康检查 `/api/health`、历史获取 `/api/qq/history` 等基础接口请参阅源码注释)*

## 项目目录

```text
cloudecho/
├── app.py              # Flask API 主入口
├── cli.py              # CLI 交互式调试客户端
├── agent.py            # ReAct 多轮 Agent 核心逻辑
├── rag_engine.py       # RAG 检索、重排与上下文组装
├── vector_store.py     # LanceDB 向量存储生命周期管理
├── embedding_client.py # 本地 ONNX 嵌入推理组件
├── reranker.py         # 本地 ONNX 重排序推理组件
├── requirements.txt    # 依赖清单
└── README.md           # 项目说明
```

## 贡献指南

欢迎提交 PR 或 Issue：
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -m 'Add some feature'`)
4. 推送分支 (`git push origin feature/your-feature`)
5. 开启 Pull Request

## 许可证

本项目基于 [GPL-3.0 License](LICENSE) 协议开源。
