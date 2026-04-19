#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent CLI
交互式命令行界面，流式显示模型思考、回答和工具调用过程
支持运行时查看和修改配置（.env 持久化）
"""

import argparse
import sys
import io
import os
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*has conflict with protected namespace.*",
    category=UserWarning,
)

from dotenv import load_dotenv, set_key

from agent import Agent
from config import Config
from llm_client import LLMClient
from rag_engine import RAGEngine
from embedding_client import EmbeddingClient
from reranker import ONNXReranker
from vector_store import LanceDBManager
from database import DatabaseManager
from model_manager import ensure_all_models

# Windows 管道输入默认使用系统编码（常为 GBK），强制改为 UTF-8
if os.name == 'nt':
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (AttributeError, OSError):
        pass

# ANSI 颜色码
COLOR_GRAY = "\033[90m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_CYAN = "\033[96m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"

# 受 CLI 管理的配置项（键 -> 说明）
MANAGED_CONFIGS = {
    "BOT_API_KEY": "LLM API 密钥",
    "BOT_API_BASE": "LLM API 基础地址",
    "BOT_API_MODEL": "LLM 模型名称",
    "DB_PATH": "SQLite 数据库路径",
    "VECTOR_DB_PATH": "LanceDB 向量数据库路径",
    "EMBEDDING_MODEL_DIR": "Embedding 模型本地目录",
    "RERANKER_MODEL_DIR": "Reranker 模型本地目录",
    "CXDATA_PATH": "群配置文件 CxData.json 路径",
}


def get_env_path() -> str:
    """获取程序所在目录（AI 目录）下的 .env 文件路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, ".env")


def load_env_file():
    """加载 .env 文件到当前进程环境变量"""
    env_path = get_env_path()
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)


def mask_value(key: str, value: str) -> str:
    """对敏感值进行脱敏显示"""
    if not value:
        return "(未设置)"
    if "KEY" in key or "SECRET" in key or "TOKEN" in key or "PASSWORD" in key:
        if len(value) <= 8:
            return "***"
        return value[:4] + "***" + value[-4:]
    return value


def show_config():
    """打印当前配置"""
    print(f"\n{COLOR_GREEN}[当前配置]{COLOR_RESET}")
    for key, desc in MANAGED_CONFIGS.items():
        raw = os.getenv(key, "")
        print(f"  {COLOR_CYAN}{key}{COLOR_RESET}: {mask_value(key, raw)}  {COLOR_GRAY}# {desc}{COLOR_RESET}")
    print()


def set_config(key: str, value: str):
    """设置单个配置项，持久化到 .env 并更新当前进程环境变量"""
    key = key.strip()
    if key not in MANAGED_CONFIGS:
        print(f"{COLOR_RED}未知配置项：{key}{COLOR_RESET}")
        print(f"支持的项：{', '.join(MANAGED_CONFIGS.keys())}")
        return None

    env_path = get_env_path()
    set_key(env_path, key, value, quote_mode="never")
    os.environ[key] = value
    print(f"{COLOR_GREEN}[OK]{COLOR_RESET} {key} 已更新并写入 {env_path}")
    return key


def rebuild_agent() -> Agent:
    """根据最新配置重建 Agent 及所有依赖"""
    # 重建各层组件，使新配置生效
    db_manager = DatabaseManager(db_path=Config.DB_PATH)
    db_manager.init_database()
    llm_client = LLMClient()
    rag_engine = RAGEngine(
        embedding_client=EmbeddingClient(model_dir=Config.EMBEDDING_MODEL_DIR),
        reranker=ONNXReranker(model_dir=Config.RERANKER_MODEL_DIR),
        vector_store=LanceDBManager(
            db_path=Config.VECTOR_DB_PATH,
            embedding_client=EmbeddingClient(model_dir=Config.EMBEDDING_MODEL_DIR),
        ),
        db_manager=db_manager,
    )
    return Agent(llm_client=llm_client, rag_engine=rag_engine, db_manager=db_manager)


def print_help():
    """打印 CLI 内置命令帮助"""
    print(f"\n{COLOR_GREEN}[CLI 命令帮助]{COLOR_RESET}")
    print("  /quit, /exit          退出程序")
    print("  /config               查看当前配置")
    print("  /config set KEY=VALUE 修改配置项并持久化到 .env")
    print("  /config reload        重新加载 .env 并重建 Agent")
    print("  /help                 显示本帮助")
    print()


def handle_config_command(text: str) -> bool:
    """
    处理以 /config 开头的命令
    返回 True 表示已处理，False 表示非配置命令
    """
    parts = text.split(None, 2)
    if not parts or parts[0] != "/config":
        return False

    if len(parts) == 1:
        show_config()
        return True

    sub = parts[1].lower()

    if sub == "set" and len(parts) == 3:
        expr = parts[2].strip()
        if "=" not in expr:
            print(f"{COLOR_RED}格式错误，应为：/config set KEY=VALUE{COLOR_RESET}")
            return True
        k, v = expr.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        set_config(k, v)
        return True

    if sub == "reload":
        load_env_file()
        print(f"{COLOR_GREEN}[OK]{COLOR_RESET} 已重新加载 .env")
        return True

    print(f"{COLOR_RED}未知子命令：{sub}{COLOR_RESET}")
    return True


def main():
    parser = argparse.ArgumentParser(description="云忆 CloudEcho - 统一入口")
    parser.add_argument("--api", action="store_true", help="启动 Flask API 服务")
    parser.add_argument("--group", default="150204969", help="QQ 群号（仅 CLI 模式）")
    parser.add_argument("--user-id", default="123456", help="用户 QQ（仅 CLI 模式）")
    parser.add_argument("--user-name", default="CLI用户", help="用户昵称（仅 CLI 模式）")
    args = parser.parse_args()

    # 启动时加载 .env
    load_env_file()

    # 首次启动自动检测并下载模型
    ensure_all_models()

    if args.api:
        from app import main as app_main
        app_main()
        return

    agent = rebuild_agent()

    print(f"{COLOR_GREEN}[Agent CLI] 已启动{COLOR_RESET}")
    print(f"群号: {args.group} | 用户: {args.user_name}({args.user_id})")
    print("输入消息与 Agent 对话，输入 /help 查看命令，输入 /quit 退出\n")

    while True:
        try:
            user_input = input(f"{COLOR_GREEN}[{args.user_name}] >>> {COLOR_RESET}")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        text = user_input.strip()
        if text in ("/quit", "/exit", "quit", "exit"):
            break

        if not text:
            continue

        if text == "/help":
            print_help()
            continue

        if text.startswith("/config"):
            if handle_config_command(text):
                # 如果修改了配置，建议重建 Agent（除了 reload 外，set 也建议重建）
                if text.startswith("/config set") or text.startswith("/config reload"):
                    agent = rebuild_agent()
                continue

        print(f"\n{COLOR_GRAY}--- Agent 开始思考 ---{COLOR_RESET}")

        reasoning_active = False
        token_active = False
        first_round = True

        for event in agent.run_stream(
            group_id=args.group,
            user_id=args.user_id,
            user_name=args.user_name,
            content=text,
            cache_messages=[],
        ):
            etype = event["type"]

            if etype == "round_start":
                if not first_round:
                    if reasoning_active or token_active:
                        print()
                    print(f"\n{COLOR_GRAY}--- 第 {event['round']} 轮推理 ---{COLOR_RESET}")
                    reasoning_active = False
                    token_active = False
                first_round = False

            elif etype == "reasoning":
                if not reasoning_active:
                    print(f"\n{COLOR_GRAY}[思考过程]{COLOR_RESET} ", end="", flush=True)
                    reasoning_active = True
                print(event["content"], end="", flush=True)

            elif etype == "token":
                if reasoning_active and not token_active:
                    print(f"\n\n{COLOR_RESET}[回复]{COLOR_RESET} ", end="", flush=True)
                    reasoning_active = False
                    token_active = True
                elif not token_active:
                    print(f"\n{COLOR_RESET}[回复]{COLOR_RESET} ", end="", flush=True)
                    token_active = True
                print(event["content"], end="", flush=True)

            elif etype == "tool_call":
                if reasoning_active or token_active:
                    print()
                    reasoning_active = False
                    token_active = False
                print(f"\n{COLOR_YELLOW}[TOOL CALL] {event['tool']}{COLOR_RESET}")
                args_json = event.get("arguments", {})
                for k, v in args_json.items():
                    print(f"  {COLOR_CYAN}{k}{COLOR_RESET}: {v}")

            elif etype == "tool_result":
                result = event["result"]
                preview = result[:400] + "..." if len(result) > 400 else result
                lines = preview.splitlines()
                print(f"{COLOR_BLUE}[TOOL RESULT]{COLOR_RESET}")
                for line in lines[:6]:
                    print(f"  {line}")
                if len(lines) > 6:
                    print(f"  ... ({len(result)} 字符)")

            elif etype == "response_complete":
                if reasoning_active or token_active:
                    print()
                print(f"\n{COLOR_GRAY}--- Agent 回复结束 ---{COLOR_RESET}\n")

        reasoning_active = False
        token_active = False

    print(f"\n{COLOR_GREEN}再见！{COLOR_RESET}")


if __name__ == "__main__":
    main()
