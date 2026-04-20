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
import re
import warnings
import time

warnings.filterwarnings(
    "ignore",
    message=".*has conflict with protected namespace.*",
    category=UserWarning,
)

from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.status import Status
from rich.theme import Theme
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PromptStyle

from agent import Agent
from config import Config
from llm_client import LLMClient
from rag_engine import RAGEngine
from embedding_client import EmbeddingClient
from reranker import ONNXReranker
from vector_store import LanceDBManager
from database import DatabaseManager
from model_manager import ensure_all_models

# 自定义 Rich 主题
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green",
    "agent": "bold blue",
    "user": "bold green",
    "tool": "bold yellow",
    "thinking": "italic grey50",
})

console = Console(theme=custom_theme)

# prompt_toolkit 样式
prompt_style = PromptStyle.from_dict({
    'prompt': 'bold green',
})

# Windows 管道输入默认使用系统编码（常为 GBK），强制改为 UTF-8
if os.name == 'nt':
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (AttributeError, OSError):
        pass

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
    """使用 Rich Table 打印当前配置"""
    table = Table(title="[bold green]当前配置[/bold green]", show_header=True, header_style="bold cyan")
    table.add_column("配置项", style="cyan")
    table.add_column("当前值", style="white")
    table.add_column("说明", style="dim")

    for key, desc in MANAGED_CONFIGS.items():
        raw = os.getenv(key, "")
        table.add_row(key, mask_value(key, raw), desc)
    
    console.print(table)


def set_config(key: str, value: str):
    """设置单个配置项，持久化到 .env 并更新当前进程环境变量"""
    key = key.strip()
    if key not in MANAGED_CONFIGS:
        console.print(f"[error]未知配置项：{key}[/error]")
        console.print(f"支持的项：{', '.join(MANAGED_CONFIGS.keys())}")
        return None

    env_path = get_env_path()
    set_key(env_path, key, value, quote_mode="never")
    os.environ[key] = value
    console.print(f"[success]OK[/success] {key} 已更新并写入 {env_path}")
    return key


def rebuild_agent() -> Agent:
    """根据最新配置重建 Agent 及所有依赖"""
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
    table = Table(title="[bold green]CLI 命令帮助[/bold green]", show_header=False, padding=(0, 2))
    table.add_row("/quit, /exit", "退出程序")
    table.add_row("/config", "查看当前配置")
    table.add_row("/config set KEY=VALUE", "修改配置项并持久化到 .env")
    table.add_row("/config reload", "重新加载 .env 并重建 Agent")
    table.add_row("/help", "显示本帮助")
    console.print(table)


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
            console.print(f"[error]格式错误，应为：/config set KEY=VALUE[/error]")
            return True
        k, v = expr.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        set_config(k, v)
        return True

    if sub == "reload":
        load_env_file()
        console.print(f"[success]OK[/success] 已重新加载 .env")
        return True

    console.print(f"[error]未知子命令：{sub}[/error]")
    return True


def check_is_initialized() -> bool:
    """检查关键配置是否已完成初始化"""
    api_key = os.getenv("BOT_API_KEY", "").strip()
    return bool(api_key)


def run_initialization_wizard():
    """全屏向导：引导用户完成首次配置"""
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]欢迎使用 CloudEcho 云忆[/bold cyan]\n"
        "[dim]检测到您是首次启动或尚未配置核心参数，我们将引导您完成初始化。[/dim]",
        border_style="blue",
        padding=(1, 4)
    ))

    # 利用现有 Config 的推导逻辑进行环境提示
    cxdata_exists = os.path.exists(Config.CXDATA_PATH)
    db_exists = os.path.exists(Config.DB_PATH)
    
    if cxdata_exists:
        console.print(f"[success]✨ 自动检测到插件环境！[/success]")
        console.print(f"  配置文件: [dim]{Config.CXDATA_PATH}[/dim]")
        console.print(f"  数据目录: [dim]{os.path.dirname(Config.DB_PATH)}[/dim]\n")
    else:
        console.print(f"[info]ℹ️ 未检测到预设的插件配置文件，将使用独立模式运行。[/info]\n")

    # 1. 配置 LLM
    console.print("[bold]第 1 步：配置 LLM (DeepSeek 或 OpenAI 兼容 API)[/bold]")
    
    api_base = session.prompt("  请输入 API Base (默认 DeepSeek): ", 
                              default=os.getenv("BOT_API_BASE", "https://api.deepseek.com/v1"))
    set_config("BOT_API_BASE", api_base)

    api_key = ""
    while not api_key:
        api_key = session.prompt("  请输入 API Key (必填): ").strip()
        if not api_key:
            console.print("  [error]API Key 不能为空！[/error]")
    set_config("BOT_API_KEY", api_key)

    api_model = session.prompt("  请输入模型名称 (默认 deepseek-chat): ", 
                               default=os.getenv("BOT_API_MODEL", "deepseek-chat"))
    set_config("BOT_API_MODEL", api_model)

    # 2. 确认路径配置
    console.print("\n[bold]第 2 步：确认数据存储路径[/bold]")
    table = Table(show_header=True, header_style="dim", box=None)
    table.add_column("配置项", style="cyan")
    table.add_column("当前路径", style="dim")
    
    table.add_row("数据库 (SQLite)", Config.DB_PATH)
    table.add_row("向量库 (LanceDB)", Config.VECTOR_DB_PATH)
    table.add_row("配置文件 (CxData)", Config.CXDATA_PATH)
    console.print(table)
    
    confirm_paths = session.prompt("  是否维持这些路径？(Y/n): ", default="Y").lower()
    if confirm_paths != "y":
        db_path = session.prompt("  请输入 SQLite 路径: ", default=Config.DB_PATH)
        set_config("DB_PATH", db_path)
        vdb_path = session.prompt("  请输入 LanceDB 路径: ", default=Config.VECTOR_DB_PATH)
        set_config("VECTOR_DB_PATH", vdb_path)
        cx_path = session.prompt("  请输入 CxData.json 路径: ", default=Config.CXDATA_PATH)
        set_config("CXDATA_PATH", cx_path)

    # 3. 模型下载
    console.print("\n[bold]第 3 步：初始化本地 AI 模型[/bold]")
    console.print("  CloudEcho 需要下载约 500MB 的 ONNX 模型文件进行本地 Embedding 和 Rerank。")
    
    # 这里直接调用 ensure_all_models，它内部已经有检测逻辑
    with console.status("[bold cyan]正在检测并准备模型环境...[/bold cyan]", spinner="bouncingBar"):
        # 如果不是插件环境，ensure_all_models 默认可能跳过下载，这里在向导中我们强制检查
        ensure_all_models()
    
    console.print("\n" + "="*50)
    console.print("[success]🎉 配置完成！CloudEcho 已准备就绪。[/success]")
    time.sleep(1.5)
    console.clear()


def main():
    parser = argparse.ArgumentParser(description="云忆 CloudEcho - 统一入口")
    parser.add_argument("--api", action="store_true", help="启动 Flask API 服务")
    parser.add_argument("--group", default="150204969", help="QQ 群号（仅 CLI 模式）")
    parser.add_argument("--user-id", default="123456", help="用户 QQ（仅 CLI 模式）")
    parser.add_argument("--user-name", default="CLI用户", help="用户昵称（仅 CLI 模式）")
    args = parser.parse_args()

    # 启动时加载 .env
    load_env_file()

    # 初始化 PromptSession
    history_path = os.path.join(os.path.dirname(get_env_path()), ".cli_history")
    global session
    session = PromptSession(history=FileHistory(history_path))

    # 检查是否需要初始化向导
    if not check_is_initialized() and not args.api:
        run_initialization_wizard()
    else:
        # 非首次启动或 API 模式，仍需确保模型存在
        with console.status("[bold cyan]正在初始化环境...[/bold cyan]", spinner="dots"):
            ensure_all_models()

    if args.api:
        from app import main as app_main
        app_main()
        return

    with console.status("[bold cyan]正在构建 Agent...[/bold cyan]", spinner="dots"):
        agent = rebuild_agent()

    # 自动同步最近 3 天的历史消息到向量库
    with console.status("[bold cyan]正在同步最近 3 天的历史消息...[/bold cyan]", spinner="bouncingBar"):
        try:
            # 获取所有有记录的群组
            active_groups = agent.db_manager.list_active_groups()
            if args.group not in active_groups:
                active_groups.append(args.group)
            
            with agent.db_manager._get_connection() as conn:
                for gid in active_groups:
                    try:
                        agent.rag_engine.vector_store.sync_recent_tables(conn, gid, days=3)
                    except Exception as e:
                        console.print(f"[warning]同步群 {gid} 失败: {e}[/warning]")
                    
            console.print(f"[success]✨ 已自动同步 {len(active_groups)} 个群组的最近 3 天消息。[/success]")
        except Exception as e:
            console.print(f"[error]同步过程出错: {e}[/error]")

    console.print(Panel.fit(
        f"[bold blue]CloudEcho 云忆[/bold blue] Agent CLI 已启动\n"
        f"群号: [cyan]{args.group}[/cyan] | 用户: [cyan]{args.user_name}({args.user_id})[/cyan]\n"
        f"输入消息对话，输入 [bold yellow]/help[/bold yellow] 查看命令",
        border_style="green"
    ))

    # 初始化 PromptSession 支持历史记录
    history_path = os.path.join(os.path.dirname(get_env_path()), ".cli_history")
    session = PromptSession(history=FileHistory(history_path))

    while True:
        try:
            user_input = session.prompt(f"[{args.user_name}] >>> ", style=prompt_style)
        except (EOFError, KeyboardInterrupt):
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
                if text.startswith("/config set") or text.startswith("/config reload"):
                    with console.status("[bold cyan]正在重建 Agent...[/bold cyan]"):
                        agent = rebuild_agent()
                continue
        
        # 拦截所有未识别的 / 命令
        if text.startswith("/"):
            console.print(f"[error]未知指令: {text.split()[0]}[/error]")
            console.print("[dim]输入 [bold yellow]/help[/bold yellow] 查看可用命令列表[/dim]")
            continue

        # 开始推理流
        reasoning_content = ""
        reply_content = ""
        
        status = Status("[thinking]Agent 正在思考...[/thinking]", console=console, spinner="bouncingBar")
        status.start()
        is_thinking = True

        try:
            for event in agent.run_stream(
                group_id=args.group,
                user_id=args.user_id,
                user_name=args.user_name,
                content=text,
                cache_messages=[],
            ):
                etype = event["type"]

                if etype == "round_start":
                    pass

                elif etype == "reasoning":
                    reasoning_content += event["content"]

                elif etype == "token":
                    if is_thinking:
                        status.stop()
                        is_thinking = False
                    
                    if not reply_content:
                        console.print(f"\n[agent]DeepSleep[/agent] > ", end="")
                    
                    token = event["content"]
                    reply_content += token
                    console.print(token, end="")

                elif etype == "tool_call":
                    if is_thinking:
                        status.stop()
                    
                    tool_name = event['tool']
                    args_json = event.get("arguments", {})
                    arg_str = ", ".join([f"{k}={v}" for k, v in args_json.items()])
                    
                    if tool_name == "rag_search":
                        query = args_json.get("query", "未知关键词")
                        console.print(f"\n[tool]🔍 正在召回: [cyan]{query}[/cyan][/tool]")
                        status.update(f"[thinking]正在检索向量库和缓存...[/thinking]")
                    elif tool_name == "time_filter":
                        time_range = args_json.get("time_range", "未知时间")
                        keywords = args_json.get("keywords")
                        kw_str = f"，关键词：[cyan]{keywords}[/cyan]" if keywords else ""
                        console.print(f"\n[tool]📅 正在按时间筛选: [cyan]{time_range}[/cyan]{kw_str}[/tool]")
                        status.update(f"[thinking]正在按时间段检索聊天记录...[/thinking]")
                    else:
                        console.print(f"\n[tool]调用工具 {tool_name}({arg_str})[/tool]")
                        status.update(f"[thinking]正在执行 {tool_name}...[/thinking]")
                    
                    status.start()
                    is_thinking = True

                elif etype == "tool_result":
                    if is_thinking:
                        status.stop()
                    
                    tool_name = event.get("tool", "unknown")
                    result = event["result"]
                    
                    if tool_name == "rag_search":
                        # 提取隐藏的统计信息：<!-- RAG_STATS:v,f,n -->
                        stats_match = re.search(r"<!-- RAG_STATS:(\d+),(\d+),(\d+) -->", result)

                        if stats_match:
                            v_count = stats_match.group(1)
                            f_count = stats_match.group(2)
                            n_count = stats_match.group(3)

                            console.print(f"[success]✅ 召回完成：向量召回 [cyan]{v_count}[/cyan] 条，文本匹配 [cyan]{f_count}[/cyan] 条，最终筛选 [cyan]{n_count}[/cyan] 条。[/success]")
                        else:
                            # 兼容旧逻辑
                            count = len(re.findall(r"\[\d{2}-\d{2}\s\d{2}:\d{2}\]", result))
                            if "暂无相关历史" in result:
                                console.print(f"[warning]💡 召回失败：未找到相关历史内容。[/warning]")
                            else:
                                console.print(f"[success]✅ 召回了 {count} 条内容。[/success]")

                        # 移除预览中的统计注释，保持界面整洁
                        clean_preview = re.sub(r"<!-- RAG_STATS:.*? -->", "", result).strip()
                    elif tool_name == "time_filter":
                        # 提取 Time_Range_Query_Results 的 count/shown 属性
                        tf_match = re.search(r"count='(\d+)'\s+shown='(\d+)'", result)
                        if tf_match:
                            total = tf_match.group(1)
                            shown = tf_match.group(2)
                            console.print(f"[success]✅ 时间筛选完成：共 [cyan]{total}[/cyan] 条，展示 [cyan]{shown}[/cyan] 条。[/success]")
                        else:
                            if "未找到" in result:
                                console.print(f"[warning]💡 该时间段内未找到聊天记录。[/warning]")
                            else:
                                console.print(f"[success]✅ 时间筛选完成。[/success]")
                        clean_preview = result.strip()
                    else:
                        clean_preview = result.strip()

                    preview = clean_preview[:200] + "..." if len(clean_preview) > 200 else clean_preview
                    console.print(Panel(preview, title=f"工具结果 [{tool_name}]", border_style="dim", subtitle=f"共 {len(clean_preview)} 字符"))

                    status.update("[thinking]Agent 正在分析结果...[/thinking]")
                    status.start()
                    is_thinking = True

                elif etype == "response_complete":
                    if is_thinking:
                        status.stop()
                        is_thinking = False
                    console.print("\n") # 换行

        except Exception as e:
            if is_thinking:
                status.stop()
                is_thinking = False
            console.print(f"[error]运行时错误: {e}[/error]")
            import traceback
            console.print(traceback.format_exc(), style="dim")

    console.print(f"\n[success]再见！[/success]")


if __name__ == "__main__":
    main()
