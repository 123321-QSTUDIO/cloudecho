#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持多轮工具调用的 Agent 引擎
内置工具：rag_search（语义检索）、time_filter（时间线筛选）
架构可扩展，未来可加入更多工具
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from llm_client import LLMClient
from rag_engine import RAGEngine
from database import DatabaseManager

logger = logging.getLogger(__name__)


class Tool:
    """工具基类"""
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}

    def execute(self, **kwargs) -> str:
        raise NotImplementedError

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class RAGSearchTool(Tool):
    """RAG 检索工具：搜索群聊历史档案（长期记忆）"""

    name = "rag_search"
    description = (
        "检索该 QQ 群的长期聊天档案。仅当用户询问过去发生过的事情、"
        "某人的历史言论、或需要了解群内话题的历史背景时使用。"
        "注意：此工具仅包含已持久化的历史记录，不包含当前正在进行的对话内容。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "用于历史档案检索的关键词或查询语句",
            },
        },
        "required": ["query"],
    }

    def __init__(self, rag_engine: RAGEngine, group_id: str, llm_client: LLMClient):
        self.rag_engine = rag_engine
        self.group_id = group_id
        self.llm_client = llm_client

    def execute(self, query: str) -> str:
        logger.info(f"Agent 调用 rag_search (长期记忆): query='{query}'")
        try:
            res = self.rag_engine.query(self.group_id, query, self.llm_client)
            context = res["context"]
            self._last_sources = res["sources"]
            
            # 统计信息透传
            stats = res.get("stats", {})
            stats_str = f"RAG_STATS:{stats.get('vector_count', 0)},{stats.get('fts_count', 0)},{stats.get('final_count', 0)}"
            
            return f"{context}\n\n<!-- {stats_str} -->"
        except Exception as e:
            logger.error(f"rag_search 执行失败：{str(e)}")
            return f"（历史检索失败：{str(e)}）"


class TimeFilterTool(Tool):
    """时间筛选工具：按时间段检索群聊历史消息"""

    name = "time_filter"
    description = (
        "按时间段检索该 QQ 群的聊天记录。当用户询问'昨天说了什么'、"
        "'上周的讨论'、'3天前的情况'或指定具体日期时使用。"
        "与 rag_search 不同，此工具按精确时间范围查询，适合时间线回顾。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "time_range": {
                "type": "string",
                "description": "时间范围描述，如：'昨天'、'上周'、'今天'、'3天前'、'2026-04-15'、'最近一周'",
            },
            "keywords": {
                "type": "string",
                "description": "可选，关键词过滤，只返回包含该关键词的消息",
            },
            "limit": {
                "type": "integer",
                "description": "最大返回条数，默认30",
            },
        },
        "required": ["time_range"],
    }

    def __init__(self, db_manager: DatabaseManager, group_id: str):
        self.db_manager = db_manager
        self.group_id = group_id
        self._last_count = 0

    def execute(self, time_range: str, keywords: Optional[str] = None, limit: int = 30) -> str:
        logger.info(f"Agent 调用 time_filter: time_range='{time_range}', keywords='{keywords}'")
        try:
            start_time, end_time = self._parse_time_range(time_range)
            results = self.db_manager.get_messages_by_time_range(
                self.group_id, start_time, end_time, keywords=keywords, limit=limit
            )
            self._last_count = len(results)

            if not results:
                return f"（在 '{time_range}' 时间段内未找到相关消息）"

            # Token 安全截断：中文约 1.5-2 字/token，6000 字 ≈ 3000-4000 token
            MAX_TOTAL_CHARS = 6000
            MAX_SINGLE_CHARS = 300
            total_chars = 0
            lines = []
            truncated_count = 0

            for item in results:
                user_name = item.get("user_name", "未知用户")
                user_id = item.get("user_id", "")
                content = item.get("content", "")
                raw_time = item.get("time", "")
                # 单条截断
                if len(content) > MAX_SINGLE_CHARS:
                    content = content[:MAX_SINGLE_CHARS] + "..."
                # 格式化时间
                if len(raw_time) >= 12 and raw_time.isdigit():
                    formatted_time = f"{raw_time[4:6]}-{raw_time[6:8]} {raw_time[8:10]}:{raw_time[10:12]}"
                else:
                    formatted_time = raw_time
                line = f"[{formatted_time}] {user_name}({user_id}): {content}"
                # 总长度截断
                if total_chars + len(line) > MAX_TOTAL_CHARS:
                    truncated_count = len(results) - len(lines)
                    break
                lines.append(line)
                total_chars += len(line) + 1  # +1 for newline

            context = "\n".join(lines)
            extra_note = f"\n（另有 {truncated_count} 条消息因长度限制未展示）" if truncated_count else ""
            return (
                f"<Time_Range_Query_Results time_range='{time_range}' count='{len(results)}' shown='{len(lines)}'>\n"
                f"{context}\n"
                f"</Time_Range_Query_Results>{extra_note}"
            )
        except Exception as e:
            logger.error(f"time_filter 执行失败：{str(e)}")
            return f"（时间筛选失败：{str(e)}）"

    @staticmethod
    def _parse_time_range(time_desc: str) -> tuple:
        """解析自然语言时间描述为 (start_time, end_time) 的 YYYYMMDDHHMMSS 字符串"""
        import re
        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        time_desc = time_desc.strip()

        # 今天
        if time_desc in ("今天", "今日"):
            return (today.strftime("%Y%m%d%H%M%S"), now.strftime("%Y%m%d%H%M%S"))

        # 昨天
        if time_desc in ("昨天", "昨日"):
            start = today - timedelta(days=1)
            end = start.replace(hour=23, minute=59, second=59)
            return (start.strftime("%Y%m%d%H%M%S"), end.strftime("%Y%m%d%H%M%S"))

        # 前天
        if time_desc in ("前天", "前日"):
            start = today - timedelta(days=2)
            end = start.replace(hour=23, minute=59, second=59)
            return (start.strftime("%Y%m%d%H%M%S"), end.strftime("%Y%m%d%H%M%S"))

        # 本周
        if time_desc in ("本周", "这周", "这一周"):
            weekday = today.weekday()  # Monday=0
            start = today - timedelta(days=weekday)
            return (start.strftime("%Y%m%d%H%M%S"), now.strftime("%Y%m%d%H%M%S"))

        # 上周
        if time_desc in ("上周", "上一周"):
            weekday = today.weekday()
            end = today - timedelta(days=weekday + 1)
            start = end.replace(hour=0, minute=0, second=0)
            return (start.strftime("%Y%m%d%H%M%S"), end.strftime("%Y%m%d%H%M%S"))

        # 最近N天 / N天前
        m = re.match(r'最近?(\d+)\s*天[前]?', time_desc)
        if m:
            n = int(m.group(1))
            if "前" in time_desc:
                start = today - timedelta(days=n)
                end = start.replace(hour=23, minute=59, second=59)
                return (start.strftime("%Y%m%d%H%M%S"), end.strftime("%Y%m%d%H%M%S"))
            else:
                start = today - timedelta(days=n - 1)
                return (start.strftime("%Y%m%d%H%M%S"), now.strftime("%Y%m%d%H%M%S"))

        # 最近N小时 / N小时前
        m = re.match(r'最近?(\d+)\s*个?小时[前]?', time_desc)
        if m:
            n = int(m.group(1))
            start = now - timedelta(hours=n)
            return (start.strftime("%Y%m%d%H%M%S"), now.strftime("%Y%m%d%H%M%S"))

        # 尝试 YYYY-MM-DD 或 YYYYMMDD
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                dt = datetime.strptime(time_desc, fmt)
                start = dt.replace(hour=0, minute=0, second=0)
                end = dt.replace(hour=23, minute=59, second=59)
                return (start.strftime("%Y%m%d%H%M%S"), end.strftime("%Y%m%d%H%M%S"))
            except ValueError:
                pass

        # 默认最近24小时
        start = now - timedelta(hours=24)
        return (start.strftime("%Y%m%d%H%M%S"), now.strftime("%Y%m%d%H%M%S"))


class Agent:
    """
    多轮工具调用 Agent
    支持推理 -> 调用工具 -> 观察结果 -> 再推理 的循环
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rag_engine: Optional[RAGEngine] = None,
        db_manager: Optional[DatabaseManager] = None,
        max_tool_rounds: int = 3,
        max_history_turns: int = 10,
    ):
        self.llm_client = llm_client or LLMClient()
        self.rag_engine = rag_engine or RAGEngine()
        self.db_manager = db_manager
        self.max_tool_rounds = max_tool_rounds
        self.max_history_turns = max_history_turns

    def _build_system_prompt(self) -> str:
        return (
            "你是群聊成员 DeepSleep，也是一个可以调用工具的 Agent。\n\n"
            "【记忆层级优先原则 - 严格遵循】\n"
            "第一优先级（Immediate Context）：当前 messages 数组中的直接对话。\n"
            "  用户提到的'刚才'、'刚才那句'、'你刚才说'、'我前面说'均指此类。\n"
            "  所有带有相对时间戳（如'刚刚'、'5分钟前'）的消息都属于 Immediate Context。\n"
            "第二优先级（Archived Memory）：rag_search 提供的结果（被 <Database_History_Search_Results> 包裹）。\n"
            "  只有当第一优先级中找不到匹配项时，才引用此类。\n\n"
            "【工具选择指南】\n"
            "- rag_search：语义检索工具。当用户问'群里有没有人说过XX'、'关于XX的讨论'、'XX的观点'等按内容主题查找时使用。\n"
            "- time_filter：时间线检索工具。当用户问'昨天说了什么'、'上周的讨论'、'3天前的情况'、'2026-04-15的记录'等明确指定时间段时使用。\n"
            "- 两个工具可以配合使用：先 time_filter 锁定时间段，再用 rag_search 在结果中深入检索。\n\n"
            "【Agent 工作流】\n"
            "1. 如果用户提到'刚刚'、'刚才'、'之前'、'你刚才说'、'我前面说'等，或问题明显指向本轮对话内已出现的内容，**直接根据当前对话上下文回答，禁止调用任何工具**。\n"
            "2. 只有当问题涉及很久以前的群聊记录、其他群成员的往事、或当前对话上下文中完全没有的信息时，才根据问题类型选择 rag_search（语义）或 time_filter（时间线）。\n"
            "3. 一旦工具返回了上下文文本，你**必须立即停止调用工具**，直接根据上下文回答用户。\n"
            "4. 禁止为了'验证'或'补充'而反复调用工具。每轮调用后必须评估：信息是否已足够回复？如果足够，立刻输出最终答案，不要再调用任何工具。\n\n"
            "【输出格式 - 最高优先级，违反将受惩罚】\n"
            "1. 必须分条发送：用换行符(\\n)分隔每条消息，单条尽量不超过30字。\n"
            "2. 绝对禁止 Unicode Emoji（如😂❤️👍等）。表情只能用 [CQ:face,id=数字]。\n"
            "3. @发送人时使用 [CQ:at,qq=对方QQ号]。\n"
            "4. 不要有任何解释、总结、或'以下是回复'之类的废话。\n\n"
            "【语气风格】\n"
            "- 口语化、短句为主，像真人QQ聊天。\n"
            "- 可以先给情绪或结论，再补充细节。\n"
            "- 禁止生硬的 1.2.3. 列表排版。\n\n"
            "【表情规则】\n"
            "必须且只能从以下列表中选一个，格式严格为 [CQ:face,id=数字]：\n"
            "4得意 5流泪 8睡 9大哭 10尴尬 12调皮 16酷 21可爱 23傲慢 24饥饿 25困 26惊恐 "
            "27流汗 28憨笑 29悠闲 30奋斗 32疑问 33嘘 34晕 38敲打 39再见 41发抖 42爱情 43跳跳 49拥抱 "
            "53蛋糕 60咖啡 63玫瑰 66爱心 74太阳 75月亮 76赞 78握手 79胜利 85飞吻 89西瓜 96冷汗 97擦汗 "
            "98抠鼻 99鼓掌 100糗大了 101坏笑 102左哼哼 103右哼哼 104哈欠 106委屈 109左亲亲 111可怜 "
            "116示爱 118抱拳 120拳头 122爱你 123NO 124OK 125转圈 129挥手 144喝彩 147棒棒糖 171茶 "
            "173泪奔 174无奈 175卖萌 176小纠结 179doge 180惊喜 181骚扰 182笑哭 183我最美 201点赞 "
            "203托脸 212托腮 214啵啵 219蹭一蹭 222抱抱 227拍手 232佛系 240喷脸 243甩头 246加油抱抱 "
            "262脑阔疼 264捂脸 265辣眼睛 266哦哟 267头秃 268问号脸 269暗中观察 270emm 271吃瓜 272呵呵哒 "
            "273我酸了 277汪汪 278汗 281无眼笑 282敬礼 284面无表情 285摸鱼 287哦 289睁眼 290敲开心 "
            "293摸锦鲤 294期待 297拜谢 298元宝 299牛啊 305右亲亲 306牛气冲天 307喵喵 314仔细分析 "
            "315加油 318崇拜 319比心 320庆祝 322拒绝 324吃糖 326生气"
        )

    def _build_user_prompt(self, group_id: str, user_id: str, user_name: str, content: str) -> str:
        return (
            f"【当前群号】{group_id}\n"
            f"【发送人】{user_name}({user_id})\n"
            f"【用户消息】{content}"
        )

    @staticmethod
    def _format_relative_time(dt_str: str) -> str:
        """将 SQLite CURRENT_TIMESTAMP 转为相对时间描述"""
        if not dt_str:
            return ""
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            now = datetime.now()
            delta = now - dt
            seconds = int(delta.total_seconds())
            if seconds < 60:
                return "刚刚"
            elif seconds < 3600:
                return f"{seconds // 60}分钟前"
            elif seconds < 86400:
                return f"{seconds // 3600}小时前"
            else:
                return f"{seconds // 86400}天前"
        except Exception:
            return ""

    def _format_history_message(self, h: Dict) -> str:
        """将 Conversation_History 记录统一封装为与当前消息一致的群聊格式"""
        role = h["role"]
        content = h["content"]
        user_name = h.get("user_name", "")
        created_at = h.get("created_at", "")
        relative_time = self._format_relative_time(created_at)
        time_tag = f"（{relative_time}）" if relative_time else ""

        if role == "user":
            return f"【发送人】{user_name}{time_tag}\n【用户消息】{content}"
        else:  # assistant
            return f"【发送人】DeepSleep{time_tag}\n【回复】{content}"

    def run(
        self,
        group_id: str,
        user_id: str,
        user_name: str,
        content: str,
        cache_messages: List[Dict],
    ) -> Dict[str, Any]:
        """
        Agent 主入口
        :return: 包含 response、tool_calls_history、rewritten_query 的字典
        """
        logger.info(f"Agent 收到请求：群 {group_id}，用户 {user_id}，消息 '{content[:30]}...'")

        # 初始化工具
        rag_tool = RAGSearchTool(self.rag_engine, group_id, self.llm_client)
        time_filter_tool = TimeFilterTool(self.db_manager, group_id) if self.db_manager else None
        tools = [rag_tool] + ([time_filter_tool] if time_filter_tool else [])
        tools_schema = [t.to_openai_schema() for t in tools]
        tool_map = {t.name: t for t in tools}

        # 构建对话历史
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
        ]
        if self.db_manager:
            history = self.db_manager.get_conversation_history(group_id, user_id, limit=self.max_history_turns)
            for h in history:
                messages.append({"role": h["role"], "content": self._format_history_message(h)})
        messages.append({"role": "user", "content": self._build_user_prompt(group_id, user_id, user_name, content)})

        tool_calls_history = []
        rewritten_query = None

        for round_idx in range(self.max_tool_rounds):
            logger.info(f"Agent 进入第 {round_idx + 1} 轮推理")

            try:
                result = self.llm_client.chat_completion_stream(
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto",
                    temperature=0.7,
                    max_tokens=800,
                )
            except Exception as e:
                logger.error(f"LLM 流式调用失败：{str(e)}")
                raise RuntimeError(f"LLM 流式调用失败：{str(e)}")

            # 提取模型返回内容
            assistant_message = result.get("message", {})
            finish_reason = result.get("finish_reason", "stop")

            # 记录改写后的查询（如果模型在第一轮回复中直接暴露，或从 tool_call 参数中提取）
            assistant_content = assistant_message.get("content", "")

            # 判断是否调用工具
            tool_calls = assistant_message.get("tool_calls", [])
            if finish_reason != "tool_calls" or not tool_calls:
                # 模型不再调用工具，输出最终回复
                logger.info(f"Agent 在第 {round_idx + 1} 轮结束，生成最终回复")
                final_response = assistant_content or "（模型未返回内容）"
                if self.db_manager:
                    self.db_manager.save_conversation_turn(group_id, user_id, user_name, "user", content)
                    self.db_manager.save_conversation_turn(group_id, user_id, "DeepSleep", "assistant", final_response)
                return {
                    "response": final_response,
                    "sources": self._extract_sources(rag_tool),
                    "tool_calls_history": tool_calls_history,
                    "rewritten_query": rewritten_query or content,
                }

            # 模型要求调用工具：将 assistant message 加入历史
            messages.append(assistant_message)

            # 逐条执行工具调用
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                tc_func = tc.get("function", {})
                tc_name = tc_func.get("name", "")
                tc_args_str = tc_func.get("arguments", "{}")

                try:
                    tc_args = json.loads(tc_args_str) if tc_args_str else {}
                except json.JSONDecodeError:
                    tc_args = {}

                # 记录改写查询（rag_search 的第一个 query 参数）
                if tc_name == "rag_search" and tc_args.get("query") and not rewritten_query:
                    rewritten_query = tc_args.get("query")

                tool_calls_history.append({
                    "round": round_idx + 1,
                    "tool": tc_name,
                    "arguments": tc_args,
                })

                if tc_name in tool_map:
                    try:
                        tool_result = tool_map[tc_name].execute(**tc_args)
                    except Exception as e:
                        tool_result = f"工具执行出错：{str(e)}"
                else:
                    tool_result = f"未知工具：{tc_name}"

                # 追加停止调用提示，抑制模型反复检索
                tool_result += (
                    "\n\n[系统提示] 以上是你请求的上下文。"
                    "如果群聊历史中没有相关信息，请先检查当前对话上下文（已包含在消息历史中），"
                    "不要因为没有找到群聊记录就否认用户提到的内容。"
                    "信息足够时立即停止调用工具，直接输出最终回复。"
                )

                # 构造 tool 结果消息（OpenAI 兼容格式）
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tc_name,
                    "content": str(tool_result),
                })
                logger.info(f"Tool {tc_name} 执行完成，结果长度={len(str(tool_result))}")

        # 达到最大轮次，强制生成最终回复
        logger.info("Agent 达到最大工具调用轮次，强制生成最终回复")
        try:
            result = self.llm_client.chat_completion_stream(
                messages=messages,
                tools=tools_schema,
                tool_choice="none",
                temperature=0.7,
                max_tokens=800,
            )
            final_content = result.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"最终回复生成失败：{str(e)}")
            final_content = "（Agent 思考超时，请稍后重试）"

        if self.db_manager:
            self.db_manager.save_conversation_turn(group_id, user_id, user_name, "user", content)
            self.db_manager.save_conversation_turn(group_id, user_id, "DeepSleep", "assistant", final_content)
        return {
            "response": final_content,
            "sources": self._extract_sources(rag_tool),
            "tool_calls_history": tool_calls_history,
            "rewritten_query": rewritten_query or content,
        }

    def run_stream(
        self,
        group_id: str,
        user_id: str,
        user_name: str,
        content: str,
        cache_messages: List[Dict],
    ):
        """
        Agent 流式主入口（Generator）。
        产生的事件类型：
        - round_start: {"type": "round_start", "round": int}
        - reasoning:   {"type": "reasoning", "content": str}
        - token:       {"type": "token", "content": str}
        - tool_call:   {"type": "tool_call", "tool": str, "arguments": dict}
        - tool_result: {"type": "tool_result", "tool": str, "result": str}
        - response_complete: {"type": "response_complete", "content": str}
        """
        logger.info(f"Agent 收到请求：群 {group_id}，用户 {user_id}，消息 '{content[:30]}...'")

        rag_tool = RAGSearchTool(self.rag_engine, group_id, self.llm_client)
        time_filter_tool = TimeFilterTool(self.db_manager, group_id) if self.db_manager else None
        tools = [rag_tool] + ([time_filter_tool] if time_filter_tool else [])
        tools_schema = [t.to_openai_schema() for t in tools]
        tool_map = {t.name: t for t in tools}

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
        ]
        if self.db_manager:
            history = self.db_manager.get_conversation_history(group_id, user_id, limit=self.max_history_turns)
            for h in history:
                messages.append({"role": h["role"], "content": self._format_history_message(h)})
        messages.append({"role": "user", "content": self._build_user_prompt(group_id, user_id, user_name, content)})

        rewritten_query = None

        for round_idx in range(self.max_tool_rounds):
            logger.info(f"Agent 进入第 {round_idx + 1} 轮推理")
            yield {"type": "round_start", "round": round_idx + 1}

            full_content = ""
            full_reasoning = ""
            current_tool_calls = {}
            final_finish_reason = None

            try:
                for event in self.llm_client.chat_completion_stream_iter(
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto",
                    temperature=0.7,
                    max_tokens=800,
                ):
                    delta = event["delta"]
                    finish_reason = event["finish_reason"]

                    if delta.get("reasoning_content"):
                        chunk = delta["reasoning_content"]
                        full_reasoning += chunk
                        yield {"type": "reasoning", "content": chunk}

                    if delta.get("content"):
                        chunk = delta["content"]
                        full_content += chunk
                        yield {"type": "token", "content": chunk}

                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            if idx not in current_tool_calls:
                                current_tool_calls[idx] = {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if tc.get("id"):
                                current_tool_calls[idx]["id"] = tc["id"]
                            if tc.get("type"):
                                current_tool_calls[idx]["type"] = tc["type"]
                            func = tc.get("function", {})
                            if func.get("name"):
                                current_tool_calls[idx]["function"]["name"] = func["name"]
                            if func.get("arguments"):
                                current_tool_calls[idx]["function"]["arguments"] += func["arguments"]

                    if finish_reason:
                        final_finish_reason = finish_reason

            except Exception as e:
                logger.error(f"LLM 流式调用失败：{str(e)}")
                raise RuntimeError(f"LLM 流式调用失败：{str(e)}")

            tool_calls_list = [current_tool_calls[i] for i in sorted(current_tool_calls.keys())]
            if tool_calls_list and final_finish_reason != "tool_calls":
                final_finish_reason = "tool_calls"

            if final_finish_reason != "tool_calls" or not tool_calls_list:
                yield {"type": "response_complete", "content": full_content}
                if self.db_manager:
                    self.db_manager.save_conversation_turn(group_id, user_id, user_name, "user", content)
                    self.db_manager.save_conversation_turn(group_id, user_id, "DeepSleep", "assistant", full_content)
                return

            # 将 assistant message 加入历史
            messages.append({
                "role": "assistant",
                "content": full_content,
                "tool_calls": tool_calls_list,
            })

            # 执行工具调用
            for tc in tool_calls_list:
                tc_id = tc.get("id", "")
                tc_func = tc.get("function", {})
                tc_name = tc_func.get("name", "")
                tc_args_str = tc_func.get("arguments", "{}")

                try:
                    tc_args = json.loads(tc_args_str) if tc_args_str else {}
                except json.JSONDecodeError:
                    tc_args = {}

                if tc_name == "rag_search" and tc_args.get("query") and not rewritten_query:
                    rewritten_query = tc_args.get("query")

                yield {"type": "tool_call", "tool": tc_name, "arguments": tc_args}

                if tc_name in tool_map:
                    try:
                        tool_result = tool_map[tc_name].execute(**tc_args)
                    except Exception as e:
                        tool_result = f"工具执行出错：{str(e)}"
                else:
                    tool_result = f"未知工具：{tc_name}"

                tool_result += (
                    "\n\n[系统提示] 以上是你请求的上下文。"
                    "如果群聊历史中没有相关信息，请先检查当前对话上下文（已包含在消息历史中），"
                    "不要因为没有找到群聊记录就否认用户提到的内容。"
                    "信息足够时立即停止调用工具，直接输出最终回复。"
                )

                yield {"type": "tool_result", "tool": tc_name, "result": tool_result}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tc_name,
                    "content": str(tool_result),
                })
                logger.info(f"Tool {tc_name} 执行完成，结果长度={len(str(tool_result))}")

        # 达到最大轮次，强制生成最终回复
        logger.info("Agent 达到最大工具调用轮次，强制生成最终回复")
        try:
            full_content = ""
            for event in self.llm_client.chat_completion_stream_iter(
                messages=messages,
                tools=tools_schema,
                tool_choice="none",
                temperature=0.7,
                max_tokens=800,
            ):
                delta = event["delta"]
                if delta.get("content"):
                    full_content += delta["content"]
                    yield {"type": "token", "content": delta["content"]}
        except Exception as e:
            logger.error(f"最终回复生成失败：{str(e)}")
            full_content = "（Agent 思考超时，请稍后重试）"

        yield {"type": "response_complete", "content": full_content}
        if self.db_manager:
            self.db_manager.save_conversation_turn(group_id, user_id, user_name, "user", content)
            self.db_manager.save_conversation_turn(group_id, user_id, "DeepSleep", "assistant", full_content)

    def _extract_sources(self, rag_tool: RAGSearchTool) -> List[Dict]:
        """从 RAG 工具的执行状态中回溯来源摘要"""
        return getattr(rag_tool, "_last_sources", [])
