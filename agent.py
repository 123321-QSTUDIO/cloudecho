#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持多轮工具调用的 Agent 引擎
目前内置工具：rag_search（群聊历史检索）
架构可扩展，未来可加入更多工具
"""

import json
import logging
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
    """RAG 检索工具：搜索群聊历史和缓存消息"""

    name = "rag_search"
    description = (
        "搜索指定 QQ 群的聊天记录（包括已持久化到数据库的历史消息和未写入数据库的近期缓存消息），"
        "返回与查询话题相关的上下文文本。当用户问题可能涉及群内往事、人物、事件、讨论内容时，"
        "必须优先调用此工具获取上下文后再作答。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "用于搜索的查询语句，建议简洁明确，能准确描述用户想了解的话题",
            },
        },
        "required": ["query"],
    }

    def __init__(self, rag_engine: RAGEngine, group_id: str, cache_messages: List[Dict]):
        self.rag_engine = rag_engine
        self.group_id = group_id
        self.cache_messages = cache_messages

    def execute(self, query: str) -> str:
        logger.info(f"Agent 调用 rag_search: query='{query}'")
        try:
            candidates = self.rag_engine.hybrid_retrieve(self.group_id, query, self.cache_messages)
            ranked = self.rag_engine.rerank_candidates(query, candidates)
            context = self.rag_engine.build_context(ranked)
            # 缓存来源摘要，供最终响应返回
            self._last_sources = [
                {
                    "user_name": c.get("user_name", ""),
                    "user_id": c.get("user_id", ""),
                    "time": c.get("time", ""),
                    "content": c.get("content", "")[:100],
                    "score": round(c.get("rerank_score", 0.0), 4),
                    "source": c.get("source", "history"),
                }
                for c in ranked
            ]
            return context
        except Exception as e:
            logger.error(f"rag_search 执行失败：{str(e)}")
            return f"（检索失败：{str(e)}）"


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
            "【Agent 工作流 - 严格遵循】\n"
            "1. 当用户问题涉及群内历史、往事、人物、讨论时，调用 rag_search 获取上下文。\n"
            "2. 一旦 rag_search 返回了上下文文本，你**必须立即停止调用工具**，直接根据上下文回答用户。\n"
            "3. 禁止为了'验证'或'补充'而反复调用 rag_search。每轮调用后必须评估：信息是否已足够回复？如果足够，立刻输出最终答案，不要再调用任何工具。\n\n"
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
        rag_tool = RAGSearchTool(self.rag_engine, group_id, cache_messages)
        tools = [rag_tool]
        tools_schema = [t.to_openai_schema() for t in tools]
        tool_map = {t.name: t for t in tools}

        # 构建对话历史
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
        ]
        if self.db_manager:
            history = self.db_manager.get_conversation_history(group_id, user_id, limit=self.max_history_turns)
            for h in history:
                messages.append({"role": h["role"], "content": h["content"]})
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
                    self.db_manager.save_conversation_turn(group_id, user_id, "user", content)
                    self.db_manager.save_conversation_turn(group_id, user_id, "assistant", final_response)
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
                    "如果这些信息已足够回答用户问题，请立即停止调用任何工具，直接输出最终回复。"
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
            self.db_manager.save_conversation_turn(group_id, user_id, "user", content)
            self.db_manager.save_conversation_turn(group_id, user_id, "assistant", final_content)
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

        rag_tool = RAGSearchTool(self.rag_engine, group_id, cache_messages)
        tools = [rag_tool]
        tools_schema = [t.to_openai_schema() for t in tools]
        tool_map = {t.name: t for t in tools}

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
        ]
        if self.db_manager:
            history = self.db_manager.get_conversation_history(group_id, user_id, limit=self.max_history_turns)
            for h in history:
                messages.append({"role": h["role"], "content": h["content"]})
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
                    self.db_manager.save_conversation_turn(group_id, user_id, "user", content)
                    self.db_manager.save_conversation_turn(group_id, user_id, "assistant", full_content)
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
                    "如果这些信息已足够回答用户问题，请立即停止调用任何工具，直接输出最终回复。"
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
            self.db_manager.save_conversation_turn(group_id, user_id, "user", content)
            self.db_manager.save_conversation_turn(group_id, user_id, "assistant", full_content)

    def _extract_sources(self, rag_tool: RAGSearchTool) -> List[Dict]:
        """从 RAG 工具的执行状态中回溯来源摘要"""
        return getattr(rag_tool, "_last_sources", [])
