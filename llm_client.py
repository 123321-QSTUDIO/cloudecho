#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 客户端
处理与 LLM API 的通信（对应易语言中的 _AI聊天_请求LLM 功能）
"""

import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.base_url = os.getenv('BOT_API_BASE', 'https://api.deepseek.com/v1/chat/completions')
        self.api_key = os.getenv('BOT_API_KEY')
        self.model = os.getenv('BOT_API_MODEL', 'deepseek-chat')
        self.timeout = 30

        if not self.api_key:
            logger.warning("未找到 API 密钥，LLM 功能将受限")

    def _build_messages(self, group_id: str, user_id: str, message_content: str) -> List[Dict[str, str]]:
        """
        构建消息数组（类似易语言的角色设定）
        包含 system 角色（群聊特定指令）和 user 角色（用户消息）
        """
        system_prompt = (
            "你是群聊成员 DeepSleep。\n\n"
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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message_content}
        ]
        return messages

    def get_response(self, group_id: str, user_id: str, message_content: str) -> str:
        """获取 LLM 对群消息的回复"""
        if not self.api_key:
            raise ValueError("未配置 API 密钥，请设置 BOT_API_KEY 环境变量")

        messages = self._build_messages(group_id, user_id, message_content)
        result = self.chat_completion(messages=messages)
        return result.get("message", {}).get("content", "")

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = "auto",
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> Dict[str, Any]:
        """
        通用 LLM 对话接口，支持工具调用（OpenAI / DeepSeek 兼容格式）
        :return: 包含 message、finish_reason 的字典
        """
        if not self.api_key:
            raise ValueError("未配置 API 密钥，请设置 BOT_API_KEY 环境变量")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            logger.info(f"请求 LLM，消息数={len(messages)}，tools={'有' if tools else '无'}")
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            response_data = response.json()

            if "choices" not in response_data or len(response_data["choices"]) == 0:
                raise ValueError("LLM 响应格式异常：缺少 choices")

            choice = response_data["choices"][0]
            finish_reason = choice.get("finish_reason", "stop")
            message = choice.get("message", {})

            # 确保 tool_calls 结构完整
            if "tool_calls" in message and message["tool_calls"]:
                # 修正 finish_reason（部分模型可能不返回 tool_calls）
                finish_reason = "tool_calls"

            logger.info(f"收到 LLM 回复，finish_reason={finish_reason}")
            return {
                "message": message,
                "finish_reason": finish_reason,
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API 请求失败：{str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get('error', {}).get('message', str(e))
                except Exception:
                    error_msg = str(e)
            else:
                error_msg = str(e)
            raise RuntimeError(f"LLM API 错误：{error_msg}")

        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"解析 LLM 响应失败：{str(e)}")
            raise RuntimeError(f"解析 LLM 响应失败：{str(e)}")

    def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = "auto",
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> Dict[str, Any]:
        """
        流式 LLM 对话接口，支持在流中检测工具调用并立即聚合。
        一旦 finish_reason 为 tool_calls，即可中断并开始执行工具。
        返回格式与 chat_completion 完全一致：{"message": ..., "finish_reason": ...}
        """
        if not self.api_key:
            raise ValueError("未配置 API 密钥，请设置 BOT_API_KEY 环境变量")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(f"请求 LLM 流式接口，消息数={len(messages)}，tools={'有' if tools else '无'}")

        content_parts = []
        tool_calls_agg: Dict[int, Dict[str, Any]] = {}
        finish_reason = "stop"

        try:
            with requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta", {})

                    if delta.get("content"):
                        content_parts.append(delta["content"])

                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            if idx not in tool_calls_agg:
                                tool_calls_agg[idx] = {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if tc.get("id"):
                                tool_calls_agg[idx]["id"] = tc["id"]
                            if tc.get("type"):
                                tool_calls_agg[idx]["type"] = tc["type"]
                            func = tc.get("function", {})
                            if func.get("name"):
                                tool_calls_agg[idx]["function"]["name"] = func["name"]
                            if func.get("arguments"):
                                tool_calls_agg[idx]["function"]["arguments"] += func["arguments"]

                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

        except requests.exceptions.RequestException as e:
            logger.error(f"LLM 流式请求失败：{str(e)}")
            raise RuntimeError(f"LLM API 错误：{str(e)}")

        tool_calls_list = [tool_calls_agg[i] for i in sorted(tool_calls_agg.keys())]
        if tool_calls_list and finish_reason != "tool_calls":
            finish_reason = "tool_calls"

        logger.info(f"流式接收完成，finish_reason={finish_reason}，content_len={len(''.join(content_parts))}")
        return {
            "message": {
                "role": "assistant",
                "content": "".join(content_parts),
                "tool_calls": tool_calls_list,
            },
            "finish_reason": finish_reason,
        }

    def chat_completion_stream_iter(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = "auto",
        temperature: float = 0.7,
        max_tokens: int = 800,
    ):
        """
        流式 LLM 对话接口（Generator 版本）。
        逐 chunk yield delta 和 finish_reason，供 CLI/TUI 实时消费。
        """
        if not self.api_key:
            raise ValueError("未配置 API 密钥，请设置 BOT_API_KEY 环境变量")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(f"请求 LLM 流式接口（iter），消息数={len(messages)}，tools={'有' if tools else '无'}")

        try:
            with requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choice = chunk.get("choices", [{}])[0]
                    yield {
                        "delta": choice.get("delta", {}),
                        "finish_reason": choice.get("finish_reason"),
                    }
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM 流式请求失败：{str(e)}")
            raise RuntimeError(f"LLM API 错误：{str(e)}")
