#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 引擎（检索增强生成）
提供查询重写、混合检索、重排序、上下文组装等底层能力
LLM 生成逻辑已上移至 Agent 层
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

from embedding_client import EmbeddingClient
from reranker import ONNXReranker
from vector_store import LanceDBManager
from database import DatabaseManager
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG 引擎：查询 -> 检索 -> 重排 -> 组装上下文 -> LLM 回复"""

    def __init__(
        self,
        embedding_client: Optional[EmbeddingClient] = None,
        reranker: Optional[ONNXReranker] = None,
        vector_store: Optional[LanceDBManager] = None,
        db_manager: Optional[DatabaseManager] = None,
        retrieve_days: int = 3,
        rerank_top_k: int = 30,
        final_top_n: int = 10,
    ):
        self.embedding_client = embedding_client or EmbeddingClient()
        self.reranker = reranker or ONNXReranker()
        self.vector_store = vector_store or LanceDBManager(embedding_client=self.embedding_client)
        self.db_manager = db_manager or DatabaseManager()
        self.retrieve_days = retrieve_days
        self.rerank_top_k = rerank_top_k
        self.final_top_n = final_top_n

    def retrieve_historical(self, group_id: str, query: str) -> List[Dict]:
        """从历史数据库中检索相关消息（自动同步 + 向量检索）"""
        import sqlite3
        conn = sqlite3.connect(self.db_manager.db_path, check_same_thread=False)
        try:
            # 自动同步最近 N 天的数据到 LanceDB
            self.vector_store.sync_recent_tables(conn, group_id, days=self.retrieve_days)
        finally:
            conn.close()

        # 编码查询向量
        query_vector = self.embedding_client.encode([query])[0]

        # 跨天检索
        candidates = self.vector_store.search_multi_days(
            group_id=group_id,
            query_vector=query_vector,
            days=self.retrieve_days,
            top_k_per_day=self.rerank_top_k // self.retrieve_days + 5,
        )

        logger.info(f"历史检索完成：群 {group_id} 召回 {len(candidates)} 条候选")
        return candidates

    def retrieve_from_cache(self, query: str, cache_messages: List[Dict]) -> List[Dict]:
        """从缓存消息中检索与查询最相关的条目"""
        if not cache_messages:
            return []

        # 过滤掉无效项
        valid_cache = [m for m in cache_messages if isinstance(m, dict) and m.get("content")]
        if not valid_cache:
            return []

        contents = [m["content"] for m in valid_cache]
        embeddings = self.embedding_client.encode(contents)
        query_vector = self.embedding_client.encode([query])[0]

        # 计算余弦相似度并排序
        import numpy as np
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized = embeddings / norms
        query_norm = np.linalg.norm(query_vector)
        query_norm = query_norm if query_norm > 0 else 1e-10
        query_normalized = query_vector / query_norm

        similarities = np.dot(normalized, query_normalized)

        # 为缓存项补充字段，使其与历史消息字段一致
        results = []
        for idx, msg in enumerate(valid_cache):
            results.append({
                "message_id": msg.get("id", 0),
                "group_id": msg.get("group_id", ""),
                "user_name": msg.get("user_name", ""),
                "user_id": msg.get("user_id", ""),
                "time": msg.get("time", datetime.now().isoformat()),
                "content": msg["content"],
                "table_name": "cache",
                "_distance": 1.0 - float(similarities[idx]),
                "source": "cache",
            })

        # 按相似度降序取前 K
        results.sort(key=lambda x: x["_distance"])
        top_k = min(len(results), self.rerank_top_k // 2)
        return results[:top_k]

    def hybrid_retrieve(self, group_id: str, query: str, cache_messages: List[Dict]) -> List[Dict]:
        """混合检索：合并历史消息和缓存消息"""
        historical = self.retrieve_historical(group_id, query)
        cache = self.retrieve_from_cache(query, cache_messages)

        # 合并并去重（按 content + user_id + time 去重）
        seen = set()
        merged = []
        for item in historical + cache:
            key = f"{item.get('user_id', '')}|{item.get('time', '')}|{item.get('content', '')}"
            if key not in seen:
                seen.add(key)
                merged.append(item)

        logger.info(f"混合检索完成：历史 {len(historical)} 条 + 缓存 {len(cache)} 条，去重后 {len(merged)} 条")
        return merged

    def rerank_candidates(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """使用 Cross-encoder 对候选结果重排序"""
        if not candidates:
            return []

        reranked = self.reranker.rerank(query, candidates)
        return reranked[:self.final_top_n]

    def build_context(self, candidates: List[Dict]) -> str:
        """
        将重排序后的候选消息组装为 LLM 上下文
        按时间正序排列，保留发言人、时间、内容
        """
        if not candidates:
            return "（暂无相关历史上下文）"

        # 按时间排序
        sorted_candidates = sorted(candidates, key=lambda x: x.get("time", ""))

        lines = []
        for item in sorted_candidates:
            user_name = item.get("user_name", "未知用户")
            user_id = item.get("user_id", "")
            time_str = item.get("time", "")[11:16] if item.get("time") else ""
            content = item.get("content", "")
            source_tag = "[缓存]" if item.get("source") == "cache" else ""
            lines.append(f"{source_tag}[{time_str}] {user_name}({user_id}): {content}")

        context = "\n".join(lines)

        # 简单截断保护（避免上下文过长）
        max_len = 8000
        if len(context) > max_len:
            context = context[:max_len] + "\n...（上下文已截断）"

        return context

    def rewrite_query(self, content: str, llm_client: LLMClient) -> str:
        """
        查询重写：使用 LLM 将口语化问题改写为更适合向量检索的查询语句。
        若改写失败则回退到原始问题。
        """
        prompt = (
            "请将用户的口语化问题改写为一个适合向量检索的简洁查询语句。"
            "保留核心实体（人名、物品、事件、时间等），去除闲聊语气。"
            "只返回改写后的查询文本，不要任何解释、引号或额外内容。\n\n"
            f"用户问题：{content}"
        )
        messages = [
            {"role": "system", "content": "你是一个查询优化助手，专门将口语化输入改写为检索友好的关键词查询。"},
            {"role": "user", "content": prompt},
        ]
        try:
            result = llm_client.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=120,
            )
            rewritten = result.get("message", {}).get("content", "").strip().strip('"').strip("'")
            if rewritten:
                logger.info(f"查询重写：'{content}' -> '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.error(f"查询重写失败：{str(e)}")
        return content

    def query(
        self,
        group_id: str,
        content: str,
        cache_messages: List[Dict],
        llm_client: LLMClient,
    ) -> Dict:
        """
        完整 RAG 链路入口（阶段 1~4）。
        返回：{
            "rewritten_query": str,
            "context": str,
            "sources": List[Dict],
        }
        """
        rewritten = self.rewrite_query(content, llm_client)
        candidates = self.hybrid_retrieve(group_id, rewritten, cache_messages)
        ranked = self.rerank_candidates(rewritten, candidates)
        context = self.build_context(ranked)
        sources = [
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
        return {
            "rewritten_query": rewritten,
            "context": context,
            "sources": sources,
        }

