#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 引擎（检索增强生成）
专注于长期记忆检索：向量召回 + 全文搜索
短期记忆由 Agent 上下文窗口直接处理
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from embedding_client import EmbeddingClient
from reranker import ONNXReranker
from vector_store import LanceDBManager
from database import DatabaseManager
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG 引擎：专注于长期记忆检索"""

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

    def retrieve_historical(self, group_id: str, query: str) -> Dict[str, Any]:
        """从历史数据库中混合检索（向量 + 全文）"""
        import sqlite3
        conn = sqlite3.connect(self.db_manager.db_path, check_same_thread=False)
        try:
            self.vector_store.sync_recent_tables(conn, group_id, days=self.retrieve_days)
        finally:
            conn.close()

        # 1. 向量检索
        query_vector = self.embedding_client.encode([query])[0]
        vector_results = self.vector_store.search_multi_days(
            group_id=group_id,
            query_vector=query_vector,
            days=self.retrieve_days,
            top_k_per_day=self.rerank_top_k // self.retrieve_days + 5,
        )
        for r in vector_results:
            r["source"] = "history_vector"

        # 2. 全文检索 (FTS)
        fts_results = self.vector_store.search_multi_days_fts(
            group_id=group_id,
            query=query,
            days=self.retrieve_days,
            top_k_per_day=self.rerank_top_k // self.retrieve_days + 5,
        )

        logger.info(f"历史检索完成：向量召回 {len(vector_results)} 条，全文召回 {len(fts_results)} 条")
        return {
            "vector_count": len(vector_results),
            "fts_count": len(fts_results),
            "merged_candidates": vector_results + fts_results
        }

    def hybrid_retrieve(self, group_id: str, query: str) -> Dict[str, Any]:
        """综合混合检索：合并长期记忆中的向量结果和全文结果"""
        hist_data = self.retrieve_historical(group_id, query)
        candidates = hist_data["merged_candidates"]

        # 去重
        seen = set()
        merged = []
        for item in candidates:
            key = f"{item.get('user_id', '')}|{item.get('time', '')}|{item.get('content', '')}"
            if key not in seen:
                seen.add(key)
                merged.append(item)

        return {
            "vector_count": hist_data["vector_count"],
            "fts_count": hist_data["fts_count"],
            "merged_count": len(merged),
            "candidates": merged
        }

    def rerank_candidates(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """使用 Cross-encoder 对候选结果重排序"""
        if not candidates:
            return []

        reranked = self.reranker.rerank(query, candidates)
        return reranked[:self.final_top_n]

    def build_context(self, candidates: List[Dict]) -> str:
        """组装 LLM 上下文（长期记忆专用标记）"""
        if not candidates:
            return "（暂无相关历史档案）"

        # Token 安全截断：中文约 1.5-2 字/token，8000 字 ≈ 4000-5000 token
        MAX_TOTAL_CHARS = 8000
        MAX_SINGLE_CHARS = 300
        total_chars = 0
        lines = []
        truncated_count = 0

        sorted_candidates = sorted(candidates, key=lambda x: x.get("time", ""))
        for item in sorted_candidates:
            user_name = item.get("user_name", "未知用户")
            user_id = item.get("user_id", "")
            content = item.get("content", "")
            # 单条截断
            if len(content) > MAX_SINGLE_CHARS:
                content = content[:MAX_SINGLE_CHARS] + "..."

            raw_time = item.get("time", "")
            formatted_time = ""
            if raw_time:
                if len(raw_time) >= 16 and "-" in raw_time and ":" in raw_time:
                    formatted_time = f"{raw_time[5:10]} {raw_time[11:16]}"
                elif len(raw_time) >= 12 and raw_time.isdigit():
                    formatted_time = f"{raw_time[4:6]}-{raw_time[6:8]} {raw_time[8:10]}:{raw_time[10:12]}"
                else:
                    formatted_time = raw_time

            line = f"[{formatted_time}] {user_name}({user_id}): {content}"
            if total_chars + len(line) > MAX_TOTAL_CHARS:
                truncated_count = len(sorted_candidates) - len(lines)
                break
            lines.append(line)
            total_chars += len(line) + 1  # +1 for newline

        context = "\n".join(lines)
        extra_note = f"\n（另有 {truncated_count} 条档案因长度限制未展示）" if truncated_count else ""
        return (
            "<Long_Term_Memory_Database_Results>\n"
            f"{context}\n"
            "</Long_Term_Memory_Database_Results>\n"
            f"注：以上为长期历史档案，非当前对话内容。{extra_note}"
        )

    def rewrite_query(self, content: str, llm_client: LLMClient) -> str:
        """查询重写"""
        prompt = (
            "请将用户的提问改写为一个适合检索历史聊天记录的简洁关键词查询。\n"
            f"用户问题：{content}"
        )
        messages = [
            {"role": "system", "content": "你是一个历史档案检索助手。"},
            {"role": "user", "content": prompt},
        ]
        try:
            result = llm_client.chat_completion(messages=messages, temperature=0.3, max_tokens=120)
            rewritten = result.get("message", {}).get("content", "").strip().strip('"').strip("'")
            if rewritten:
                return rewritten
        except Exception as e:
            logger.error(f"查询重写失败：{str(e)}")
        return content

    def query(
        self,
        group_id: str,
        content: str,
        llm_client: LLMClient,
    ) -> Dict:
        """完整长期记忆检索链路"""
        rewritten = self.rewrite_query(content, llm_client)
        
        # 仅从长期库检索
        retrieve_result = self.hybrid_retrieve(group_id, rewritten)
        candidates = retrieve_result["candidates"]
        
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
            "stats": {
                "vector_count": retrieve_result["vector_count"],
                "fts_count": retrieve_result["fts_count"],
                "final_count": len(ranked)
            }
        }
