#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LanceDB 向量存储管理器
按群号_日期分表，支持从 SQLite 自动增量同步
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import lancedb
import pyarrow as pa
from embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)


class LanceDBManager:
    """LanceDB 向量存储管理器"""

    def __init__(self, db_path: str = None, embedding_client: Optional[EmbeddingClient] = None):
        from config import Config
        self.db_path = db_path or Config.VECTOR_DB_PATH
        self.embedding_client = embedding_client or EmbeddingClient()
        self._ensure_dir()
        self.db = lancedb.connect(self.db_path)
        self._sync_meta_table = "_sync_metadata"
        self._ensure_sync_meta_table()

    def _ensure_dir(self):
        """确保向量数据库目录存在"""
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path, exist_ok=True)

    def _ensure_sync_meta_table(self):
        """确保同步元数据表存在，用于记录每张表的已同步最大 message_id"""
        try:
            self.db.open_table(self._sync_meta_table)
        except Exception:
            schema = pa.schema([
                pa.field("table_name", pa.string()),
                pa.field("last_message_id", pa.int64()),
                pa.field("updated_at", pa.string()),
            ])
            self.db.create_table(self._sync_meta_table, schema=schema)

    def _get_table_name(self, group_id: str, date_str: str) -> str:
        """生成 LanceDB 表名：group_{group_id}_{YYYYMMDD}"""
        return f"group_{group_id}_{date_str}"

    def _get_sync_record(self, table_name: str) -> int:
        """获取某张表已同步的最大 message_id"""
        try:
            tbl = self.db.open_table(self._sync_meta_table)
            df = tbl.to_pandas()
            row = df[df["table_name"] == table_name]
            if not row.empty:
                return int(row.iloc[0]["last_message_id"])
        except Exception:
            pass
        return 0

    def _update_sync_record(self, table_name: str, last_message_id: int):
        """更新同步元数据"""
        try:
            tbl = self.db.open_table(self._sync_meta_table)
            # LanceDB 暂不支持单行 update，采用先读全表、替换行、再覆盖的方式
            df = tbl.to_pandas()
            # 过滤掉旧记录
            records = []
            for _, row in df.iterrows():
                if row["table_name"] != table_name:
                    records.append({
                        "table_name": str(row["table_name"]),
                        "last_message_id": int(row["last_message_id"]),
                        "updated_at": str(row["updated_at"]),
                    })
            records.append({
                "table_name": table_name,
                "last_message_id": last_message_id,
                "updated_at": datetime.now().isoformat(),
            })
            # 转换为 pyarrow Table
            new_table = pa.Table.from_pylist(records)
            self.db.drop_table(self._sync_meta_table)
            self.db.create_table(self._sync_meta_table, data=new_table)
        except Exception as e:
            logger.error(f"更新同步元数据失败：{str(e)}")

    def sync_table(self, sqlite_conn, group_id: str, date_str: str) -> bool:
        """
        从 SQLite 指定日期表增量同步到 LanceDB
        :param sqlite_conn: SQLite 连接对象
        :param group_id: QQ 群号
        :param date_str: 日期字符串（YYYYMMDD）
        :return: 是否成功同步
        """
        sqlite_table = f"Chat_{date_str}"
        lance_table = self._get_table_name(group_id, date_str)
        last_synced_id = self._get_sync_record(lance_table)

        try:
            cursor = sqlite_conn.execute(
                f"SELECT id, group_id, user_name, user_id, time, content FROM {sqlite_table} "
                f"WHERE group_id = ? AND id > ? ORDER BY id ASC",
                (group_id, last_synced_id)
            )
            rows = cursor.fetchall()
        except Exception as e:
            logger.warning(f"读取 SQLite 表 {sqlite_table} 失败（可能表不存在）：{str(e)}")
            return False

        if not rows:
            logger.debug(f"表 {lance_table} 无新增数据，跳过同步")
            return True

        logger.info(f"正在同步 {len(rows)} 条消息到 {lance_table}（分批编码，每批 32 条）")

        # 分批处理，避免一次性 encode 大 batch 导致 ONNX Runtime 内存暴涨
        batch_size = 32
        total = len(rows)
        max_id = 0

        for start in range(0, total, batch_size):
            chunk = rows[start:start + batch_size]
            contents = [row[5] for row in chunk]
            embeddings = self.embedding_client.encode(contents)

            if embeddings.shape[0] != len(chunk):
                logger.error(f"批次 {start}-{start + len(chunk)} 嵌入数量不匹配，同步失败")
                return False

            data = {
                "vector": [embeddings[i].tolist() for i in range(len(chunk))],
                "message_id": [row[0] for row in chunk],
                "group_id": [row[1] for row in chunk],
                "user_name": [row[2] for row in chunk],
                "user_id": [row[3] for row in chunk],
                "time": [row[4] for row in chunk],
                "content": [row[5] for row in chunk],
                "table_name": [sqlite_table] * len(chunk),
            }

            try:
                pa_table = pa.Table.from_pydict(data)
                try:
                    tbl = self.db.open_table(lance_table)
                    tbl.add(pa_table)
                except Exception:
                    self.db.create_table(lance_table, data=pa_table)
            except Exception as e:
                logger.error(f"写入 LanceDB 批次 {start} 失败：{str(e)}")
                return False

            max_id = max(max_id, max(row[0] for row in chunk))

        self._update_sync_record(lance_table, max_id)
        logger.info(f"同步完成：{lance_table}，最大 message_id = {max_id}")
        return True

    def sync_recent_tables(self, sqlite_conn, group_id: str, days: int = 3, cleanup_days: int = 7):
        """
        同步最近 N 天的所有表，并自动清理超过保留天数的旧表
        :param sqlite_conn: SQLite 连接对象
        :param group_id: QQ 群号
        :param days: 回溯天数（默认 3 天）
        :param cleanup_days: 数据保留天数（默认 7 天）
        """
        today = datetime.now()
        for i in range(days):
            date_str = (today - timedelta(days=i)).strftime("%Y%m%d")
            self.sync_table(sqlite_conn, group_id, date_str)
        self.cleanup_group_tables(group_id, retention_days=cleanup_days)

    def cleanup_group_tables(self, group_id: str, retention_days: int = 7):
        """
        清理指定群中超过保留天数的旧向量表
        :param group_id: QQ 群号
        :param retention_days: 保留天数（默认 7 天）
        :return: 删除的表名列表
        """
        cutoff = (datetime.now() - timedelta(days=retention_days)).strftime("%Y%m%d")
        prefix = f"group_{group_id}_"
        deleted = []

        try:
            tables = self.db.list_tables()
            # 兼容不同 lancedb 版本的返回格式
            table_names = []
            if hasattr(tables, '__iter__'):
                for t in tables:
                    name = t[0] if isinstance(t, tuple) else str(t)
                    table_names.append(name)

            for name in table_names:
                if name.startswith(prefix):
                    date_part = name.replace(prefix, "")
                    if date_part.isdigit() and len(date_part) == 8 and date_part < cutoff:
                        try:
                            self.db.drop_table(name)
                            deleted.append(name)
                            logger.info(f"已清理旧向量表：{name}（超过 {retention_days} 天）")
                        except Exception as e:
                            logger.warning(f"清理向量表 {name} 失败：{str(e)}")
        except Exception as e:
            logger.error(f"清理群 {group_id} 旧向量表时出错：{str(e)}")

        return deleted

    def cleanup_all_groups(self, retention_days: int = 7):
        """
        扫描所有 group_ 开头的向量表，清理超过保留天数的旧表
        :param retention_days: 保留天数（默认 7 天）
        :return: 总共删除的表名列表
        """
        cutoff = (datetime.now() - timedelta(days=retention_days)).strftime("%Y%m%d")
        deleted = []

        try:
            tables = self.db.list_tables()
            table_names = []
            if hasattr(tables, '__iter__'):
                for t in tables:
                    name = t[0] if isinstance(t, tuple) else str(t)
                    table_names.append(name)

            for name in table_names:
                if name.startswith("group_"):
                    parts = name.split("_")
                    if len(parts) >= 3:
                        date_part = parts[-1]
                        if date_part.isdigit() and len(date_part) == 8 and date_part < cutoff:
                            try:
                                self.db.drop_table(name)
                                deleted.append(name)
                                logger.info(f"已清理旧向量表：{name}（超过 {retention_days} 天）")
                            except Exception as e:
                                logger.warning(f"清理向量表 {name} 失败：{str(e)}")
        except Exception as e:
            logger.error(f"全局清理旧向量表时出错：{str(e)}")

        return deleted

    def search(self, group_id: str, query_vector: np.ndarray, date_str: str, top_k: int = 20) -> List[Dict]:
        """
        在指定群_日期的 LanceDB 表中搜索最近邻
        :param group_id: QQ 群号
        :param query_vector: 查询向量
        :param date_str: 日期字符串（YYYYMMDD）
        :param top_k: 返回结果数
        :return: 候选消息列表
        """
        lance_table = self._get_table_name(group_id, date_str)
        try:
            tbl = self.db.open_table(lance_table)
            results = (
                tbl.search(query_vector.tolist())
                .limit(top_k)
                .to_pandas()
            )
            return results.to_dict("records")
        except Exception:
            # 表不存在或无数据时返回空列表
            return []

    def search_multi_days(self, group_id: str, query_vector: np.ndarray, days: int = 3, top_k_per_day: int = 20) -> List[Dict]:
        """
        在最近 N 天的多个表中搜索并合并结果
        :param group_id: QQ 群号
        :param query_vector: 查询向量
        :param days: 回溯天数
        :param top_k_per_day: 每天返回的结果数
        :return: 合并后的候选消息列表
        """
        today = datetime.now()
        all_results = []
        for i in range(days):
            date_str = (today - timedelta(days=i)).strftime("%Y%m%d")
            day_results = self.search(group_id, query_vector, date_str, top_k_per_day)
            all_results.extend(day_results)
        return all_results
