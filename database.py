#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QQ 群聊天数据库管理器
处理 SQLite 操作，采用按日期分表的表结构
"""

import sqlite3
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path=None):
        from config import Config
        self.db_path = db_path or Config.DB_PATH
        self._ensure_directories_exist()

    def _ensure_directories_exist(self):
        """确保数据库所在目录存在"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    def _get_connection(self):
        """创建数据库连接"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode = WAL')
        return conn

    def _get_table_name(self):
        """获取当前日期的表名（格式：Chat_YYYYMMDD）"""
        return f"Chat_{datetime.now().strftime('%Y%m%d')}"

    def _get_status_table_name(self):
        """获取状态表名"""
        return "Chunk_Status"

    def init_database(self):
        """初始化数据库，创建必需的表"""
        with self._get_connection() as conn:
            # 创建状态表
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._get_status_table_name()} (
                    group_id TEXT,
                    table_name TEXT,
                    last_id INTEGER,
                    PRIMARY KEY(group_id, table_name)
                )
            """)
            conn.commit()
            logger.info("数据库初始化完成")

    def _ensure_table_exists(self, conn, table_name):
        """确保聊天表存在"""
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT,
                user_name TEXT,
                user_id TEXT,
                time TEXT,
                content TEXT
            )
        """)

    def get_history(self, group_id, limit=100, start_time=None):
        """获取指定群的聊天历史"""
        table_name = self._get_table_name()

        with self._get_connection() as conn:
            self._ensure_table_exists(conn, table_name)

            query = f"SELECT id, group_id, user_name, user_id, time, content FROM {table_name} WHERE group_id = ?"
            params = [group_id]

            if start_time:
                query += " AND time >= ?"
                params.append(start_time)

            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            # 按时间正序返回
            return [
                {
                    'id': row[0],
                    'group_id': row[1],
                    'user_name': row[2],
                    'user_id': row[3],
                    'time': row[4],
                    'content': row[5]
                }
                for row in reversed(rows)
            ]

    def list_active_groups(self):
        """列出所有有聊天记录的活跃群"""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'Chat_%'")
            tables = [row[0] for row in cursor.fetchall()]

            groups = set()
            for table in tables:
                try:
                    cursor = conn.execute(f"SELECT DISTINCT group_id FROM {table}")
                    for row in cursor.fetchall():
                        if row[0]:
                            groups.add(row[0])
                except Exception:
                    continue

            # 也检查状态表
            try:
                cursor = conn.execute(f"SELECT DISTINCT group_id FROM {self._get_status_table_name()}")
                for row in cursor.fetchall():
                    if row[0]:
                        groups.add(row[0])
            except Exception:
                pass

            return sorted(list(groups))

    def get_stats(self):
        """获取数据库统计信息"""
        with self._get_connection() as conn:
            # 获取所有聊天表
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'Chat_%'")
            tables = [row[0] for row in cursor.fetchall()]

            total_messages = 0
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                total_messages += cursor.fetchone()[0]

            return {
                'total_tables': len(tables),
                'total_messages': total_messages,
                'tables': tables
            }

    def insert_message(self, group_id: str, user_id: str, user_name: str, time: str, content: str) -> int:
        """
        插入单条消息到当前日期的 Chat_YYYYMMDD 表
        :return: 插入的行 id
        """
        table_name = self._get_table_name()
        with self._get_connection() as conn:
            self._ensure_table_exists(conn, table_name)
            cursor = conn.execute(
                f"INSERT INTO {table_name} (group_id, user_name, user_id, time, content) VALUES (?, ?, ?, ?, ?)",
                (group_id, user_name, user_id, time, content)
            )
            conn.commit()
            return cursor.lastrowid

    def cleanup_old_tables(self, days=30):
        """清理旧的数据表（可选维护操作）"""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'Chat_%'")
            tables = [row[0] for row in cursor.fetchall()]
            return len(tables)
