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
from typing import List, Dict, Optional

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
            # 创建对话历史表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS Conversation_History (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    user_name TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 兼容旧表：添加 user_name 字段
            try:
                conn.execute("ALTER TABLE Conversation_History ADD COLUMN user_name TEXT")
            except sqlite3.OperationalError:
                pass
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_group_user
                ON Conversation_History(group_id, user_id, created_at)
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

    def save_conversation_turn(self, group_id: str, user_id: str, user_name: str, role: str, content: str) -> int:
        """保存单条对话记录"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO Conversation_History (group_id, user_id, user_name, role, content) VALUES (?, ?, ?, ?, ?)",
                (group_id, user_id, user_name, role, content)
            )
            conn.commit()
            return cursor.lastrowid

    def get_conversation_history(self, group_id: str, user_id: str, limit: int = 10) -> List[Dict]:
        """获取指定用户在指定群的最近对话历史（按时间正序）"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT role, content, user_name, created_at FROM Conversation_History WHERE group_id = ? AND user_id = ? ORDER BY id DESC LIMIT ?",
                (group_id, user_id, limit * 2)
            )
            rows = cursor.fetchall()
            # 按时间正序返回
            return [
                {"role": row[0], "content": row[1], "user_name": row[2], "created_at": row[3]}
                for row in reversed(rows)
            ]

    @staticmethod
    def _normalize_time(raw_time: str, table_date: str = None) -> str:
        """
        将各种时间格式归一化为 YYYYMMDDHHMMSS 的 14 位字符串，用于比较。
        支持的输入格式：
          - 14位数字：YYYYMMDDHHMMSS
          - 6位数字：HHMMSS → 用 table_date 或今天补齐日期
          - 带分隔符：YYYY-MM-DD HH:MM:SS / YYYY/MM/DD HH:MM:SS
        无法解析时返回空字符串。
        """
        if not raw_time:
            return ""
        s = str(raw_time).strip()

        # 14位纯数字
        if s.isdigit() and len(s) == 14:
            return s

        # 6位纯数字（HHMMSS），需要补齐日期
        if s.isdigit() and len(s) == 6:
            date_prefix = table_date or datetime.now().strftime("%Y%m%d")
            return date_prefix + s

        # 8位纯数字（YYYYMMDD），补齐时间
        if s.isdigit() and len(s) == 8:
            return s + "000000"

        # 标准格式带分隔符
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt).strftime("%Y%m%d%H%M%S")
            except ValueError:
                continue

        # 12位纯数字：YYMMDDHHMMSS
        if s.isdigit() and len(s) == 12:
            return "20" + s

        return ""

    def get_messages_by_time_range(
        self,
        group_id: str,
        start_time: str,
        end_time: str,
        keywords: Optional[str] = None,
        limit: int = 30,
    ) -> List[Dict]:
        """
        按时间范围查询消息（支持跨天表）。
        由于 SQLite 中 time 字段格式不统一（6位 HHMMSS / 14位 YYYYMMDDHHMMSS），
        不在 SQL 中直接比较，而是先按 group_id 和 keywords 查出后在 Python 中过滤。
        """
        # 从表名提取日期，用于补齐 6 位时间
        def _extract_date_from_table(table_name: str) -> str:
            if table_name.startswith("Chat_") and len(table_name) == 13:
                return table_name[5:]  # Chat_YYYYMMDD → YYYYMMDD
            return datetime.now().strftime("%Y%m%d")

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'Chat_%'"
            )
            tables = [row[0] for row in cursor.fetchall()]

            results = []
            for table in tables:
                table_date = _extract_date_from_table(table)
                query = (
                    f"SELECT id, group_id, user_name, user_id, time, content "
                    f"FROM {table} WHERE group_id = ?"
                )
                params = [group_id]

                if keywords:
                    query += " AND content LIKE ?"
                    params.append(f"%{keywords}%")

                query += " ORDER BY id DESC LIMIT ?"
                # 放宽 limit，防止过滤后数据太少
                params.append(limit * 3)

                try:
                    cursor = conn.execute(query, params)
                    for row in cursor.fetchall():
                        raw_time = row[4] if row[4] else ""
                        norm_time = self._normalize_time(raw_time, table_date)
                        # 只有成功解析且落在范围内才保留
                        if norm_time and start_time <= norm_time <= end_time:
                            results.append({
                                'id': row[0],
                                'group_id': row[1],
                                'user_name': row[2],
                                'user_id': row[3],
                                'time': raw_time,
                                'content': row[5],
                            })
                except Exception:
                    continue

            # 按归一化时间正序排列，截断到 limit
            def _sort_key(item):
                return self._normalize_time(item.get('time', ''), _extract_date_from_table(f"Chat_{datetime.now().strftime('%Y%m%d')}"))

            results.sort(key=_sort_key)
            return results[:limit]

    def cleanup_conversation_history(self, days: int = 30) -> int:
        """清理超过 N 天的旧对话记录"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM Conversation_History WHERE created_at < datetime('now', ?)",
                (f"-{days} days",)
            )
            conn.commit()
            return cursor.rowcount
