#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QQ 群 LLM API 服务
提供用于处理 QQ 群交互的 REST API 接口
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*has conflict with protected namespace.*",
    category=UserWarning,
)

from flask import Flask, request, jsonify
import logging
import os
import json
from datetime import datetime, timedelta
from database import DatabaseManager
from llm_client import LLMClient
from rag_engine import RAGEngine
from agent import Agent
from embedding_client import EmbeddingClient
from reranker import ONNXReranker
from vector_store import LanceDBManager
from model_manager import ensure_all_models
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# 初始化组件（路径统一从 Config 读取）
db_manager = DatabaseManager(db_path=Config.DB_PATH)
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
agent = Agent(llm_client=llm_client, rag_engine=rag_engine, db_manager=db_manager)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/qq/message', methods=['POST'])
def receive_qq_message():
    """
    接收单条 QQ 群消息并写入 SQLite。
    供 MiraiCQ 插件或消息采集脚本调用。
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '未提供 JSON 数据'}), 400

        required_fields = ['group_id', 'user_id', 'user_name', 'content']
        missing = [field for field in required_fields if field not in data]
        if missing:
            missing_str = ', '.join(missing)
            return jsonify({'error': f'缺少必需字段：{missing_str}'}), 400

        time_str = data.get('time')
        if not time_str:
            time_str = datetime.now().strftime('%Y%m%d%H%M%S')

        msg_id = db_manager.insert_message(
            group_id=str(data['group_id']),
            user_id=str(data['user_id']),
            user_name=str(data['user_name']),
            time=time_str,
            content=str(data['content']),
        )

        logger.info(f"写入消息：群 {data['group_id']}，id={msg_id}")
        return jsonify({
            'status': 'success',
            'message_id': msg_id,
            'group_id': data['group_id'],
        })

    except Exception as e:
        logger.error(f"处理消息接收请求时出错：{str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/qq/history', methods=['GET'])
def get_qq_history():
    """
    获取群聊历史记录（只读）
    参数：group_id（必需）、limit（默认 100）、start_time（可选）
    """
    try:
        group_id = request.args.get('group_id')
        if not group_id:
            return jsonify({'error': '缺少 group_id 参数'}), 400

        limit = int(request.args.get('limit', 100))
        start_time = request.args.get('start_time')

        messages = db_manager.get_history(
            group_id=group_id,
            limit=limit,
            start_time=start_time
        )

        return jsonify({
            'status': 'success',
            'count': len(messages),
            'messages': messages
        })

    except Exception as e:
        logger.error(f"获取历史记录时出错：{str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm/query', methods=['POST'])
def query_llm():
    """
    基础 LLM 回复（无 RAG，仅角色设定）
    必需参数：group_id, user_id, message_content
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '未提供 JSON 数据'}), 400

        required_fields = ['group_id', 'user_id', 'message_content']
        missing = [field for field in required_fields if field not in data]
        if missing:
            missing_str = ', '.join(missing)
            return jsonify({'error': f'缺少必需字段：{missing_str}'}), 400

        response = llm_client.get_response(
            group_id=data['group_id'],
            user_id=data['user_id'],
            message_content=data['message_content']
        )

        return jsonify({
            'status': 'success',
            'response': response
        })

    except Exception as e:
        logger.error(f"查询 LLM 时出错：{str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm/rag', methods=['POST'])
def query_llm_rag():
    """
    商用级 RAG 查询端点
    必需参数：group_id, user_id, user_name, content
    可选参数：cache（未写入数据库的近期消息列表）
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '未提供 JSON 数据'}), 400

        required_fields = ['group_id', 'user_id', 'user_name', 'content']
        missing = [field for field in required_fields if field not in data]
        if missing:
            missing_str = ', '.join(missing)
            return jsonify({'error': f'缺少必需字段：{missing_str}'}), 400

        cache = data.get('cache', [])
        if not isinstance(cache, list):
            return jsonify({'error': 'cache 必须是数组类型'}), 400

        # 通过 Agent 执行多轮工具调用 + RAG 回复
        result = agent.run(
            group_id=str(data['group_id']),
            user_id=str(data['user_id']),
            user_name=str(data['user_name']),
            content=str(data['content']),
            cache_messages=cache,
        )

        return jsonify({
            'status': 'success',
            'response': result['response'],
            'sources': result.get('sources', []),
            'rewritten_query': result.get('rewritten_query', ''),
            'tool_calls': result.get('tool_calls_history', []),
        })

    except Exception as e:
        logger.error(f"RAG 查询时出错：{str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sync/vectorstore', methods=['POST'])
def sync_vectorstore():
    """
    接收易语言插件写入数据库后的完整 JSON 缓存数据，
    解析群号及消息日期，触发对应群的向量数据库增量同步。

    期望的 JSON 格式与插件缓存结构一致：
    {
      "123456789": [
        {"QQ号": "123", "昵称": "User", "时间": "20240114120000", "内容": "msg"},
        ...
      ],
      "987654321": [
        ...
      ]
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '未提供 JSON 数据'}), 400

        if not isinstance(data, dict):
            return jsonify({'error': '数据必须是对象（键为群号）'}), 400

        conn = db_manager._get_connection()
        sync_results = []

        try:
            for group_id, messages in data.items():
                if not isinstance(messages, list):
                    continue

                # 从消息时间中提取涉及的所有日期
                date_set = set()
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    time_str = msg.get("时间", "")
                    parsed_date = _parse_e_time_to_date(time_str)
                    if parsed_date:
                        date_set.add(parsed_date)

                if not date_set:
                    # 没有时间信息时，默认同步最近 3 天
                    date_set = {
                        (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
                        for i in range(3)
                    }

                synced_tables = []
                for date_str in sorted(date_set):
                    success = rag_engine.vector_store.sync_table(conn, group_id, date_str)
                    synced_tables.append({
                        "date": date_str,
                        "table": f"group_{group_id}_{date_str}",
                        "success": success
                    })

                sync_results.append({
                    "group_id": group_id,
                    "message_count": len(messages),
                    "synced_tables": synced_tables
                })
                logger.info(f"向量库同步完成：群 {group_id}，涉及 {len(synced_tables)} 个日期表")

                # 自动清理该群超过 7 天的旧向量表
                deleted = rag_engine.vector_store.cleanup_group_tables(group_id, retention_days=7)
                if deleted:
                    logger.info(f"自动清理群 {group_id} 旧表：{deleted}")

        finally:
            conn.close()

        return jsonify({
            'status': 'success',
            'synced_groups': len(sync_results),
            'details': sync_results
        })

    except Exception as e:
        logger.error(f"向量库同步时出错：{str(e)}")
        return jsonify({'error': str(e)}), 500


def _parse_e_time_to_date(time_str: str) -> str:
    """
    解析易语言的时间格式为 YYYYMMDD 字符串。
    支持格式：
      - "181844"（6位纯时间 HHMMSS，默认返回当天日期）
      - "20240114120000"（14位日期时间）
      - "2024-01-14 12:00:00"
      - "2024/01/14 12:00:00"
    """
    if not time_str or not isinstance(time_str, str):
        return ""

    time_str = time_str.strip()

    # 6 位纯数字：HHMMSS，无日期信息，返回当前日期
    if time_str.isdigit() and len(time_str) == 6:
        return datetime.now().strftime("%Y%m%d")

    # 14 位纯数字：YYYYMMDDHHMMSS
    if time_str.isdigit() and len(time_str) >= 8:
        return time_str[:8]

    # 标准日期格式分隔符
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y%m%d%H%M%S"]:
        try:
            dt = datetime.strptime(time_str, fmt)
            return dt.strftime("%Y%m%d")
        except ValueError:
            continue

    # 兜底：只要前 8 位是数字就取前 8 位
    if time_str[:8].isdigit():
        return time_str[:8]

    return ""


@app.route('/api/qq/groups', methods=['GET'])
def list_groups():
    """列出活跃的 QQ 群（只读）"""
    try:
        groups = db_manager.list_active_groups()
        return jsonify({
            'status': 'success',
            'groups': groups
        })
    except Exception as e:
        logger.error(f"获取群列表时出错：{str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/qq/stats', methods=['GET'])
def get_stats():
    """获取数据库统计信息（只读）"""
    try:
        stats = db_manager.get_stats()
        return jsonify({
            'status': 'success',
            'stats': stats
        })
    except Exception as e:
        logger.error(f"获取统计信息时出错：{str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/cleanup', methods=['POST'])
def admin_cleanup():
    """
    手动触发向量数据库清理
    删除所有群超过 7 天的旧向量表
    """
    try:
        deleted = rag_engine.vector_store.cleanup_all_groups(retention_days=7)
        return jsonify({
            'status': 'success',
            'deleted_tables': deleted,
            'count': len(deleted)
        })
    except Exception as e:
        logger.error(f"手动清理向量库时出错：{str(e)}")
        return jsonify({'error': str(e)}), 500


def _load_cxdata_json(path: str):
    """加载 CxData.json，自动尝试 UTF-8 和 GBK/GB2312 编码"""
    for encoding in ["utf-8", "gbk", "gb2312"]:
        try:
            with open(path, "r", encoding=encoding) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeError("无法解析 CxData.json，所有支持的编码均失败")


def preload_vector_databases():
    """
    启动时预加载向量数据库：
    读取 CxData.json，为开启 AI 的群自动同步最近 3 天的聊天记录到 LanceDB
    """
    cxdata_path = Config.CXDATA_PATH
    if not os.path.exists(cxdata_path):
        logger.warning(f"未找到 CxData.json：{cxdata_path}，跳过启动预同步")
        return

    try:
        cxdata = _load_cxdata_json(cxdata_path)
    except Exception as e:
        logger.error(f"读取 CxData.json 失败：{str(e)}")
        return

    # 提取开启 AI 的群号（对应易语言：json.取通用属性("['群号'].AI") == "true"）
    ai_groups = []
    for group_id, group_config in cxdata.items():
        if isinstance(group_config, dict) and str(group_config.get("AI")).lower() == "true":
            ai_groups.append(group_id)

    if not ai_groups:
        logger.info("没有群开启 AI 功能，跳过向量数据库预同步")
        return

    logger.info(f"发现 {len(ai_groups)} 个群开启 AI，开始预同步向量数据库...")

    conn = db_manager._get_connection()
    try:
        for group_id in ai_groups:
            logger.info(f"预同步群 {group_id} 的向量数据（最近 3 天）...")
            try:
                rag_engine.vector_store.sync_recent_tables(conn, group_id, days=3)
            except Exception as e:
                logger.error(f"预同步群 {group_id} 失败：{str(e)}")

            # 清理该群超过 7 天的旧向量表
            try:
                deleted = rag_engine.vector_store.cleanup_group_tables(group_id, retention_days=7)
                if deleted:
                    logger.info(f"启动清理：群 {group_id} 删除 {len(deleted)} 个旧表")
            except Exception as e:
                logger.error(f"启动清理群 {group_id} 旧表失败：{str(e)}")
    finally:
        conn.close()

    logger.info("向量数据库预同步完成")


def main():
    # 首次启动自动检测并下载模型
    ensure_all_models()

    # 初始化数据库（仅创建缺失的表结构，不写入任何数据）
    db_manager.init_database()

    # 启动时预同步：为开启 AI 的群创建/更新向量数据库（最近 3 天）
    preload_vector_databases()

    # 运行服务器
    host = app.config.get('HOST', '0.0.0.0')
    port = app.config.get('PORT', 5000)
    debug = app.config.get('DEBUG', False)

    logger.info(f"在 {host}:{port} 启动 QQ LLM API 服务")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    main()
