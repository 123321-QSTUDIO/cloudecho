#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QQ 群 LLM API 服务配置
默认路径基于本文件所在目录自动推导，适配酷Q目录结构：
  酷Q根目录/app/CX Bot Test/AI/  <-- 本文件位置
"""

import os

# 计算固定基准目录
_AI_DIR = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_DIR = os.path.dirname(_AI_DIR)


class Config:
    """API 服务器配置类"""
    # 服务器配置
    HOST = os.getenv('API_HOST', '0.0.0.0')
    PORT = int(os.getenv('API_PORT', 5000))
    DEBUG = os.getenv('API_DEBUG', 'False').lower() == 'true'

    # 数据库配置（默认在 AI 子目录内）
    DB_PATH = os.getenv('DB_PATH', os.path.join(_AI_DIR, 'Data.db'))
    VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', os.path.join(_AI_DIR, 'VectorDB'))

    # 模型路径配置（默认在 AI 子目录内的 models/ 下）
    EMBEDDING_MODEL_DIR = os.getenv(
        'EMBEDDING_MODEL_DIR',
        os.path.join(_AI_DIR, 'models', 'bge-large-zh-v1.5')
    )
    RERANKER_MODEL_DIR = os.getenv(
        'RERANKER_MODEL_DIR',
        os.path.join(_AI_DIR, 'models', 'bge-reranker-base')
    )

    # 群配置文件（默认在 AI 的父目录 CX Bot Test 下）
    CXDATA_PATH = os.getenv('CXDATA_PATH', os.path.join(_PLUGIN_DIR, 'CxData.json'))

    # LLM 配置
    BOT_API_KEY = os.getenv('BOT_API_KEY', '')
    BOT_API_BASE = os.getenv('BOT_API_BASE', 'https://api.deepseek.com/v1/chat/completions')
    BOT_API_MODEL = os.getenv('BOT_API_MODEL', 'deepseek-chat')

    # 日志配置
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    # 请求配置
    REQUEST_TIMEOUT = 30  # 超时时间（秒）
    MAX_RETRY_ATTEMPTS = 3  # 最大重试次数
