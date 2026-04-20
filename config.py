#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QQ 群 LLM API 服务配置
默认路径基于本文件所在目录自动推导，适配酷Q目录结构：
  酷Q根目录/app/CX Bot Test/AI/  <-- 本文件位置
"""

import os


def _resolve_ai_dir():
    """
    推导 AI 数据目录的优先级：
    1. 环境变量 AI_DIR（显式指定）
    2. 当前工作目录的 AI/ 子目录存在 → 使用 cwd/AI（独立模式）
    3. 当前工作目录本身就是 AI/ → 使用 cwd
    4. 源码所在目录的 AI/ 子目录存在 → 使用源码目录/AI（开发模式）
    5. 源码所在目录本身 → 回退
    6. 最终回退：当前工作目录
    """
    cwd = os.getcwd()
    src_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 环境变量
    env_ai_dir = os.getenv("AI_DIR")
    if env_ai_dir:
        return os.path.abspath(env_ai_dir)

    # 2. 当前目录下有 AI/ 子目录（独立部署：用户在项目根目录运行）
    cwd_ai = os.path.join(cwd, "AI")
    if os.path.isdir(cwd_ai):
        return cwd_ai

    # 3. 当前目录本身就是 AI/
    if os.path.basename(cwd).upper() == "AI":
        return cwd

    # 4. 源码目录下有 AI/ 子目录（开发模式：从 git 仓库运行）
    src_ai = os.path.join(src_dir, "AI")
    if os.path.isdir(src_ai):
        return src_ai

    # 5. 源码目录本身
    if os.path.basename(src_dir).upper() == "AI":
        return src_dir

    # 6. 默认：当前工作目录（最符合服务器部署直觉）
    return cwd


_AI_DIR = _resolve_ai_dir()
_PLUGIN_DIR = os.path.dirname(_AI_DIR) if os.path.basename(_AI_DIR).upper() == "AI" else _AI_DIR


class Config:
    """API 服务器配置类"""
    # 服务器配置
    HOST = os.getenv('API_HOST', '0.0.0.0')
    PORT = int(os.getenv('API_PORT', 5000))
    DEBUG = os.getenv('API_DEBUG', 'False').lower() == 'true'

    # 数据库配置（确保指向正确的 Data.db）
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
