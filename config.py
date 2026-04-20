#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QQ 群 LLM API 服务配置
默认路径基于运行目录自动推导，适配两种部署模式：
  1. CQ 插件模式：运行目录/app/CX Bot Test/AI/
  2. 独立部署模式：运行目录本身即为数据根目录
"""

import os


def _resolve_dirs():
    """
    推导 AI 数据目录和插件目录。
    返回 (ai_dir, plugin_dir) 元组。

    优先级：
    1. CQ 插件环境：运行目录下存在 app/CX Bot Test/CxData.json
       → ai_dir = app/CX Bot Test/AI, plugin_dir = app/CX Bot Test
    2. 环境变量 AI_DIR（显式指定）
    3. 当前目录下有 AI/ 子目录 → 使用 cwd/AI
    4. 当前目录本身就是 AI/ → 使用 cwd
    5. 源码所在目录下有 AI/ 子目录 → 开发模式回退
    6. 源码所在目录本身 → 回退
    7. 最终回退：当前工作目录
    """
    cwd = os.getcwd()
    src_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. CQ 插件环境检测（最高优先级）
    cq_plugin_dir = os.path.join(cwd, "app", "CX Bot Test")
    cq_cxdata = os.path.join(cq_plugin_dir, "CxData.json")
    if os.path.isfile(cq_cxdata):
        return os.path.join(cq_plugin_dir, "AI"), cq_plugin_dir

    # 2. 环境变量
    env_ai_dir = os.getenv("AI_DIR")
    if env_ai_dir:
        ai = os.path.abspath(env_ai_dir)
        return ai, os.path.dirname(ai)

    # 3. 当前目录下有 AI/ 子目录
    cwd_ai = os.path.join(cwd, "AI")
    if os.path.isdir(cwd_ai):
        return cwd_ai, cwd

    # 4. 当前目录本身就是 AI/
    if os.path.basename(cwd).upper() == "AI":
        return cwd, os.path.dirname(cwd)

    # 5. 源码目录下有 AI/ 子目录（开发模式）
    src_ai = os.path.join(src_dir, "AI")
    if os.path.isdir(src_ai):
        return src_ai, src_dir

    # 6. 源码目录本身
    if os.path.basename(src_dir).upper() == "AI":
        return src_dir, os.path.dirname(src_dir)

    # 7. 默认：当前工作目录
    return cwd, cwd


_AI_DIR, _PLUGIN_DIR = _resolve_dirs()


class Config:
    """API 服务器配置类"""
    # 服务器配置
    HOST = os.getenv('API_HOST', '0.0.0.0')
    PORT = int(os.getenv('API_PORT', '5000'))
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

    # 群配置文件（默认在插件目录下）
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
