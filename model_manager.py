#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型自动下载管理器
首次启动时检测模型缺失，从远程下载 tar.gz 并解压
"""

import os
import logging
import tarfile
from typing import Optional
import requests
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)


MODEL_DOWNLOAD_URLS = {
    "bge-large-zh-v1.5": os.getenv(
        "MODEL_URL_EMBEDDING",
        "https://heiservers.top/download/bge-large-zh-v1.5.tar.gz"
    ),
    "bge-reranker-base": os.getenv(
        "MODEL_URL_RERANKER",
        "https://heiservers.top/download/bge-reranker-base.tar.gz"
    ),
}


def _human_readable_size(size_bytes: int) -> str:
    """将字节数转换为人类可读格式"""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def _download_file(url: str, dest_path: str, chunk_size: int = 8192) -> bool:
    """流式下载文件并显示 tqdm 进度条"""
    logger.info(f"开始下载：{url}")
    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        total_str = _human_readable_size(total) if total else "未知大小"

        print(f"  文件大小：{total_str}")
        with open(dest_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="  下载进度",
            ncols=80,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        logger.info(f"下载完成：{dest_path}")
        return True
    except Exception as e:
        logger.error(f"下载失败：{url}，错误：{str(e)}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def _extract_tar(tar_path: str, extract_to: str) -> bool:
    """解压 tar.gz 文件并显示 tqdm 进度条"""
    logger.info(f"解压 {tar_path} 到 {extract_to}")
    try:
        os.makedirs(extract_to, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            members = tf.getmembers()
            with tqdm(
                total=len(members),
                desc="  解压进度",
                ncols=80,
                unit="file",
            ) as bar:
                for member in members:
                    tf.extract(member, extract_to)
                    bar.update(1)
        logger.info("解压完成")
        return True
    except Exception as e:
        logger.error(f"解压失败：{str(e)}")
        return False


def ensure_model(model_name: str, model_dir: str, url: Optional[str] = None) -> bool:
    """
    确保单个模型已存在。
    通过检查目录内是否有 model_int8.onnx 来判断。
    """
    onnx_path = os.path.join(model_dir, "model_int8.onnx")
    if os.path.exists(onnx_path):
        logger.info(f"模型 {model_name} 已存在，跳过下载")
        return True

    download_url = url or MODEL_DOWNLOAD_URLS.get(model_name)
    if not download_url or download_url.startswith("https://your-cdn"):
        logger.warning(f"模型 {model_name} 缺失，但未配置有效下载链接")
        return False

    models_root = os.path.dirname(model_dir)
    os.makedirs(models_root, exist_ok=True)
    tar_path = os.path.join(models_root, f"{model_name}.tar.gz")

    print(f"\n[模型下载] {model_name} 未找到，开始从远程下载...")
    if not _download_file(download_url, tar_path):
        return False

    print(f"[模型解压] {model_name} ...")
    if not _extract_tar(tar_path, models_root):
        return False

    # 删除 tar.gz 包释放空间
    try:
        os.remove(tar_path)
        logger.info(f"已清理临时 tar.gz：{tar_path}")
    except Exception:
        pass

    # 再次检查模型文件
    if os.path.exists(onnx_path):
        print(f"[模型就绪] {model_name}\n")
        return True
    else:
        logger.error(f"解压后仍未找到 {onnx_path}，请检查 tar.gz 包内部路径")
        return False


def ensure_all_models() -> bool:
    """
    主入口：检查 CxData.json 和 Data.db 是否存在，
    若存在且模型缺失，则自动下载并解压。
    """
    cxdata_exists = os.path.exists(Config.CXDATA_PATH)
    db_exists = os.path.exists(Config.DB_PATH)

    if not cxdata_exists and not db_exists:
        logger.info("未检测到 CxData.json 和 Data.db，跳过模型自动下载（可能尚未部署到插件目录）")
        return True

    if not cxdata_exists:
        logger.warning(f"CxData.json 不存在：{Config.CXDATA_PATH}")
    if not db_exists:
        logger.warning(f"Data.db 不存在：{Config.DB_PATH}")

    logger.info("检测到插件环境，开始检查本地模型...")

    results = []
    results.append(ensure_model("bge-large-zh-v1.5", Config.EMBEDDING_MODEL_DIR))
    results.append(ensure_model("bge-reranker-base", Config.RERANKER_MODEL_DIR))

    return all(results)
