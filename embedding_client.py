#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地 ONNX 嵌入模型客户端（轻量版）
使用 int8 量化模型，适配 8C16G 云服务器
"""

import os
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(
        self,
        model_dir: str = None
    ):
        from config import Config
        self.model_dir = os.path.abspath(model_dir or Config.EMBEDDING_MODEL_DIR)
        self.onnx_path = os.path.join(self.model_dir, "model_int8.onnx")
        self.tokenizer = None
        self.session = None
        self._dim = 0
        self._load()

    def _load(self):
        from transformers import AutoTokenizer
        import onnxruntime as ort

        logger.info(f"加载本地 tokenizer: {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)

        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX 模型不存在: {self.onnx_path}")

        logger.info(f"加载 ONNX: {self.onnx_path}")
        sess = ort.SessionOptions()
        sess.enable_cpu_mem_arena = False
        sess.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess.intra_op_num_threads = 2  # 8核服务器留有余量
        self.session = ort.InferenceSession(
            self.onnx_path, sess, providers=["CPUExecutionProvider"]
        )

        self._dim = self.encode(["test"]).shape[1]
        logger.info(f"Embedding 就绪，维度={self._dim}")

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        tok = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        inputs = {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}
        if "token_type_ids" in {x.name for x in self.session.get_inputs()}:
            inputs["token_type_ids"] = tok.get("token_type_ids")

        out = self.session.run(None, inputs)[0]

        # mean pooling
        mask = np.expand_dims(tok["attention_mask"], -1).astype(np.float32)
        pooled = np.sum(out * mask, axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)

        # L2 normalize
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return (pooled / norms).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dim
