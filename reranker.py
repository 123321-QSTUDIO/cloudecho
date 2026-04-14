#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 重排序器（轻量版）
使用 int8 量化模型，适配 8C16G 云服务器
"""

import os
import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)


class ONNXReranker:
    def __init__(
        self,
        model_dir: str = None
    ):
        from config import Config
        self.model_dir = os.path.abspath(model_dir or Config.RERANKER_MODEL_DIR)
        self.onnx_path = os.path.join(self.model_dir, "model_int8.onnx")
        self.tokenizer = None
        self.session = None
        self._load()

    def _load(self):
        from transformers import AutoTokenizer
        import onnxruntime as ort

        logger.info(f"加载本地 tokenizer: {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX 模型不存在: {self.onnx_path}")

        logger.info(f"加载 ONNX: {self.onnx_path}")
        sess = ort.SessionOptions()
        sess.enable_cpu_mem_arena = False
        sess.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            self.onnx_path, sess, providers=["CPUExecutionProvider"]
        )
        logger.info("Reranker 就绪")

    def rerank(self, query: str, candidates: List[Dict], batch_size: int = 8) -> List[Dict]:
        if not candidates:
            return []

        pairs = [(query, c.get("content", "")) for c in candidates]
        scores = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            tok = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="np"
            )
            inputs = {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}
            if "token_type_ids" in {x.name for x in self.session.get_inputs()}:
                inputs["token_type_ids"] = tok.get("token_type_ids")

            scores.extend(self.session.run(None, inputs)[0].flatten().tolist())

        for idx, c in enumerate(candidates):
            c["rerank_score"] = float(scores[idx])

        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
