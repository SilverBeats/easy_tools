#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数值与词向量工具。"""

import numpy as np



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算两个向量的余弦相似度

    Args:
        a: shape = (a_len, emb_dim)
        b: shape = (b_len, emb_dim)

    Returns:
        np.ndarray: 余弦相似度矩阵 shape = (a_len, b_len)
    """
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    sim_matrix = np.dot(a, b.T) / (a_norm * b_norm.T)
    return sim_matrix

