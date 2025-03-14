#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-14 22:43
# @Desc   : 混合索引模块，负责管理向量索引和结构索引
# --------------------------------------------------------
"""

import faiss
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import torch
from collections import defaultdict
from utils.logger import Logger

class HybridIndex:
    """混合索引类
    
    包含两种索引：
    1. 向量索引：使用FAISS进行语义相似度搜索
    2. 结构索引：使用字典进行图结构特征的快速检索
    """
    
    def __init__(
        self,
        vector_dim: int = 768,
        index_type: str = "IVF100,Flat",
        nprobe: int = 10
    ):
        """初始化混合索引
        
        Args:
            vector_dim: 向量维度
            index_type: FAISS索引类型
            nprobe: FAISS搜索时探测的聚类数量
        """
        self.logger = Logger(self.__class__.__name__)
        
        # 向量索引
        self.quantizer = faiss.IndexFlatL2(vector_dim)
        self.vector_index = faiss.index_factory(vector_dim, index_type)
        
        if isinstance(self.vector_index, faiss.IndexIVF):
            self.vector_index.nprobe = nprobe
            
        # 结构索引
        self.structural_index = {
            "type": defaultdict(list),  # 按节点类型索引
            "degree": defaultdict(list),  # 按度数范围索引
            "attributes": defaultdict(list)  # 按属性值索引
        }
        
        # ID映射
        self.id_to_data = {}  # 存储原始数据
        self.id_to_vector = {}  # 存储向量表示
        
        # 是否已训练
        self.is_trained = False
        
    def add_item(
        self,
        item_id: str,
        vector: np.ndarray,
        type_label: str,
        degree: int,
        attributes: Dict[str, Any]
    ):
        """添加一个项目到索引
        
        Args:
            item_id: 项目ID
            vector: 向量表示
            type_label: 类型标签
            degree: 节点度数
            attributes: 属性字典
        """
        # 存储原始数据
        self.id_to_data[item_id] = {
            "type": type_label,
            "degree": degree,
            "attributes": attributes
        }
        
        # 存储向量表示
        self.id_to_vector[item_id] = vector
        
        # 更新结构索引
        self.structural_index["type"][type_label].append(item_id)
        
        # 按度数范围索引
        degree_range = self._get_degree_range(degree)
        self.structural_index["degree"][degree_range].append(item_id)
        
        # 按属性值索引
        for key, value in attributes.items():
            if isinstance(value, (str, int, float, bool)):
                self.structural_index["attributes"][f"{key}:{value}"].append(item_id)
                
    def build_index(self, batch_size: int = 1000):
        """构建向量索引
        
        Args:
            batch_size: 批处理大小
        """
        vectors = []
        ids = []
        
        # 收集所有向量
        for item_id, vector in self.id_to_vector.items():
            vectors.append(vector)
            ids.append(item_id)
            
        vectors = np.array(vectors).astype('float32')
        
        # 训练索引
        if not self.is_trained and isinstance(self.vector_index, faiss.IndexIVF):
            self.logger.info("训练向量索引...")
            self.vector_index.train(vectors)
            self.is_trained = True
            
        # 分批添加向量
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            self.vector_index.add(batch_vectors)
            
        self.logger.info(f"向量索引构建完成，共索引了 {len(self.id_to_data)} 个项目")
        
    def _get_degree_range(self, degree: int) -> str:
        """获取度数范围标签"""
        if degree <= 5:
            return "1-5"
        elif degree <= 10:
            return "6-10"
        elif degree <= 20:
            return "11-20"
        elif degree <= 50:
            return "21-50"
        else:
            return "50+"
            
    def search(
        self,
        query_vector: Optional[np.ndarray] = None,
        type_label: Optional[str] = None,
        min_degree: Optional[int] = None,
        max_degree: Optional[int] = None,
        attributes: Optional[Dict[str, Any]] = None,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """混合搜索
        
        Args:
            query_vector: 查询向量
            type_label: 类型标签
            min_degree: 最小度数
            max_degree: 最大度数
            attributes: 属性条件
            k: 返回的结果数量
            
        Returns:
            结果列表，每个元素是(item_id, score)元组
        """
        # 候选集
        candidates = set(self.id_to_data.keys())
        
        # 应用结构过滤
        if type_label:
            type_candidates = set(self.structural_index["type"][type_label])
            candidates &= type_candidates
            
        if min_degree is not None or max_degree is not None:
            degree_candidates = set()
            for item_id in candidates:
                degree = self.id_to_data[item_id]["degree"]
                if (min_degree is None or degree >= min_degree) and \
                   (max_degree is None or degree <= max_degree):
                    degree_candidates.add(item_id)
            candidates &= degree_candidates
            
        if attributes:
            attr_candidates = set()
            for key, value in attributes.items():
                attr_key = f"{key}:{value}"
                attr_candidates.update(self.structural_index["attributes"][attr_key])
            candidates &= attr_candidates
            
        if not candidates:
            return []
            
        # 如果没有查询向量，按ID排序返回
        if query_vector is None:
            return [(item_id, 1.0) for item_id in sorted(candidates)][:k]
            
        # 准备候选向量
        candidate_vectors = []
        candidate_ids = []
        
        for item_id in candidates:
            candidate_vectors.append(self.id_to_vector[item_id])
            candidate_ids.append(item_id)
            
        candidate_vectors = np.array(candidate_vectors).astype('float32')
        
        # 向量搜索
        if len(candidate_vectors) > 0:
            D, I = self.vector_index.search(
                query_vector.reshape(1, -1).astype('float32'),
                min(k, len(candidate_vectors))
            )
            results = [(candidate_ids[idx], score) for score, idx in zip(D[0], I[0])]
            return results
            
        return []
        
    def save(self, path: str):
        """保存索引到文件
        
        Args:
            path: 文件路径
        """
        import pickle
        
        # 保存FAISS索引
        faiss.write_index(self.vector_index, f"{path}.faiss")
        
        # 保存其他数据
        data = {
            "structural_index": self.structural_index,
            "id_to_data": self.id_to_data,
            "id_to_vector": self.id_to_vector,
            "is_trained": self.is_trained
        }
        
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(data, f)
            
        self.logger.info(f"索引已保存到 {path}")
        
    @classmethod
    def load(cls, path: str) -> "HybridIndex":
        """从文件加载索引
        
        Args:
            path: 文件路径
            
        Returns:
            加载的索引对象
        """
        import pickle
        
        # 加载FAISS索引
        vector_index = faiss.read_index(f"{path}.faiss")
        
        # 加载其他数据
        with open(f"{path}.pkl", "rb") as f:
            data = pickle.load(f)
            
        # 创建实例
        instance = cls(vector_dim=vector_index.d)
        instance.vector_index = vector_index
        instance.structural_index = data["structural_index"]
        instance.id_to_data = data["id_to_data"]
        instance.id_to_vector = data["id_to_vector"]
        instance.is_trained = data["is_trained"]
        
        return instance
