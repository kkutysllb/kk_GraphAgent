#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询处理器模块，负责处理自然语言查询并转换为Cypher查询
"""

import torch
from typing import Dict, List, Any, Optional, Tuple
import re
from utils.logger import Logger
from .encoder import DualEncoder
from .indexer import HybridIndex

class QueryProcessor:
    """查询处理器类
    
    负责：
    1. 解析自然语言查询
    2. 提取查询意图和实体
    3. 路由到合适的查询模板
    4. 生成Cypher查询
    """
    
    def __init__(
        self,
        model: DualEncoder,
        index: HybridIndex,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """初始化查询处理器
        
        Args:
            model: 双通道编码器模型
            index: 混合索引
            device: 运行设备
        """
        self.logger = Logger(self.__class__.__name__)
        self.model = model.to(device)
        self.model.eval()
        self.index = index
        self.device = device
        
        # 查询模板
        self.query_templates = {
            "node_info": """
            MATCH (n:{node_type} {id: $id})
            RETURN n
            """,
            
            "node_neighbors": """
            MATCH (n:{node_type} {id: $id})-[r]->(m)
            RETURN type(r) as relationship_type,
                   m.id as neighbor_id,
                   m.type as neighbor_type,
                   m.name as neighbor_name
            """,
            
            "path_between": """
            MATCH path = shortestPath(
                (source:{source_type} {id: $source_id})-[*..{max_hops}]->(target:{target_type} {id: $target_id})
            )
            RETURN path
            """,
            
            "subgraph": """
            MATCH path = (n:{node_type} {id: $id})-[*1..{depth}]-(m)
            RETURN path
            """
        }
        
    def process_query(
        self,
        query: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """处理自然语言查询
        
        Args:
            query: 自然语言查询
            max_results: 最大返回结果数
            
        Returns:
            包含查询结果的字典
        """
        # 提取查询意图和实体
        intent, entities = self._extract_intent_entities(query)
        
        # 获取查询向量
        query_vector = self._encode_query(query)
        
        # 根据意图路由到不同的处理逻辑
        if intent == "node_info":
            return self._process_node_info(entities, query_vector, max_results)
        elif intent == "node_neighbors":
            return self._process_node_neighbors(entities, query_vector, max_results)
        elif intent == "path_between":
            return self._process_path_between(entities, query_vector, max_results)
        elif intent == "subgraph":
            return self._process_subgraph(entities, query_vector, max_results)
        else:
            return self._process_semantic_search(query_vector, max_results)
            
    def _extract_intent_entities(
        self,
        query: str
    ) -> Tuple[str, Dict[str, Any]]:
        """提取查询意图和实体
        
        Args:
            query: 自然语言查询
            
        Returns:
            意图和实体字典的元组
        """
        # 意图识别模式
        intent_patterns = {
            "node_info": r"查看|显示|获取.*节点信息",
            "node_neighbors": r"查找.*邻居|相邻|关联",
            "path_between": r"路径|连接|之间",
            "subgraph": r"子图|局部|范围"
        }
        
        # 实体识别模式
        entity_patterns = {
            "node_type": r"(VM|HOST|DC|TENANT|NE|HA|TRU)",
            "node_id": r"ID[:\s]+(\w+)",
            "max_hops": r"(\d+)跳",
            "depth": r"深度[:\s]+(\d+)"
        }
        
        # 识别意图
        intent = "semantic_search"  # 默认意图
        for intent_name, pattern in intent_patterns.items():
            if re.search(pattern, query):
                intent = intent_name
                break
                
        # 提取实体
        entities = {}
        for entity_name, pattern in entity_patterns.items():
            match = re.search(pattern, query)
            if match:
                entities[entity_name] = match.group(1)
                
        return intent, entities
        
    def _encode_query(self, query: str) -> np.ndarray:
        """编码查询文本
        
        Args:
            query: 查询文本
            
        Returns:
            查询向量
        """
        with torch.no_grad():
            text_embeddings = self.model.encode_text([query])
            return text_embeddings.cpu().numpy()
            
    def _process_node_info(
        self,
        entities: Dict[str, Any],
        query_vector: np.ndarray,
        max_results: int
    ) -> Dict[str, Any]:
        """处理节点信息查询
        
        Args:
            entities: 实体字典
            query_vector: 查询向量
            max_results: 最大结果数
            
        Returns:
            查询结果
        """
        node_type = entities.get("node_type")
        node_id = entities.get("node_id")
        
        # 构建Cypher查询
        if node_id:
            cypher = self.query_templates["node_info"].format(
                node_type=node_type or "BUSINESS"
            )
            params = {"id": node_id}
        else:
            # 使用向量搜索找到相关节点
            results = self.index.search(
                query_vector=query_vector,
                type_label=node_type,
                k=max_results
            )
            
            if not results:
                return {"message": "未找到相关节点"}
                
            node_ids = [item_id for item_id, _ in results]
            cypher = f"""
            MATCH (n)
            WHERE n.id IN $node_ids
            RETURN n
            """
            params = {"node_ids": node_ids}
            
        return {
            "cypher": cypher,
            "params": params
        }
        
    def _process_node_neighbors(
        self,
        entities: Dict[str, Any],
        query_vector: np.ndarray,
        max_results: int
    ) -> Dict[str, Any]:
        """处理节点邻居查询
        
        Args:
            entities: 实体字典
            query_vector: 查询向量
            max_results: 最大结果数
            
        Returns:
            查询结果
        """
        node_type = entities.get("node_type")
        node_id = entities.get("node_id")
        
        if node_id:
            cypher = self.query_templates["node_neighbors"].format(
                node_type=node_type or "BUSINESS"
            )
            params = {"id": node_id}
        else:
            # 使用向量搜索找到相关节点
            results = self.index.search(
                query_vector=query_vector,
                type_label=node_type,
                k=1
            )
            
            if not results:
                return {"message": "未找到相关节点"}
                
            node_id = results[0][0]
            cypher = self.query_templates["node_neighbors"].format(
                node_type=node_type or "BUSINESS"
            )
            params = {"id": node_id}
            
        return {
            "cypher": cypher,
            "params": params
        }
        
    def _process_path_between(
        self,
        entities: Dict[str, Any],
        query_vector: np.ndarray,
        max_results: int
    ) -> Dict[str, Any]:
        """处理路径查询
        
        Args:
            entities: 实体字典
            query_vector: 查询向量
            max_results: 最大结果数
            
        Returns:
            查询结果
        """
        source_type = entities.get("source_type", "BUSINESS")
        target_type = entities.get("target_type", "BUSINESS")
        max_hops = int(entities.get("max_hops", 3))
        
        # 使用向量搜索找到相关节点
        results = self.index.search(
            query_vector=query_vector,
            type_label=source_type,
            k=2
        )
        
        if len(results) < 2:
            return {"message": "未找到足够的节点"}
            
        source_id = results[0][0]
        target_id = results[1][0]
        
        cypher = self.query_templates["path_between"].format(
            source_type=source_type,
            target_type=target_type,
            max_hops=max_hops
        )
        
        params = {
            "source_id": source_id,
            "target_id": target_id
        }
        
        return {
            "cypher": cypher,
            "params": params
        }
        
    def _process_subgraph(
        self,
        entities: Dict[str, Any],
        query_vector: np.ndarray,
        max_results: int
    ) -> Dict[str, Any]:
        """处理子图查询
        
        Args:
            entities: 实体字典
            query_vector: 查询向量
            max_results: 最大结果数
            
        Returns:
            查询结果
        """
        node_type = entities.get("node_type")
        node_id = entities.get("node_id")
        depth = int(entities.get("depth", 2))
        
        if node_id:
            cypher = self.query_templates["subgraph"].format(
                node_type=node_type or "BUSINESS",
                depth=depth
            )
            params = {"id": node_id}
        else:
            # 使用向量搜索找到相关节点
            results = self.index.search(
                query_vector=query_vector,
                type_label=node_type,
                k=1
            )
            
            if not results:
                return {"message": "未找到相关节点"}
                
            node_id = results[0][0]
            cypher = self.query_templates["subgraph"].format(
                node_type=node_type or "BUSINESS",
                depth=depth
            )
            params = {"id": node_id}
            
        return {
            "cypher": cypher,
            "params": params
        }
        
    def _process_semantic_search(
        self,
        query_vector: np.ndarray,
        max_results: int
    ) -> Dict[str, Any]:
        """处理语义搜索查询
        
        Args:
            query_vector: 查询向量
            max_results: 最大结果数
            
        Returns:
            查询结果
        """
        # 使用向量搜索找到相关节点
        results = self.index.search(
            query_vector=query_vector,
            k=max_results
        )
        
        if not results:
            return {"message": "未找到相关结果"}
            
        node_ids = [item_id for item_id, _ in results]
        
        cypher = """
        MATCH (n)
        WHERE n.id IN $node_ids
        RETURN n
        """
        
        params = {"node_ids": node_ids}
        
        return {
            "cypher": cypher,
            "params": params
        }
