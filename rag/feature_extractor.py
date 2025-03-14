#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-14 22:43
# @Desc   : 特征提取器模块，负责从Neo4j中提取节点和关系特征
# --------------------------------------------------------
"""

"""
特征提取器模块，负责从Neo4j中提取节点和关系特征
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from neo4j import GraphDatabase
from tqdm import tqdm
import numpy as np
from utils.logger import Logger

class FeatureExtractor:
    """特征提取器类
    
    从Neo4j图数据库中提取节点和关系的特征，包括：
    1. 节点静态特征（类型、属性等）
    2. 节点动态特征（性能指标、日志等）
    3. 关系静态特征（类型、权重等）
    4. 关系动态特征（状态变化等）
    5. 子图结构特征
    """
    
    def __init__(self, uri: str, user: str, password: str):
        """初始化特征提取器
        
        Args:
            uri: Neo4j数据库URI
            user: 用户名
            password: 密码
        """
        self.logger = Logger(self.__class__.__name__)
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.node_types = self._get_node_types()
        self.relationship_types = self._get_relationship_types()
        
    def close(self):
        """关闭数据库连接"""
        if self._driver:
            self._driver.close()
            
    def _get_node_types(self) -> List[str]:
        """获取所有节点类型"""
        with self._driver.session() as session:
            result = session.run("CALL db.labels()")
            return [record["label"] for record in result]
            
    def _get_relationship_types(self) -> List[str]:
        """获取所有关系类型"""
        with self._driver.session() as session:
            result = session.run("CALL db.relationshipTypes()")
            return [record["relationshipType"] for record in result]
            
    def extract_node_static_features(self, batch_size: int = 100) -> List[Dict]:
        """提取节点静态特征
        
        Args:
            batch_size: 批处理大小
            
        Returns:
            节点特征列表，每个元素是一个字典，包含节点ID、类型、属性等
        """
        features = []
        
        with self._driver.session() as session:
            for node_type in self.node_types:
                query = f"""
                MATCH (n:{node_type})
                RETURN n
                """
                
                result = session.run(query)
                for record in tqdm(result, desc=f"提取 {node_type} 节点特征"):
                    node = record["n"]
                    feature = {
                        "id": node.id,
                        "type": node_type,
                        "attributes": dict(node.items()),
                        "degree": self._get_node_degree(session, node.id)
                    }
                    features.append(feature)
                    
                    if len(features) >= batch_size:
                        yield features
                        features = []
                        
        if features:
            yield features
            
    def _get_node_degree(self, session, node_id: int) -> Dict[str, int]:
        """获取节点的度数信息"""
        query = """
        MATCH (n) WHERE id(n) = $node_id
        RETURN size((n)-->()) as out_degree,
               size((n)<--()) as in_degree
        """
        result = session.run(query, node_id=node_id).single()
        return {
            "in_degree": result["in_degree"],
            "out_degree": result["out_degree"],
            "total_degree": result["in_degree"] + result["out_degree"]
        }
        
    def extract_node_dynamic_features(self, batch_size: int = 100) -> List[Dict]:
        """提取节点动态特征
        
        Args:
            batch_size: 批处理大小
            
        Returns:
            节点动态特征列表
        """
        features = []
        
        with self._driver.session() as session:
            for node_type in self.node_types:
                query = f"""
                MATCH (n:{node_type})
                WHERE n.metrics_data IS NOT NULL OR n.log_data IS NOT NULL
                RETURN n
                """
                
                result = session.run(query)
                for record in tqdm(result, desc=f"提取 {node_type} 节点动态特征"):
                    node = record["n"]
                    feature = {
                        "id": node.id,
                        "type": node_type,
                        "metrics": json.loads(node["metrics_data"]) if node.get("metrics_data") else {},
                        "logs": json.loads(node["log_data"]) if node.get("log_data") else {}
                    }
                    features.append(feature)
                    
                    if len(features) >= batch_size:
                        yield features
                        features = []
                        
        if features:
            yield features
            
    def extract_relationship_features(self, batch_size: int = 100) -> List[Dict]:
        """提取关系特征
        
        Args:
            batch_size: 批处理大小
            
        Returns:
            关系特征列表
        """
        features = []
        
        with self._driver.session() as session:
            for rel_type in self.relationship_types:
                query = f"""
                MATCH ()-[r:`{rel_type}`]->()
                RETURN r, startNode(r) as source, endNode(r) as target
                """
                
                result = session.run(query)
                for record in tqdm(result, desc=f"提取 {rel_type} 关系特征"):
                    rel = record["r"]
                    source = record["source"]
                    target = record["target"]
                    
                    feature = {
                        "id": rel.id,
                        "type": rel_type,
                        "source_id": source.id,
                        "target_id": target.id,
                        "attributes": dict(rel.items()),
                        "dynamics": json.loads(rel["dynamics_data"]) if rel.get("dynamics_data") else {}
                    }
                    features.append(feature)
                    
                    if len(features) >= batch_size:
                        yield features
                        features = []
                        
        if features:
            yield features
            
    def extract_subgraph_features(self, node_id: int, max_depth: int = 2) -> Dict:
        """提取以指定节点为中心的子图特征
        
        Args:
            node_id: 中心节点ID
            max_depth: 最大深度
            
        Returns:
            子图特征字典
        """
        with self._driver.session() as session:
            # 获取子图
            query = """
            MATCH path = (n)-[*1..{max_depth}]-(m)
            WHERE id(n) = $node_id
            RETURN path
            """
            
            result = session.run(query, node_id=node_id, max_depth=max_depth)
            
            # 收集子图中的所有节点和边
            nodes = set()
            relationships = set()
            
            for record in result:
                path = record["path"]
                for node in path.nodes:
                    nodes.add(node.id)
                for rel in path.relationships:
                    relationships.add(rel.id)
                    
            # 构建子图特征
            subgraph = {
                "center_node_id": node_id,
                "depth": max_depth,
                "node_count": len(nodes),
                "relationship_count": len(relationships),
                "nodes": list(nodes),
                "relationships": list(relationships)
            }
            
            return subgraph
            
    def generate_text_description(self, node_id: int) -> str:
        """生成节点的文本描述
        
        Args:
            node_id: 节点ID
            
        Returns:
            文本描述
        """
        with self._driver.session() as session:
            # 获取节点信息
            query = """
            MATCH (n) WHERE id(n) = $node_id
            RETURN n
            """
            
            result = session.run(query, node_id=node_id).single()
            if not result:
                return ""
                
            node = result["n"]
            node_type = list(node.labels)[0]
            
            # 构建基本描述
            description = f"这是一个类型为 {node_type} 的节点，"
            
            # 添加属性描述
            attributes = dict(node.items())
            if "name" in attributes:
                description += f"名称为 {attributes['name']}，"
            if "id" in attributes:
                description += f"ID为 {attributes['id']}，"
                
            # 添加动态特征描述
            if node.get("metrics_data"):
                metrics = json.loads(node["metrics_data"])
                description += "具有以下性能指标："
                for metric, value in metrics.items():
                    description += f"{metric}: {value}，"
                    
            if node.get("log_data"):
                logs = json.loads(node["log_data"])
                description += "最近的日志状态："
                for status, details in logs.items():
                    description += f"{status}: {details}，"
                    
            # 添加关系描述
            query = """
            MATCH (n)-[r]->(m)
            WHERE id(n) = $node_id
            RETURN type(r) as rel_type, m.name as target_name, count(*) as count
            """
            
            result = session.run(query, node_id=node_id)
            for record in result:
                description += f"与 {record['count']} 个 {record['target_name']} 节点存在 {record['rel_type']} 关系，"
                
            return description.rstrip("，") + "。"
