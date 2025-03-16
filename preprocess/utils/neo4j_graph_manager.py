#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-03 14:30
# @Desc   : Neo4j图数据库管理模块
# --------------------------------------------------------
"""

import json
from typing import Dict, List, Tuple, Any, Optional
from neo4j import GraphDatabase
from utils.logger import Logger
from tqdm import tqdm
import threading

# 添加全局锁，用于控制日志输出
_log_lock = threading.Lock()
_connection_logged = False

class Neo4jGraphManager:
    """Neo4j图数据库管理器
    
    用于将JSON格式的图数据导入Neo4j图数据库，并提供Cypher查询接口。
    支持与自然语言查询处理器集成，实现更复杂的跨层端到端拓扑查询。
    
    主要功能：
    1. 连接Neo4j数据库
    2. 导入图数据
    3. 构建索引
    4. 执行Cypher查询
    5. 查询结果处理
    """
    
    def __init__(self, uri: str, user: str, password: str, batch_size: int = 100):
        """初始化Neo4j图数据库管理器
        
        Args:
            uri: Neo4j数据库URI
            user: Neo4j用户名
            password: Neo4j密码
            batch_size: 批量导入的大小
        """
        self.logger = Logger(self.__class__.__name__)
        self.uri = uri
        self.user = user
        self.password = password
        self.batch_size = batch_size
        self._driver = None
        self.connect()
    
    def connect(self):
        """连接到Neo4j数据库"""
        global _connection_logged
        
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            with self._driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                count = result.single()["count"]
                
                # 使用线程锁控制日志输出，避免多线程重复打印
                with _log_lock:
                    if not _connection_logged:
                        self.logger.info(f"成功连接Neo4j数据库，当前数据库包含 {count} 个节点")
                        _connection_logged = True
        except Exception as e:
            self.logger.error(f"连接Neo4j数据库失败: {str(e)}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if self._driver:
            self._driver.close()
            self.logger.info("已关闭Neo4j数据库连接")
    
    def _create_indexes(self, session):
        """创建所需的索引"""
        self.logger.info("创建索引...")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:BUSINESS) ON (n.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:COMPUTE) ON (n.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:STORAGE) ON (n.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:DC) ON (n.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:TENANT) ON (n.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:NE) ON (n.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:VM) ON (n.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:HOST) ON (n.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:HA) ON (n.id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (n:TRU) ON (n.id)")
    
    def _execute_node_batch(self, session, batch):
        """执行节点批量导入"""
        try:
            for node in batch:
                node_type = node["type"]
                node_data = {
                    "id": node["id"],
                    "type": node["type"],
                    "attributes": node.get("attributes", {}),
                    "metrics_data": json.dumps(node.get("metrics", {})) if node.get("metrics") else None,
                    "log_data": json.dumps(node.get("logs", {})) if node.get("logs") else None
                }
                
                query = f"""
                MERGE (n:{node_type} {{id: $id}})
                SET n.type = $type
                SET n += $attributes
                SET n.metrics_data = $metrics_data
                SET n.log_data = $log_data
                """
                
                session.run(query, node_data)
        except Exception as e:
            self.logger.error(f"批量导入节点失败: {str(e)}")
            if batch:
                self.logger.error(f"示例节点数据: {json.dumps(batch[0], indent=2)}")
            raise

    def _batch_import_nodes(self, session, nodes):
        """批量导入节点"""
        node_count = 0
        batch = []
        
        for node in tqdm(nodes, desc="导入节点"):
            if not isinstance(node, dict) or "id" not in node or "type" not in node:
                self.logger.warning(f"跳过无效节点: {node}")
                continue
            batch.append(node)
            
            if len(batch) >= self.batch_size:
                self._execute_node_batch(session, batch)
                node_count += len(batch)
                batch = []
        
        if batch:
            self._execute_node_batch(session, batch)
            node_count += len(batch)
        
        return node_count
    
    def _execute_edge_batch(self, session, batch):
        """执行边批量导入"""
        try:
            for edge in batch:
                edge_type = edge["type"]
                edge_data = {
                    "source": edge["source"],
                    "target": edge["target"],
                    "attributes": edge.get("attributes", {}),
                    "dynamics_data": json.dumps(edge.get("dynamics", {})) if edge.get("dynamics") else None
                }
                
                query = f"""
                MATCH (source {{id: $source}})
                MATCH (target {{id: $target}})
                MERGE (source)-[r:`{edge_type}`]->(target)
                SET r = $attributes
                SET r.dynamics_data = $dynamics_data
                """
                
                session.run(query, edge_data)
        except Exception as e:
            self.logger.error(f"批量导入边失败: {str(e)}")
            if batch:
                self.logger.error(f"示例边数据: {json.dumps(batch[0], indent=2)}")
            raise

    def _batch_import_edges(self, session, edges):
        """批量导入边"""
        edge_count = 0
        batch = []
        
        for edge in tqdm(edges, desc="导入边"):
            if not isinstance(edge, dict) or "source" not in edge or "target" not in edge or "type" not in edge:
                self.logger.warning(f"跳过无效边: {edge}")
                continue
            batch.append(edge)
            
            if len(batch) >= self.batch_size:
                self._execute_edge_batch(session, batch)
                edge_count += len(batch)
                batch = []
        
        if batch:
            self._execute_edge_batch(session, batch)
            edge_count += len(batch)
        
        return edge_count
    
    def import_graph_data(self, graph_data_path: str, clear_existing: bool = False) -> bool:
        """导入图数据到Neo4j"""
        try:
            # 读取图数据
            with open(graph_data_path, 'r') as f:
                graph_data = json.load(f)
            
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])
            
            self.logger.info(f"从 {graph_data_path} 读取图数据")
            self.logger.info(f"节点数量: {len(nodes)}")
            self.logger.info(f"边数量: {len(edges)}")
            
            with self._driver.session() as session:
                # 清除现有数据
                if clear_existing:
                    self.logger.info("清除现有数据...")
                    session.run("MATCH (n) DETACH DELETE n")
                
                # 创建索引
                self._create_indexes(session)
                
                # 导入节点
                node_count = self._batch_import_nodes(session, nodes)
                self.logger.info(f"成功导入 {node_count} 个节点")
                
                # 导入边
                edge_count = self._batch_import_edges(session, edges)
                self.logger.info(f"成功导入 {edge_count} 条边")
                
                # 验证导入结果
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                actual_node_count = result.single()["node_count"]
                
                result = session.run("MATCH ()-[r]->() RETURN count(r) as edge_count")
                actual_edge_count = result.single()["edge_count"]
                
                self.logger.info(f"验证结果:")
                self.logger.info(f"实际节点数量: {actual_node_count}")
                self.logger.info(f"实际边数量: {actual_edge_count}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"导入图数据失败: {str(e)}")
            return False
    
    def execute_cypher_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """执行Cypher查询并返回结果

        Args:
            query: Cypher查询语句
            params: 查询参数

        Returns:
            查询结果列表
        """
        try:
            with self._driver.session() as session:
                result = session.run(query, params or {})
                return [dict(record) for record in result]
        except Exception as e:
            self.logger.error(f"执行Cypher查询失败: {str(e)}")
            raise

    def _clean_query(self, query: str) -> str:
        """清理查询语句，移除不必要的格式和确保语法正确

        Args:
            query: 原始查询语句

        Returns:
            清理后的查询语句
        """
        # 移除Markdown代码块标记
        query = query.replace("```cypher", "").replace("```", "")
        
        # 移除开头和结尾的空白
        query = query.strip()
        
        # 确保查询以分号结尾
        if not query.endswith(";"):
            query += ";"
            
        # 移除多余的空行，保持基本格式
        lines = [line.strip() for line in query.splitlines() if line.strip()]
        query = " ".join(lines)
        
        # 记录清理后的查询语句（用于调试）
        self.logger.debug(f"清理后的查询语句: {query}")
        
        return query
    
    def build_cypher_query_from_template(self, template: Dict, entities: Dict) -> Tuple[str, Dict]:
        """根据模板构建Cypher查询
        
        Args:
            template: 查询模板
            entities: 实体字典
            
        Returns:
            Cypher查询语句和参数字典
        """
        try:
            template_type = template.get("template_type")
            path_template = template.get("path_template")
            constraints = template.get("constraints", {})
            
            self.logger.info(f"根据模板构建Cypher查询，类型: {template_type}，路径模板: {path_template}")
            
            # 解析路径模板
            node_types = []
            for segment in path_template.split('-'):
                if segment.startswith('[') and segment.endswith(']'):
                    # 处理可选路径段
                    optional_segment = segment[1:-1]
                    sub_types = optional_segment.split('-')
                    node_types.extend(sub_types)
                else:
                    node_types.append(segment)
            
            # 构建Cypher查询
            cypher_query = ""
            params = {}
            
            # 根据模板类型构建不同的查询
            if template_type == "vm_distribution":
                # 虚拟机分布查询
                vm_name = entities.get("vm_name")
                cypher_query = """
                MATCH (vm:COMPUTE {name: $vm_name})-[:CONNECTS_TO*]->(host:COMPUTE)
                WHERE host.type = 'HOST'
                RETURN vm, host
                """
                params = {"vm_name": vm_name}
                
            elif template_type == "tenant_resources":
                # 租户资源查询
                tenant_name = entities.get("tenant_name")
                cypher_query = """
                MATCH (tenant:BUSINESS {name: $tenant_name})-[:CONNECTS_TO*]->(resource)
                WHERE resource.type IN ['VM', 'HOST', 'HA', 'TRU']
                RETURN tenant, resource
                """
                params = {"tenant_name": tenant_name}
                
            elif template_type == "ne_distribution":
                # 网元分布查询
                ne_name = entities.get("ne_name")
                cypher_query = """
                MATCH (ne:BUSINESS {name: $ne_name})-[:CONNECTS_TO*]->(resource)
                WHERE resource.type IN ['VM', 'HOST']
                RETURN ne, resource
                """
                params = {"ne_name": ne_name}
                
            else:
                # 通用查询构建
                # 构建路径匹配部分
                path_parts = []
                for i in range(len(node_types) - 1):
                    source_type = node_types[i]
                    target_type = node_types[i + 1]
                    path_parts.append(f"({source_type.lower()}:{source_type})-[:CONNECTS_TO]->({target_type.lower()}:{target_type})")
                
                path_match = " MATCH " + ", ".join(path_parts)
                
                # 构建WHERE条件
                where_conditions = []
                for entity_name, entity_value in entities.items():
                    if "_name" in entity_name:
                        node_type = entity_name.split("_")[0].upper()
                        where_conditions.append(f"{node_type.lower()}.name = ${entity_name}")
                        params[entity_name] = entity_value
                
                where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
                
                # 构建RETURN语句
                return_nodes = [node_type.lower() for node_type in node_types]
                return_clause = " RETURN " + ", ".join(return_nodes)
                
                cypher_query = path_match + where_clause + return_clause
            
            self.logger.info(f"构建的Cypher查询: {cypher_query}")
            return cypher_query, params
            
        except Exception as e:
            self.logger.error(f"构建Cypher查询失败: {str(e)}")
            raise
    
    def query_with_template(self, template: Dict, entities: Dict) -> List[Dict]:
        """使用模板执行查询
        
        Args:
            template: 查询模板
            entities: 实体字典
            
        Returns:
            查询结果列表
        """
        try:
            # 构建Cypher查询
            cypher_query, params = self.build_cypher_query_from_template(template, entities)
            
            # 执行查询
            results = self.execute_cypher_query(cypher_query, params)
            
            # 处理结果
            processed_results = self._process_query_results(results, template)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"使用模板执行查询失败: {str(e)}")
            raise
    
    def _process_query_results(self, results: List[Dict], template: Dict) -> List[Dict]:
        """处理查询结果
        
        Args:
            results: 查询结果列表
            template: 查询模板
            
        Returns:
            处理后的结果列表
        """
        try:
            processed_results = []
            template_type = template.get("template_type")
            
            for record in results:
                processed_record = {}
                
                # 根据不同的模板类型处理结果
                if template_type == "vm_distribution":
                    # 虚拟机分布查询结果处理
                    if "vm" in record and "host" in record:
                        vm_data = record["vm"]
                        host_data = record["host"]
                        processed_record = {
                            "vm_id": vm_data.get("id"),
                            "vm_name": vm_data.get("name"),
                            "host_id": host_data.get("id"),
                            "host_name": host_data.get("name"),
                            "relation": "部署在"
                        }
                
                elif template_type == "tenant_resources":
                    # 租户资源查询结果处理
                    if "tenant" in record and "resource" in record:
                        tenant_data = record["tenant"]
                        resource_data = record["resource"]
                        processed_record = {
                            "tenant_id": tenant_data.get("id"),
                            "tenant_name": tenant_data.get("name"),
                            "resource_id": resource_data.get("id"),
                            "resource_name": resource_data.get("name"),
                            "resource_type": resource_data.get("type"),
                            "relation": "拥有"
                        }
                
                elif template_type == "ne_distribution":
                    # 网元分布查询结果处理
                    if "ne" in record and "resource" in record:
                        ne_data = record["ne"]
                        resource_data = record["resource"]
                        processed_record = {
                            "ne_id": ne_data.get("id"),
                            "ne_name": ne_data.get("name"),
                            "resource_id": resource_data.get("id"),
                            "resource_name": resource_data.get("name"),
                            "resource_type": resource_data.get("type"),
                            "relation": "使用"
                        }
                
                else:
                    # 通用结果处理
                    processed_record = record
                
                processed_results.append(processed_record)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"处理查询结果失败: {str(e)}")
            return results  # 如果处理失败，返回原始结果
    
    def get_schema(self) -> Dict[str, Any]:
        """获取数据库schema信息
        
        Returns:
            包含节点类型、关系类型和属性的字典
        """
        # 获取节点类型和属性
        node_query = """
        CALL db.schema.nodeTypeProperties()
        YIELD nodeType, propertyName, propertyTypes
        RETURN nodeType, propertyName, propertyTypes
        """
        
        # 获取关系类型和属性
        rel_query = """
        MATCH (a)-[r]->(b)
        WITH type(r) as rel_type, 
             labels(a) as source_labels,
             labels(b) as target_labels,
             count(*) as count,
             keys(r) as properties
        RETURN DISTINCT 
            rel_type,
            source_labels,
            target_labels,
            properties,
            count
        ORDER BY count DESC
        """
        
        try:
            # 执行查询
            node_results = self.execute_cypher_query(node_query)
            rel_results = self.execute_cypher_query(rel_query)
            
            # 处理结果
            schema = {
                'nodes': {},
                'relationships': {}
            }
            
            # 处理节点信息
            for record in node_results:
                node_type = record['nodeType']
                if node_type not in schema['nodes']:
                    schema['nodes'][node_type] = {
                        'properties': {}
                    }
                schema['nodes'][node_type]['properties'][record['propertyName']] = record['propertyTypes']
                
            # 处理关系信息
            for record in rel_results:
                rel_type = record['rel_type']
                if rel_type not in schema['relationships']:
                    schema['relationships'][rel_type] = {
                        'source_labels': record['source_labels'],
                        'target_labels': record['target_labels'],
                        'properties': record['properties'],
                        'count': record['count']
                    }
                    
            return schema
            
        except Exception as e:
            self.logger.error(f"获取schema失败: {str(e)}")
            raise