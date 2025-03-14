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
from datetime import datetime

from typing import Dict, List, Tuple, Any, Optional
from neo4j import GraphDatabase
from graph_rag.utils.logger import Logger
from graph_rag.utils.config import Config



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
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """初始化Neo4j图数据库管理器
        
        Args:
            uri: Neo4j数据库URI，如果为None则从配置文件读取
            user: Neo4j用户名，如果为None则从配置文件读取
            password: Neo4j密码，如果为None则从配置文件读取
        """
        self.logger = Logger(self.__class__.__name__)
        
        # 获取配置实例
        self.config = Config.get_instance()
        
        # 获取数据库连接参数
        neo4j_config = self.config.get('neo4j', {})
        self.uri = uri or neo4j_config.get('uri', 'bolt://localhost:7687')
        self.user = user or neo4j_config.get('user', 'neo4j')
        self.password = password or neo4j_config.get('password', 'Oms_2600a')
        
        self.logger.info(f"初始化Neo4j图数据库管理器，URI: {self.uri}")
        
        # 初始化驱动
        self._driver = None
        self.connect()
    
    def connect(self):
        """连接到Neo4j数据库"""
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=30
            )
            # 验证连接
            with self._driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                count = result.single()["count"]
                self.logger.info(f"成功连接Neo4j数据库，当前数据库包含 {count} 个节点")
        except Exception as e:
            self.logger.error(f"连接Neo4j数据库失败: {str(e)}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if self._driver:
            self._driver.close()
            self.logger.info("已关闭Neo4j数据库连接")
    
    def import_graph_data(self, graph_data_path: str, clear_existing: bool = False) -> bool:
        """导入图数据到Neo4j
        
        Args:
            graph_data_path: 图数据文件路径
            clear_existing: 是否清除现有数据
            
        Returns:
            是否成功导入
        """
        try:
            self.logger.info(f"正在导入图数据: {graph_data_path}")
            
            # 读取图数据
            with open(graph_data_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])
            
            self.logger.info(f"读取到 {len(nodes)} 个节点和 {len(edges)} 条边")
            
            with self._driver.session() as session:
                # 清除现有数据
                if clear_existing:
                    self.logger.info("清除现有数据")
                    session.run("MATCH (n) DETACH DELETE n")
                
                # 导入节点
                self.logger.info("正在导入节点...")
                node_count = 0
                for node in nodes:
                    node_id = node.get("id")
                    layer_type = node.get("type")  # 层级类型（如DC/TENANT/NE等）
                    attributes = node.get("attributes", {})
                    business_type = attributes.get("business_type")  # 业务类型（BUSINESS/COMPUTE/STORAGE）
                    
                    # 处理动态数据
                    metrics = node.get("metrics", {}).get("metrics_data", {}).get(node_id, [])
                    logs = node.get("logs", {}).get("log_data", {}).get(node_id, [])
                    
                    # 跳过无效节点
                    if not node_id or not layer_type or not business_type:
                        self.logger.warning(f"跳过无效节点: {node}")
                        continue
                    
                    # 构建节点属性，将复杂对象序列化为JSON字符串
                    node_props = {
                        "id": node_id,
                        "name": attributes.get("name"),
                        "layer": attributes.get("layer"),
                        "layer_type": attributes.get("layer_type"),
                        "business_type": attributes.get("business_type"),
                        "capacity": attributes.get("capacity"),
                        "timestamp": attributes.get("timestamp"),
                        "metrics_data": json.dumps(metrics) if metrics else None,
                        "logs_data": json.dumps(logs) if logs else None,
                        "last_update": datetime.now().isoformat()
                    }
                    
                    # 构建Cypher查询，同时使用业务类型和层级类型作为标签
                    query = """
                    MERGE (n:{business_type}:{layer_type} {{id: $props.id}})
                    SET n = $props
                    RETURN n
                    """.format(business_type=business_type, layer_type=layer_type)
                    
                    session.run(query, props=node_props)
                    node_count += 1
                
                self.logger.info(f"成功导入 {node_count} 个节点")
                
                # 导入边
                self.logger.info("正在导入边...")
                edge_count = 0
                skipped_edges = 0
                for edge in edges:
                    source = edge.get("source")
                    target = edge.get("target")
                    edge_type = edge.get("type", "CONNECTS_TO")
                    attributes = edge.get("attributes", {})
                    
                    # 处理边的动态数据
                    dynamics = edge.get("dynamics", {})
                    
                    # 跳过无效边
                    if not source or not target:
                        self.logger.warning(f"跳过无效边 (缺少源/目标): {edge}")
                        skipped_edges += 1
                        continue
                    
                    try:
                        # 构建边属性，将复杂对象序列化为JSON字符串
                        edge_props = {
                            "weight": attributes.get("weight"),
                            "source_type": attributes.get("source_type"),
                            "target_type": attributes.get("target_type"),
                            "timestamp": attributes.get("timestamp"),
                            "dynamics_data": json.dumps(dynamics) if dynamics else None,
                            "last_update": datetime.now().isoformat()
                        }
                        
                        # 构建Cypher查询
                        query = """
                        MATCH (source {{id: $source}})
                        MATCH (target {{id: $target}})
                        MERGE (source)-[r:{edge_type}]->(target)
                        SET r = $props
                        RETURN r
                        """.format(edge_type=edge_type)
                        
                        session.run(query, source=source, target=target, props=edge_props)
                        edge_count += 1
                    except Exception as e:
                        self.logger.warning(f"导入边失败: 源={source}, 目标={target}, 错误={str(e)}")
                        skipped_edges += 1
                
                self.logger.info(f"成功导入 {edge_count} 条边, 跳过 {skipped_edges} 条无效边")
                
                # 创建索引
                self.logger.info("正在创建索引...")
                # 为业务类型创建索引
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:BUSINESS) ON (n.id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:COMPUTE) ON (n.id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:STORAGE) ON (n.id)")
                # 为层级类型创建索引
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:DC) ON (n.id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:TENANT) ON (n.id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:NE) ON (n.id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:VM) ON (n.id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:HOST) ON (n.id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:HA) ON (n.id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:TRU) ON (n.id)")
                
                self.logger.info("图数据导入完成")
                return True
                
        except Exception as e:
            self.logger.error(f"导入图数据失败: {str(e)}")
            return False
        finally:
            self.close()
    
    def execute_cypher_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """执行Cypher查询并返回结果

        Args:
            query: Cypher查询语句
            params: 查询参数

        Returns:
            查询结果列表
        """
        # 清理查询语句
        query = self._clean_query(query)
        
        with self._driver.session() as session:
            try:
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