"""
模式获取智能体实现
负责从Neo4j数据库获取图谱模式信息
"""
from typing import Dict, Any, List
import logging
from py2neo import Graph
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .base import Agent, Message

logger = logging.getLogger("agno.agent.schema_fetcher")

class SchemaFetcherAgent(Agent):
    """
    模式获取智能体
    负责从Neo4j数据库获取图谱模式信息，包括节点标签、关系类型和属性
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__("schema_fetcher")
        self.config = config
        self.neo4j_uri = config.get("neo4j_uri", "bolt://localhost:7687")
        self.neo4j_user = config.get("neo4j_user", "neo4j")
        self.neo4j_password = config.get("neo4j_password", "agno123")
        self.cached_schema = None
        self.cache_expiry = 0  # 缓存过期时间戳
        self.cache_ttl = config.get("schema_cache_ttl", 3600)  # 缓存有效期（秒）
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    async def on_message(self, msg: Message) -> Message:
        """
        处理消息，获取Neo4j数据库模式
        """
        try:
            # 检查缓存是否有效
            current_time = asyncio.get_event_loop().time()
            if self.cached_schema and current_time < self.cache_expiry:
                self.logger.info("使用缓存的数据库模式")
                return msg.update(schema=self.cached_schema)
            
            # 获取最新模式
            self.logger.info("从Neo4j获取数据库模式")
            schema = await self._fetch_database_schema()
            
            # 更新缓存
            self.cached_schema = schema
            self.cache_expiry = current_time + self.cache_ttl
            
            # 更新消息
            return msg.update(schema=schema)
            
        except Exception as e:
            self.logger.error(f"获取数据库模式失败: {str(e)}", exc_info=True)
            return msg.update(error=f"获取数据库模式失败: {str(e)}")
    
    async def _fetch_database_schema(self) -> Dict[str, Any]:
        """
        从Neo4j数据库获取模式信息
        """
        # 在线程池中执行数据库操作（防止阻塞事件循环）
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self._fetch_schema_sync
        )
    
    def _fetch_schema_sync(self) -> Dict[str, Any]:
        """
        同步获取数据库模式
        """
        # 连接到Neo4j
        graph = Graph(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # 获取节点标签
        node_labels = self._get_node_labels(graph)
        
        # 获取关系类型
        relationship_types = self._get_relationship_types(graph)
        
        # 获取属性
        properties = self._get_properties(graph, node_labels, relationship_types)
        
        # 获取完整的关系模式
        full_relationship_types = self._get_full_relationship_schema(graph)
        
        # 返回完整模式
        return {
            "node_labels": node_labels,
            "relationship_types": full_relationship_types,
            "properties": properties
        }
    
    def _get_node_labels(self, graph: Graph) -> List[str]:
        """
        获取所有节点标签
        """
        labels_query = "CALL db.labels() YIELD label RETURN label ORDER BY label"
        labels_result = graph.run(labels_query).data()
        return [record["label"] for record in labels_result]
    
    def _get_relationship_types(self, graph: Graph) -> List[str]:
        """
        获取所有关系类型
        """
        rel_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
        rel_result = graph.run(rel_query).data()
        return [record["relationshipType"] for record in rel_result]
    
    def _get_properties(self, graph: Graph, node_labels: List[str], rel_types: List[str]) -> Dict[str, List[str]]:
        """
        获取节点和关系的属性
        """
        properties = {}
        
        # 获取节点属性
        for label in node_labels:
            props_query = f"""
            MATCH (n:{label})
            UNWIND keys(n) AS prop
            RETURN DISTINCT prop
            ORDER BY prop
            """
            props_result = graph.run(props_query).data()
            properties[label] = [record["prop"] for record in props_result]
        
        # 获取关系属性
        for rel_type in rel_types:
            props_query = f"""
            MATCH ()-[r:{rel_type}]->()
            UNWIND keys(r) AS prop
            RETURN DISTINCT prop
            ORDER BY prop
            """
            props_result = graph.run(props_query).data()
            properties[rel_type] = [record["prop"] for record in props_result]
        
        return properties
    
    def _get_full_relationship_schema(self, graph: Graph) -> List[Dict[str, str]]:
        """
        获取完整的关系模式信息，包括起始和终止节点类型
        """
        rel_schema_query = """
        CALL db.schema.visualization() YIELD nodes, relationships
        UNWIND relationships AS rel
        RETURN 
            labels(startNode(rel))[0] AS start,
            type(rel) AS type,
            labels(endNode(rel))[0] AS end
        """
        
        result = graph.run(rel_schema_query).data()
        
        rel_schema = []
        for record in result:
            rel_schema.append({
                "start": record["start"],
                "type": record["type"],
                "end": record["end"]
            })
        
        return rel_schema 