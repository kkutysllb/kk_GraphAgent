"""
查询执行智能体实现
负责在Neo4j数据库上执行Cypher查询
"""
from typing import Dict, Any, List
import logging
from py2neo import Graph
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from .base import Agent, Message

logger = logging.getLogger("agno.agent.query_executor")

class QueryExecutorAgent(Agent):
    """
    查询执行智能体
    负责在Neo4j数据库上执行Cypher查询
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__("query_executor")
        self.config = config
        self.neo4j_uri = config.get("neo4j_uri", "bolt://localhost:7687")
        self.neo4j_user = config.get("neo4j_user", "neo4j")
        self.neo4j_password = config.get("neo4j_password", "agno123")
        self.max_execution_time = config.get("max_execution_time", 30)  # 最大执行时间（秒）
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    async def on_message(self, msg: Message) -> Message:
        """
        处理消息，执行Cypher查询
        """
        # 获取Cypher查询
        cypher = msg.data.get("cypher")
        if not cypher:
            return msg.update(error="未提供Cypher查询")
        
        # 执行查询
        try:
            self.logger.info(f"执行Cypher查询: {cypher}")
            start_time = time.time()
            
            # 执行查询
            results, execution_time, has_more = await self._execute_query(cypher)
            
            self.logger.info(f"查询执行完成，耗时: {execution_time:.2f}秒，返回{len(results)}条结果")
            
            # 更新消息
            return msg.update(
                results=results,
                execution_time=execution_time,
                has_more_results=has_more
            )
            
        except Exception as e:
            self.logger.error(f"执行查询失败: {str(e)}", exc_info=True)
            return msg.update(error=f"执行查询失败: {str(e)}")
    
    async def _execute_query(self, cypher: str) -> tuple:
        """
        执行Cypher查询
        返回(结果列表, 执行时间, 是否有更多结果)
        """
        # 创建任务
        start_time = time.time()
        
        # 在线程池中执行数据库操作（防止阻塞事件循环）
        try:
            results = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor, self._execute_query_sync, cypher
                ),
                timeout=self.max_execution_time
            )
            
            execution_time = time.time() - start_time
            
            # 检查是否有更多结果（基于结果数量和是否有LIMIT子句）
            has_more = self._check_for_more_results(results, cypher)
            
            return results, execution_time, has_more
            
        except asyncio.TimeoutError:
            self.logger.error(f"查询执行超时，已超过{self.max_execution_time}秒")
            raise Exception(f"查询执行超时，已超过{self.max_execution_time}秒")
    
    def _execute_query_sync(self, cypher: str) -> List[Dict[str, Any]]:
        """
        同步执行Cypher查询
        """
        # 连接到Neo4j
        graph = Graph(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # 执行查询
        cursor = graph.run(cypher)
        
        # 获取结果
        results = cursor.data()
        
        return results
    
    def _check_for_more_results(self, results: List[Dict[str, Any]], cypher: str) -> bool:
        """
        检查是否可能有更多结果（基于结果数量和LIMIT子句）
        """
        # 检查结果是否为空
        if not results:
            return False
        
        # 尝试提取LIMIT值
        limit_value = self._extract_limit_value(cypher)
        
        # 如果找到LIMIT值并且结果数量等于或接近LIMIT值，则可能有更多结果
        if limit_value and len(results) >= int(limit_value) * 0.9:  # 90%以上的LIMIT值
            return True
        
        return False
    
    def _extract_limit_value(self, cypher: str) -> str:
        """
        从Cypher查询中提取LIMIT值
        """
        import re
        
        # 使用正则表达式查找LIMIT子句
        limit_pattern = re.compile(r"LIMIT\s+(\d+)", re.IGNORECASE)
        match = limit_pattern.search(cypher)
        
        if match:
            return match.group(1)  # 返回第一个捕获组（数字）
        
        return None 