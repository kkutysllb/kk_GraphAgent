"""
编排协调器智能体实现
负责协调多智能体系统的执行流程
"""
from typing import Dict, Any, List, Optional
import logging
import asyncio
import time
import json
import traceback

from .base import Agent, Message
from .intent_analyzer import IntentAnalyzerAgent
from .schema_fetcher import SchemaFetcherAgent
from .entity_recognizer import EntityRecognizerAgent
from .cypher_generator import CypherGeneratorAgent
from .syntax_checker import SyntaxCheckerAgent
from .query_executor import QueryExecutorAgent
from .result_formatter import ResultFormatterAgent
from .graph_embedding import GraphEmbeddingAgent
from .vector_retrieval import VectorRetrievalAgent
from .context_manager import ContextManagerAgent

logger = logging.getLogger("agno.agent.orchestrator")

class OrchestratorAgent(Agent):
    """
    编排协调器智能体
    负责调度和协调各智能体的执行流程
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__("orchestrator")
        self.config = config
        
        # 执行流程中的智能体
        self.agents = {}
        
        # 加载智能体时间戳记录
        self.init_ts = time.time()
        
        # 创建并初始化所有智能体
        self._init_agents()
        
        self.logger.info(f"编排器初始化完成，加载了{len(self.agents)}个智能体，耗时：{time.time() - self.init_ts:.2f}秒")
    
    def _init_agents(self):
        """
        初始化所有智能体
        """
        try:
            # 创建智能体实例
            self.agents["intent_analyzer"] = IntentAnalyzerAgent(self.config)
            self.agents["schema_fetcher"] = SchemaFetcherAgent(self.config)
            self.agents["entity_recognizer"] = EntityRecognizerAgent(self.config)
            self.agents["graph_embedding"] = GraphEmbeddingAgent(self.config)
            self.agents["vector_retrieval"] = VectorRetrievalAgent(self.config)
            self.agents["cypher_generator"] = CypherGeneratorAgent(self.config)
            self.agents["syntax_checker"] = SyntaxCheckerAgent(self.config)
            self.agents["query_executor"] = QueryExecutorAgent(self.config)
            self.agents["result_formatter"] = ResultFormatterAgent(self.config)
            self.agents["context_manager"] = ContextManagerAgent(self.config)
            
            # 根据需要添加其他专业智能体
            if self.config.get("enable_topology_analysis", True):
                from .topology_analyzer import TopologyAnalyzerAgent
                self.agents["topology_analyzer"] = TopologyAnalyzerAgent(self.config)
            
            if self.config.get("enable_performance_analysis", True):
                from .performance_analyzer import PerformanceAnalyzerAgent
                self.agents["performance_analyzer"] = PerformanceAnalyzerAgent(self.config)
                
            if self.config.get("enable_fault_diagnosis", True):
                from .fault_diagnoser import FaultDiagnoserAgent
                self.agents["fault_diagnoser"] = FaultDiagnoserAgent(self.config)
                
        except Exception as e:
            self.logger.error(f"初始化智能体失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"初始化智能体失败: {str(e)}")
    
    async def on_message(self, msg: Message) -> Message:
        """
        处理消息
        实现主要协调逻辑
        """
        start_time = time.time()
        session_id = msg.session_id
        
        self.logger.info(f"开始处理会话 {session_id} 的查询: {msg.data.get('query', '')}")
        
        # 基本查询处理流程
        try:
            # 1. 检索历史上下文 (如果有)
            if "context_manager" in self.agents:
                msg = await self.agents["context_manager"].on_message(msg)
                if msg.has_error():
                    self.logger.warning(f"上下文管理失败: {msg.error}")
            
            # 2. 意图分析
            msg = await self.agents["intent_analyzer"].on_message(msg)
            if msg.has_error():
                return self._format_error_response(msg, "意图分析失败")
            
            intent = msg.data.get("intent", "basic_query")
            self.logger.info(f"查询意图: {intent}")
            
            # 3. 获取图数据库模式
            msg = await self.agents["schema_fetcher"].on_message(msg)
            if msg.has_error():
                return self._format_error_response(msg, "获取数据库模式失败")
            
            # 4. 实体识别
            msg = await self.agents["entity_recognizer"].on_message(msg)
            if msg.has_error():
                self.logger.warning(f"实体识别警告: {msg.error}")
                # 继续处理，实体识别失败不一定影响后续步骤
            
            # 5. 向量检索相似查询 (如果启用)
            if self.config.get("enable_vector_retrieval", True):
                try:
                    # 获取查询嵌入
                    msg = await self.agents["graph_embedding"].on_message(msg)
                    
                    # 检索相似查询
                    if not msg.has_error():
                        msg = await self.agents["vector_retrieval"].on_message(msg)
                except Exception as e:
                    self.logger.warning(f"向量检索失败: {str(e)}")
                    # 继续处理，向量检索失败不阻止主流程
            
            # 6. 根据意图执行不同流程
            if intent == "topology_analysis" and "topology_analyzer" in self.agents:
                # 拓扑分析流程
                msg = await self._execute_topology_analysis(msg)
            elif intent == "performance_analysis" and "performance_analyzer" in self.agents:
                # 性能分析流程
                msg = await self._execute_performance_analysis(msg)
            elif intent == "fault_diagnosis" and "fault_diagnoser" in self.agents:
                # 故障诊断流程
                msg = await self._execute_fault_diagnosis(msg)
            else:
                # 基础查询流程 (默认)
                msg = await self._execute_basic_query(msg)
            
            # 保存查询上下文
            if "context_manager" in self.agents:
                context_msg = await self.agents["context_manager"].on_message(
                    msg.update(action="save_context")
                )
                if context_msg.has_error():
                    self.logger.warning(f"保存上下文失败: {context_msg.error}")
            
            # 记录处理时间
            total_time = time.time() - start_time
            msg = msg.update(processing_time=total_time)
            
            self.logger.info(f"会话 {session_id} 查询处理完成，耗时: {total_time:.2f}秒")
            
            return msg
            
        except Exception as e:
            self.logger.error(f"查询处理过程发生错误: {str(e)}", exc_info=True)
            return msg.update(
                error=f"处理查询时出错: {str(e)}",
                traceback=traceback.format_exc()
            )
    
    async def _execute_basic_query(self, msg: Message) -> Message:
        """
        执行基础查询流程
        """
        # 1. 生成Cypher查询
        msg = await self.agents["cypher_generator"].on_message(msg)
        if msg.has_error():
            return self._format_error_response(msg, "Cypher生成失败")
        
        # 2. 语法检查
        msg = await self.agents["syntax_checker"].on_message(msg)
        if msg.has_error():
            return self._format_error_response(msg, "Cypher语法检查失败")
        
        # 3. 执行查询
        msg = await self.agents["query_executor"].on_message(msg)
        if msg.has_error():
            return self._format_error_response(msg, "执行查询失败")
        
        # 4. 格式化结果
        msg = await self.agents["result_formatter"].on_message(msg)
        if msg.has_error():
            return self._format_error_response(msg, "结果格式化失败")
        
        return msg
    
    async def _execute_topology_analysis(self, msg: Message) -> Message:
        """
        执行拓扑分析流程
        """
        # 调用拓扑分析智能体
        msg = await self.agents["topology_analyzer"].on_message(msg)
        if msg.has_error():
            return self._format_error_response(msg, "拓扑分析失败")
        
        # 根据分析结果执行Cypher查询
        if "cypher_query" in msg.data:
            # 执行生成的查询
            msg = await self.agents["query_executor"].on_message(msg)
            if msg.has_error():
                return self._format_error_response(msg, "执行拓扑分析查询失败")
            
            # 格式化结果
            msg = await self.agents["result_formatter"].on_message(
                msg.update(format_type="topology")
            )
        
        return msg
    
    async def _execute_performance_analysis(self, msg: Message) -> Message:
        """
        执行性能分析流程
        """
        # 调用性能分析智能体
        msg = await self.agents["performance_analyzer"].on_message(msg)
        if msg.has_error():
            return self._format_error_response(msg, "性能分析失败")
        
        # 根据分析结果执行Cypher查询
        if "cypher_query" in msg.data:
            # 执行生成的查询
            msg = await self.agents["query_executor"].on_message(msg)
            if msg.has_error():
                return self._format_error_response(msg, "执行性能分析查询失败")
            
            # 格式化结果
            msg = await self.agents["result_formatter"].on_message(
                msg.update(format_type="performance")
            )
        
        return msg
    
    async def _execute_fault_diagnosis(self, msg: Message) -> Message:
        """
        执行故障诊断流程
        """
        # 调用故障诊断智能体
        msg = await self.agents["fault_diagnoser"].on_message(msg)
        if msg.has_error():
            return self._format_error_response(msg, "故障诊断失败")
        
        # 根据诊断结果执行Cypher查询
        if "cypher_query" in msg.data:
            # 执行生成的查询
            msg = await self.agents["query_executor"].on_message(msg)
            if msg.has_error():
                return self._format_error_response(msg, "执行故障诊断查询失败")
            
            # 格式化结果
            msg = await self.agents["result_formatter"].on_message(
                msg.update(format_type="fault")
            )
        
        return msg
    
    def _format_error_response(self, msg: Message, default_error: str) -> Message:
        """
        格式化错误响应
        """
        error = msg.error or default_error
        return msg.update(
            status="error",
            error=error
        ) 