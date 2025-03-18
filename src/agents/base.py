"""
智能体基类定义
所有智能体都继承自Agent基类
"""
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
import uuid
import time
import logging
import asyncio

logger = logging.getLogger("agno.agent")

@dataclass
class Message:
    """消息数据结构，用于智能体之间的通信"""
    data: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    workflow_results: List[Dict[str, Any]] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    
    def update(self, **kwargs) -> 'Message':
        """
        更新消息，创建一个包含新数据的消息副本
        """
        new_msg = Message(
            data={**self.data},
            meta={**self.meta},
            workflow_results=[*self.workflow_results],
            id=self.id,
            created_at=self.created_at
        )
        
        # 更新数据字段
        for key, value in kwargs.items():
            new_msg.data[key] = value
            
        return new_msg
    
    def record_workflow(self, agent_name: str, result: Dict[str, Any]) -> None:
        """
        记录工作流执行结果
        """
        self.workflow_results.append({
            "agent": agent_name,
            "timestamp": time.time(),
            "result": result
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将消息转换为字典
        """
        return asdict(self)

class Agent:
    """
    智能体基类
    所有特定功能的智能体都应该继承这个类
    """
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"agno.agent.{name}")
    
    async def process(self, msg: Message) -> Message:
        """
        处理消息的主方法
        在执行具体的on_message前后添加日志和错误处理
        """
        self.logger.info(f"处理消息: {msg.id}")
        start_time = time.time()
        
        try:
            result = await self.on_message(msg)
            
            # 记录工作流结果
            result.record_workflow(
                self.name, 
                {"success": True, "time_taken": time.time() - start_time}
            )
            
            self.logger.info(f"完成消息处理: {msg.id}")
            return result
            
        except Exception as e:
            self.logger.error(f"处理消息时出错: {str(e)}", exc_info=True)
            
            # 记录工作流错误
            msg.record_workflow(
                self.name,
                {
                    "success": False,
                    "error": str(e),
                    "time_taken": time.time() - start_time
                }
            )
            
            # 在消息中标记错误
            return msg.update(error=str(e))
    
    async def on_message(self, msg: Message) -> Message:
        """
        具体的消息处理逻辑
        子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现on_message方法")

class MessageBus:
    """
    消息总线
    用于智能体之间的通信和消息路由
    """
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.logger = logging.getLogger("agno.message_bus")
    
    def register(self, agent: Agent) -> None:
        """
        注册智能体到消息总线
        """
        self.agents[agent.name] = agent
        self.logger.info(f"注册智能体: {agent.name}")
    
    async def process(self, agent_name: str, msg: Message) -> Message:
        """
        通过指定的智能体处理消息
        """
        if agent_name not in self.agents:
            self.logger.error(f"未找到智能体: {agent_name}")
            return msg.update(error=f"未找到智能体: {agent_name}")
        
        agent = self.agents[agent_name]
        return await agent.process(msg)
    
    async def broadcast(self, msg: Message, exclude: Optional[List[str]] = None) -> Dict[str, Message]:
        """
        向所有智能体广播消息(除了排除的智能体)
        """
        exclude = exclude or []
        tasks = []
        
        for name, agent in self.agents.items():
            if name not in exclude:
                tasks.append(self.process(name, msg))
        
        results = await asyncio.gather(*tasks)
        return {name: result for name, result in zip(
            [n for n in self.agents.keys() if n not in exclude], 
            results
        )}

def configure_message_bus(config: Dict[str, Any]) -> MessageBus:
    """
    配置消息总线并注册所有智能体
    """
    from .orchestrator import OrchestratorAgent
    from .intent_analyzer import IntentAnalyzerAgent
    from .schema_fetcher import SchemaFetcherAgent
    from .entity_recognizer import EntityRecognizerAgent
    from .graph_embedding import GraphEmbeddingAgent
    from .vector_retrieval import VectorRetrievalAgent
    from .cypher_generator import CypherGeneratorAgent
    from .syntax_checker import SyntaxCheckerAgent
    from .query_executor import QueryExecutorAgent
    from .result_formatter import ResultFormatterAgent
    from .topology_analyzer import TopologyAnalyzerAgent
    from .performance_analyzer import PerformanceAnalyzerAgent
    from .fault_diagnoser import FaultDiagnoserAgent
    from .context_manager import ContextManagerAgent
    
    bus = MessageBus()
    
    # 注册智能体
    bus.register(OrchestratorAgent(config))
    bus.register(IntentAnalyzerAgent(config))
    bus.register(SchemaFetcherAgent(config))
    bus.register(EntityRecognizerAgent(config))
    bus.register(GraphEmbeddingAgent(config.get("embedding_config", {})))
    bus.register(VectorRetrievalAgent(config))
    bus.register(CypherGeneratorAgent(config))
    bus.register(SyntaxCheckerAgent(config))
    bus.register(QueryExecutorAgent(config))
    bus.register(ResultFormatterAgent(config))
    bus.register(TopologyAnalyzerAgent(config))
    bus.register(PerformanceAnalyzerAgent(config))
    bus.register(FaultDiagnoserAgent(config))
    bus.register(ContextManagerAgent(config))
    
    return bus 