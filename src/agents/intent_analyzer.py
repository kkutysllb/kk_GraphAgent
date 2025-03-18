"""
意图分析智能体实现
负责识别用户查询的意图
"""
from typing import Dict, Any, List
import logging
import json
import hashlib

from .base import Agent, Message
from src.models.llm_service import LLMService

logger = logging.getLogger("agno.agent.intent_analyzer")

class IntentAnalyzerAgent(Agent):
    """
    意图分析智能体
    负责分析用户输入查询的意图
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__("intent_analyzer")
        self.config = config
        
        # 初始化LLM服务
        self.llm_service = LLMService(
            api_url=config.get("llm_endpoint"),
            default_model=config.get("llm_model"),
            cache_dir=config.get("cache_dir")
        )
        
        # 意图分类规则
        self.intent_definitions = {
            "basic_query": "基础查询：直接查询图数据库中的实体和关系",
            "topology_analysis": "拓扑分析：分析网络拓扑结构，查找关键节点、路径或瓶颈",
            "performance_analysis": "性能分析：分析系统性能指标，查找性能瓶颈或异常",
            "fault_diagnosis": "故障诊断：诊断系统故障，查找根因和影响范围"
        }
    
    async def on_message(self, msg: Message) -> Message:
        """
        处理消息，分析用户查询的意图
        """
        # 获取用户查询
        query = msg.data.get("query")
        if not query:
            return msg.update(error="未提供查询文本")
        
        # 提取上下文（如果有）
        context = msg.data.get("context", {})
        
        # 确定查询意图
        try:
            intent, confidence = await self._classify_intent(query, context)
            
            self.logger.info(f"查询意图: {intent}, 置信度: {confidence}")
            
            # 更新消息
            return msg.update(
                intent=intent,
                intent_confidence=confidence
            )
        except Exception as e:
            self.logger.error(f"意图分析失败: {str(e)}", exc_info=True)
            return msg.update(error=f"意图分析失败: {str(e)}")
    
    async def _classify_intent(self, query: str, context: Dict[str, Any] = None) -> tuple:
        """
        使用LLM对查询进行意图分类
        返回(意图, 置信度)
        """
        # 构建分类提示
        intent_definitions = "\n".join([
            f"- {intent}: {definition}" 
            for intent, definition in self.intent_definitions.items()
        ])
        
        # 构建提示
        system_prompt = f"""
        你是一个专业的5G核心网资源图谱查询意图分析器。你的任务是分析用户查询，并将其分类为以下意图类别之一：
        
        {intent_definitions}
        
        请只返回以下格式的JSON:
        {{
            "intent": "意图类别名称",
            "confidence": 0.0到1.0之间的数字,
            "explanation": "简短解释"
        }}
        
        不要返回任何其他格式或多余文本。
        """
        
        # 准备上下文（如果有）
        context_str = ""
        if context:
            context_str = f"\n\n相关上下文信息:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
        
        # 构建用户消息
        user_message = f"用户查询: {query}{context_str}"
        
        # 生成缓存键
        cache_key = None
        if self.config.get("use_cache", True):
            hash_input = f"{system_prompt}|{user_message}"
            cache_key = f"intent_{hashlib.md5(hash_input.encode()).hexdigest()}"
        
        # 使用LLM服务生成JSON响应
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            # 调用LLM服务
            result = await self.llm_service.generate_json(
                messages=messages,
                temperature=0.1,  # 低温度以提高确定性
                cache_key=cache_key
            )
            
            # 提取结果
            intent = result.get("intent")
            confidence = result.get("confidence", 0.0)
            
            # 验证意图是否有效
            if intent not in self.intent_definitions:
                self.logger.warning(f"无效的意图: {intent}，回退到basic_query")
                intent = "basic_query"
                confidence = 0.5
            
            return intent, confidence
            
        except Exception as e:
            self.logger.error(f"调用LLM服务失败: {str(e)}", exc_info=True)
            # 回退到默认意图
            return "basic_query", 0.5 