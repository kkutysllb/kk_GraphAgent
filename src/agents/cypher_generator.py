"""
Cypher生成智能体实现
负责将自然语言查询转换为Cypher查询语句
"""
from typing import Dict, Any, List, Optional
import logging
import json
import hashlib

from .base import Agent, Message
from src.models.llm_service import LLMService

logger = logging.getLogger("agno.agent.cypher_generator")

class CypherGeneratorAgent(Agent):
    """
    Cypher生成智能体
    负责将自然语言查询转换为Cypher查询语句
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__("cypher_generator")
        self.config = config
        
        # 初始化LLM服务
        self.llm_service = LLMService(
            api_url=config.get("llm_endpoint"),
            default_model=config.get("llm_model"),
            cache_dir=config.get("cache_dir")
        )
    
    async def on_message(self, msg: Message) -> Message:
        """
        处理消息，生成Cypher查询
        """
        # 检查必要输入
        query = msg.data.get("query")
        if not query:
            return msg.update(error="未提供查询文本")
        
        schema = msg.data.get("schema")
        if not schema:
            return msg.update(error="未提供数据模式")
        
        entities = msg.data.get("entities", [])
        intent = msg.data.get("intent", "basic_query")
        
        # 生成Cypher查询
        try:
            cypher_query, explanation = await self._generate_cypher(
                query=query, 
                schema=schema, 
                entities=entities, 
                intent=intent,
                examples=msg.data.get("examples", []),
                similar_queries=msg.data.get("similar_queries", [])
            )
            
            self.logger.info(f"生成的Cypher查询: {cypher_query}")
            
            # 更新消息
            return msg.update(
                cypher_query=cypher_query,
                cypher_explanation=explanation
            )
        except Exception as e:
            self.logger.error(f"Cypher生成失败: {str(e)}", exc_info=True)
            return msg.update(error=f"Cypher生成失败: {str(e)}")
    
    async def _generate_cypher(
        self, 
        query: str, 
        schema: Dict[str, Any], 
        entities: List[Dict[str, Any]],
        intent: str,
        examples: List[Dict[str, Any]] = None,
        similar_queries: List[Dict[str, Any]] = None
    ) -> tuple:
        """
        生成Cypher查询
        返回(Cypher查询, 解释)
        """
        # 准备模式信息
        if isinstance(schema, dict):
            schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        else:
            schema_str = str(schema)
        
        # 准备实体信息
        entities_str = ""
        if entities:
            entities_str = "识别到的实体:\n" + json.dumps(entities, ensure_ascii=False, indent=2)
        
        # 准备示例（如果有）
        examples_str = ""
        if examples and len(examples) > 0:
            examples_text = []
            for i, example in enumerate(examples[:3]):  # 最多使用3个示例
                example_query = example.get("query", "")
                example_cypher = example.get("cypher", "")
                if example_query and example_cypher:
                    examples_text.append(f"示例 {i+1}:\n查询: {example_query}\nCypher: {example_cypher}")
            
            if examples_text:
                examples_str = "\n\n相关示例:\n" + "\n\n".join(examples_text)
        
        # 准备相似查询（如果有）
        similar_str = ""
        if similar_queries and len(similar_queries) > 0:
            similar_text = []
            for i, similar in enumerate(similar_queries[:2]):  # 最多使用2个相似查询
                similar_query = similar.get("query", "")
                similar_cypher = similar.get("cypher", "")
                if similar_query and similar_cypher:
                    similar_text.append(f"相似查询 {i+1}:\n查询: {similar_query}\nCypher: {similar_cypher}")
            
            if similar_text:
                similar_str = "\n\n相似历史查询:\n" + "\n\n".join(similar_text)
        
        # 根据意图调整提示
        intent_str = f"查询意图: {intent}"
        
        # 构建提示
        system_prompt = f"""
        你是一个专业的Neo4j Cypher查询生成专家。
        你的任务是将自然语言查询转换为准确的Cypher查询语句。
        请严格按照提供的模式信息构建查询，确保生成的Cypher语法正确。

        请返回以下JSON格式：
        {{
            "cypher": "生成的Cypher查询语句",
            "explanation": "对生成的Cypher查询的简短解释"
        }}

        不要返回任何其他格式或多余文本。
        """
        
        # 构建用户消息
        user_message = f"""
        请为以下自然语言查询生成Cypher查询语句:

        查询: {query}

        {intent_str}

        图数据库模式:
        {schema_str}

        {entities_str}
        {examples_str}
        {similar_str}
        """
        
        # 生成缓存键
        cache_key = None
        if self.config.get("use_cache", True):
            hash_input = f"{query}|{schema_str}|{entities_str}|{intent}"
            cache_key = f"cypher_{hashlib.md5(hash_input.encode()).hexdigest()}"
        
        # 使用LLM服务生成JSON响应
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # 调用LLM服务
        result = await self.llm_service.generate_json(
            messages=messages,
            temperature=0.2,  # 适当的温度平衡创造性和准确性
            cache_key=cache_key
        )
        
        # 提取结果
        cypher = result.get("cypher", "")
        explanation = result.get("explanation", "")
        
        if not cypher:
            raise Exception("生成的Cypher查询为空")
        
        return cypher, explanation 