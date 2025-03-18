"""
LLM服务实现
提供大语言模型服务，封装vLLM客户端
"""
import json
import logging
import asyncio
import os
from typing import Dict, Any, List, Optional, Union, Callable

from .vllm_client import VLLMClient

logger = logging.getLogger("agno.models.llm_service")

class LLMService:
    """
    LLM服务
    封装vLLM客户端，提供高级接口
    """
    def __init__(
        self,
        api_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        cache_dir: Optional[str] = None
    ):
        """
        初始化LLM服务
        
        Args:
            api_url: vLLM服务API地址，如果未指定则使用环境变量
            default_model: 默认使用的模型名称，如果未指定则使用环境变量
            timeout: API请求超时时间(秒)
            max_retries: 最大重试次数
            cache_dir: 缓存目录，用于缓存响应
        """
        # 从环境变量中获取配置（如果未指定）
        self.api_url = api_url or os.environ.get("LLM_ENDPOINT", "http://localhost:8000/v1")
        self.default_model = default_model or os.environ.get("LLM_MODEL", "qwen2.5-7b")
        self.cache_dir = cache_dir
        
        # 创建vLLM客户端
        self.client = VLLMClient(
            api_url=self.api_url,
            default_model=self.default_model,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # 初始化缓存目录（如果指定）
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"LLM服务已初始化，使用API: {self.api_url}，默认模型: {self.default_model}")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        生成文本响应
        
        Args:
            messages: 对话消息列表
            model: 模型名称
            temperature: 温度
            max_tokens: 最大令牌数
            stop: 停止词列表
            cache_key: 缓存键，如果指定则使用缓存
            **kwargs: 其他参数
        
        Returns:
            生成的文本
        """
        # 检查缓存
        if cache_key and self.cache_dir:
            cached_result = self._check_cache(cache_key)
            if cached_result:
                logger.info(f"使用缓存结果，缓存键: {cache_key}")
                return cached_result
        
        # 生成响应
        response = await self.client.generate(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs
        )
        
        # 提取生成的文本
        text = ""
        try:
            text = response["choices"][0]["message"]["content"]
            
            # 缓存结果
            if cache_key and self.cache_dir:
                self._cache_result(cache_key, text)
            
            return text
        except (KeyError, IndexError) as e:
            logger.error(f"解析LLM响应失败: {str(e)}")
            logger.debug(f"原始响应: {response}")
            raise Exception("解析LLM响应失败")
    
    async def generate_json(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,  # 低温度，提高确定性
        max_tokens: int = 1024,
        json_schema: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成JSON格式的响应
        
        Args:
            messages: 对话消息列表
            model: 模型名称
            temperature: 温度
            max_tokens: 最大令牌数
            json_schema: JSON模式定义
            cache_key: 缓存键
            **kwargs: 其他参数
        
        Returns:
            生成的JSON对象
        """
        # 检查缓存
        if cache_key and self.cache_dir:
            cached_result = self._check_cache(cache_key)
            if cached_result:
                try:
                    return json.loads(cached_result)
                except json.JSONDecodeError:
                    logger.warning(f"缓存的结果不是有效的JSON，将重新生成，缓存键: {cache_key}")
        
        # 添加JSON相关配置
        response_format = {"type": "json_object"}
        if json_schema:
            response_format["schema"] = json_schema
        
        # 确保系统消息说明需要JSON输出
        system_msg_found = False
        for msg in messages:
            if msg.get("role") == "system":
                system_msg_found = True
                if "JSON" not in msg.get("content", ""):
                    msg["content"] += "\n请以有效的JSON格式返回回答，不要包含任何非JSON内容。"
                break
        
        if not system_msg_found:
            messages.insert(0, {
                "role": "system",
                "content": "你是一个有用的助手，总是以有效的JSON格式返回回答，不要包含任何非JSON内容。"
            })
        
        # 生成响应
        response = await self.client.generate(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs
        )
        
        # 提取生成的文本
        try:
            text = response["choices"][0]["message"]["content"]
            
            # 解析JSON
            try:
                data = json.loads(text)
                
                # 缓存结果
                if cache_key and self.cache_dir:
                    self._cache_result(cache_key, text)
                
                return data
            except json.JSONDecodeError as e:
                logger.error(f"解析JSON失败: {str(e)}")
                logger.debug(f"原始文本: {text}")
                
                # 尝试从文本中提取JSON部分
                json_start = text.find("{")
                json_end = text.rfind("}")
                
                if json_start >= 0 and json_end >= json_start:
                    try:
                        json_text = text[json_start:json_end+1]
                        return json.loads(json_text)
                    except json.JSONDecodeError:
                        pass
                
                raise Exception("LLM未返回有效的JSON")
                
        except (KeyError, IndexError) as e:
            logger.error(f"解析LLM响应失败: {str(e)}")
            logger.debug(f"原始响应: {response}")
            raise Exception("解析LLM响应失败")
    
    async def classify(
        self,
        text: str,
        categories: List[str],
        model: Optional[str] = None,
        temperature: float = 0.1,
        cache_key: Optional[str] = None
    ) -> Dict[str, float]:
        """
        文本分类，返回每个类别的概率
        
        Args:
            text: 要分类的文本
            categories: 类别列表
            model: 模型名称
            temperature: 温度
            cache_key: 缓存键
        
        Returns:
            每个类别的概率字典
        """
        # 构建分类提示
        prompt = f"""对以下文本进行分类，选择最匹配的类别:

文本: "{text}"

类别:
{chr(10).join([f"- {cat}" for cat in categories])}

请以JSON格式返回每个类别的匹配概率(0.0到1.0)。例如:
{{
  "类别1": 0.8,
  "类别2": 0.2,
  ...
}}

只返回JSON，不要包含任何其他文本。所有概率之和应为1.0。
"""
        
        # 调用 generate_json 方法
        messages = [{"role": "user", "content": prompt}]
        result = await self.generate_json(
            messages=messages,
            model=model,
            temperature=temperature,
            cache_key=cache_key
        )
        
        # 规范化结果（确保所有类别都存在且概率之和为1.0）
        normalized = {}
        total = 0.0
        
        # 首先获取所有有效概率值
        for cat in categories:
            prob = float(result.get(cat, 0.0))
            if prob < 0.0:
                prob = 0.0
            normalized[cat] = prob
            total += prob
        
        # 规范化概率值
        if total > 0:
            for cat in normalized:
                normalized[cat] /= total
        else:
            # 如果所有概率都是0，则使用均匀分布
            for cat in normalized:
                normalized[cat] = 1.0 / len(categories)
        
        return normalized
    
    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        cache_key: Optional[str] = None
    ) -> Union[List[float], List[List[float]]]:
        """
        生成文本嵌入向量
        
        Args:
            texts: 单个文本或文本列表
            model: 模型名称
            cache_key: 缓存键
        
        Returns:
            单个向量或向量列表
        """
        # 处理单个文本
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        # 检查缓存
        if cache_key and self.cache_dir:
            cached_result = self._check_cache(cache_key)
            if cached_result:
                try:
                    embeddings = json.loads(cached_result)
                    return embeddings[0] if is_single else embeddings
                except json.JSONDecodeError:
                    logger.warning(f"缓存的结果不是有效的JSON，将重新生成，缓存键: {cache_key}")
        
        # 生成嵌入
        response = await self.client.generate_embeddings(
            texts=texts,
            model=model
        )
        
        try:
            data = response.get("data", [])
            embeddings = [item.get("embedding") for item in data]
            
            # 缓存结果
            if cache_key and self.cache_dir:
                self._cache_result(cache_key, json.dumps(embeddings))
            
            return embeddings[0] if is_single else embeddings
        except (KeyError, IndexError) as e:
            logger.error(f"解析嵌入响应失败: {str(e)}")
            logger.debug(f"原始响应: {response}")
            raise Exception("解析嵌入响应失败")
    
    def _check_cache(self, key: str) -> Optional[str]:
        """
        检查缓存
        
        Args:
            key: 缓存键
        
        Returns:
            缓存的结果或None
        """
        if not self.cache_dir:
            return None
        
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"读取缓存失败: {str(e)}")
        
        return None
    
    def _cache_result(self, key: str, result: str) -> None:
        """
        缓存结果
        
        Args:
            key: 缓存键
            result: 要缓存的结果
        """
        if not self.cache_dir:
            return
        
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(result)
        except Exception as e:
            logger.warning(f"写入缓存失败: {str(e)}")
    
    async def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            服务是否健康
        """
        return await self.client.health_check() 