"""
vLLM客户端实现
负责与vLLM服务通信，并处理模型推理请求
"""
import json
import logging
import httpx
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger("agno.models.vllm_client")

class VLLMClient:
    """
    vLLM客户端
    负责与vLLM服务进行通信，发送推理请求并处理响应
    """
    def __init__(
        self, 
        api_url: str = "http://localhost:8000/v1",
        default_model: str = "qwen2.5-7b",
        timeout: float = 60.0,
        max_retries: int = 3
    ):
        """
        初始化vLLM客户端
        
        Args:
            api_url: vLLM服务API地址
            default_model: 默认使用的模型名称
            timeout: API请求超时时间(秒)
            max_retries: 最大重试次数
        """
        self.api_url = api_url
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 自适应URL格式
        if not self.api_url.endswith("/"):
            self.api_url += "/"
            
        # 确保URL格式正确
        if not self.api_url.endswith("v1/"):
            self.api_url = self.api_url.rstrip("/") + "/v1/"
        
        # 检查API是否可用
        self._check_api_availability()
    
    def _check_api_availability(self) -> None:
        """
        检查vLLM API是否可用
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.api_url}models")
                if response.status_code == 200:
                    logger.info(f"vLLM API可用: {self.api_url}")
                    models = response.json().get("data", [])
                    if models:
                        model_ids = [model.get("id") for model in models]
                        logger.info(f"可用模型: {', '.join(model_ids)}")
                else:
                    logger.warning(f"vLLM API返回非200状态码: {response.status_code}")
        except Exception as e:
            logger.warning(f"检查vLLM API可用性时出错: {str(e)}")
            logger.info("启动时vLLM服务可能尚未准备好，将在运行时重试")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用vLLM进行文本生成
        
        Args:
            messages: 对话消息列表，格式为[{"role": "user", "content": "消息内容"}, ...]
            model: 模型名称，如果未指定则使用默认模型
            temperature: 采样温度，值越低生成的文本越确定
            top_p: 核采样概率，控制生成文本的多样性
            max_tokens: 生成的最大token数
            stop: 停止生成的字符串列表
            stream: 是否流式返回生成结果
            **kwargs: 其他参数，将直接传递给API
        
        Returns:
            生成结果字典
        """
        # 使用默认模型（如果未指定）
        model = model or self.default_model
        
        # 准备请求数据
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # 添加可选参数
        if stop:
            request_data["stop"] = stop
        
        # 添加其他参数
        for key, value in kwargs.items():
            request_data[key] = value
        
        # 发送请求
        endpoint = f"{self.api_url}chat/completions"
        headers = {"Content-Type": "application/json"}
        
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        endpoint,
                        json=request_data,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        error_msg = f"vLLM API错误: {response.status_code} - {response.text}"
                        logger.error(error_msg)
                        retry_count += 1
                        if retry_count >= self.max_retries:
                            raise Exception(error_msg)
                            
            except httpx.TimeoutException:
                logger.warning(f"vLLM API请求超时 (重试 {retry_count+1}/{self.max_retries})")
                retry_count += 1
                if retry_count >= self.max_retries:
                    raise Exception(f"vLLM API请求在{self.max_retries}次重试后仍然超时")
            
            except Exception as e:
                logger.error(f"vLLM API请求失败: {str(e)}")
                retry_count += 1
                if retry_count >= self.max_retries:
                    raise
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成文本嵌入向量
        
        Args:
            texts: 要生成嵌入的文本列表
            model: 嵌入模型名称，如果未指定则使用环境变量中的嵌入模型
        
        Returns:
            包含嵌入向量的字典
        """
        # 使用默认模型（如果未指定）
        model = model or self.default_model
        
        # 准备请求数据
        request_data = {
            "model": model,
            "input": texts
        }
        
        # 发送请求
        endpoint = f"{self.api_url}embeddings"
        headers = {"Content-Type": "application/json"}
        
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        endpoint,
                        json=request_data,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        error_msg = f"vLLM嵌入API错误: {response.status_code} - {response.text}"
                        logger.error(error_msg)
                        retry_count += 1
                        if retry_count >= self.max_retries:
                            raise Exception(error_msg)
                            
            except httpx.TimeoutException:
                logger.warning(f"vLLM嵌入API请求超时 (重试 {retry_count+1}/{self.max_retries})")
                retry_count += 1
                if retry_count >= self.max_retries:
                    raise Exception(f"vLLM嵌入API请求在{self.max_retries}次重试后仍然超时")
            
            except Exception as e:
                logger.error(f"vLLM嵌入API请求失败: {str(e)}")
                retry_count += 1
                if retry_count >= self.max_retries:
                    raise
    
    async def health_check(self) -> bool:
        """
        检查vLLM服务健康状态
        
        Returns:
            服务是否健康
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.api_url}health")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"vLLM健康检查失败: {str(e)}")
            return False 