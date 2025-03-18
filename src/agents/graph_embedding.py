"""
图嵌入智能体实现
负责处理图嵌入相关操作
"""
from typing import Dict, Any, List, Optional, Union
import logging
import torch
import numpy as np
import os
import json
import httpx

from .base import Agent, Message

logger = logging.getLogger("agno.agent.graph_embedding")

class GraphEmbeddingAgent(Agent):
    """
    图嵌入智能体
    负责生成图数据的向量表示
    支持查询文本、节点和子图的嵌入
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__("graph_embedding")
        self.config = config
        self.model_path = config.get("model_path", "models/dual_stream_encoder.pt")
        self.embedding_dim = config.get("embedding_dim", 768)
        self.model = None
        self.text_encoder = None
        self.embedding_api_endpoint = config.get("embedding_api_endpoint", None)
        
        # 初始化模型或API客户端
        self._initialize_embedding_service()
    
    def _initialize_embedding_service(self):
        """
        初始化嵌入服务，可以是本地模型或API
        """
        if self.embedding_api_endpoint:
            self.logger.info(f"使用远程嵌入API: {self.embedding_api_endpoint}")
            # 使用API，不需要加载模型
            return
        
        # 检查模型文件是否存在
        if os.path.exists(self.model_path):
            try:
                self.logger.info(f"加载图嵌入模型: {self.model_path}")
                # 加载模型（根据实际模型实现加载逻辑）
                self._load_model()
            except Exception as e:
                self.logger.error(f"加载模型失败: {str(e)}", exc_info=True)
                self.logger.warning("回退到基本文本嵌入服务")
                self._initialize_text_encoder()
        else:
            self.logger.warning(f"模型文件不存在: {self.model_path}，回退到基本文本嵌入服务")
            self._initialize_text_encoder()
    
    def _load_model(self):
        """
        加载双流图嵌入模型
        """
        # 实际实现中，这里应该加载DualStreamGraphEncoder模型
        # 简化实现，仅做示意
        try:
            # 模拟模型加载
            # self.model = torch.load(self.model_path)
            # self.model.eval()
            pass
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}", exc_info=True)
            raise
    
    def _initialize_text_encoder(self):
        """
        初始化基本文本编码器（作为回退选项）
        """
        # 实际实现中，这里可以加载轻量级文本编码模型
        # 简化实现，仅做示意
        self.text_encoder = "sentence-transformers/all-MiniLM-L6-v2"
    
    async def on_message(self, msg: Message) -> Message:
        """
        处理消息，生成嵌入向量
        """
        embed_type = msg.data.get("embed_type", "query")
        
        try:
            if embed_type == "query":
                # 查询文本嵌入
                query = msg.data.get("query", "")
                if not query:
                    return msg.update(error="未提供查询文本")
                
                vector = await self._encode_query(query)
                
            elif embed_type == "node":
                # 节点嵌入
                node_data = msg.data.get("node_data", {})
                if not node_data:
                    return msg.update(error="未提供节点数据")
                
                vector = await self._encode_node(node_data)
                
            elif embed_type == "subgraph":
                # 子图嵌入
                subgraph_data = msg.data.get("subgraph_data", {})
                if not subgraph_data:
                    return msg.update(error="未提供子图数据")
                
                vector = await self._encode_subgraph(subgraph_data)
                
            else:
                return msg.update(error=f"不支持的嵌入类型: {embed_type}")
            
            # 更新消息
            return msg.update(
                embedding_vector=vector,
                embedding_dimension=len(vector)
            )
            
        except Exception as e:
            self.logger.error(f"生成嵌入向量失败: {str(e)}", exc_info=True)
            return msg.update(error=f"生成嵌入向量失败: {str(e)}")
    
    async def _encode_query(self, query: str) -> List[float]:
        """
        编码查询文本为向量
        """
        if self.embedding_api_endpoint:
            # 使用API生成嵌入
            return await self._call_embedding_api("query", {"text": query})
        
        elif self.model:
            # 使用加载的模型生成嵌入
            # 实际实现中，应该调用模型的文本编码方法
            # 简化实现，返回随机向量
            return self._generate_random_embedding()
            
        else:
            # 使用基本文本编码器生成嵌入
            # 简化实现，返回随机向量
            return self._generate_random_embedding()
    
    async def _encode_node(self, node_data: Dict[str, Any]) -> List[float]:
        """
        编码节点数据为向量
        """
        if self.embedding_api_endpoint:
            # 使用API生成嵌入
            return await self._call_embedding_api("node", node_data)
        
        elif self.model:
            # 使用加载的模型生成嵌入
            # 简化实现，返回随机向量
            return self._generate_random_embedding()
            
        else:
            # 回退到基于文本的节点表示
            node_text = self._node_to_text(node_data)
            return await self._encode_query(node_text)
    
    async def _encode_subgraph(self, subgraph_data: Dict[str, Any]) -> List[float]:
        """
        编码子图数据为向量
        """
        if self.embedding_api_endpoint:
            # 使用API生成嵌入
            return await self._call_embedding_api("subgraph", subgraph_data)
        
        elif self.model:
            # 使用加载的模型生成嵌入
            # 简化实现，返回随机向量
            return self._generate_random_embedding()
            
        else:
            # 回退到基于文本的子图表示
            subgraph_text = self._subgraph_to_text(subgraph_data)
            return await self._encode_query(subgraph_text)
    
    async def _call_embedding_api(self, embed_type: str, data: Dict[str, Any]) -> List[float]:
        """
        调用嵌入API
        """
        headers = {"Content-Type": "application/json"}
        payload = {
            "embed_type": embed_type,
            "data": data
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.embedding_api_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "embedding" in result:
                        return result["embedding"]
                    else:
                        raise Exception("API响应中未包含嵌入向量")
                else:
                    self.logger.error(f"嵌入API错误: {response.status_code} - {response.text}")
                    raise Exception(f"嵌入API错误: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"调用嵌入API时出错: {str(e)}", exc_info=True)
                raise Exception(f"调用嵌入API失败: {str(e)}")
    
    def _generate_random_embedding(self) -> List[float]:
        """
        生成随机嵌入向量（用于测试或回退）
        """
        # 生成随机向量
        vector = np.random.normal(0, 1, self.embedding_dim)
        
        # 归一化
        vector = vector / np.linalg.norm(vector)
        
        return vector.tolist()
    
    def _node_to_text(self, node_data: Dict[str, Any]) -> str:
        """
        将节点数据转换为文本表示
        """
        node_type = node_data.get("type", "")
        props = node_data.get("properties", {})
        
        # 构建文本表示
        text_parts = [f"节点类型: {node_type}"]
        
        for key, value in props.items():
            text_parts.append(f"{key}: {value}")
        
        return ", ".join(text_parts)
    
    def _subgraph_to_text(self, subgraph_data: Dict[str, Any]) -> str:
        """
        将子图数据转换为文本表示
        """
        nodes = subgraph_data.get("nodes", [])
        relationships = subgraph_data.get("relationships", [])
        
        # 构建文本表示
        text_parts = []
        
        # 添加节点信息
        for node in nodes:
            node_text = self._node_to_text(node)
            text_parts.append(node_text)
        
        # 添加关系信息
        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            rel_type = rel.get("type", "")
            text_parts.append(f"关系: {source} -{rel_type}-> {target}")
        
        return "; ".join(text_parts) 