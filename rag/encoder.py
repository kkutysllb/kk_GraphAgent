#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-14 22:43
# @Desc   : 双通道编码器模块，负责将图数据和文本数据编码到同一语义空间
# --------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Dict, List, Optional, Tuple
import numpy as np
from utils.logger import Logger

class DualEncoder(nn.Module):
    """双通道编码器
    
    包含两个编码器：
    1. 文本编码器：使用预训练语言模型编码文本
    2. 图编码器：使用图注意力网络编码图结构
    """
    
    def __init__(
        self,
        text_model_name: str = "bert-base-chinese",
        node_dim: int = 256,
        edge_dim: int = 128,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """初始化双通道编码器
        
        Args:
            text_model_name: 预训练语言模型名称
            node_dim: 节点特征维度
            edge_dim: 边特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super().__init__()
        self.logger = Logger(self.__class__.__name__)
        
        # 文本编码器
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # 图编码器
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
        # 图注意力层
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=hidden_dim
            )
            for _ in range(2)  # 2层GAT
        ])
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 对比学习温度参数
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
    def encode_text(
        self,
        texts: List[str],
        max_length: int = 512
    ) -> torch.Tensor:
        """编码文本序列
        
        Args:
            texts: 文本列表
            max_length: 最大序列长度
            
        Returns:
            文本编码张量
        """
        # 对文本进行分词
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # 将输入移到GPU（如果可用）
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        # 获取文本编码
        outputs = self.text_encoder(**inputs)
        text_embeddings = outputs.pooler_output  # [batch_size, hidden_size]
        
        return text_embeddings
        
    def encode_graph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """编码图结构
        
        Args:
            node_features: 节点特征矩阵 [num_nodes, node_dim]
            edge_index: 边索引矩阵 [2, num_edges]
            edge_features: 边特征矩阵 [num_edges, edge_dim]
            batch: 批处理索引 [num_nodes]
            
        Returns:
            图编码张量
        """
        # 编码节点和边特征
        x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_features) if edge_features is not None else None
        
        # 图注意力层
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr)
            x = F.elu(x)
            
        # 如果是批处理，则对每个图进行池化
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
            
        # 输出投影
        graph_embeddings = self.output_proj(x)
        
        return graph_embeddings
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor],
        texts: List[str],
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            node_features: 节点特征矩阵
            edge_index: 边索引矩阵
            edge_features: 边特征矩阵
            texts: 文本列表
            batch: 批处理索引
            
        Returns:
            图编码和文本编码的元组
        """
        # 获取图编码
        graph_embeddings = self.encode_graph(
            node_features,
            edge_index,
            edge_features,
            batch
        )
        
        # 获取文本编码
        text_embeddings = self.encode_text(texts)
        
        return graph_embeddings, text_embeddings
        
    def compute_similarity(
        self,
        graph_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """计算图编码和文本编码的相似度
        
        Args:
            graph_embeddings: 图编码矩阵 [batch_size, output_dim]
            text_embeddings: 文本编码矩阵 [batch_size, output_dim]
            
        Returns:
            相似度矩阵 [batch_size, batch_size]
        """
        # 归一化编码
        graph_embeddings = F.normalize(graph_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        # 计算余弦相似度
        similarity = torch.matmul(graph_embeddings, text_embeddings.t())
        
        # 应用温度缩放
        similarity = similarity / self.temperature
        
        return similarity
        
    def compute_loss(
        self,
        similarity: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算对比学习损失
        
        Args:
            similarity: 相似度矩阵 [batch_size, batch_size]
            labels: 标签矩阵 [batch_size, batch_size]，如果为None则使用对角矩阵
            
        Returns:
            对比学习损失
        """
        batch_size = similarity.size(0)
        
        # 如果没有提供标签，则使用对角矩阵（每个图编码应该与对应的文本编码最相似）
        if labels is None:
            labels = torch.eye(batch_size).to(similarity.device)
            
        # 计算交叉熵损失
        loss = -torch.sum(F.log_softmax(similarity, dim=1) * labels) / batch_size
        
        return loss
