#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:40
# @Desc   : 混合文本和图编码器的双编码器模块
# --------------------------------------------------------
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List

from .text_encoder import TextEncoder
from .dynamic_heterogeneous_graph_encoder import DynamicHeterogeneousGraphEncoder

class DualEncoder(nn.Module):
    """混合文本和图编码器的双编码器模块"""
    
    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        node_dim: int = 256,
        edge_dim: int = 64,
        time_series_dim: int = 32,
        hidden_dim: int = 256,
        output_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_text: bool = False,
        node_types: List[str] = ["DC", "TENANT", "NE", "VM", "HOST", "HA", "TRU"],
        edge_types: List[str] = ["DC_TO_TENANT", "TENANT_TO_NE", "NE_TO_VM", "VM_TO_HOST", "HOST_TO_HA", "HA_TO_TRU"],
        seq_len: int = 24,
        num_hierarchies: int = 7
    ):
        """
        初始化双编码器
        
        Args:
            text_model_name: 预训练文本模型名称
            node_dim: 输入节点特征维度
            edge_dim: 输入边特征维度
            time_series_dim: 输入时间序列特征维度
            hidden_dim: 隐藏维度
            output_dim: 输出维度
            num_layers: 图层数
            num_heads: 注意力头数
            dropout: 丢弃率
            freeze_text: 是否冻结文本编码器
            node_types: 节点类型列表
            edge_types: 边类型列表
            seq_len: 时间序列序列长度
            num_hierarchies: 层次级别数
        """
        super().__init__()
        
        # 文本编码器
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            output_dim=output_dim,
            dropout=dropout,
            freeze_base=freeze_text
        )
        
        # 图编码器
        self.graph_encoder = DynamicHeterogeneousGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            time_series_dim=time_series_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            edge_types=edge_types,
            num_levels=num_hierarchies,
            seq_len=seq_len,
            use_edge_features=True,
            dropout=dropout
        )
        
        # 对齐投影
        self.text_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
        self.graph_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        编码文本输入
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs
            
        Returns:
            包含文本嵌入的字典
        """
        # 获取文本嵌入
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 投影嵌入
        text_embedding = self.text_projection(text_outputs['pooled'])
        
        return {
            'text_embedding': text_embedding,
            'text_outputs': text_outputs
        }
        
    def encode_graph(
        self,
        node_features: torch.Tensor,
        edge_indices_dict: Dict[str, torch.Tensor],
        edge_features_dict: Dict[str, torch.Tensor],
        time_series_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        编码图输入
        
        Args:
            node_features: 节点特征矩阵
            edge_indices_dict: 边索引字典
            edge_features_dict: 边特征字典
            time_series_features: 时间序列特征
            batch: 批量分配矩阵（不使用）
            
        Returns:
            包含图嵌入的字典
        """
        # 获取图嵌入
        graph_outputs = self.graph_encoder(
            node_features=node_features,
            edge_indices_dict=edge_indices_dict,
            edge_features_dict=edge_features_dict,
            time_series_features=time_series_features
        )
        
        # 检查graph_outputs的格式
        if isinstance(graph_outputs, torch.Tensor):
            # 如果graph_outputs是张量，直接使用
            graph_embedding = self.graph_projection(graph_outputs)
            graph_outputs_dict = {'raw_embedding': graph_outputs}
        elif isinstance(graph_outputs, dict) and 'graph_embedding' in graph_outputs:
            # 如果graph_outputs是字典且包含graph_embedding键
            graph_embedding = self.graph_projection(graph_outputs['graph_embedding'])
            graph_outputs_dict = graph_outputs
        else:
            # 其他情况，假设graph_outputs是节点嵌入，进行全局池化
            # 使用平均池化作为简单的全局表示
            graph_embedding = torch.mean(graph_outputs, dim=0, keepdim=True)
            graph_embedding = self.graph_projection(graph_embedding)
            graph_outputs_dict = {'node_embeddings': graph_outputs}
        
        return {
            'graph_embedding': graph_embedding,
            'graph_outputs': graph_outputs_dict
        }
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        node_features: torch.Tensor,
        edge_indices_dict: Dict[str, torch.Tensor],
        edge_features_dict: Dict[str, torch.Tensor],
        time_series_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            node_features: 节点特征矩阵
            edge_indices_dict: 边索引字典
            edge_features_dict: 边特征字典
            time_series_features: 时间序列特征
            batch: 批量分配矩阵（不使用）
            token_type_ids: token类型IDs
            
        Returns:
            包含:
                - text_embedding: 文本嵌入
                - graph_embedding: 图嵌入
                - similarity: 嵌入之间的余弦相似度
        """
        # 编码文本和图
        text_outputs = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        graph_outputs = self.encode_graph(
            node_features=node_features,
            edge_indices_dict=edge_indices_dict,
            edge_features_dict=edge_features_dict,
            time_series_features=time_series_features
        )
        
        # 获取嵌入
        text_embedding = text_outputs['text_embedding']
        graph_embedding = graph_outputs['graph_embedding']
        
        # 计算相似度
        similarity = torch.cosine_similarity(
            text_embedding.unsqueeze(1),
            graph_embedding.unsqueeze(0),
            dim=-1
        )
        
        return {
            'text_embedding': text_embedding,
            'graph_embedding': graph_embedding,
            'similarity': similarity,
            'text_outputs': text_outputs['text_outputs'],
            'graph_outputs': graph_outputs['graph_outputs']
        }
        
    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.text_encoder.get_embedding_dim() 