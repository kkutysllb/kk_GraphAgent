#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:40
# @Desc   : 基于GAT的图编码器
# --------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Dict, List, Optional

class GraphEncoder(nn.Module):
    """基于GAT的图编码器"""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        residual: bool = True
    ):
        """
        初始化图编码器
        
        Args:
            node_dim: 输入节点特征维度
            edge_dim: 输入边特征维度
            hidden_dim: 隐藏维度
            output_dim: 输出维度
            num_layers: GAT层数
            num_heads: 注意力头数
            dropout: 丢弃率
            residual: 是否使用残差连接
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        
        # 节点特征投影
        self.node_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 边特征投影
        self.edge_projection = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GAT层
        self.gat_layers = nn.ModuleList()
        
        # 第一层
        self.gat_layers.append(
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=hidden_dim,
                concat=True
            )
        )
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=hidden_dim,
                    concat=True
                )
            )
            
        # 输出层
        self.gat_layers.append(
            GATConv(
                in_channels=hidden_dim,
                out_channels=output_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=hidden_dim,
                concat=True
            )
        )
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
        ])
        self.layer_norms.append(nn.LayerNorm(output_dim))
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            node_features: Node feature matrix
            edge_index: Edge index matrix
            edge_features: Edge feature matrix
            batch: Batch assignment matrix
            
        Returns:
            Dictionary containing:
                - node_embeddings: Node embeddings
                - graph_embedding: Graph embedding
        """
        # Project features
        x = self.node_projection(node_features)
        edge_attr = self.edge_projection(edge_features)
        
        # Initial features
        node_embeddings = [x]
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            # 残差连接
            if self.residual and i > 0:
                x = x + node_embeddings[-1]
                
            # 应用GAT
            x = gat_layer(x, edge_index, edge_attr=edge_attr)
            
            # 应用归一化和激活
            x = self.layer_norms[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
            node_embeddings.append(x)
            
        # 池化图嵌入
        if batch is not None:
            graph_embedding = global_mean_pool(x, batch)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)
            
        return {
            'node_embeddings': node_embeddings[-1],
            'graph_embedding': graph_embedding,
            'all_embeddings': node_embeddings
        }
        
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.output_dim 