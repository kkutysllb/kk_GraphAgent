#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-22 10:00
# @Desc   : 动态异构图编码器，基于DyHAN模型架构，针对5G核心网资源图谱进行优化
# --------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class NodeLevelAttention(nn.Module):
    """节点级注意力层
    
    为每种边类型学习邻居节点的重要性，并聚合邻居特征。
    
    Args:
        input_dim: 输入节点特征维度
        edge_dim: 输入边特征维度
        hidden_dim: 隐藏层维度
        edge_types: 边类型列表
        dropout: Dropout比率
        use_edge_features: 是否在注意力计算中使用边特征
    """
    
    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden_dim: int,
        edge_types: List[str],
        dropout: float = 0.1,
        use_edge_features: bool = True
    ):
        super().__init__()
        
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.use_edge_features = use_edge_features
        self.input_dim = input_dim
        
        # 节点特征投影（为每种边类型创建独立的变换）
        self.node_transforms = nn.ModuleDict({
            edge_type: nn.Linear(input_dim, hidden_dim)
            for edge_type in edge_types
        })
        
        # 边特征投影（如果使用边特征）
        if use_edge_features:
            self.edge_transforms = nn.ModuleDict({
                edge_type: nn.Linear(edge_dim, hidden_dim)
                for edge_type in edge_types
            })
            
            # 注意力MLP（考虑源节点、目标节点和边特征）
            self.attention_mlp_with_edge = nn.ModuleDict({
                edge_type: nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1)
                )
                for edge_type in edge_types
            })
        
        # 注意力MLP（仅考虑源节点和目标节点）
        self.attention_mlp = nn.ModuleDict({
            edge_type: nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            for edge_type in edge_types
        })
        
        # 输出变换
        self.output_transforms = nn.ModuleDict({
            edge_type: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for edge_type in edge_types
        })
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_indices_dict: Dict[str, torch.Tensor],
        edge_features_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            node_features: 节点特征矩阵 [num_nodes, input_dim]
            edge_indices_dict: 边索引字典 {edge_type: [2, num_edges]}
            edge_features_dict: 边特征字典 {edge_type: [num_edges, edge_dim]}
            
        Returns:
            edge_type_embeddings: 每种边类型的节点嵌入 {edge_type: [num_nodes, hidden_dim]}
        """
        edge_type_embeddings = {}
        num_nodes = node_features.size(0)
        device = node_features.device
        
        for edge_type in self.edge_types:
            if edge_type not in edge_indices_dict:
                # 如果没有这种类型的边，创建零张量
                edge_type_embeddings[edge_type] = torch.zeros(
                    num_nodes, self.hidden_dim, device=device
                )
                continue
                
            # 获取边索引
            edge_index = edge_indices_dict[edge_type]
            
            # 变换节点特征
            transformed_features = self.node_transforms[edge_type](node_features)
            
            # 计算注意力系数
            if self.use_edge_features and edge_features_dict is not None and edge_type in edge_features_dict:
                # 使用边特征
                edge_features = edge_features_dict[edge_type]
                transformed_edge_features = self.edge_transforms[edge_type](edge_features)
                
                # 获取源节点和目标节点
                src, dst = edge_index
                
                # 获取源节点、目标节点和边的特征
                h_src = transformed_features[src]  # [num_edges, hidden_dim]
                h_dst = transformed_features[dst]  # [num_edges, hidden_dim]
                
                # 连接源节点、目标节点和边的特征
                concat_features = torch.cat([h_src, h_dst, transformed_edge_features], dim=1)  # [num_edges, hidden_dim*3]
                
                # 计算未归一化的注意力系数
                attention = self.attention_mlp_with_edge[edge_type](concat_features)  # [num_edges, 1]
                
                # 对每个节点的邻居进行归一化
                attention_coefficients = self._normalize_attention(attention, src, num_nodes)
            else:
                # 不使用边特征
                # 获取源节点和目标节点
                src, dst = edge_index
                
                # 获取源节点和目标节点的特征
                h_src = transformed_features[src]  # [num_edges, hidden_dim]
                h_dst = transformed_features[dst]  # [num_edges, hidden_dim]
                
                # 连接源节点和目标节点的特征
                concat_features = torch.cat([h_src, h_dst], dim=1)  # [num_edges, hidden_dim*2]
                
                # 计算未归一化的注意力系数
                attention = self.attention_mlp[edge_type](concat_features)  # [num_edges, 1]
                
                # 对每个节点的邻居进行归一化
                attention_coefficients = self._normalize_attention(attention, src, num_nodes)
            
            # 聚合邻居特征
            edge_type_embedding = self._aggregate_neighbors(
                transformed_features, 
                edge_index, 
                attention_coefficients
            )
            
            # 应用输出变换
            edge_type_embedding = self.output_transforms[edge_type](edge_type_embedding)
            
            edge_type_embeddings[edge_type] = edge_type_embedding
        
        return edge_type_embeddings
    
    def _normalize_attention(self, attention, src_indices, num_nodes):
        """对注意力系数进行归一化"""
        # 创建一个稀疏矩阵来存储每个源节点的所有邻居的注意力系数
        src_indices_expanded = src_indices.unsqueeze(-1).expand(-1, 1)
        
        # 对每个源节点的邻居进行softmax归一化
        # 首先，我们需要计算每个源节点的邻居数量
        ones = torch.ones_like(attention)
        src_node_degrees = torch.zeros(num_nodes, 1, device=attention.device)
        src_node_degrees.scatter_add_(0, src_indices_expanded, ones)
        
        # 为了避免除以零，将度为0的节点设为1
        src_node_degrees[src_node_degrees == 0] = 1
        
        # 计算每个源节点的邻居注意力系数之和
        attention_sum = torch.zeros(num_nodes, 1, device=attention.device)
        attention_exp = torch.exp(attention)
        attention_sum.scatter_add_(0, src_indices_expanded, attention_exp)
        
        # 归一化
        attention_sum_gathered = attention_sum[src_indices]
        attention_normalized = attention_exp / attention_sum_gathered
        
        # 应用dropout
        attention_normalized = self.dropout(attention_normalized)
        
        return attention_normalized
    
    def _aggregate_neighbors(self, features, edge_index, attention_coefficients):
        """聚合邻居特征"""
        # 获取源节点和目标节点
        src, dst = edge_index
        
        # 加权聚合
        weighted_features = features[dst] * attention_coefficients
        
        # 使用scatter_add聚合到源节点
        output = torch.zeros_like(features)
        src_expanded = src.unsqueeze(-1).expand(-1, features.size(1))
        output.scatter_add_(0, src_expanded, weighted_features)
        
        return output


class EdgeLevelAttention(nn.Module):
    """边级注意力层
    
    学习不同边类型的重要性，并将边类型特定的节点嵌入融合为统一的节点表示。
    
    Args:
        hidden_dim: 隐藏层维度
        edge_types: 边类型列表
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        hidden_dim: int,
        edge_types: List[str],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.edge_types = edge_types
        
        # 为每种边类型创建单独的注意力计算
        self.edge_type_attentions = nn.ModuleDict({
            edge_type: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                nn.Dropout(dropout)
            ) for edge_type in edge_types
        })
        
        # 输出变换
        self.output_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        edge_type_embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            edge_type_embeddings: 每种边类型的节点嵌入 {edge_type: [num_nodes, hidden_dim]}
            
        Returns:
            fused_embeddings: 融合后的节点嵌入 [num_nodes, hidden_dim]
        """
        # 检查是否有边类型嵌入
        if not edge_type_embeddings:
            # 如果没有边类型嵌入，返回零张量
            num_nodes = next(iter(edge_type_embeddings.values())).size(0) if edge_type_embeddings else 0
            hidden_dim = next(iter(edge_type_embeddings.values())).size(1) if edge_type_embeddings else 0
            device = next(iter(edge_type_embeddings.values())).device if edge_type_embeddings else torch.device('cpu')
            return torch.zeros(num_nodes, hidden_dim, device=device)
        
        # 获取节点数和设备
        num_nodes = next(iter(edge_type_embeddings.values())).size(0)
        hidden_dim = next(iter(edge_type_embeddings.values())).size(1)
        device = next(iter(edge_type_embeddings.values())).device
        
        # 计算每种边类型的注意力分数
        valid_edge_types = []
        attention_scores = []
        
        for edge_type in self.edge_types:
            if edge_type in edge_type_embeddings:
                valid_edge_types.append(edge_type)
                # 使用对应边类型的注意力网络计算分数
                score = self.edge_type_attentions[edge_type](edge_type_embeddings[edge_type])  # [num_nodes, 1]
                attention_scores.append(score)
        
        # 如果没有有效的边类型，返回零张量
        if not valid_edge_types:
            return torch.zeros(num_nodes, hidden_dim, device=device)
        
        # 对注意力分数进行softmax归一化
        stacked_scores = torch.cat(attention_scores, dim=1)  # [num_nodes, num_valid_edge_types]
        normalized_scores = F.softmax(stacked_scores, dim=1)  # [num_nodes, num_valid_edge_types]
        
        # 使用注意力分数加权聚合不同边类型的嵌入
        fused_embedding = torch.zeros(num_nodes, hidden_dim, device=device)
        
        for i, edge_type in enumerate(valid_edge_types):
            # 获取该边类型的归一化分数
            score = normalized_scores[:, i:i+1]  # [num_nodes, 1]
            # 加权聚合
            fused_embedding += edge_type_embeddings[edge_type] * score
        
        # 应用输出变换
        fused_embedding = self.output_transform(fused_embedding)
        
        return fused_embedding


class TimeSeriesEncoder(nn.Module):
    """时间序列编码器
    
    处理节点的时间序列动态属性，捕捉时间维度的模式。
    
    Args:
        input_dim: 输入时间序列特征维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        seq_len: 时间序列长度
        num_layers: LSTM层数
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 时间序列数据，形状为 [batch_size, seq_len, input_dim]
            
        Returns:
            编码后的时间序列表示，形状为 [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # 应用LSTM
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]
        
        # 应用注意力机制
        attention_scores = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        
        # 加权聚合
        context = torch.sum(lstm_out * attention_weights, dim=1)  # [batch_size, hidden_dim*2]
        
        # 输出投影
        output = self.output_projection(context)  # [batch_size, output_dim]
        
        return output


class TemporalLevelAttention(nn.Module):
    """时间级注意力层
    
    学习不同时间点的重要性，并将时间序列特征与静态特征融合。
    
    Args:
        static_dim: 静态特征维度
        temporal_dim: 时间特征维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        static_dim: int,
        temporal_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.static_dim = static_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 静态特征投影
        self.static_projection = nn.Linear(static_dim, hidden_dim)
        
        # 时间特征投影
        self.temporal_projection = nn.Linear(temporal_dim, hidden_dim)
        
        # 注意力计算
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        # 注意力向量
        self.attention_vector = nn.Parameter(torch.randn(hidden_dim, 1))
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        static_features: torch.Tensor,
        temporal_features: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            static_features: 静态特征，形状为 [batch_size, static_dim]
            temporal_features: 时间特征，形状为 [batch_size, temporal_dim]
            
        Returns:
            fused_features: 融合后的特征，形状为 [batch_size, output_dim]
        """
        batch_size = static_features.size(0)
        
        # 投影静态特征和时间特征
        h_static = self.static_projection(static_features)  # [batch_size, hidden_dim]
        h_temporal = self.temporal_projection(temporal_features)  # [batch_size, hidden_dim]
        
        # 计算注意力权重
        concat_features = torch.cat([h_static, h_temporal], dim=1)  # [batch_size, hidden_dim*2]
        attention_hidden = self.attention_mlp(concat_features)  # [batch_size, hidden_dim]
        attention_score = torch.matmul(attention_hidden, self.attention_vector)  # [batch_size, 1]
        attention_weights = F.softmax(attention_score, dim=0)  # [batch_size, 1]
        
        # 加权融合
        weighted_temporal = h_temporal * attention_weights  # [batch_size, hidden_dim]
        
        # 连接静态特征和加权时间特征
        fused_features = torch.cat([h_static, weighted_temporal], dim=1)  # [batch_size, hidden_dim*2]
        
        # 应用融合层
        output = self.fusion_layer(fused_features)  # [batch_size, output_dim]
        
        return output


class HierarchicalAwarenessModule(nn.Module):
    """层级感知模块
    
    处理节点的层级关系，捕捉不同层级之间的依赖关系。
    
    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        num_levels: 层级数量
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_levels: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_levels = num_levels
        
        # 层级特征投影
        self.level_projections = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_levels)
        ])
        
        # 层级间注意力
        self.inter_level_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        node_features: torch.Tensor,
        node_levels: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: 节点特征，形状为 [num_nodes, input_dim]
            node_levels: 节点层级，形状为 [num_nodes]，值范围为0到num_levels-1
            
        Returns:
            hierarchical_features: 层级感知特征，形状为 [num_nodes, output_dim]
        """
        num_nodes = node_features.size(0)
        device = node_features.device
        
        # 初始化输出特征
        hierarchical_features = torch.zeros(num_nodes, self.hidden_dim, device=device)
        
        # 对每个层级进行处理
        for level in range(self.num_levels):
            # 获取当前层级的节点掩码
            level_mask = (node_levels == level)
            
            # 如果当前层级没有节点，则跳过
            if not torch.any(level_mask):
                continue
            
            # 获取当前层级的节点特征
            level_node_features = node_features[level_mask]
            
            # 投影当前层级的节点特征
            projected_features = self.level_projections[level](level_node_features)
            
            # 将投影后的特征存储到对应位置
            hierarchical_features[level_mask] = projected_features
        
        # 计算层级间注意力
        enhanced_features = torch.zeros_like(hierarchical_features)
        
        for i in range(num_nodes):
            current_level = node_levels[i].item()
            current_feature = hierarchical_features[i].unsqueeze(0)  # [1, hidden_dim]
            
            # 初始化注意力分数
            attention_scores = []
            level_features = []
            
            # 对每个层级计算注意力分数
            for level in range(self.num_levels):
                # 获取当前层级的节点掩码
                level_mask = (node_levels == level)
                
                # 如果当前层级没有节点，则跳过
                if not torch.any(level_mask):
                    continue
                
                # 获取当前层级的平均特征
                level_feature = torch.mean(hierarchical_features[level_mask], dim=0, keepdim=True)  # [1, hidden_dim]
                
                # 计算注意力分数
                concat_feature = torch.cat([current_feature, level_feature], dim=1)  # [1, hidden_dim*2]
                score = self.inter_level_attention(concat_feature)  # [1, 1]
                
                attention_scores.append(score)
                level_features.append(level_feature)
            
            # 如果没有有效的层级，则跳过
            if not attention_scores:
                enhanced_features[i] = hierarchical_features[i]
                continue
            
            # 对注意力分数进行softmax归一化
            attention_scores = torch.cat(attention_scores, dim=0)  # [num_valid_levels, 1]
            attention_weights = F.softmax(attention_scores, dim=0)  # [num_valid_levels, 1]
            
            # 加权聚合不同层级的特征
            level_features = torch.cat(level_features, dim=0)  # [num_valid_levels, hidden_dim]
            weighted_features = torch.sum(level_features * attention_weights, dim=0)  # [hidden_dim]
            
            # 存储增强后的特征
            enhanced_features[i] = weighted_features
        
        # 应用输出投影
        output = self.output_projection(enhanced_features)  # [num_nodes, output_dim]
        
        return output


class DynamicHeterogeneousGraphEncoder(nn.Module):
    """动态异构图编码器
    
    基于DyHAN模型架构，针对5G核心网资源图谱进行优化的图编码器。
    能够处理层级异构图结构和时间序列动态属性。
    
    Args:
        node_dim: 节点特征维度
        edge_dim: 边特征维度
        time_series_dim: 时间序列特征维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        edge_types: 边类型列表
        num_levels: 层级数量
        seq_len: 时间序列长度
        use_edge_features: 是否在注意力计算中使用边特征
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        time_series_dim: int,
        hidden_dim: int,
        output_dim: int,
        edge_types: List[str],
        num_levels: int,
        seq_len: int,
        use_edge_features: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_series_dim = time_series_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_types = edge_types
        self.num_levels = num_levels
        self.seq_len = seq_len
        self.use_edge_features = use_edge_features
        
        # 节点级注意力层
        self.node_level_attention = NodeLevelAttention(
            input_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            edge_types=edge_types,
            dropout=dropout,
            use_edge_features=use_edge_features
        )
        
        # 边级注意力层
        self.edge_level_attention = EdgeLevelAttention(
            hidden_dim=hidden_dim,
            edge_types=edge_types,
            dropout=dropout
        )
        
        # 时间序列编码器
        self.time_series_encoder = TimeSeriesEncoder(
            input_dim=time_series_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            seq_len=seq_len,
            dropout=dropout
        )
        
        # 时间级注意力层
        self.temporal_level_attention = TemporalLevelAttention(
            static_dim=hidden_dim,
            temporal_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # 层级感知模块
        self.hierarchical_awareness = HierarchicalAwarenessModule(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_levels=num_levels,
            dropout=dropout
        )
        
        # 最终输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_indices_dict: Dict[str, torch.Tensor],
        edge_features_dict: Optional[Dict[str, torch.Tensor]] = None,
        time_series_features: Optional[torch.Tensor] = None,
        node_levels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: 节点特征矩阵 [num_nodes, node_dim]
            edge_indices_dict: 边索引字典 {edge_type: [2, num_edges]}
            edge_features_dict: 边特征字典 {edge_type: [num_edges, edge_dim]}
            time_series_features: 时间序列特征 [num_nodes, seq_len, time_series_dim]
            node_levels: 节点层级 [num_nodes]
            
        Returns:
            node_embeddings: 节点嵌入 [num_nodes, output_dim]
        """
        # 1. 节点级注意力
        edge_type_embeddings = self.node_level_attention(
            node_features, edge_indices_dict, edge_features_dict
        )
        
        # 2. 边级注意力
        static_embeddings = self.edge_level_attention(edge_type_embeddings)
        
        # 初始化最终嵌入为静态嵌入
        final_embeddings = static_embeddings
        
        # 3. 时间序列处理（如果提供）
        if time_series_features is not None:
            # 编码时间序列
            temporal_embeddings = self.time_series_encoder(time_series_features)
            
            # 4. 时间级注意力
            final_embeddings = self.temporal_level_attention(
                static_embeddings, temporal_embeddings
            )
        
        # 5. 层级感知（如果提供）
        if node_levels is not None:
            final_embeddings = self.hierarchical_awareness(
                final_embeddings, node_levels
            )
        
        # 6. 最终输出投影
        node_embeddings = self.output_projection(final_embeddings)
        
        return node_embeddings
