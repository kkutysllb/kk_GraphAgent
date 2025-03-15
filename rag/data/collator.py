#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:34
# @Desc   : 用于批量处理和预处理数据的收集器模块
# --------------------------------------------------------
"""

import torch
from typing import Dict, List, Any, Tuple
from transformers import PreTrainedTokenizer
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GraphTextCollator:
    """用于批量处理和预处理数据的收集器模块"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化收集器
        
        Args:
            tokenizer: 用于文本处理的tokenizer
            max_length: 最大序列长度
            device: 用于放置张量的设备
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        处理和收集一批样本
        
        Args:
            batch: 数据集中的样本列表
            
        Returns:
            包含张量的批量字典
        """
        # 分离数据
        texts = [sample['text'] for sample in batch]
        node_ids = [sample['node_id'] for sample in batch]
        node_types = [sample['node_type'] for sample in batch]
        subgraphs = [sample['subgraph'] for sample in batch]
        
        # 处理文本
        text_batch = self._process_text(texts)
        
        # 处理图
        graph_batch = self._process_graphs(subgraphs)
        
        # 创建批量
        collated_batch = {
            'node_ids': node_ids,
            'node_types': node_types,
            'input_ids': text_batch['input_ids'],
            'attention_mask': text_batch['attention_mask'],
            'node_features': graph_batch['node_features'],
            'edge_index': graph_batch['edge_index'],
            'edge_features': graph_batch['edge_features'],
            'batch_idx': graph_batch['batch_idx'],
            'center_node_idx': graph_batch['center_node_idx']
        }
        
        return collated_batch
        
    def _process_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """处理文本序列"""
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'].to(self.device),
            'attention_mask': tokenized['attention_mask'].to(self.device)
        }
        
    def _process_graphs(self, subgraphs: List[Dict]) -> Dict[str, torch.Tensor]:
        """处理图结构"""
        batch_size = len(subgraphs)
        
        # 初始化列表用于批量处理
        all_node_features = []
        all_edge_features = []
        edge_indices = []
        batch_indices = []
        center_node_indices = []
        
        node_offset = 0
        for batch_idx, subgraph in enumerate(subgraphs):
            # 获取节点和边
            nodes = subgraph['nodes']
            edges = subgraph['edges']
            center_node_id = subgraph['center_node_id']
            
            # 创建节点映射（node_id -> 特征矩阵中的索引）
            node_map = {}
            for idx, node_id in enumerate(nodes.keys()):
                node_map[node_id] = idx + node_offset
            
            # 找到中心节点索引
            center_node_idx = node_map[center_node_id]
            center_node_indices.append(center_node_idx)
            
            # 提取节点特征
            node_features = self._extract_node_features(list(nodes.values()))
            all_node_features.append(node_features)
            
            # 提取边特征并创建边索引
            if edges:
                edge_features, edge_index = self._extract_edge_features_and_index(
                    list(edges.values()), node_map
                )
                all_edge_features.append(edge_features)
                edge_indices.append(edge_index)
            
            # 跟踪批量成员资格
            batch_indices.extend([batch_idx] * len(nodes))
            
            # 更新偏移量
            node_offset += len(nodes)
        
        # 连接特征和索引
        node_features_tensor = torch.cat(all_node_features, dim=0) if all_node_features else torch.empty((0, 0))
        
        if all_edge_features:
            edge_features_tensor = torch.cat(all_edge_features, dim=0)
            edge_index_tensor = torch.cat(edge_indices, dim=1)
        else:
            # 如果没有边，创建空张量
            edge_features_tensor = torch.empty((0, 0))
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        
        return {
            'node_features': node_features_tensor.to(self.device),
            'edge_features': edge_features_tensor.to(self.device),
            'edge_index': edge_index_tensor.to(self.device),
            'batch_idx': torch.tensor(batch_indices, dtype=torch.long).to(self.device),
            'center_node_idx': torch.tensor(center_node_indices, dtype=torch.long).to(self.device)
        }
        
    def _extract_node_features(self, nodes: List[Dict]) -> torch.Tensor:
        """从节点字典中提取节点特征"""
        features_list = []
        
        for node in nodes:
            # 使用预提取的特征从feature_extractor
            node_features = node['features']
            
            # 转换为张量友好的格式
            feature_vector = self._convert_features_to_vector(node_features)
            features_list.append(feature_vector)
        
        # 将特征堆叠成一个张量
        features_tensor = torch.stack(features_list) if features_list else torch.empty((0, 0))
        return features_tensor
    
    def _extract_edge_features_and_index(self, edges: List[Dict], node_map: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取边特征和边索引"""
        features_list = []
        edge_indices = []
        
        for edge in edges:
            # 使用预提取的特征从feature_extractor
            edge_features = edge['features']
            
            # 转换为张量友好的格式
            feature_vector = self._convert_features_to_vector(edge_features)
            features_list.append(feature_vector)
            
            # 创建边索引
            source_idx = node_map[edge['source']]
            target_idx = node_map[edge['target']]
            edge_indices.append([source_idx, target_idx])
        
        # 确保所有特征向量具有相同的维度
        if features_list:
            # 找到最大维度
            max_dim = max(f.size(0) for f in features_list)
            # 填充所有向量到相同维度
            padded_features = []
            for f in features_list:
                if f.size(0) < max_dim:
                    # 使用零填充
                    padded = torch.zeros(max_dim, dtype=f.dtype)
                    padded[:f.size(0)] = f
                    padded_features.append(padded)
                else:
                    padded_features.append(f)
            features_tensor = torch.stack(padded_features)
        else:
            features_tensor = torch.empty((0, 0))
        
        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t() if edge_indices else torch.empty((2, 0), dtype=torch.long)
        
        return features_tensor, edge_index_tensor
    
    def _convert_features_to_vector(self, features: Dict) -> torch.Tensor:
        """将特征字典转换为特征向量"""
        # 提取数值特征
        numerical_features = []
        
        # 处理静态特征
        for key in ['id', 'name', 'type', 'business_type', 'layer', 'layer_type']:
            if key in features and features[key]:
                # 跳过非数值特征
                continue
        
        # 如果可用，处理metrics数据
        if 'metrics_data' in features and features['metrics_data']:
            try:
                metrics = features['metrics_data']
                if isinstance(metrics, str):
                    import json
                    metrics = json.loads(metrics)
                
                # 从metrics中提取数值特征
                for metric_name, values in metrics.items():
                    if isinstance(values, list) and values:
                        # 使用最新值
                        numerical_features.append(float(values[-1]))
                    elif isinstance(values, (int, float)):
                        numerical_features.append(float(values))
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                logger.warning(f"Error processing metrics_data: {e}")
        
        # 如果可用，处理log数据
        if 'log_data' in features and features['log_data']:
            try:
                logs = features['log_data']
                if isinstance(logs, str):
                    import json
                    logs = json.loads(logs)
                
                # 从logs中提取数值特征
                for log_type, entries in logs.items():
                    if isinstance(entries, list) and entries:
                        # 使用计数作为特征
                        numerical_features.append(float(len(entries)))
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                logger.warning(f"Error processing log_data: {e}")
        
        # 如果可用，处理dynamics数据
        if 'dynamics_data' in features and features['dynamics_data']:
            try:
                dynamics = features['dynamics_data']
                if isinstance(dynamics, str):
                    import json
                    dynamics = json.loads(dynamics)
                
                # 从dynamics中提取数值特征
                if 'propagation_data' in dynamics and dynamics['propagation_data']:
                    prop_data = dynamics['propagation_data']
                    if isinstance(prop_data, list) and prop_data:
                        # 使用最新条目
                        latest = prop_data[-1]
                        if 'effects' in latest:
                            effects = latest['effects']
                            for effect_name, effect_data in effects.items():
                                if isinstance(effect_data, dict) and 'probability' in effect_data:
                                    numerical_features.append(float(effect_data['probability']))
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                logger.warning(f"Error processing dynamics_data: {e}")
        
        # 如果没有数值特征，使用默认特征
        if not numerical_features:
            numerical_features = [0.0]
        
        # 转换为张量
        return torch.tensor(numerical_features, dtype=torch.float) 