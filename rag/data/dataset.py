#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:33
# @Desc   : 用于加载和预处理图和文本数据的数据集模块。
# --------------------------------------------------------
"""

from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Counter
import json
import logging
import random
from tqdm import tqdm
import math
from collections import defaultdict
import threading


from ..feature_extractor import FeatureExtractor
from preprocess.utils.neo4j_graph_manager import Neo4jGraphManager
from ..utils.logging import LoggerMixin

logger = logging.getLogger(__name__)

# 添加全局锁，用于控制日志输出
_dataset_log_lock = threading.Lock()
_node_loading_logged = {}
_edge_loading_logged = {}

class GraphTextDataset(Dataset, LoggerMixin):
    """用于生成图和文本数据集的类"""
    
    def __init__(
        self,
        graph_manager: Neo4jGraphManager,
        feature_extractor: FeatureExtractor,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[str]] = None,
        max_text_length: int = 512,
        max_node_size: int = 100,
        max_edge_size: int = 200,
        include_dynamic: bool = True,
        data_augmentation: bool = True,
        balance_node_types: bool = True,
        adaptive_subgraph_size: bool = True,
        negative_sample_ratio: float = 0.5,
        split: str = "train",
        split_ratio: Dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1},
        seed: int = 42,
        skip_internal_split=False
    ):
        """
        初始化数据集
        
        Args:
            graph_manager: Neo4j图管理器实例
            feature_extractor: 特征提取器实例
            node_types: 要包含的节点类型列表（如果为None，则使用feature_extractor中的所有类型）
            edge_types: 要包含的边类型列表（如果为None，则使用feature_extractor中的所有类型）
            max_text_length: 最大文本序列长度
            max_node_size: 子图中的最大节点数
            max_edge_size: 子图中的最大边数
            include_dynamic: 是否包含动态特征
            data_augmentation: 是否应用数据增强
            balance_node_types: 是否平衡不同节点类型的样本数量
            adaptive_subgraph_size: 是否根据节点类型自适应调整子图大小
            negative_sample_ratio: 负样本比例
            split: 数据集分割（"train", "val", or "test"）
            split_ratio: 数据集分割比例
            seed: 随机种子
            skip_internal_split: 是否跳过内部的数据集分割
        """
        global _dataset_log_lock
        
        super().__init__()
        self.graph_manager = graph_manager
        self.feature_extractor = feature_extractor
        self.node_types = node_types or feature_extractor.node_types
        self.edge_types = edge_types or feature_extractor.relationship_types
        self.max_text_length = max_text_length
        self.max_node_size = max_node_size
        self.max_edge_size = max_edge_size
        self.include_dynamic = include_dynamic
        self.data_augmentation = data_augmentation
        self.balance_node_types = balance_node_types
        self.adaptive_subgraph_size = adaptive_subgraph_size
        self.negative_sample_ratio = negative_sample_ratio
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 记载图数据
        self.nodes, self.edges = self._load_graph_data()
        
        # 生成文本描述
        with _dataset_log_lock:
            self.log_info("生成节点文本描述...")
        self.text_descriptions = self._generate_text_descriptions()
        
        # 创建图-文对
        with _dataset_log_lock:
            self.log_info("Creating graph-text pairs...")
        self.pairs = self._create_pairs()
        
        # 应用数据集分割（如果需要）
        if not skip_internal_split:
            self.pairs = self._apply_dataset_split()
        
        # 平衡节点类型（如果启用）
        if self.balance_node_types and self.split == "train":
            self.pairs = self._balance_node_types()
        
        # 生成负样本（如果启用）
        if self.negative_sample_ratio > 0 and self.split == "train":
            negative_pairs = self._generate_negative_samples()
            self.pairs.extend(negative_pairs)
            with _dataset_log_lock:
                self.log_info(f"Added {len(negative_pairs)} negative samples")
        
        # 记录数据集统计信息
        with _dataset_log_lock:
            self.log_info(f"Created {split} dataset with {len(self.pairs)} graph-text pairs")
            self.log_info(f"Node types: {self.node_types}, Edge types: {self.edge_types}")
            
            # 统计节点类型分布
            node_type_counts = Counter([pair["node_type"] for pair in self.pairs])
            for node_type, count in node_type_counts.items():
                self.log_info(f"Node type {node_type}: {count} samples")
            
            # 统计正负样本比例
            if self.negative_sample_ratio > 0 and self.split == "train":
                positive_count = sum(1 for pair in self.pairs if not pair.get("is_negative", False))
                negative_count = sum(1 for pair in self.pairs if pair.get("is_negative", False))
                self.log_info(f"Positive:Negative ratio = {positive_count}:{negative_count} = {positive_count/negative_count:.2f}")
        
        # 初始化数据集大小
        self.dataset_size = len(self.pairs)
    
    def save(self, file_path: str):
        """保存数据集到文件"""
        import torch
        
        # 准备要保存的数据
        data_to_save = {
            'pairs': self.pairs,
            'config': {
                'node_types': self.node_types,
                'edge_types': self.edge_types,
                'max_text_length': self.max_text_length,
                'max_node_size': self.max_node_size,
                'max_edge_size': self.max_edge_size,
                'include_dynamic': self.include_dynamic,
                'data_augmentation': self.data_augmentation,
                'balance_node_types': self.balance_node_types,
                'adaptive_subgraph_size': self.adaptive_subgraph_size,
                'negative_sample_ratio': self.negative_sample_ratio,
                'split': self.split,
                'split_ratio': self.split_ratio,
                'seed': self.seed
            }
        }
        
        # 使用torch.save保存数据
        torch.save(data_to_save, file_path)
    
    @classmethod
    def load(cls, file_path: str):
        """从文件加载数据集"""
        import torch
        
        # 加载数据
        data = torch.load(file_path)
        
        # 创建数据集实例
        dataset = cls.__new__(cls)
        
        # 设置配置
        config = data['config']
        dataset.node_types = config['node_types']
        dataset.edge_types = config['edge_types']
        dataset.max_text_length = config['max_text_length']
        dataset.max_node_size = config['max_node_size']
        dataset.max_edge_size = config['max_edge_size']
        dataset.include_dynamic = config['include_dynamic']
        dataset.data_augmentation = config['data_augmentation']
        dataset.balance_node_types = config['balance_node_types']
        dataset.adaptive_subgraph_size = config['adaptive_subgraph_size']
        dataset.negative_sample_ratio = config['negative_sample_ratio']
        dataset.split = config['split']
        dataset.split_ratio = config['split_ratio']
        dataset.seed = config['seed']
        
        # 设置数据
        dataset.pairs = data['pairs']
        dataset.dataset_size = len(dataset.pairs)
        
        # 设置其他属性
        dataset.graph_manager = None
        dataset.feature_extractor = None
        
        return dataset
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.dataset_size
    
    def __getitem__(self, idx: int) -> Dict:
        """获取数据项"""
        pair = self.pairs[idx]
        return {
            'node_id': pair['node_id'],
            'text': pair['text'],
            'subgraph': pair['subgraph'],
            'node_type': pair['node_type']
        }
    
    def _load_graph_data(self) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """从Neo4j加载节点和边及其特征"""
        global _dataset_log_lock, _node_loading_logged, _edge_loading_logged
        
        # 使用线程锁控制日志输出
        with _dataset_log_lock:
            if self.split not in _node_loading_logged:
                self.log_info("从Neo4j加载图数据...")
                _node_loading_logged[self.split] = True
        
        # 按类型获取节点及其特征
        nodes = {}
        for node_type in tqdm(self.node_types, desc="加载节点"):
            # 从Neo4j获取节点ID
            query = f"""
            MATCH (n:{node_type})
            RETURN DISTINCT id(n) as node_id, n.id as node_name
            """
            with self.graph_manager._driver.session() as session:
                result = session.run(query)
                for record in result:
                    node_id = record["node_id"]
                    node_name = record["node_name"]
                    # 使用特征提取器提取节点特征
                    node_data = self.feature_extractor.extract_node_features(node_id, node_type)
                    nodes[str(node_id)] = node_data
        
        # 按类型获取边及其特征
        edges = {}
        for edge_type in tqdm(self.edge_types, desc="加载边"):
            # 从Neo4j获取边信息
            query = f"""
            MATCH (source)-[r:{edge_type}]->(target)
            WITH DISTINCT source, target, r, id(source) as source_id, id(target) as target_id, id(r) as edge_id
            RETURN source_id, target_id, edge_id
            """
            with self.graph_manager._driver.session() as session:
                result = session.run(query)
                for record in result:
                    source_id = record["source_id"]
                    target_id = record["target_id"]
                    edge_id = record["edge_id"]
                    # 使用特征提取器提取边特征
                    features = self.feature_extractor.extract_edge_features(edge_id)
                    edge_key = f"{source_id}->{target_id}"
                    edges[edge_key] = {
                        "id": str(edge_id),
                        "source": str(source_id),
                        "target": str(target_id),
                        "type": edge_type,
                        "features": features
                    }
        
        # 使用线程锁控制日志输出
        with _dataset_log_lock:
            if self.split not in _edge_loading_logged:
                self.log_info(f"Loaded {len(nodes)} nodes and {len(edges)} edges")
                _edge_loading_logged[self.split] = True
        
        return nodes, edges
    
    def _generate_text_descriptions(self) -> Dict[str, str]:
        """生成节点文本描述""" 
        descriptions = {}
        for node_id, node in tqdm(self.nodes.items(), desc="生成描述"):
            # 获取连接的节点
            connected_nodes = self._get_connected_nodes(node_id)
            
            # 使用节点类型、特征和连接生成描述
            desc = self._format_description(node["type"], node, connected_nodes)
            descriptions[node_id] = desc
            
        return descriptions
    
    def _get_connected_nodes(self, node_id: str) -> List[Dict]:
        """获取连接到给定节点的节点"""
        connected = []
        
        for edge_key, edge in self.edges.items():
            if edge["source"] == node_id:
                target_id = edge["target"]
                if target_id in self.nodes:
                    connected.append({
                        "id": target_id,
                        "type": self.nodes[target_id]["type"],
                        "name": self.nodes[target_id]["name"],
                        "relation": edge["type"],
                        "direction": "outgoing"
                    })
            elif edge["target"] == node_id:
                source_id = edge["source"]
                if source_id in self.nodes:
                    connected.append({
                        "id": source_id,
                        "type": self.nodes[source_id]["type"],
                        "name": self.nodes[source_id]["name"],
                        "relation": edge["type"],
                        "direction": "incoming"
                    })
        
        return connected
    
    def _format_description(
        self,
        node_type: str,
        node_data: Dict,
        connected_nodes: List[Dict]
    ) -> str:
        """格式化节点描述，基于节点类型、特征和连接关系"""
        # 中文模板
        templates = {
            "DC": "这是一个数据中心 (DC)，ID为{id}。{details}",
            "TENANT": "这是5G核心网中的一个租户，ID为{id}。{details}",
            "NE": "这是一个网元 (NE)，ID为{id}。{details}",
            "VM": "这是一个虚拟机 (VM)，ID为{id}。{details}",
            "HOST": "这是一个主机服务器，ID为{id}。{details}",
            "HA": "这是一个高可用或主机组 (HA) 集群，ID为{id}。{details}",
            "TRU": "这是一个存储单元 (TRU)，ID为{id}。{details}"
        }
        
        # 获取模板
        template = templates.get(node_type, "这是一个{type}节点，ID为{id}。{details}")
        
        # 格式化基本信息
        basic_info = template.format(
            type=node_type,
            id=node_data.get("id", "unknown"),
            details=""
        )
        
        # 添加属性
        properties = []
        for key, value in node_data.items():
            if key not in ["id", "type", "metrics_data", "log_data", "dynamics_data", "features"] and value:
                properties.append(f"{key}: {value}")
        
        property_text = ""
        if properties:
            property_text = "它具有以下属性：" + "，".join(properties) + "。"
        
        # 添加指标（如果可用）
        metrics_text = ""
        if "metrics_data" in node_data and node_data["metrics_data"]:
            try:
                metrics = json.loads(node_data["metrics_data"]) if isinstance(node_data["metrics_data"], str) else node_data["metrics_data"]
                if metrics:
                    metrics_items = []
                    for metric_name, metric_values in metrics.items():
                        if isinstance(metric_values, dict):
                            # 处理嵌套的指标数据
                            for sub_metric, sub_value in metric_values.items():
                                if isinstance(sub_value, list) and sub_value:
                                    metrics_items.append(f"{metric_name}.{sub_metric}: 最新值 {sub_value[-1]}, 平均值 {sum(sub_value)/len(sub_value):.2f}, 最大值 {max(sub_value)}, 最小值 {min(sub_value)}")
                                elif sub_value:
                                    metrics_items.append(f"{metric_name}.{sub_metric}: {sub_value}")
                        elif isinstance(metric_values, list) and metric_values:
                            # 处理时间序列指标数据
                            metrics_items.append(f"{metric_name}: 最新值 {metric_values[-1]}, 平均值 {sum(metric_values)/len(metric_values):.2f}, 最大值 {max(metric_values)}, 最小值 {min(metric_values)}")
                        elif metric_values:
                            metrics_items.append(f"{metric_name}: {metric_values}")
                    if metrics_items:
                        metrics_text = "它的指标包括：" + "，".join(metrics_items) + "。"
            except (json.JSONDecodeError, TypeError):
                pass
        
        # 添加日志数据（如果可用）
        log_text = ""
        if "log_data" in node_data and node_data["log_data"]:
            try:
                logs = json.loads(node_data["log_data"]) if isinstance(node_data["log_data"], str) else node_data["log_data"]
                if logs and isinstance(logs, dict) and "log_data" in logs:
                    log_entries = []
                    for node_id, log_list in logs["log_data"].items():
                        if isinstance(log_list, list) and log_list:
                            # 获取最近的日志条目
                            recent_logs = log_list[-3:] if len(log_list) > 3 else log_list
                            for entry in recent_logs:
                                if "timestamp" in entry and "status" in entry:
                                    status_info = []
                                    for status_key, status_val in entry["status"].items():
                                        status_info.append(f"{status_key}: {status_val}")
                                    log_entries.append(f"{entry['timestamp']} - {', '.join(status_info)}")
                    if log_entries:
                        log_text = "最近的日志记录：" + "；".join(log_entries) + "。"
            except (json.JSONDecodeError, TypeError):
                pass
        
        # 添加动态数据（如果可用）
        dynamics_text = ""
        if "dynamics_data" in node_data and node_data["dynamics_data"]:
            try:
                dynamics = json.loads(node_data["dynamics_data"]) if isinstance(node_data["dynamics_data"], str) else node_data["dynamics_data"]
                if dynamics:
                    dynamics_items = []
                    # 处理不同类型的动态数据
                    if isinstance(dynamics, dict):
                        for key, value in dynamics.items():
                            if key == "propagation_data" and isinstance(value, list):
                                # 处理传播数据
                                recent_propagations = value[-3:] if len(value) > 3 else value
                                for prop in recent_propagations:
                                    if "timestamp" in prop and "effects" in prop:
                                        effects = []
                                        for effect_key, effect_val in prop["effects"].items():
                                            if isinstance(effect_val, dict) and "probability" in effect_val:
                                                effects.append(f"{effect_key}(概率:{effect_val['probability']:.2f})")
                                        dynamics_items.append(f"{prop['timestamp']} - 影响: {', '.join(effects)}")
                            else:
                                dynamics_items.append(f"{key}: {value}")
                    if dynamics_items:
                        dynamics_text = "动态特征数据：" + "；".join(dynamics_items) + "。"
            except (json.JSONDecodeError, TypeError):
                pass
        
        # 添加连接
        connection_text = ""
        if connected_nodes:
            outgoing = [n for n in connected_nodes if n["direction"] == "outgoing"]
            incoming = [n for n in connected_nodes if n["direction"] == "incoming"]
            
            if outgoing:
                outgoing_by_type = {}
                for n in outgoing:
                    if n["type"] not in outgoing_by_type:
                        outgoing_by_type[n["type"]] = []
                    outgoing_by_type[n["type"]].append(n)
                
                outgoing_text_parts = []
                for type_name, nodes in outgoing_by_type.items():
                    node_text = "，".join([f"{n['name']}" for n in nodes[:5]])
                    if len(nodes) > 5:
                        node_text += f"等{len(nodes)}个"
                    outgoing_text_parts.append(f"{len(nodes)}个{type_name}节点（{node_text}）")
                
                connection_text += f"它连接到：{'，'.join(outgoing_text_parts)}。"
            
            if incoming:
                incoming_by_type = {}
                for n in incoming:
                    if n["type"] not in incoming_by_type:
                        incoming_by_type[n["type"]] = []
                    incoming_by_type[n["type"]].append(n)
                
                incoming_text_parts = []
                for type_name, nodes in incoming_by_type.items():
                    node_text = "，".join([f"{n['name']}" for n in nodes[:5]])
                    if len(nodes) > 5:
                        node_text += f"等{len(nodes)}个"
                    incoming_text_parts.append(f"{len(nodes)}个{type_name}节点（{node_text}）")
                
                connection_text += f"它被以下节点连接：{'，'.join(incoming_text_parts)}。"
        
        # 组合所有部分
        full_description = basic_info + " " + property_text + metrics_text + log_text + dynamics_text + connection_text
        
        # 添加节点类型特定的详细信息
        if node_type == "VM":
            vm_details = self._add_vm_specific_details(node_data)
            full_description += vm_details
        elif node_type == "HOST":
            host_details = self._add_host_specific_details(node_data)
            full_description += host_details
        elif node_type == "DC":
            dc_details = self._add_dc_specific_details(node_data)
            full_description += dc_details
        elif node_type == "TENANT":
            tenant_details = self._add_tenant_specific_details(node_data)
            full_description += tenant_details
        elif node_type == "NE":
            ne_details = self._add_ne_specific_details(node_data)
            full_description += ne_details
        elif node_type == "HA":
            ha_details = self._add_ha_specific_details(node_data)
            full_description += ha_details
        elif node_type == "TRU":
            tru_details = self._add_tru_specific_details(node_data)
            full_description += tru_details
        
        return full_description.strip()
    
    def _add_vm_specific_details(self, features: Dict) -> str:
        """添加VM特定的详细信息到描述中"""
        details = ""
        
        # 添加CPU、内存、磁盘使用情况（如果可用）
        if "metrics_data" in features and features["metrics_data"]:
            try:
                metrics = json.loads(features["metrics_data"]) if isinstance(features["metrics_data"], str) else features["metrics_data"]
                if metrics:
                    # CPU使用率
                    if "cpu_usage" in metrics and metrics["cpu_usage"]:
                        cpu = metrics["cpu_usage"][-1] if isinstance(metrics["cpu_usage"], list) and metrics["cpu_usage"] else metrics["cpu_usage"]
                        details += f" 该虚拟机的CPU使用率为{cpu}%。"
                        
                        # 添加CPU使用趋势分析
                        if isinstance(metrics["cpu_usage"], list) and len(metrics["cpu_usage"]) > 1:
                            cpu_trend = metrics["cpu_usage"][-1] - metrics["cpu_usage"][0]
                            if cpu_trend > 5:
                                details += f" CPU使用率呈上升趋势，增长了{cpu_trend:.2f}%。"
                            elif cpu_trend < -5:
                                details += f" CPU使用率呈下降趋势，降低了{abs(cpu_trend):.2f}%。"
                            else:
                                details += " CPU使用率保持稳定。"
                    
                    # 内存使用率
                    if "memory_usage" in metrics and metrics["memory_usage"]:
                        memory = metrics["memory_usage"][-1] if isinstance(metrics["memory_usage"], list) and metrics["memory_usage"] else metrics["memory_usage"]
                        details += f" 内存使用率为{memory}%。"
                        
                        # 添加内存使用趋势分析
                        if isinstance(metrics["memory_usage"], list) and len(metrics["memory_usage"]) > 1:
                            memory_trend = metrics["memory_usage"][-1] - metrics["memory_usage"][0]
                            if memory_trend > 5:
                                details += f" 内存使用率呈上升趋势，增长了{memory_trend:.2f}%。"
                            elif memory_trend < -5:
                                details += f" 内存使用率呈下降趋势，降低了{abs(memory_trend):.2f}%。"
                            else:
                                details += " 内存使用率保持稳定。"
                    
                    # 磁盘使用率
                    if "disk_usage" in metrics and metrics["disk_usage"]:
                        disk = metrics["disk_usage"][-1] if isinstance(metrics["disk_usage"], list) and metrics["disk_usage"] else metrics["disk_usage"]
                        details += f" 磁盘使用率为{disk}%。"
                    
                    # 网络流量
                    if "network_traffic" in metrics and metrics["network_traffic"]:
                        network = metrics["network_traffic"][-1] if isinstance(metrics["network_traffic"], list) and metrics["network_traffic"] else metrics["network_traffic"]
                        details += f" 网络流量为{network}Mbps。"
            except (json.JSONDecodeError, TypeError):
                pass
        
        # 添加日志状态信息
        if "log_data" in features and features["log_data"]:
            try:
                logs = json.loads(features["log_data"]) if isinstance(features["log_data"], str) else features["log_data"]
                if logs and isinstance(logs, dict) and "log_data" in logs:
                    # 分析日志状态
                    status_counts = {"cpu_log_status": {}, "memory_log_status": {}, "disk_log_status": {}, "network_log_status": {}}
                    
                    for node_id, log_list in logs["log_data"].items():
                        if isinstance(log_list, list):
                            # 只分析最近30条日志
                            recent_logs = log_list[-30:] if len(log_list) > 30 else log_list
                            for entry in recent_logs:
                                if "status" in entry:
                                    for status_key, status_val in entry["status"].items():
                                        if status_key in status_counts:
                                            if status_val not in status_counts[status_key]:
                                                status_counts[status_key][status_val] = 0
                                            status_counts[status_key][status_val] += 1
                    
                    # 生成状态摘要
                    status_summary = []
                    for status_key, counts in status_counts.items():
                        if counts:
                            # 找出最常见的状态
                            most_common_status = max(counts.items(), key=lambda x: x[1])
                            status_name = status_key.replace("_log_status", "").capitalize()
                            status_summary.append(f"{status_name}状态主要为{most_common_status[0]}（占比{most_common_status[1]/sum(counts.values())*100:.1f}%）")
                    
                    if status_summary:
                        details += f" 日志状态分析：{'，'.join(status_summary)}。"
            except (json.JSONDecodeError, TypeError):
                pass
        
        return details
    
    def _add_host_specific_details(self, features: Dict) -> str:
        """添加主机特定的详细信息到描述中"""
        details = ""
        
        # 添加主机特定信息
        if "business_type" in features:
            details += f" 该主机是{features['business_type']}业务的一部分。"
        
        # 添加硬件配置信息
        if "cpu_cores" in features:
            details += f" CPU核心数：{features['cpu_cores']}。"
        
        if "memory_size" in features:
            details += f" 内存大小：{features['memory_size']}GB。"
        
        if "disk_size" in features:
            details += f" 磁盘容量：{features['disk_size']}GB。"
        
        # 添加性能指标
        if "metrics_data" in features and features["metrics_data"]:
            try:
                metrics = json.loads(features["metrics_data"]) if isinstance(features["metrics_data"], str) else features["metrics_data"]
                if metrics:
                    # CPU负载
                    if "cpu_load" in metrics and metrics["cpu_load"]:
                        cpu_load = metrics["cpu_load"][-1] if isinstance(metrics["cpu_load"], list) and metrics["cpu_load"] else metrics["cpu_load"]
                        details += f" CPU负载：{cpu_load}。"
                    
                    # 内存使用情况
                    if "memory_used" in metrics and metrics["memory_used"]:
                        memory_used = metrics["memory_used"][-1] if isinstance(metrics["memory_used"], list) and metrics["memory_used"] else metrics["memory_used"]
                        details += f" 内存使用：{memory_used}GB。"
                    
                    # 磁盘I/O
                    if "disk_io" in metrics and metrics["disk_io"]:
                        disk_io = metrics["disk_io"][-1] if isinstance(metrics["disk_io"], list) and metrics["disk_io"] else metrics["disk_io"]
                        details += f" 磁盘I/O：{disk_io}MB/s。"
                    
                    # 网络吞吐量
                    if "network_throughput" in metrics and metrics["network_throughput"]:
                        network = metrics["network_throughput"][-1] if isinstance(metrics["network_throughput"], list) and metrics["network_throughput"] else metrics["network_throughput"]
                        details += f" 网络吞吐量：{network}Mbps。"
            except (json.JSONDecodeError, TypeError):
                pass
        
        # 添加虚拟机数量信息
        if "vm_count" in features:
            details += f" 该主机上运行了{features['vm_count']}个虚拟机。"
        
        return details
    
    def _add_dc_specific_details(self, features: Dict) -> str:
        """添加数据中心特定的详细信息到描述中"""
        details = ""
        
        # 添加数据中心特定信息
        if "business_type" in features and features["business_type"]:
            details += f" 该数据中心支持{features['business_type']}业务。"
        
        if "layer" in features and features["layer"]:
            details += f" 它位于{features['layer']}层。"
            
        # 添加数据中心容量信息
        if "capacity" in features:
            details += f" 数据中心容量为{features['capacity']}。"
            
        # 添加地理位置信息
        if "location" in features:
            details += f" 位于{features['location']}。"
            
        # 添加运行状态
        if "status" in features:
            details += f" 当前运行状态：{features['status']}。"
        
        return details
    
    def _add_tenant_specific_details(self, features: Dict) -> str:
        """添加租户特定的详细信息到描述中"""
        details = ""
        
        # 添加租户特定信息
        if "business_type" in features and features["business_type"]:
            details += f" 该租户提供{features['business_type']}服务。"
        
        if "layer" in features and features["layer"]:
            details += f" 它位于{features['layer']}层。"
            
        # 添加租户资源配额
        if "resource_quota" in features:
            details += f" 资源配额：{features['resource_quota']}。"
            
        # 添加租户创建时间
        if "create_time" in features:
            details += f" 创建于{features['create_time']}。"
            
        # 添加租户状态
        if "status" in features:
            details += f" 当前状态：{features['status']}。"
        
        return details
    
    def _add_ne_specific_details(self, features: Dict) -> str:
        """添加网元特定的详细信息到描述中"""
        details = ""
        
        # 添加网元特定信息
        if "business_type" in features and features["business_type"]:
            details += f" 该网元支持{features['business_type']}功能。"
        
        if "layer" in features and features["layer"]:
            details += f" 它位于{features['layer']}层。"
        
        if "layer_type" in features and features["layer_type"]:
            details += f" 它是{features['layer_type']}类型的网元。"
            
        # 添加连接和会话统计
        if "metrics_data" in features and features["metrics_data"]:
            try:
                metrics = json.loads(features["metrics_data"]) if isinstance(features["metrics_data"], str) else features["metrics_data"]
                if metrics and isinstance(metrics, dict):
                    for node_id, metric_list in metrics.items():
                        if isinstance(metric_list, list) and metric_list:
                            # 获取最新的指标
                            latest_metric = metric_list[-1]
                            if isinstance(latest_metric, dict) and "values" in latest_metric:
                                values = latest_metric["values"]
                                if "connection_count" in values:
                                    details += f" 连接数：{values['connection_count']}。"
                                if "session_count" in values:
                                    details += f" 会话数：{values['session_count']}。"
                                if "connection_success_rate" in values:
                                    details += f" 连接成功率：{values['connection_success_rate']:.2f}%。"
                                if "session_success_rate" in values:
                                    details += f" 会话成功率：{values['session_success_rate']:.2f}%。"
            except (json.JSONDecodeError, TypeError):
                pass
                
        # 添加日志状态分析
        if "log_data" in features and features["log_data"]:
            try:
                logs = json.loads(features["log_data"]) if isinstance(features["log_data"], str) else features["log_data"]
                if logs and isinstance(logs, dict) and "log_data" in logs:
                    # 分析日志状态
                    status_counts = {"connection_log_status": {}, "session_log_status": {}, 
                                    "session_success_log_status": {}, "connection_success_log_status": {}}
                    
                    for node_id, log_list in logs["log_data"].items():
                        if isinstance(log_list, list):
                            # 只分析最近30条日志
                            recent_logs = log_list[-30:] if len(log_list) > 30 else log_list
                            for entry in recent_logs:
                                if "status" in entry:
                                    for status_key, status_val in entry["status"].items():
                                        if status_key in status_counts:
                                            if status_val not in status_counts[status_key]:
                                                status_counts[status_key][status_val] = 0
                                            status_counts[status_key][status_val] += 1
                    
                    # 生成状态摘要
                    status_summary = []
                    for status_key, counts in status_counts.items():
                        if counts:
                            # 找出最常见的状态
                            most_common_status = max(counts.items(), key=lambda x: x[1])
                            status_name = status_key.replace("_log_status", "").replace("_", " ").capitalize()
                            status_summary.append(f"{status_name}主要为{most_common_status[0]}（占比{most_common_status[1]/sum(counts.values())*100:.1f}%）")
                    
                    if status_summary:
                        details += f" 日志状态分析：{'，'.join(status_summary)}。"
            except (json.JSONDecodeError, TypeError):
                pass
        
        return details
    
    def _add_ha_specific_details(self, features: Dict) -> str:
        """添加高可用集群特定的详细信息到描述中"""
        details = ""
        
        # 添加高可用集群特定信息
        if "metrics_data" in features and features["metrics_data"]:
            try:
                metrics = json.loads(features["metrics_data"]) if isinstance(features["metrics_data"], str) else features["metrics_data"]
                if metrics:
                    if "availability" in metrics and metrics["availability"]:
                        availability = metrics["availability"][-1] if isinstance(metrics["availability"], list) and metrics["availability"] else metrics["availability"]
                        details += f" 该高可用集群的可用性为{availability}%。"
                        
                        # 添加可用性趋势分析
                        if isinstance(metrics["availability"], list) and len(metrics["availability"]) > 1:
                            avail_trend = metrics["availability"][-1] - metrics["availability"][0]
                            if avail_trend > 1:
                                details += f" 可用性呈上升趋势，增长了{avail_trend:.2f}%。"
                            elif avail_trend < -1:
                                details += f" 可用性呈下降趋势，降低了{abs(avail_trend):.2f}%。"
                            else:
                                details += " 可用性保持稳定。"
                    
                    # 添加故障转移次数
                    if "failover_count" in metrics:
                        failover = metrics["failover_count"][-1] if isinstance(metrics["failover_count"], list) and metrics["failover_count"] else metrics["failover_count"]
                        details += f" 故障转移次数：{failover}次。"
                    
                    # 添加响应时间
                    if "response_time" in metrics:
                        response = metrics["response_time"][-1] if isinstance(metrics["response_time"], list) and metrics["response_time"] else metrics["response_time"]
                        details += f" 平均响应时间：{response}ms。"
            except (json.JSONDecodeError, TypeError):
                pass
        
        if "layer" in features and features["layer"]:
            details += f" 它位于{features['layer']}层。"
            
        # 添加集群节点数
        if "node_count" in features:
            details += f" 集群包含{features['node_count']}个节点。"
        
        return details
    
    def _add_tru_specific_details(self, features: Dict) -> str:
        """添加存储单元特定的详细信息到描述中"""
        details = ""
        
        # 添加存储单元特定信息
        if "metrics_data" in features and features["metrics_data"]:
            try:
                metrics = json.loads(features["metrics_data"]) if isinstance(features["metrics_data"], str) else features["metrics_data"]
                if metrics:
                    if "storage_capacity" in metrics and metrics["storage_capacity"]:
                        capacity = metrics["storage_capacity"][-1] if isinstance(metrics["storage_capacity"], list) and metrics["storage_capacity"] else metrics["storage_capacity"]
                        details += f" 该存储单元的容量为{capacity}GB。"
                    
                    if "storage_usage" in metrics and metrics["storage_usage"]:
                        usage = metrics["storage_usage"][-1] if isinstance(metrics["storage_usage"], list) and metrics["storage_usage"] else metrics["storage_usage"]
                        details += f" 存储使用率为{usage}%。"
                        
                        # 添加存储使用趋势分析
                        if isinstance(metrics["storage_usage"], list) and len(metrics["storage_usage"]) > 1:
                            usage_trend = metrics["storage_usage"][-1] - metrics["storage_usage"][0]
                            if usage_trend > 5:
                                details += f" 存储使用率呈上升趋势，增长了{usage_trend:.2f}%。"
                            elif usage_trend < -5:
                                details += f" 存储使用率呈下降趋势，降低了{abs(usage_trend):.2f}%。"
                            else:
                                details += " 存储使用率保持稳定。"
                    
                    # 添加I/O性能指标
                    if "iops" in metrics:
                        iops = metrics["iops"][-1] if isinstance(metrics["iops"], list) and metrics["iops"] else metrics["iops"]
                        details += f" IOPS：{iops}。"
                    
                    if "throughput" in metrics:
                        throughput = metrics["throughput"][-1] if isinstance(metrics["throughput"], list) and metrics["throughput"] else metrics["throughput"]
                        details += f" 吞吐量：{throughput}MB/s。"
                    
                    if "latency" in metrics:
                        latency = metrics["latency"][-1] if isinstance(metrics["latency"], list) and metrics["latency"] else metrics["latency"]
                        details += f" 延迟：{latency}ms。"
            except (json.JSONDecodeError, TypeError):
                pass
        
        if "layer" in features and features["layer"]:
            details += f" 它位于{features['layer']}层。"
            
        # 添加存储类型
        if "storage_type" in features:
            details += f" 存储类型：{features['storage_type']}。"
        
        return details
    
    def _create_pairs(self) -> List[Dict]:
        """Create graph-text pairs for training"""
        pairs = []
        for node_id, node in tqdm(self.nodes.items(), desc="Creating pairs"):
            if node_id in self.text_descriptions:
                # Extract subgraph
                subgraph = self._extract_subgraph(node_id)
                
                # Create pair
                pair = {
                    'node_id': node_id,
                    'text': self.text_descriptions[node_id],
                    'subgraph': subgraph,
                    'node_type': node["type"],
                    'is_negative': False  # 标记为正样本
                }
                pairs.append(pair)
        
        # Apply data augmentation if enabled
        if self.data_augmentation and self.split == "train":
            augmented_pairs = self._apply_data_augmentation(pairs)
            pairs.extend(augmented_pairs)
            self.log_info(f"Added {len(augmented_pairs)} augmented pairs")
        
        return pairs
    
    def _extract_subgraph(self, center_node_id: str) -> Dict:
        """Extract local subgraph centered at given node"""
        # Initialize subgraph
        subgraph_nodes = {center_node_id: self.nodes[center_node_id]}
        subgraph_edges = {}
        
        # 获取节点类型
        node_type = self.nodes[center_node_id]["type"]
        
        # 根据节点类型自适应调整子图大小
        max_node_size = self.max_node_size
        max_edge_size = self.max_edge_size
        
        if self.adaptive_subgraph_size:
            # 为稀有节点类型增加子图大小
            if node_type in ["DC", "HA", "TRU"]:
                max_node_size = min(200, self.max_node_size * 2)
                max_edge_size = min(200, self.max_edge_size * 2)
            # 为常见节点类型减小子图大小
            elif node_type in ["VM"]:
                max_node_size = max(20, int(self.max_node_size * 0.8))
                max_edge_size = max(20, int(self.max_edge_size * 0.8))
        
        # 计算节点重要性分数
        node_importance = self._calculate_node_importance()
        
        # Add 1-hop neighbors
        neighbor_scores = []
        for edge_key, edge in self.edges.items():
            if edge["source"] == center_node_id:
                target_id = edge["target"]
                if target_id in self.nodes:
                    importance = node_importance.get(target_id, 1.0)
                    neighbor_scores.append((target_id, edge_key, importance))
            elif edge["target"] == center_node_id:
                source_id = edge["source"]
                if source_id in self.nodes:
                    importance = node_importance.get(source_id, 1.0)
                    neighbor_scores.append((source_id, edge_key, importance))
        
        # 按重要性排序邻居节点
        neighbor_scores.sort(key=lambda x: x[2], reverse=True)
        
        # 添加重要的邻居节点
        for neighbor_id, edge_key, _ in neighbor_scores:
            if len(subgraph_nodes) >= max_node_size or len(subgraph_edges) >= max_edge_size:
                break
            
            subgraph_nodes[neighbor_id] = self.nodes[neighbor_id]
            
            # 找到对应的边
            for e_key, edge in self.edges.items():
                if (edge["source"] == center_node_id and edge["target"] == neighbor_id) or \
                   (edge["target"] == center_node_id and edge["source"] == neighbor_id):
                    subgraph_edges[e_key] = edge
                    break
        
        # Add 2-hop neighbors (limited by max_node_size and max_edge_size)
        if len(subgraph_nodes) < max_node_size and len(subgraph_edges) < max_edge_size:
            two_hop_neighbors = []
            
            for node_id in list(subgraph_nodes.keys()):
                if node_id != center_node_id:  # Skip center node (already processed)
                    for edge_key, edge in self.edges.items():
                        if edge["source"] == node_id and edge["target"] not in subgraph_nodes:
                            target_id = edge["target"]
                            if target_id in self.nodes:
                                importance = node_importance.get(target_id, 1.0)
                                two_hop_neighbors.append((target_id, edge_key, importance))
                        elif edge["target"] == node_id and edge["source"] not in subgraph_nodes:
                            source_id = edge["source"]
                            if source_id in self.nodes:
                                importance = node_importance.get(source_id, 1.0)
                                two_hop_neighbors.append((source_id, edge_key, importance))
            
            # 按重要性排序二阶邻居
            two_hop_neighbors.sort(key=lambda x: x[2], reverse=True)
            
            # 添加重要的二阶邻居
            for neighbor_id, edge_key, _ in two_hop_neighbors:
                if len(subgraph_nodes) >= max_node_size or len(subgraph_edges) >= max_edge_size:
                    break
                
                subgraph_nodes[neighbor_id] = self.nodes[neighbor_id]
                
                # 找到对应的边
                edge = self.edges.get(edge_key)
                if edge:
                    subgraph_edges[edge_key] = edge
        
        return {
            'nodes': subgraph_nodes,
            'edges': subgraph_edges,
            'center_node_id': center_node_id
        }
    
    def _calculate_node_importance(self) -> Dict[str, float]:
        """计算节点重要性分数"""
        # 初始化节点重要性
        importance = {}
        
        # 计算节点的度
        node_degrees = defaultdict(int)
        for edge in self.edges.values():
            node_degrees[edge["source"]] += 1
            node_degrees[edge["target"]] += 1
        
        # 计算节点类型的稀有度
        node_type_counts = Counter([node["type"] for node in self.nodes.values()])
        type_rarity = {
            node_type: 1.0 / (count + 1) 
            for node_type, count in node_type_counts.items()
        }
        
        # 计算节点重要性分数
        for node_id, node in self.nodes.items():
            # 基于度和节点类型稀有度的重要性
            degree = node_degrees.get(node_id, 0)
            rarity = type_rarity.get(node["type"], 1.0)
            
            # 重要性分数 = 度 * 稀有度
            importance[node_id] = degree * rarity
            
            # 为特定类型的节点增加重要性
            if node["type"] in ["DC", "HA", "TRU"]:
                importance[node_id] *= 2.0
        
        # 归一化重要性分数
        max_importance = max(importance.values()) if importance else 1.0
        for node_id in importance:
            importance[node_id] /= max_importance
        
        return importance
    
    def _apply_dataset_split(self) -> List[Dict]:
        """应用数据集分割"""
        global _dataset_log_lock
        
        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {self.split}")
        
        # 按节点类型分组
        pairs_by_type = defaultdict(list)
        for pair in self.pairs:
            pairs_by_type[pair["node_type"]].append(pair)
        
        # 为每种节点类型应用分割
        split_pairs = []
        for node_type, type_pairs in pairs_by_type.items():
            # 打乱顺序
            random.shuffle(type_pairs)
            
            # 计算分割点
            train_end = int(len(type_pairs) * self.split_ratio["train"])
            val_end = train_end + int(len(type_pairs) * self.split_ratio["val"])
            
            # 根据当前分割选择对应的数据
            if self.split == "train":
                split_pairs.extend(type_pairs[:train_end])
            elif self.split == "val":
                split_pairs.extend(type_pairs[train_end:val_end])
            else:  # test
                split_pairs.extend(type_pairs[val_end:])
        
        with _dataset_log_lock:
            self.log_info(f"Applied {self.split} split: {len(split_pairs)}/{len(self.pairs)} pairs")
        return split_pairs
    
    def _balance_node_types(self) -> List[Dict]:
        """平衡不同节点类型的样本数量"""
        global _dataset_log_lock
        
        # 按节点类型分组
        pairs_by_type = defaultdict(list)
        for pair in self.pairs:
            pairs_by_type[pair["node_type"]].append(pair)
        
        # 找出最多样本的类型
        max_count = max(len(pairs) for pairs in pairs_by_type.values())
        
        # 平衡各类型样本数量
        balanced_pairs = []
        for node_type, type_pairs in pairs_by_type.items():
            # 对于样本较少的类型，通过过采样增加样本
            if len(type_pairs) < max_count:
                # 计算需要复制的次数
                repeat_factor = math.ceil(max_count / len(type_pairs))
                
                # 复制样本
                augmented_pairs = []
                for _ in range(repeat_factor):
                    # 每次复制时稍微打乱顺序
                    shuffled = type_pairs.copy()
                    random.shuffle(shuffled)
                    augmented_pairs.extend(shuffled)
                
                # 截取所需数量
                balanced_pairs.extend(augmented_pairs[:max_count])
                with _dataset_log_lock:
                    self.log_info(f"Balanced {node_type}: {len(type_pairs)} -> {max_count} samples")
            else:
                # 对于样本较多的类型，保持不变
                balanced_pairs.extend(type_pairs)
        
        return balanced_pairs
    
    def _generate_negative_samples(self) -> List[Dict]:
        """生成负样本"""
        global _dataset_log_lock
        
        negative_pairs = []
        positive_count = len(self.pairs)
        target_negative_count = int(positive_count * self.negative_sample_ratio)
        
        with _dataset_log_lock:
            self.log_info(f"Generating {target_negative_count} negative samples...")
        
        # 按节点类型分组
        pairs_by_type = defaultdict(list)
        for pair in self.pairs:
            pairs_by_type[pair["node_type"]].append(pair)
        
        # 为每种节点类型生成负样本
        for node_type, type_pairs in pairs_by_type.items():
            # 计算该类型需要生成的负样本数量
            type_ratio = len(type_pairs) / positive_count
            type_negative_count = int(target_negative_count * type_ratio)
            
            # 生成三种类型的负样本
            
            # 1. 文本-图不匹配负样本
            text_graph_mismatch = self._generate_text_graph_mismatch(type_pairs, int(type_negative_count * 0.4))
            negative_pairs.extend(text_graph_mismatch)
            
            # 2. 困难负样本（同类型不同节点）
            hard_negatives = self._generate_hard_negatives(type_pairs, int(type_negative_count * 0.4))
            negative_pairs.extend(hard_negatives)
            
            # 3. 子图扰动负样本
            subgraph_perturbation = self._generate_subgraph_perturbation(type_pairs, int(type_negative_count * 0.2))
            negative_pairs.extend(subgraph_perturbation)
        
        return negative_pairs
    
    def _generate_text_graph_mismatch(self, pairs: List[Dict], count: int) -> List[Dict]:
        """生成文本-图不匹配的负样本"""
        if len(pairs) < 2:
            return []
        
        negative_pairs = []
        for _ in range(min(count, len(pairs))):
            # 随机选择一个样本
            pair = random.choice(pairs)
            
            # 随机选择一个不同的文本
            other_pairs = [p for p in pairs if p["node_id"] != pair["node_id"]]
            if not other_pairs:
                continue
            
            other_pair = random.choice(other_pairs)
            
            # 创建负样本：使用原图和不匹配的文本
            negative_pair = {
                'node_id': pair["node_id"],
                'text': other_pair["text"],
                'subgraph': pair["subgraph"],
                'node_type': pair["node_type"],
                'is_negative': True,
                'negative_type': 'text_graph_mismatch'
            }
            
            negative_pairs.append(negative_pair)
        
        return negative_pairs
    
    def _generate_hard_negatives(self, pairs: List[Dict], count: int) -> List[Dict]:
        """生成困难负样本（同类型不同节点）"""
        if len(pairs) < 2:
            return []
        
        # 计算节点相似度
        node_similarities = {}
        for i, pair1 in enumerate(pairs):
            for j, pair2 in enumerate(pairs):
                if i >= j:
                    continue
                
                # 计算两个节点的相似度
                similarity = self._calculate_node_similarity(pair1, pair2)
                node_similarities[(pair1["node_id"], pair2["node_id"])] = similarity
        
        # 按相似度排序
        sorted_similarities = sorted(
            node_similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 生成困难负样本
        negative_pairs = []
        for (node_id1, node_id2), similarity in sorted_similarities[:count]:
            # 找到对应的样本
            pair1 = next(p for p in pairs if p["node_id"] == node_id1)
            pair2 = next(p for p in pairs if p["node_id"] == node_id2)
            
            # 创建负样本：使用相似节点的文本和图
            negative_pair = {
                'node_id': pair1["node_id"],
                'text': pair2["text"],
                'subgraph': pair1["subgraph"],
                'node_type': pair1["node_type"],
                'is_negative': True,
                'negative_type': 'hard_negative',
                'similarity': similarity
            }
            
            negative_pairs.append(negative_pair)
        
        return negative_pairs
    
    def _calculate_node_similarity(self, pair1: Dict, pair2: Dict) -> float:
        """计算两个节点的相似度"""
        # 简单实现：基于共同邻居的比例
        nodes1 = set(pair1["subgraph"]["nodes"].keys())
        nodes2 = set(pair2["subgraph"]["nodes"].keys())
        
        # 计算Jaccard相似度
        intersection = len(nodes1.intersection(nodes2))
        union = len(nodes1.union(nodes2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _generate_subgraph_perturbation(self, pairs: List[Dict], count: int) -> List[Dict]:
        """生成子图扰动负样本"""
        negative_pairs = []
        
        for _ in range(min(count, len(pairs))):
            # 随机选择一个样本
            pair = random.choice(pairs)
            
            # 复制子图
            perturbed_subgraph = {
                'nodes': pair["subgraph"]["nodes"].copy(),
                'edges': pair["subgraph"]["edges"].copy(),
                'center_node_id': pair["subgraph"]["center_node_id"]
            }
            
            # 扰动子图：随机删除一些节点和边，并添加一些不相关的节点和边
            self._perturb_subgraph(perturbed_subgraph)
            
            # 创建负样本
            negative_pair = {
                'node_id': pair["node_id"],
                'text': pair["text"],
                'subgraph': perturbed_subgraph,
                'node_type': pair["node_type"],
                'is_negative': True,
                'negative_type': 'subgraph_perturbation'
            }
            
            negative_pairs.append(negative_pair)
        
        return negative_pairs
    
    def _perturb_subgraph(self, subgraph: Dict) -> None:
        """扰动子图结构"""
        # 保留中心节点
        center_node_id = subgraph["center_node_id"]
        
        # 随机删除30%的非中心节点
        nodes_to_remove = []
        for node_id in subgraph["nodes"]:
            if node_id != center_node_id and random.random() < 0.3:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            if node_id in subgraph["nodes"]:
                del subgraph["nodes"][node_id]
        
        # 删除与已删除节点相连的边
        edges_to_remove = []
        for edge_key, edge in subgraph["edges"].items():
            if edge["source"] in nodes_to_remove or edge["target"] in nodes_to_remove:
                edges_to_remove.append(edge_key)
        
        for edge_key in edges_to_remove:
            if edge_key in subgraph["edges"]:
                del subgraph["edges"][edge_key]
        
        # 添加一些随机节点（从全局节点中选择）
        available_nodes = [
            node_id for node_id in self.nodes
            if node_id not in subgraph["nodes"] and node_id != center_node_id
        ]
        
        if available_nodes:
            num_to_add = min(5, len(available_nodes))
            for _ in range(num_to_add):
                if not available_nodes:
                    break
                
                random_node_id = random.choice(available_nodes)
                available_nodes.remove(random_node_id)
                
                subgraph["nodes"][random_node_id] = self.nodes[random_node_id]
        
        # 添加一些随机边（从全局边中选择）
        available_edges = [
            edge_key for edge_key, edge in self.edges.items()
            if edge_key not in subgraph["edges"] and
            edge["source"] in subgraph["nodes"] and
            edge["target"] in subgraph["nodes"]
        ]
        
        if available_edges:
            num_to_add = min(5, len(available_edges))
            for _ in range(num_to_add):
                if not available_edges:
                    break
                
                random_edge_key = random.choice(available_edges)
                available_edges.remove(random_edge_key)
                
                subgraph["edges"][random_edge_key] = self.edges[random_edge_key]
    
    def _apply_data_augmentation(self, pairs: List[Dict]) -> List[Dict]:
        """Apply data augmentation to create more training pairs"""
        augmented_pairs = []
        
        for pair in tqdm(pairs, desc="Applying data augmentation"):
            # 1. Text augmentation: mask random words
            masked_text = self._mask_random_words(pair["text"])
            if masked_text != pair["text"]:
                augmented_pair = pair.copy()
                augmented_pair["text"] = masked_text
                augmented_pairs.append(augmented_pair)
            
            # 2. Text augmentation: synonym replacement
            synonym_text = self._replace_with_synonyms(pair["text"])
            if synonym_text != pair["text"]:
                augmented_pair = pair.copy()
                augmented_pair["text"] = synonym_text
                augmented_pairs.append(augmented_pair)
            
            # 3. Subgraph augmentation: random node dropout
            augmented_subgraph = self._random_node_dropout(pair["subgraph"])
            if augmented_subgraph != pair["subgraph"]:
                augmented_pair = pair.copy()
                augmented_pair["subgraph"] = augmented_subgraph
                augmented_pairs.append(augmented_pair)
            
            # 4. Subgraph augmentation: edge perturbation
            perturbed_subgraph = self._edge_perturbation(pair["subgraph"])
            if perturbed_subgraph != pair["subgraph"]:
                augmented_pair = pair.copy()
                augmented_pair["subgraph"] = perturbed_subgraph
                augmented_pairs.append(augmented_pair)
        
        return augmented_pairs
    
    def _mask_random_words(self, text: str, mask_prob: float = 0.15) -> str:
        """Mask random words in the text"""
        words = text.split()
        masked_words = []
        
        for word in words:
            if random.random() < mask_prob:
                masked_words.append("[MASK]")
            else:
                masked_words.append(word)
        
        return " ".join(masked_words)
    
    def _replace_with_synonyms(self, text: str, replace_prob: float = 0.1) -> str:
        """Replace words with synonyms"""
        # 简单的同义词替换（实际应用中可以使用更复杂的同义词库）
        synonyms = {
            "数据中心": ["数据中心", "DC", "机房"],
            "租户": ["租户", "客户", "用户"],
            "网元": ["网元", "NE", "网络设备"],
            "虚拟机": ["虚拟机", "VM", "云主机"],
            "主机": ["主机", "服务器", "物理机"],
            "高可用": ["高可用", "主机组", "HA", "容灾"],
            "存储": ["存储", "存储单元", "TRU"],
            "CPU": ["CPU", "处理器", "计算资源"],
            "内存": ["内存", "RAM", "存储资源"],
            "磁盘": ["磁盘", "硬盘", "存储设备"],
            "网络": ["网络", "网络连接", "通信"],
            "状态": ["状态", "运行状态", "工作状态"],
            "告警": ["告警", "警报", "异常提示"],
            "日志": ["日志", "记录", "系统记录"],
            "指标": ["指标", "性能指标", "监控指标"]
        }
        
        words = text.split()
        replaced_words = []
        
        for word in words:
            if random.random() < replace_prob:
                # 查找可能的同义词
                for key, values in synonyms.items():
                    if key in word:
                        # 随机选择一个同义词替换
                        replacement = random.choice(values)
                        word = word.replace(key, replacement)
                        break
            
            replaced_words.append(word)
        
        return " ".join(replaced_words)
    
    def _random_node_dropout(self, subgraph: Dict, dropout_prob: float = 0.2) -> Dict:
        """Randomly drop nodes from the subgraph (except the center node)"""
        center_node_id = subgraph["center_node_id"]
        nodes = subgraph["nodes"].copy()
        edges = subgraph["edges"].copy()
        
        # Get non-center nodes
        non_center_nodes = [node_id for node_id in nodes.keys() if node_id != center_node_id]
        
        # Randomly select nodes to drop
        nodes_to_drop = []
        for node_id in non_center_nodes:
            if random.random() < dropout_prob:
                nodes_to_drop.append(node_id)
        
        # Drop selected nodes
        for node_id in nodes_to_drop:
            if node_id in nodes:
                del nodes[node_id]
        
        # Drop edges connected to dropped nodes
        edges_to_drop = []
        for edge_key, edge in edges.items():
            if edge["source"] in nodes_to_drop or edge["target"] in nodes_to_drop:
                edges_to_drop.append(edge_key)
        
        for edge_key in edges_to_drop:
            if edge_key in edges:
                del edges[edge_key]
        
        return {
            'nodes': nodes,
            'edges': edges,
            'center_node_id': center_node_id
        }
    
    def _edge_perturbation(self, subgraph: Dict, perturb_prob: float = 0.1) -> Dict:
        """Perturb edges in the subgraph"""
        nodes = subgraph["nodes"].copy()
        edges = subgraph["edges"].copy()
        center_node_id = subgraph["center_node_id"]
        
        # 随机删除一些边
        edges_to_remove = []
        for edge_key, edge in edges.items():
            # 保留与中心节点相连的边
            if edge["source"] == center_node_id or edge["target"] == center_node_id:
                continue
                
            if random.random() < perturb_prob:
                edges_to_remove.append(edge_key)
        
        for edge_key in edges_to_remove:
            if edge_key in edges:
                del edges[edge_key]
        
        # 随机添加一些边
        node_ids = list(nodes.keys())
        for _ in range(int(len(edges) * perturb_prob)):
            if len(node_ids) < 2:
                break
                
            # 随机选择两个节点
            source_id = random.choice(node_ids)
            target_id = random.choice([n for n in node_ids if n != source_id])
            
            # 检查边是否已存在
            edge_exists = False
            for edge in edges.values():
                if (edge["source"] == source_id and edge["target"] == target_id) or \
                   (edge["source"] == target_id and edge["target"] == source_id):
                    edge_exists = True
                    break
            
            if not edge_exists:
                # 创建新边
                edge_key = f"{source_id}->{target_id}"
                edges[edge_key] = {
                    "id": edge_key,
                    "source": source_id,
                    "target": target_id,
                    "type": random.choice(self.edge_types),
                    "features": {}  # 简化的特征
                }
        
        return {
            'nodes': nodes,
            'edges': edges,
            'center_node_id': center_node_id
        }
    
    def __len__(self) -> int:
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]
        return {
            'node_id': pair['node_id'],
            'text': pair['text'],
            'subgraph': pair['subgraph'],
            'node_type': pair['node_type']
        }
        
    def __getstate__(self):
        """自定义序列化方法，排除无法序列化的Neo4j连接对象"""
        state = self.__dict__.copy()
        # 移除无法序列化的对象
        if 'graph_manager' in state:
            del state['graph_manager']
        if 'feature_extractor' in state:
            del state['feature_extractor']
        return state
    
    def __setstate__(self, state):
        """自定义反序列化方法"""
        self.__dict__.update(state)
        # 注意：反序列化后，graph_manager和feature_extractor将为None
        # 如果需要使用这些对象，需要在加载后重新初始化它们
        self.graph_manager = None
        self.feature_extractor = None

    @classmethod
    def from_samples(cls, samples: List[Dict], config: Dict):
        """
        从预生成的样本创建数据集
        
        Args:
            samples: 样本列表
            config: 数据集配置
            
        Returns:
            GraphTextDataset实例
        """
        dataset = cls.__new__(cls)
        dataset.samples = samples
        dataset.config = config
        dataset.node_types = config['node_types']
        dataset.edge_types = config['edge_types']
        dataset.max_text_length = config.get('max_text_length', 512)
        dataset.max_node_size = config.get('max_node_size', 100)
        dataset.max_edge_size = config.get('max_edge_size', 200)
        
        return dataset
        
    @classmethod
    def from_dataset(cls, dataset_path: str, config: Dict):
        """
        从预生成的dataset文件创建数据集,使用分块加载方式
        
        Args:
            dataset_path: 数据集文件路径
            config: 数据集配置
            
        Returns:
            GraphTextDataset实例
        """
        import torch
        
        dataset_instance = cls.__new__(cls)
        dataset_instance.config = config
        dataset_instance.node_types = config['node_types']
        dataset_instance.edge_types = config['edge_types']
        dataset_instance.max_text_length = config.get('max_text_length', 512)
        dataset_instance.max_node_size = config.get('max_node_size', 100)
        dataset_instance.max_edge_size = config.get('max_edge_size', 200)
        
        # 加载数据集
        data = torch.load(dataset_path)
        dataset_instance.pairs = data['pairs']
        dataset_instance.dataset_size = len(dataset_instance.pairs)
        
        # 设置其他属性
        dataset_instance.graph_manager = None
        dataset_instance.feature_extractor = None
        
        return dataset_instance
    
    def __len__(self):
        """返回数据集大小"""
        return self.dataset_size
    
    def __getitem__(self, idx):
        """获取数据项"""
        pair = self.pairs[idx]
        return {
            'node_id': pair['node_id'],
            'text': pair['text'],
            'subgraph': pair['subgraph'],
            'node_type': pair['node_type']
        }
    
    @classmethod
    def collate_fn(cls, batch):
        """
        批处理函数，将多个样本组合成一个批次
        
        Args:
            batch: 样本列表
            
        Returns:
            批处理后的数据
        """
        import torch
        from transformers import AutoTokenizer
        
        # 过滤掉空样本
        batch = [b for b in batch if b]
        
        if not batch:
            return {}
        
        # 提取文本和图特征
        texts = [item.get('text', '') for item in batch]
        node_ids = [item.get('node_id', '') for item in batch]
        
        # 使用tokenizer处理文本
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        text_encodings = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 提取节点特征
        node_features_list = []
        for item in batch:
            if 'node_features' in item and item['node_features'] is not None:
                if isinstance(item['node_features'], list):
                    node_features_list.append(torch.tensor(item['node_features'], dtype=torch.float))
                else:
                    node_features_list.append(torch.tensor(item['node_features'], dtype=torch.float))
            else:
                node_features_list.append(torch.randn(256))
        
        # 堆叠节点特征
        node_features = torch.stack(node_features_list)
        
        # 处理边特征和边索引
        batch_size = len(batch)
        edge_indices = []
        edge_features_list = []
        edge_types = []
        
        # 为每个样本创建边索引和特征
        for i, item in enumerate(batch):
            subgraph = item.get('subgraph', {})
            edges = subgraph.get('edges', [])
            
            # 为每条边创建索引和特征
            for edge in edges:
                src_idx = i
                dst_idx = i
                edge_indices.append([src_idx, dst_idx])
                
                edge_id = edge.get('id')
                edge_type = edge.get('type', 'default')
                
                if 'edge_features' in item and item['edge_features'] is not None and edge_id in item['edge_features']:
                    edge_feat = item['edge_features'][edge_id]
                    if isinstance(edge_feat, list):
                        edge_features_list.append(torch.tensor(edge_feat, dtype=torch.float))
                    else:
                        edge_features_list.append(torch.tensor(edge_feat, dtype=torch.float))
                else:
                    edge_features_list.append(torch.randn(64))
                
                edge_types.append(edge_type)
        
        # 如果没有边，创建一个默认边
        if not edge_indices:
            for i in range(batch_size):
                edge_indices.append([i, i])
                edge_features_list.append(torch.randn(64))
                edge_types.append('default')
        
        # 转换为张量
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_features = torch.stack(edge_features_list) if edge_features_list else torch.randn(1, 64)
        
        # 创建边类型字典
        edge_indices_dict = {'default': edge_index}
        edge_features_dict = {'default': edge_features}
        
        # 组合成批次
        batch_data = {
            'input_ids': text_encodings.input_ids,
            'attention_mask': text_encodings.attention_mask,
            'token_type_ids': text_encodings.token_type_ids if hasattr(text_encodings, 'token_type_ids') else None,
            'node_features': node_features,
            'edge_features': edge_features,
            'edge_index': edge_index,
            'edge_indices_dict': edge_indices_dict,
            'edge_features_dict': edge_features_dict,
            'batch': torch.arange(batch_size),
            'node_ids': node_ids,
            'texts': texts
        }
        
        return batch_data