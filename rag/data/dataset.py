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


import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
import random
from tqdm import tqdm

from ..feature_extractor import FeatureExtractor
from preprocess.utils.neo4j_graph_manager import Neo4jGraphManager
from ..utils.logging import LoggerMixin

logger = logging.getLogger(__name__)

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
        split: str = "train"
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
            split: 数据集分割（"train", "val", or "test"）
        """
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
        self.split = split
        
        # 记载图数据
        self.nodes, self.edges = self._load_graph_data()
        
        # 生成文本描述
        self.text_descriptions = self._generate_text_descriptions()
        
        # 创建图-文对
        self.pairs = self._create_pairs()
        
        # 记录数据集统计信息
        self.log_info(f"Created {split} dataset with {len(self.pairs)} graph-text pairs")
        self.log_info(f"Node types: {self.node_types}")
        self.log_info(f"Edge types: {self.edge_types}")
        
    def _load_graph_data(self) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """从Neo4j加载节点和边及其特征"""
        self.log_info("从Neo4j加载图数据...")
        
        # 按类型获取节点及其特征
        nodes = {}
        for node_type in tqdm(self.node_types, desc="加载节点"):
            # 从Neo4j获取节点ID
            query = f"""
            MATCH (n:{node_type})
            RETURN id(n) as node_id, n.id as node_name
            """
            with self.graph_manager._driver.session() as session:
                result = session.run(query)
                for record in result:
                    node_id = record["node_id"]
                    node_name = record["node_name"]
                    # 使用特征提取器提取节点特征
                    features = self.feature_extractor.extract_node_features(node_id, node_type)
                    nodes[str(node_id)] = {
                        "id": str(node_id),
                        "name": node_name,
                        "type": node_type,
                        "features": features
                    }
        
        # 按类型获取边及其特征
        edges = {}
        for edge_type in tqdm(self.edge_types, desc="加载边"):
            # 从Neo4j获取边信息
            query = f"""
            MATCH (source)-[r:{edge_type}]->(target)
            RETURN id(source) as source_id, id(target) as target_id, id(r) as edge_id
            """
            with self.graph_manager._driver.session() as session:
                result = session.run(query)
                for record in result:
                    source_id = record["source_id"]
                    target_id = record["target_id"]
                    edge_id = record["edge_id"]
                    # 使用特征提取器提取边特征
                    features = self.feature_extractor.extract_edge_features(
                        source_id, target_id, edge_type
                    )
                    edge_key = f"{source_id}->{target_id}"
                    edges[edge_key] = {
                        "id": str(edge_id),
                        "source": str(source_id),
                        "target": str(target_id),
                        "type": edge_type,
                        "features": features
                    }
        
        self.log_info(f"Loaded {len(nodes)} nodes and {len(edges)} edges")
        return nodes, edges
    
    def _generate_text_descriptions(self) -> Dict[str, str]:
        """生成节点文本描述""" 
        self.log_info("生成节点文本描述...")
        
        descriptions = {}
        for node_id, node in tqdm(self.nodes.items(), desc="生成描述"):
            # 从特征获取节点属性
            node_type = node["type"]
            node_features = node["features"]
            
            # 获取连接的节点
            connected_nodes = self._get_connected_nodes(node_id)
            
            # 使用节点类型、特征和连接生成描述
            desc = self._format_description(node_type, node_features, connected_nodes)
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
        features: Dict,
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
            "HA": "这是一个高可用性 (HA) 集群，ID为{id}。{details}",
            "TRU": "这是一个存储单元 (TRU)，ID为{id}。{details}"
        }
        
        # 获取模板
        template = templates.get(node_type, "这是一个{type}节点，ID为{id}。{details}")
        
        # 格式化基本信息
        basic_info = template.format(
            type=node_type,
            id=features.get("id", "unknown"),
            details=""
        )
        
        # 添加属性
        properties = []
        for key, value in features.items():
            if key not in ["id", "type", "metrics_data", "log_data", "dynamics_data"] and value:
                properties.append(f"{key}: {value}")
        
        property_text = ""
        if properties:
            property_text = "它具有以下属性：" + "，".join(properties) + "。"
        
        # 添加指标（如果可用）
        metrics_text = ""
        if "metrics_data" in features and features["metrics_data"]:
            try:
                metrics = json.loads(features["metrics_data"]) if isinstance(features["metrics_data"], str) else features["metrics_data"]
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
        if "log_data" in features and features["log_data"]:
            try:
                logs = json.loads(features["log_data"]) if isinstance(features["log_data"], str) else features["log_data"]
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
        if "dynamics_data" in features and features["dynamics_data"]:
            try:
                dynamics = json.loads(features["dynamics_data"]) if isinstance(features["dynamics_data"], str) else features["dynamics_data"]
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
            vm_details = self._add_vm_specific_details(features)
            full_description += vm_details
        elif node_type == "HOST":
            host_details = self._add_host_specific_details(features)
            full_description += host_details
        elif node_type == "DC":
            dc_details = self._add_dc_specific_details(features)
            full_description += dc_details
        elif node_type == "TENANT":
            tenant_details = self._add_tenant_specific_details(features)
            full_description += tenant_details
        elif node_type == "NE":
            ne_details = self._add_ne_specific_details(features)
            full_description += ne_details
        elif node_type == "HA":
            ha_details = self._add_ha_specific_details(features)
            full_description += ha_details
        elif node_type == "TRU":
            tru_details = self._add_tru_specific_details(features)
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
        self.log_info("Creating graph-text pairs...")
        
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
                    'node_type': node["type"]
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
        
        # Add 1-hop neighbors
        for edge_key, edge in self.edges.items():
            if edge["source"] == center_node_id:
                target_id = edge["target"]
                if target_id in self.nodes:
                    subgraph_nodes[target_id] = self.nodes[target_id]
                    subgraph_edges[edge_key] = edge
            elif edge["target"] == center_node_id:
                source_id = edge["source"]
                if source_id in self.nodes:
                    subgraph_nodes[source_id] = self.nodes[source_id]
                    subgraph_edges[edge_key] = edge
        
        # Add 2-hop neighbors (limited by max_node_size and max_edge_size)
        if len(subgraph_nodes) < self.max_node_size and len(subgraph_edges) < self.max_edge_size:
            for node_id in list(subgraph_nodes.keys()):
                if node_id != center_node_id:  # Skip center node (already processed)
                    for edge_key, edge in self.edges.items():
                        if len(subgraph_nodes) >= self.max_node_size or len(subgraph_edges) >= self.max_edge_size:
                            break
                        
                        if edge["source"] == node_id and edge["target"] not in subgraph_nodes:
                            target_id = edge["target"]
                            if target_id in self.nodes:
                                subgraph_nodes[target_id] = self.nodes[target_id]
                                subgraph_edges[edge_key] = edge
                        elif edge["target"] == node_id and edge["source"] not in subgraph_nodes:
                            source_id = edge["source"]
                            if source_id in self.nodes:
                                subgraph_nodes[source_id] = self.nodes[source_id]
                                subgraph_edges[edge_key] = edge
        
        return {
            'nodes': subgraph_nodes,
            'edges': subgraph_edges,
            'center_node_id': center_node_id
        }
    
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
            
            # 2. Subgraph augmentation: random node dropout
            augmented_subgraph = self._random_node_dropout(pair["subgraph"])
            if augmented_subgraph != pair["subgraph"]:
                augmented_pair = pair.copy()
                augmented_pair["subgraph"] = augmented_subgraph
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