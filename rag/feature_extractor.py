#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-14 22:43
# @Desc   : 特征提取器模块，负责从Neo4j中提取节点和关系特征
# --------------------------------------------------------
"""

import json
from typing import Dict, List, Any, Optional
import numpy as np
import logging
from preprocess.utils.neo4j_graph_manager import Neo4jGraphManager

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """从Neo4j数据库中提取基于真实数据结构的特征"""
    
    def __init__(
        self,
        graph_manager: Neo4jGraphManager,
        node_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None
    ):
        """
        初始化特征提取器
        
        Args:
            graph_manager: Neo4j图管理器实例
            node_types: 要提取特征的节点类型列表
            relationship_types: 要提取特征的关系类型列表
        """
        self.graph_manager = graph_manager
        
        # 真实的默认节点类型和关系
        self.node_types = node_types or [
            'DC', 'TENANT', 'NE', 'VM', 'HOST', 'HA', 'TRU'
        ]
        
        self.relationship_types = relationship_types or [
            'DC_TO_TENANT', 'TENANT_TO_NE', 'NE_TO_VM',
            'VM_TO_HOST', 'HOST_TO_HA', 'HA_TO_TRU'
        ]
        
        # 每个类型包括层信息的节点特征键
        self.node_feature_keys = {
            'DC': ['id', 'name', 'type', 'business_type', 'layer', 'layer_type'],
            'TENANT': ['id', 'name', 'type', 'business_type', 'layer', 'layer_type'],
            'NE': ['id', 'name', 'type', 'business_type', 'layer', 'layer_type'],
            'VM': ['id', 'name', 'type', 'business_type', 'layer', 'layer_type', 'metrics_data', 'log_data'],
            'HOST': ['id', 'name', 'type', 'business_type', 'layer', 'layer_type', 'metrics_data', 'log_data'],
            'HA': ['id', 'name', 'type', 'business_type', 'layer', 'layer_type', 'metrics_data', 'log_data'],
            'TRU': ['id', 'name', 'type', 'business_type', 'layer', 'layer_type', 'metrics_data', 'log_data']
        }
        
        # 每个类型包括动态属性的边特征键
        self.edge_feature_keys = {
            'DC_TO_TENANT': ['source_type', 'target_type', 'weight'],
            'TENANT_TO_NE': ['source_type', 'target_type', 'weight'],
            'NE_TO_VM': ['source_type', 'target_type', 'weight', 'dynamics_data'],
            'VM_TO_HOST': ['source_type', 'target_type', 'weight', 'dynamics_data'],
            'HOST_TO_HA': ['source_type', 'target_type', 'weight', 'dynamics_data'],
            'HA_TO_TRU': ['source_type', 'target_type', 'weight', 'dynamics_data']
        }
        
        # 层配置
        self.layer_config = {
            'business': ['DC', 'TENANT', 'NE', 'VM'],
            'resource': ['VM', 'HOST', 'HA', 'TRU']
        }
        
    def extract_node_features(self, node_id: int, node_type: str = None) -> Dict:
        """提取节点特征
        
        Args:
            node_id: 节点ID
            node_type: 节点类型（可选）
            
        Returns:
            包含节点特征的字典
        """
        try:
            # 查询节点属性
            query = f"""
            MATCH (n) WHERE id(n) = $node_id
            RETURN n
            """
            with self.graph_manager._driver.session() as session:
                result = session.run(query, node_id=node_id)
                record = result.single()
                if not record:
                    # 如果找不到节点，返回默认特征
                    return {
                        "id": str(node_id),
                        "type": node_type or "unknown",
                        "name": "unknown",
                        "business_type": "",
                        "layer": "",
                        "layer_type": "",
                        "metrics_data": "",
                        "log_data": "",
                        "features": np.zeros(256, dtype=np.float32)
                    }
                
                node = record["n"]
                node_props = dict(node.items())
                
                # 构建特征字典
                features = {
                    "id": str(node_id),
                    "type": node_type or node_props.get("type", "unknown"),
                    "name": node_props.get("name", ""),
                    "business_type": node_props.get("business_type", ""),
                    "layer": node_props.get("layer", ""),
                    "layer_type": node_props.get("layer_type", ""),
                    "metrics_data": node_props.get("metrics_data", ""),
                    "log_data": node_props.get("log_data", ""),
                    "features": np.zeros(256, dtype=np.float32)  # 预留特征向量
                }
                
                return features
                
        except Exception as e:
            self.log_error(f"提取节点 {node_id} 特征时出错: {str(e)}")
            # 返回默认特征
            return {
                "id": str(node_id),
                "type": node_type or "unknown",
                "name": "unknown",
                "business_type": "",
                "layer": "",
                "layer_type": "",
                "metrics_data": "",
                "log_data": "",
                "features": np.zeros(256, dtype=np.float32)
            }
    
    def extract_edge_features(
        self,
        edge_id: str
    ) -> np.ndarray:
        """
        根据边ID提取边特征，返回固定维度的特征向量
        
        Args:
            edge_id: 边ID
            
        Returns:
            边特征向量（64维）
        """
        try:
            # 获取边属性
            query = f"""
            MATCH ()-[r]-()
            WHERE id(r) = $edge_id
            RETURN r, type(r) as type
            """
            
            with self.graph_manager._driver.session() as session:
                result = session.run(query, {"edge_id": edge_id})
                record = result.single()
                
                if not record:
                    logger.warning(f"Edge {edge_id} not found")
                    return np.zeros(64)
                
                edge = dict(record["r"].items())
                edge_type = record["type"]
                
                # 提取基本特征
                basic_features = []
                
                # 添加边类型的one-hot编码
                type_one_hot = np.zeros(len(self.relationship_types))
                if edge_type in self.relationship_types:
                    type_one_hot[self.relationship_types.index(edge_type)] = 1
                basic_features.extend(type_one_hot)
                
                # 处理dynamics_data
                dynamics_features = np.zeros(32)  # 为动态数据预留32维
                if 'dynamics_data' in edge and edge['dynamics_data']:
                    try:
                        dynamics = json.loads(edge['dynamics_data']) if isinstance(edge['dynamics_data'], str) else edge['dynamics_data']
                        if dynamics:
                            dynamics_features = self._process_dynamics_data(dynamics)
                    except (json.JSONDecodeError, TypeError):
                        pass
                basic_features.extend(dynamics_features)
                
                # 确保特征向量长度为64
                features = np.array(basic_features, dtype=np.float32)
                if len(features) < 64:
                    features = np.pad(features, (0, 64 - len(features)))
                elif len(features) > 64:
                    features = features[:64]
                
                return features
                
        except Exception as e:
            logger.error(f"Error extracting features for edge {edge_id}: {str(e)}")
            return np.zeros(64)
    
    def _process_metrics_data(self, metrics: Dict, node_type: str) -> np.ndarray:
        """处理指标数据，返回固定维度的特征向量"""
        features = []
        
        if node_type == 'VM':
            # VM特定的指标
            cpu_usage = self._get_latest_metric_value(metrics, 'cpu_usage', 0)
            memory_usage = self._get_latest_metric_value(metrics, 'memory_usage', 0)
            disk_usage = self._get_latest_metric_value(metrics, 'disk_usage', 0)
            network_traffic = self._get_latest_metric_value(metrics, 'network_traffic', 0)
            
            features.extend([cpu_usage, memory_usage, disk_usage, network_traffic])
            
        elif node_type == 'HOST':
            # HOST特定的指标
            cpu_load = self._get_latest_metric_value(metrics, 'cpu_load', 0)
            memory_used = self._get_latest_metric_value(metrics, 'memory_used', 0)
            disk_io = self._get_latest_metric_value(metrics, 'disk_io', 0)
            network_throughput = self._get_latest_metric_value(metrics, 'network_throughput', 0)
            
            features.extend([cpu_load, memory_used, disk_io, network_throughput])
            
        # 确保返回128维特征
        features = np.array(features, dtype=np.float32)
        if len(features) < 128:
            features = np.pad(features, (0, 128 - len(features)))
        return features[:128]
    
    def _process_log_data(self, logs: Dict) -> np.ndarray:
        """处理日志数据，返回固定维度的特征向量"""
        features = []
        
        if isinstance(logs, dict) and 'log_data' in logs:
            # 统计不同状态的出现次数
            status_counts = {'normal': 0, 'warning': 0, 'error': 0}
            
            for node_id, log_list in logs['log_data'].items():
                if isinstance(log_list, list):
                    # 只处理最近的30条日志
                    recent_logs = log_list[-30:] if len(log_list) > 30 else log_list
                    for entry in recent_logs:
                        if 'status' in entry:
                            for status_key, status_val in entry['status'].items():
                                # 确保status_val是字符串类型
                                status_str = str(status_val).lower() if status_val is not None else ""
                                if status_str in status_counts:
                                    status_counts[status_str] += 1
            
            # 归一化状态计数
            total = sum(status_counts.values()) or 1
            features.extend([count/total for count in status_counts.values()])
        
        # 确保返回64维特征
        features = np.array(features, dtype=np.float32)
        if len(features) < 64:
            features = np.pad(features, (0, 64 - len(features)))
        return features[:64]
    
    def _process_dynamics_data(self, dynamics: Dict) -> np.ndarray:
        """处理动态数据，返回固定维度的特征向量"""
        features = []
        
        if isinstance(dynamics, dict):
            # 处理传播数据
            if 'propagation_data' in dynamics and isinstance(dynamics['propagation_data'], list):
                recent_props = dynamics['propagation_data'][-5:] if len(dynamics['propagation_data']) > 5 else dynamics['propagation_data']
                
                # 提取传播概率
                probs = []
                for prop in recent_props:
                    if 'effects' in prop:
                        for effect in prop['effects'].values():
                            if isinstance(effect, dict) and 'probability' in effect:
                                probs.append(effect['probability'])
                
                # 计算平均传播概率
                if probs:
                    features.append(np.mean(probs))
                else:
                    features.append(0)
        
        # 确保返回32维特征
        features = np.array(features, dtype=np.float32)
        if len(features) < 32:
            features = np.pad(features, (0, 32 - len(features)))
        return features[:32]
    
    def _get_latest_metric_value(self, metrics: Dict, metric_name: str, default: float = 0) -> float:
        """获取指标的最新值"""
        if metric_name in metrics:
            values = metrics[metric_name]
            if isinstance(values, list) and values:
                return float(values[-1])
            elif isinstance(values, (int, float)):
                return float(values)
        return default
    
    def extract_chain_features(
        self,
        dc_id: str,
        chain_type: str = 'both'
    ) -> Dict[str, Any]:
        """
        提取从DC开始的完整链的特征
        
        Args:
            dc_id: DC节点ID
            chain_type: 要提取的链类型 ('business', 'resource', or 'both')
            
        Returns:
            包含完整链的节点和边特征的字典
        """
        try:
            chains = {
                'business': {
                    'path': '(dc:DC)-[:DC_TO_TENANT]->(tenant:TENANT)-[:TENANT_TO_NE]->(ne:NE)-[:NE_TO_VM]->(vm:VM)',
                    'nodes': ['DC', 'TENANT', 'NE', 'VM'],
                    'edges': ['DC_TO_TENANT', 'TENANT_TO_NE', 'NE_TO_VM']
                },
                'resource': {
                    'path': '(vm:VM)-[:VM_TO_HOST]->(host:HOST)-[:HOST_TO_HA]->(ha:HA)-[:HA_TO_TRU]->(tru:TRU)',
                    'nodes': ['VM', 'HOST', 'HA', 'TRU'],
                    'edges': ['VM_TO_HOST', 'HOST_TO_HA', 'HA_TO_TRU']
                }
            }
            
            result = {
                'nodes': {},
                'edges': {},
                'chain_info': {
                    'dc_id': dc_id,
                    'chain_type': chain_type,
                    'layer_info': self.layer_config
                }
            }
            
            if chain_type in ['business', 'both']:
                business_chain = self._extract_specific_chain(dc_id, chains['business'])
                result['nodes'].update(business_chain['nodes'])
                result['edges'].update(business_chain['edges'])
                
            if chain_type in ['resource', 'both']:
                # 从业务链中获取VM节点
                vm_ids = set()
                if 'VM' in result['nodes']:
                    vm_ids = set(result['nodes']['VM'].keys())
                
                # 提取每个VM的资源链
                for vm_id in vm_ids:
                    resource_chain = self._extract_specific_chain(vm_id, chains['resource'])
                    for node_type, nodes in resource_chain['nodes'].items():
                        if node_type not in result['nodes']:
                            result['nodes'][node_type] = {}
                        result['nodes'][node_type].update(nodes)
                    
                    for edge_type, edges in resource_chain['edges'].items():
                        if edge_type not in result['edges']:
                            result['edges'][edge_type] = {}
                        result['edges'][edge_type].update(edges)
                        
            return result
            
        except Exception as e:
            logger.error(f"Error extracting chain features for DC {dc_id}: {str(e)}")
            return {'nodes': {}, 'edges': {}, 'chain_info': {'dc_id': dc_id, 'chain_type': chain_type, 'error': str(e)}}
            
    def _extract_specific_chain(self, start_node_id: str, chain_config: Dict) -> Dict[str, Dict]:
        """提取特定链配置的特征"""
        result = {
            'nodes': {node_type: {} for node_type in chain_config['nodes']},
            'edges': {edge_type: {} for edge_type in chain_config['edges']}
        }
        
        # 修改查询，先匹配起始节点，然后再匹配路径
        query = f"""
        MATCH (start)
        WHERE id(start) = $start_node_id
        MATCH path = {chain_config['path']}
        WHERE start = nodes(path)[0]
        UNWIND nodes(path) as node
        WITH path, collect(node) as nodes
        UNWIND relationships(path) as rel
        RETURN nodes, collect(rel) as rels
        """
        
        try:
            with self.graph_manager._driver.session() as session:
                records = session.run(query, start_node_id=start_node_id)
                
                for record in records:
                    # 处理节点
                    for node in record['nodes']:
                        node_type = list(node.labels)[0]  # 获取第一个标签
                        if node_type in result['nodes']:
                            result['nodes'][node_type][node.id] = self.extract_node_features(node.id, node_type)
                    
                    # 处理关系
                    for rel in record['rels']:
                        edge_type = rel.type
                        if edge_type in result['edges']:
                            edge_key = f"{rel.start_node.id}->{rel.end_node.id}"
                            result['edges'][edge_type][edge_key] = self.extract_edge_features(rel.id)
        except Exception as e:
            logger.error(f"Error in _extract_specific_chain: {str(e)}")
                        
        return result
            
    def _get_latest_metrics(self, metrics: Dict) -> Dict:
        """处理性能数据以获取最新值"""
        if not metrics or not isinstance(metrics, dict):
            return {}
            
        latest_metrics = {}
        for metric_name, values in metrics.items():
            if isinstance(values, list) and values:
                # 假设最后一个值是最新的
                latest_metrics[f"{metric_name}_latest"] = values[-1]
                # 添加一些基本统计信息
                latest_metrics[f"{metric_name}_mean"] = np.mean(values)
                latest_metrics[f"{metric_name}_std"] = np.std(values)
                
        return latest_metrics
        
    def _get_latest_dynamics(self, dynamics_data: List[Dict]) -> Dict:
        """
        处理边的动态数据以获取最新值和统计信息
        
        Args:
            dynamics_data: 包含timestamp和effects的列表
            
        Returns:
            处理后的特征字典
        """
        if not dynamics_data or not isinstance(dynamics_data, list) or len(dynamics_data) == 0:
            return {}
            
        # 按时间戳排序，确保最新的在最后
        sorted_dynamics = sorted(dynamics_data, key=lambda x: x.get('timestamp', ''))
        
        # 获取最新的动态数据
        latest_entry = sorted_dynamics[-1]
        latest_dynamics = {}
        
        # 添加最新时间戳
        latest_dynamics['latest_timestamp'] = latest_entry.get('timestamp', '')
        
        # 处理最新的effects数据
        effects = latest_entry.get('effects', {})
        for effect_name, effect_data in effects.items():
            if isinstance(effect_data, dict):
                for key, value in effect_data.items():
                    latest_dynamics[f"{effect_name}_{key}"] = value
        
        # 计算各种状态的统计信息
        status_counts = {}
        for entry in sorted_dynamics:
            effects = entry.get('effects', {})
            for effect_name, effect_data in effects.items():
                if isinstance(effect_data, dict) and 'source_status' in effect_data:
                    status = effect_data['source_status']
                    if effect_name not in status_counts:
                        status_counts[effect_name] = {}
                    if status not in status_counts[effect_name]:
                        status_counts[effect_name][status] = 0
                    status_counts[effect_name][status] += 1
        
        # 添加状态统计信息到特征中
        for effect_name, counts in status_counts.items():
            for status, count in counts.items():
                latest_dynamics[f"{effect_name}_status_{status}_count"] = count
            total = sum(counts.values())
            for status, count in counts.items():
                latest_dynamics[f"{effect_name}_status_{status}_ratio"] = count / total if total > 0 else 0
                
        return latest_dynamics
        
    def _get_latest_logs(self, logs: Dict) -> Dict:
        """处理日志数据以获取最新条目"""
        if not logs or not isinstance(logs, dict):
            return {}
            
        latest_logs = {}
        for log_type, entries in logs.items():
            if isinstance(entries, list) and entries:
                # 获取最新的日志条目
                latest_logs[f"{log_type}_latest"] = entries[-1]
                # 统计日志条目
                latest_logs[f"{log_type}_count"] = len(entries)
                
        return latest_logs
        
    def _get_default_node_features(self, node_type: str) -> Dict:
        """获取默认节点特征"""
        default_features = {}
        for key in self.node_feature_keys.get(node_type, []):
            if key in ['metrics_data', 'log_data']:
                default_features[key] = ""
            else:
                default_features[key] = ""
        return default_features
       
    def _get_default_edge_features(self, edge_type: str) -> Dict:
        """获取默认边特征"""
        default_features = {
            "source_type": "",
            "target_type": "",
            "weight": 1.0,
            "dynamics_data": ""
        }
        
        # 添加边类型特定的默认特征
        for key in self.edge_feature_keys.get(edge_type, []):
            if key not in default_features:
                default_features[key] = ""
                
        return default_features
