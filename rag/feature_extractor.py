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
        
    def extract_node_features(
        self,
        node_id: str,
        node_type: str
    ) -> Dict:
        """
        根据节点类型提取节点特征
        
        Args:
            node_id: 节点ID
            node_type: 节点类型
            
        Returns:
            节点特征字典
        """
        try:
            # 获取节点属性
            query = f"""
            MATCH (n:{node_type})
            WHERE id(n) = $node_id
            RETURN n
            """
            
            with self.graph_manager._driver.session() as session:
                result = session.run(query, {"node_id": node_id})
                record = result.single()
                
                if not record:
                    logger.warning(f"Node {node_id} of type {node_type} not found")
                    return self._get_default_node_features(node_type)
                    
                node = dict(record["n"].items())
                features = {}
                
                # 提取所有可用的属性
                for key in node:
                    # 直接保留原始属性值
                    features[key] = node[key]
                
                # 确保基本特征键存在
                for key in self.node_feature_keys.get(node_type, []):
                    if key not in features:
                        features[key] = ""
                
                # 处理metrics_data以获取派生特征
                if 'metrics_data' in features and isinstance(features['metrics_data'], str) and features['metrics_data']:
                    try:
                        metrics = json.loads(features['metrics_data'])
                        if metrics:
                            latest_metrics = self._get_latest_metrics(metrics)
                            features.update(latest_metrics)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse metrics_data for node {node_id}")
                
                # 处理log_data以获取派生特征
                if 'log_data' in features and isinstance(features['log_data'], str) and features['log_data']:
                    try:
                        logs = json.loads(features['log_data'])
                        if logs:
                            latest_logs = self._get_latest_logs(logs)
                            features.update(latest_logs)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse log_data for node {node_id}")
                        
                return features
                
        except Exception as e:
            logger.error(f"Error extracting features for node {node_id}: {str(e)}")
            return self._get_default_node_features(node_type)

    def extract_edge_features(
        self,
        source_id: str,
        target_id: str,
        edge_type: str
    ) -> Dict:
        """
        根据边类型提取边特征
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            edge_type: 边类型
            
        Returns:
            边特征字典
        """
        try:
            # 获取默认特征作为基础
            features = self._get_default_edge_features(edge_type)
            
            # 确保ID是整数类型
            try:
                source_id_int = int(source_id)
                target_id_int = int(target_id)
            except (ValueError, TypeError):
                logger.warning(f"无法将ID转换为整数: source_id={source_id}, target_id={target_id}")
                return features
            
            # 首先获取源节点和目标节点的类型
            type_query = """
            MATCH (source) WHERE id(source) = $source_id
            MATCH (target) WHERE id(target) = $target_id
            RETURN labels(source) as source_labels, labels(target) as target_labels
            """
            
            with self.graph_manager._driver.session() as session:
                type_result = session.run(type_query, {"source_id": source_id_int, "target_id": target_id_int})
                type_record = type_result.single()
                
                if not type_record:
                    logger.warning(f"无法获取节点类型: source_id={source_id_int}, target_id={target_id_int}")
                    return features
                
                source_labels = type_record['source_labels']
                target_labels = type_record['target_labels']
                
                # 获取第一个标签作为节点类型
                source_type = source_labels[0] if source_labels else ""
                target_type = target_labels[0] if target_labels else ""
                
                # 设置源节点和目标节点类型
                features["source_type"] = source_type
                features["target_type"] = target_type
        
            
            # 获取边属性
            edge_query = f"""
            MATCH (source)-[r:{edge_type}]->(target)
            WHERE id(source) = $source_id AND id(target) = $target_id
            RETURN r
            """
            
            with self.graph_manager._driver.session() as session:
                edge_result = session.run(edge_query, {"source_id": source_id_int, "target_id": target_id_int})
                edge_record = edge_result.single()
                
                if not edge_record:
                    logger.warning(f"Edge {source_id}->{target_id} of type {edge_type} not found")
                    return features
                
                # 提取边的所有属性
                edge = dict(edge_record["r"].items())
                
                # 添加所有边属性到特征中
                for key, value in edge.items():
                    if key == 'weight':
                        features[key] = float(value)
                    else:
                        features[key] = value
                
                # 处理dynamics_data以获取派生特征
                if 'dynamics_data' in features and features['dynamics_data']:
                    try:
                        # 确保dynamics_data是字符串类型
                        dynamics_data_str = features['dynamics_data']
                        if not isinstance(dynamics_data_str, str):
                            dynamics_data_str = str(dynamics_data_str)
                            
                        dynamics = json.loads(dynamics_data_str)
                        if dynamics and 'propagation_data' in dynamics:
                            latest_dynamics = self._get_latest_dynamics(dynamics['propagation_data'])
                            features.update(latest_dynamics)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse dynamics_data for edge {source_id}->{target_id}: {str(e)}")
                    except Exception as e:
                        logger.warning(f"Error processing dynamics_data for edge {source_id}->{target_id}: {str(e)}")
                
                return features
                
        except Exception as e:
            logger.error(f"Error extracting features for edge {source_id}->{target_id}: {str(e)}")
            return self._get_default_edge_features(edge_type)
            
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
                            result['edges'][edge_type][edge_key] = self.extract_edge_features(
                                rel.start_node.id,
                                rel.end_node.id,
                                edge_type
                            )
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
