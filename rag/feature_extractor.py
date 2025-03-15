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
from neo4j import GraphDatabase
import numpy as np
from datetime import datetime
import logging
from preprocess.utils.neo4j_graph_manager import Neo4jGraphManager

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from Neo4j database based on real data structure"""
    
    def __init__(
        self,
        graph_manager: Neo4jGraphManager,
        node_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None
    ):
        """
        Initialize feature extractor
        
        Args:
            graph_manager: Neo4j graph manager instance
            node_types: List of node types to extract features from
            relationship_types: List of relationship types to extract features from
        """
        self.graph_manager = graph_manager
        
        # Default node and relationship types based on real data
        self.node_types = node_types or [
            'DC', 'TENANT', 'NE', 'VM', 'HOST', 'HA', 'TRU'
        ]
        
        self.relationship_types = relationship_types or [
            'DC_TO_TENANT', 'TENANT_TO_NE', 'NE_TO_VM',
            'VM_TO_HOST', 'HOST_TO_HA', 'HA_TO_TRU'
        ]
        
        # Node feature keys for each type
        self.node_feature_keys = {
            'DC': ['id', 'name', 'type', 'business_type'],
            'TENANT': ['id', 'name', 'type', 'business_type'],
            'NE': ['id', 'name', 'type', 'business_type'],
            'VM': ['id', 'name', 'type', 'metrics_data', 'log_data'],
            'HOST': ['id', 'name', 'type', 'metrics_data', 'log_data'],
            'HA': ['id', 'name', 'type'],
            'TRU': ['id', 'name', 'type']
        }
        
        # Edge feature keys for each type
        self.edge_feature_keys = {
            'DC_TO_TENANT': ['source_type', 'target_type', 'weight'],
            'TENANT_TO_NE': ['source_type', 'target_type', 'weight'],
            'NE_TO_VM': ['source_type', 'target_type', 'weight'],
            'VM_TO_HOST': ['source_type', 'target_type', 'weight'],
            'HOST_TO_HA': ['source_type', 'target_type', 'weight'],
            'HA_TO_TRU': ['source_type', 'target_type', 'weight']
        }
        
    def extract_node_features(self, node_id: str, node_type: str) -> Dict:
        """
        Extract node features based on node type
        
        Args:
            node_id: Node ID
            node_type: Type of the node
            
        Returns:
            Dictionary of node features
        """
        try:
            # Get node properties
            node = self.graph_manager.get_node_by_id(node_id)
            
            if not node:
                logger.warning(f"Node {node_id} not found")
                return self._get_default_node_features(node_type)
                
            features = {}
            
            # Extract basic features
            for key in ['id', 'name', 'type', 'business_type']:
                if key in self.node_feature_keys[node_type]:
                    features[key] = node.get(key, '')
                    
            # Extract metrics data if available
            if 'metrics_data' in self.node_feature_keys[node_type]:
                metrics = node.get('metrics_data', {})
                if isinstance(metrics, str):
                    try:
                        metrics = json.loads(metrics)
                    except json.JSONDecodeError:
                        metrics = {}
                        
                # Process metrics data
                if metrics:
                    latest_metrics = self._get_latest_metrics(metrics)
                    features.update(latest_metrics)
                    
            # Extract log data if available
            if 'log_data' in self.node_feature_keys[node_type]:
                log_data = node.get('log_data', {})
                if isinstance(log_data, str):
                    try:
                        log_data = json.loads(log_data)
                    except json.JSONDecodeError:
                        log_data = {}
                        
                # Process log data
                if log_data:
                    latest_logs = self._get_latest_logs(log_data)
                    features.update(latest_logs)
                    
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
        Extract edge features based on edge type
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of the edge
            
        Returns:
            Dictionary of edge features
        """
        try:
            # Get edge properties
            edge = self.graph_manager.get_edge(source_id, target_id, edge_type)
            
            if not edge:
                logger.warning(f"Edge {source_id}->{target_id} of type {edge_type} not found")
                return self._get_default_edge_features(edge_type)
                
            features = {}
            
            # Extract basic features
            for key in self.edge_feature_keys[edge_type]:
                features[key] = edge.get(key, '')
                
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
        Extract features for complete chain starting from DC
        
        Args:
            dc_id: DC node ID
            chain_type: Type of chain to extract ('business', 'resource', or 'both')
            
        Returns:
            Dictionary containing node and edge features for the complete chain
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
                    'chain_type': chain_type
                }
            }
            
            if chain_type in ['business', 'both']:
                business_chain = self._extract_specific_chain(dc_id, chains['business'])
                result['nodes'].update(business_chain['nodes'])
                result['edges'].update(business_chain['edges'])
                
            if chain_type in ['resource', 'both']:
                # Get VM nodes from business chain
                vm_ids = set()
                if 'VM' in result['nodes']:
                    vm_ids = set(result['nodes']['VM'].keys())
                
                # Extract resource chain for each VM
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
        """Extract features for a specific chain configuration"""
        result = {
            'nodes': {node_type: {} for node_type in chain_config['nodes']},
            'edges': {edge_type: {} for edge_type in chain_config['edges']}
        }
        
        # Query to get the chain
        query = f"""
        MATCH path = {chain_config['path']}
        WHERE id(startNode(path)) = $start_node_id
        UNWIND nodes(path) as node
        WITH path, collect(node) as nodes
        UNWIND relationships(path) as rel
        RETURN nodes, collect(rel) as rels
        """
        
        with self.graph_manager.driver.session() as session:
            records = session.run(query, start_node_id=start_node_id)
            
            for record in records:
                # Process nodes
                for node in record['nodes']:
                    node_type = list(node.labels)[0]  # Get the first label
                    if node_type in result['nodes']:
                        result['nodes'][node_type][node.id] = self.extract_node_features(node.id, node_type)
                
                # Process relationships
                for rel in record['rels']:
                    edge_type = rel.type
                    if edge_type in result['edges']:
                        edge_key = f"{rel.start_node.id}->{rel.end_node.id}"
                        result['edges'][edge_type][edge_key] = self.extract_edge_features(
                            rel.start_node.id,
                            rel.end_node.id,
                            edge_type
                        )
                        
        return result
            
    def _get_latest_metrics(self, metrics: Dict) -> Dict:
        """Process metrics data to get latest values"""
        if not metrics or not isinstance(metrics, dict):
            return {}
            
        latest_metrics = {}
        for metric_name, values in metrics.items():
            if isinstance(values, list) and values:
                # Assume the last value is the latest
                latest_metrics[f"{metric_name}_latest"] = values[-1]
                # Add some basic statistics
                latest_metrics[f"{metric_name}_mean"] = np.mean(values)
                latest_metrics[f"{metric_name}_std"] = np.std(values)
                
        return latest_metrics
        
    def _get_latest_logs(self, logs: Dict) -> Dict:
        """Process log data to get latest entries"""
        if not logs or not isinstance(logs, dict):
            return {}
            
        latest_logs = {}
        for log_type, entries in logs.items():
            if isinstance(entries, list) and entries:
                # Get the latest log entry
                latest_logs[f"{log_type}_latest"] = entries[-1]
                # Count log entries
                latest_logs[f"{log_type}_count"] = len(entries)
                
        return latest_logs
        
    def _get_default_node_features(self, node_type: str) -> Dict:
        """Get default features for a node type"""
        features = {key: '' for key in self.node_feature_keys[node_type]}
        return features
        
    def _get_default_edge_features(self, edge_type: str) -> Dict:
        """Get default features for an edge type"""
        features = {key: '' for key in self.edge_feature_keys[edge_type]}
        return features 