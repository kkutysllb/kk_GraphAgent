#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:33
# @Desc   : Dataset module for loading and preprocessing graph and text data.
# --------------------------------------------------------
"""

"""
Dataset module for loading and preprocessing graph and text data.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import logging
from tqdm import tqdm

from ..feature_extractor import FeatureExtractor
from preprocess.utils.neo4j_graph_manager import Neo4jGraphManager
from ..utils.logging import LoggerMixin

logger = logging.getLogger(__name__)

class GraphTextDataset(Dataset, LoggerMixin):
    """Dataset class for graph-text pairs"""
    
    def __init__(
        self,
        graph_manager: Neo4jGraphManager,
        feature_extractor: FeatureExtractor,
        node_types: List[str],
        edge_types: List[str],
        max_text_length: int = 512,
        max_node_size: int = 100,
        max_edge_size: int = 200,
        include_dynamic: bool = True
    ):
        """
        Initialize the dataset
        
        Args:
            graph_manager: Neo4j graph manager instance
            feature_extractor: Feature extractor instance
            node_types: List of node types to include
            edge_types: List of edge types to include
            max_text_length: Maximum text sequence length
            max_node_size: Maximum number of nodes in a subgraph
            max_edge_size: Maximum number of edges in a subgraph
            include_dynamic: Whether to include dynamic features
        """
        super().__init__()
        self.graph_manager = graph_manager
        self.feature_extractor = feature_extractor
        self.node_types = node_types
        self.edge_types = edge_types
        self.max_text_length = max_text_length
        self.max_node_size = max_node_size
        self.max_edge_size = max_edge_size
        self.include_dynamic = include_dynamic
        
        # Load graph data
        self.nodes, self.edges = self._load_graph_data()
        
        # Generate text descriptions
        self.text_descriptions = self._generate_text_descriptions()
        
        # Create graph-text pairs
        self.pairs = self._create_pairs()
        
        # Get feature dimensions
        self.node_dim = feature_extractor.get_node_feature_dim(include_dynamic)
        self.edge_dim = feature_extractor.get_edge_feature_dim(include_dynamic)
        
    def _load_graph_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load nodes and edges from Neo4j"""
        self.log_info("Loading graph data from Neo4j...")
        
        # Get nodes by types
        nodes = []
        for node_type in tqdm(self.node_types, desc="Loading nodes"):
            type_nodes = self.graph_manager.get_nodes_by_type(node_type)
            nodes.extend(type_nodes)
            
        # Get edges by types
        edges = []
        for edge_type in tqdm(self.edge_types, desc="Loading edges"):
            type_edges = self.graph_manager.get_edges_by_type(edge_type)
            edges.extend(type_edges)
            
        return nodes, edges
    
    def _generate_text_descriptions(self) -> Dict[str, str]:
        """Generate text descriptions for nodes"""
        self.log_info("Generating text descriptions...")
        
        descriptions = {}
        for node in tqdm(self.nodes, desc="Generating descriptions"):
            node_id = node['id']
            # Get node properties and relationships
            props = self.graph_manager.get_node_properties(node_id)
            rels = self.graph_manager.get_node_relationships(node_id)
            
            # Generate description using properties and relationships
            desc = self._format_description(node, props, rels)
            descriptions[node_id] = desc
            
        return descriptions
    
    def _format_description(
        self,
        node: Dict,
        properties: Dict,
        relationships: List[Dict]
    ) -> str:
        """Format node description using properties and relationships"""
        # Basic node info
        desc = f"This is a {node['type']} node with ID {node['id']}. "
        
        # Add properties
        if properties:
            desc += "It has the following properties: "
            props = [f"{k}={v}" for k,v in properties.items()]
            desc += ", ".join(props) + ". "
            
        # Add relationships
        if relationships:
            desc += "It is connected to: "
            rels = [f"{r['type']} to {r['target']}" for r in relationships]
            desc += ", ".join(rels) + "."
            
        return desc
    
    def _create_pairs(self) -> List[Dict]:
        """Create graph-text pairs for training"""
        self.log_info("Creating graph-text pairs...")
        
        pairs = []
        for node in tqdm(self.nodes, desc="Creating pairs"):
            node_id = node['id']
            if node_id in self.text_descriptions:
                # Extract subgraph
                subgraph = self._extract_subgraph(node_id)
                
                # Extract features
                features = self.feature_extractor.extract_subgraph_features(
                    center_node_id=node_id,
                    neighbor_ids=subgraph['neighbor_ids'],
                    edge_pairs=subgraph['edge_pairs']
                )
                
                pair = {
                    'node_id': node_id,
                    'text': self.text_descriptions[node_id],
                    'node_features': features['node_features'],
                    'edge_features': features['edge_features'],
                    'edge_index': self._create_edge_index(
                        features['node_map'],
                        subgraph['edge_pairs']
                    )
                }
                pairs.append(pair)
                
        return pairs
    
    def _extract_subgraph(self, center_node_id: str) -> Dict:
        """Extract local subgraph centered at given node"""
        # Get k-hop neighbors
        subgraph = self.graph_manager.get_k_hop_subgraph(
            center_node_id,
            k=2,
            max_nodes=self.max_node_size,
            max_edges=self.max_edge_size
        )
        
        return {
            'neighbor_ids': subgraph['node_ids'],
            'edge_pairs': subgraph['edge_pairs']
        }
        
    def _create_edge_index(
        self,
        node_map: Dict[str, int],
        edge_pairs: List[Tuple[str, str]]
    ) -> torch.Tensor:
        """Create edge index tensor from edge pairs"""
        edge_index = []
        for source_id, target_id in edge_pairs:
            source_idx = node_map[source_id]
            target_idx = node_map[target_id]
            edge_index.append([source_idx, target_idx])
            
        return torch.tensor(edge_index, dtype=torch.long).t()
        
    def __len__(self) -> int:
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]
        return {
            'node_id': pair['node_id'],
            'text': pair['text'],
            'node_features': torch.FloatTensor(pair['node_features']),
            'edge_features': torch.FloatTensor(pair['edge_features']),
            'edge_index': pair['edge_index']
        } 