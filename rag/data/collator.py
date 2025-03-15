#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:34
# @Desc   : Collator module for batching and preprocessing data.
# --------------------------------------------------------
"""

"""
Collator module for batching and preprocessing data.
"""

import torch
from typing import Dict, List
from transformers import PreTrainedTokenizer
import numpy as np

class GraphTextCollator:
    """Collator class for batching graph-text pairs"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the collator
        
        Args:
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            device: Device to put tensors on
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Process and collate batch of samples
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batch dictionary with tensors
        """
        # Separate text and graph data
        texts = [sample['text'] for sample in batch]
        graphs = [sample['graph'] for sample in batch]
        node_ids = [sample['node_id'] for sample in batch]
        
        # Process text
        text_batch = self._process_text(texts)
        
        # Process graphs
        graph_batch = self._process_graphs(graphs)
        
        return {
            'node_ids': node_ids,
            'input_ids': text_batch['input_ids'],
            'attention_mask': text_batch['attention_mask'],
            'node_features': graph_batch['node_features'],
            'edge_index': graph_batch['edge_index'],
            'edge_features': graph_batch['edge_features'],
            'batch_idx': graph_batch['batch_idx']
        }
        
    def _process_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Process text sequences"""
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'].to(self.device),
            'attention_mask': tokenized['attention_mask'].to(self.device)
        }
        
    def _process_graphs(self, graphs: List[Dict]) -> Dict[str, torch.Tensor]:
        """Process graph structures"""
        batch_size = len(graphs)
        
        # Initialize lists for batching
        all_nodes = []
        all_edges = []
        edge_index = []
        batch_idx = []
        
        node_offset = 0
        for batch_id, graph in enumerate(graphs):
            nodes = graph['nodes']
            edges = graph['edges']
            
            # Add nodes
            all_nodes.extend(nodes)
            
            # Add edges
            all_edges.extend(edges)
            
            # Create edge index
            src_nodes = [edge['source'] for edge in edges]
            dst_nodes = [edge['target'] for edge in edges]
            batch_edge_index = torch.tensor([src_nodes, dst_nodes])
            
            # Offset node indices
            batch_edge_index = batch_edge_index + node_offset
            edge_index.append(batch_edge_index)
            
            # Track batch membership
            batch_idx.extend([batch_id] * len(nodes))
            
            # Update offset
            node_offset += len(nodes)
            
        # Concatenate edge indices
        edge_index = torch.cat(edge_index, dim=1)
        
        # Convert node features
        node_features = self._extract_node_features(all_nodes)
        
        # Convert edge features
        edge_features = self._extract_edge_features(all_edges)
        
        return {
            'node_features': node_features.to(self.device),
            'edge_index': edge_index.to(self.device),
            'edge_features': edge_features.to(self.device),
            'batch_idx': torch.tensor(batch_idx).to(self.device)
        }
        
    def _extract_node_features(self, nodes: List[Dict]) -> torch.Tensor:
        """Extract and normalize node features"""
        # Extract relevant features
        features = []
        for node in nodes:
            node_feats = [
                float(node.get('cpu_usage', 0)),
                float(node.get('memory_usage', 0)),
                float(node.get('disk_usage', 0)),
                float(node.get('network_in', 0)),
                float(node.get('network_out', 0))
            ]
            features.append(node_feats)
            
        # Convert to tensor and normalize
        features = torch.tensor(features, dtype=torch.float)
        features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
        
        return features
        
    def _extract_edge_features(self, edges: List[Dict]) -> torch.Tensor:
        """Extract and normalize edge features"""
        # Extract relevant features
        features = []
        for edge in edges:
            edge_feats = [
                float(edge.get('latency', 0)),
                float(edge.get('bandwidth', 0)),
                float(edge.get('packet_loss', 0))
            ]
            features.append(edge_feats)
            
        # Convert to tensor and normalize
        features = torch.tensor(features, dtype=torch.float)
        features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
        
        return features 