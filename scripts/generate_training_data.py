#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:52
# @Desc   : Script to generate training dataset.
# --------------------------------------------------------
"""

"""
Script to generate training dataset.
"""

import os
import logging
import torch
from torch.utils.data import random_split
import json

from rag.data.dataset import GraphTextDataset
from rag.feature_extractor import FeatureExtractor
from preprocess.utils.neo4j_graph_manager import Neo4jGraphManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Neo4j connection parameters
    neo4j_params = {
        'uri': 'bolt://localhost:7687',
        'user': 'neo4j',
        'password': 'password'
    }
    
    # Create graph manager
    graph_manager = Neo4jGraphManager(**neo4j_params)
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(
        graph_manager=graph_manager,
        node_feature_keys=[
            'cpu_usage',
            'memory_usage',
            'disk_usage',
            'network_in',
            'network_out'
        ],
        edge_feature_keys=[
            'latency',
            'bandwidth',
            'packet_loss'
        ]
    )
    
    # Create dataset
    dataset = GraphTextDataset(
        graph_manager=graph_manager,
        feature_extractor=feature_extractor,
        node_types=['Device'],
        edge_types=['CONNECTS_TO'],
        max_text_length=512,
        max_node_size=100,
        max_edge_size=200,
        include_dynamic=True
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )
    
    # Create output directory
    os.makedirs('data', exist_ok=True)
    
    # Save dataset splits
    logger.info("Saving dataset splits...")
    
    torch.save(train_dataset, 'data/train_dataset.pt')
    torch.save(val_dataset, 'data/val_dataset.pt')
    torch.save(test_dataset, 'data/test_dataset.pt')
    
    # Save dataset info
    dataset_info = {
        'total_size': len(dataset),
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'node_feature_dim': dataset.node_dim,
        'edge_feature_dim': dataset.edge_dim,
        'node_types': dataset.node_types,
        'edge_types': dataset.edge_types
    }
    
    with open('data/dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
        
    logger.info("Dataset generation completed successfully!")
    logger.info(f"Dataset info: {dataset_info}")

if __name__ == "__main__":
    main() 