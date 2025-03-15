#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:40
# @Desc   : Graph encoder module based on Graph Attention Networks.
# --------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Dict, List, Optional

class GraphEncoder(nn.Module):
    """Graph encoder based on Graph Attention Networks"""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        residual: bool = True
    ):
        """
        Initialize graph encoder
        
        Args:
            node_dim: Input node feature dimension
            edge_dim: Input edge feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            residual: Whether to use residual connections
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        
        # Node feature projection
        self.node_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Edge feature projection
        self.edge_projection = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=hidden_dim,
                concat=True
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=hidden_dim,
                    concat=True
                )
            )
            
        # Output layer
        self.gat_layers.append(
            GATConv(
                in_channels=hidden_dim,
                out_channels=output_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=hidden_dim,
                concat=True
            )
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
        ])
        self.layer_norms.append(nn.LayerNorm(output_dim))
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            node_features: Node feature matrix
            edge_index: Edge index matrix
            edge_features: Edge feature matrix
            batch: Batch assignment matrix
            
        Returns:
            Dictionary containing:
                - node_embeddings: Node embeddings
                - graph_embedding: Graph embedding
        """
        # Project features
        x = self.node_projection(node_features)
        edge_attr = self.edge_projection(edge_features)
        
        # Initial features
        node_embeddings = [x]
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            # Residual connection
            if self.residual and i > 0:
                x = x + node_embeddings[-1]
                
            # Apply GAT
            x = gat_layer(x, edge_index, edge_attr=edge_attr)
            
            # Apply normalization and activation
            x = self.layer_norms[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
            node_embeddings.append(x)
            
        # Pool graph embedding
        if batch is not None:
            graph_embedding = global_mean_pool(x, batch)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)
            
        return {
            'node_embeddings': node_embeddings[-1],
            'graph_embedding': graph_embedding,
            'all_embeddings': node_embeddings
        }
        
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.output_dim 