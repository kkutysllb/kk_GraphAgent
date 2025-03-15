#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:40
# @Desc   : Dual encoder module that combines text and graph encoders.
# --------------------------------------------------------
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .text_encoder import TextEncoder
from .graph_encoder import GraphEncoder

class DualEncoder(nn.Module):
    """Dual encoder combining text and graph encoders"""
    
    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        node_dim: int = 256,
        edge_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_text: bool = False
    ):
        """
        Initialize dual encoder
        
        Args:
            text_model_name: Name of pre-trained text model
            node_dim: Input node feature dimension
            edge_dim: Input edge feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of graph layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            freeze_text: Whether to freeze text encoder
        """
        super().__init__()
        
        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            output_dim=output_dim,
            dropout=dropout,
            freeze_base=freeze_text
        )
        
        # Graph encoder
        self.graph_encoder = GraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Alignment projection
        self.text_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
        self.graph_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text input
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            
        Returns:
            Dictionary containing text embeddings
        """
        # Get text embeddings
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Project embeddings
        text_embedding = self.text_projection(text_outputs['pooled'])
        
        return {
            'text_embedding': text_embedding,
            'text_outputs': text_outputs
        }
        
    def encode_graph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode graph input
        
        Args:
            node_features: Node feature matrix
            edge_index: Edge index matrix
            edge_features: Edge feature matrix
            batch: Batch assignment matrix
            
        Returns:
            Dictionary containing graph embeddings
        """
        # Get graph embeddings
        graph_outputs = self.graph_encoder(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            batch=batch
        )
        
        # Project embeddings
        graph_embedding = self.graph_projection(graph_outputs['graph_embedding'])
        
        return {
            'graph_embedding': graph_embedding,
            'graph_outputs': graph_outputs
        }
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            node_features: Node feature matrix
            edge_index: Edge index matrix
            edge_features: Edge feature matrix
            batch: Batch assignment matrix
            token_type_ids: Token type IDs
            
        Returns:
            Dictionary containing:
                - text_embedding: Text embedding
                - graph_embedding: Graph embedding
                - similarity: Cosine similarity between embeddings
        """
        # Encode text and graph
        text_outputs = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        graph_outputs = self.encode_graph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            batch=batch
        )
        
        # Get embeddings
        text_embedding = text_outputs['text_embedding']
        graph_embedding = graph_outputs['graph_embedding']
        
        # Compute similarity
        similarity = torch.cosine_similarity(
            text_embedding.unsqueeze(1),
            graph_embedding.unsqueeze(0),
            dim=-1
        )
        
        return {
            'text_embedding': text_embedding,
            'graph_embedding': graph_embedding,
            'similarity': similarity,
            'text_outputs': text_outputs['text_outputs'],
            'graph_outputs': graph_outputs['graph_outputs']
        }
        
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.text_encoder.get_embedding_dim() 