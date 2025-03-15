#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:38
# @Desc   : Text encoder module based on pre-trained language models.
# --------------------------------------------------------
"""


"""
Text encoder module based on pre-trained language models.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional

class TextEncoder(nn.Module):
    """Text encoder based on pre-trained language models"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dim: int = 768,
        dropout: float = 0.1,
        freeze_base: bool = False
    ):
        """
        Initialize text encoder
        
        Args:
            model_name: Name of pre-trained model
            output_dim: Output dimension
            dropout: Dropout rate
            freeze_base: Whether to freeze base model
        """
        super().__init__()
        
        # Load pre-trained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
                
        # Output projection
        if self.config.hidden_size != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.config.hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout)
            )
        else:
            self.projection = nn.Identity()
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional)
            
        Returns:
            Dictionary containing:
                - embeddings: Text embeddings
                - pooled: Pooled text representation
                - hidden_states: All hidden states (if output_hidden_states=True)
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get sequence output and pooled output
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # Project outputs
        sequence_output = self.projection(sequence_output)
        pooled_output = self.projection(pooled_output)
        
        return {
            'embeddings': sequence_output,
            'pooled': pooled_output,
            'hidden_states': outputs.hidden_states
        }
        
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.config.hidden_size 