#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:40
# @Desc   : Loss module with contrastive learning and other loss functions.
# --------------------------------------------------------
"""

"""
Loss module with contrastive learning and other loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class ContrastiveLoss(nn.Module):
    """Contrastive loss for similarity learning"""
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean"
    ):
        """
        Initialize contrastive loss
        
        Args:
            temperature: Temperature parameter
            reduction: Reduction method (mean or sum)
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(
        self,
        text_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            text_embeddings: Text embeddings (batch_size x dim)
            graph_embeddings: Graph embeddings (batch_size x dim)
            labels: Optional labels for supervised contrastive loss
            
        Returns:
            Dictionary containing:
                - loss: Contrastive loss value
                - accuracy: Prediction accuracy
                - similarity: Similarity matrix
        """
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        graph_embeddings = F.normalize(graph_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(text_embeddings, graph_embeddings.t()) / self.temperature
        
        # Get batch size
        batch_size = text_embeddings.size(0)
        
        # Default labels are identity matrix (diagonal is positive pairs)
        if labels is None:
            labels = torch.eye(batch_size, device=text_embeddings.device)
            
        # Compute log softmax
        log_probs = F.log_softmax(similarity, dim=1)
        
        # Compute loss
        loss = -torch.sum(labels * log_probs)
        if self.reduction == "mean":
            loss = loss / batch_size
            
        # Compute accuracy
        predictions = similarity.argmax(dim=1)
        targets = labels.argmax(dim=1)
        accuracy = (predictions == targets).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'similarity': similarity
        }
        
class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning"""
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean"
    ):
        """
        Initialize InfoNCE loss
        
        Args:
            temperature: Temperature parameter
            reduction: Reduction method (mean or sum)
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        negative_keys: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            query: Query embeddings (batch_size x dim)
            positive_key: Positive key embeddings (batch_size x dim)
            negative_keys: Negative key embeddings (num_negative x dim)
            
        Returns:
            Dictionary containing:
                - loss: InfoNCE loss value
                - accuracy: Prediction accuracy
                - similarity: Similarity scores
        """
        # Normalize embeddings
        query = F.normalize(query, p=2, dim=1)
        positive_key = F.normalize(positive_key, p=2, dim=1)
        negative_keys = F.normalize(negative_keys, p=2, dim=1)
        
        # Compute positive similarity
        positive_similarity = torch.sum(
            query * positive_key,
            dim=1,
            keepdim=True
        ) / self.temperature
        
        # Compute negative similarity
        negative_similarity = torch.matmul(
            query,
            negative_keys.t()
        ) / self.temperature
        
        # Concatenate similarities
        logits = torch.cat([positive_similarity, negative_similarity], dim=1)
        
        # Create labels (positive pair is at index 0)
        labels = torch.zeros(
            logits.size(0),
            dtype=torch.long,
            device=query.device
        )
        
        # Compute cross entropy loss
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        
        # Compute accuracy
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'similarity': logits
        }
        
class TripletLoss(nn.Module):
    """Triplet loss with hard negative mining"""
    
    def __init__(
        self,
        margin: float = 0.3,
        reduction: str = "mean"
    ):
        """
        Initialize triplet loss
        
        Args:
            margin: Margin parameter
            reduction: Reduction method (mean or sum)
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            anchor: Anchor embeddings (batch_size x dim)
            positive: Positive embeddings (batch_size x dim)
            negative: Negative embeddings (batch_size x dim)
            
        Returns:
            Dictionary containing:
                - loss: Triplet loss value
                - positive_distance: Distance to positive samples
                - negative_distance: Distance to negative samples
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # Compute distances
        positive_distance = torch.sum((anchor - positive) ** 2, dim=1)
        negative_distance = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Compute loss
        loss = F.relu(positive_distance - negative_distance + self.margin)
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            
        return {
            'loss': loss,
            'positive_distance': positive_distance.mean(),
            'negative_distance': negative_distance.mean()
        } 