#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:40
# @Desc   : Loss module with contrastive learning and other loss functions.
# --------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Union, Tuple

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

class BatchContrastiveLoss(nn.Module):
    """Batch-based contrastive loss for dual encoder training"""
    
    def __init__(
        self,
        temperature: float = 0.07,
        use_hard_negatives: bool = False,
        hard_negative_ratio: float = 0.5
    ):
        """
        Initialize batch contrastive loss
        
        Args:
            temperature: Temperature parameter
            use_hard_negatives: Whether to use hard negative mining
            hard_negative_ratio: Ratio of hard negatives to use
        """
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_ratio = hard_negative_ratio
        
    def forward(
        self,
        text_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor,
        hard_negatives: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            text_embeddings: Text embeddings (batch_size x dim)
            graph_embeddings: Graph embeddings (batch_size x dim)
            hard_negatives: Optional tuple of (text_hard_negatives, graph_hard_negatives)
            
        Returns:
            Dictionary containing:
                - loss: Contrastive loss value
                - text_to_graph_accuracy: Text to graph retrieval accuracy
                - graph_to_text_accuracy: Graph to text retrieval accuracy
                - similarity: Similarity matrix
        """
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        graph_embeddings = F.normalize(graph_embeddings, p=2, dim=1)
        
        batch_size = text_embeddings.size(0)
        
        # Compute similarity matrix for in-batch samples
        similarity = torch.matmul(text_embeddings, graph_embeddings.t()) / self.temperature
        
        # Create labels (diagonal is positive pairs)
        labels = torch.arange(batch_size, device=text_embeddings.device)
        
        # Incorporate hard negatives if provided
        if self.use_hard_negatives and hard_negatives is not None:
            text_hard_negatives, graph_hard_negatives = hard_negatives
            
            # Normalize hard negatives
            text_hard_negatives = F.normalize(text_hard_negatives, p=2, dim=1)
            graph_hard_negatives = F.normalize(graph_hard_negatives, p=2, dim=1)
            
            # Compute similarity with hard negatives
            text_to_hard_graph = torch.matmul(text_embeddings, graph_hard_negatives.t()) / self.temperature
            hard_text_to_graph = torch.matmul(text_hard_negatives, graph_embeddings.t()) / self.temperature
            
            # Combine with in-batch similarities
            text_to_graph_sim = torch.cat([similarity, text_to_hard_graph], dim=1)
            graph_to_text_sim = torch.cat([similarity.t(), hard_text_to_graph.t()], dim=1)
        else:
            text_to_graph_sim = similarity
            graph_to_text_sim = similarity.t()
        
        # Compute text-to-graph loss
        text_to_graph_loss = F.cross_entropy(text_to_graph_sim, labels)
        
        # Compute graph-to-text loss
        graph_to_text_loss = F.cross_entropy(graph_to_text_sim, labels)
        
        # Combine losses
        loss = (text_to_graph_loss + graph_to_text_loss) / 2
        
        # Compute accuracies
        text_to_graph_accuracy = (text_to_graph_sim.argmax(dim=1) == labels).float().mean()
        graph_to_text_accuracy = (graph_to_text_sim.argmax(dim=1) == labels).float().mean()
        
        return {
            'loss': loss,
            'text_to_graph_accuracy': text_to_graph_accuracy,
            'graph_to_text_accuracy': graph_to_text_accuracy,
            'similarity': similarity
        }

class MultiPositiveLoss(nn.Module):
    """Contrastive loss with multiple positive samples per anchor"""
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean"
    ):
        """
        Initialize multi-positive contrastive loss
        
        Args:
            temperature: Temperature parameter
            reduction: Reduction method (mean or sum)
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        positive_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            anchors: Anchor embeddings (batch_size x dim)
            positives: Positive embeddings (num_positives x dim)
            positive_mask: Binary mask indicating positive pairs (batch_size x num_positives)
            
        Returns:
            Dictionary containing:
                - loss: Multi-positive contrastive loss value
                - accuracy: Prediction accuracy
                - similarity: Similarity matrix
        """
        # Normalize embeddings
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(anchors, positives.t()) / self.temperature
        
        # Apply log softmax
        log_probs = F.log_softmax(similarity, dim=1)
        
        # Compute loss using positive mask
        loss = -(positive_mask * log_probs).sum(dim=1)
        
        if self.reduction == "mean":
            # Normalize by number of positive pairs per anchor
            num_positives = positive_mask.sum(dim=1)
            loss = (loss / torch.clamp(num_positives, min=1)).mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            
        # Compute accuracy
        max_sim_indices = similarity.argmax(dim=1)
        has_positive = (positive_mask.sum(dim=1) > 0)
        correct = torch.zeros_like(max_sim_indices, dtype=torch.bool)
        
        for i in range(anchors.size(0)):
            if has_positive[i]:
                correct[i] = positive_mask[i, max_sim_indices[i]]
                
        accuracy = correct.float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'similarity': similarity
        }

class CombinedLoss(nn.Module):
    """Combined loss function for dual encoder training"""
    
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        triplet_weight: float = 0.5,
        temperature: float = 0.07,
        margin: float = 0.3,
        use_hard_negatives: bool = False
    ):
        """
        Initialize combined loss
        
        Args:
            contrastive_weight: Weight for contrastive loss
            triplet_weight: Weight for triplet loss
            temperature: Temperature parameter for contrastive loss
            margin: Margin parameter for triplet loss
            use_hard_negatives: Whether to use hard negative mining
        """
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.triplet_weight = triplet_weight
        self.batch_contrastive = BatchContrastiveLoss(
            temperature=temperature,
            use_hard_negatives=use_hard_negatives
        )
        self.triplet = TripletLoss(margin=margin)
        
    def forward(
        self,
        text_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor,
        triplet_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        hard_negatives: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            text_embeddings: Text embeddings (batch_size x dim)
            graph_embeddings: Graph embeddings (batch_size x dim)
            triplet_data: Optional tuple of (anchors, positives, negatives) for triplet loss
            hard_negatives: Optional tuple of (text_hard_negatives, graph_hard_negatives)
            
        Returns:
            Dictionary containing:
                - loss: Combined loss value
                - contrastive_loss: Contrastive loss component
                - triplet_loss: Triplet loss component (if used)
                - accuracy: Overall accuracy
        """
        # Compute contrastive loss
        contrastive_results = self.batch_contrastive(
            text_embeddings, 
            graph_embeddings,
            hard_negatives
        )
        contrastive_loss = contrastive_results['loss']
        
        # Initialize result dictionary
        result_dict = {
            'contrastive_loss': contrastive_loss,
            'text_to_graph_accuracy': contrastive_results['text_to_graph_accuracy'],
            'graph_to_text_accuracy': contrastive_results['graph_to_text_accuracy'],
            'accuracy': (contrastive_results['text_to_graph_accuracy'] + 
                        contrastive_results['graph_to_text_accuracy']) / 2
        }
        
        # Compute triplet loss if weight > 0 and data is provided
        if self.triplet_weight > 0 and triplet_data is not None:
            anchors, positives, negatives = triplet_data
            triplet_results = self.triplet(anchors, positives, negatives)
            triplet_loss = triplet_results['loss']
            
            # Add triplet loss to result dictionary
            result_dict['triplet_loss'] = triplet_loss
            
            # Add triplet metrics to result dictionary
            for k, v in triplet_results.items():
                if k != 'loss':
                    result_dict[f'triplet_{k}'] = v
                    
            # Combine losses
            combined_loss = self.contrastive_weight * contrastive_loss + self.triplet_weight * triplet_loss
        else:
            # Only use contrastive loss
            combined_loss = contrastive_loss
        
        # Add combined loss to result dictionary
        result_dict['loss'] = combined_loss
        
        return result_dict

class HardNegativeMiningLoss(nn.Module):
    """Contrastive loss with online hard negative mining"""
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.3,
        mining_strategy: str = "semi-hard",
        negative_ratio: float = 0.5
    ):
        """
        Initialize hard negative mining loss
        
        Args:
            temperature: Temperature parameter
            margin: Margin parameter for semi-hard negatives
            mining_strategy: Mining strategy ('hard', 'semi-hard', or 'distance')
            negative_ratio: Ratio of negatives to mine
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.mining_strategy = mining_strategy
        self.negative_ratio = negative_ratio
        
    def mine_hard_negatives(
        self,
        anchors: torch.Tensor,
        candidates: torch.Tensor,
        positive_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mine hard negatives
        
        Args:
            anchors: Anchor embeddings (batch_size x dim)
            candidates: Candidate embeddings (num_candidates x dim)
            positive_mask: Binary mask indicating positive pairs (batch_size x num_candidates)
            
        Returns:
            Indices of hard negative samples
        """
        # Compute similarity matrix
        similarity = torch.matmul(anchors, candidates.t())
        
        # Create negative mask (complement of positive mask)
        negative_mask = ~positive_mask.bool()
        
        # Apply mining strategy
        if self.mining_strategy == "hard":
            # Hard negatives: most similar negatives
            similarity_masked = similarity.clone()
            similarity_masked[~negative_mask] = -float('inf')
            hard_indices = similarity_masked.argsort(dim=1, descending=True)
        
        elif self.mining_strategy == "semi-hard":
            # Semi-hard negatives: negatives that are closer than positives + margin
            positive_sim = (similarity * positive_mask).sum(dim=1, keepdim=True) / positive_mask.sum(dim=1, keepdim=True).clamp(min=1)
            semi_hard_mask = (similarity > positive_sim - self.margin) & negative_mask
            
            # If no semi-hard negatives, fall back to hard negatives
            if semi_hard_mask.sum() == 0:
                similarity_masked = similarity.clone()
                similarity_masked[~negative_mask] = -float('inf')
                hard_indices = similarity_masked.argsort(dim=1, descending=True)
            else:
                similarity_masked = similarity.clone()
                similarity_masked[~semi_hard_mask] = -float('inf')
                hard_indices = similarity_masked.argsort(dim=1, descending=True)
        
        elif self.mining_strategy == "distance":
            # Distance-based: sample based on similarity distribution
            similarity_masked = similarity.clone()
            similarity_masked[~negative_mask] = -float('inf')
            
            # Apply softmax to get sampling probabilities
            prob = F.softmax(similarity_masked / self.temperature, dim=1)
            
            # Sample indices based on probabilities
            hard_indices = torch.multinomial(prob, num_samples=min(int(candidates.size(0) * self.negative_ratio), 
                                                                 candidates.size(0)))
        
        return hard_indices
    
    def forward(
        self,
        anchors: torch.Tensor,
        candidates: torch.Tensor,
        positive_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            anchors: Anchor embeddings (batch_size x dim)
            candidates: Candidate embeddings (num_candidates x dim)
            positive_mask: Binary mask indicating positive pairs (batch_size x num_candidates)
            
        Returns:
            Dictionary containing:
                - loss: Hard negative mining loss value
                - accuracy: Prediction accuracy
                - hard_negative_indices: Indices of hard negatives
        """
        # Normalize embeddings
        anchors = F.normalize(anchors, p=2, dim=1)
        candidates = F.normalize(candidates, p=2, dim=1)
        
        # Mine hard negatives
        hard_negative_indices = self.mine_hard_negatives(anchors, candidates, positive_mask)
        
        # Gather hard negatives
        batch_size = anchors.size(0)
        num_hard_negatives = hard_negative_indices.size(1)
        hard_negatives = candidates[hard_negative_indices.view(-1)].view(batch_size, num_hard_negatives, -1)
        
        # Compute positive similarity
        positive_sim = (torch.matmul(anchors, candidates.t()) * positive_mask).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
        
        # Compute negative similarity
        negative_sim = torch.bmm(anchors.unsqueeze(1), hard_negatives.transpose(1, 2)).squeeze(1)
        
        # Compute loss
        logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1) / self.temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchors.device)
        loss = F.cross_entropy(logits, labels)
        
        # Compute accuracy
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'hard_negative_indices': hard_negative_indices
        } 