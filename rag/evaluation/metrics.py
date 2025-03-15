#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:34
# @Desc   : Metrics module for model evaluation.
# --------------------------------------------------------
"""

"""
Metrics module for model evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculator for various evaluation metrics"""
    
    def __init__(self, k_values: List[int] = [1, 5, 10]):
        """
        Initialize metrics calculator
        
        Args:
            k_values: List of k values for top-k metrics
        """
        self.k_values = k_values
        
    def compute_retrieval_metrics(
        self,
        query_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics
        
        Args:
            query_embeddings: Query embedding matrix (n_queries x dim)
            item_embeddings: Item embedding matrix (n_items x dim)
            labels: Ground truth labels (n_queries x n_items)
            
        Returns:
            Dictionary of metric names and values
        """
        # Compute similarity scores
        scores = torch.matmul(query_embeddings, item_embeddings.t())
        
        # Get rankings
        _, rankings = scores.sort(dim=1, descending=True)
        
        metrics = {}
        
        # Compute metrics for each k
        for k in self.k_values:
            # Hits@k
            hits_k = self._compute_hits_at_k(rankings, labels, k)
            metrics[f'hits@{k}'] = hits_k
            
            # MRR@k
            mrr_k = self._compute_mrr_at_k(rankings, labels, k)
            metrics[f'mrr@{k}'] = mrr_k
            
            # Precision@k
            precision_k = self._compute_precision_at_k(rankings, labels, k)
            metrics[f'precision@{k}'] = precision_k
            
            # Recall@k
            recall_k = self._compute_recall_at_k(rankings, labels, k)
            metrics[f'recall@{k}'] = recall_k
            
        # Mean metrics
        metrics['mean_hits'] = np.mean([metrics[f'hits@{k}'] for k in self.k_values])
        metrics['mean_mrr'] = np.mean([metrics[f'mrr@{k}'] for k in self.k_values])
        metrics['mean_precision'] = np.mean([metrics[f'precision@{k}'] for k in self.k_values])
        metrics['mean_recall'] = np.mean([metrics[f'recall@{k}'] for k in self.k_values])
        
        return metrics
    
    def _compute_hits_at_k(
        self,
        rankings: torch.Tensor,
        labels: torch.Tensor,
        k: int
    ) -> float:
        """Compute Hits@K metric"""
        # Get top k predictions
        top_k = rankings[:, :k]
        
        # Check if true items are in top k
        hits = torch.zeros(len(rankings))
        for i, ranking in enumerate(top_k):
            hits[i] = torch.any(labels[i][ranking]).float()
            
        return hits.mean().item()
    
    def _compute_mrr_at_k(
        self,
        rankings: torch.Tensor,
        labels: torch.Tensor,
        k: int
    ) -> float:
        """Compute MRR@K metric"""
        # Get top k predictions
        top_k = rankings[:, :k]
        
        # Calculate reciprocal ranks
        mrr = torch.zeros(len(rankings))
        for i, ranking in enumerate(top_k):
            # Find rank of first relevant item
            for rank, idx in enumerate(ranking):
                if labels[i][idx]:
                    mrr[i] = 1.0 / (rank + 1)
                    break
                    
        return mrr.mean().item()
    
    def _compute_precision_at_k(
        self,
        rankings: torch.Tensor,
        labels: torch.Tensor,
        k: int
    ) -> float:
        """Compute Precision@K metric"""
        # Get top k predictions
        top_k = rankings[:, :k]
        
        # Calculate precision
        precision = torch.zeros(len(rankings))
        for i, ranking in enumerate(top_k):
            true_positives = torch.sum(labels[i][ranking]).float()
            precision[i] = true_positives / k
            
        return precision.mean().item()
    
    def _compute_recall_at_k(
        self,
        rankings: torch.Tensor,
        labels: torch.Tensor,
        k: int
    ) -> float:
        """Compute Recall@K metric"""
        # Get top k predictions
        top_k = rankings[:, :k]
        
        # Calculate recall
        recall = torch.zeros(len(rankings))
        for i, ranking in enumerate(top_k):
            true_positives = torch.sum(labels[i][ranking]).float()
            total_positives = torch.sum(labels[i]).float()
            if total_positives > 0:
                recall[i] = true_positives / total_positives
                
        return recall.mean().item()
    
    def compute_classification_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute classification metrics
        
        Args:
            predictions: Predicted class probabilities
            labels: Ground truth labels
            
        Returns:
            Dictionary of metric names and values
        """
        # Convert to numpy
        preds = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # Get predicted classes
        pred_classes = np.argmax(preds, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            pred_classes,
            average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    def compute_regression_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute regression metrics
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            
        Returns:
            Dictionary of metric names and values
        """
        # Convert to numpy
        preds = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        # Calculate metrics
        mse = np.mean((preds - targets) ** 2)
        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        } 