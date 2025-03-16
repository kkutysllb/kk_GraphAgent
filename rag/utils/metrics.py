#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-25 15:30
# @Desc   : 检索评估指标计算工具
# --------------------------------------------------------
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

def compute_similarity(
    query_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    similarity_metric: str = "cosine"
) -> torch.Tensor:
    """
    计算查询嵌入和项目嵌入之间的相似度
    
    Args:
        query_embeddings: 查询嵌入 (num_queries x dim)
        item_embeddings: 项目嵌入 (num_items x dim)
        similarity_metric: 相似度度量方式 ("cosine", "dot", "euclidean")
        
    Returns:
        相似度矩阵 (num_queries x num_items)
    """
    # 归一化嵌入
    if similarity_metric == "cosine":
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        item_embeddings = torch.nn.functional.normalize(item_embeddings, p=2, dim=1)
        
    # 计算相似度
    if similarity_metric in ["cosine", "dot"]:
        similarity = torch.matmul(query_embeddings, item_embeddings.t())
    elif similarity_metric == "euclidean":
        similarity = -torch.cdist(query_embeddings, item_embeddings, p=2)
    else:
        raise ValueError(f"不支持的相似度度量方式: {similarity_metric}")
        
    return similarity

def compute_retrieval_metrics(
    similarity: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    计算检索指标
    
    Args:
        similarity: 相似度矩阵 (num_queries x num_items)
        k_values: 计算指标的k值列表
        
    Returns:
        包含各种检索指标的字典
    """
    # 获取相似度矩阵的形状
    num_queries, num_items = similarity.shape
    
    # 对角线上的元素是正确的匹配（假设查询和项目是一一对应的）
    # 如果不是一一对应，需要提供正确的匹配标签
    labels = torch.arange(num_queries, device=similarity.device)
    
    # 计算每个查询的排名
    sorted_indices = torch.argsort(similarity, dim=1, descending=True)
    ranks = torch.zeros(num_queries, dtype=torch.long, device=similarity.device)
    
    for i in range(num_queries):
        # 找到正确匹配的排名
        ranks[i] = torch.where(sorted_indices[i] == labels[i])[0][0] + 1
    
    # 计算MRR (Mean Reciprocal Rank)
    mrr = (1.0 / ranks.float()).mean().item()
    
    # 计算各种k值的Recall@k和Precision@k
    metrics = {'mrr': mrr}
    
    for k in k_values:
        # 计算Recall@k
        recall_at_k = (ranks <= k).float().mean().item()
        metrics[f'recall@{k}'] = recall_at_k
        
        # 计算Precision@k
        precision_at_k = (ranks <= k).float().sum().item() / (min(k, num_items) * num_queries)
        metrics[f'precision@{k}'] = precision_at_k
        
        # 计算NDCG@k (Normalized Discounted Cumulative Gain)
        dcg_at_k = torch.zeros(num_queries, device=similarity.device)
        for i in range(num_queries):
            if ranks[i] <= k:
                dcg_at_k[i] = 1.0 / torch.log2(torch.tensor(ranks[i].item() + 1, device=similarity.device))
        
        # 理想DCG是1.0 / log2(1 + 1) = 1.0
        idcg = 1.0
        ndcg_at_k = dcg_at_k.mean().item() / idcg
        metrics[f'ndcg@{k}'] = ndcg_at_k
    
    return metrics

def compute_metrics(
    text_embeddings: torch.Tensor,
    graph_embeddings: torch.Tensor,
    similarity_metric: str = "cosine",
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    计算文本到图和图到文本的检索指标
    
    Args:
        text_embeddings: 文本嵌入 (batch_size x dim)
        graph_embeddings: 图嵌入 (batch_size x dim)
        similarity_metric: 相似度度量方式 ("cosine", "dot", "euclidean")
        k_values: 计算指标的k值列表
        
    Returns:
        包含各种检索指标的字典
    """
    # 计算相似度矩阵
    similarity = compute_similarity(text_embeddings, graph_embeddings, similarity_metric)
    
    # 计算文本到图的检索指标
    text_to_graph_metrics = compute_retrieval_metrics(similarity, k_values)
    text_to_graph_metrics = {f'text_to_graph_{k}': v for k, v in text_to_graph_metrics.items()}
    
    # 计算图到文本的检索指标
    graph_to_text_metrics = compute_retrieval_metrics(similarity.t(), k_values)
    graph_to_text_metrics = {f'graph_to_text_{k}': v for k, v in graph_to_text_metrics.items()}
    
    # 合并指标
    metrics = {**text_to_graph_metrics, **graph_to_text_metrics}
    
    # 计算平均指标
    avg_metrics = {}
    for k in k_values:
        avg_metrics[f'avg_recall@{k}'] = (metrics[f'text_to_graph_recall@{k}'] + metrics[f'graph_to_text_recall@{k}']) / 2
        avg_metrics[f'avg_precision@{k}'] = (metrics[f'text_to_graph_precision@{k}'] + metrics[f'graph_to_text_precision@{k}']) / 2
        avg_metrics[f'avg_ndcg@{k}'] = (metrics[f'text_to_graph_ndcg@{k}'] + metrics[f'graph_to_text_ndcg@{k}']) / 2
    
    avg_metrics['avg_mrr'] = (metrics['text_to_graph_mrr'] + metrics['graph_to_text_mrr']) / 2
    
    # 添加平均指标
    metrics.update(avg_metrics)
    
    return metrics

def compute_hard_negative_metrics(
    query_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    similarity_metric: str = "cosine"
) -> Dict[str, float]:
    """
    计算硬负样本的指标
    
    Args:
        query_embeddings: 查询嵌入 (batch_size x dim)
        positive_embeddings: 正样本嵌入 (batch_size x dim)
        negative_embeddings: 负样本嵌入 (num_negatives x dim)
        similarity_metric: 相似度度量方式 ("cosine", "dot", "euclidean")
        
    Returns:
        包含硬负样本指标的字典
    """
    # 计算查询与正样本的相似度
    positive_similarity = compute_similarity(
        query_embeddings, 
        positive_embeddings, 
        similarity_metric
    ).diag()
    
    # 计算查询与负样本的相似度
    negative_similarity = compute_similarity(
        query_embeddings, 
        negative_embeddings, 
        similarity_metric
    )
    
    # 计算硬负样本比例
    # 硬负样本定义为与查询的相似度高于正样本的负样本
    hard_negative_ratio = (negative_similarity > positive_similarity.unsqueeze(1)).float().mean().item()
    
    # 计算相似度差异
    similarity_gap = (positive_similarity.mean() - negative_similarity.mean()).item()
    
    # 计算最小相似度差异
    min_similarity_gap = (positive_similarity - negative_similarity.max(dim=1)[0]).mean().item()
    
    return {
        'hard_negative_ratio': hard_negative_ratio,
        'similarity_gap': similarity_gap,
        'min_similarity_gap': min_similarity_gap
    }

def compute_triplet_metrics(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    margin: float = 0.3,
    similarity_metric: str = "cosine"
) -> Dict[str, float]:
    """
    计算三元组指标
    
    Args:
        anchor_embeddings: 锚点嵌入 (batch_size x dim)
        positive_embeddings: 正样本嵌入 (batch_size x dim)
        negative_embeddings: 负样本嵌入 (batch_size x dim)
        margin: 边界参数
        similarity_metric: 相似度度量方式 ("cosine", "dot", "euclidean")
        
    Returns:
        包含三元组指标的字典
    """
    # 归一化嵌入
    if similarity_metric == "cosine":
        anchor_embeddings = torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = torch.nn.functional.normalize(positive_embeddings, p=2, dim=1)
        negative_embeddings = torch.nn.functional.normalize(negative_embeddings, p=2, dim=1)
    
    # 计算正样本距离
    if similarity_metric in ["cosine", "dot"]:
        positive_distance = 1.0 - torch.sum(anchor_embeddings * positive_embeddings, dim=1)
        negative_distance = 1.0 - torch.sum(anchor_embeddings * negative_embeddings, dim=1)
    elif similarity_metric == "euclidean":
        positive_distance = torch.sum((anchor_embeddings - positive_embeddings) ** 2, dim=1)
        negative_distance = torch.sum((anchor_embeddings - negative_embeddings) ** 2, dim=1)
    else:
        raise ValueError(f"不支持的相似度度量方式: {similarity_metric}")
    
    # 计算三元组损失
    triplet_loss = torch.nn.functional.relu(positive_distance - negative_distance + margin).mean().item()
    
    # 计算三元组准确率（负样本距离 > 正样本距离 + margin）
    triplet_accuracy = (negative_distance > positive_distance + margin).float().mean().item()
    
    # 计算距离比率
    distance_ratio = (positive_distance / negative_distance.clamp(min=1e-8)).mean().item()
    
    return {
        'triplet_loss': triplet_loss,
        'triplet_accuracy': triplet_accuracy,
        'positive_distance': positive_distance.mean().item(),
        'negative_distance': negative_distance.mean().item(),
        'distance_ratio': distance_ratio
    } 