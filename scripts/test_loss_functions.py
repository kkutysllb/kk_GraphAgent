#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-25 10:30
# @Desc   : 损失函数测试脚本
# --------------------------------------------------------
"""

import os
import sys
import torch
import argparse
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.models.loss import (
    ContrastiveLoss,
    InfoNCELoss,
    TripletLoss,
    BatchContrastiveLoss,
    MultiPositiveLoss,
    CombinedLoss,
    HardNegativeMiningLoss
)
from utils.logger import setup_logger

# 创建测试结果目录
TEST_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_results")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# 设置日志
logger = setup_logger("loss_functions_test")

def generate_test_embeddings(
    batch_size: int = 16,
    embedding_dim: int = 768,
    num_negatives: int = 32,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    生成测试嵌入向量
    
    Args:
        batch_size: 批量大小
        embedding_dim: 嵌入维度
        num_negatives: 负样本数量
        seed: 随机种子
        
    Returns:
        包含测试嵌入的字典
    """
    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)
    
    # 生成文本嵌入
    text_embeddings = torch.randn(batch_size, embedding_dim)
    
    # 生成图嵌入（正样本）
    graph_embeddings = text_embeddings + 0.1 * torch.randn(batch_size, embedding_dim)
    
    # 生成负样本嵌入
    negative_embeddings = torch.randn(num_negatives, embedding_dim)
    
    # 生成锚点、正样本和负样本（用于三元组损失）
    anchors = torch.randn(batch_size, embedding_dim)
    positives = anchors + 0.1 * torch.randn(batch_size, embedding_dim)
    negatives = torch.randn(batch_size, embedding_dim)
    
    # 生成多正样本的掩码
    num_candidates = batch_size + num_negatives
    positive_mask = torch.zeros(batch_size, num_candidates)
    for i in range(batch_size):
        # 每个锚点有1-3个正样本
        num_pos = torch.randint(1, 4, (1,)).item()
        pos_indices = torch.randperm(num_candidates)[:num_pos]
        positive_mask[i, pos_indices] = 1.0
    
    # 生成硬负样本
    text_hard_negatives = torch.randn(batch_size // 2, embedding_dim)
    graph_hard_negatives = torch.randn(batch_size // 2, embedding_dim)
    
    return {
        'text_embeddings': text_embeddings,
        'graph_embeddings': graph_embeddings,
        'negative_embeddings': negative_embeddings,
        'anchors': anchors,
        'positives': positives,
        'negatives': negatives,
        'positive_mask': positive_mask,
        'text_hard_negatives': text_hard_negatives,
        'graph_hard_negatives': graph_hard_negatives
    }

def test_contrastive_loss(
    embeddings: Dict[str, torch.Tensor],
    temperature: float = 0.07
) -> Dict[str, Any]:
    """
    测试对比损失
    
    Args:
        embeddings: 嵌入向量字典
        temperature: 温度参数
        
    Returns:
        测试结果字典
    """
    logger.info("测试对比损失...")
    
    # 创建损失函数
    loss_fn = ContrastiveLoss(temperature=temperature)
    
    # 计算损失
    results = loss_fn(
        embeddings['text_embeddings'],
        embeddings['graph_embeddings']
    )
    
    # 记录结果
    logger.info(f"对比损失: {results['loss']:.4f}")
    logger.info(f"准确率: {results['accuracy']:.4f}")
    
    return {
        'name': 'ContrastiveLoss',
        'loss': results['loss'].item(),
        'accuracy': results['accuracy'].item(),
        'similarity_shape': list(results['similarity'].shape)
    }

def test_infonce_loss(
    embeddings: Dict[str, torch.Tensor],
    temperature: float = 0.07
) -> Dict[str, Any]:
    """
    测试InfoNCE损失
    
    Args:
        embeddings: 嵌入向量字典
        temperature: 温度参数
        
    Returns:
        测试结果字典
    """
    logger.info("测试InfoNCE损失...")
    
    # 创建损失函数
    loss_fn = InfoNCELoss(temperature=temperature)
    
    # 计算损失
    results = loss_fn(
        embeddings['text_embeddings'],
        embeddings['graph_embeddings'],
        embeddings['negative_embeddings']
    )
    
    # 记录结果
    logger.info(f"InfoNCE损失: {results['loss']:.4f}")
    logger.info(f"准确率: {results['accuracy']:.4f}")
    
    return {
        'name': 'InfoNCELoss',
        'loss': results['loss'].item(),
        'accuracy': results['accuracy'].item(),
        'similarity_shape': list(results['similarity'].shape)
    }

def test_triplet_loss(
    embeddings: Dict[str, torch.Tensor],
    margin: float = 0.3
) -> Dict[str, Any]:
    """
    测试三元组损失
    
    Args:
        embeddings: 嵌入向量字典
        margin: 边界参数
        
    Returns:
        测试结果字典
    """
    logger.info("测试三元组损失...")
    
    # 创建损失函数
    loss_fn = TripletLoss(margin=margin)
    
    # 计算损失
    results = loss_fn(
        embeddings['anchors'],
        embeddings['positives'],
        embeddings['negatives']
    )
    
    # 记录结果
    logger.info(f"三元组损失: {results['loss']:.4f}")
    logger.info(f"正样本距离: {results['positive_distance']:.4f}")
    logger.info(f"负样本距离: {results['negative_distance']:.4f}")
    
    return {
        'name': 'TripletLoss',
        'loss': results['loss'].item(),
        'positive_distance': results['positive_distance'].item(),
        'negative_distance': results['negative_distance'].item()
    }

def test_batch_contrastive_loss(
    embeddings: Dict[str, torch.Tensor],
    temperature: float = 0.07,
    use_hard_negatives: bool = True
) -> Dict[str, Any]:
    """
    测试批量对比损失
    
    Args:
        embeddings: 嵌入向量字典
        temperature: 温度参数
        use_hard_negatives: 是否使用硬负样本
        
    Returns:
        测试结果字典
    """
    logger.info("测试批量对比损失...")
    
    # 创建损失函数
    loss_fn = BatchContrastiveLoss(
        temperature=temperature,
        use_hard_negatives=use_hard_negatives
    )
    
    # 准备硬负样本
    hard_negatives = (
        embeddings['text_hard_negatives'],
        embeddings['graph_hard_negatives']
    ) if use_hard_negatives else None
    
    # 计算损失
    results = loss_fn(
        embeddings['text_embeddings'],
        embeddings['graph_embeddings'],
        hard_negatives
    )
    
    # 记录结果
    logger.info(f"批量对比损失: {results['loss']:.4f}")
    logger.info(f"文本到图检索准确率: {results['text_to_graph_accuracy']:.4f}")
    logger.info(f"图到文本检索准确率: {results['graph_to_text_accuracy']:.4f}")
    
    return {
        'name': 'BatchContrastiveLoss',
        'loss': results['loss'].item(),
        'text_to_graph_accuracy': results['text_to_graph_accuracy'].item(),
        'graph_to_text_accuracy': results['graph_to_text_accuracy'].item(),
        'use_hard_negatives': use_hard_negatives
    }

def test_multi_positive_loss(
    embeddings: Dict[str, torch.Tensor],
    temperature: float = 0.07
) -> Dict[str, Any]:
    """
    测试多正样本损失
    
    Args:
        embeddings: 嵌入向量字典
        temperature: 温度参数
        
    Returns:
        测试结果字典
    """
    logger.info("测试多正样本损失...")
    
    # 创建损失函数
    loss_fn = MultiPositiveLoss(temperature=temperature)
    
    # 准备候选样本
    candidates = torch.cat([
        embeddings['graph_embeddings'],
        embeddings['negative_embeddings']
    ], dim=0)
    
    # 计算损失
    results = loss_fn(
        embeddings['text_embeddings'],
        candidates,
        embeddings['positive_mask']
    )
    
    # 记录结果
    logger.info(f"多正样本损失: {results['loss']:.4f}")
    logger.info(f"准确率: {results['accuracy']:.4f}")
    
    return {
        'name': 'MultiPositiveLoss',
        'loss': results['loss'].item(),
        'accuracy': results['accuracy'].item(),
        'similarity_shape': list(results['similarity'].shape)
    }

def test_combined_loss(
    embeddings: Dict[str, torch.Tensor],
    contrastive_weight: float = 1.0,
    triplet_weight: float = 0.5,
    temperature: float = 0.07,
    margin: float = 0.3,
    use_hard_negatives: bool = True
) -> Dict[str, Any]:
    """
    测试组合损失
    
    Args:
        embeddings: 嵌入向量字典
        contrastive_weight: 对比损失权重
        triplet_weight: 三元组损失权重
        temperature: 温度参数
        margin: 边界参数
        use_hard_negatives: 是否使用硬负样本
        
    Returns:
        测试结果字典
    """
    logger.info("测试组合损失...")
    
    # 创建损失函数
    loss_fn = CombinedLoss(
        contrastive_weight=contrastive_weight,
        triplet_weight=triplet_weight,
        temperature=temperature,
        margin=margin,
        use_hard_negatives=use_hard_negatives
    )
    
    # 准备三元组数据
    triplet_data = (
        embeddings['anchors'],
        embeddings['positives'],
        embeddings['negatives']
    )
    
    # 准备硬负样本
    hard_negatives = (
        embeddings['text_hard_negatives'],
        embeddings['graph_hard_negatives']
    ) if use_hard_negatives else None
    
    # 计算损失
    results = loss_fn(
        embeddings['text_embeddings'],
        embeddings['graph_embeddings'],
        triplet_data,
        hard_negatives
    )
    
    # 记录结果
    logger.info(f"组合损失: {results['loss']:.4f}")
    logger.info(f"对比损失: {results.get('contrastive_loss', 0.0):.4f}")
    
    # 只有当三元组损失权重大于0时才记录三元组损失
    if triplet_weight > 0 and 'triplet_loss' in results:
        logger.info(f"三元组损失: {results['triplet_loss']:.4f}")
    
    logger.info(f"准确率: {results['accuracy']:.4f}")
    
    # 构建返回结果
    result_dict = {
        'name': f'CombinedLoss(c={contrastive_weight:.1f},t={triplet_weight:.1f})',
        'loss': results['loss'].item(),
        'accuracy': results['accuracy'].item(),
        'contrastive_weight': contrastive_weight,
        'triplet_weight': triplet_weight,
        'use_hard_negatives': use_hard_negatives
    }
    
    # 添加可选的损失组件
    if 'contrastive_loss' in results:
        result_dict['contrastive_loss'] = results['contrastive_loss'].item()
    if 'triplet_loss' in results:
        result_dict['triplet_loss'] = results['triplet_loss'].item()
    
    return result_dict

def test_hard_negative_mining_loss(
    embeddings: Dict[str, torch.Tensor],
    temperature: float = 0.07,
    margin: float = 0.3,
    mining_strategy: str = "semi-hard"
) -> Dict[str, Any]:
    """
    测试硬负样本挖掘损失
    
    Args:
        embeddings: 嵌入向量字典
        temperature: 温度参数
        margin: 边界参数
        mining_strategy: 挖掘策略
        
    Returns:
        测试结果字典
    """
    logger.info(f"测试硬负样本挖掘损失 (策略: {mining_strategy})...")
    
    # 创建损失函数
    loss_fn = HardNegativeMiningLoss(
        temperature=temperature,
        margin=margin,
        mining_strategy=mining_strategy
    )
    
    # 准备候选样本
    candidates = torch.cat([
        embeddings['graph_embeddings'],
        embeddings['negative_embeddings']
    ], dim=0)
    
    # 计算损失
    results = loss_fn(
        embeddings['text_embeddings'],
        candidates,
        embeddings['positive_mask']
    )
    
    # 记录结果
    logger.info(f"硬负样本挖掘损失: {results['loss']:.4f}")
    logger.info(f"准确率: {results['accuracy']:.4f}")
    
    return {
        'name': 'HardNegativeMiningLoss',
        'loss': results['loss'].item(),
        'accuracy': results['accuracy'].item(),
        'mining_strategy': mining_strategy,
        'hard_negative_indices_shape': list(results['hard_negative_indices'].shape)
    }

def compare_loss_functions(
    batch_size: int = 16,
    embedding_dim: int = 768,
    num_negatives: int = 32,
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    比较不同的损失函数
    
    Args:
        batch_size: 批量大小
        embedding_dim: 嵌入维度
        num_negatives: 负样本数量
        seed: 随机种子
        
    Returns:
        比较结果字典
    """
    logger.info("比较不同的损失函数...")
    
    # 生成测试嵌入
    embeddings = generate_test_embeddings(
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        num_negatives=num_negatives,
        seed=seed
    )
    
    # 测试不同的损失函数
    results = []
    
    # 测试对比损失
    results.append(test_contrastive_loss(embeddings))
    
    # 测试InfoNCE损失
    results.append(test_infonce_loss(embeddings))
    
    # 测试三元组损失
    results.append(test_triplet_loss(embeddings))
    
    # 测试批量对比损失（不使用硬负样本）
    results.append(test_batch_contrastive_loss(embeddings, use_hard_negatives=False))
    
    # 测试批量对比损失（使用硬负样本）
    results.append(test_batch_contrastive_loss(embeddings, use_hard_negatives=True))
    
    # 测试多正样本损失
    results.append(test_multi_positive_loss(embeddings))
    
    # 测试组合损失（不同权重组合）
    results.append(test_combined_loss(embeddings, contrastive_weight=1.0, triplet_weight=0.0))
    results.append(test_combined_loss(embeddings, contrastive_weight=0.0, triplet_weight=1.0))
    results.append(test_combined_loss(embeddings, contrastive_weight=0.5, triplet_weight=0.5))
    
    # 测试硬负样本挖掘损失（不同挖掘策略）
    results.append(test_hard_negative_mining_loss(embeddings, mining_strategy="hard"))
    results.append(test_hard_negative_mining_loss(embeddings, mining_strategy="semi-hard"))
    results.append(test_hard_negative_mining_loss(embeddings, mining_strategy="distance"))
    
    # 保存比较结果
    comparison_results = {
        'batch_size': batch_size,
        'embedding_dim': embedding_dim,
        'num_negatives': num_negatives,
        'seed': seed,
        'results': results
    }
    
    return comparison_results

def visualize_results(comparison_results: Dict[str, Any]) -> None:
    """
    可视化比较结果
    
    Args:
        comparison_results: 比较结果字典
    """
    # 提取损失值和准确率
    loss_values = []
    accuracy_values = []
    loss_names = []
    accuracy_names = []
    
    for result in comparison_results['results']:
        if 'loss' in result:
            loss_values.append(result['loss'])
            loss_names.append(result['name'])
            
        if 'accuracy' in result:
            accuracy_values.append(result.get('accuracy', 0))
            accuracy_names.append(result['name'])
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制损失值
    plt.subplot(2, 1, 1)
    plt.bar(loss_names, loss_values)
    plt.title('Loss Values Comparison')
    plt.ylabel('Loss')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 绘制准确率
    plt.subplot(2, 1, 2)
    plt.bar(accuracy_names, accuracy_values)
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_path = os.path.join(TEST_RESULTS_DIR, f"loss_functions_comparison_{timestamp}.png")
    plt.savefig(chart_path)
    logger.info(f"比较图表已保存到: {chart_path}")
    
    # 关闭图表
    plt.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="损失函数测试")
    parser.add_argument("--batch_size", type=int, default=16, help="批量大小")
    parser.add_argument("--embedding_dim", type=int, default=768, help="嵌入维度")
    parser.add_argument("--num_negatives", type=int, default=32, help="负样本数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--visualize", action="store_true", help="是否可视化结果")
    args = parser.parse_args()
    
    # 比较不同的损失函数
    comparison_results = compare_loss_functions(
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        num_negatives=args.num_negatives,
        seed=args.seed
    )
    
    # 保存比较结果
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(TEST_RESULTS_DIR, f"loss_functions_comparison_{timestamp}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    logger.info(f"比较结果已保存到: {result_file}")
    
    # 可视化结果
    if args.visualize:
        visualize_results(comparison_results)

if __name__ == "__main__":
    main() 