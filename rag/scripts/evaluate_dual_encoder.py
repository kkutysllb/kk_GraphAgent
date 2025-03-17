#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-17 11:15
# @Desc   : 双通道编码器评估脚本
# --------------------------------------------------------
"""

import os
import sys
import torch
import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.models.dual_encoder import DualEncoder
from rag.utils.logging import setup_logger
from rag.utils.experiment_manager import ExperimentManager
from rag.utils.tools import set_seed, get_device, move_to_device

# 设置日志
logger = setup_logger("evaluate_dual_encoder")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估双通道编码器")
    
    # 数据参数
    parser.add_argument("--test_data", type=str, required=True, help="测试数据集路径")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作线程数")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    
    # 评估参数
    parser.add_argument("--similarity_threshold", type=float, default=0.5, help="相似度阈值")
    parser.add_argument("--top_k", type=int, default=[1, 5, 10], nargs="+", help="Top-K评估")
    
    # 实验参数
    parser.add_argument("--experiment_dir", type=str, default="experiments", help="实验目录")
    parser.add_argument("--experiment_name", type=str, default=None, help="实验名称")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default=None, help="设备")
    
    return parser.parse_args()

def load_dataset(data_path: str, batch_size: int, num_workers: int) -> DataLoader:
    """
    加载数据集
    
    Args:
        data_path: 数据集路径
        batch_size: 批次大小
        num_workers: 工作线程数
        
    Returns:
        数据加载器
    """
    logger.info(f"加载数据集: {data_path}")
    
    # 加载数据集
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"数据集大小: {len(data)}")
    
    # 创建数据集
    dataset = GraphTextDatasetWrapper(data)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

class GraphTextDatasetWrapper(torch.utils.data.Dataset):
    """图文数据集包装器"""
    
    def __init__(self, data: List[Dict[str, Any]]):
        """
        初始化数据集
        
        Args:
            data: 数据列表
        """
        self.data = data
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取数据项"""
        item = self.data[idx]
        
        # 处理文本
        text = item['text']
        
        # 处理子图
        subgraph = item['subgraph']
        
        # 处理标签
        is_negative = item.get('is_negative', False)
        
        return {
            'node_id': item['node_id'],
            'text': text,
            'subgraph': subgraph,
            'node_type': item['node_type'],
            'is_negative': is_negative
        }

def load_model(model_path: str, device: torch.device) -> DualEncoder:
    """
    加载模型
    
    Args:
        model_path: 模型检查点路径
        device: 设备
        
    Returns:
        双通道编码器模型
    """
    logger.info(f"加载模型: {model_path}")
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取配置
    config = checkpoint.get('config', {})
    
    # 创建模型
    # 注意：这里假设模型结构可以从检查点中恢复
    # 实际应用中可能需要更多的配置信息
    model = DualEncoder.from_pretrained(config)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 移动到设备
    model = model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    return model

def compute_similarity(text_embeddings: torch.Tensor, graph_embeddings: torch.Tensor) -> torch.Tensor:
    """
    计算文本和图嵌入之间的余弦相似度
    
    Args:
        text_embeddings: 文本嵌入 [batch_size, embedding_dim]
        graph_embeddings: 图嵌入 [batch_size, embedding_dim]
        
    Returns:
        相似度矩阵 [batch_size, batch_size]
    """
    # 归一化嵌入
    text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
    graph_embeddings = torch.nn.functional.normalize(graph_embeddings, p=2, dim=1)
    
    # 计算余弦相似度
    similarity = torch.matmul(text_embeddings, graph_embeddings.transpose(0, 1))
    
    return similarity

def evaluate_retrieval(
    model: DualEncoder,
    dataloader: DataLoader,
    device: torch.device,
    top_k: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    评估检索性能
    
    Args:
        model: 双通道编码器模型
        dataloader: 数据加载器
        device: 设备
        top_k: Top-K评估列表
        
    Returns:
        评估指标
    """
    logger.info("评估检索性能...")
    
    # 收集所有嵌入
    all_text_embeddings = []
    all_graph_embeddings = []
    all_node_ids = []
    all_node_types = []
    all_is_negative = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="计算嵌入"):
            # 移动数据到设备
            batch = move_to_device(batch, device)
            
            # 前向传播
            text_embeddings, graph_embeddings = model(
                texts=batch['text'],
                subgraphs=batch['subgraph']
            )
            
            # 收集嵌入
            all_text_embeddings.append(text_embeddings)
            all_graph_embeddings.append(graph_embeddings)
            
            # 收集元数据
            all_node_ids.extend(batch['node_id'])
            all_node_types.extend(batch['node_type'])
            
            # 收集标签
            if 'is_negative' in batch:
                all_is_negative.extend(batch['is_negative'].cpu().numpy())
    
    # 连接所有嵌入
    text_embeddings = torch.cat(all_text_embeddings, dim=0)
    graph_embeddings = torch.cat(all_graph_embeddings, dim=0)
    
    # 计算相似度矩阵
    similarity = compute_similarity(text_embeddings, graph_embeddings)
    
    # 计算检索指标
    metrics = {}
    
    # 计算Hits@K
    for k in top_k:
        # 获取每行的Top-K索引
        _, topk_indices = torch.topk(similarity, k=min(k, similarity.size(1)), dim=1)
        
        # 计算正确检索的数量
        correct = 0
        for i in range(similarity.size(0)):
            if i in topk_indices[i]:
                correct += 1
        
        # 计算Hits@K
        hits_at_k = correct / similarity.size(0)
        metrics[f'Hits@{k}'] = hits_at_k
        logger.info(f"Hits@{k}: {hits_at_k:.4f}")
    
    # 计算MRR (Mean Reciprocal Rank)
    reciprocal_ranks = []
    for i in range(similarity.size(0)):
        # 获取正确答案的排名
        _, indices = torch.sort(similarity[i], descending=True)
        rank = torch.where(indices == i)[0].item() + 1
        reciprocal_ranks.append(1.0 / rank)
    
    mrr = np.mean(reciprocal_ranks)
    metrics['MRR'] = mrr
    logger.info(f"MRR: {mrr:.4f}")
    
    # 计算按节点类型的性能
    node_types = set(all_node_types)
    for node_type in node_types:
        # 获取该类型的索引
        indices = [i for i, t in enumerate(all_node_types) if t == node_type]
        
        if not indices:
            continue
        
        # 计算该类型的Hits@K
        for k in top_k:
            correct = 0
            for i in indices:
                _, topk_indices = torch.topk(similarity[i], k=min(k, similarity.size(1)), dim=1)
                if i in topk_indices[0]:
                    correct += 1
            
            hits_at_k = correct / len(indices)
            metrics[f'{node_type}_Hits@{k}'] = hits_at_k
            logger.info(f"{node_type} Hits@{k}: {hits_at_k:.4f}")
    
    return metrics

def evaluate_classification(
    model: DualEncoder,
    dataloader: DataLoader,
    device: torch.device,
    similarity_threshold: float = 0.5
) -> Dict[str, float]:
    """
    评估分类性能
    
    Args:
        model: 双通道编码器模型
        dataloader: 数据加载器
        device: 设备
        similarity_threshold: 相似度阈值
        
    Returns:
        评估指标
    """
    logger.info("评估分类性能...")
    
    # 收集所有嵌入和标签
    all_text_embeddings = []
    all_graph_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="计算嵌入"):
            # 移动数据到设备
            batch = move_to_device(batch, device)
            
            # 前向传播
            text_embeddings, graph_embeddings = model(
                texts=batch['text'],
                subgraphs=batch['subgraph']
            )
            
            # 收集嵌入
            all_text_embeddings.append(text_embeddings)
            all_graph_embeddings.append(graph_embeddings)
            
            # 收集标签
            if 'is_negative' in batch:
                all_labels.append(batch['is_negative'])
    
    # 连接所有嵌入和标签
    text_embeddings = torch.cat(all_text_embeddings, dim=0)
    graph_embeddings = torch.cat(all_graph_embeddings, dim=0)
    
    if all_labels:
        labels = torch.cat(all_labels, dim=0).cpu().numpy()
        
        # 计算余弦相似度
        text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
        graph_embeddings = torch.nn.functional.normalize(graph_embeddings, p=2, dim=1)
        similarities = torch.sum(text_embeddings * graph_embeddings, dim=1).cpu().numpy()
        
        # 根据相似度阈值预测标签
        predictions = (similarities >= similarity_threshold).astype(int)
        
        # 计算混淆矩阵
        cm = confusion_matrix(labels, predictions)
        
        # 计算精确率、召回率、F1分数
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tn = cm[0, 0]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # 计算PR曲线和ROC曲线
        precision_curve, recall_curve, _ = precision_recall_curve(labels, similarities)
        fpr, tpr, _ = roc_curve(labels, similarities)
        pr_auc = auc(recall_curve, precision_curve)
        roc_auc = auc(fpr, tpr)
        
        # 返回指标
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'precision_recall_curve': {
                'all': (precision_curve, recall_curve, pr_auc)
            },
            'roc_curve': {
                'all': (fpr, tpr, roc_auc)
            }
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1: {f1:.4f}")
        logger.info(f"PR-AUC: {pr_auc:.4f}")
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        
        return metrics
    else:
        logger.warning("没有标签信息，无法评估分类性能")
        return {}

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device) if args.device else get_device()
    logger.info(f"使用设备: {device}")
    
    # 创建实验管理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"evaluation_{timestamp}"
    experiment_manager = ExperimentManager(
        base_dir=args.experiment_dir,
        experiment_name=experiment_name,
        config=vars(args),
        use_tensorboard=False
    )
    
    # 加载数据集
    test_loader = load_dataset(
        data_path=args.test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 加载模型
    model = load_model(args.model_path, device)
    
    # 评估检索性能
    retrieval_metrics = evaluate_retrieval(
        model=model,
        dataloader=test_loader,
        device=device,
        top_k=args.top_k
    )
    
    # 评估分类性能
    classification_metrics = evaluate_classification(
        model=model,
        dataloader=test_loader,
        device=device,
        similarity_threshold=args.similarity_threshold
    )
    
    # 合并指标
    all_metrics = {**retrieval_metrics, **classification_metrics}
    
    # 记录指标
    experiment_manager.log_metrics(all_metrics, 0, prefix="test")
    
    # 可视化评估结果
    experiment_manager.plot_evaluation_results(all_metrics)
    
    # 关闭实验管理器
    experiment_manager.close()
    
    logger.info("评估完成")

if __name__ == "__main__":
    main() 