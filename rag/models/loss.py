#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:40
# @Desc   : 损失模块，包含对比学习和其他损失函数。
# --------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class ContrastiveLoss(nn.Module):
    """对比损失用于相似性学习"""
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean"
    ):
        """
        初始化对比损失
        
        Args:
            temperature: 温度参数
            reduction: 减少方法（mean或sum）
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
        前向传播
        
        Args:
            text_embeddings: 文本嵌入（batch_size x dim）
            graph_embeddings: 图嵌入（batch_size x dim）
            labels: 可选标签用于监督对比损失
            
        Returns:
            包含:
                - loss: 对比损失值
                - accuracy: 预测准确率
                - similarity: 相似度矩阵
        """
        # 归一化嵌入
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        graph_embeddings = F.normalize(graph_embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(text_embeddings, graph_embeddings.t()) / self.temperature
        
        # 获取批量大小
        batch_size = text_embeddings.size(0)
        
        # 默认标签是单位矩阵（对角线是正样本）
        if labels is None:
            labels = torch.eye(batch_size, device=text_embeddings.device)
            
        # 计算log softmax
        log_probs = F.log_softmax(similarity, dim=1)
        
        # 计算损失
        loss = -torch.sum(labels * log_probs)
        if self.reduction == "mean":
            loss = loss / batch_size
            
        # 计算准确率
        predictions = similarity.argmax(dim=1)
        targets = labels.argmax(dim=1)
        accuracy = (predictions == targets).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'similarity': similarity
        }
        
class InfoNCELoss(nn.Module):
    """InfoNCE损失用于对比学习"""
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean"
    ):
        """
        初始化InfoNCE损失
        
        Args:
            temperature: 温度参数
            reduction: 减少方法（mean或sum）
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
        前向传播
        
        Args:
            query: 查询嵌入（batch_size x dim）
            positive_key: 正样本键嵌入（batch_size x dim）
            negative_keys: 负样本键嵌入（num_negative x dim）
            
        Returns:
            包含:
                - loss: InfoNCE损失值
                - accuracy: 预测准确率
                - similarity: 相似度得分
        """
        # 归一化嵌入
        query = F.normalize(query, p=2, dim=1)
        positive_key = F.normalize(positive_key, p=2, dim=1)
        negative_keys = F.normalize(negative_keys, p=2, dim=1)
        
        # 计算正相似度
        positive_similarity = torch.sum(
            query * positive_key,
            dim=1,
            keepdim=True
        ) / self.temperature
        
        # 计算负相似度
        negative_similarity = torch.matmul(
            query,
            negative_keys.t()
        ) / self.temperature
        
        # 连接相似度
        logits = torch.cat([positive_similarity, negative_similarity], dim=1)
        
        # 创建标签（正样本对在索引0）
        labels = torch.zeros(
            logits.size(0),
            dtype=torch.long,
            device=query.device
        )
        
        # 计算交叉熵损失
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
    """Triplet损失用于硬负挖掘"""
    
    def __init__(
        self,
        margin: float = 0.3,
        reduction: str = "mean"
    ):
        """
        初始化三元组损失
        
        Args:
            margin: 边距参数
            reduction: 减少方法（mean或sum）
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
        前向传播
        
        Args:
            anchor: 锚点嵌入（batch_size x dim）
            positive: 正样本嵌入（batch_size x dim）
            negative: 负样本嵌入（batch_size x dim）
            
        Returns:
            包含:
                - loss: 三元组损失值
                - positive_distance: 正样本距离
                - negative_distance: 负样本距离
        """
        # 归一化嵌入
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # 计算距离
        positive_distance = torch.sum((anchor - positive) ** 2, dim=1)
        negative_distance = torch.sum((anchor - negative) ** 2, dim=1)
        
        # 计算损失
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
    """基于批次的对比损失用于双编码器训练"""
    
    def __init__(
        self,
        temperature: float = 0.07,
        use_hard_negatives: bool = False,
        hard_negative_ratio: float = 0.5
    ):
        """
        初始化批量对比损失
        
        Args:
            temperature: 温度参数
            use_hard_negatives: 是否使用硬负挖掘
            hard_negative_ratio: 硬负挖掘比例
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
        计算批次对比损失
        
        Args:
            text_embeddings: 文本嵌入 [batch_size, embedding_dim]
            graph_embeddings: 图嵌入 [batch_size, embedding_dim]
            hard_negatives: 硬负样本 (text_embeddings, graph_embeddings)
            
        Returns:
            损失字典
        """
        # 确保输入维度匹配
        batch_size = text_embeddings.size(0)
        if batch_size != graph_embeddings.size(0):
            raise ValueError(f"Text embeddings batch size ({batch_size}) does not match graph embeddings batch size ({graph_embeddings.size(0)})")
        
        # 计算相似度矩阵
        similarity = F.cosine_similarity(text_embeddings.unsqueeze(1), graph_embeddings.unsqueeze(0), dim=2)
        similarity = similarity / self.temperature
        
        # 创建标签（对角线上的元素为正样本）
        labels = torch.arange(batch_size, device=similarity.device)
        
        # 计算基本损失（不使用硬负样本）
        text_to_graph_loss = F.cross_entropy(similarity, labels)
        # 对转置后的相似度矩阵计算损失，确保维度与标签匹配
        graph_to_text_loss = F.cross_entropy(similarity.t(), labels)
        
        # 计算准确率
        text_to_graph_acc = (similarity.argmax(dim=1) == labels).float().mean()
        graph_to_text_acc = (similarity.t().argmax(dim=1) == labels).float().mean()
        
        return {
            "loss": text_to_graph_loss + graph_to_text_loss,
            "text_to_graph_loss": text_to_graph_loss,
            "graph_to_text_loss": graph_to_text_loss,
            "text_to_graph_acc": text_to_graph_acc,
            "graph_to_text_acc": graph_to_text_acc
        }

class MultiPositiveLoss(nn.Module):
    """基于锚点的对比损失，每个锚点有多个正样本"""
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean"
    ):
        """
        初始化多正样本对比损失
        
        Args:
            temperature: 温度参数
            reduction: 减少方法（mean或sum）
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
        前向传播
        
        Args:
            anchors: 锚点嵌入（batch_size x dim）
            positives: 正样本嵌入（num_positives x dim）
            positive_mask: 二进制掩码指示正样本对（batch_size x num_positives）
            
        Returns:
            包含:
                - loss: 多正样本对比损失值
                - accuracy: 预测准确率
                - similarity: 相似度矩阵
        """
        # 归一化嵌入
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(anchors, positives.t()) / self.temperature
        
        # 应用log softmax
        log_probs = F.log_softmax(similarity, dim=1)
        
        # 使用正样本掩码计算损失
        loss = -(positive_mask * log_probs).sum(dim=1)
        
        if self.reduction == "mean":
            # 按锚点正样本对数归一化
            num_positives = positive_mask.sum(dim=1)
            loss = (loss / torch.clamp(num_positives, min=1)).mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            
        # 计算准确率
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
    """用于双编码器训练的组合损失函数"""
    
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        triplet_weight: float = 0.5,
        temperature: float = 0.07,
        margin: float = 0.3,
        use_hard_negatives: bool = False
    ):
        """
        初始化组合损失
        
        Args:
            contrastive_weight: 对比损失权重
            triplet_weight: 三元组损失权重
            temperature: 对比损失温度参数
            margin: 三元组损失边距参数
            use_hard_negatives: 是否使用硬负挖掘
        """
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.triplet_weight = triplet_weight
        
        # 创建一个简单版本的BatchContrastiveLoss
        # 这个版本不会在forward方法中尝试处理硬负样本
        self.batch_contrastive = BatchContrastiveLoss(
            temperature=temperature,
            use_hard_negatives=False  # 禁用硬负样本处理
        )
        
        self.triplet = TripletLoss(margin=margin)
        self.use_hard_negatives = use_hard_negatives
        self.temperature = temperature
        
    def forward(
        self,
        text_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor,
        triplet_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        hard_negatives: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            text_embeddings: 文本嵌入（batch_size x dim）
            graph_embeddings: 图嵌入（batch_size x dim）
            triplet_data: 可选元组（anchors, positives, negatives）用于三元组损失
            hard_negatives: 可选元组（text_hard_negatives, graph_hard_negatives）
            
        Returns:
            包含:
                - loss: 组合损失值
                - contrastive_loss: 对比损失组件
                - triplet_loss: 三元组损失组件（如果使用）
                - accuracy: 总体准确率
        """
        # 计算标准对比损失，不使用硬负样本
        contrastive_results = self.batch_contrastive(text_embeddings, graph_embeddings)
        contrastive_loss = contrastive_results['loss']
        
        # 如果有硬负样本并启用了硬负样本功能
        # 但在这里单独处理，而不是通过BatchContrastiveLoss
        if hard_negatives is not None and self.use_hard_negatives:
            hn_text_emb, hn_graph_emb = hard_negatives
            batch_size = text_embeddings.size(0)
            
            # 归一化所有嵌入
            text_norm = F.normalize(text_embeddings, dim=1)
            graph_norm = F.normalize(graph_embeddings, dim=1)
            hn_text_norm = F.normalize(hn_text_emb, dim=1)
            hn_graph_norm = F.normalize(hn_graph_emb, dim=1)
            
            # 计算基本相似度矩阵
            similarity = torch.matmul(text_norm, graph_norm.t()) / self.temperature
            
            # 文本到硬负图的相似度
            text_to_hn_graph = torch.matmul(text_norm, hn_graph_norm.t()) / self.temperature
            
            # 只扩展文本到图的方向
            text_to_all_graph = torch.cat([similarity, text_to_hn_graph], dim=1)
            
            # 计算硬负样本增强的文本到图损失
            labels = torch.arange(batch_size, device=text_embeddings.device)
            enhanced_t2g_loss = F.cross_entropy(text_to_all_graph, labels)
            
            # 计算标准图到文本损失（不使用硬负样本）
            g2t_loss = F.cross_entropy(similarity.t(), labels)
            
            # 更新对比损失为增强版本
            contrastive_loss = (enhanced_t2g_loss + g2t_loss) / 2
        
        # 初始化结果字典
        result_dict = {
            'contrastive_loss': contrastive_loss,
            'text_to_graph_acc': contrastive_results['text_to_graph_acc'],
            'graph_to_text_acc': contrastive_results['graph_to_text_acc'],
            'accuracy': (contrastive_results['text_to_graph_acc'] + 
                        contrastive_results['graph_to_text_acc']) / 2
        }
        
        # 如果权重大于0且数据提供，则计算三元组损失
        if self.triplet_weight > 0 and triplet_data is not None:
            anchors, positives, negatives = triplet_data
            triplet_results = self.triplet(anchors, positives, negatives)
            triplet_loss = triplet_results['loss']
            
            # 将三元组损失添加到结果字典
            result_dict['triplet_loss'] = triplet_loss
            
            # 将三元组指标添加到结果字典
            for k, v in triplet_results.items():
                if k != 'loss':
                    result_dict[f'triplet_{k}'] = v
                    
            # 组合损失
            combined_loss = self.contrastive_weight * contrastive_loss + self.triplet_weight * triplet_loss
        else:
            # 仅使用对比损失
            combined_loss = contrastive_loss
        
        # 将组合损失添加到结果字典
        result_dict['loss'] = combined_loss
        
        return result_dict

class HardNegativeMiningLoss(nn.Module):
    """基于在线硬负挖掘的对比损失"""
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.3,
        mining_strategy: str = "semi-hard",
        negative_ratio: float = 0.5
    ):
        """
        初始化硬负挖掘损失
        
        Args:
            temperature: 温度参数
            margin: 半硬负挖掘边距参数
            mining_strategy: 挖掘策略（'hard', 'semi-hard', or 'distance'）
            negative_ratio: 负挖掘比例
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
        挖掘硬负样本
        
        Args:
            anchors: 锚点嵌入（batch_size x dim）
            candidates: 候选嵌入（num_candidates x dim）
            positive_mask: 二进制掩码指示正样本对（batch_size x num_candidates）
            
        Returns:
            硬负样本的索引
        """
        # 计算相似度矩阵
        similarity = torch.matmul(anchors, candidates.t())
        
        # 创建负样本掩码（正样本掩码的补集）
        negative_mask = ~positive_mask.bool()
        
        # 应用挖掘策略
        if self.mining_strategy == "hard":
            # 硬负样本：最相似的负样本
            similarity_masked = similarity.clone()
            similarity_masked[~negative_mask] = -float('inf')
            hard_indices = similarity_masked.argsort(dim=1, descending=True)
        
        elif self.mining_strategy == "semi-hard":
            # 半硬负样本：比正样本更相似的负样本
            positive_sim = (similarity * positive_mask).sum(dim=1, keepdim=True) / positive_mask.sum(dim=1, keepdim=True).clamp(min=1)
            semi_hard_mask = (similarity > positive_sim - self.margin) & negative_mask
            
            # 如果没有半硬负样本，则回退到硬负样本
            if semi_hard_mask.sum() == 0:
                similarity_masked = similarity.clone()
                similarity_masked[~negative_mask] = -float('inf')
                hard_indices = similarity_masked.argsort(dim=1, descending=True)
            else:
                similarity_masked = similarity.clone()
                similarity_masked[~semi_hard_mask] = -float('inf')
                hard_indices = similarity_masked.argsort(dim=1, descending=True)
        
        elif self.mining_strategy == "distance":
            # 基于相似度分布的样本：根据相似度分布采样
            similarity_masked = similarity.clone()
            similarity_masked[~negative_mask] = -float('inf')
            
            # 应用softmax获取采样概率
            prob = F.softmax(similarity_masked / self.temperature, dim=1)
            
            # 根据概率采样索引
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
        前向传播
        
        Args:
            anchors: 锚点嵌入（batch_size x dim）
            candidates: 候选嵌入（num_candidates x dim）
            positive_mask: 二进制掩码指示正样本对（batch_size x num_candidates）
            
        Returns:
            Dictionary containing:
                - loss: 硬负挖掘损失值
                - accuracy: 预测准确率
                - hard_negative_indices: 硬负样本的索引
        """
        # 归一化嵌入
        anchors = F.normalize(anchors, p=2, dim=1)
        candidates = F.normalize(candidates, p=2, dim=1)
        
        # 挖掘硬负样本
        hard_negative_indices = self.mine_hard_negatives(anchors, candidates, positive_mask)
        
        # 收集硬负样本
        batch_size = anchors.size(0)
        num_hard_negatives = hard_negative_indices.size(1)
        hard_negatives = candidates[hard_negative_indices.view(-1)].view(batch_size, num_hard_negatives, -1)
        
        # 计算正样本相似度
        positive_sim = (torch.matmul(anchors, candidates.t()) * positive_mask).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
        
        # 计算负样本相似度
        negative_sim = torch.bmm(anchors.unsqueeze(1), hard_negatives.transpose(1, 2)).squeeze(1)
        
        # 计算损失
        logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1) / self.temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchors.device)
        loss = F.cross_entropy(logits, labels)
        
        # 计算准确率
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'hard_negative_indices': hard_negative_indices
        } 