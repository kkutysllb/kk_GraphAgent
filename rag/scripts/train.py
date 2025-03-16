#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-25 14:30
# @Desc   : 双通道编码器训练脚本
# --------------------------------------------------------
"""

import os
import sys
import torch
import argparse
import json
import datetime
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.models.dual_encoder import DualEncoder
from rag.models.loss import (
    ContrastiveLoss,
    InfoNCELoss,
    TripletLoss,
    BatchContrastiveLoss,
    MultiPositiveLoss,
    CombinedLoss,
    HardNegativeMiningLoss
)
from rag.data.graph_text_dataset import GraphTextDataset
from rag.training.trainer import Trainer
from rag.utils.logging import setup_logger
from rag.utils.metrics import compute_metrics
from rag.utils.tools import save_checkpoint, load_checkpoint

# 创建训练结果目录
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 设置日志
logger = setup_logger("dual_encoder_training")

class TrainingManager:
    """双通道编码器训练管理器"""
    
    def __init__(
        self,
        config_path: str,
        device: Optional[str] = None
    ):
        """
        初始化训练管理器
        
        Args:
            config_path: 配置文件路径
            device: 设备（'cuda'或'cpu'）
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 设置设备
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 创建结果目录
        self.experiment_name = self.config.get('experiment_name', f"dual_encoder_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir = os.path.join(RESULTS_DIR, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 保存配置
        self._save_config()
        
        # 初始化组件
        self.model = None
        self.trainer = None
        self.train_loader = None
        self.val_loader = None
        
        # 训练状态
        self.train_metrics = []
        self.val_metrics = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _save_config(self) -> None:
        """保存配置到实验目录"""
        config_path = os.path.join(self.experiment_dir, 'config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"配置已保存到: {config_path}")
    
    def setup(self) -> None:
        """设置训练环境"""
        # 设置随机种子
        seed = self.config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 创建数据集和数据加载器
        self._setup_data()
        
        # 创建模型
        self._setup_model()
        
        # 创建训练器
        self._setup_trainer()
        
        logger.info("训练环境设置完成")
    
    def _setup_data(self) -> None:
        """设置数据集和数据加载器"""
        data_config = self.config.get('data', {})
        
        # 创建数据集
        train_dataset = GraphTextDataset(
            neo4j_uri=data_config.get('neo4j_uri', 'bolt://localhost:7687'),
            neo4j_user=data_config.get('neo4j_user', 'neo4j'),
            neo4j_password=data_config.get('neo4j_password', 'password'),
            split='train',
            max_graph_size=data_config.get('max_graph_size', 100),
            max_text_length=data_config.get('max_text_length', 128),
            augmentation=data_config.get('augmentation', True),
            negative_sampling=data_config.get('negative_sampling', 'hard')
        )
        
        val_dataset = GraphTextDataset(
            neo4j_uri=data_config.get('neo4j_uri', 'bolt://localhost:7687'),
            neo4j_user=data_config.get('neo4j_user', 'neo4j'),
            neo4j_password=data_config.get('neo4j_password', 'password'),
            split='val',
            max_graph_size=data_config.get('max_graph_size', 100),
            max_text_length=data_config.get('max_text_length', 128),
            augmentation=False,
            negative_sampling=data_config.get('negative_sampling', 'hard')
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=data_config.get('batch_size', 32),
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=train_dataset.collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=data_config.get('batch_size', 32),
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=val_dataset.collate_fn
        )
        
        logger.info(f"训练数据集大小: {len(train_dataset)}")
        logger.info(f"验证数据集大小: {len(val_dataset)}")
    
    def _setup_model(self) -> None:
        """设置模型"""
        model_config = self.config.get('model', {})
        
        # 创建双通道编码器
        self.model = DualEncoder(
            text_encoder_name=model_config.get('text_encoder_name', 'bert-base-chinese'),
            text_pooling_strategy=model_config.get('text_pooling_strategy', 'cls'),
            graph_hidden_dim=model_config.get('graph_hidden_dim', 256),
            graph_num_layers=model_config.get('graph_num_layers', 3),
            projection_dim=model_config.get('projection_dim', 768),
            dropout=model_config.get('dropout', 0.1)
        )
        
        # 打印模型信息
        logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def _setup_trainer(self) -> None:
        """设置训练器"""
        training_config = self.config.get('training', {})
        optim_config = self.config.get('optimizer', {})
        loss_config = self.config.get('loss', {})
        
        # 获取损失函数类型
        loss_type = loss_config.get('type', 'combined')
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            loss_type=loss_type,
            learning_rate=optim_config.get('lr', 2e-5),
            weight_decay=optim_config.get('weight_decay', 0.01),
            warmup_steps=training_config.get('warmup_steps', 1000),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            num_epochs=training_config.get('num_epochs', 10),
            device=self.device,
            checkpoint_dir=self.experiment_dir,
            early_stopping_patience=training_config.get('early_stopping', {}).get('patience', 3)
        )
        
        # 如果使用自定义损失函数，需要手动设置
        if loss_type in ['batch_contrastive', 'multi_positive', 'hard_negative_mining', 'combined']:
            if loss_type == 'batch_contrastive':
                self.trainer.loss_fn = BatchContrastiveLoss(
                    temperature=loss_config.get('temperature', 0.07),
                    use_hard_negatives=loss_config.get('use_hard_negatives', True)
                )
            elif loss_type == 'multi_positive':
                self.trainer.loss_fn = MultiPositiveLoss(
                    temperature=loss_config.get('temperature', 0.07)
                )
            elif loss_type == 'hard_negative_mining':
                self.trainer.loss_fn = HardNegativeMiningLoss(
                    temperature=loss_config.get('temperature', 0.07),
                    margin=loss_config.get('margin', 0.3),
                    mining_strategy=loss_config.get('mining_strategy', 'semi-hard')
                )
            elif loss_type == 'combined':
                self.trainer.loss_fn = CombinedLoss(
                    contrastive_weight=loss_config.get('contrastive_weight', 1.0),
                    triplet_weight=loss_config.get('triplet_weight', 0.5),
                    temperature=loss_config.get('temperature', 0.07),
                    margin=loss_config.get('margin', 0.3),
                    use_hard_negatives=loss_config.get('use_hard_negatives', True)
                )
            
            # 将损失函数移动到设备上
            self.trainer.loss_fn = self.trainer.loss_fn.to(self.device)
        
        logger.info(f"使用损失函数: {loss_type}")
    
    def train(self) -> None:
        """训练模型"""
        logger.info("开始训练")
        
        # 使用Trainer进行训练
        self.trainer.train()
        
        # 可视化训练过程
        self._visualize_training()
        
        logger.info("训练完成")
    
    def _visualize_training(self) -> None:
        """可视化训练过程"""
        # 提取训练历史
        history = self.trainer.history
        
        # 创建图表
        plt.figure(figsize=(12, 10))
        
        # 绘制损失
        plt.subplot(2, 1, 1)
        plt.plot(history['train_loss'], 'b-', label='训练损失')
        plt.plot(history['val_loss'], 'r-', label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('周期')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        # 绘制准确率
        plt.subplot(2, 1, 2)
        plt.plot(history['train_accuracy'], 'b-', label='训练准确率')
        plt.plot(history['val_accuracy'], 'r-', label='验证准确率')
        plt.title('训练和验证准确率')
        plt.xlabel('周期')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = os.path.join(self.experiment_dir, 'training_progress.png')
        plt.savefig(chart_path)
        plt.close()
        
        logger.info(f"训练进度图表已保存到: {chart_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        # 加载检查点
        checkpoint = load_checkpoint(checkpoint_path, self.model, self.trainer.optimizer, device=self.device)
        
        # 更新训练器状态
        self.trainer.global_step = checkpoint.get('global_step', 0)
        self.trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.trainer.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        self.trainer.patience_counter = checkpoint.get('patience_counter', 0)
        
        logger.info(f"从检查点加载模型: {checkpoint_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="双通道编码器训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, help="检查点路径（用于恢复训练）")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], help="设备")
    args = parser.parse_args()
    
    # 创建训练管理器
    manager = TrainingManager(
        config_path=args.config,
        device=args.device
    )
    
    # 设置训练环境
    manager.setup()
    
    # 加载检查点（如果提供）
    if args.checkpoint:
        manager.load_checkpoint(args.checkpoint)
    
    # 开始训练
    manager.train()

if __name__ == "__main__":
    main() 