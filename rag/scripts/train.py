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
from typing import Dict, Optional, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.models.dual_encoder import DualEncoder
from rag.models.loss import (
    BatchContrastiveLoss,
    MultiPositiveLoss,
    CombinedLoss,
    HardNegativeMiningLoss
)
from rag.data.dataset import GraphTextDataset
from rag.training.trainer import Trainer
from rag.utils.logging import setup_logging
from rag.utils.tools import load_checkpoint

# 创建训练结果目录
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 设置日志
logger = None  # 初始化为None，在TrainingManager中设置

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
        
        # 创建实验目录结构（不使用logger）
        self._create_experiment_dirs()
        
        # 设置日志
        global logger
        logger = setup_logging("dual_encoder_training", os.path.join(self.log_dir, 'training.log'))
        
        # 现在可以使用logger了
        logger.info(f"使用设备: {self.device}")
        logger.info(f"创建实验目录: {self.experiment_dir}")
        logger.info(f"日志目录: {self.log_dir}")
        logger.info(f"检查点目录: {self.checkpoint_dir}")
        logger.info(f"评估结果目录: {self.eval_dir}")
        logger.info(f"可视化结果目录: {self.vis_dir}")
        
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
        
    def _create_experiment_dirs(self) -> None:
        """创建实验目录结构（不使用logger）"""
        # 生成实验名称（使用时间戳）
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = self.config.get('experiment_name', 'dual_encoder')
        self.experiment_name = f"{exp_name}_{timestamp}"
        
        # 创建主实验目录
        self.experiment_dir = os.path.join(RESULTS_DIR, self.experiment_name)
        
        # 创建子目录
        self.log_dir = os.path.join(self.experiment_dir, 'logs')  # 日志目录
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')  # 检查点目录
        self.eval_dir = os.path.join(self.experiment_dir, 'evaluation')  # 评估结果目录
        self.vis_dir = os.path.join(self.experiment_dir, 'visualization')  # 可视化结果目录
        
        # 创建所有目录
        for dir_path in [self.log_dir, self.checkpoint_dir, self.eval_dir, self.vis_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
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
        
        # 设置日志
        self._setup_logging()
        
        logger.info("训练环境设置完成")
    
    def _setup_data(self) -> None:
        """设置数据集和数据加载器"""
        data_config = self.config.get('data', {})
        dataset_path = data_config.get('dataset_path', 'datasets/full_dataset')
        
        logger.info(f"从 {dataset_path} 加载数据集...")
        
        # 加载数据集配置
        dataset_stats = torch.load(os.path.join(dataset_path, 'dataset_stats.pt'))
        dataset_config = dataset_stats['config']
            
        logger.info(f"数据集配置: {dataset_config}")
        
        # 创建数据集
        train_dataset = GraphTextDataset.load(
            os.path.join(dataset_path, 'train.pt')
        )
        val_dataset = GraphTextDataset.load(
            os.path.join(dataset_path, 'val.pt')
        )
        
        # 确保数据集有collate_fn方法
        if not hasattr(train_dataset, 'collate_fn'):
            train_dataset.collate_fn = GraphTextDataset.collate_fn
        if not hasattr(val_dataset, 'collate_fn'):
            val_dataset.collate_fn = GraphTextDataset.collate_fn
        
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
            text_model_name=model_config.get('text_model_name', 'bert-base-chinese'),
            node_dim=model_config.get('node_dim', 256),
            edge_dim=model_config.get('edge_dim', 64),
            time_series_dim=model_config.get('time_series_dim', 32),
            hidden_dim=model_config.get('hidden_dim', 256),
            output_dim=model_config.get('output_dim', 768),
            num_layers=model_config.get('num_layers', 2),
            num_heads=model_config.get('num_heads', 8),
            dropout=model_config.get('dropout', 0.1),
            freeze_text=model_config.get('freeze_text', False),
            node_types=model_config.get('node_types', ["DC", "TENANT", "NE", "VM", "HOST", "HA", "TRU"]),
            edge_types=model_config.get('edge_types', ["DC_TO_TENANT", "TENANT_TO_NE", "NE_TO_VM", "VM_TO_HOST", "HOST_TO_HA", "HA_TO_TRU"]),
            seq_len=model_config.get('seq_len', 24),
            num_hierarchies=model_config.get('num_hierarchies', 7)
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
        
        # 创建训练器，使用checkpoint_dir保存检查点
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
            checkpoint_dir=self.checkpoint_dir,
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
        
        # 保存图表到可视化目录
        chart_path = os.path.join(self.vis_dir, 'training_progress.png')
        plt.savefig(chart_path)
        plt.close()
        
        logger.info(f"训练进度图表已保存到: {chart_path}")
        
        # 保存训练历史数据
        history_path = os.path.join(self.vis_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
        logger.info(f"训练历史数据已保存到: {history_path}")
    
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

    def _setup_logging(self) -> None:
        """设置日志"""
        # 创建日志目录
        log_dir = os.path.join(self.experiment_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志文件
        log_file = os.path.join(log_dir, 'training.log')
        
        # 配置日志处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        logger.addHandler(file_handler)
        
        logger.info(f"日志文件保存在: {log_file}")

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