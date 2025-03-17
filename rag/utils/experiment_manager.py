#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-17 10:30
# @Desc   : 实验管理器，用于管理训练实验
# --------------------------------------------------------
"""

import os
import json
import yaml
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import shutil
from pathlib import Path
import pandas as pd
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

from .logging import LoggerMixin

logger = logging.getLogger(__name__)

class ExperimentManager(LoggerMixin):
    """实验管理器，用于管理训练实验"""
    
    def __init__(
        self,
        base_dir: str = "experiments",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        use_tensorboard: bool = True
    ):
        """
        初始化实验管理器
        
        Args:
            base_dir: 实验基础目录
            experiment_name: 实验名称，如果为None则使用时间戳
            config: 实验配置
            use_tensorboard: 是否使用TensorBoard
        """
        super().__init__()
        
        # 设置实验名称和目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"experiment_{timestamp}"
        self.experiment_dir = os.path.join(base_dir, self.experiment_name)
        
        # 创建实验目录结构
        self._create_experiment_dirs()
        
        # 保存配置
        self.config = config or {}
        if self.config:
            self.save_config(self.config)
        
        # 初始化TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.get_tensorboard_dir())
        
        self.log_info(f"创建实验: {self.experiment_name}")
        self.log_info(f"实验目录: {self.experiment_dir}")
    
    def _create_experiment_dirs(self):
        """创建实验目录结构"""
        # 主目录
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 子目录
        os.makedirs(self.get_checkpoints_dir(), exist_ok=True)
        os.makedirs(self.get_logs_dir(), exist_ok=True)
        os.makedirs(self.get_results_dir(), exist_ok=True)
        os.makedirs(self.get_figures_dir(), exist_ok=True)
        os.makedirs(self.get_tensorboard_dir(), exist_ok=True)
    
    def get_experiment_dir(self) -> str:
        """获取实验目录"""
        return self.experiment_dir
    
    def get_checkpoints_dir(self) -> str:
        """获取检查点目录"""
        return os.path.join(self.experiment_dir, "checkpoints")
    
    def get_logs_dir(self) -> str:
        """获取日志目录"""
        return os.path.join(self.experiment_dir, "logs")
    
    def get_results_dir(self) -> str:
        """获取结果目录"""
        return os.path.join(self.experiment_dir, "results")
    
    def get_figures_dir(self) -> str:
        """获取图表目录"""
        return os.path.join(self.experiment_dir, "figures")
    
    def get_tensorboard_dir(self) -> str:
        """获取TensorBoard目录"""
        return os.path.join(self.experiment_dir, "tensorboard")
    
    def save_config(self, config: Dict[str, Any]):
        """
        保存实验配置
        
        Args:
            config: 实验配置
        """
        self.config = config
        
        # 保存为YAML
        yaml_path = os.path.join(self.experiment_dir, "config.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # 保存为JSON（便于其他程序读取）
        json_path = os.path.join(self.experiment_dir, "config.json")
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.log_info(f"保存配置到 {yaml_path} 和 {json_path}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        is_best: bool = False,
        is_last: bool = False,
        name: Optional[str] = None
    ):
        """
        保存模型检查点
        
        Args:
            model: PyTorch模型
            optimizer: PyTorch优化器
            epoch: 当前轮次
            loss: 当前损失值
            metrics: 指标字典
            is_best: 是否是最佳模型
            is_last: 是否是最后一轮模型
            name: 可选的检查点名称
        """
        # 创建检查点目录
        checkpoint_dir = self.get_checkpoints_dir()
        
        # 创建检查点名称
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"checkpoint_epoch{epoch:03d}_{timestamp}"
        
        checkpoint_path = os.path.join(checkpoint_dir, f"{name}.pt")
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.log_info(f"保存检查点到 {checkpoint_path}")
        
        # 如果是最佳模型，复制一份
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            shutil.copy(checkpoint_path, best_path)
            self.log_info(f"保存最佳模型到 {best_path}")
        
        # 如果是最后一轮模型，复制一份
        if is_last:
            last_path = os.path.join(checkpoint_dir, "last_model.pt")
            shutil.copy(checkpoint_path, last_path)
            self.log_info(f"保存最后模型到 {last_path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Dict[str, Any]]:
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点路径
            model: PyTorch模型
            optimizer: 可选的PyTorch优化器
            
        Returns:
            Tuple of (model, optimizer, checkpoint_info)
        """
        # 加载检查点
        checkpoint = torch.load(checkpoint_path)
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 获取检查点信息
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', 0.0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {})
        }
        
        self.log_info(f"加载检查点从 {checkpoint_path}")
        return model, optimizer, checkpoint_info
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ""
    ):
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 当前步骤
            prefix: 指标前缀
        """
        # 保存到JSON文件
        metrics_dir = os.path.join(self.get_results_dir(), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        metrics_file = os.path.join(metrics_dir, f"{prefix}_metrics.jsonl")
        
        # 添加时间戳和步骤
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            **metrics
        }
        
        # 追加到文件
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # 记录到TensorBoard
        if self.use_tensorboard:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{prefix}/{name}", value, step)
        
        # 记录到日志
        metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        self.log_info(f"{prefix} 步骤 {step}: {metrics_str}")
    
    def plot_training_curves(
        self,
        train_metrics: Dict[str, List[float]],
        val_metrics: Dict[str, List[float]],
        save_name: str = "training_curves"
    ):
        """
        绘制训练曲线
        
        Args:
            train_metrics: 训练指标字典
            val_metrics: 验证指标字典
            save_name: 保存文件名
        """
        # 创建图表目录
        figures_dir = self.get_figures_dir()
        
        # 获取所有指标名称
        all_metrics = set(train_metrics.keys()).union(set(val_metrics.keys()))
        
        # 为每个指标创建一个子图
        n_metrics = len(all_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
        
        # 如果只有一个指标，确保axes是一个列表
        if n_metrics == 1:
            axes = [axes]
        
        # 绘制每个指标的曲线
        for i, metric_name in enumerate(sorted(all_metrics)):
            ax = axes[i]
            
            # 绘制训练曲线
            if metric_name in train_metrics:
                ax.plot(train_metrics[metric_name], label=f'Train {metric_name}')
            
            # 绘制验证曲线
            if metric_name in val_metrics:
                ax.plot(val_metrics[metric_name], label=f'Val {metric_name}')
            
            ax.set_title(f'{metric_name} vs. Epoch')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(figures_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300)
        self.log_info(f"保存训练曲线到 {save_path}")
        
        # 关闭图表
        plt.close(fig)
    
    def plot_evaluation_results(
        self,
        results: Dict[str, Any],
        save_name: str = "evaluation_results"
    ):
        """
        绘制评估结果
        
        Args:
            results: 评估结果字典
            save_name: 保存文件名
        """
        # 创建图表目录
        figures_dir = self.get_figures_dir()
        
        # 根据结果类型创建不同的可视化
        if "confusion_matrix" in results:
            # 绘制混淆矩阵
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                results["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=results.get("labels", None),
                yticklabels=results.get("labels", None)
            )
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            
            # 保存图表
            save_path = os.path.join(figures_dir, f"{save_name}_confusion_matrix.png")
            plt.savefig(save_path, dpi=300)
            self.log_info(f"保存混淆矩阵到 {save_path}")
            plt.close()
        
        if "precision_recall_curve" in results:
            # 绘制精确率-召回率曲线
            plt.figure(figsize=(10, 8))
            for label, (precision, recall, _) in results["precision_recall_curve"].items():
                plt.plot(recall, precision, label=f"{label}")
            
            plt.title("Precision-Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            plt.grid(True)
            
            # 保存图表
            save_path = os.path.join(figures_dir, f"{save_name}_precision_recall.png")
            plt.savefig(save_path, dpi=300)
            self.log_info(f"保存精确率-召回率曲线到 {save_path}")
            plt.close()
        
        if "roc_curve" in results:
            # 绘制ROC曲线
            plt.figure(figsize=(10, 8))
            for label, (fpr, tpr, _) in results["roc_curve"].items():
                plt.plot(fpr, tpr, label=f"{label}")
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title("ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.grid(True)
            
            # 保存图表
            save_path = os.path.join(figures_dir, f"{save_name}_roc_curve.png")
            plt.savefig(save_path, dpi=300)
            self.log_info(f"保存ROC曲线到 {save_path}")
            plt.close()
        
        # 保存评估结果摘要
        summary = {k: v for k, v in results.items() if not isinstance(v, (dict, list, np.ndarray))}
        if summary:
            # 保存为JSON
            summary_path = os.path.join(self.get_results_dir(), f"{save_name}_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            self.log_info(f"保存评估结果摘要到 {summary_path}")
    
    def save_model_summary(
        self,
        model: torch.nn.Module,
        input_shapes: Optional[Dict[str, Tuple]] = None
    ):
        """
        保存模型摘要
        
        Args:
            model: PyTorch模型
            input_shapes: 输入形状字典
        """
        # 创建摘要文件
        summary_path = os.path.join(self.experiment_dir, "model_summary.txt")
        
        # 获取模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 写入摘要
        with open(summary_path, 'w') as f:
            f.write(f"Model Summary\n")
            f.write(f"=============\n\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n\n")
            
            # 写入模型结构
            f.write(f"Model Structure\n")
            f.write(f"--------------\n\n")
            f.write(str(model))
            
            # 如果提供了输入形状，尝试写入每层的输出形状
            if input_shapes:
                f.write(f"\n\nLayer Output Shapes\n")
                f.write(f"------------------\n\n")
                f.write("Not implemented yet")
        
        self.log_info(f"保存模型摘要到 {summary_path}")
    
    def close(self):
        """关闭实验管理器"""
        if self.use_tensorboard:
            self.writer.close()
        
        self.log_info(f"关闭实验: {self.experiment_name}") 