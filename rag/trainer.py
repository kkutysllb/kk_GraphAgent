#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练器模块，负责模型训练和评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import os
import json
from utils.logger import Logger
from .encoder import DualEncoder

class Trainer:
    """训练器类
    
    负责模型训练、评估和保存
    """
    
    def __init__(
        self,
        model: DualEncoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """初始化训练器
        
        Args:
            model: 双通道编码器模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            learning_rate: 学习率
            weight_decay: 权重衰减
            device: 训练设备
        """
        self.logger = Logger(self.__class__.__name__)
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10
        )
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # 训练历史
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": []
        }
        
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch
        
        Returns:
            训练损失和准确率的元组
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}"):
            # 将数据移到设备
            node_features = batch["node_features"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            edge_features = batch["edge_features"].to(self.device)
            texts = batch["texts"]
            batch_idx = batch["batch"].to(self.device)
            
            # 前向传播
            graph_embeddings, text_embeddings = self.model(
                node_features,
                edge_index,
                edge_features,
                texts,
                batch_idx
            )
            
            # 计算相似度和损失
            similarity = self.model.compute_similarity(
                graph_embeddings,
                text_embeddings
            )
            loss = self.model.compute_loss(similarity)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 计算准确率
            predictions = torch.argmax(similarity, dim=1)
            labels = torch.arange(len(texts)).to(self.device)
            correct = (predictions == labels).sum().item()
            
            # 更新统计
            total_loss += loss.item()
            total_correct += correct
            total_samples += len(texts)
            
        # 更新学习率
        self.scheduler.step()
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
        
    def validate(self) -> Tuple[float, float]:
        """验证模型
        
        Returns:
            验证损失和准确率的元组
        """
        if not self.val_loader:
            return 0.0, 0.0
            
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # 将数据移到设备
                node_features = batch["node_features"].to(self.device)
                edge_index = batch["edge_index"].to(self.device)
                edge_features = batch["edge_features"].to(self.device)
                texts = batch["texts"]
                batch_idx = batch["batch"].to(self.device)
                
                # 前向传播
                graph_embeddings, text_embeddings = self.model(
                    node_features,
                    edge_index,
                    edge_features,
                    texts,
                    batch_idx
                )
                
                # 计算相似度和损失
                similarity = self.model.compute_similarity(
                    graph_embeddings,
                    text_embeddings
                )
                loss = self.model.compute_loss(similarity)
                
                # 计算准确率
                predictions = torch.argmax(similarity, dim=1)
                labels = torch.arange(len(texts)).to(self.device)
                correct = (predictions == labels).sum().item()
                
                # 更新统计
                total_loss += loss.item()
                total_correct += correct
                total_samples += len(texts)
                
        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
        
    def train(
        self,
        num_epochs: int,
        save_dir: str,
        save_freq: int = 1,
        early_stopping: int = 5
    ):
        """训练模型
        
        Args:
            num_epochs: 训练轮数
            save_dir: 模型保存目录
            save_freq: 保存频率（每多少个epoch保存一次）
            early_stopping: 早停轮数
        """
        os.makedirs(save_dir, exist_ok=True)
        no_improvement = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            
            # 验证
            if self.val_loader:
                val_loss, val_acc = self.validate()
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)
                
                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_path = os.path.join(save_dir, "best_model.pt")
                    self.save_checkpoint(self.best_model_path)
                    no_improvement = 0
                else:
                    no_improvement += 1
                    
                # 早停
                if no_improvement >= early_stopping:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                    
            # 定期保存
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                self.save_checkpoint(checkpoint_path)
                
            # 打印进度
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
            )
            if self.val_loader:
                self.logger.info(
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
                
        # 保存训练历史
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
            
    def save_checkpoint(self, path: str):
        """保存检查点
        
        Args:
            path: 保存路径
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "history": self.history
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"模型已保存到 {path}")
        
    def load_checkpoint(self, path: str):
        """加载检查点
        
        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["current_epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]
        
        self.logger.info(f"模型已从 {path} 加载")
