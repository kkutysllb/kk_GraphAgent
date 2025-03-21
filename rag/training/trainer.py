#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:49
# @Desc   : Training module for the RAG model.
# --------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Tuple
import logging
import os
from tqdm import tqdm

from ..models.dual_encoder import DualEncoder
from ..models.loss import (
    ContrastiveLoss, 
    InfoNCELoss, 
    TripletLoss,
    BatchContrastiveLoss,
    MultiPositiveLoss,
    CombinedLoss,
    HardNegativeMiningLoss
)
from ..utils.logging import LoggerMixin
from ..utils.tools import save_checkpoint, load_checkpoint, get_lr

logger = logging.getLogger(__name__)

class Trainer(LoggerMixin):
    """Trainer class for the RAG model"""
    
    def __init__(
        self,
        model: DualEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_type: str = "contrastive",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        num_epochs: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        early_stopping_patience: int = 10
    ):
        """
        Initialize trainer
        
        Args:
            model: DualEncoder model
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_type: Type of loss function (contrastive, infonce, triplet, batch_contrastive, multi_positive, hard_negative_mining, combined)
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm
            num_epochs: Number of training epochs
            device: Device to use
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Number of epochs to wait before early stopping
        """
        super().__init__()
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience
        
        # Create loss function
        if loss_type == "contrastive":
            self.loss_fn = ContrastiveLoss()
        elif loss_type == "infonce":
            self.loss_fn = InfoNCELoss()
        elif loss_type == "triplet":
            self.loss_fn = TripletLoss()
        elif loss_type == "batch_contrastive":
            self.loss_fn = BatchContrastiveLoss()
        elif loss_type == "multi_positive":
            self.loss_fn = MultiPositiveLoss()
        elif loss_type == "hard_negative_mining":
            self.loss_fn = HardNegativeMiningLoss()
        elif loss_type == "combined":
            self.loss_fn = CombinedLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Move loss function to device
        self.loss_fn = self.loss_fn.to(device)
            
        # Create optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.1
        )
        
        # Training parameters
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.global_step = 0
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # 检查批次是否为空
            if not batch:
                continue
                
            # 处理批次数据
            text_encodings = batch['text_encodings']
            input_ids = text_encodings['input_ids'].to(self.device)
            attention_mask = text_encodings['attention_mask'].to(self.device)
            
            node_features = batch['node_features'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_features = batch['edge_features'].to(self.device)
            
            # 前向传播
            text_embeddings, graph_embeddings = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features
            )
            
            # 计算损失
            loss_dict = self.loss_fn(
                text_embeddings=text_embeddings,
                graph_embeddings=graph_embeddings
            )
            
            loss = loss_dict['loss']
            
            # 获取准确率指标
            accuracy = (loss_dict.get('text_to_graph_acc', 0.0) + loss_dict.get('graph_to_text_acc', 0.0)) / 2
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
            # Update weights
            self.optimizer.step()
            
            # Update learning rate
            if self.global_step < self.warmup_steps:
                lr_scale = min(1.0, float(self.global_step + 1) / self.warmup_steps)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr_scale * self.scheduler.get_last_lr()[0]
                    
            # Update metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'accuracy': accuracy,
                'lr': get_lr(self.optimizer)
            })
            
            self.global_step += 1
            
        # Update scheduler
        self.scheduler.step()
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
        
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for batch in pbar:
                # 检查批次是否为空
                if not batch:
                    continue
                    
                # 处理批次数据
                text_encodings = batch['text_encodings']
                input_ids = text_encodings['input_ids'].to(self.device)
                attention_mask = text_encodings['attention_mask'].to(self.device)
                
                node_features = batch['node_features'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                edge_features = batch['edge_features'].to(self.device)
                
                # 前向传播
                text_embeddings, graph_embeddings = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_features=edge_features
                )
                
                # 计算损失
                loss_dict = self.loss_fn(
                    text_embeddings=text_embeddings,
                    graph_embeddings=graph_embeddings
                )
                
                loss = loss_dict['loss']
                
                # 获取准确率
                accuracy = (loss_dict.get('text_to_graph_acc', 0.0) + loss_dict.get('graph_to_text_acc', 0.0)) / 2
                
                # 更新指标
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': loss.item(),
                    'accuracy': accuracy
                })
        
        # 计算平均指标
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
        
    def train(self) -> Dict[str, list]:
        """Train the model"""
        self.log_info("Starting training...")
        
        for epoch in range(self.num_epochs):
            self.log_info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Log metrics
            self.log_info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Accuracy: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Accuracy: {val_metrics['accuracy']:.4f}"
            )
            
            # Save checkpoint if validation loss improved
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_accuracy = val_metrics['accuracy']
                self.patience_counter = 0
                
                # Save checkpoint
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=val_metrics['loss'],
                    metrics=val_metrics,
                    checkpoint_dir=self.checkpoint_dir,
                    name="best_model"
                )
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                self.log_info(
                    f"Early stopping triggered after {epoch + 1} epochs. "
                    f"Best validation loss: {self.best_val_loss:.4f}, "
                    f"Best validation accuracy: {self.best_val_accuracy:.4f}"
                )
                break
                
        return self.history
        
    def load_best_model(self) -> Tuple[nn.Module, Dict[str, float]]:
        """Load the best model from checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        model, _, checkpoint_info = load_checkpoint(
            checkpoint_path,
            self.model
        )
        return model, checkpoint_info 