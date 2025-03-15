#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:36
# @Desc   : Utility tools module.
# --------------------------------------------------------
"""

"""
Utility tools module.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def get_device() -> torch.device:
    """Get PyTorch device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def save_json(data: Dict[str, Any], filepath: str):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
        
def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)
        
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    name: Optional[str] = None
):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        loss: Current loss value
        metrics: Dictionary of metric values
        checkpoint_dir: Directory to save checkpoint
        name: Optional name for checkpoint file
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint name
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"checkpoint_{timestamp}"
        
    checkpoint_path = os.path.join(checkpoint_dir, f"{name}.pt")
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Dict[str, Any]]:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optional PyTorch optimizer
        
    Returns:
        Tuple of (model, optimizer, checkpoint_info)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    # Get checkpoint info
    checkpoint_info = {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'metrics': checkpoint['metrics']
    }
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return model, optimizer, checkpoint_info
    
def count_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def move_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Move batch of tensors to device"""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()} 