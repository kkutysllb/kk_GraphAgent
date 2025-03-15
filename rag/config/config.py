#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:35
# @Desc   : 配置管理模块
# --------------------------------------------------------
"""

import yaml
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration"""
    text_model_name: str = "bert-base-chinese"  # 更改为中文预训练模型
    node_dim: int = 256
    edge_dim: int = 64
    hidden_dim: int = 256
    output_dim: int = 768
    num_heads: int = 8
    dropout: float = 0.1
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    
@dataclass
class DataConfig:
    """Data configuration"""
    max_text_length: int = 512
    max_node_size: int = 100
    max_edge_size: int = 200
    node_types: List[str] = None
    edge_types: List[str] = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
@dataclass
class Neo4jConfig:
    """Neo4j database configuration"""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "Oms_2600a"
    database: str = "neo4j"
    
@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    neo4j: Neo4jConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            neo4j=Neo4jConfig(**config_dict.get('neo4j', {}))
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': {
                'text_model_name': self.model.text_model_name,
                'node_dim': self.model.node_dim,
                'edge_dim': self.model.edge_dim,
                'hidden_dim': self.model.hidden_dim,
                'output_dim': self.model.output_dim,
                'num_heads': self.model.num_heads,
                'dropout': self.model.dropout
            },
            'training': {
                'batch_size': self.training.batch_size,
                'num_epochs': self.training.num_epochs,
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'warmup_steps': self.training.warmup_steps,
                'max_grad_norm': self.training.max_grad_norm,
                'early_stopping_patience': self.training.early_stopping_patience
            },
            'data': {
                'max_text_length': self.data.max_text_length,
                'max_node_size': self.data.max_node_size,
                'max_edge_size': self.data.max_edge_size,
                'node_types': self.data.node_types,
                'edge_types': self.data.edge_types,
                'train_ratio': self.data.train_ratio,
                'val_ratio': self.data.val_ratio,
                'test_ratio': self.data.test_ratio
            },
            'neo4j': {
                'uri': self.neo4j.uri,
                'user': self.neo4j.user,
                'password': self.neo4j.password,
                'database': self.neo4j.database
            }
        }
        
    def save(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
                
def load_config(config_path: str) -> Config:
    """Helper function to load configuration"""
    return Config.from_yaml(config_path)

def get_config(config_path: str = None) -> Dict[str, Any]:
    """
    获取配置，如果提供了配置文件路径则从文件加载，否则返回默认配置
    
    Args:
        config_path: 配置文件路径（可选）
        
    Returns:
        配置字典
    """
    if config_path and os.path.exists(config_path):
        try:
            config = load_config(config_path)
            return config.to_dict()
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            logger.info("使用默认配置")
    
    # 创建默认配置
    config = Config(
        model=ModelConfig(),
        training=TrainingConfig(),
        data=DataConfig(),
        neo4j=Neo4jConfig()
    )
    
    return config.to_dict() 