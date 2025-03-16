#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-26 10:30
# @Desc   : 配置加载工具
# --------------------------------------------------------
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise

def get_database_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    获取数据库配置
    
    Args:
        config_path: 配置文件路径，如果为None，则使用默认路径
        
    Returns:
        数据库配置字典
    """
    if config_path is None:
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(root_dir, 'configs', 'database_config.yaml')
    
    config = load_config(config_path)
    return config.get('neo4j', {})

def get_dataset_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    获取数据集配置
    
    Args:
        config_path: 配置文件路径，如果为None，则使用默认路径
        
    Returns:
        数据集配置字典
    """
    if config_path is None:
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(root_dir, 'configs', 'database_config.yaml')
    
    config = load_config(config_path)
    return config.get('dataset', {})

def get_graph_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    获取图配置
    
    Args:
        config_path: 配置文件路径，如果为None，则使用默认路径
        
    Returns:
        图配置字典
    """
    if config_path is None:
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(root_dir, 'configs', 'database_config.yaml')
    
    config = load_config(config_path)
    return config.get('graph', {}) 