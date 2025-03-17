#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:35
# @Desc   : 日志实用程序模块。
# --------------------------------------------------------
"""

import logging
import sys
from typing import Optional
import os
from datetime import datetime

def setup_logging(name: str, log_file: str = None) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
    
def get_experiment_logger(
    experiment_name: str,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    获取具有时间戳的实验日志记录器
    
    Args:
        experiment_name: 实验名称
        log_dir: 存储日志文件的目录
        
    Returns:
        日志实例
    """
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建日志文件路径
    log_file = os.path.join(
        log_dir,
        f"{experiment_name}_{timestamp}.log"
    )
    
    return setup_logging(experiment_name, log_file)
    
class LoggerMixin:
    """Mixin类，用于为任何类添加日志记录功能"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def log_info(self, message: str):
        """记录信息消息"""
        self.logger.info(message)
        
    def log_warning(self, message: str):
        """记录警告消息"""
        self.logger.warning(message)
        
    def log_error(self, message: str):
        """记录错误消息"""
        self.logger.error(message)
        
    def log_debug(self, message: str):
        """记录调试消息"""
        self.logger.debug(message)
        
    def log_exception(self, message: str):
        """记录异常消息"""
        self.logger.exception(message) 