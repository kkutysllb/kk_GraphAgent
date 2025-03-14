#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import sys
from datetime import datetime

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别，默认为INFO
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 如果已经有处理器，不重复添加
    if logger.handlers:
        return logger
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 创建文件处理器
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'logs'
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 设置格式化器
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

class Logger:
    """日志记录器类"""
    
    def __init__(self, name):
        """初始化日志记录器
        
        Args:
            name: 日志记录器名称
        """
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 如果已经有处理器，不再添加
        if not self.logger.handlers:
            # 创建控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # 创建格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            
            # 添加处理器到日志记录器
            self.logger.addHandler(console_handler)
    
    def info(self, message):
        """记录信息级别的日志"""
        self.logger.info(message)
    
    def warning(self, message):
        """记录警告级别的日志"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录错误级别的日志"""
        self.logger.error(message)
    
    def debug(self, message):
        """记录调试级别的日志"""
        self.logger.debug(message)
    
    def critical(self, message):
        """记录严重错误级别的日志"""
        self.logger.critical(message) 