#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : ${1:kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 09:35
# @Desc   : Logging utility module.
# --------------------------------------------------------
"""

"""
Logging utility module.
"""

import logging
import sys
from typing import Optional
import os
from datetime import datetime

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file. If None, logs to stdout only
        level: Logging level
        log_format: Custom log format string
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Default format
    if log_format is None:
        log_format = '[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
        
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
    
def get_experiment_logger(
    experiment_name: str,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Get logger for experiment with timestamped log file
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
        
    Returns:
        Logger instance
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create log file path
    log_file = os.path.join(
        log_dir,
        f"{experiment_name}_{timestamp}.log"
    )
    
    return setup_logging(log_file)
    
class LoggerMixin:
    """Mixin class to add logging capability to any class"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)
        
    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
        
    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)
        
    def log_debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
        
    def log_exception(self, message: str):
        """Log exception message"""
        self.logger.exception(message) 