"""
配置工具模块
用于加载和管理应用配置
"""
import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("agno.utils.config")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    从JSON文件加载配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}，将使用默认配置")
        return get_default_config()
    
    try:
        # 从文件加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"成功从 {config_path} 加载配置")
        
        # 与默认配置合并
        merged_config = {**get_default_config(), **config}
        
        # 从环境变量覆盖敏感配置
        merged_config = override_from_env(merged_config)
        
        return merged_config
    
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}", exc_info=True)
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    
    Returns:
        默认配置字典
    """
    return {
        "version": "1.0.0",
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        },
        "vector_db": {
            "host": "localhost",
            "port": 19530,
            "collection": "agno_vectors"
        },
        "llm_endpoint": "http://localhost:8000/v1",
        "llm_model": "qwen2.5-7b",
        "embedding_model": "bge-base-zh-v1.5",
        "cache_dir": "cache",
        "use_cache": True,
        "log_level": "INFO",
        "max_tokens": 1024,
        "max_results": 20,
        "enable_vector_retrieval": True,
        "enable_topology_analysis": True,
        "enable_performance_analysis": True,
        "enable_fault_diagnosis": True
    }

def override_from_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    从环境变量覆盖配置
    
    Args:
        config: 原始配置字典
    
    Returns:
        更新后的配置字典
    """
    # 复制配置以避免修改原始配置
    result = config.copy()
    
    # Neo4j配置
    if os.environ.get("NEO4J_URI"):
        result["neo4j"]["uri"] = os.environ.get("NEO4J_URI")
    
    if os.environ.get("NEO4J_USER"):
        result["neo4j"]["user"] = os.environ.get("NEO4J_USER")
    
    if os.environ.get("NEO4J_PASSWORD"):
        result["neo4j"]["password"] = os.environ.get("NEO4J_PASSWORD")
    
    # 向量数据库配置
    if os.environ.get("VECTOR_DB_HOST"):
        result["vector_db"]["host"] = os.environ.get("VECTOR_DB_HOST")
    
    if os.environ.get("VECTOR_DB_PORT"):
        result["vector_db"]["port"] = int(os.environ.get("VECTOR_DB_PORT"))
    
    # LLM配置
    if os.environ.get("LLM_ENDPOINT"):
        result["llm_endpoint"] = os.environ.get("LLM_ENDPOINT")
    
    if os.environ.get("LLM_MODEL"):
        result["llm_model"] = os.environ.get("LLM_MODEL")
    
    if os.environ.get("EMBEDDING_MODEL"):
        result["embedding_model"] = os.environ.get("EMBEDDING_MODEL")
    
    # 其他配置
    if os.environ.get("CACHE_DIR"):
        result["cache_dir"] = os.environ.get("CACHE_DIR")
    
    if os.environ.get("USE_CACHE"):
        result["use_cache"] = os.environ.get("USE_CACHE").lower() in ("yes", "true", "1")
    
    if os.environ.get("LOG_LEVEL"):
        result["log_level"] = os.environ.get("LOG_LEVEL")
    
    return result

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    保存配置到JSON文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径
    
    Returns:
        是否成功保存
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # 保存配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"成功保存配置到 {config_path}")
        return True
    
    except Exception as e:
        logger.error(f"保存配置文件失败: {str(e)}", exc_info=True)
        return False 