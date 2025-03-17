#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-16 15:00
# @Desc   : 生成完整的训练、验证和测试数据集
# --------------------------------------------------------
"""

import os
import sys
import logging
import json
import time
import argparse
import random
import torch
import numpy as np
from pathlib import Path
import concurrent.futures
import threading
from tqdm import tqdm
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.utils.neo4j_graph_manager import Neo4jGraphManager
from rag.feature_extractor import FeatureExtractor
from rag.data.dataset import GraphTextDataset
from rag.utils.config import get_database_config, get_dataset_config, get_graph_config

# 添加线程本地存储，避免多线程共享连接
thread_local = threading.local()

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_thread_local():
    """初始化线程局部变量"""
    if not hasattr(thread_local, "graph_manager"):
        db_config = get_database_config()
        thread_local.graph_manager = Neo4jGraphManager(
            uri=db_config.get('uri'),
            user=db_config.get('user'),
            password=db_config.get('password')
        )
        logger.debug("为线程创建了新的数据库连接")

def get_thread_graph_manager():
    """获取线程局部的图管理器"""
    if not hasattr(thread_local, "graph_manager"):
        init_thread_local()
    return thread_local.graph_manager

def generate_dataset(
    output_dir: str,
    balance_node_types: bool = True,
    adaptive_subgraph_size: bool = True,
    negative_sample_ratio: float = 0.3,
    data_augmentation: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 8,
    seed: int = 42
):
    """生成完整的训练、验证和测试数据集"""
    start_time = time.time()
    logger.info("开始生成数据集...")
    logger.info(f"参数: balance_node_types={balance_node_types}, adaptive_subgraph_size={adaptive_subgraph_size}, "
                f"negative_sample_ratio={negative_sample_ratio}, data_augmentation={data_augmentation}, "
                f"train_ratio={train_ratio}, val_ratio={val_ratio}, test_ratio={test_ratio}, "
                f"num_workers={num_workers}, seed={seed}")
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 从配置文件加载数据库连接信息
    db_config = get_database_config()
    dataset_config = get_dataset_config()
    graph_config = get_graph_config()
    
    # 连接Neo4j数据库
    try:
        # 创建主图管理器
        graph_manager = Neo4jGraphManager(
            uri=db_config.get('uri'),
            user=db_config.get('user'),
            password=db_config.get('password')
        )
        logger.info("成功连接到Neo4j数据库")
        
        # 获取节点类型统计信息
        query_time_start = time.time()
        results = graph_manager.execute_cypher_query(
            "MATCH (n) RETURN labels(n)[0] as type, count(n) as count"
        )
        
        # 统计节点类型
        node_type_counts = {}
        for record in results:
            if record.get('type'):
                node_type = record['type']
                count = record['count']
                node_type_counts[node_type] = count
        
        query_time = time.time() - query_time_start
        logger.info(f"获取节点类型统计信息耗时: {query_time:.2f}秒")
        logger.info(f"数据库中共有 {sum(node_type_counts.values())} 个节点，分为 {len(node_type_counts)} 种类型")
        
        for node_type, count in node_type_counts.items():
            logger.info(f"节点类型 {node_type}: {count} 个")
        
        # 创建特征提取器
        feature_extractor = FeatureExtractor(
            graph_manager=graph_manager,
            node_types=graph_config.get('node_types'),
            relationship_types=graph_config.get('edge_types')
        )
        
        # 设置日志级别，暂时降低GraphTextDataset的日志级别，避免重复日志
        dataset_logger = logging.getLogger('GraphTextDataset')
        original_level = dataset_logger.level
        dataset_logger.setLevel(logging.WARNING)
        
        try:
            # 设置数据集分割比例
            split_ratio = {
                "train": train_ratio,
                "val": val_ratio,
                "test": test_ratio
            }
            
            # 创建训练集
            logger.info("创建训练集...")
            train_dataset = GraphTextDataset(
                graph_manager=graph_manager,
                feature_extractor=feature_extractor,
                node_types=graph_config.get('node_types'),
                edge_types=graph_config.get('edge_types'),
                max_node_size=dataset_config.get('max_node_size', 50),
                max_edge_size=dataset_config.get('max_edge_size', 100),
                include_dynamic=dataset_config.get('include_dynamic', True),
                data_augmentation=data_augmentation,
                balance_node_types=balance_node_types,
                adaptive_subgraph_size=adaptive_subgraph_size,
                negative_sample_ratio=negative_sample_ratio,
                split="train",
                split_ratio=split_ratio,
                seed=seed,
                skip_internal_split=True
            )
            logger.info(f"训练集创建完成，包含 {len(train_dataset.pairs)} 个样本")
            
            # 创建验证集
            logger.info("创建验证集...")
            val_dataset = GraphTextDataset(
                graph_manager=graph_manager,
                feature_extractor=feature_extractor,
                node_types=graph_config.get('node_types'),
                edge_types=graph_config.get('edge_types'),
                max_node_size=dataset_config.get('max_node_size', 50),
                max_edge_size=dataset_config.get('max_edge_size', 100),
                include_dynamic=dataset_config.get('include_dynamic', True),
                data_augmentation=False,  # 验证集不使用数据增强
                balance_node_types=False,  # 验证集不平衡节点类型
                adaptive_subgraph_size=adaptive_subgraph_size,
                negative_sample_ratio=negative_sample_ratio * 0.5,  # 验证集负样本比例减半
                split="val",
                split_ratio=split_ratio,
                seed=seed,
                skip_internal_split=True
            )
            logger.info(f"验证集创建完成，包含 {len(val_dataset.pairs)} 个样本")
            
            # 创建测试集
            logger.info("创建测试集...")
            test_dataset = GraphTextDataset(
                graph_manager=graph_manager,
                feature_extractor=feature_extractor,
                node_types=graph_config.get('node_types'),
                edge_types=graph_config.get('edge_types'),
                max_node_size=dataset_config.get('max_node_size', 50),
                max_edge_size=dataset_config.get('max_edge_size', 100),
                include_dynamic=dataset_config.get('include_dynamic', True),
                data_augmentation=False,  # 测试集不使用数据增强
                balance_node_types=False,  # 测试集不平衡节点类型
                adaptive_subgraph_size=adaptive_subgraph_size,
                negative_sample_ratio=negative_sample_ratio * 0.5,  # 测试集负样本比例减半
                split="test",
                split_ratio=split_ratio,
                seed=seed,
                skip_internal_split=True
            )
            logger.info(f"测试集创建完成，包含 {len(test_dataset.pairs)} 个样本")
            
            # 保存数据集
            logger.info("保存数据集...")
            
            # 创建保存数据的目录
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 定义保存数据集的函数
            def save_dataset_to_json(dataset, file_path):
                # 提取需要保存的数据
                data_to_save = []
                for i in range(len(dataset)):
                    item = dataset[i]
                    
                    # 提取或生成特征
                    node_id = item['node_id']
                    subgraph = item['subgraph']
                    
                    # 使用特征提取器获取节点特征
                    node_features = feature_extractor.extract_node_features(node_id)
                    
                    # 提取边特征
                    edge_features = {}
                    for edge in subgraph.get('edges', []):
                        edge_id = edge.get('id')
                        if edge_id:
                            edge_features[edge_id] = feature_extractor.extract_edge_features(edge_id)
                    
                    # 保存完整的数据，包括特征
                    data_to_save.append({
                        'node_id': item['node_id'],
                        'text': item['text'],
                        'subgraph': item['subgraph'],
                        'node_type': item['node_type'],
                        'node_features': node_features.tolist() if isinstance(node_features, torch.Tensor) else node_features,
                        'edge_features': {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in edge_features.items()},
                        'label': item.get('label', 1),  # 正样本默认标签为1
                        'is_negative': item.get('is_negative', False),
                        'negative_type': item.get('negative_type', None)
                    })
                
                # 保存为JSON文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"数据集已保存到 {file_path}，包含 {len(data_to_save)} 个样本")
            
            # 保存训练集
            train_file = output_path / "train_dataset.json"
            save_dataset_to_json(train_dataset, train_file)
            logger.info(f"训练集已保存到 {train_file}")
            
            # 保存验证集
            val_file = output_path / "val_dataset.json"
            save_dataset_to_json(val_dataset, val_file)
            logger.info(f"验证集已保存到 {val_file}")
            
            # 保存测试集
            test_file = output_path / "test_dataset.json"
            save_dataset_to_json(test_dataset, test_file)
            logger.info(f"测试集已保存到 {test_file}")
            
            # 保存数据集配置信息
            config_file = output_path / "dataset_config.json"
            node_types = graph_config.get('node_types')
            edge_types = graph_config.get('edge_types')
            config_info = {
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "test_size": len(test_dataset),
                "node_types": node_types,
                "edge_types": edge_types,
                "balance_node_types": balance_node_types,
                "adaptive_subgraph_size": adaptive_subgraph_size,
                "negative_sample_ratio": negative_sample_ratio,
                "data_augmentation": data_augmentation,
                "split_ratio": split_ratio,
                "seed": seed,
                "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, ensure_ascii=False, indent=2)
            logger.info(f"数据集配置信息已保存到 {config_file}")
            
            # 保存样本示例
            logger.info("保存样本示例...")
            
            def save_sample_examples(dataset, output_file, count=10):
                """保存样本示例"""
                # 按节点类型分组
                samples_by_type = {}
                for i in range(min(len(dataset.pairs), 100)):  # 只处理前100个样本
                    sample = dataset.pairs[i]
                    node_type = sample["node_type"]
                    
                    if node_type not in samples_by_type:
                        samples_by_type[node_type] = []
                    
                    if len(samples_by_type[node_type]) < count:  # 每种类型最多保存count个示例
                        # 创建简化版本的样本
                        simplified_sample = {
                            "node_id": sample["node_id"],
                            "node_type": sample["node_type"],
                            "text": sample["text"],
                            "is_negative": sample.get("is_negative", False),
                            "negative_type": sample.get("negative_type", None),
                            "subgraph_summary": {
                                "node_count": len(sample["subgraph"]["nodes"]),
                                "edge_count": len(sample["subgraph"]["edges"]),
                                "center_node_id": sample["subgraph"]["center_node_id"]
                            }
                        }
                        samples_by_type[node_type].append(simplified_sample)
                
                # 合并所有示例
                examples = []
                for node_type, samples in samples_by_type.items():
                    examples.extend(samples)
                
                # 保存示例
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(examples, f, ensure_ascii=False, indent=2)
                logger.info(f"保存了 {len(examples)} 个样本示例到 {output_file}")
            
            # 并行保存样本示例
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                train_future = executor.submit(save_sample_examples, train_dataset, output_path / "train_samples.json")
                val_future = executor.submit(save_sample_examples, val_dataset, output_path / "val_samples.json")
                test_future = executor.submit(save_sample_examples, test_dataset, output_path / "test_samples.json")
                
                # 等待所有保存任务完成
                for future in concurrent.futures.as_completed([train_future, val_future, test_future]):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"保存样本示例出错: {e}")
            
            total_time = time.time() - start_time
            logger.info(f"数据集生成完成，总耗时: {total_time:.2f}秒，结果保存在 {output_path} 目录")
            
        finally:
            # 恢复原来的日志级别
            dataset_logger.setLevel(original_level)
            
            # 关闭所有线程本地的数据库连接
            try:
                if hasattr(thread_local, "graph_manager"):
                    thread_local.graph_manager.close()
            except:
                pass
            
            # 关闭主数据库连接
            try:
                graph_manager.close()
            except:
                pass
        
    except Exception as e:
        logger.error(f"数据集生成过程出错: {e}")
        raise

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成完整的训练、验证和测试数据集")
    parser.add_argument("--output_dir", type=str, default="datasets", help="输出目录")
    parser.add_argument("--balance", action="store_true", help="是否平衡节点类型")
    parser.add_argument("--adaptive", action="store_true", help="是否使用自适应子图大小")
    parser.add_argument("--negative_ratio", type=float, default=0.3, help="负样本比例")
    parser.add_argument("--augmentation", action="store_true", help="是否使用数据增强")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="测试集比例")
    parser.add_argument("--workers", type=int, default=8, help="工作线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    # 生成数据集
    generate_dataset(
        output_dir=args.output_dir,
        balance_node_types=args.balance,
        adaptive_subgraph_size=args.adaptive,
        negative_sample_ratio=args.negative_ratio,
        data_augmentation=args.augmentation,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.workers,
        seed=args.seed
    ) 