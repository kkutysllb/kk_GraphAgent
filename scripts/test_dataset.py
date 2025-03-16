#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-26 10:00
# @Desc   : 测试图文数据集的功能
# --------------------------------------------------------
"""

import os
import sys
import logging
import json
from typing import Dict, List, Tuple
# 在导入matplotlib之前设置后端为Agg
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前设置
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import concurrent.futures
import argparse
import random
import threading

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

def test_dataset_creation(
    balance_node_types: bool = True,
    adaptive_subgraph_size: bool = True,
    negative_sample_ratio: float = 0.5,
    data_augmentation: bool = True,
    sample_size: int = None,
    num_workers: int = 4
):
    """测试数据集创建功能"""
    start_time = time.time()
    logger.info("测试数据集创建...")
    logger.info(f"参数: balance_node_types={balance_node_types}, adaptive_subgraph_size={adaptive_subgraph_size}, "
                f"negative_sample_ratio={negative_sample_ratio}, data_augmentation={data_augmentation}, "
                f"sample_size={sample_size}, num_workers={num_workers}")
    
    # 创建结果目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = f"test_results/dataset_test_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
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
        
        # 获取所有节点ID和类型（优化查询，一次性获取节点ID和类型）
        query_time_start = time.time()
        results = graph_manager.execute_cypher_query(
            "MATCH (n) RETURN n.id as id, labels(n)[0] as type"
        )
        
        # 按节点类型分组
        nodes_by_type = {}
        all_node_ids = []
        
        for record in results:
            if record.get('id') and record.get('type'):
                node_id = record['id']
                node_type = record['type']
                
                if node_type not in nodes_by_type:
                    nodes_by_type[node_type] = []
                
                nodes_by_type[node_type].append(node_id)
                all_node_ids.append(node_id)
        
        query_time = time.time() - query_time_start
        logger.info(f"获取节点信息耗时: {query_time:.2f}秒")
        logger.info(f"获取到 {len(all_node_ids)} 个节点，分为 {len(nodes_by_type)} 种类型")
        
        for node_type, ids in nodes_by_type.items():
            logger.info(f"节点类型 {node_type}: {len(ids)} 个")
        
        # 如果需要采样，先对节点ID进行采样，确保各类型节点均衡
        if sample_size:
            # 修改配置以加快处理速度
            dataset_config['max_node_size'] = min(dataset_config.get('max_node_size', 50), 30)
            dataset_config['max_edge_size'] = min(dataset_config.get('max_edge_size', 100), 50)
            
            # 计算每种类型需要的样本数
            total_samples_needed = sample_size * 3  # 训练、验证、测试集总共需要的样本数
            samples_per_type = max(1, total_samples_needed // len(nodes_by_type))
            
            # 均衡采样各类型节点
            sampled_node_ids = []
            for node_type, ids in nodes_by_type.items():
                if len(ids) > samples_per_type:
                    sampled_node_ids.extend(random.sample(ids, samples_per_type))
                else:
                    sampled_node_ids.extend(ids)
            
            # 如果采样后的节点数仍然超过需要的数量，再次随机采样
            if len(sampled_node_ids) > total_samples_needed:
                sampled_node_ids = random.sample(sampled_node_ids, total_samples_needed)
            
            all_node_ids = sampled_node_ids
            logger.info(f"采样后节点数量: {len(all_node_ids)}")
        
        # 划分数据集，确保各类型节点在训练、验证和测试集中的分布均衡
        train_node_ids = []
        val_node_ids = []
        test_node_ids = []
        
        # 按类型划分
        for node_type, ids in nodes_by_type.items():
            # 过滤出当前类型在采样后的节点ID
            type_ids = [node_id for node_id in all_node_ids if node_id in ids]
            
            if not type_ids:
                continue
                
            # 计算每个集合应分配的数量
            train_size = int(len(type_ids) * 0.6)
            val_size = int(len(type_ids) * 0.2)
            
            # 随机打乱
            random.shuffle(type_ids)
            
            # 分配到各集合
            train_node_ids.extend(type_ids[:train_size])
            val_node_ids.extend(type_ids[train_size:train_size+val_size])
            test_node_ids.extend(type_ids[train_size+val_size:])
        
        logger.info(f"数据集划分: 训练集 {len(train_node_ids)} 节点, 验证集 {len(val_node_ids)} 节点, 测试集 {len(test_node_ids)} 节点")
        
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
            # 并行创建数据集
            logger.info(f"使用 {num_workers} 个工作线程并行创建数据集...")
            
            # 创建数据集的函数
            def create_dataset(split_name, node_ids, balance, negative_ratio, augmentation):
                logger.info(f"开始创建{split_name}数据集...")
                start_time = time.time()
                dataset = create_dataset_with_retry(
                    graph_manager=None,  # 不传递共享的图管理器，让每个线程使用自己的
                    feature_extractor=feature_extractor,
                    node_ids=node_ids,
                    dataset_config=dataset_config,
                    balance=balance,
                    adaptive=adaptive_subgraph_size,
                    negative_ratio=negative_ratio,
                    augmentation=augmentation,
                    split=split_name
                )
                elapsed = time.time() - start_time
                logger.info(f"{split_name}数据集创建完成，包含 {len(dataset.pairs)} 个样本，耗时: {elapsed:.2f}秒")
                return dataset
            
            # 使用线程池并行创建数据集
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # 提交创建数据集的任务
                train_future = executor.submit(
                    create_dataset, 
                    "train", 
                    train_node_ids, 
                    balance_node_types, 
                    negative_sample_ratio, 
                    data_augmentation
                )
                
                val_future = executor.submit(
                    create_dataset, 
                    "val", 
                    val_node_ids, 
                    False, 
                    negative_sample_ratio * 0.5, 
                    False
                )
                
                test_future = executor.submit(
                    create_dataset, 
                    "test", 
                    test_node_ids, 
                    False, 
                    negative_sample_ratio * 0.5, 
                    False
                )
                
                # 获取结果
                train_dataset = train_future.result()
                val_dataset = val_future.result()
                test_dataset = test_future.result()
            
            logger.info(f"所有数据集创建完成: 训练集 {len(train_dataset.pairs)} 样本, 验证集 {len(val_dataset.pairs)} 样本, 测试集 {len(test_dataset.pairs)} 样本")
            
            # 如果指定了样本大小，则对数据集进行采样
            if sample_size and len(train_dataset.pairs) > sample_size:
                logger.info("对数据集进行采样...")
                sample_start = time.time()
                
                # 并行采样
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    train_sample_future = executor.submit(
                        parallel_sample_dataset, 
                        train_dataset.pairs, 
                        sample_size, 
                        num_workers
                    )
                    
                    val_sample_future = executor.submit(
                        parallel_sample_dataset, 
                        val_dataset.pairs, 
                        sample_size // 5, 
                        num_workers
                    )
                    
                    test_sample_future = executor.submit(
                        parallel_sample_dataset, 
                        test_dataset.pairs, 
                        sample_size // 5, 
                        num_workers
                    )
                    
                    # 获取结果
                    train_dataset.pairs = train_sample_future.result()
                    val_dataset.pairs = val_sample_future.result()
                    test_dataset.pairs = test_sample_future.result()
                
                sample_time = time.time() - sample_start
                logger.info(f"采样完成，耗时: {sample_time:.2f}秒")
                logger.info(f"采样后: 训练集 {len(train_dataset.pairs)} 样本, 验证集 {len(val_dataset.pairs)} 样本, 测试集 {len(test_dataset.pairs)} 样本")
            
            # 使用多线程分析数据集
            logger.info(f"使用 {num_workers} 个工作线程进行数据集分析...")
            
            # 直接分析原始数据，避免使用整个数据集对象
            train_pairs = train_dataset.pairs
            val_pairs = val_dataset.pairs
            test_pairs = test_dataset.pairs
            
            # 并行分析数据集
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                train_analysis_future = executor.submit(
                    analyze_dataset_statistics_parallel, 
                    train_pairs, 
                    f"{result_dir}/train", 
                    num_workers
                )
                
                val_analysis_future = executor.submit(
                    analyze_dataset_statistics_parallel, 
                    val_pairs, 
                    f"{result_dir}/val", 
                    num_workers
                )
                
                test_analysis_future = executor.submit(
                    analyze_dataset_statistics_parallel, 
                    test_pairs, 
                    f"{result_dir}/test", 
                    num_workers
                )
                
                # 等待分析完成
                for future in concurrent.futures.as_completed([train_analysis_future, val_analysis_future, test_analysis_future]):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"数据集分析出错: {e}")
            
            # 分析负样本和子图大小
            if negative_sample_ratio > 0:
                logger.info("开始分析负样本...")
                analyze_negative_samples_parallel(train_pairs, f"{result_dir}/negative_samples", num_workers)
            
            if adaptive_subgraph_size:
                logger.info("开始分析自适应子图大小...")
                analyze_adaptive_subgraph_size_parallel(train_pairs, f"{result_dir}/adaptive_subgraph", num_workers)
            
            # 并行保存样本示例
            logger.info("保存样本示例...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                train_future = executor.submit(save_sample_examples, train_pairs, f"{result_dir}/train_samples.json")
                val_future = executor.submit(save_sample_examples, val_pairs, f"{result_dir}/val_samples.json")
                test_future = executor.submit(save_sample_examples, test_pairs, f"{result_dir}/test_samples.json")
                
                # 等待所有保存任务完成
                for future in concurrent.futures.as_completed([train_future, val_future, test_future]):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"保存样本示例出错: {e}")
            
            total_time = time.time() - start_time
            logger.info(f"测试完成，总耗时: {total_time:.2f}秒，结果保存在 {result_dir} 目录")
            
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
        logger.error(f"测试过程出错: {e}")
        raise

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

def create_dataset_with_retry(graph_manager, feature_extractor, node_ids, dataset_config, balance, adaptive, negative_ratio, augmentation, split, max_retries=3):
    """创建数据集并在失败时重试"""
    # 使用线程本地的图管理器，避免多线程共享连接
    local_graph_manager = get_thread_graph_manager()
    
    for attempt in range(max_retries):
        try:
            # 设置随机种子，确保每次尝试都使用不同的种子
            random_seed = 42 + attempt
            
            # 创建数据集，使用线程本地的图管理器
            dataset = GraphTextDataset(
                graph_manager=local_graph_manager,
                feature_extractor=feature_extractor,
                max_node_size=dataset_config.get('max_node_size', 50),
                max_edge_size=dataset_config.get('max_edge_size', 100),
                include_dynamic=dataset_config.get('include_dynamic', True),
                data_augmentation=augmentation,
                balance_node_types=balance,
                adaptive_subgraph_size=adaptive,
                negative_sample_ratio=negative_ratio,
                split=split,
                seed=random_seed
            )
            return dataset
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"创建{split}数据集失败 (尝试 {attempt+1}/{max_retries}): {e}")
                time.sleep(1)  # 短暂等待后重试
            else:
                logger.error(f"创建{split}数据集失败，已达到最大重试次数: {e}")
                raise

def parallel_sample_dataset(pairs: List[Dict], sample_size: int, num_workers: int) -> List[Dict]:
    """并行采样数据集"""
    if len(pairs) <= sample_size:
        return pairs
    
    # 按节点类型分组
    pairs_by_type = {}
    for pair in pairs:
        node_type = pair["node_type"]
        if node_type not in pairs_by_type:
            pairs_by_type[node_type] = []
        pairs_by_type[node_type].append(pair)
    
    # 并行处理每种类型的采样
    sampled_pairs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for node_type, type_pairs in pairs_by_type.items():
            # 计算该类型需要采样的数量
            type_sample_size = max(1, int(len(type_pairs) / len(pairs) * sample_size))
            
            # 提交采样任务
            future = executor.submit(
                random.sample,
                type_pairs,
                min(type_sample_size, len(type_pairs))
            )
            futures.append(future)
        
        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            sampled_pairs.extend(future.result())
    
    # 如果总数超过目标大小，再次采样
    if len(sampled_pairs) > sample_size:
        sampled_pairs = random.sample(sampled_pairs, sample_size)
    
    return sampled_pairs

def process_sample_batch(batch_data: Tuple[List[Dict], int, int]) -> Dict:
    """处理样本批次，用于多线程分析"""
    samples, start_idx, end_idx = batch_data
    
    result = {
        "node_type_counts": {},
        "text_lengths": [],
        "subgraph_node_counts": [],
        "subgraph_edge_counts": [],
        "positive_count": 0,
        "negative_count": 0,
        "negative_types": {}
    }
    
    for i in range(start_idx, min(end_idx, len(samples))):
        sample = samples[i]
        
        # 统计节点类型
        node_type = sample["node_type"]
        if node_type not in result["node_type_counts"]:
            result["node_type_counts"][node_type] = 0
        result["node_type_counts"][node_type] += 1
        
        # 统计文本长度
        result["text_lengths"].append(len(sample["text"]))
        
        # 统计子图大小
        result["subgraph_node_counts"].append(len(sample["subgraph"]["nodes"]))
        result["subgraph_edge_counts"].append(len(sample["subgraph"]["edges"]))
        
        # 统计正负样本
        if sample.get("is_negative", False):
            result["negative_count"] += 1
            negative_type = sample.get("negative_type", "unknown")
            if negative_type not in result["negative_types"]:
                result["negative_types"][negative_type] = 0
            result["negative_types"][negative_type] += 1
        else:
            result["positive_count"] += 1
    
    return result

def merge_batch_results(results: List[Dict]) -> Dict:
    """合并多个批次的分析结果"""
    merged = {
        "node_type_counts": {},
        "text_lengths": [],
        "subgraph_node_counts": [],
        "subgraph_edge_counts": [],
        "positive_count": 0,
        "negative_count": 0,
        "negative_types": {}
    }
    
    for result in results:
        # 合并节点类型计数
        for node_type, count in result["node_type_counts"].items():
            if node_type not in merged["node_type_counts"]:
                merged["node_type_counts"][node_type] = 0
            merged["node_type_counts"][node_type] += count
        
        # 合并列表
        merged["text_lengths"].extend(result["text_lengths"])
        merged["subgraph_node_counts"].extend(result["subgraph_node_counts"])
        merged["subgraph_edge_counts"].extend(result["subgraph_edge_counts"])
        
        # 合并计数
        merged["positive_count"] += result["positive_count"]
        merged["negative_count"] += result["negative_count"]
        
        # 合并负样本类型
        for neg_type, count in result["negative_types"].items():
            if neg_type not in merged["negative_types"]:
                merged["negative_types"][neg_type] = 0
            merged["negative_types"][neg_type] += count
    
    return merged

def analyze_dataset_statistics_parallel(pairs: List[Dict], output_prefix: str, num_workers: int = 4):
    """并行分析数据集统计信息"""
    analysis_start = time.time()
    logger.info(f"分析数据集统计信息，使用 {num_workers} 个工作线程...")
    
    # 创建结果目录
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # 划分批次
    batch_size = max(1, len(pairs) // num_workers)
    batches = [(pairs, i, min(i + batch_size, len(pairs))) for i in range(0, len(pairs), batch_size)]
    
    # 并行处理批次
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_sample_batch, batch) for batch in batches]
        
        # 收集结果
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="分析数据集"):
            results.append(future.result())
    
    # 合并结果
    merged = merge_batch_results(results)
    
    # 保存统计信息
    statistics = {
        "node_type_counts": merged["node_type_counts"],
        "text_length_stats": {
            "min": min(merged["text_lengths"]),
            "max": max(merged["text_lengths"]),
            "mean": np.mean(merged["text_lengths"]),
            "median": np.median(merged["text_lengths"])
        },
        "subgraph_node_stats": {
            "min": min(merged["subgraph_node_counts"]),
            "max": max(merged["subgraph_node_counts"]),
            "mean": np.mean(merged["subgraph_node_counts"]),
            "median": np.median(merged["subgraph_node_counts"])
        },
        "subgraph_edge_stats": {
            "min": min(merged["subgraph_edge_counts"]),
            "max": max(merged["subgraph_edge_counts"]),
            "mean": np.mean(merged["subgraph_edge_counts"]),
            "median": np.median(merged["subgraph_edge_counts"])
        },
        "sample_counts": {
            "total": merged["positive_count"] + merged["negative_count"],
            "positive": merged["positive_count"],
            "negative": merged["negative_count"],
            "negative_types": merged["negative_types"]
        }
    }
    
    with open(f"{output_prefix}_statistics.json", "w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    
    # 单线程顺序生成所有图表，避免多线程问题
    logger.info("生成数据集统计图表...")
    
    # 生成节点类型分布图
    generate_node_type_distribution(merged["node_type_counts"], f"{output_prefix}_node_type_distribution.png")
    
    # 生成文本长度分布图
    generate_text_length_distribution(merged["text_lengths"], f"{output_prefix}_text_length_distribution.png")
    
    # 生成子图大小分布图
    generate_subgraph_size_distribution(
        merged["subgraph_node_counts"], 
        merged["subgraph_edge_counts"], 
        f"{output_prefix}_subgraph_size_distribution.png"
    )
    
    # 生成正负样本比例图（如果有负样本）
    if merged["negative_count"] > 0:
        generate_positive_negative_ratio(
            merged["positive_count"], 
            merged["negative_count"], 
            f"{output_prefix}_positive_negative_ratio.png"
        )
        
        # 生成负样本类型分布图（如果有多种类型）
        if len(merged["negative_types"]) > 1:
            generate_negative_type_distribution(
                merged["negative_types"], 
                f"{output_prefix}_negative_type_distribution.png"
            )
    
    analysis_time = time.time() - analysis_start
    logger.info(f"数据集分析完成，耗时: {analysis_time:.2f}秒")

# 新的图表生成函数，避免多线程问题
def generate_node_type_distribution(node_type_counts, output_file):
    """生成节点类型分布图"""
    try:
        # 创建新的图表
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(list(node_type_counts.keys()), list(node_type_counts.values()))
        ax.set_title("Node Type Distribution")
        ax.set_xlabel("Node Type")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        logger.info(f"已保存节点类型分布图: {output_file}")
    except Exception as e:
        logger.error(f"生成节点类型分布图时出错: {e}")
        plt.close('all')

def generate_text_length_distribution(text_lengths, output_file):
    """生成文本长度分布图"""
    try:
        # 创建新的图表
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(text_lengths, bins=20)
        ax.set_title("Text Length Distribution")
        ax.set_xlabel("Text Length")
        ax.set_ylabel("Sample Count")
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        logger.info(f"已保存文本长度分布图: {output_file}")
    except Exception as e:
        logger.error(f"生成文本长度分布图时出错: {e}")
        plt.close('all')

def generate_subgraph_size_distribution(node_counts, edge_counts, output_file):
    """生成子图大小分布图"""
    try:
        # 创建新的图表
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(node_counts, bins=20, alpha=0.7, label="节点数")
        ax.hist(edge_counts, bins=20, alpha=0.7, label="边数")
        ax.set_title("Subgraph Size Distribution")
        ax.set_xlabel("Count")
        ax.set_ylabel("Sample Count")
        ax.legend()
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        logger.info(f"已保存子图大小分布图: {output_file}")
    except Exception as e:
        logger.error(f"生成子图大小分布图时出错: {e}")
        plt.close('all')

def generate_positive_negative_ratio(positive_count, negative_count, output_file):
    """生成正负样本比例图"""
    try:
        # 创建新的图表
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie([positive_count, negative_count], 
               labels=["Positive", "Negative"], 
               autopct='%1.1f%%',
               colors=['#66b3ff', '#ff9999'])
        ax.set_title("Positive Negative Ratio")
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        logger.info(f"已保存正负样本比例图: {output_file}")
    except Exception as e:
        logger.error(f"生成正负样本比例图时出错: {e}")
        plt.close('all')

def generate_negative_type_distribution(negative_types, output_file):
    """生成负样本类型分布图"""
    try:
        # 创建新的图表
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(list(negative_types.keys()), list(negative_types.values()))
        ax.set_title("Negative Type Distribution")
        ax.set_xlabel("Negative Type")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        logger.info(f"已保存负样本类型分布图: {output_file}")
    except Exception as e:
        logger.error(f"生成负样本类型分布图时出错: {e}")
        plt.close('all')

def analyze_negative_samples_parallel(pairs: List[Dict], output_prefix: str, num_workers: int = 4):
    """并行分析负样本"""
    analysis_start = time.time()
    logger.info("分析负样本...")
    
    # 创建结果目录
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # 并行收集负样本
    negative_samples = []
    
    def collect_negatives(batch_pairs):
        return [p for p in batch_pairs if p.get("is_negative", False)]
    
    # 划分批次
    batch_size = max(1, len(pairs) // num_workers)
    batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(collect_negatives, batch) for batch in batches]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="收集负样本"):
            negative_samples.extend(future.result())
    
    if not negative_samples:
        logger.info("没有找到负样本")
        return
    
    # 按负样本类型分组
    samples_by_type = {}
    for sample in negative_samples:
        negative_type = sample.get("negative_type", "unknown")
        if negative_type not in samples_by_type:
            samples_by_type[negative_type] = []
        samples_by_type[negative_type].append(sample)
    
    # 保存每种类型的示例
    save_futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for negative_type, samples in samples_by_type.items():
            # 选择最多5个示例
            examples = samples[:5]
            
            # 保存示例
            save_futures.append(executor.submit(
                save_json_file,
                examples,
                f"{output_prefix}_{negative_type}_examples.json",
                f"保存了 {len(examples)} 个 {negative_type} 类型的负样本示例"
            ))
    
    # 等待所有保存任务完成
    for future in concurrent.futures.as_completed(save_futures):
        try:
            future.result()
        except Exception as e:
            logger.error(f"保存负样本示例时出错: {e}")
    
    # 分析困难负样本的相似度分布（如果有）
    if "hard_negative" in samples_by_type:
        hard_negatives = samples_by_type["hard_negative"]
        similarities = [sample.get("similarity", 0) for sample in hard_negatives]
        
        # 单线程生成图表
        try:
            # 创建新的图表
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(similarities, bins=20)
            ax.set_title("Hard Negative Similarity Distribution")
            ax.set_xlabel("Similarity")
            ax.set_ylabel("Sample Count")
            plt.tight_layout()
            
            # 保存图表
            output_file = f"{output_prefix}_hard_negative_similarity.png"
            plt.savefig(output_file, dpi=300)
            plt.close(fig)
            logger.info(f"已保存困难负样本相似度分布图: {output_file}")
        except Exception as e:
            logger.error(f"生成困难负样本相似度分布图时出错: {e}")
            plt.close('all')
        
        logger.info(f"困难负样本相似度统计: 最小={min(similarities):.4f}, 最大={max(similarities):.4f}, "
                   f"平均={np.mean(similarities):.4f}, 中位数={np.median(similarities):.4f}")
    
    analysis_time = time.time() - analysis_start
    logger.info(f"负样本分析完成，耗时: {analysis_time:.2f}秒")

def analyze_adaptive_subgraph_size_parallel(pairs: List[Dict], output_prefix: str, num_workers: int = 4):
    """并行分析自适应子图大小"""
    analysis_start = time.time()
    logger.info("分析自适应子图大小...")
    
    # 创建结果目录
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # 按节点类型统计子图大小
    subgraph_sizes_by_type = {}
    
    def process_subgraph_batch(batch_pairs):
        result = {}
        for sample in batch_pairs:
            node_type = sample["node_type"]
            if node_type not in result:
                result[node_type] = {
                    "node_counts": [],
                    "edge_counts": []
                }
            result[node_type]["node_counts"].append(len(sample["subgraph"]["nodes"]))
            result[node_type]["edge_counts"].append(len(sample["subgraph"]["edges"]))
        return result
    
    # 划分批次
    batch_size = max(1, len(pairs) // num_workers)
    batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
    
    # 并行处理批次
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_subgraph_batch, batch) for batch in batches]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="分析子图大小"):
            batch_result = future.result()
            # 合并结果
            for node_type, sizes in batch_result.items():
                if node_type not in subgraph_sizes_by_type:
                    subgraph_sizes_by_type[node_type] = {
                        "node_counts": [],
                        "edge_counts": []
                    }
                subgraph_sizes_by_type[node_type]["node_counts"].extend(sizes["node_counts"])
                subgraph_sizes_by_type[node_type]["edge_counts"].extend(sizes["edge_counts"])
    
    # 保存统计信息
    statistics = {}
    for node_type, sizes in subgraph_sizes_by_type.items():
        node_counts = sizes["node_counts"]
        edge_counts = sizes["edge_counts"]
        
        statistics[node_type] = {
            "node_count_stats": {
                "min": min(node_counts),
                "max": max(node_counts),
                "mean": np.mean(node_counts),
                "median": np.median(node_counts)
            },
            "edge_count_stats": {
                "min": min(edge_counts),
                "max": max(edge_counts),
                "mean": np.mean(edge_counts),
                "median": np.median(edge_counts)
            }
        }
    
    with open(f"{output_prefix}_statistics.json", "w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    
    # 单线程生成图表
    try:
        # 创建新的图表
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        
        # 节点数分布
        node_types = list(subgraph_sizes_by_type.keys())
        node_data = [subgraph_sizes_by_type[nt]["node_counts"] for nt in node_types]
        
        axs[0].boxplot(node_data, labels=node_types)
        axs[0].set_title("Subgraph Node Count Distribution by Node Type")
        axs[0].set_ylabel("Node Count")
        axs[0].grid(True, linestyle='--', alpha=0.7)
        
        # 边数分布
        edge_data = [subgraph_sizes_by_type[nt]["edge_counts"] for nt in node_types]
        
        axs[1].boxplot(edge_data, labels=node_types)
        axs[1].set_title("Subgraph Edge Count Distribution by Node Type")
        axs[1].set_ylabel("Edge Count")
        axs[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = f"{output_prefix}_size_by_type.png"
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        logger.info(f"已保存不同节点类型的子图大小分布图: {output_file}")
    except Exception as e:
        logger.error(f"生成不同节点类型的子图大小分布图时出错: {e}")
        plt.close('all')
    
    analysis_time = time.time() - analysis_start
    logger.info(f"自适应子图大小分析完成，耗时: {analysis_time:.2f}秒")

def save_json_file(data, output_file, log_message=None):
    """保存JSON文件"""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if log_message:
            logger.info(log_message)
        return True
    except Exception as e:
        logger.error(f"保存JSON文件 {output_file} 时出错: {e}")
        return False

def save_sample_examples(pairs: List[Dict], output_file: str):
    """保存样本示例"""
    # 按节点类型分组
    samples_by_type = {}
    for i in range(min(len(pairs), 100)):  # 只处理前100个样本
        sample = pairs[i]
        node_type = sample["node_type"]
        
        if node_type not in samples_by_type:
            samples_by_type[node_type] = []
        
        if len(samples_by_type[node_type]) < 2:  # 每种类型最多保存2个示例
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
    return save_json_file(examples, output_file, f"保存了 {len(examples)} 个样本示例到 {output_file}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试图文数据集")
    parser.add_argument("--balance", action="store_true", help="是否平衡节点类型")
    parser.add_argument("--adaptive", action="store_true", help="是否使用自适应子图大小")
    parser.add_argument("--negative_ratio", type=float, default=0.5, help="负样本比例")
    parser.add_argument("--augmentation", action="store_true", help="是否使用数据增强")
    parser.add_argument("--sample_size", type=int, default=None, help="采样大小，不指定则使用全部数据")
    parser.add_argument("--workers", type=int, default=4, help="工作线程数")
    args = parser.parse_args()
    
    # 测试数据集
    test_dataset_creation(
        balance_node_types=args.balance,
        adaptive_subgraph_size=args.adaptive,
        negative_sample_ratio=args.negative_ratio,
        data_augmentation=args.augmentation,
        sample_size=args.sample_size,
        num_workers=args.workers
    ) 