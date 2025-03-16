#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-24 11:30
# @Desc   : 文本编码器测试脚本
# --------------------------------------------------------
"""

import os
import sys
import torch
import argparse
import logging
import json
import datetime
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.models.text_encoder import TextEncoder
from utils.logger import setup_logger

# 创建测试结果目录
TEST_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_results")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# 设置日志
logger = setup_logger("text_encoder_test")

def test_text_encoder(
    model_name: str = "bert-base-chinese",
    output_dim: int = 768,
    pooling_strategy: str = "cls",
    test_texts: List[str] = None
) -> Dict[str, Any]:
    """
    测试文本编码器
    
    Args:
        model_name: 预训练模型名称
        output_dim: 输出维度
        pooling_strategy: 池化策略
        test_texts: 测试文本列表
        
    Returns:
        测试结果字典
    """
    logger.info(f"测试文本编码器: model_name={model_name}, output_dim={output_dim}, pooling_strategy={pooling_strategy}")
    
    # 默认测试文本
    if test_texts is None:
        test_texts = [
            "查询DC001下所有虚拟机的状态",
            "TENANT002租户下有多少网元设备",
            "找出所有状态异常的主机",
            "VM005的CPU使用率是多少",
            "哪些虚拟机的内存使用率超过80%"
        ]
    
    # 创建编码器
    encoder = TextEncoder(
        model_name=model_name,
        output_dim=output_dim,
        pooling_strategy=pooling_strategy,
        use_layer_weights=(pooling_strategy == "weighted")
    )
    
    # 编码文本
    logger.info("编码文本...")
    outputs = encoder.encode_text(test_texts)
    
    # 提取结果
    embeddings = outputs['embeddings']
    pooled = outputs['pooled']
    
    # 计算相似度矩阵
    similarity_matrix = torch.nn.functional.cosine_similarity(
        pooled.unsqueeze(1), pooled.unsqueeze(0), dim=-1
    )
    
    # 打印结果
    logger.info(f"文本数量: {len(test_texts)}")
    logger.info(f"序列嵌入形状: {embeddings.shape}")
    logger.info(f"池化嵌入形状: {pooled.shape}")
    
    # 打印相似度矩阵
    logger.info("相似度矩阵:")
    for i in range(len(test_texts)):
        sim_str = " ".join([f"{similarity_matrix[i, j]:.4f}" for j in range(len(test_texts))])
        logger.info(f"  {sim_str}")
    
    # 返回结果
    return {
        "embeddings_shape": list(embeddings.shape),
        "pooled_shape": list(pooled.shape),
        "similarity_matrix": similarity_matrix.tolist(),
        "model_name": model_name,
        "output_dim": output_dim,
        "pooling_strategy": pooling_strategy
    }

def compare_pooling_strategies(
    model_name: str = "bert-base-chinese",
    output_dim: int = 768,
    test_texts: List[str] = None
) -> Dict[str, Any]:
    """
    比较不同的池化策略
    
    Args:
        model_name: 预训练模型名称
        output_dim: 输出维度
        test_texts: 测试文本列表
        
    Returns:
        比较结果字典
    """
    logger.info("比较不同的池化策略...")
    
    # 默认测试文本
    if test_texts is None:
        test_texts = [
            "查询DC001下所有虚拟机的状态",
            "TENANT002租户下有多少网元设备",
            "找出所有状态异常的主机",
            "VM005的CPU使用率是多少",
            "哪些虚拟机的内存使用率超过80%"
        ]
    
    # 池化策略
    strategies = ["cls", "mean", "max", "attention", "weighted"]
    
    # 测试每种策略
    results = {}
    for strategy in strategies:
        logger.info(f"测试 {strategy} 池化策略...")
        results[strategy] = test_text_encoder(
            model_name=model_name,
            output_dim=output_dim,
            pooling_strategy=strategy,
            test_texts=test_texts
        )
    
    # 比较相似度矩阵
    comparison_results = {
        "text_pairs": [],
        "strategies": strategies,
        "model_name": model_name,
        "output_dim": output_dim,
        "test_texts": test_texts
    }
    
    logger.info("相似度矩阵比较:")
    for i, text_i in enumerate(test_texts):
        for j, text_j in enumerate(test_texts):
            if i < j:  # 只比较上三角矩阵
                pair_result = {
                    "text1_index": i,
                    "text2_index": j,
                    "text1": text_i,
                    "text2": text_j,
                    "similarities": {}
                }
                
                logger.info(f"文本 {i+1} 和文本 {j+1} 的相似度:")
                for strategy in strategies:
                    sim = results[strategy]["similarity_matrix"][i][j]
                    pair_result["similarities"][strategy] = sim
                    logger.info(f"  {strategy}: {sim:.4f}")
                logger.info("")
                
                comparison_results["text_pairs"].append(pair_result)
    
    # 保存比较结果
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(TEST_RESULTS_DIR, f"pooling_strategies_comparison_{timestamp}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    logger.info(f"比较结果已保存到: {result_file}")
    
    return comparison_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文本编码器测试")
    parser.add_argument("--model_name", type=str, default="bert-base-chinese", help="预训练模型名称")
    parser.add_argument("--output_dim", type=int, default=768, help="输出维度")
    parser.add_argument("--pooling_strategy", type=str, default="cls", help="池化策略")
    parser.add_argument("--compare", action="store_true", help="比较不同的池化策略")
    args = parser.parse_args()
    
    # 测试文本
    test_texts = [
        "查询DC001下所有虚拟机的状态",
        "TENANT002租户下有多少网元设备",
        "找出所有状态异常的主机",
        "VM005的CPU使用率是多少",
        "哪些虚拟机的内存使用率超过80%"
    ]
    
    if args.compare:
        # 比较不同的池化策略
        comparison_results = compare_pooling_strategies(
            model_name=args.model_name,
            output_dim=args.output_dim,
            test_texts=test_texts
        )
        
        # 保存单个测试结果
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(TEST_RESULTS_DIR, f"text_encoder_test_{timestamp}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model_name": args.model_name,
                "output_dim": args.output_dim,
                "test_texts": test_texts,
                "comparison_results": comparison_results
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"测试结果已保存到: {result_file}")
    else:
        # 测试单个池化策略
        result = test_text_encoder(
            model_name=args.model_name,
            output_dim=args.output_dim,
            pooling_strategy=args.pooling_strategy,
            test_texts=test_texts
        )
        
        # 保存测试结果
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(TEST_RESULTS_DIR, f"text_encoder_test_{timestamp}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model_name": args.model_name,
                "output_dim": args.output_dim,
                "pooling_strategy": args.pooling_strategy,
                "test_texts": test_texts,
                "result": result
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"测试结果已保存到: {result_file}")

if __name__ == "__main__":
    main() 