#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-24 10:30
# @Desc   : 文本编码器测试脚本
# --------------------------------------------------------
"""

import os
import sys
import torch
import unittest
import logging
import json
import datetime
import traceback
from typing import Dict, List, Optional, Tuple, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.models.text_encoder import TextEncoder

# 创建测试结果目录
TEST_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_results")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# 配置日志
LOG_FILE = os.path.join(TEST_RESULTS_DIR, f"text_encoder_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    print(f"日志文件将保存到: {LOG_FILE}")
except Exception as e:
    print(f"无法创建日志文件: {str(e)}")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)


class TestTextEncoder(unittest.TestCase):
    """测试文本编码器"""
    
    def setUp(self):
        """设置测试环境"""
        # 设置随机种子以确保可重复性
        torch.manual_seed(42)
        
        # 设置测试参数
        self.model_name = "bert-base-chinese"
        self.output_dim = 768
        self.dropout = 0.1
        self.max_length = 128
        
        # 测试文本
        self.test_texts = [
            "查询DC001下所有虚拟机的状态",
            "TENANT002租户下有多少网元设备",
            "找出所有状态异常的主机",
            "VM005的CPU使用率是多少",
            "哪些虚拟机的内存使用率超过80%"
        ]
        
        # 测试结果
        self.test_results = {
            "basic": {},
            "pooling_strategies": {},
            "projection": {},
            "layer_weights": {},
            "encode_text": {}
        }
        
    def tearDown(self):
        """测试结束后的清理工作"""
        # 保存测试结果
        result_file = os.path.join(TEST_RESULTS_DIR, f"text_encoder_test_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        logger.info(f"测试结果已保存到: {result_file}")
        
    def test_basic_functionality(self):
        """测试基本功能"""
        logger.info("测试基本功能...")
        
        try:
            # 创建编码器
            encoder = TextEncoder(
                model_name=self.model_name,
                output_dim=self.output_dim,
                dropout=self.dropout
            )
            
            # 创建输入
            tokenizer = encoder.tokenizer
            inputs = tokenizer(
                self.test_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 前向传播
            outputs = encoder(**inputs)
            
            # 检查输出
            self.assertIn('embeddings', outputs)
            self.assertIn('pooled', outputs)
            self.assertIn('hidden_states', outputs)
            
            # 检查输出形状
            batch_size = len(self.test_texts)
            seq_length = inputs['input_ids'].shape[1]
            
            self.assertEqual(outputs['embeddings'].shape, (batch_size, seq_length, self.output_dim))
            self.assertEqual(outputs['pooled'].shape, (batch_size, self.output_dim))
            
            # 记录结果
            self.test_results["basic"] = {
                "status": "通过",
                "embeddings_shape": list(outputs['embeddings'].shape),
                "pooled_shape": list(outputs['pooled'].shape),
                "hidden_states_count": len(outputs['hidden_states'])
            }
            
            logger.info("基本功能测试通过")
            
        except Exception as e:
            logger.error(f"基本功能测试失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.test_results["basic"] = {
                "status": "失败",
                "error": str(e)
            }
            raise
            
    def test_pooling_strategies(self):
        """测试不同的池化策略"""
        logger.info("测试不同的池化策略...")
        
        pooling_strategies = ["cls", "mean", "max", "attention", "weighted"]
        
        for strategy in pooling_strategies:
            try:
                logger.info(f"测试 {strategy} 池化策略...")
                
                # 创建编码器
                encoder = TextEncoder(
                    model_name=self.model_name,
                    output_dim=self.output_dim,
                    dropout=self.dropout,
                    pooling_strategy=strategy,
                    use_layer_weights=(strategy == "weighted")
                )
                
                # 创建输入
                tokenizer = encoder.tokenizer
                inputs = tokenizer(
                    self.test_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # 前向传播
                outputs = encoder(**inputs)
                
                # 检查输出
                self.assertIn('pooled', outputs)
                
                # 检查输出形状
                batch_size = len(self.test_texts)
                self.assertEqual(outputs['pooled'].shape, (batch_size, self.output_dim))
                
                # 记录结果
                self.test_results["pooling_strategies"][strategy] = {
                    "status": "通过",
                    "pooled_shape": list(outputs['pooled'].shape),
                    "pooled_norm": float(torch.norm(outputs['pooled']).item())
                }
                
                logger.info(f"{strategy} 池化策略测试通过")
                
            except Exception as e:
                logger.error(f"{strategy} 池化策略测试失败: {str(e)}")
                logger.error(traceback.format_exc())
                self.test_results["pooling_strategies"][strategy] = {
                    "status": "失败",
                    "error": str(e)
                }
                
    def test_projection(self):
        """测试投影层"""
        logger.info("测试投影层...")
        
        output_dims = [128, 256, 512, 768, 1024]
        
        for dim in output_dims:
            try:
                logger.info(f"测试输出维度 {dim}...")
                
                # 创建编码器
                encoder = TextEncoder(
                    model_name=self.model_name,
                    output_dim=dim,
                    dropout=self.dropout
                )
                
                # 创建输入
                tokenizer = encoder.tokenizer
                inputs = tokenizer(
                    self.test_texts[0],  # 只使用一个文本以加快测试速度
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # 前向传播
                outputs = encoder(**inputs)
                
                # 检查输出维度
                self.assertEqual(outputs['pooled'].shape[-1], dim)
                self.assertEqual(outputs['embeddings'].shape[-1], dim)
                
                # 检查get_output_dim方法
                self.assertEqual(encoder.get_output_dim(), dim)
                
                # 记录结果
                self.test_results["projection"][str(dim)] = {
                    "status": "通过",
                    "pooled_shape": list(outputs['pooled'].shape),
                    "embeddings_shape": list(outputs['embeddings'].shape),
                    "get_output_dim": encoder.get_output_dim()
                }
                
                logger.info(f"输出维度 {dim} 测试通过")
                
            except Exception as e:
                logger.error(f"输出维度 {dim} 测试失败: {str(e)}")
                logger.error(traceback.format_exc())
                self.test_results["projection"][str(dim)] = {
                    "status": "失败",
                    "error": str(e)
                }
                
    def test_layer_weights(self):
        """测试层权重"""
        logger.info("测试层权重...")
        
        try:
            # 创建编码器
            encoder = TextEncoder(
                model_name=self.model_name,
                output_dim=self.output_dim,
                dropout=self.dropout,
                pooling_strategy="weighted",
                use_layer_weights=True
            )
            
            # 检查层权重
            self.assertTrue(hasattr(encoder, 'layer_weights'))
            self.assertTrue(isinstance(encoder.layer_weights, torch.nn.Parameter))
            
            # 创建输入
            tokenizer = encoder.tokenizer
            inputs = tokenizer(
                self.test_texts[0],  # 只使用一个文本以加快测试速度
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 前向传播
            outputs = encoder(**inputs)
            
            # 检查输出
            self.assertIn('pooled', outputs)
            
            # 记录结果
            self.test_results["layer_weights"] = {
                "status": "通过",
                "layer_weights_shape": list(encoder.layer_weights.shape),
                "layer_weights_sum": float(encoder.layer_weights.sum().item())
            }
            
            logger.info("层权重测试通过")
            
        except Exception as e:
            logger.error(f"层权重测试失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.test_results["layer_weights"] = {
                "status": "失败",
                "error": str(e)
            }
            
    def test_encode_text(self):
        """测试直接编码文本的功能"""
        logger.info("测试直接编码文本的功能...")
        
        try:
            # 创建编码器
            encoder = TextEncoder(
                model_name=self.model_name,
                output_dim=self.output_dim,
                dropout=self.dropout
            )
            
            # 单个文本
            single_text = self.test_texts[0]
            single_outputs = encoder.encode_text(single_text)
            
            # 多个文本
            batch_outputs = encoder.encode_text(self.test_texts)
            
            # 检查输出
            self.assertIn('embeddings', single_outputs)
            self.assertIn('pooled', single_outputs)
            self.assertIn('embeddings', batch_outputs)
            self.assertIn('pooled', batch_outputs)
            
            # 检查形状
            self.assertEqual(single_outputs['pooled'].shape[0], 1)
            self.assertEqual(batch_outputs['pooled'].shape[0], len(self.test_texts))
            
            # 记录结果
            self.test_results["encode_text"] = {
                "status": "通过",
                "single_text_pooled_shape": list(single_outputs['pooled'].shape),
                "batch_text_pooled_shape": list(batch_outputs['pooled'].shape)
            }
            
            logger.info("直接编码文本功能测试通过")
            
        except Exception as e:
            logger.error(f"直接编码文本功能测试失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.test_results["encode_text"] = {
                "status": "失败",
                "error": str(e)
            }


if __name__ == "__main__":
    unittest.main() 