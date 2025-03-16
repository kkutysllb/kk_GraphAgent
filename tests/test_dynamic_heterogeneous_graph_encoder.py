#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-22 11:30
# @Desc   : 动态异构图编码器测试脚本
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

from rag.models.dynamic_heterogeneous_graph_encoder import (
    NodeLevelAttention,
    EdgeLevelAttention,
    TimeSeriesEncoder,
    TemporalLevelAttention,
    HierarchicalAwarenessModule,
    DynamicHeterogeneousGraphEncoder
)

# 创建测试结果目录
TEST_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_results")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# 配置日志
LOG_FILE = os.path.join(TEST_RESULTS_DIR, f"dynamic_heterogeneous_graph_encoder_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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


class TestDynamicHeterogeneousGraphEncoder(unittest.TestCase):
    """测试动态异构图编码器"""
    
    def setUp(self):
        """设置测试环境"""
        # 设置随机种子以确保可重复性
        torch.manual_seed(42)
        
        # 设置测试参数
        self.batch_size = 8
        self.num_nodes = 20
        self.node_dim = 16
        self.edge_dim = 8
        self.time_series_dim = 12
        self.hidden_dim = 32
        self.output_dim = 64
        self.seq_len = 10
        self.edge_types = ["CONNECTS", "CONTAINS", "DEPENDS_ON"]
        self.num_levels = 3
        
        # 创建测试数据
        self.node_features = torch.randn(self.num_nodes, self.node_dim)
        
        # 创建边索引和边特征
        self.edge_indices_dict = {}
        self.edge_features_dict = {}
        
        for edge_type in self.edge_types:
            # 为每种边类型随机生成边
            num_edges = torch.randint(5, 15, (1,)).item()
            src_nodes = torch.randint(0, self.num_nodes, (num_edges,))
            dst_nodes = torch.randint(0, self.num_nodes, (num_edges,))
            
            # 确保源节点和目标节点不同
            for i in range(num_edges):
                while src_nodes[i] == dst_nodes[i]:
                    dst_nodes[i] = torch.randint(0, self.num_nodes, (1,))
            
            # 创建边索引
            edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
            self.edge_indices_dict[edge_type] = edge_index
            
            # 创建边特征
            edge_features = torch.randn(num_edges, self.edge_dim)
            self.edge_features_dict[edge_type] = edge_features
        
        # 创建时间序列特征
        self.time_series_features = torch.randn(self.num_nodes, self.seq_len, self.time_series_dim)
        
        # 创建节点层级
        self.node_levels = torch.randint(0, self.num_levels, (self.num_nodes,))
        
        # 初始化测试结果
        if not hasattr(TestDynamicHeterogeneousGraphEncoder, 'test_results'):
            TestDynamicHeterogeneousGraphEncoder.test_results = {
                "test_name": "动态异构图编码器测试",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tests": []
            }
    
    def tearDown(self):
        """测试完成后的清理工作"""
        pass
    
    def test_node_level_attention(self):
        """测试节点级注意力层"""
        test_result = {"name": "节点级注意力层测试", "status": "失败", "details": {}}
        
        try:
            logger.info("测试节点级注意力层...")
            
            # 创建节点级注意力层
            node_level_attention = NodeLevelAttention(
                input_dim=self.node_dim,
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                edge_types=self.edge_types,
                dropout=0.1,
                use_edge_features=True
            )
            
            # 打印注意力MLP的维度 - 使用第一个边类型
            edge_type = self.edge_types[0]
            attention_mlp_input_dim = node_level_attention.attention_mlp[edge_type][0].in_features
            logger.info(f"NodeLevelAttention with edge features - attention MLP input dim for {edge_type}: {attention_mlp_input_dim}, hidden dim: {node_level_attention.hidden_dim}")
            
            # 前向传播
            edge_type_embeddings = node_level_attention(
                self.node_features,
                self.edge_indices_dict,
                self.edge_features_dict
            )
            
            # 验证输出
            self.assertIsInstance(edge_type_embeddings, dict)
            self.assertEqual(len(edge_type_embeddings), len(self.edge_types))
            
            for edge_type in self.edge_types:
                self.assertIn(edge_type, edge_type_embeddings)
                self.assertEqual(edge_type_embeddings[edge_type].shape, (self.num_nodes, self.hidden_dim))
                logger.info(f"Edge type {edge_type} embeddings shape: {edge_type_embeddings[edge_type].shape}")
            
            # 测试不使用边特征的情况
            logger.info("测试节点级注意力层（不使用边特征）...")
            node_level_attention_no_edge = NodeLevelAttention(
                input_dim=self.node_dim,
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                edge_types=self.edge_types,
                dropout=0.1,
                use_edge_features=False
            )
            
            # 打印注意力MLP的维度 - 使用第一个边类型
            edge_type = self.edge_types[0]
            attention_mlp_no_edge_input_dim = node_level_attention_no_edge.attention_mlp[edge_type][0].in_features
            logger.info(f"NodeLevelAttention without edge features - attention MLP input dim for {edge_type}: {attention_mlp_no_edge_input_dim}, hidden dim: {node_level_attention_no_edge.hidden_dim}")
            
            # 前向传播
            edge_type_embeddings_no_edge = node_level_attention_no_edge(
                self.node_features,
                self.edge_indices_dict,
                None
            )
            
            # 验证输出
            self.assertIsInstance(edge_type_embeddings_no_edge, dict)
            self.assertEqual(len(edge_type_embeddings_no_edge), len(self.edge_types))
            
            for edge_type in self.edge_types:
                self.assertIn(edge_type, edge_type_embeddings_no_edge)
                self.assertEqual(edge_type_embeddings_no_edge[edge_type].shape, (self.num_nodes, self.hidden_dim))
                logger.info(f"Edge type {edge_type} embeddings shape (no edge features): {edge_type_embeddings_no_edge[edge_type].shape}")
            
            logger.info("节点级注意力层测试通过！")
            test_result["status"] = "通过"
            test_result["details"] = {
                "input_dim": self.node_dim,
                "edge_dim": self.edge_dim,
                "hidden_dim": self.hidden_dim,
                "output_shape": f"[{self.num_nodes}, {self.hidden_dim}] for each edge type",
                "attention_mlp_with_edge_features": f"Input dim: {attention_mlp_input_dim}",
                "attention_mlp_without_edge_features": f"Input dim: {attention_mlp_no_edge_input_dim}"
            }
        except Exception as e:
            logger.error(f"节点级注意力层测试失败: {str(e)}")
            test_result["details"]["error"] = str(e)
        
        TestDynamicHeterogeneousGraphEncoder.test_results["tests"].append(test_result)
    
    def test_edge_level_attention(self):
        """测试边级注意力层"""
        test_result = {"name": "边级注意力层测试", "status": "失败", "details": {}}
        
        try:
            logger.info("测试边级注意力层...")
            
            # 创建边级注意力层
            edge_level_attention = EdgeLevelAttention(
                hidden_dim=self.hidden_dim,
                edge_types=self.edge_types,
                dropout=0.1
            )
            
            # 创建边类型嵌入
            edge_type_embeddings = {}
            for edge_type in self.edge_types:
                edge_type_embeddings[edge_type] = torch.randn(self.num_nodes, self.hidden_dim)
            
            # 前向传播
            fused_embeddings = edge_level_attention(edge_type_embeddings)
            
            # 验证输出
            self.assertEqual(fused_embeddings.shape, (self.num_nodes, self.hidden_dim))
            
            logger.info("边级注意力层测试通过！")
            test_result["status"] = "通过"
            test_result["details"] = {
                "hidden_dim": self.hidden_dim,
                "output_shape": f"[{self.num_nodes}, {self.hidden_dim}]"
            }
        except Exception as e:
            logger.error(f"边级注意力层测试失败: {str(e)}")
            test_result["details"]["error"] = str(e)
        
        TestDynamicHeterogeneousGraphEncoder.test_results["tests"].append(test_result)
    
    def test_time_series_encoder(self):
        """测试时间序列编码器"""
        test_result = {"name": "时间序列编码器测试", "status": "失败", "details": {}}
        
        try:
            logger.info("测试时间序列编码器...")
            
            # 创建时间序列编码器
            time_series_encoder = TimeSeriesEncoder(
                input_dim=self.time_series_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                seq_len=self.seq_len,
                dropout=0.1
            )
            
            # 前向传播
            temporal_embeddings = time_series_encoder(self.time_series_features)
            
            # 验证输出
            self.assertEqual(temporal_embeddings.shape, (self.num_nodes, self.hidden_dim))
            
            logger.info("时间序列编码器测试通过！")
            test_result["status"] = "通过"
            test_result["details"] = {
                "input_dim": self.time_series_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.hidden_dim,
                "seq_len": self.seq_len,
                "output_shape": f"[{self.num_nodes}, {self.hidden_dim}]"
            }
        except Exception as e:
            logger.error(f"时间序列编码器测试失败: {str(e)}")
            test_result["details"]["error"] = str(e)
        
        TestDynamicHeterogeneousGraphEncoder.test_results["tests"].append(test_result)
    
    def test_temporal_level_attention(self):
        """测试时间级注意力层"""
        test_result = {"name": "时间级注意力层测试", "status": "失败", "details": {}}
        
        try:
            logger.info("测试时间级注意力层...")
            
            # 创建时间级注意力层
            temporal_level_attention = TemporalLevelAttention(
                static_dim=self.hidden_dim,
                temporal_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                dropout=0.1
            )
            
            # 创建静态特征和时间特征
            static_features = torch.randn(self.num_nodes, self.hidden_dim)
            temporal_features = torch.randn(self.num_nodes, self.hidden_dim)
            
            # 前向传播
            fused_features = temporal_level_attention(static_features, temporal_features)
            
            # 验证输出
            self.assertEqual(fused_features.shape, (self.num_nodes, self.hidden_dim))
            
            logger.info("时间级注意力层测试通过！")
            test_result["status"] = "通过"
            test_result["details"] = {
                "static_dim": self.hidden_dim,
                "temporal_dim": self.hidden_dim,
                "output_dim": self.hidden_dim,
                "output_shape": f"[{self.num_nodes}, {self.hidden_dim}]"
            }
        except Exception as e:
            logger.error(f"时间级注意力层测试失败: {str(e)}")
            test_result["details"]["error"] = str(e)
        
        TestDynamicHeterogeneousGraphEncoder.test_results["tests"].append(test_result)
    
    def test_hierarchical_awareness_module(self):
        """测试层级感知模块"""
        test_result = {"name": "层级感知模块测试", "status": "失败", "details": {}}
        
        try:
            logger.info("测试层级感知模块...")
            
            # 创建层级感知模块
            hierarchical_awareness = HierarchicalAwarenessModule(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                num_levels=self.num_levels,
                dropout=0.1
            )
            
            # 创建节点特征
            node_features = torch.randn(self.num_nodes, self.hidden_dim)
            
            # 前向传播
            hierarchical_features = hierarchical_awareness(node_features, self.node_levels)
            
            # 验证输出
            self.assertEqual(hierarchical_features.shape, (self.num_nodes, self.hidden_dim))
            
            logger.info("层级感知模块测试通过！")
            test_result["status"] = "通过"
            test_result["details"] = {
                "input_dim": self.hidden_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.hidden_dim,
                "num_levels": self.num_levels,
                "output_shape": f"[{self.num_nodes}, {self.hidden_dim}]"
            }
        except Exception as e:
            logger.error(f"层级感知模块测试失败: {str(e)}")
            test_result["details"]["error"] = str(e)
        
        TestDynamicHeterogeneousGraphEncoder.test_results["tests"].append(test_result)
    
    def test_dynamic_heterogeneous_graph_encoder(self):
        """测试动态异构图编码器"""
        test_result = {"name": "动态异构图编码器测试", "status": "失败", "details": {}, "subtests": []}
        
        try:
            logger.info("测试动态异构图编码器...")
            
            # 创建动态异构图编码器
            graph_encoder = DynamicHeterogeneousGraphEncoder(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                time_series_dim=self.time_series_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                edge_types=self.edge_types,
                num_levels=self.num_levels,
                seq_len=self.seq_len,
                use_edge_features=True,
                dropout=0.1
            )
            
            # 打印关键组件的配置
            logger.info(f"DynamicHeterogeneousGraphEncoder configuration:")
            logger.info(f"  - Node dim: {self.node_dim}")
            logger.info(f"  - Edge dim: {self.edge_dim}")
            logger.info(f"  - Hidden dim: {self.hidden_dim}")
            logger.info(f"  - Output dim: {self.output_dim}")
            logger.info(f"  - Edge types: {self.edge_types}")
            
            edge_type = self.edge_types[0]
            node_level_attention_mlp_input_dim = graph_encoder.node_level_attention.attention_mlp[edge_type][0].in_features
            logger.info(f"  - Node level attention MLP input dim for {edge_type}: {node_level_attention_mlp_input_dim}")
            
            # 测试1：使用所有特征
            subtest_result = {"name": "使用所有特征", "status": "失败", "details": {}}
            try:
                logger.info("测试1：使用所有特征")
                node_embeddings = graph_encoder(
                    self.node_features,
                    self.edge_indices_dict,
                    self.edge_features_dict,
                    self.time_series_features,
                    self.node_levels
                )
                
                # 验证输出
                self.assertEqual(node_embeddings.shape, (self.num_nodes, self.output_dim))
                logger.info(f"Test 1 - Output shape: {node_embeddings.shape}")
                subtest_result["status"] = "通过"
                subtest_result["details"]["output_shape"] = f"[{self.num_nodes}, {self.output_dim}]"
            except Exception as e:
                logger.error(f"测试1失败: {str(e)}")
                subtest_result["details"]["error"] = str(e)
            test_result["subtests"].append(subtest_result)
            
            # 测试2：不使用时间序列特征
            subtest_result = {"name": "不使用时间序列特征", "status": "失败", "details": {}}
            try:
                logger.info("测试2：不使用时间序列特征")
                node_embeddings = graph_encoder(
                    self.node_features,
                    self.edge_indices_dict,
                    self.edge_features_dict,
                    None,
                    self.node_levels
                )
                
                # 验证输出
                self.assertEqual(node_embeddings.shape, (self.num_nodes, self.output_dim))
                logger.info(f"Test 2 - Output shape: {node_embeddings.shape}")
                subtest_result["status"] = "通过"
                subtest_result["details"]["output_shape"] = f"[{self.num_nodes}, {self.output_dim}]"
            except Exception as e:
                logger.error(f"测试2失败: {str(e)}")
                subtest_result["details"]["error"] = str(e)
            test_result["subtests"].append(subtest_result)
            
            # 测试3：不使用层级信息
            subtest_result = {"name": "不使用层级信息", "status": "失败", "details": {}}
            try:
                logger.info("测试3：不使用层级信息")
                node_embeddings = graph_encoder(
                    self.node_features,
                    self.edge_indices_dict,
                    self.edge_features_dict,
                    self.time_series_features,
                    None
                )
                
                # 验证输出
                self.assertEqual(node_embeddings.shape, (self.num_nodes, self.output_dim))
                logger.info(f"Test 3 - Output shape: {node_embeddings.shape}")
                subtest_result["status"] = "通过"
                subtest_result["details"]["output_shape"] = f"[{self.num_nodes}, {self.output_dim}]"
            except Exception as e:
                logger.error(f"测试3失败: {str(e)}")
                subtest_result["details"]["error"] = str(e)
            test_result["subtests"].append(subtest_result)
            
            # 测试4：不使用边特征
            subtest_result = {"name": "不使用边特征", "status": "失败", "details": {}}
            try:
                logger.info("测试4：不使用边特征")
                # 创建一个新的编码器，明确设置不使用边特征
                graph_encoder_no_edge = DynamicHeterogeneousGraphEncoder(
                    node_dim=self.node_dim,
                    edge_dim=self.edge_dim,
                    time_series_dim=self.time_series_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.output_dim,
                    edge_types=self.edge_types,
                    num_levels=self.num_levels,
                    seq_len=self.seq_len,
                    use_edge_features=False,
                    dropout=0.1
                )
                
                edge_type = self.edge_types[0]
                node_level_attention_no_edge_mlp_input_dim = graph_encoder_no_edge.node_level_attention.attention_mlp[edge_type][0].in_features
                logger.info(f"DynamicHeterogeneousGraphEncoder without edge features:")
                logger.info(f"  - Node level attention MLP input dim for {edge_type}: {node_level_attention_no_edge_mlp_input_dim}")
                
                node_embeddings = graph_encoder_no_edge(
                    self.node_features,
                    self.edge_indices_dict,
                    None,
                    self.time_series_features,
                    self.node_levels
                )
                
                # 验证输出
                self.assertEqual(node_embeddings.shape, (self.num_nodes, self.output_dim))
                logger.info(f"Test 4 - Output shape: {node_embeddings.shape}")
                subtest_result["status"] = "通过"
                subtest_result["details"]["output_shape"] = f"[{self.num_nodes}, {self.output_dim}]"
                subtest_result["details"]["node_level_attention_mlp_input_dim"] = f"{node_level_attention_no_edge_mlp_input_dim}"
            except Exception as e:
                logger.error(f"测试4失败: {str(e)}")
                subtest_result["details"]["error"] = str(e)
            test_result["subtests"].append(subtest_result)
            
            # 测试5：只使用基本特征
            subtest_result = {"name": "只使用基本特征", "status": "失败", "details": {}}
            try:
                logger.info("测试5：只使用基本特征")
                node_embeddings = graph_encoder_no_edge(
                    self.node_features,
                    self.edge_indices_dict,
                    None,
                    None,
                    None
                )
                
                # 验证输出
                self.assertEqual(node_embeddings.shape, (self.num_nodes, self.output_dim))
                logger.info(f"Test 5 - Output shape: {node_embeddings.shape}")
                subtest_result["status"] = "通过"
                subtest_result["details"]["output_shape"] = f"[{self.num_nodes}, {self.output_dim}]"
            except Exception as e:
                logger.error(f"测试5失败: {str(e)}")
                subtest_result["details"]["error"] = str(e)
            test_result["subtests"].append(subtest_result)
            
            # 如果所有子测试都通过，则整体测试通过
            all_passed = all(subtest["status"] == "通过" for subtest in test_result["subtests"])
            if all_passed:
                test_result["status"] = "通过"
                test_result["details"] = {
                    "node_dim": self.node_dim,
                    "edge_dim": self.edge_dim,
                    "hidden_dim": self.hidden_dim,
                    "output_dim": self.output_dim,
                    "num_levels": self.num_levels,
                    "seq_len": self.seq_len
                }
                logger.info("动态异构图编码器测试通过！")
            else:
                logger.error("动态异构图编码器测试部分失败！")
        except Exception as e:
            logger.error(f"动态异构图编码器测试失败: {str(e)}")
            test_result["details"]["error"] = str(e)
        
        TestDynamicHeterogeneousGraphEncoder.test_results["tests"].append(test_result)


if __name__ == "__main__":
    # 运行测试
    unittest.main(exit=False)
    
    # 保存测试结果到JSON文件
    try:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(TEST_RESULTS_DIR, f"dynamic_heterogeneous_graph_encoder_test_{timestamp}.json")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(TestDynamicHeterogeneousGraphEncoder.test_results, f, ensure_ascii=False, indent=4)
        
        logger.info(f"测试结果已保存到: {result_file}")
        print(f"测试结果已保存到: {result_file}")
    except Exception as e:
        error_msg = f"无法保存测试结果: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        print(error_msg) 