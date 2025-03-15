#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-16 10:00
# @Desc   : 测试GraphTextDataset类的功能
# --------------------------------------------------------
"""

import os
import sys
import json
import logging
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.data.dataset import GraphTextDataset
from rag.data.collator import GraphTextCollator
from rag.feature_extractor import FeatureExtractor
from preprocess.utils.neo4j_graph_manager import Neo4jGraphManager
from rag.config.config import get_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试GraphTextDataset类的功能")
    parser.add_argument("--sample_count", type=int, default=10, help="每种节点类型要保存的样本数量")
    parser.add_argument("--output_file", type=str, default="graph_text_samples.json", help="输出文件名")
    parser.add_argument("--node_type", type=str, default=None, help="指定要获取的节点类型，不指定则获取所有类型")
    parser.add_argument("--comprehensive", action="store_true", help="是否进行全面覆盖测试")
    parser.add_argument("--path_length", type=int, default=3, help="复杂路径的最大长度")
    parser.add_argument("--complex_query_count", type=int, default=5, help="复杂查询样本数量")
    parser.add_argument("--stats_query_count", type=int, default=5, help="统计信息查询样本数量")
    return parser.parse_args()

def get_node_statistics(dataset):
    """获取数据集中节点类型的统计信息"""
    node_types = Counter()
    for i in range(len(dataset)):
        sample = dataset[i]
        node_types[sample["node_type"]] += 1
    
    logger.info("节点类型统计:")
    for node_type, count in node_types.items():
        logger.info(f"  - {node_type}: {count}个节点")
    
    return node_types

def get_edge_statistics(dataset):
    """获取数据集中边类型的统计信息"""
    edge_types = Counter()
    edge_pairs = Counter()  # 记录节点类型对之间的边
    
    for i in range(len(dataset)):
        sample = dataset[i]
        subgraph = sample["subgraph"]
        
        for edge_key, edge in subgraph["edges"].items():
            if "type" in edge:
                edge_types[edge["type"]] += 1
                
                # 获取源节点和目标节点的类型
                source_id = edge["source"]
                target_id = edge["target"]
                
                if source_id in subgraph["nodes"] and target_id in subgraph["nodes"]:
                    source_type = subgraph["nodes"][source_id]["type"]
                    target_type = subgraph["nodes"][target_id]["type"]
                    edge_pair = f"{source_type}-[{edge['type']}]->{target_type}"
                    edge_pairs[edge_pair] += 1
    
    logger.info("边类型统计:")
    for edge_type, count in edge_types.items():
        logger.info(f"  - {edge_type}: {count}条边")
    
    logger.info("节点类型对之间的边统计:")
    for edge_pair, count in edge_pairs.items():
        logger.info(f"  - {edge_pair}: {count}条边")
    
    return edge_types, edge_pairs

def has_dynamic_attributes(node):
    """检查节点是否具有动态属性"""
    features = node["features"]
    return (
        ("metrics_data" in features and features["metrics_data"]) or
        ("log_data" in features and features["log_data"]) or
        ("dynamics_data" in features and features["dynamics_data"])
    )

def get_dynamic_attribute_statistics(dataset):
    """获取具有动态属性的节点统计信息"""
    dynamic_nodes = defaultdict(int)
    metrics_nodes = defaultdict(int)
    log_nodes = defaultdict(int)
    dynamics_nodes = defaultdict(int)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        subgraph = sample["subgraph"]
        
        for node_id, node in subgraph["nodes"].items():
            node_type = node["type"]
            features = node["features"]
            
            if has_dynamic_attributes(node):
                dynamic_nodes[node_type] += 1
            
            if "metrics_data" in features and features["metrics_data"]:
                metrics_nodes[node_type] += 1
            
            if "log_data" in features and features["log_data"]:
                log_nodes[node_type] += 1
            
            if "dynamics_data" in features and features["dynamics_data"]:
                dynamics_nodes[node_type] += 1
    
    logger.info("具有动态属性的节点统计:")
    for node_type, count in dynamic_nodes.items():
        logger.info(f"  - {node_type}: {count}个节点具有动态属性")
    
    logger.info("具有指标数据的节点统计:")
    for node_type, count in metrics_nodes.items():
        logger.info(f"  - {node_type}: {count}个节点具有指标数据")
    
    logger.info("具有日志数据的节点统计:")
    for node_type, count in log_nodes.items():
        logger.info(f"  - {node_type}: {count}个节点具有日志数据")
    
    logger.info("具有动态特征数据的节点统计:")
    for node_type, count in dynamics_nodes.items():
        logger.info(f"  - {node_type}: {count}个节点具有动态特征数据")
    
    return dynamic_nodes, metrics_nodes, log_nodes, dynamics_nodes

def find_complex_paths(dataset, max_length=3):
    """查找复杂路径（多跳路径）"""
    complex_paths = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        subgraph = sample["subgraph"]
        
        # 构建邻接表
        adjacency = defaultdict(list)
        for edge_key, edge in subgraph["edges"].items():
            source_id = edge["source"]
            target_id = edge["target"]
            edge_type = edge.get("type", "UNKNOWN")
            
            if source_id in subgraph["nodes"] and target_id in subgraph["nodes"]:
                adjacency[source_id].append((target_id, edge_type))
        
        # 从中心节点开始，查找路径
        center_node_id = subgraph["center_node_id"]
        if center_node_id in adjacency:
            paths = find_paths_dfs(subgraph, adjacency, center_node_id, [], max_length)
            for path in paths:
                if len(path) >= 2:  # 至少包含2条边的路径
                    complex_paths.append({
                        "sample_id": i,
                        "center_node_id": center_node_id,
                        "center_node_type": subgraph["nodes"][center_node_id]["type"],
                        "path": path
                    })
    
    logger.info(f"找到 {len(complex_paths)} 条复杂路径")
    return complex_paths

def find_paths_dfs(subgraph, adjacency, current_node, current_path, max_length):
    """使用深度优先搜索查找路径"""
    if len(current_path) >= max_length:
        return [current_path]
    
    paths = []
    if current_path:
        paths.append(current_path)
    
    for neighbor, edge_type in adjacency[current_node]:
        if neighbor not in [node for node, _ in current_path]:
            new_path = current_path + [(neighbor, edge_type)]
            paths.extend(find_paths_dfs(subgraph, adjacency, neighbor, new_path, max_length))
    
    return paths

def generate_complex_query_samples(dataset, complex_paths, count=5):
    """生成复杂语义组合的查询样本"""
    if not complex_paths:
        logger.warning("没有找到复杂路径，无法生成复杂查询样本")
        return []
    
    complex_queries = []
    selected_paths = random.sample(complex_paths, min(count, len(complex_paths)))
    
    for path_info in selected_paths:
        sample_id = path_info["sample_id"]
        sample = dataset[sample_id]
        subgraph = sample["subgraph"]
        
        # 构建查询描述
        path = path_info["path"]
        query_parts = []
        
        center_node_id = path_info["center_node_id"]
        center_node = subgraph["nodes"][center_node_id]
        center_node_type = center_node["type"]
        center_node_name = center_node.get("name", f"ID为{center_node_id}的{center_node_type}")
        
        query_parts.append(f"从{center_node_name}开始")
        
        for i, (node_id, edge_type) in enumerate(path):
            node = subgraph["nodes"][node_id]
            node_type = node["type"]
            node_name = node.get("name", f"ID为{node_id}的{node_type}")
            
            if i > 0:
                query_parts.append(f"然后通过{edge_type}关系")
            else:
                query_parts.append(f"通过{edge_type}关系")
            
            query_parts.append(f"连接到{node_name}")
        
        query_text = "，".join(query_parts) + "的路径是什么？"
        
        # 构建答案描述
        answer_parts = [f"路径从{center_node_name}开始"]
        
        for i, (node_id, edge_type) in enumerate(path):
            node = subgraph["nodes"][node_id]
            node_type = node["type"]
            node_name = node.get("name", f"ID为{node_id}的{node_type}")
            
            answer_parts.append(f"通过{edge_type}关系连接到{node_name}")
        
        answer_text = "，".join(answer_parts) + "。"
        
        complex_queries.append({
            "query": query_text,
            "answer": answer_text,
            "path": [{"node_id": center_node_id, "node_type": center_node_type}] + 
                   [{"node_id": node_id, "node_type": subgraph["nodes"][node_id]["type"], "edge_type": edge_type} 
                    for node_id, edge_type in path]
        })
    
    return complex_queries

def check_neo4j_edges(graph_manager):
    """检查Neo4j数据库中的边"""
    logger.info("检查Neo4j数据库中的边...")
    
    # 获取所有边类型
    query = """
    MATCH ()-[r]->()
    RETURN DISTINCT type(r) as edge_type, count(r) as count
    """
    
    edge_types = {}
    with graph_manager._driver.session() as session:
        result = session.run(query)
        for record in result:
            edge_type = record["edge_type"]
            count = record["count"]
            edge_types[edge_type] = count
    
    if edge_types:
        logger.info("Neo4j数据库中的边类型:")
        for edge_type, count in edge_types.items():
            logger.info(f"  - {edge_type}: {count}条边")
    else:
        logger.warning("Neo4j数据库中没有边！")
    
    # 获取节点类型之间的边
    query = """
    MATCH (source)-[r]->(target)
    RETURN DISTINCT labels(source)[0] as source_type, type(r) as edge_type, labels(target)[0] as target_type, count(r) as count
    """
    
    edge_pairs = {}
    with graph_manager._driver.session() as session:
        result = session.run(query)
        for record in result:
            source_type = record["source_type"]
            edge_type = record["edge_type"]
            target_type = record["target_type"]
            count = record["count"]
            edge_pair = f"{source_type}-[{edge_type}]->{target_type}"
            edge_pairs[edge_pair] = count
    
    if edge_pairs:
        logger.info("Neo4j数据库中的节点类型对之间的边:")
        for edge_pair, count in edge_pairs.items():
            logger.info(f"  - {edge_pair}: {count}条边")
    
    return edge_types, edge_pairs

def generate_resource_usage_queries(dataset, count=5):
    """生成资源使用情况的统计信息查询样本"""
    resource_queries = []
    
    # 找到具有指标数据的节点
    nodes_with_metrics = []
    for i in range(len(dataset)):
        sample = dataset[i]
        node_id = sample["node_id"]
        node_type = sample["node_type"]
        node = sample["subgraph"]["nodes"][node_id]
        
        if "features" in node and "metrics_data" in node["features"] and node["features"]["metrics_data"]:
            nodes_with_metrics.append((i, node_id, node_type))
    
    if not nodes_with_metrics:
        logger.warning("没有找到具有指标数据的节点，无法生成资源使用情况查询")
        return []
    
    # 随机选择节点生成查询
    selected_nodes = random.sample(nodes_with_metrics, min(count, len(nodes_with_metrics)))
    
    for i, node_id, node_type in selected_nodes:
        sample = dataset[i]
        node = sample["subgraph"]["nodes"][node_id]
        node_name = node.get("name", f"ID为{node_id}的{node_type}")
        
        # 生成时间范围
        time_ranges = ["过去24小时", "过去一周", "过去一个月", "过去三个月"]
        time_range = random.choice(time_ranges)
        
        # 生成资源类型
        resource_types = {
            "VM": ["CPU", "内存", "磁盘", "网络"],
            "HOST": ["CPU", "内存", "磁盘", "网络"],
            "NE": ["连接数", "会话数", "吞吐量"],
            "HA": ["可用性", "响应时间", "故障切换次数"],
            "TRU": ["存储容量", "IOPS", "吞吐量", "延迟"]
        }
        
        resource_type = "资源"
        if node_type in resource_types:
            resource_type = random.choice(resource_types[node_type])
        
        # 生成查询
        query_text = f"{node_name}在{time_range}内的{resource_type}使用情况如何？有没有异常波动？"
        
        # 生成答案
        answer_parts = [f"{node_name}在{time_range}内的{resource_type}使用情况"]
        
        # 根据节点类型生成不同的答案
        if node_type == "VM":
            if resource_type == "CPU":
                answer_parts.append(f"平均使用率为{random.randint(10, 90)}%")
                if random.random() < 0.3:  # 30%的概率有异常
                    answer_parts.append(f"在{random.choice(['周一', '周二', '周三', '周四', '周五'])}出现了峰值{random.randint(90, 100)}%")
                    answer_parts.append("这可能是由于应用负载突增导致的")
                else:
                    answer_parts.append("使用率稳定，没有明显异常波动")
            elif resource_type == "内存":
                answer_parts.append(f"平均使用率为{random.randint(20, 80)}%")
                if random.random() < 0.3:
                    answer_parts.append(f"有{random.randint(1, 5)}次内存使用率超过90%的情况")
                    answer_parts.append("可能存在内存泄漏问题")
                else:
                    answer_parts.append("使用率稳定，没有明显异常波动")
        elif node_type == "NE":
            if resource_type == "连接数":
                answer_parts.append(f"平均连接数为{random.randint(1000, 10000)}")
                if random.random() < 0.3:
                    answer_parts.append(f"在高峰期连接数达到{random.randint(10000, 20000)}")
                    answer_parts.append("但连接成功率保持在99%以上，性能正常")
                else:
                    answer_parts.append("连接数稳定，没有明显异常波动")
        
        # 通用结尾
        if random.random() < 0.5:
            answer_parts.append(f"建议继续监控{resource_type}使用情况，确保系统稳定运行")
        
        answer_text = "。".join(answer_parts) + "。"
        
        resource_queries.append({
            "query": query_text,
            "answer": answer_text,
            "node_id": node_id,
            "node_type": node_type,
            "query_type": "resource_usage"
        })
    
    return resource_queries

def generate_anomaly_propagation_queries(dataset, count=5):
    """生成异常状态传播路径的统计信息查询样本"""
    anomaly_queries = []
    
    # 找到具有日志数据的节点
    nodes_with_logs = []
    for i in range(len(dataset)):
        sample = dataset[i]
        node_id = sample["node_id"]
        node_type = sample["node_type"]
        node = sample["subgraph"]["nodes"][node_id]
        
        if "features" in node and "log_data" in node["features"] and node["features"]["log_data"]:
            nodes_with_logs.append((i, node_id, node_type))
    
    if not nodes_with_logs:
        logger.warning("没有找到具有日志数据的节点，无法生成异常状态传播路径查询")
        return []
    
    # 随机选择节点生成查询
    selected_nodes = random.sample(nodes_with_logs, min(count, len(nodes_with_logs)))
    
    for i, node_id, node_type in selected_nodes:
        sample = dataset[i]
        subgraph = sample["subgraph"]
        node = subgraph["nodes"][node_id]
        node_name = node.get("name", f"ID为{node_id}的{node_type}")
        
        # 生成异常类型
        anomaly_types = {
            "VM": ["CPU高负载", "内存不足", "磁盘空间不足", "网络延迟高"],
            "HOST": ["CPU高负载", "内存不足", "磁盘空间不足", "网络延迟高"],
            "NE": ["连接失败率高", "会话建立失败", "吞吐量下降"],
            "HA": ["可用性下降", "响应时间增加", "频繁故障切换"],
            "TRU": ["存储容量不足", "IOPS下降", "吞吐量下降", "延迟增加"]
        }
        
        anomaly_type = "异常"
        if node_type in anomaly_types:
            anomaly_type = random.choice(anomaly_types[node_type])
        
        # 查找连接的节点，构建可能的传播路径
        connected_nodes = []
        for edge_key, edge in subgraph["edges"].items():
            if edge["source"] == node_id:
                target_id = edge["target"]
                if target_id in subgraph["nodes"]:
                    target_node = subgraph["nodes"][target_id]
                    connected_nodes.append((target_id, target_node["type"], edge["type"], "outgoing"))
            elif edge["target"] == node_id:
                source_id = edge["source"]
                if source_id in subgraph["nodes"]:
                    source_node = subgraph["nodes"][source_id]
                    connected_nodes.append((source_id, source_node["type"], edge["type"], "incoming"))
        
        # 生成查询
        query_text = f"如果{node_name}出现{anomaly_type}，可能的异常传播路径是什么？"
        
        # 生成答案
        answer_parts = [f"如果{node_name}出现{anomaly_type}"]
        
        if connected_nodes:
            # 选择1-3个连接的节点作为传播路径
            propagation_count = min(random.randint(1, 3), len(connected_nodes))
            propagation_nodes = random.sample(connected_nodes, propagation_count)
            
            for prop_node_id, prop_node_type, edge_type, direction in propagation_nodes:
                prop_node = subgraph["nodes"][prop_node_id]
                prop_node_name = prop_node.get("name", f"ID为{prop_node_id}的{prop_node_type}")
                
                if direction == "outgoing":
                    answer_parts.append(f"可能通过{edge_type}关系影响到{prop_node_name}")
                else:
                    answer_parts.append(f"可能受到{prop_node_name}通过{edge_type}关系的影响")
            
            # 添加影响描述
            effects = [
                "导致服务质量下降",
                "引起性能瓶颈",
                "造成用户体验下降",
                "影响业务连续性",
                "触发告警"
            ]
            answer_parts.append(random.choice(effects))
        else:
            answer_parts.append("由于该节点没有连接的节点，异常不会传播到其他节点")
        
        # 添加建议
        suggestions = [
            "建议及时处理异常，防止影响扩大",
            "建议监控相关节点的状态变化",
            "建议检查相关配置和资源分配",
            "建议进行根因分析，找出异常的根本原因"
        ]
        answer_parts.append(random.choice(suggestions))
        
        answer_text = "，".join(answer_parts) + "。"
        
        anomaly_queries.append({
            "query": query_text,
            "answer": answer_text,
            "node_id": node_id,
            "node_type": node_type,
            "query_type": "anomaly_propagation"
        })
    
    return anomaly_queries

def generate_hierarchy_statistics_queries(dataset, count=5):
    """生成层级统计信息的查询样本"""
    hierarchy_queries = []
    
    # 找到具有下级节点的节点
    nodes_with_children = []
    for i in range(len(dataset)):
        sample = dataset[i]
        node_id = sample["node_id"]
        node_type = sample["node_type"]
        subgraph = sample["subgraph"]
        
        # 检查是否有出边
        has_children = False
        for edge_key, edge in subgraph["edges"].items():
            if edge["source"] == node_id:
                has_children = True
                break
        
        if has_children:
            nodes_with_children.append((i, node_id, node_type))
    
    if not nodes_with_children:
        logger.warning("没有找到具有下级节点的节点，无法生成层级统计信息查询")
        return []
    
    # 随机选择节点生成查询
    selected_nodes = random.sample(nodes_with_children, min(count, len(nodes_with_children)))
    
    for i, node_id, node_type in selected_nodes:
        sample = dataset[i]
        subgraph = sample["subgraph"]
        node = subgraph["nodes"][node_id]
        node_name = node.get("name", f"ID为{node_id}的{node_type}")
        
        # 统计直接下级节点
        direct_children = {}
        for edge_key, edge in subgraph["edges"].items():
            if edge["source"] == node_id:
                target_id = edge["target"]
                if target_id in subgraph["nodes"]:
                    target_type = subgraph["nodes"][target_id]["type"]
                    if target_type not in direct_children:
                        direct_children[target_type] = []
                    direct_children[target_type].append(target_id)
        
        # 统计二级下级节点
        second_level_children = {}
        for child_type, child_ids in direct_children.items():
            for child_id in child_ids:
                for edge_key, edge in subgraph["edges"].items():
                    if edge["source"] == child_id:
                        target_id = edge["target"]
                        if target_id in subgraph["nodes"]:
                            target_type = subgraph["nodes"][target_id]["type"]
                            if target_type not in second_level_children:
                                second_level_children[target_type] = []
                            second_level_children[target_type].append(target_id)
        
        # 生成查询
        query_text = f"{node_name}包含哪些下级节点和下下级节点？各有多少个？"
        
        # 生成答案
        answer_parts = [f"{node_name}的层级结构如下"]
        
        # 添加直接下级节点信息
        if direct_children:
            answer_parts.append("直接下级节点包括")
            child_descriptions = []
            for child_type, child_ids in direct_children.items():
                child_count = len(set(child_ids))  # 去重计数
                child_descriptions.append(f"{child_count}个{child_type}节点")
            answer_parts.append("、".join(child_descriptions))
        else:
            answer_parts.append("没有直接下级节点")
        
        # 添加二级下级节点信息
        if second_level_children:
            answer_parts.append("二级下级节点包括")
            child_descriptions = []
            for child_type, child_ids in second_level_children.items():
                child_count = len(set(child_ids))  # 去重计数
                child_descriptions.append(f"{child_count}个{child_type}节点")
            answer_parts.append("、".join(child_descriptions))
        else:
            answer_parts.append("没有二级下级节点")
        
        # 添加总结
        total_direct = sum(len(set(ids)) for ids in direct_children.values())
        total_second = sum(len(set(ids)) for ids in second_level_children.values())
        answer_parts.append(f"总计：直接下级节点{total_direct}个，二级下级节点{total_second}个")
        
        answer_text = "。".join(answer_parts) + "。"
        
        hierarchy_queries.append({
            "query": query_text,
            "answer": answer_text,
            "node_id": node_id,
            "node_type": node_type,
            "query_type": "hierarchy_statistics"
        })
    
    return hierarchy_queries

def generate_statistics_queries(dataset, count=5):
    """生成各类统计信息查询样本"""
    # 生成不同类型的统计信息查询
    resource_queries = generate_resource_usage_queries(dataset, count)
    anomaly_queries = generate_anomaly_propagation_queries(dataset, count)
    hierarchy_queries = generate_hierarchy_statistics_queries(dataset, count)
    
    # 合并所有查询
    all_queries = resource_queries + anomaly_queries + hierarchy_queries
    
    # 如果查询数量不足，记录警告
    if len(all_queries) < count:
        logger.warning(f"只生成了{len(all_queries)}个统计信息查询样本，少于请求的{count}个")
    
    # 随机选择指定数量的查询
    if len(all_queries) > count:
        all_queries = random.sample(all_queries, count)
    
    logger.info(f"生成了{len(all_queries)}个统计信息查询样本")
    logger.info(f"  - 资源使用情况查询: {len(resource_queries)}个")
    logger.info(f"  - 异常传播路径查询: {len(anomaly_queries)}个")
    logger.info(f"  - 层级统计信息查询: {len(hierarchy_queries)}个")
    
    return all_queries

def test_graph_text_dataset(sample_count=10, output_file="graph_text_samples.json", node_type=None, 
                           comprehensive=False, path_length=3, complex_query_count=5, stats_query_count=5):
    """测试GraphTextDataset类的功能"""
    # 加载配置
    config = get_config()
    
    # 创建输出目录
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    # 初始化Neo4j连接
    neo4j_config = config.get("neo4j", {})
    graph_manager = Neo4jGraphManager(
        uri=neo4j_config.get("uri", "bolt://localhost:7687"),
        user=neo4j_config.get("user", "neo4j"),
        password=neo4j_config.get("password", "password")
    )
    
    # 检查Neo4j数据库中的边
    neo4j_edge_types, neo4j_edge_pairs = check_neo4j_edges(graph_manager)
    
    # 初始化特征提取器
    feature_extractor = FeatureExtractor(graph_manager)
    
    # 获取所有可用的节点类型和边类型
    all_node_types = ["DC", "TENANT", "NE", "VM", "HOST", "HA", "TRU"]
    all_edge_types = list(neo4j_edge_types.keys()) if neo4j_edge_types else ["CONTAINS", "DEPLOYED_ON", "CONNECTS_TO"]
    
    # 创建数据集
    logger.info("创建GraphTextDataset...")
    dataset = GraphTextDataset(
        graph_manager=graph_manager,
        feature_extractor=feature_extractor,
        node_types=all_node_types,  # 使用所有节点类型
        edge_types=all_edge_types,  # 使用所有边类型
        max_node_size=50,  # 限制子图大小
        max_edge_size=100,
        include_dynamic=True,
        data_augmentation=False,  # 测试时不使用数据增强
        split="test"
    )
    
    # 打印数据集统计信息
    logger.info(f"数据集大小: {len(dataset)}")
    
    # 如果进行全面覆盖测试，则收集统计信息
    if comprehensive:
        logger.info("进行全面覆盖测试...")
        node_stats = get_node_statistics(dataset)
        edge_stats, edge_pair_stats = get_edge_statistics(dataset)
        dynamic_stats, metrics_stats, log_stats, dynamics_stats = get_dynamic_attribute_statistics(dataset)
        complex_paths = find_complex_paths(dataset, max_length=path_length)
        
        # 生成复杂查询样本
        complex_queries = generate_complex_query_samples(dataset, complex_paths, count=complex_query_count)
        
        # 保存复杂查询样本
        if complex_queries:
            complex_query_path = output_dir / "complex_queries.json"
            with open(complex_query_path, "w", encoding="utf-8") as f:
                json.dump(complex_queries, f, ensure_ascii=False, indent=2)
            logger.info(f"保存了 {len(complex_queries)} 个复杂查询样本到 {complex_query_path}")
        
        # 生成统计信息查询样本
        stats_queries = generate_statistics_queries(dataset, count=stats_query_count)
        
        # 保存统计信息查询样本
        if stats_queries:
            stats_query_path = output_dir / "stats_queries.json"
            with open(stats_query_path, "w", encoding="utf-8") as f:
                json.dump(stats_queries, f, ensure_ascii=False, indent=2)
            logger.info(f"保存了 {len(stats_queries)} 个统计信息查询样本到 {stats_query_path}")
    
    # 保存样本
    samples = []
    
    # 如果指定了节点类型，则只获取该类型的节点
    if node_type:
        logger.info(f"只获取类型为 {node_type} 的节点")
        filtered_indices = [i for i, sample in enumerate(dataset) if sample["node_type"] == node_type]
        if not filtered_indices:
            logger.warning(f"未找到类型为 {node_type} 的节点")
            return
        indices = filtered_indices[:min(sample_count, len(filtered_indices))]
        
    # 如果进行全面覆盖测试，则确保每种节点类型都有样本
    elif comprehensive:
        indices = []
        for node_type in all_node_types:
            type_indices = [i for i, sample in enumerate(dataset) if sample["node_type"] == node_type]
            if type_indices:
                # 为每种节点类型选择样本
                selected = type_indices[:min(sample_count, len(type_indices))]
                indices.extend(selected)
                logger.info(f"为 {node_type} 类型选择了 {len(selected)} 个样本")
            else:
                logger.warning(f"未找到类型为 {node_type} 的节点")
        
        # 确保包含具有动态属性的节点
        dynamic_indices = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if i not in indices and has_dynamic_attributes(sample["subgraph"]["nodes"][sample["node_id"]]):
                dynamic_indices.append(i)
        
        if dynamic_indices:
            selected_dynamic = dynamic_indices[:min(sample_count, len(dynamic_indices))]
            indices.extend(selected_dynamic)
            logger.info(f"额外选择了 {len(selected_dynamic)} 个具有动态属性的节点样本")
    else:
        indices = range(min(sample_count, len(dataset)))
    
    for i in indices:
        sample = dataset[i]
        
        # 简化子图以便于查看
        simplified_subgraph = {
            "center_node_id": sample["subgraph"]["center_node_id"],
            "node_count": len(sample["subgraph"]["nodes"]),
            "edge_count": len(sample["subgraph"]["edges"]),
            "node_types": list(set(node["type"] for node in sample["subgraph"]["nodes"].values())),
            "edge_types": list(set(edge["type"] for edge in sample["subgraph"]["edges"].values() if "type" in edge))
        }
        
        # 创建简化的样本
        simplified_sample = {
            "node_id": sample["node_id"],
            "node_type": sample["node_type"],
            "text": sample["text"],
            "subgraph": simplified_subgraph
        }
        
        samples.append(simplified_sample)
    
    # 保存样本到文件
    output_path = output_dir / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"保存了 {len(samples)} 个样本到 {output_path}")
    
    # 如果进行全面覆盖测试，则保存统计信息
    if comprehensive:
        stats = {
            "node_types": {k: v for k, v in node_stats.items()},
            "edge_types": {k: v for k, v in edge_stats.items()},
            "edge_pairs": {k: v for k, v in edge_pair_stats.items()},
            "dynamic_nodes": {k: v for k, v in dynamic_stats.items()},
            "metrics_nodes": {k: v for k, v in metrics_stats.items()},
            "log_nodes": {k: v for k, v in log_stats.items()},
            "dynamics_nodes": {k: v for k, v in dynamics_stats.items()},
            "complex_paths_count": len(complex_paths)
        }
        
        stats_path = output_dir / "coverage_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"保存了覆盖统计信息到 {stats_path}")
    
    # 测试数据加载器
    logger.info("测试数据加载器...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    collator = GraphTextCollator(tokenizer=tokenizer)
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collator,
        num_workers=0
    )
    
    # 获取一个批次
    for batch in dataloader:
        logger.info(f"批次大小: {len(batch['node_ids'])}")
        logger.info(f"输入ID形状: {batch['input_ids'].shape}")
        logger.info(f"注意力掩码形状: {batch['attention_mask'].shape}")
        logger.info(f"节点特征形状: {batch['node_features'].shape}")
        logger.info(f"边索引形状: {batch['edge_index'].shape}")
        logger.info(f"边特征形状: {batch['edge_features'].shape}")
        logger.info(f"批次索引形状: {batch['batch_idx'].shape}")
        logger.info(f"中心节点索引: {batch['center_node_idx']}")
        break
    
    logger.info("GraphTextDataset测试完成")

if __name__ == "__main__":
    args = parse_args()
    test_graph_text_dataset(
        sample_count=args.sample_count, 
        output_file=args.output_file, 
        node_type=args.node_type,
        comprehensive=args.comprehensive,
        path_length=args.path_length,
        complex_query_count=args.complex_query_count,
        stats_query_count=args.stats_query_count
    ) 