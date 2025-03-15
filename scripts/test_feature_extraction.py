#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试特征提取器，验证提取的特征是否符合设计要求
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.feature_extractor import FeatureExtractor
from preprocess.utils.neo4j_graph_manager import Neo4jGraphManager
import json
from pprint import pprint

def test_node_feature_extraction():
    """测试节点特征提取"""
    
    # 初始化Neo4j连接和特征提取器
    graph_manager = Neo4jGraphManager(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="Oms_2600a"
    )
    
    extractor = FeatureExtractor(graph_manager)
    
    # 获取每种类型的示例节点
    node_samples = {}
    
    print("开始提取节点特征样本...")
    for node_type in extractor.node_types:
        query = f"""
        MATCH (n:{node_type})
        RETURN n, id(n) as node_id LIMIT 1
        """
        with graph_manager._driver.session() as session:
            result = session.run(query)
            record = result.single()
            if record:
                node = record['n']
                node_id = record['node_id']  # 使用Neo4j内部ID
                features = extractor.extract_node_features(node_id, node_type)
                node_samples[node_type] = {
                    'raw_data': dict(node.items()),
                    'extracted_features': features
                }
    
    # 保存结果
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "node_feature_samples.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(node_samples, f, ensure_ascii=False, indent=2)
    
    print("\n节点特征提取结果:")
    for node_type, data in node_samples.items():
        print("原始数据:")
        pprint(data['raw_data'])
        print("\n提取的特征:")
        pprint(data['extracted_features'])
        print("-" * 80)
    
    print(f"\n结果已保存至: {output_file}")

def test_edge_feature_extraction():
    """测试边特征提取"""
    
    # 初始化Neo4j连接和特征提取器
    graph_manager = Neo4jGraphManager(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="Oms_2600a"
    )
    
    extractor = FeatureExtractor(graph_manager)
    
    # 获取每种类型的示例边
    edge_samples = {}
    
    print("\n开始提取边特征样本...")
    for edge_type in extractor.relationship_types:
        query = f"""
        MATCH ()-[r:{edge_type}]->()
        RETURN r, id(startNode(r)) as source_id, id(endNode(r)) as target_id LIMIT 1
        """
        with graph_manager._driver.session() as session:
            result = session.run(query)
            record = result.single()
            if record:
                rel = record['r']
                source_id = record['source_id']  # 使用Neo4j内部ID
                target_id = record['target_id']  # 使用Neo4j内部ID
    
                features = extractor.extract_edge_features(source_id, target_id, edge_type)
                edge_samples[edge_type] = {
                    'raw_data': dict(rel.items()),
                    'extracted_features': features
                }
    
    # 保存结果
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "edge_feature_samples.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(edge_samples, f, ensure_ascii=False, indent=2)
    
    print("\n边特征提取结果:")
    for edge_type, data in edge_samples.items():
        print(f"\n{edge_type} 边:")
        print("原始数据:")
        pprint(data['raw_data'])
        print("\n提取的特征:")
        pprint(data['extracted_features'])
        print("-" * 80)
    
    print(f"\n结果已保存至: {output_file}")

def test_chain_feature_extraction():
    """测试链路特征提取"""
    
    # 初始化Neo4j连接和特征提取器
    graph_manager = Neo4jGraphManager(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="Oms_2600a"
    )
    
    extractor = FeatureExtractor(graph_manager)
    
    # 获取一个DC节点作为起点
    dc_id = None
    
    # 先检查数据库中是否存在DC节点
    check_query = """
    MATCH (n:DC)
    RETURN count(n) as dc_count
    """
    
    with graph_manager._driver.session() as session:
        result = session.run(check_query)
        record = result.single()
        if record:
            dc_count = record['dc_count']
            print(f"数据库中存在 {dc_count} 个DC节点")
    
    # 使用更精确的查询获取DC节点
    query = """
    MATCH (dc:DC)
    WHERE dc.type = 'DC'
    RETURN id(dc) as dc_id, dc.id as dc_name LIMIT 1
    """
    
    with graph_manager._driver.session() as session:
        result = session.run(query)
        record = result.single()
        if record:
            dc_id = record['dc_id']
            dc_name = record['dc_name']
            print(f"找到DC节点: ID={dc_id}, Name={dc_name}")
    
    # 如果第一个查询没有找到，尝试使用备用查询
    if dc_id is None:
        print("未找到DC节点，尝试使用备用查询...")
        # 备用查询，不使用type属性
        backup_query = """
        MATCH (dc:DC)
        RETURN id(dc) as dc_id, dc.id as dc_name LIMIT 1
        """
        
        with graph_manager._driver.session() as session:
            result = session.run(backup_query)
            record = result.single()
            if record:
                dc_id = record['dc_id']
                dc_name = record['dc_name']
                print(f"使用备用查询找到DC节点: ID={dc_id}, Name={dc_name}")
    
    # 确保dc_id不为None且为整数
    if dc_id is not None:
        try:
            dc_id = int(dc_id)
            print(f"将使用DC节点ID: {dc_id} (转换为整数) 进行链路特征提取")
        except (ValueError, TypeError):
            print(f"无法将DC节点ID转换为整数: {dc_id}")
            dc_id = None
    
    if dc_id is None:
        print("未找到有效的DC节点ID，无法测试链路特征提取")
        return
    
    print(f"\n开始提取链路特征，起始DC节点ID: {dc_id}")
    
    # 测试三种链路类型
    chain_types = ['business', 'resource', 'both']
    chain_samples = {}
    
    for chain_type in chain_types:
        print(f"\n提取 {chain_type} 类型的链路特征...")
        chain_features = extractor.extract_chain_features(dc_id, chain_type)
        
        # 统计节点和边的数量
        node_counts = {node_type: len(nodes) for node_type, nodes in chain_features['nodes'].items() if nodes}
        edge_counts = {edge_type: len(edges) for edge_type, edges in chain_features['edges'].items() if edges}
        
        chain_samples[chain_type] = {
            'chain_info': chain_features['chain_info'],
            'node_counts': node_counts,
            'edge_counts': edge_counts,
            'full_features': chain_features
        }
    
    # 保存结果
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "chain_feature_samples.json")
    
    # 保存完整结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chain_samples, f, ensure_ascii=False, indent=2)
    
    # 打印摘要信息
    print("\n链路特征提取结果摘要:")
    for chain_type, data in chain_samples.items():
        print(f"\n{chain_type} 链路:")
        print("链路信息:")
        pprint(data['chain_info'])
        print("\n节点数量:")
        pprint(data['node_counts'])
        print("\n边数量:")
        pprint(data['edge_counts'])
        print("-" * 80)
    
    print(f"\n结果已保存至: {output_file}")

def main():
    print("开始测试特征提取...")
    test_node_feature_extraction()
    test_edge_feature_extraction()
    test_chain_feature_extraction()
    print("\n测试完成!")

if __name__ == "__main__":
    main() 