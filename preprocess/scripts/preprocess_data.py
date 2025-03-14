#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-03-11 16:30
# @Desc   : 5G资源拓扑数据预处理
# --------------------------------------------------------
"""

import os
import sys
import argparse
import json
from datetime import datetime
from collections import defaultdict


# 将项目根目录添加到系统路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

import yaml
from preprocess.utils.data_preprocessor import ResourcePreprocessor
from preprocess.utils.metrics_generator import MetricsGenerator
from preprocess.utils.log_generator import LogGenerator
from preprocess.utils.edge_dynamics_generator import EdgeDynamicsGenerator

def main():
    parser = argparse.ArgumentParser(description='Process resource topology data and generate graph data')
    parser.add_argument('--input', type=str, required=True, help='Input Excel file path')
    parser.add_argument('--output', type=str, required=True, help='Output directory path')
    parser.add_argument('--time_window', type=int, default=168, help='Time window in hours (default: 168 hours/1 week)')
    parser.add_argument('--interval', type=int, default=60, help='Sampling interval in minutes (default: 60 minutes)')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads (default: 8)')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 1. 处理静态拓扑数据
    print("1. 处理静态拓扑数据...")
    preprocessor = ResourcePreprocessor(args.input)
    graph_data = preprocessor.preprocess(args.input)
    
    # 保存原始图数据（用于调试）
    graph_data_path = os.path.join(args.output, 'graph_data.json')
    with open(graph_data_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    # 2. 生成动态性能指标数据
    print("2. 生成动态性能指标数据...")
    metrics_output_path = os.path.join(args.output, 'metrics_data.json')
    metrics_generator = MetricsGenerator(graph_data_path, metrics_output_path, max_workers=args.workers)
    metrics_data = metrics_generator.generate_metrics()
    
    # 保存性能指标中间结果
    metrics_intermediate = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics_by_node': metrics_data,
        'statistics': {
            'total_nodes': len(graph_data['nodes']),
            'nodes_with_metrics': sum(1 for node_id in metrics_data if metrics_data[node_id]),
            'total_metrics': sum(len(metrics_data[node_id]) for node_id in metrics_data if metrics_data[node_id])
        }
    }
    metrics_intermediate_path = os.path.join(args.output, 'metrics_intermediate.json')
    with open(metrics_intermediate_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_intermediate, f, ensure_ascii=False, indent=2)
    
    # 3. 将性能指标数据集成到图数据中
    print("3. 集成性能指标数据...")
    for node in graph_data['nodes']:
        if node['type'] in ['NE', 'VM', 'HOST', 'HA', 'TRU']:
            node['metrics'] = {
                'metrics_data': {
                    node['id']: metrics_data.get(node['id'], [])
                }
            }
    
    # 保存带性能指标的图数据
    metrics_graph_path = os.path.join(args.output, 'graph_data_with_metrics.json')
    with open(metrics_graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    # 4. 生成日志状态数据
    print("4. 生成日志状态数据...")
    log_generator = LogGenerator(metrics_graph_path, max_workers=args.workers)
    graph_data = log_generator.generate_logs()
    
    # 保存日志中间结果
    logs_intermediate = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'logs_by_node': {},
        'statistics': {
            'total_nodes': len(graph_data['nodes']),
            'nodes_with_logs': 0,
            'total_log_entries': 0
        }
    }
    
    # 收集日志数据统计
    for node in graph_data['nodes']:
        if 'logs' in node and 'log_data' in node['logs']:
            node_logs = node['logs']['log_data'].get(node['id'], [])
            if node_logs:
                logs_intermediate['logs_by_node'][node['id']] = {
                    'node_type': node['type'],
                    'log_entries': len(node_logs),
                    'status_distribution': defaultdict(lambda: defaultdict(int))
                }
                logs_intermediate['statistics']['nodes_with_logs'] += 1
                logs_intermediate['statistics']['total_log_entries'] += len(node_logs)
                
                # 统计每个字段的状态分布
                for log_entry in node_logs:
                    for field, status in log_entry['status'].items():
                        logs_intermediate['logs_by_node'][node['id']]['status_distribution'][field][str(status)] += 1
    
    # 将defaultdict转换为普通dict以便JSON序列化
    for node_id in logs_intermediate['logs_by_node']:
        logs_intermediate['logs_by_node'][node_id]['status_distribution'] = {
            field: dict(status_counts)
            for field, status_counts in logs_intermediate['logs_by_node'][node_id]['status_distribution'].items()
        }
    
    logs_intermediate_path = os.path.join(args.output, 'logs_intermediate.json')
    with open(logs_intermediate_path, 'w', encoding='utf-8') as f:
        json.dump(logs_intermediate, f, ensure_ascii=False, indent=2)
    
    # 保存带日志的图数据
    logs_graph_path = os.path.join(args.output, 'graph_data_with_logs.json')
    with open(logs_graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    # 5. 生成边的动态数据
    print("5. 生成边的动态数据...")
    edge_dynamics_generator = EdgeDynamicsGenerator(logs_graph_path, max_workers=args.workers)
    graph_data = edge_dynamics_generator.generate_edge_dynamics()
    
    # 保存边动态数据中间结果
    edge_dynamics_intermediate = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dynamics_by_edge': {},
        'statistics': {
            'total_edges': len(graph_data['edges']),
            'edges_with_dynamics': 0,
            'total_dynamics_entries': 0
        }
    }
    
    # 收集边动态数据统计
    for edge in graph_data['edges']:
        if 'dynamics' in edge and edge['dynamics']:
            edge_id = f"{edge['source']}->{edge['target']}"
            dynamics_data = edge['dynamics']
            
            if isinstance(dynamics_data, dict):
                edge_dynamics_intermediate['dynamics_by_edge'][edge_id] = {
                    'source': edge['source'],
                    'target': edge['target'],
                    'dynamics_entries': len(dynamics_data),
                    'propagation_data': dynamics_data
                }
                edge_dynamics_intermediate['statistics']['edges_with_dynamics'] += 1
                edge_dynamics_intermediate['statistics']['total_dynamics_entries'] += len(dynamics_data)
            elif isinstance(dynamics_data, list):
                edge_dynamics_intermediate['dynamics_by_edge'][edge_id] = {
                    'source': edge['source'],
                    'target': edge['target'],
                    'dynamics_entries': len(dynamics_data),
                    'propagation_data': [
                        entry if isinstance(entry, dict) else {'timestamp': str(entry), 'propagation_prob': 0.0}
                        for entry in dynamics_data
                    ]
                }
                edge_dynamics_intermediate['statistics']['edges_with_dynamics'] += 1
                edge_dynamics_intermediate['statistics']['total_dynamics_entries'] += len(dynamics_data)
    
    edge_dynamics_intermediate_path = os.path.join(args.output, 'edge_dynamics_intermediate.json')
    with open(edge_dynamics_intermediate_path, 'w', encoding='utf-8') as f:
        json.dump(edge_dynamics_intermediate, f, ensure_ascii=False, indent=2)
    
    # 6. 保存完整的图数据
    output_path = os.path.join(args.output, 'graph_data_complete.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    print("\n数据处理完成！")
    print(f"- 原始图数据: {graph_data_path}")
    print(f"- 性能指标中间结果: {metrics_intermediate_path}")
    print(f"- 性能指标数据: {metrics_output_path}")
    print(f"- 带性能指标的图数据: {metrics_graph_path}")
    print(f"- 日志中间结果: {logs_intermediate_path}")
    print(f"- 带日志的图数据: {logs_graph_path}")
    print(f"- 边动态数据中间结果: {edge_dynamics_intermediate_path}")
    print(f"- 完整图数据（含指标、日志和边动态）: {output_path}")
    
    # 打印统计信息
    total_nodes = len(graph_data['nodes'])
    total_edges = len(graph_data['edges'])
    nodes_with_metrics = metrics_intermediate['statistics']['nodes_with_metrics']
    nodes_with_logs = logs_intermediate['statistics']['nodes_with_logs']
    edges_with_dynamics = edge_dynamics_intermediate['statistics']['edges_with_dynamics']
    
    print(f"\n统计信息:")
    print(f"- 总节点数: {total_nodes}")
    print(f"- 总边数: {total_edges}")
    print(f"- 包含性能指标的节点数: {nodes_with_metrics}")
    print(f"- 包含日志状态的节点数: {nodes_with_logs}")
    print(f"- 包含动态数据的边数: {edges_with_dynamics}")
    print(f"- 性能指标覆盖率: {nodes_with_metrics/total_nodes*100:.2f}%")
    print(f"- 日志状态覆盖率: {nodes_with_logs/total_nodes*100:.2f}%")
    print(f"- 边动态数据覆盖率: {edges_with_dynamics/total_edges*100:.2f}%")

if __name__ == "__main__":
    main() 