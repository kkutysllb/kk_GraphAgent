#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-03-11 16:30
# @Desc   : 边的动态数据生成器
# --------------------------------------------------------
"""
import json
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .topology_dynamics_generator import TopologyDynamicsGenerator

class EdgeDynamicsGenerator:
    def __init__(self, graph_data_path, max_workers=8):
        """初始化边的动态数据生成器
        
        Args:
            graph_data_path: 包含节点动态数据的图数据文件路径
            max_workers: 最大线程数，默认为8
        """
        self.graph_data_path = graph_data_path
        self.max_workers = max_workers
        self.graph_data = self._load_data()
        
        # 初始化拓扑动态生成器
        self.topology_generator = TopologyDynamicsGenerator(graph_data_path)
        
        # 定义传播影响权重
        self.propagation_weights = {
            # TRU -> HOST
            ('TRU', 'HOST'): {
                'storage_log_status': {
                    'disk_log_status': 0.9,  # 存储异常对主机磁盘的影响权重
                    'network_log_status': 0.3  # 存储异常对主机网络的影响权重
                }
            },
            # HOST -> VM
            ('HOST', 'VM'): {
                'cpu_log_status': {
                    'cpu_log_status': 0.8
                },
                'memory_log_status': {
                    'memory_log_status': 0.8
                },
                'disk_log_status': {
                    'disk_log_status': 0.7
                },
                'network_log_status': {
                    'network_log_status': 0.9
                }
            },
            # VM -> NE
            ('VM', 'NE'): {
                'cpu_log_status': {
                    'session_success_log_status': 0.6,
                    'connection_success_log_status': 0.4
                },
                'memory_log_status': {
                    'session_success_log_status': 0.5
                },
                'network_log_status': {
                    'connection_success_log_status': 0.8,
                    'connection_log_status': 0.7
                },
                'disk_log_status': {
                    'session_success_log_status': 0.4
                }
            },
            # NE -> VM (反向影响)
            ('NE', 'VM'): {
                'session_log_status': {
                    'cpu_log_status': 0.4,
                    'memory_log_status': 0.4
                },
                'connection_log_status': {
                    'network_log_status': 0.6
                }
            }
        }
    
    def _load_data(self):
        """加载图数据"""
        with open(self.graph_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_node_by_id(self, node_id):
        """通过ID获取节点"""
        return next((node for node in self.graph_data['nodes'] if node['id'] == node_id), None)
    
    def _calculate_propagation_probability(self, source_node, target_node, edge, timestamp):
        """计算传播概率
        
        Args:
            source_node: 源节点
            target_node: 目标节点
            edge: 边数据
            timestamp: 时间戳
        """
        # 获取节点类型
        source_type = source_node['type']
        target_type = target_node['type']
        
        # 获取对应的权重配置
        weights = self.propagation_weights.get((source_type, target_type), {})
        if not weights:
            return None
        
        # 获取源节点的状态
        source_logs = source_node.get('logs', {}).get('log_data', {}).get(source_node['id'], [])
        if not source_logs:
            return None
        
        # 获取指定时间点的状态
        source_status = {}
        for log in source_logs:
            if log['timestamp'] == timestamp:
                source_status = log['status']
                break
        
        if not source_status:
            return None
        
        # 计算传播效果
        propagation_effects = {}
        
        for source_field, source_value in source_status.items():
            if source_field in weights:
                for target_field, weight in weights[source_field].items():
                    # 基于源状态值和权重计算传播概率
                    probability = (source_value / 3.0) * weight  # 归一化状态值(0-3)并应用权重
                    
                    # 添加随机扰动
                    noise = np.random.normal(0, 0.1)  # 添加少量随机性
                    probability = max(0, min(1, probability + noise))  # 确保概率在[0,1]范围内
                    
                    propagation_effects[target_field] = {
                        'probability': probability,
                        'source_status': source_value
                    }
        
        return propagation_effects if propagation_effects else None
    
    def generate_edge_dynamics(self):
        """生成边的动态数据，包括状态传播和拓扑变化
        
        Returns:
            dict: 更新后的图数据
        """
        edges = self.graph_data['edges']
        print(f"开始生成 {len(edges)} 条边的动态数据...")
        
        # 1. 生成拓扑动态变化数据
        # 设置时间范围为未来7天
        start_time = datetime(2025, 3, 4, 0, 0, 0)  # 从2025年3月4日开始
        end_time = start_time + timedelta(days=7)
        topology_dynamics = self.topology_generator.generate_topology_dynamics(start_time, end_time)
        
        # 创建时间点到拓扑变化的映射
        topology_changes = {}
        for change in topology_dynamics:
            topology_changes[change['timestamp']] = change['topology_change']
        
        # 2. 生成状态传播动态数据
        # 创建边ID到边的映射
        edge_map = {(edge['source'], edge['target']): edge for edge in edges}
        
        # 收集所有需要处理的任务
        tasks = []
        for edge in edges:
            source_node = self._get_node_by_id(edge['source'])
            target_node = self._get_node_by_id(edge['target'])
            
            if not source_node or not target_node:
                continue
            
            # 获取源节点的日志时间序列
            source_logs = source_node.get('logs', {}).get('log_data', {}).get(source_node['id'], [])
            if not source_logs:
                continue
            
            # 初始化边的动态数据
            edge['dynamics'] = {
                'propagation_data': []
            }
            
            # 为每个时间点创建任务
            for log in source_logs:
                tasks.append({
                    'source_node': source_node,
                    'target_node': target_node,
                    'edge': edge,
                    'timestamp': log['timestamp'],
                    'edge_key': (edge['source'], edge['target'])
                })
        
        # 使用线程池处理任务
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = []
            for task in tasks:
                futures.append(
                    executor.submit(
                        self._calculate_propagation_probability,
                        task['source_node'],
                        task['target_node'],
                        task['edge'],
                        task['timestamp']
                    )
                )
            
            # 使用tqdm创建进度条
            with tqdm(total=len(futures), desc="生成边动态数据") as pbar:
                # 处理完成的任务
                for future, task in zip(as_completed(futures), tasks):
                    propagation_effects = future.result()
                    if propagation_effects:
                        edge = edge_map[task['edge_key']]
                        
                        # 获取当前时间点的拓扑变化
                        topology_change = topology_changes.get(task['timestamp'])
                        
                        # 构建完整的动态数据
                        dynamic_data = {
                            'timestamp': task['timestamp'],
                            'effects': propagation_effects
                        }
                        
                        # 如果存在拓扑变化，添加到动态数据中
                        if topology_change:
                            # 检查当前边是否受到拓扑变化影响
                            edge_id = f"{task['edge_key'][0]}->{task['edge_key'][1]}"
                            for change in topology_change['related_changes']:
                                if change['edge_id'] == edge_id:
                                    dynamic_data['topology_change'] = {
                                        'status': change['status']
                                    }
                                    if 'new_edge' in change:
                                        dynamic_data['topology_change']['new_edge'] = change['new_edge']
                                    if 'migration_type' in change:
                                        dynamic_data['topology_change']['migration_type'] = change['migration_type']
                                    break
                        
                        # 记录动态数据
                        edge['dynamics']['propagation_data'].append(dynamic_data)
                    pbar.update(1)
        
        # 清理没有动态数据的边
        for edge in edges:
            if not edge.get('dynamics', {}).get('propagation_data'):
                edge.pop('dynamics', None)
        
        return self.graph_data
    
    def save_data(self, output_path):
        """保存更新后的图数据
        
        Args:
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.graph_data, f, ensure_ascii=False, indent=2) 