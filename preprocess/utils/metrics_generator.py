#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-03-11 16:30
# @Desc   : 5G资源拓扑性能指标模拟数据生成
# --------------------------------------------------------
"""
import json
import random
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class MetricsGenerator:
    def __init__(self, graph_data_path, output_path, max_workers=8):
        """初始化性能指标生成器
        
        Args:
            graph_data_path: 静态图数据文件路径
            output_path: 性能指标数据输出路径
            max_workers: 最大线程数
        """
        self.graph_data_path = graph_data_path
        self.output_path = output_path
        self.max_workers = max_workers
        self.graph_data = self._load_graph_data()
        self.nodes = {node['id']: node for node in self.graph_data['nodes']}
        self.edges = self._process_edges()
        
        # 采样参数
        self.start_time = datetime.now() - timedelta(days=7)
        self.end_time = datetime.now()
        self.sample_interval = timedelta(minutes=5)
        
        # 性能指标基准值和波动范围
        self.metrics_config = {
            'NE': {
                'connection_count': {'base': 1500, 'std': 300, 'min': 0, 'max': 3000},
                'session_count': {'base': 800, 'std': 150, 'min': 0, 'max': 1500},
                'session_success_rate': {'base': 99.8, 'std': 0.3, 'min': 90, 'max': 100},
                'connection_success_rate': {'base': 99.9, 'std': 0.2, 'min': 90, 'max': 100}
            },
            'VM': {
                'vcpu_usage': {'base': 40, 'std': 10, 'min': 0, 'max': 100},
                'vmem_usage': {'base': 50, 'std': 10, 'min': 0, 'max': 100},
                'disk_io': {'base': 2000, 'std': 500, 'min': 0, 'max': 10000},
                'network_io': {'base': 200, 'std': 50, 'min': 0, 'max': 1000}
            },
            'HOST': {
                'cpu_usage': {'base': 50, 'std': 10, 'min': 0, 'max': 100},
                'mem_usage': {'base': 60, 'std': 10, 'min': 0, 'max': 100},
                'disk_io': {'base': 8000, 'std': 2000, 'min': 0, 'max': 30000},
                'network_io': {'base': 800, 'std': 200, 'min': 0, 'max': 3000}
            },
            'HA': {
                'avg_cpu_usage': {'base': 50, 'std': 8, 'min': 0, 'max': 100},
                'avg_mem_usage': {'base': 60, 'std': 8, 'min': 0, 'max': 100}
            },
            'TRU': {
                'storage_usage': {'base': 70, 'std': 5, 'min': 0, 'max': 100}
            }
        }
        
        # 资源关联影响配置
        self.impact_config = {
            'TRU_HOST': 0.3,  # TRU性能对HOST的影响权重
            'HOST_VM': 0.5,   # HOST性能对VM的影响权重
            'VM_NE': 0.7,     # VM性能对NE的影响权重
            'HA_HOST': 0.4    # HA状态对HOST的影响权重
        }
        
        # 异常场景配置
        self.anomaly_config = {
            'probability': 0.01,  # 每个时间点发生异常的概率
            'duration': {         # 异常持续时间范围（分钟）
                'min': 15,
                'max': 60
            },
            'scenarios': {
                'resource_exhaustion': {  # 资源耗尽
                    'probability': 0.3,
                    'impact': 0.8
                },
                'performance_degradation': {  # 性能下降
                    'probability': 0.4,
                    'impact': 0.5
                },
                'service_interruption': {  # 服务中断
                    'probability': 0.3,
                    'impact': 1.0
                }
            }
        }
        
        # 记录当前活跃的异常
        self.active_anomalies = {}
        
    def _load_graph_data(self):
        """加载完整的图数据"""
        with open(self.graph_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _process_edges(self):
        """处理边信息，构建邻接表"""
        edges = defaultdict(list)
        for edge in self.graph_data['edges']:
            source_id = edge['source']
            target_id = edge['target']
            edge_type = f"{self.nodes[source_id]['type']}_{self.nodes[target_id]['type']}"
            edges[source_id].append({
                'target_id': target_id,
                'edge_type': edge_type,
                'weight': edge.get('weight', 1.0)
            })
        return edges
    
    def _get_node_metrics(self, node_id, timestamp, metrics_data):
        """获取节点最近的指标数据"""
        if node_id not in metrics_data:
            return None
        
        metrics = metrics_data[node_id]
        if not metrics:
            return None
            
        # 找到最近的指标数据
        latest_metrics = None
        for m in reversed(metrics):
            if datetime.fromisoformat(m['timestamp']) <= timestamp:
                latest_metrics = m
                break
                
        return latest_metrics['values'] if latest_metrics else None
    
    def _check_and_update_anomalies(self, node_id, timestamp):
        """检查和更新异常状态"""
        # 清理过期的异常
        current_anomalies = {k: v for k, v in self.active_anomalies.items() 
                           if v['end_time'] > timestamp}
        self.active_anomalies = current_anomalies
        
        # 检查节点是否处于异常状态
        if node_id in self.active_anomalies:
            anomaly = self.active_anomalies[node_id]
            return anomaly['impact_factor']
            
        # 随机生成新的异常
        if random.random() < self.anomaly_config['probability']:
            scenario = random.choices(
                list(self.anomaly_config['scenarios'].keys()),
                weights=[s['probability'] for s in self.anomaly_config['scenarios'].values()]
            )[0]
            
            duration = random.randint(
                self.anomaly_config['duration']['min'],
                self.anomaly_config['duration']['max']
            )
            
            self.active_anomalies[node_id] = {
                'scenario': scenario,
                'start_time': timestamp,
                'end_time': timestamp + timedelta(minutes=duration),
                'impact_factor': 1 - self.anomaly_config['scenarios'][scenario]['impact']
            }
            
            return self.active_anomalies[node_id]['impact_factor']
            
        return 1.0
    
    def _calculate_impact_factor(self, node_id, timestamp, metrics_data):
        """计算资源间的影响因子"""
        node_type = self.nodes[node_id]['type']
        impact_factor = self._check_and_update_anomalies(node_id, timestamp)
        
        # 获取影响当前节点的上游节点
        for edge in self.edges.get(node_id, []):
            source_metrics = self._get_node_metrics(edge['target_id'], timestamp, metrics_data)
            if not source_metrics:
                continue
                
            edge_type = edge['edge_type']
            if edge_type in self.impact_config:
                # 根据不同类型的资源关系计算影响
                if edge_type == 'TRU_HOST':
                    if source_metrics.get('storage_usage', 0) > 90:
                        impact_factor *= (1 - self.impact_config[edge_type])
                elif edge_type == 'HOST_VM':
                    host_impact = 1.0
                    if source_metrics.get('cpu_usage', 0) > 80:
                        host_impact *= (1 - self.impact_config[edge_type] * 0.5)
                    if source_metrics.get('mem_usage', 0) > 85:
                        host_impact *= (1 - self.impact_config[edge_type] * 0.5)
                    impact_factor *= host_impact
                elif edge_type == 'VM_NE':
                    vm_impact = 1.0
                    if source_metrics.get('vcpu_usage', 0) > 85:
                        vm_impact *= (1 - self.impact_config[edge_type] * 0.6)
                    if source_metrics.get('vmem_usage', 0) > 90:
                        vm_impact *= (1 - self.impact_config[edge_type] * 0.4)
                    impact_factor *= vm_impact
                elif edge_type == 'HA_HOST':
                    if source_metrics.get('avg_cpu_usage', 0) > 75:
                        impact_factor *= (1 - self.impact_config[edge_type])
        
        return max(0.1, min(impact_factor, 1.0))  # 限制影响因子在0.1到1.0之间

    def _generate_metric_value(self, base, std, min_val, max_val, impact_factor=1.0):
        """生成带有随机波动的指标值"""
        value = np.random.normal(base, std)
        value = value * impact_factor  # 应用影响因子
        return max(min_val, min(value, max_val))

    def _generate_node_metrics(self, node):
        """为单个节点生成性能指标
        
        Args:
            node: 节点数据
            
        Returns:
            tuple: (node_id, metrics_list)
        """
        node_type = node['type']
        node_id = node['id']
        
        if node_type not in ['NE', 'VM', 'HOST', 'HA', 'TRU']:
            return node_id, []
            
        # 获取当前时间作为结束时间
        end_time = datetime.now()
        # 设置开始时间为一周前
        start_time = end_time - timedelta(hours=168)
        # 设置采样间隔为60分钟
        interval = timedelta(minutes=60)
        
        # 初始化指标列表
        metrics_list = []
        current_time = start_time
        
        while current_time <= end_time:
            # 根据节点类型生成不同的指标
            values = {}
            
            if node_type == 'NE':
                values = {
                    'connection_count': np.random.randint(50, 3000),
                    'session_count': np.random.randint(20, 1500),
                    'session_success_rate': np.random.uniform(85, 100),
                    'connection_success_rate': np.random.uniform(90, 100)
                }
            elif node_type == 'VM':
                values = {
                    'vcpu_usage': np.random.uniform(10, 100),
                    'vmem_usage': np.random.uniform(20, 95),
                    'disk_io': np.random.uniform(100, 10000),
                    'network_io': np.random.uniform(50, 1000)
                }
            elif node_type == 'HOST':
                values = {
                    'cpu_usage': np.random.uniform(20, 100),
                    'mem_usage': np.random.uniform(30, 98),
                    'disk_io': np.random.uniform(1000, 30000),
                    'network_io': np.random.uniform(500, 3000)
                }
            elif node_type == 'HA':
                values = {
                    'avg_cpu_usage': np.random.uniform(10, 90),
                    'avg_mem_usage': np.random.uniform(20, 95)
                }
            elif node_type == 'TRU':
                values = {
                    'storage_usage': np.random.uniform(30, 95)
                }
            
            # 添加时间戳和指标值
            metrics_list.append({
                'timestamp': current_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'values': values
            })
            
            # 更新时间
            current_time += interval
        
        return node_id, metrics_list
    
    def generate_metrics(self):
        """并行生成所有节点的性能指标"""
        metrics_data = {}
        nodes = [n for n in self.graph_data['nodes'] if n['type'] in ['NE', 'VM', 'HOST', 'HA', 'TRU']]
        
        print(f"开始生成 {len(nodes)} 个节点的性能指标...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_node = {
                executor.submit(self._generate_node_metrics, node): node 
                for node in nodes
            }
            
            # 使用tqdm创建进度条
            with tqdm(total=len(nodes), desc="生成性能指标") as pbar:
                # 处理完成的任务
                for future in as_completed(future_to_node):
                    node_id, metrics_list = future.result()
                    if metrics_list:  # 只保存有指标数据的节点
                        metrics_data[node_id] = metrics_list
                    pbar.update(1)
        
        # 保存指标数据
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)
        
        return metrics_data

    def save_metrics(self, metrics_data):
        """保存性能指标数据到文件"""
        output_data = {
            'metrics_data': dict(metrics_data),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'sample_interval': str(self.sample_interval),
            'metrics_config': self.metrics_config
        }
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)