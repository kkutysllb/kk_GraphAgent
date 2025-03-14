#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-03-11 16:30
# @Desc   : 基于性能指标的日志状态生成器
# --------------------------------------------------------
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class LogGenerator:
    def __init__(self, graph_data_path, max_workers=8):
        """初始化日志生成器
        
        Args:
            graph_data_path: 包含性能指标的图数据文件路径
            max_workers: 最大线程数
        """
        self.graph_data_path = graph_data_path
        self.max_workers = max_workers
        self.graph_data = self._load_data()
        
        # 定义不同节点类型的性能指标阈值
        self.thresholds = {
            'NE': {
                'connection_count': {
                    'warning': {'min': 100, 'max': 2500},
                    'error': {'min': 50, 'max': 2800}
                },
                'session_count': {
                    'warning': {'min': 50, 'max': 1200},
                    'error': {'min': 20, 'max': 1400}
                },
                'session_success_rate': {
                    'warning': {'min': 95, 'max': None},
                    'error': {'min': 90, 'max': None}
                },
                'connection_success_rate': {
                    'warning': {'min': 96, 'max': None},
                    'error': {'min': 92, 'max': None}
                }
            },
            'VM': {
                'vcpu_usage': {
                    'warning': {'min': None, 'max': 80},
                    'error': {'min': None, 'max': 95}
                },
                'vmem_usage': {
                    'warning': {'min': None, 'max': 85},
                    'error': {'min': None, 'max': 90}
                },
                'disk_io': {
                    'warning': {'min': None, 'max': 8000},
                    'error': {'min': None, 'max': 9000}
                },
                'network_io': {
                    'warning': {'min': None, 'max': 800},
                    'error': {'min': None, 'max': 900}
                }
            },
            'HOST': {
                'cpu_usage': {
                    'warning': {'min': None, 'max': 85},
                    'error': {'min': None, 'max': 95}
                },
                'mem_usage': {
                    'warning': {'min': None, 'max': 90},
                    'error': {'min': None, 'max': 95}
                },
                'disk_io': {
                    'warning': {'min': None, 'max': 25000},
                    'error': {'min': None, 'max': 28000}
                },
                'network_io': {
                    'warning': {'min': None, 'max': 2500},
                    'error': {'min': None, 'max': 2800}
                }
            },
            'HA': {
                'avg_cpu_usage': {
                    'warning': {'min': None, 'max': 75},
                    'error': {'min': None, 'max': 85}
                },
                'avg_mem_usage': {
                    'warning': {'min': None, 'max': 80},
                    'error': {'min': None, 'max': 90}
                }
            },
            'TRU': {
                'storage_usage': {
                    'warning': {'min': None, 'max': 85},
                    'error': {'min': None, 'max': 90}
                }
            }
        }
        
        # 定义日志字段映射
        self.log_fields = {
            'NE': {
                'connection_count': 'connection_log_status',
                'session_count': 'session_log_status',
                'session_success_rate': 'session_success_log_status',
                'connection_success_rate': 'connection_success_log_status'
            },
            'VM': {
                'vcpu_usage': 'cpu_log_status',
                'vmem_usage': 'memory_log_status',
                'disk_io': 'disk_log_status',
                'network_io': 'network_log_status'
            },
            'HOST': {
                'cpu_usage': 'cpu_log_status',
                'mem_usage': 'memory_log_status',
                'disk_io': 'disk_log_status',
                'network_io': 'network_log_status'
            },
            'HA': {
                'avg_cpu_usage': 'cpu_log_status',
                'avg_mem_usage': 'memory_log_status'
            },
            'TRU': {
                'storage_usage': 'storage_log_status'
            }
        }
    
    def _load_data(self):
        """加载图数据"""
        with open(self.graph_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _check_metric_status(self, value, thresholds):
        """检查单个指标的状态
        
        Args:
            value: 指标值
            thresholds: 该指标的阈值配置
            
        Returns:
            int: 状态码 (0: 正常, 1: warning, 2: error, 3: warning+error)
        """
        status = 0
        
        # 检查 warning 阈值
        warning = thresholds['warning']
        if (warning['min'] is not None and value < warning['min']) or \
           (warning['max'] is not None and value > warning['max']):
            status |= 1
        
        # 检查 error 阈值
        error = thresholds['error']
        if (error['min'] is not None and value < error['min']) or \
           (error['max'] is not None and value > error['max']):
            status |= 2
        
        return status
    
    def _generate_node_logs(self, node):
        """为单个节点生成日志状态
        
        Args:
            node: 节点数据
            
        Returns:
            tuple: (node_id, log_data)
        """
        node_type = node['type']
        node_id = node['id']
        
        if node_type not in self.thresholds:
            return node_id, None
            
        # 获取节点的性能指标数据
        metrics = node.get('metrics', {}).get('metrics_data', {}).get(node_id, [])
        if not metrics:
            return node_id, None
        
        # 生成日志数据
        log_data = []
        for metric in metrics:
            timestamp = metric['timestamp']
            values = metric['values']
            
            # 检查每个指标的状态
            status = {}
            for metric_name, value in values.items():
                if metric_name in self.thresholds[node_type]:
                    metric_status = self._check_metric_status(
                        value,
                        self.thresholds[node_type][metric_name]
                    )
                    log_field = self.log_fields[node_type][metric_name]
                    status[log_field] = metric_status
            
            # 添加日志记录
            log_data.append({
                'timestamp': timestamp,
                'status': status
            })
        
        return node_id, log_data
    
    def generate_logs(self):
        """并行生成所有节点的日志状态"""
        nodes = [n for n in self.graph_data['nodes'] if n['type'] in self.thresholds]
        
        print(f"开始生成 {len(nodes)} 个节点的日志状态...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_node = {
                executor.submit(self._generate_node_logs, node): node 
                for node in nodes
            }
            
            # 使用tqdm创建进度条
            with tqdm(total=len(nodes), desc="生成日志状态") as pbar:
                # 处理完成的任务
                for future in as_completed(future_to_node):
                    node_id, log_data = future.result()
                    if log_data:
                        # 更新节点的日志数据
                        for node in self.graph_data['nodes']:
                            if node['id'] == node_id:
                                node['logs'] = {
                                    'log_data': {
                                        node_id: log_data
                                    }
                                }
                                break
                    pbar.update(1)
        
        return self.graph_data
    
    def save_data(self, output_path):
        """保存更新后的图数据
        
        Args:
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.graph_data, f, ensure_ascii=False, indent=2) 