#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-03-11 16:30
# @Desc   : 资源拓扑动态变化生成器
# --------------------------------------------------------
"""
import json
import random
from datetime import datetime, timedelta
from collections import defaultdict

class TopologyDynamicsGenerator:
    def __init__(self, graph_data_path):
        """初始化拓扑动态变化生成器
        
        Args:
            graph_data_path: 图数据文件路径
        """
        self.graph_data = self._load_data(graph_data_path)
        self.ha_groups = self._initialize_ha_groups()
        self.host_status = self._initialize_host_status()
        self.tru_status = self._initialize_tru_status()
        self.vm_locations = self._initialize_vm_locations()
        
    def _load_data(self, graph_data_path):
        """加载图数据"""
        with open(graph_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _initialize_ha_groups(self):
        """初始化HA组信息"""
        ha_groups = defaultdict(list)
        
        # 遍历所有节点找到HA节点
        ha_nodes = [node for node in self.graph_data['nodes'] if node['type'] == 'HA']
        
        # 遍历所有边找到HA和HOST的关系
        for edge in self.graph_data['edges']:
            source_node = self._get_node_by_id(edge['source'])
            target_node = self._get_node_by_id(edge['target'])
            
            if source_node and target_node:
                # 检查HOST_TO_HA关系
                if source_node['type'] == 'HOST' and target_node['type'] == 'HA':
                    ha_groups[target_node['id']].append(source_node['id'])
                # 检查HA_TO_HOST关系
                elif source_node['type'] == 'HA' and target_node['type'] == 'HOST':
                    ha_groups[source_node['id']].append(target_node['id'])
        
        # 如果没有找到HA组，创建默认的HA组
        if not ha_groups:
            print("警告：未找到HA组信息，创建默认HA组...")
            # 获取所有主机
            hosts = [node['id'] for node in self.graph_data['nodes'] if node['type'] == 'HOST']
            # 将主机平均分配到6个默认HA组
            chunk_size = len(hosts) // 6 + (1 if len(hosts) % 6 else 0)
            for i in range(6):
                ha_id = f"DEFAULT_HA_GROUP_{i+1}"
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(hosts))
                ha_groups[ha_id] = hosts[start_idx:end_idx]
        
        return dict(ha_groups)
    
    def _initialize_host_status(self):
        """初始化主机状态"""
        host_status = {}
        for node in self.graph_data['nodes']:
            if node['type'] == 'HOST':
                host_status[node['id']] = {
                    'status': 'active',
                    'last_failure': None,
                    'failure_count': 0
                }
        return host_status
    
    def _initialize_tru_status(self):
        """初始化存储池状态"""
        tru_status = {}
        for node in self.graph_data['nodes']:
            if node['type'] == 'TRU':
                tru_status[node['id']] = {
                    'status': 'active',
                    'last_failure': None,
                    'failure_count': 0
                }
        return tru_status
    
    def _initialize_vm_locations(self):
        """初始化VM位置信息"""
        vm_locations = {}
        for edge in self.graph_data['edges']:
            source_node = self._get_node_by_id(edge['source'])
            target_node = self._get_node_by_id(edge['target'])
            
            if source_node and target_node:
                if source_node['type'] == 'HOST' and target_node['type'] == 'VM':
                    vm_locations[target_node['id']] = {
                        'host': source_node['id'],
                        'ha_group': self._get_ha_group_for_host(source_node['id'])
                    }
        return vm_locations
    
    def _get_node_by_id(self, node_id):
        """通过ID获取节点"""
        return next((node for node in self.graph_data['nodes'] if node['id'] == node_id), None)
    
    def _get_ha_group_for_host(self, host_id):
        """获取主机所属的HA组"""
        for ha_id, hosts in self.ha_groups.items():
            if host_id in hosts:
                return ha_id
        return None
    
    def _get_host_vms(self, host_id):
        """获取主机上的所有VM"""
        return [vm_id for vm_id, info in self.vm_locations.items() if info['host'] == host_id]
    
    def _get_available_hosts(self, ha_group_id):
        """获取HA组内可用的主机"""
        if ha_group_id not in self.ha_groups:
            return []
        return [host_id for host_id in self.ha_groups[ha_group_id]
                if self.host_status[host_id]['status'] == 'active']
    
    def _get_tru_dependent_hosts(self, tru_id):
        """获取依赖存储池的主机"""
        dependent_hosts = []
        for edge in self.graph_data['edges']:
            if edge['source'] == tru_id and self._get_node_by_id(edge['target'])['type'] == 'HOST':
                dependent_hosts.append(edge['target'])
        return dependent_hosts
    
    def _create_topology_change(self, change_type, reason, changes, timestamp):
        """创建拓扑变化记录"""
        return {
            'timestamp': timestamp,
            'topology_change': {
                'change_type': change_type,
                'reason': reason,
                'related_changes': changes
            }
        }
    
    def _create_vm_migration(self, vm_id, source_host, target_host, migration_type='live_migration', scheduled=False):
        """创建VM迁移记录"""
        return {
            'edge_id': f"{source_host}->{vm_id}",
            'status': 'inactive',
            'new_edge': f"{target_host}->{vm_id}",
            'migration_type': migration_type,
            'scheduled': scheduled
        }
    
    def simulate_host_failure(self, host_id, timestamp):
        """模拟主机故障
        
        Args:
            host_id: 发生故障的主机ID
            timestamp: 故障发生时间
        """
        if self.host_status[host_id]['status'] != 'active':
            return None
        
        ha_group_id = self._get_ha_group_for_host(host_id)
        available_hosts = self._get_available_hosts(ha_group_id)
        if host_id in available_hosts:
            available_hosts.remove(host_id)
        
        if not available_hosts:
            return None  # 没有可用主机进行迁移
        
        affected_vms = self._get_host_vms(host_id)
        changes = []
        
        # 更新主机状态
        self.host_status[host_id].update({
            'status': 'failed',
            'last_failure': timestamp,
            'failure_count': self.host_status[host_id]['failure_count'] + 1
        })
        
        # 为每个受影响的VM创建迁移记录
        for vm_id in affected_vms:
            target_host = random.choice(available_hosts)
            migration_type = 'live_migration' if random.random() > 0.3 else 'restart'
            
            changes.append(self._create_vm_migration(
                vm_id, host_id, target_host,
                migration_type=migration_type,
                scheduled=False
            ))
            
            # 更新VM位置
            self.vm_locations[vm_id]['host'] = target_host
        
        return self._create_topology_change('failure', 'host_failure', changes, timestamp)
    
    def simulate_tru_failure(self, tru_id, timestamp):
        """模拟存储池故障
        
        Args:
            tru_id: 发生故障的存储池ID
            timestamp: 故障发生时间
        """
        if self.tru_status[tru_id]['status'] != 'active':
            return None
        
        affected_hosts = self._get_tru_dependent_hosts(tru_id)
        changes = []
        
        # 更新存储池状态
        self.tru_status[tru_id].update({
            'status': 'failed',
            'last_failure': timestamp,
            'failure_count': self.tru_status[tru_id]['failure_count'] + 1
        })
        
        # 记录存储池到主机的边变化
        for host_id in affected_hosts:
            changes.append({
                'edge_id': f"{tru_id}->{host_id}",
                'status': 'inactive'
            })
            
            # 如果主机严重依赖该存储池，触发主机故障
            if random.random() < 0.5:  # 50%的概率主机会受到严重影响
                host_changes = self.simulate_host_failure(host_id, timestamp)
                if host_changes:
                    changes.extend(host_changes['topology_change']['related_changes'])
        
        return self._create_topology_change('failure', 'tru_failure', changes, timestamp)
    
    def simulate_load_balancing(self, ha_group_id, timestamp):
        """模拟负载均衡
        
        Args:
            ha_group_id: 进行负载均衡的HA组ID
            timestamp: 操作时间
        """
        if ha_group_id not in self.ha_groups:
            return None
        
        available_hosts = self._get_available_hosts(ha_group_id)
        if len(available_hosts) < 2:
            return None  # 需要至少两个可用主机才能进行负载均衡
        
        changes = []
        # 随机选择一个源主机和目标主机
        source_host = random.choice(available_hosts)
        available_hosts.remove(source_host)
        target_host = random.choice(available_hosts)
        
        # 获取源主机上的VM
        vms = self._get_host_vms(source_host)
        if not vms:
            return None
        
        # 随机选择一些VM进行迁移
        vms_to_migrate = random.sample(vms, min(len(vms), random.randint(1, 3)))
        
        for vm_id in vms_to_migrate:
            changes.append(self._create_vm_migration(
                vm_id, source_host, target_host,
                migration_type='live_migration',
                scheduled=True
            ))
            
            # 更新VM位置
            self.vm_locations[vm_id]['host'] = target_host
        
        return self._create_topology_change('maintenance', 'load_balancing', changes, timestamp)
    
    def simulate_expansion(self, ha_group_id, num_hosts, timestamp):
        """模拟扩容
        
        Args:
            ha_group_id: 扩容的HA组ID
            num_hosts: 新增主机数量
            timestamp: 操作时间
        """
        if ha_group_id not in self.ha_groups:
            return None
        
        changes = []
        new_hosts = []
        
        # 创建新主机
        for i in range(num_hosts):
            host_id = f"HOST_{len(self.host_status) + 1:04d}"
            self.host_status[host_id] = {
                'status': 'active',
                'last_failure': None,
                'failure_count': 0
            }
            self.ha_groups[ha_group_id].append(host_id)
            new_hosts.append(host_id)
            
            # 为新主机分配存储池
            available_trus = [tru_id for tru_id, status in self.tru_status.items()
                            if status['status'] == 'active']
            if available_trus:
                tru_id = random.choice(available_trus)
                changes.append({
                    'edge_id': f"{tru_id}->{host_id}",
                    'status': 'active',
                    'type': 'new'
                })
        
        return self._create_topology_change('expansion', 'planned_expansion', changes, timestamp)
    
    def generate_topology_dynamics(self, start_time, end_time, interval_minutes=60):
        """生成一段时间内的拓扑动态变化数据
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            interval_minutes: 时间间隔（分钟）
        
        Returns:
            list: 拓扑动态变化数据列表
        """
        current_time = start_time
        dynamics_data = []
        
        while current_time <= end_time:
            # 随机决定是否发生变化
            if random.random() < 0.3:  # 30%的概率发生变化
                change_type = random.random()
                
                if change_type < 0.4:  # 40%的概率是主机故障
                    active_hosts = [host_id for host_id, status in self.host_status.items()
                                  if status['status'] == 'active']
                    if active_hosts:
                        host_id = random.choice(active_hosts)
                        change = self.simulate_host_failure(host_id, current_time.isoformat())
                        if change:
                            dynamics_data.append(change)
                
                elif change_type < 0.6:  # 20%的概率是存储池故障
                    active_trus = [tru_id for tru_id, status in self.tru_status.items()
                                 if status['status'] == 'active']
                    if active_trus:
                        tru_id = random.choice(active_trus)
                        change = self.simulate_tru_failure(tru_id, current_time.isoformat())
                        if change:
                            dynamics_data.append(change)
                
                elif change_type < 0.9:  # 30%的概率是负载均衡
                    ha_group_id = random.choice(list(self.ha_groups.keys()))
                    change = self.simulate_load_balancing(ha_group_id, current_time.isoformat())
                    if change:
                        dynamics_data.append(change)
                
                else:  # 10%的概率是扩容
                    ha_group_id = random.choice(list(self.ha_groups.keys()))
                    change = self.simulate_expansion(ha_group_id, random.randint(1, 2), current_time.isoformat())
                    if change:
                        dynamics_data.append(change)
            
            # 恢复一些故障的组件
            self._recover_failed_components(current_time.isoformat())
            
            current_time += timedelta(minutes=interval_minutes)
        
        return dynamics_data
    
    def _recover_failed_components(self, timestamp):
        """恢复故障的组件"""
        # 恢复主机
        for host_id, status in self.host_status.items():
            if status['status'] == 'failed' and random.random() < 0.4:  # 40%的概率恢复
                status['status'] = 'active'
        
        # 恢复存储池
        for tru_id, status in self.tru_status.items():
            if status['status'] == 'failed' and random.random() < 0.3:  # 30%的概率恢复
                status['status'] = 'active' 