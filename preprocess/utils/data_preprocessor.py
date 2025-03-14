#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-02-23 16:18
# @Desc   : 原始数据预处理
# --------------------------------------------------------
"""
import pandas as pd
import hashlib
import json
import os
from graph_rag.utils.logger import Logger
from datetime import datetime
from collections import defaultdict


class ResourceNode:
    def __init__(self, id: str, layer: int, capacity: float, type: str, metadata: dict = None):
        self.id = id
        self.layer = layer
        self.capacity = capacity 
        self.type = type
        self.metadata = metadata or {}

class ResourcePreprocessor:
    def __init__(self, config):
        self.config = config
        self.dc_id = self._generate_dc_id()
        self.logger = Logger('ResourcePreprocessor')
        
        # 定义层级映射
        self.layer_mapping = {
            'DC': 0,           # 数据中心层
            'TENANT': 1,       # 租户层
            'NE': 2,          # 网元层
            'VM': 3,          # 虚拟机层
            'HOST': 4,        # 主机层
            'HA': 5,          # HA层
            'TRU': 6,         # 存储层
        }
        
        # 定义资源类型映射
        self.resource_type_mapping = {
            'DC': 'BUSINESS',      # 业务类型
            'TENANT': 'BUSINESS',
            'NE': 'BUSINESS',
            'VM': 'COMPUTE',       # 计算类型
            'HOST': 'COMPUTE',
            'HA': 'COMPUTE',       # HA类型
            'TRU': 'STORAGE',      # 存储类型
        }
        
        # 定义默认容量映射
        self.capacity_mapping = {
            'DC': 1000.0,
            'TENANT': 100.0,
            'NE': 50.0,
            'VM': 10.0,
            'HOST': 20.0,
            'HA': 15.0,
            'TRU': 30.0
        }
        
    def _generate_dc_id(self):
        """生成全局唯一的DC ID"""
        return "DC_" + hashlib.md5("resource_pool".encode()).hexdigest()[:8]
        
    def _generate_id(self, value, prefix):
        """生成唯一ID"""
        if pd.isna(value):
            return None
        
        # 清理和标准化输入值
        value_str = str(value).strip()
        
        # 如果已经是正确格式且未使用过，直接返回
        if value_str.startswith(f"{prefix}_") and value_str not in self.used_ids:
            self.used_ids.add(value_str)
            return value_str
        
        # 生成新ID
        counter = 1
        while True:
            new_id = f"{prefix}_{counter:04d}"
            if new_id not in self.used_ids:
                self.used_ids.add(new_id)
                return new_id
            counter += 1
        
    def validate_data(self, df):
        """验证数据完整性和格式，并打印数据结构信息"""
        self.logger.info("开始详细数据分析")
        
        # 首先打印所有列名
        self.logger.info("\n实际的列名:")
        for col in df.columns:
            self.logger.info(f"  - {col}")
        
        # 检查预期的列名是否存在
        expected_columns = {
            '虚拟机ID': '虚机ID',  # 可能的映射关系
            '虚拟机名称': '虚机名称',
            '网元ID': '网元ID',
            '网元名称': '网元名称',
            '租户ID': '租户ID',
            '租户名称': '租户名称',
            '主机ID': '主机ID',
            '主机名称': '主机名称',
            'HA_ID': 'HA_ID',
            'HA': 'HA',
            'TRU_ID': 'TRU_ID',
            'TRU': 'TRU',
        }
        
        self.logger.info("\n列名映射检查:")
        for orig_col, mapped_col in expected_columns.items():
            if orig_col in df.columns:
                self.logger.info(f"  找到原始列: {orig_col}")
            elif mapped_col in df.columns:
                self.logger.info(f"  找到映射列: {mapped_col}")
            else:
                self.logger.warning(f"  未找到列: {orig_col} 或 {mapped_col}")
        
        # 打印每列的唯一值数量和示例
        for col in df.columns:
            unique_values = df[col].nunique()
            sample_values = df[col].dropna().sample(min(3, len(df[col].dropna()))).tolist()
            self.logger.info(f"\n列 {col}:")
            self.logger.info(f"  唯一值数量: {unique_values}")
            self.logger.info(f"  示例值: {sample_values}")
            self.logger.info(f"  数据类型: {df[col].dtype}")
            self.logger.info(f"  空值数量: {df[col].isnull().sum()}")
        
        # 根据实际列名更新列名映射
        self.column_mapping = {}
        for orig_col, mapped_col in expected_columns.items():
            if orig_col in df.columns:
                self.column_mapping[orig_col] = mapped_col
        
        # 应用列名映射
        if self.column_mapping:
            df.rename(columns=self.column_mapping, inplace=True)
            self.logger.info("\n应用了以下列名映射:")
            for orig, mapped in self.column_mapping.items():
                self.logger.info(f"  {orig} -> {mapped}")
        
        # 分析ID格式
        id_columns = ['网元ID', '虚机ID', '主机ID', 'HA_ID', 'TRU_ID']
        for col in id_columns:
            if col in df.columns:
                patterns = df[col].dropna().apply(lambda x: self._identify_id_pattern(x)).value_counts()
                self.logger.info(f"\n{col} ID模式分析:")
                for pattern, count in patterns.items():
                    self.logger.info(f"  {pattern}: {count}个")
        
        # 分析关联关系
        self._analyze_relationships(df)
        
    def _identify_id_pattern(self, id_str):
        """识别ID的模式"""
        if isinstance(id_str, str):
            if '-' in id_str and len(id_str) == 36:
                return 'UUID'
            elif id_str.startswith(('DC_', 'TENANT_', 'NE_', 'VM_', 'HOST_', 'HA_', 'TRU_')):
                return f'PREFIX_{id_str.split("_")[0]}'
            else:
                return 'OTHER'
        return 'NON_STRING'

    def _analyze_relationships(self, df):
        """分析节点间的关联关系"""
        self.logger.info("\n节点关联关系分析:")
        
        # 分析每种关系的数量
        relationships = [
            ('租户ID', '网元ID', 'TENANT->NE'),
            ('网元ID', '虚机ID', 'NE->VM'),
            ('网元ID', 'HA_ID', 'NE->HA'),
            ('虚机ID', '主机ID', 'VM->HOST'),
            ('HA_ID', 'TRU_ID', 'HA->TRU')
        ]
        
        for source, target, rel_name in relationships:
            if source in df.columns and target in df.columns:
                valid_pairs = df[[source, target]].dropna()
                unique_pairs = valid_pairs.drop_duplicates()
                self.logger.info(f"\n{rel_name}关系:")
                self.logger.info(f"  总关联数: {len(valid_pairs)}")
                self.logger.info(f"  唯一关联数: {len(unique_pairs)}")
                self.logger.info(f"  源节点数: {valid_pairs[source].nunique()}")
                self.logger.info(f"  目标节点数: {valid_pairs[target].nunique()}")
        
    def preprocess(self, excel_path):
        """预处理Excel数据"""
        try:
            self.logger.info(f"开始处理文件: {excel_path}")
            
            # 检查文件是否存在
            if not os.path.exists(excel_path):
                raise FileNotFoundError(f"找不到文件: {excel_path}")
            
            # 读取Excel
            df = pd.read_excel(excel_path)
            self.logger.info(f"成功读取数据，共{len(df)}行")
            
            # 验证数据
            self.logger.info("开始数据验证")
            self.validate_data(df)
            
            # 添加DC字段
            df['DC_ID'] = self.dc_id
            
            # 生成ID字段
            self.logger.info("开始生成ID字段")
            self._generate_ids(df)
            
            # 转换为图结构
            self.logger.info("开始转换为图结构")
            graph_data = self._convert_to_graph_json(df)
            
            self.logger.info(f"数据处理完成，生成{len(graph_data['nodes'])}个节点和{len(graph_data['edges'])}条边")
            
            return graph_data
            
        except Exception as e:
            self.logger.error(f"数据处理失败: {str(e)}")
            raise
            
    def _generate_ids(self, df):
        """生成所有需要的ID字段"""
        self.logger.debug("开始生成ID字段")
        
        try:
            # 用于跟踪已生成的ID，确保唯一性
            self.used_ids = set()
            self.id_mappings = {}  # 存储原始ID到新ID的映射
            
            # 1. 生成DC_ID
            df['DC_ID'] = self.dc_id
            self.used_ids.add(self.dc_id)
            
            # 2. 处理租户数据
            tenant_df = df[['租户名称']].drop_duplicates()
            for i, row in tenant_df.iterrows():
                new_id = f'TENANT_{i+1:04d}'
                self.id_mappings[row['租户名称']] = new_id
                self.used_ids.add(new_id)
            df['租户ID'] = df['租户名称'].map(self.id_mappings)
            
            # 3. 处理网元数据
            ne_df = df[['网元ID', '网元名称']].drop_duplicates()
            for i, row in ne_df.iterrows():
                new_id = f'NE_{i+1:04d}'
                self.id_mappings[row['网元ID']] = new_id
                self.used_ids.add(new_id)
            df['网元ID'] = df['网元ID'].map(self.id_mappings)
            
            # 4. 处理虚拟机数据
            vm_df = df[['虚机ID', '虚机名称']].drop_duplicates()
            for i, row in vm_df.iterrows():
                new_id = f'VM_{i+1:04d}'
                self.id_mappings[row['虚机ID']] = new_id
                self.used_ids.add(new_id)
            df['虚机ID'] = df['虚机ID'].map(self.id_mappings)
            
            # 5. 处理主机数据
            host_df = df[['主机ID', '主机名称']].drop_duplicates()
            for i, row in host_df.iterrows():
                new_id = f'HOST_{i+1:04d}'
                self.id_mappings[str(row['主机ID'])] = new_id
                self.used_ids.add(new_id)
            df['主机ID'] = df['主机ID'].astype(str).map(self.id_mappings)
            
            # 6. 处理HA数据
            ha_df = df[['HA']].drop_duplicates()
            for i, row in ha_df.iterrows():
                new_id = f'HA_{i+1:04d}'
                self.id_mappings[row['HA']] = new_id
                self.used_ids.add(new_id)
            df['HA_ID'] = df['HA'].map(self.id_mappings)
            
            # 7. 处理TRU数据
            tru_df = df[['TRU']].drop_duplicates()
            for i, row in tru_df.iterrows():
                new_id = f'TRU_{i+1:04d}'
                self.id_mappings[row['TRU']] = new_id
                self.used_ids.add(new_id)
            df['TRU_ID'] = df['TRU'].map(self.id_mappings)
            
            # 记录映射信息
            self.logger.info("\nID映射统计:")
            for entity_type in ['租户', '网元', '虚机', '主机', 'HA', 'TRU']:
                count = len([k for k in self.id_mappings.values() if k.startswith(entity_type.upper())])
                self.logger.info(f"{entity_type}: {count} 个唯一值")
            
        except Exception as e:
            self.logger.error(f"生成ID字段时出错: {str(e)}")
            raise
        
    def _create_resource_node(self, node_id: str, node_type: str, name: str = None, **kwargs) -> dict:
        """创建资源节点
        
        Args:
            node_id: 节点ID
            node_type: 节点类型（如DC、TENANT等，表示层级类型）
            name: 节点名称
            **kwargs: 其他属性
            
        Returns:
            节点字典，包含业务类型和层级信息
        """
        # 获取业务类型（BUSINESS/COMPUTE/STORAGE）
        business_type = self.resource_type_mapping.get(node_type, 'UNKNOWN')
        # 获取层级
        layer = self.layer_mapping.get(node_type, -1)
        
        # 基本属性
        attributes = {
            'name': name or node_id,  # 保留名称信息
            'layer': layer,  # 层级数值（0-6）
            'layer_type': node_type,  # 层级类型（DC/TENANT/NE等）
            'business_type': business_type,  # 业务类型（BUSINESS/COMPUTE/STORAGE）
            'capacity': self.capacity_mapping.get(node_type, 0.0),
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加其他属性
        attributes.update(kwargs)
        
        return {
            'id': node_id,
            'type': node_type,  # 使用层级类型作为主要类型
            'attributes': attributes
        }

    def _create_nodes(self, df):
        """创建所有节点"""
        nodes = []
        node_ids = set()  # 用于跟踪已创建的节点
        node_type_mapping = {}  # 用于记录节点ID到类型的映射
        
        try:
            # 创建数据中心节点
            dc_node = self._create_resource_node(
                self.dc_id,
                'DC',
                name='XBXA_DC4'
            )
            nodes.append(dc_node)
            node_ids.add(self.dc_id)
            node_type_mapping[self.dc_id] = 'DC'
            
            # 获取实际的列名
            vm_id_col = '虚拟机ID' if '虚拟机ID' in df.columns else '虚机ID'
            vm_name_col = '虚拟机名称' if '虚拟机名称' in df.columns else '虚机名称'
            
            # 创建虚拟机节点
            if vm_id_col in df.columns and vm_name_col in df.columns:
                for _, row in df[[vm_id_col, vm_name_col]].drop_duplicates().iterrows():
                    if pd.notna(row[vm_id_col]) and row[vm_id_col] not in node_ids:
                        vm_node = self._create_resource_node(
                            row[vm_id_col],
                            'VM',
                            name=row[vm_name_col]
                        )
                        nodes.append(vm_node)
                        node_ids.add(row[vm_id_col])
                        node_type_mapping[row[vm_id_col]] = 'VM'
            
            # 创建租户节点
            for _, row in df[['租户ID', '租户名称']].drop_duplicates().iterrows():
                if pd.notna(row['租户ID']) and row['租户ID'] not in node_ids:
                    tenant_node = self._create_resource_node(
                        row['租户ID'],
                        'TENANT',
                        name=row['租户名称']
                    )
                    nodes.append(tenant_node)
                    node_ids.add(row['租户ID'])
                    node_type_mapping[row['租户ID']] = 'TENANT'
            
            # 创建网元节点 - 特别处理UUID格式的ID
            for _, row in df[['网元ID', '网元名称']].drop_duplicates().iterrows():
                if pd.notna(row['网元ID']) and row['网元ID'] not in node_ids:
                    ne_node = self._create_resource_node(
                        row['网元ID'],
                        'NE',
                        name=row['网元名称']
                    )
                    nodes.append(ne_node)
                    node_ids.add(row['网元ID'])
                    node_type_mapping[row['网元ID']] = 'NE'
            
            # 创建主机节点
            for _, row in df[['主机ID', '主机名称']].drop_duplicates().iterrows():
                if pd.notna(row['主机ID']) and str(row['主机ID']) not in node_ids:
                    host_node = self._create_resource_node(
                        str(row['主机ID']),
                        'HOST',
                        name=row['主机名称']
                    )
                    nodes.append(host_node)
                    node_ids.add(str(row['主机ID']))
                    node_type_mapping[str(row['主机ID'])] = 'HOST'
            
            # 创建HA节点
            for _, row in df[['HA_ID', 'HA']].drop_duplicates().iterrows():
                if pd.notna(row['HA_ID']) and row['HA_ID'] not in node_ids:
                    ha_node = self._create_resource_node(
                        row['HA_ID'],
                        'HA',
                        name=row['HA']
                    )
                    nodes.append(ha_node)
                    node_ids.add(row['HA_ID'])
                    node_type_mapping[row['HA_ID']] = 'HA'
            
            # 创建TRU节点
            for _, row in df[['TRU_ID', 'TRU']].drop_duplicates().iterrows():
                if pd.notna(row['TRU_ID']) and row['TRU_ID'] not in node_ids:
                    tru_node = self._create_resource_node(
                        row['TRU_ID'],
                        'TRU',
                        name=row['TRU']
                    )
                    nodes.append(tru_node)
                    node_ids.add(row['TRU_ID'])
                    node_type_mapping[row['TRU_ID']] = 'TRU'
            
            
            self.node_type_mapping = node_type_mapping  # 保存为实例变量，供边创建使用
            return nodes
            
        except Exception as e:
            self.logger.error(f"创建节点失败: {str(e)}")
            raise

    def _convert_to_graph_json(self, df):
        """转换为图JSON格式"""
        try:
            self.logger.info("开始转换为图结构...")
            nodes = []
            edges = []
            added_nodes = set()
            
            # 第一步：添加所有节点
            # 1. DC节点
            dc_node = self._create_resource_node(
                node_id=df['DC_ID'].iloc[0],
                node_type='DC',
                name='XBXA_DC4'
            )
            nodes.append(dc_node)
            added_nodes.add(dc_node['id'])
            
            # 2. 租户节点
            tenant_df = df[['租户ID', '租户名称']].drop_duplicates()
            for _, row in tenant_df.iterrows():
                if row['租户ID'] not in added_nodes:
                    tenant_node = self._create_resource_node(
                        node_id=row['租户ID'],
                        node_type='TENANT',
                        name=row['租户名称']
                    )
                    nodes.append(tenant_node)
                    added_nodes.add(row['租户ID'])
            
            # 3. 网元节点
            ne_df = df[['网元ID', '网元名称']].drop_duplicates()
            for _, row in ne_df.iterrows():
                if row['网元ID'] not in added_nodes:
                    ne_node = self._create_resource_node(
                        node_id=row['网元ID'],
                        node_type='NE',
                        name=row['网元名称']
                    )
                    nodes.append(ne_node)
                    added_nodes.add(row['网元ID'])
            
            # 4. 虚拟机节点
            vm_df = df[['虚机ID', '虚机名称']].drop_duplicates()
            for _, row in vm_df.iterrows():
                if row['虚机ID'] not in added_nodes:
                    vm_node = self._create_resource_node(
                        node_id=row['虚机ID'],
                        node_type='VM',
                        name=row['虚机名称']
                    )
                    nodes.append(vm_node)
                    added_nodes.add(row['虚机ID'])
            
            # 5. 主机节点
            host_df = df[['主机ID', '主机名称']].drop_duplicates()
            for _, row in host_df.iterrows():
                if str(row['主机ID']) not in added_nodes:
                    host_node = self._create_resource_node(
                        node_id=str(row['主机ID']),
                        node_type='HOST',
                        name=row['主机名称']
                    )
                    nodes.append(host_node)
                    added_nodes.add(str(row['主机ID']))
            
            # 6. HA节点
            ha_df = df[['HA_ID', 'HA']].drop_duplicates()
            for _, row in ha_df.iterrows():
                if row['HA_ID'] not in added_nodes:
                    ha_node = self._create_resource_node(
                        node_id=row['HA_ID'],
                        node_type='HA',
                        name=row['HA']
                    )
                    nodes.append(ha_node)
                    added_nodes.add(row['HA_ID'])
            
            # 7. TRU节点
            tru_df = df[['TRU_ID', 'TRU']].drop_duplicates()
            for _, row in tru_df.iterrows():
                if row['TRU_ID'] not in added_nodes:
                    tru_node = self._create_resource_node(
                        node_id=row['TRU_ID'],
                        node_type='TRU',
                        name=row['TRU']
                    )
                    nodes.append(tru_node)
                    added_nodes.add(row['TRU_ID'])
            
            # 第二步：添加边关系
            # 1. DC->TENANT
            dc_tenant_df = df[['DC_ID', '租户ID']].drop_duplicates()
            for _, row in dc_tenant_df.iterrows():
                edges.append(self._create_edge(
                    source=row['DC_ID'],
                    target=row['租户ID'],
                    edge_type='DC_TO_TENANT'
                ))
            
            # 2. TENANT->NE
            tenant_ne_df = df[['租户ID', '网元ID']].drop_duplicates()
            for _, row in tenant_ne_df.iterrows():
                edges.append(self._create_edge(
                    source=row['租户ID'],
                    target=row['网元ID'],
                    edge_type='TENANT_TO_NE'
                ))
            
            # 3. NE->VM
            ne_vm_df = df[['网元ID', '虚机ID']].drop_duplicates()
            for _, row in ne_vm_df.iterrows():
                edges.append(self._create_edge(
                    source=row['网元ID'],
                    target=row['虚机ID'],
                    edge_type='NE_TO_VM'
                ))
            
            # 4. VM->HOST
            vm_host_df = df[['虚机ID', '主机ID']].drop_duplicates()
            for _, row in vm_host_df.iterrows():
                edges.append(self._create_edge(
                    source=row['虚机ID'],
                    target=str(row['主机ID']),
                    edge_type='VM_TO_HOST'
                ))
            
            # 5. HOST->HA
            host_ha_df = df[['主机ID', 'HA_ID']].drop_duplicates()
            for _, row in host_ha_df.iterrows():
                edges.append(self._create_edge(
                    source=str(row['主机ID']),
                    target=row['HA_ID'],
                    edge_type='HOST_TO_HA'
                ))
            
            # 6. HA->TRU
            ha_tru_df = df[['HA_ID', 'TRU_ID']].drop_duplicates()
            for _, row in ha_tru_df.iterrows():
                edges.append(self._create_edge(
                    source=row['HA_ID'],
                    target=row['TRU_ID'],
                    edge_type='HA_TO_TRU'
                ))
            
            # 打印统计信息
            self.logger.info("\n连接模式分布:")
            edge_type_counts = defaultdict(int)
            for edge in edges:
                edge_type_counts[edge['type']] += 1
            
            for edge_type, count in edge_type_counts.items():
                self.logger.info(f"  {edge_type}: {count} 条")
            
            return {
                'nodes': nodes,
                'edges': edges
            }
            
        except Exception as e:
            self.logger.error(f"转换为图结构时出错: {str(e)}")
            raise

    def validate_graph_structure(self, graph_data):
        """验证图结构"""
        self.logger.info("开始验证图结构...")
        
        # 创建节点ID集合
        node_ids = {node['id'] for node in graph_data['nodes']}
        node_types = {node['type'] for node in graph_data['nodes']}
        
        # 检查缺失的节点类型
        expected_types = set(self.resource_type_mapping.values())
        missing_types = expected_types - node_types
        if missing_types:
            self.logger.warning(f"缺失的节点类型: {missing_types}")
        
        # 验证边的节点存在性
        for edge in graph_data['edges']:
            if edge['source'] not in node_ids:
                self.logger.warning(f"边 {edge['type']} 的源节点不存在: {edge['source']}")
            if edge['target'] not in node_ids:
                self.logger.warning(f"边 {edge['type']} 的目标节点不存在: {edge['target']}")
        
        # 统计每种资源类型的节点和相关边数量
        type_stats = defaultdict(lambda: {'nodes': 0, 'edges': 0})
        
        for node in graph_data['nodes']:
            type_stats[node['type']]['nodes'] += 1
            
        for edge in graph_data['edges']:
            source_type = edge['attributes']['source_type']
            target_type = edge['attributes']['target_type']
            if source_type:
                type_stats[source_type]['edges'] += 1
            if target_type:
                type_stats[target_type]['edges'] += 1
        
        # 打印统计信息
        self.logger.info("\n节点统计:")
        for type_name, stats in type_stats.items():
            self.logger.info(f"  {type_name:10}: {stats['nodes']:4d} 节点, {stats['edges']:4d} 条相关边")

    def _create_main_path_edges(self, df):
        """创建主路径边"""
        edges = []
        edge_set = set()
        
        # 主路径配置
        edge_configs = [
            ('DC_ID', '租户ID', 'DC_TO_TENANT'),
            ('租户ID', '网元ID', 'TENANT_TO_NE'),
            ('网元ID', '虚机ID', 'NE_TO_VM'),
            ('虚机ID', '主机ID', 'VM_TO_HOST'),
            ('主机ID', 'HA_ID', 'HOST_TO_HA'),
            ('HA_ID', 'TRU_ID', 'HA_TO_TRU')
        ]
        
        for from_col, to_col, edge_type in edge_configs:
            for _, row in df[[from_col, to_col]].drop_duplicates().iterrows():
                if pd.notna(row[from_col]) and pd.notna(row[to_col]):
                    edge = self._create_edge(
                        str(row[from_col]),
                        str(row[to_col]),
                        edge_type
                    )
                    if edge:  # 只添加成功创建的边
                        edge_key = f"{edge['source']}-{edge['target']}-{edge['type']}"
                        if edge_key not in edge_set:
                            edges.append(edge)
                            edge_set.add(edge_key)
        
        return edges

    def _create_branch_path_edges(self, df):
        """创建分支路径的边"""
        edges = []
        
        # 网元 -> HA
        edges.extend(self._create_edges_between(df, '网元ID', 'HA_ID', 'NE_TO_HA'))
        
        # HA -> TRU
        edges.extend(self._create_edges_between(df, 'HA_ID', 'TRU_ID', 'HA_TO_TRU'))
        
        return edges

    def _create_edges_between(self, df, from_col, to_col, edge_type):
        """创建两个节点之间的边"""
        edges = []
        edge_set = set()  # 添加集合来防止重复边
        
        for _, row in df[[from_col, to_col]].drop_duplicates().iterrows():
            if pd.notna(row[from_col]) and pd.notna(row[to_col]):
                edge = self._create_edge(
                    str(row[from_col]),
                    str(row[to_col]),
                    edge_type
                )
                edge_key = f"{edge['source']}-{edge['target']}-{edge['type']}"
                if edge_key not in edge_set:
                    edges.append(edge)
                    edge_set.add(edge_key)
        return edges
        
    def save_json(self, graph_data, output_path):
        """保存为JSON文件"""
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"数据已保存到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存JSON文件失败: {str(e)}")
            raise 

    def _create_edge(self, source, target, edge_type, weight=1.0):
        """创建单个边"""
        try:
            # 初始化默认值
            source_type = 'UNKNOWN'
            target_type = 'UNKNOWN'
            
            # 根据ID前缀判断节点类型
            for node_type in self.layer_mapping.keys():
                if str(source).startswith(f"{node_type}_"):
                    source_type = node_type
                if str(target).startswith(f"{node_type}_"):
                    target_type = node_type
            
            # 如果提供的边类型是UNKNOWN，根据节点类型推断
            if edge_type == 'UNKNOWN':
                edge_type = f"{source_type}_TO_{target_type}"
            
            return {
                'source': source,
                'target': target,
                'type': edge_type,
                'attributes': {
                    'weight': weight,
                    'source_type': self.resource_type_mapping.get(source_type, 'UNKNOWN'),
                    'target_type': self.resource_type_mapping.get(target_type, 'UNKNOWN'),
                    'timestamp': datetime.now().isoformat()
                }
            }
        except Exception as e:
            self.logger.error(f"创建边失败 - 源: {source}, 目标: {target}, 错误: {str(e)}")
            raise

    def _create_node_id_mapping(self, G):
        """创建节点ID到索引的映射"""
        node_id_to_idx = {}
        # 使用连续整数作为索引，忽略原始ID类型
        for idx, node in enumerate(G.nodes()):
            node_id_to_idx[node] = idx  # 直接使用枚举的索引
        return node_id_to_idx 
