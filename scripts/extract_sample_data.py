#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-15 11:19
# @Desc   : 从Neo4j数据库中提取样例数据，保持完整的业务链和资源链结构    
# --------------------------------------------------------
"""

"""
从Neo4j数据库中提取样例数据，保持完整的业务链和资源链结构
业务链：DC -> TENANT -> NE -> VM
资源链：VM -> HOST -> HA -> TRU
"""

import json
from neo4j import GraphDatabase
from typing import Dict, List
import os
from datetime import datetime
from rag.utils.config import get_database_config

db_config = get_database_config()

class SampleDataExtractor:
    def __init__(self, uri: str = db_config.get('uri'), 
                 user: str = db_config.get('user'), 
                 password: str = db_config.get('password'),
                 sample_size: int = 5):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.sample_size = sample_size
        
    def close(self):
        self.driver.close()
        
    def extract_complete_chains(self) -> Dict:
        """提取完整的业务链和资源链数据"""
        with self.driver.session() as session:
            # 1. 首先找到具有完整链路的DC节点
            result = session.run("""
                MATCH path=(dc:DC)-[:DC_TO_TENANT]->(tenant:TENANT)
                    -[:TENANT_TO_NE]->(ne:NE)
                    -[:NE_TO_VM]->(vm:VM)
                    -[:VM_TO_HOST]->(host:HOST)
                    -[:HOST_TO_HA]->(ha:HA)
                    -[:HA_TO_TRU]->(tru:TRU)
                WITH dc, count(path) as chain_count
                WHERE chain_count > 0
                RETURN dc.id as dc_id
                LIMIT $sample_size
            """, sample_size=self.sample_size)
            
            dc_ids = [record["dc_id"] for record in result]
            
            if not dc_ids:
                raise Exception("未找到完整的链路数据")
            
            # 2. 对每个DC提取完整的链路数据
            complete_data = {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "sample_size": self.sample_size,
                    "chains": {
                        "business_chain": "DC -> TENANT -> NE -> VM",
                        "resource_chain": "VM -> HOST -> HA -> TRU"
                    }
                }
            }
            
            node_ids = set()  # 用于去重
            edge_pairs = set()  # 用于去重
            
            for dc_id in dc_ids:
                # 提取完整链路的所有节点和边
                result = session.run("""
                    MATCH path=(dc:DC {id: $dc_id})-[:DC_TO_TENANT]->(tenant:TENANT)
                        -[:TENANT_TO_NE]->(ne:NE)
                        -[:NE_TO_VM]->(vm:VM)
                        -[:VM_TO_HOST]->(host:HOST)
                        -[:HOST_TO_HA]->(ha:HA)
                        -[:HA_TO_TRU]->(tru:TRU)
                    UNWIND nodes(path) as node
                    WITH path, collect(node) as nodes
                    UNWIND relationships(path) as rel
                    RETURN nodes, collect(rel) as rels
                """, dc_id=dc_id)
                
                for record in result:
                    # 处理节点
                    for node in record["nodes"]:
                        if node.id not in node_ids:
                            node_data = dict(node.items())  # 获取所有属性
                            node_data["id"] = node.id
                            node_data["labels"] = list(node.labels)
                            complete_data["nodes"].append(node_data)
                            node_ids.add(node.id)
                    
                    # 处理边
                    for rel in record["rels"]:
                        edge_key = (rel.start_node.id, rel.type, rel.end_node.id)
                        if edge_key not in edge_pairs:
                            edge_data = dict(rel.items())  # 获取所有属性
                            edge_data.update({
                                "source": rel.start_node.id,
                                "target": rel.end_node.id,
                                "type": rel.type
                            })
                            complete_data["edges"].append(edge_data)
                            edge_pairs.add(edge_key)
            
            # 3. 添加统计信息
            complete_data["metadata"]["statistics"] = {
                "total_nodes": len(complete_data["nodes"]),
                "total_edges": len(complete_data["edges"]),
                "node_types": self._count_node_types(complete_data["nodes"]),
                "edge_types": self._count_edge_types(complete_data["edges"])
            }
            
            return complete_data
    
    def _count_node_types(self, nodes: List[Dict]) -> Dict:
        """统计节点类型数量"""
        counts = {}
        for node in nodes:
            for label in node["labels"]:
                counts[label] = counts.get(label, 0) + 1
        return counts
    
    def _count_edge_types(self, edges: List[Dict]) -> Dict:
        """统计边类型数量"""
        counts = {}
        for edge in edges:
            edge_type = edge["type"]
            counts[edge_type] = counts.get(edge_type, 0) + 1
        return counts

def main():
    # 创建输出目录
    output_dir = "datasets/samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取样例数据
    extractor = SampleDataExtractor(sample_size=5)
    try:
        print("开始提取样例数据...")
        sample_data = extractor.extract_complete_chains()
        
        # 保存样例数据
        output_path = os.path.join(output_dir, "sample_graph_data.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n样例数据提取完成！")
        print(f"总节点数: {sample_data['metadata']['statistics']['total_nodes']}")
        print(f"总边数: {sample_data['metadata']['statistics']['total_edges']}")
        print("\n节点类型统计:")
        for node_type, count in sample_data['metadata']['statistics']['node_types'].items():
            print(f"  {node_type}: {count}")
        print("\n边类型统计:")
        for edge_type, count in sample_data['metadata']['statistics']['edge_types'].items():
            print(f"  {edge_type}: {count}")
        print(f"\n数据已保存至: {output_path}")
        
    finally:
        extractor.close()

if __name__ == "__main__":
    main() 