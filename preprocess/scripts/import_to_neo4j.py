#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-03 15:30
# @Desc   : 将图数据导入Neo4j数据库
# --------------------------------------------------------
"""

import os
import sys
import argparse

# 添加项目根目录到系统路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from graph_rag.utils.neo4j_graph_manager import Neo4jGraphManager
from graph_rag.utils.logger import Logger


def main():
    # 创建日志记录器
    logger = Logger("ImportToNeo4j")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="将图数据导入Neo4j数据库")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="输入目录路径"
    )
    parser.add_argument(
        "--clear", 
        action="store_true", 
        help="是否清除现有数据"
    )
    parser.add_argument(
        "--uri", 
        type=str, 
        help="Neo4j数据库URI，如果不指定则从配置文件读取"
    )
    parser.add_argument(
        "--user", 
        type=str, 
        help="Neo4j用户名，如果不指定则从配置文件读取"
    )
    parser.add_argument(
        "--password", 
        type=str, 
        help="Neo4j密码，如果不指定则从配置文件读取"
    )
    
    args = parser.parse_args()
    
    # 构建图数据文件的绝对路径
    graph_data_path = os.path.join(args.input, "graph_data_complete.json")
    
    # 检查图数据文件是否存在
    if not os.path.exists(graph_data_path):
        logger.error(f"图数据文件不存在: {graph_data_path}")
        return 1
    
    logger.info(f"准备导入图数据: {graph_data_path}")
    
    # 创建Neo4j图数据库管理器
    neo4j_manager = Neo4jGraphManager(
        uri=args.uri,
        user=args.user,
        password=args.password
    )
    
    # 导入图数据
    success = neo4j_manager.import_graph_data(
        graph_data_path=graph_data_path,
        clear_existing=args.clear
    )
    
    if success:
        logger.info("图数据导入成功")
        return 0
    else:
        logger.error("图数据导入失败")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 