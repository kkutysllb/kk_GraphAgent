from neo4j import GraphDatabase

# 连接数据库
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Oms_2600a"))
session = driver.session()

try:
    # 检查节点
    print("节点统计:")
    node_result = session.run("MATCH (n) RETURN DISTINCT labels(n) as labels, count(*) as count")
    for record in node_result:
        print(f"  {record['labels']}: {record['count']}")
    
    # 检查关系
    print("\n关系统计:")
    rel_result = session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) as type, count(*) as count")
    for record in rel_result:
        print(f"  {record['type']}: {record['count']}")
        
    # 检查一个具体的边
    print("\n检查第一条边:")
    edge_result = session.run("MATCH ()-[r]->() RETURN r LIMIT 1")
    edge = edge_result.single()
    if edge:
        print(f"  找到边: {edge['r']}")
    else:
        print("  没有找到任何边")
        
finally:
    session.close()
    driver.close() 