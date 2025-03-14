#!/bin/bash
# deploy_neo4j.sh

set -e  # 遇到错误立即退出

# 检查必要的工具
command -v docker >/dev/null 2>&1 || { echo "需要安装 docker"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "需要安装 docker-compose"; exit 1; }

# 设置变量
NEO4J_PASSWORD="Oms_2600a"     # 密码
NEO4J_VERSION="5.15.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# 停止并删除旧容器
echo "清理旧的Neo4j容器..."
docker rm -f neo4j || true

# 创建必要的目录
echo "创建Neo4j目录..."
mkdir -p neo4j/data neo4j/logs neo4j/import neo4j/plugins

# 设置目录权限
echo "设置目录权限..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    chmod -R 755 neo4j/
else
    # Linux
    sudo chown -R $USER neo4j/
    sudo chmod -R 755 neo4j/
fi

# 启动Neo4j容器
echo "启动Neo4j容器..."
docker run -d --name neo4j --restart unless-stopped \
    -p 7474:7474 -p 7687:7687 \
    -v "$(pwd)/neo4j/data:/data" \
    -v "$(pwd)/neo4j/logs:/logs" \
    -v "$(pwd)/neo4j/import:/import" \
    -e "NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}" \
    -e "NEO4J_dbms_memory_pagecache_size=1G" \
    -e "NEO4J_dbms_memory_heap_initial__size=1G" \
    -e "NEO4J_dbms_memory_heap_max__size=1G" \
    neo4j:${NEO4J_VERSION}

# 等待Neo4j启动
echo "等待Neo4j启动..."
MAX_ATTEMPTS=60  # 增加等待时间到2分钟
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s -u "neo4j:${NEO4J_PASSWORD}" http://localhost:7474/db/data/ &>/dev/null; then
        echo "Neo4j已启动并可以连接"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo "等待Neo4j启动... (${ATTEMPT}/${MAX_ATTEMPTS})"
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "错误: Neo4j启动超时"
    docker logs neo4j
    exit 1
fi

echo "检查Neo4j状态..."
docker ps | grep neo4j

echo "部署完成！"
echo "Neo4j浏览器: http://localhost:7474"
echo "用户名: neo4j"
echo "密码: ${NEO4J_PASSWORD}"