# Agno：5G核心网资源图谱智能系统

Agno是一个基于多智能体协同的自然语言到Cypher查询转换系统，专为5G核心网资源图谱查询设计。系统集成了大语言模型、图嵌入模型和双数据库查询架构，能高效将用户自然语言查询转换为精确的Neo4j Cypher查询语句。

## 核心特性

- **多智能体协同架构**：采用多个专业智能体协作完成查询转换
- **本地大模型集成**：集成vLLM作为推理框架，支持部署本地大语言模型
- **图嵌入和向量检索**：利用双流图嵌入模型增强查询能力
- **专业领域知识**：针对5G核心网资源图谱进行优化
- **多类型查询支持**：支持基础查询、拓扑分析、性能分析、故障诊断等多类型查询

## 系统架构

系统采用模块化设计，主要组件包括：

1. **基础设施层**：
   - Neo4j图数据库：存储5G核心网资源图谱
   - vLLM服务：提供高性能本地大模型推理
   - Milvus向量数据库：存储和检索图嵌入向量

2. **智能体层**：
   - 编排协调器：管理智能体协作流程
   - 意图分析智能体：识别用户查询意图
   - 实体识别智能体：从查询中提取关键实体
   - Cypher生成智能体：生成Cypher查询语句
   - 以及其他专业智能体

3. **模型层**：
   - 本地大语言模型：提供自然语言理解和生成能力
   - 图嵌入模型：生成节点和查询的向量表示

## 快速开始

### 环境要求

- Docker和Docker Compose
- NVIDIA GPU（推荐用于大模型推理）
- NVIDIA Container Toolkit（用于GPU访问）

### 部署步骤

1. 克隆代码仓库：

```bash
git clone https://github.com/yourusername/agno.git
cd agno
```

2. （可选）修改配置文件：

```bash
# 编辑配置文件
nano config.json
```

3. 使用Docker Compose启动服务：

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps
```

4. 访问服务：

- Neo4j浏览器：http://localhost:7474
- Agno API服务：http://localhost:5000/api/health (健康检查)

## API使用方法

### 自然语言查询API

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "查找广州数据中心中CPU使用率超过80%的虚拟机",
    "session_id": "optional-session-id"
  }'
```

## 本地开发

### 环境设置

1. 创建Python虚拟环境：

```bash
python -m venv venv
source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 运行应用：

```bash
python src/app.py
```

### 项目结构

```
agno/
├── src/                 # 源代码目录
│   ├── agents/          # 智能体实现
│   ├── models/          # 模型相关代码
│   ├── utils/           # 工具函数
│   └── app.py           # 应用入口
├── config.json          # 配置文件
├── docker-compose.yml   # Docker Compose配置
├── Dockerfile           # Docker构建文件
└── requirements.txt     # Python依赖
```

## 许可证

本项目采用MIT许可证，详情见LICENSE文件。
