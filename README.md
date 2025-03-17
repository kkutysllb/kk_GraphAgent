# 5G核心网资源图谱智能代理

基于图数据库和深度学习的5G核心网资源智能管理系统，实现资源图谱的语义理解和智能查询。

## 功能特点

- 基于Neo4j的图数据存储和查询
- 双通道编码器实现文本和图结构的语义对齐
- 混合索引支持高效的多模态检索
- 自然语言到Cypher查询的智能转换

## 系统架构

### 核心组件

1. **特征提取器** (FeatureExtractor) ✅
   - 从Neo4j提取节点和关系特征
   - 支持静态和动态特征
   - 批量处理优化
   - 链路特征提取

2. **双通道编码器** (DualEncoder) ✅
   - BERT文本编码器
   - 动态异构图结构编码器
   - 对比学习训练
   - 相似度计算

3. **混合索引** (HybridIndex) 🔄
   - FAISS向量索引
   - 结构化过滤索引
   - 多条件组合查询
   - 批量构建支持

4. **查询处理器** (QueryProcessor) 🔄
   - 意图识别
   - 实体提取
   - 查询路由
   - 结果整合

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- Neo4j 4.4+
- CUDA 11.0+ (可选，用于GPU加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 部署neo4j数据库
```bash
./scripts/deploy.sh
```

### 原始数据预处理为图数据并校验
```bash
# 预处理图数据
python preprocess/scripts/preprocess_data.py --input datasets/raw/xbxa_dc4_topology.xlsx --output datasets/processed --interval 15 --workers 24

# 图数据校验
python preprocess/scripts/verify_graph_data.py
```

### 图数据导入到neo4j数据库
```bash
python preprocess/scripts/import_to_neo4j.py --input datasets/processed --clear --batch_size 2000
```

### 图数据库中数据抽取采样
```bash
python scripts/extract_sample_data.py
```

### 配置

1. Neo4j数据库配置
```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
```

2. 模型配置
```python
MODEL_CONFIG = {
    "text_model": "bert-base-chinese",
    "hidden_dim": 768,
    "num_heads": 8
}
```

## 使用示例

1. 特征提取

```python
from rag.feature_extractor import FeatureExtractor
from preprocess.utils.neo4j_graph_manager import Neo4jGraphManager

# 初始化Neo4j连接
graph_manager = Neo4jGraphManager(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password"
)

# 初始化特征提取器
extractor = FeatureExtractor(graph_manager)

# 提取节点特征
node_features = extractor.extract_node_features(node_id, node_type)

# 提取边特征
edge_features = extractor.extract_edge_features(source_id, target_id, edge_type)

# 提取链路特征
chain_features = extractor.extract_chain_features(dc_id, chain_type='both')
```

2. 数据集生成

```python
from rag.data.dataset import GraphTextDataset
from rag.feature_extractor import FeatureExtractor
from preprocess.utils.neo4j_graph_manager import Neo4jGraphManager
from rag.utils.config import load_config

# 加载配置
db_config = load_config("configs/database_config.yaml")

# 初始化Neo4j连接
graph_manager = Neo4jGraphManager(
    uri=db_config["neo4j"]["uri"],
    user=db_config["neo4j"]["user"],
    password=db_config["neo4j"]["password"]
)

# 初始化特征提取器
extractor = FeatureExtractor(graph_manager)

# 创建图文数据集
dataset = GraphTextDataset(
    graph_manager=graph_manager,
    feature_extractor=extractor,
    node_types=db_config["dataset"]["node_types"],
    edge_types=db_config["dataset"]["edge_types"],
    balance_node_types=True,
    adaptive_subgraph_size=True,
    data_augmentation=True,
    negative_sample_ratio=0.3
)

# 获取数据样本
sample = dataset[0]
print(f"节点ID: {sample['node_id']}")
print(f"文本描述: {sample['text']}")
print(f"子图节点数: {len(sample['subgraph']['nodes'])}")
```

3. 双通道编码器

```python
from rag.models.dual_encoder import DualEncoder
import torch

# 初始化双通道编码器
encoder = DualEncoder(
    text_embedding_dim=768,
    graph_embedding_dim=256,
    projection_dim=512
)

# 编码文本
text = "这是一个虚拟机节点，连接到多个网络设备"
text_embedding = encoder.encode_text(text)

# 编码图结构
# 假设graph_data是一个包含节点和边信息的字典
graph_embedding = encoder.encode_graph(graph_data)

# 计算相似度
similarity = encoder.compute_similarity(text_embedding, graph_embedding)
print(f"文本和图的相似度: {similarity.item()}")
```

4. 模型训练 (部分实现)

```python
from rag.models.dual_encoder import DualEncoder
from rag.models.loss import ContrastiveLoss
from torch.utils.data import DataLoader
import torch.optim as optim

# 初始化模型和损失函数
model = DualEncoder(text_embedding_dim=768, graph_embedding_dim=256, projection_dim=512)
criterion = ContrastiveLoss(margin=0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 假设train_dataset是已经创建好的GraphTextDataset实例
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练循环
for epoch in range(10):
    for batch in train_loader:
        # 获取文本和图数据
        texts = batch['text']
        graphs = batch['subgraph']
        
        # 前向传播
        text_embeddings = model.encode_text(texts)
        graph_embeddings = model.encode_graph(graphs)
        
        # 计算损失
        loss = criterion(text_embeddings, graph_embeddings)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

5. 查询处理 (即将实现)

```python
from rag.query_processor import QueryProcessor

processor = QueryProcessor(model, index)
result = processor.process_query("查找与VM-001相关的所有主机")
```

## 项目结构

```
.
├── configs/ # 配置文件
│ ├── database_config.yaml # 数据库配置
│ └── train_config.yaml # 训练配置
├── docs/ # 文档
│ ├── graph_encoder_design.md # 图编码器设计文档
│ ├── progress.md # 进度文档
│ ├── rag_design.md # RAG设计文档
│ ├── text_encoder_design.md # 文本编码器设计文档
│ └── work_log.md # 工作日志
├── preprocess/ # 预处理代码
│ └── utils/ # 工具函数
│ └── neo4j_graph_manager.py # Neo4j图管理器
├── rag/ # 主要代码
│ ├── data/ # 数据处理
│ │ └── dataset.py # 图文数据集
│ ├── models/ # 模型
│ │ ├── dual_encoder.py # 双通道编码器
│ │ └── loss.py # 损失函数
│ ├── utils/ # 工具函数
│ │ ├── config.py # 配置加载
│ │ └── logging.py # 日志工具
│ ├── feature_extractor.py # 特征提取器
│ └── test_dynamic_heterogeneous_graph_encoder.py # 图编码器测试
├── scripts/ # 工具脚本
│ ├── extract_sample_data.py # 样本数据提取
│ ├── generate_dataset.py # 数据集生成
│ ├── test_dataset.py # 数据集测试
│ ├── test_feature_extraction.py # 特征提取测试
│ ├── test_text_encoder.py # 文本编码器测试
│ └── visualize_results.py # 结果可视化
├── datasets/ # 数据集
│ ├── full_dataset/ # 完整数据集
│ └── samples/ # 样本数据
├── test_results/ # 测试结果
├── .gitignore # Git忽略文件
├── .gitattributes # Git属性文件
├── requirements.txt # 依赖
└── README.md # 说明文档
```

## 当前进度

- ✅ 特征提取模块完成 (2025-03-15)
  - 节点特征提取
  - 边特征提取
  - 链路特征提取
  - 测试脚本

- ✅ 图文对生成完成 (2025-03-20)
  - GraphTextDataset类实现
  - 中文文本描述生成
  - 图文对创建与测试
  - 复杂查询样本生成
  - 统计信息查询样本生成

- ✅ 动态异构图编码器 (2025-03-22)
  - 节点级注意力层
  - 边级注意力层
  - 时间序列编码器
  - 层级感知模块
  - 集成与优化

- ✅ 双通道编码器集成 (2025-03-23)
  - 架构优化
  - 接口统一
  - 参数调整
  - 兼容性保证

- ✅ 文本编码器 (2025-03-24)
  - 基于中文BERT的编码器
  - 多种池化策略支持
  - 层权重学习机制
  - 特征投影与维度调整
  - 全面测试与性能分析

- 🔄 训练流程 (计划开始: 2025-03-26)
  - 对比学习损失函数
  - 优化策略设计
  - 训练与验证流程
  - 模型保存机制

- 🔄 数据集生成优化 (2024-03-15)
  - 修复了特征提取和描述生成问题
  - 优化了数据加载性能
  - 改进了边查询逻辑
  - 完善了错误处理机制

详细进度请查看 [进度报告](./docs/progress_report.md) 和 [工作日志](./docs/work_log.md)

## 注意事项

1. 大文件管理
   - 模型检查点保存在`checkpoints/`目录
   - 向量索引保存在`indices/`目录
   - 使用Git LFS管理大文件

2. 性能优化
   - 批量处理数据
   - 使用GPU加速
   - 索引预热

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 许可证

MIT License
