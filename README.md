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

2. **双通道编码器** (DualEncoder) 🔄
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

### 使用示例

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

2. 模型训练 (即将实现)
```python
from rag.trainer import Trainer

trainer = Trainer(model, train_loader, val_loader)
trainer.train(num_epochs=10)
```

3. 查询处理 (即将实现)
```python
from rag.query_processor import QueryProcessor

processor = QueryProcessor(model, index)
result = processor.process_query("查找与VM-001相关的所有主机")
```

## 项目结构

```
.
├── docs/                   # 文档
│   ├── rag_design_v2.md    # RAG设计文档V2
│   ├── llm_enhanced_design.md # LLM增强设计
│   └── progress_report.md  # 进度报告
├── rag/                    # 主要代码
│   ├── feature_extractor.py # 特征提取器
│   ├── encoder.py          # 编码器(即将实现)
│   ├── indexer.py          # 索引器(即将实现)
│   ├── query_processor.py  # 查询处理器(即将实现)
│   └── trainer.py          # 训练器(即将实现)
├── preprocess/             # 预处理代码
│   └── utils/              # 工具函数
│       └── neo4j_graph_manager.py # Neo4j图管理器
├── scripts/                # 工具脚本
│   └── test_feature_extraction.py # 特征提取测试脚本
├── test_results/           # 测试结果
├── requirements.txt        # 依赖
└── README.md               # 说明文档
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
