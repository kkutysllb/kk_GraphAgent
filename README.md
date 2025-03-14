# 5G核心网资源图谱智能代理

基于图数据库和深度学习的5G核心网资源智能管理系统，实现资源图谱的语义理解和智能查询。

## 功能特点

- 基于Neo4j的图数据存储和查询
- 双通道编码器实现文本和图结构的语义对齐
- 混合索引支持高效的多模态检索
- 自然语言到Cypher查询的智能转换

## 系统架构

### 核心组件

1. **特征提取器** (FeatureExtractor)
   - 从Neo4j提取节点和关系特征
   - 支持静态和动态特征
   - 批量处理优化
   - 文本描述生成

2. **双通道编码器** (DualEncoder)
   - BERT文本编码器
   - GAT图结构编码器
   - 对比学习训练
   - 相似度计算

3. **混合索引** (HybridIndex)
   - FAISS向量索引
   - 结构化过滤索引
   - 多条件组合查询
   - 批量构建支持

4. **查询处理器** (QueryProcessor)
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

extractor = FeatureExtractor(uri, user, password)
features = extractor.extract_node_static_features()
```

2. 模型训练
```python
from rag.trainer import Trainer

trainer = Trainer(model, train_loader, val_loader)
trainer.train(num_epochs=10)
```

3. 查询处理
```python
from rag.query_processor import QueryProcessor

processor = QueryProcessor(model, index)
result = processor.process_query("查找与VM-001相关的所有主机")
```

## 项目结构

```
.
├── docs/                    # 文档
├── rag/                    # 主要代码
│   ├── feature_extractor.py
│   ├── encoder.py
│   ├── indexer.py
│   ├── query_processor.py
│   └── trainer.py
├── tests/                  # 测试代码
├── scripts/                # 工具脚本
├── requirements.txt        # 依赖
└── README.md              # 说明文档
```

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
