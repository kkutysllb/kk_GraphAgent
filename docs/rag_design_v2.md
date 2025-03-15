# RAG 知识库系统设计文档 V2

## 1. 系统架构

### 1.1 整体架构
```
                                    ┌─────────────────┐
                                    │     用户查询     │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │   查询处理模块    │
                                    └────────┬────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     │                                               │
             ┌───────▼───────┐                             ┌────────▼────────┐
             │  文本编码通道   │                             │   图编码通道     │
             └───────┬───────┘                             └────────┬────────┘
                     │                                              │
             ┌───────▼───────┐                             ┌────────▼────────┐
             │   BERT 编码    │                             │    GAT 编码     │
             └───────┬───────┘                             └────────┬────────┘
                     │                                              │
                     └──────────────────┬───────────────────────────┘
                                       │
                              ┌────────▼────────┐
                              │    混合索引      │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │   Neo4j 数据库   │
                              └─────────────────┘
```

### 1.2 核心模块设计

#### 1.2.1 文本编码器
- 基于预训练语言模型（BERT）
- 支持文本序列和池化表示
- 可配置的输出维度和dropout
- 可选的模型参数冻结

#### 1.2.2 图编码器
- 基于图注意力网络（GAT）
- 多层注意力机制
- 边特征融合
- 残差连接和层归一化
- 可配置的注意力头数和隐藏维度

#### 1.2.3 双通道编码器
- 文本和图特征对齐
- 投影层用于特征转换
- 余弦相似度计算
- 批处理支持

## 2. 数据处理流程

### 2.1 数据集设计
```python
class GraphTextDataset:
    def __init__(self):
        # 初始化数据集
        self.nodes = []  # 节点列表
        self.edges = []  # 边列表
        self.text_descriptions = {}  # 文本描述字典
        
    def _load_graph_data(self):
        # 从Neo4j加载图数据
        pass
        
    def _generate_text_descriptions(self):
        # 生成节点文本描述
        pass
        
    def _create_pairs(self):
        # 创建图文对
        pass
```

### 2.2 数据预处理
- 节点特征提取和归一化
- 边特征提取和归一化
- 文本标记化和截断
- 子图采样和批处理

### 2.3 数据增强
- 随机掩码
- 子图采样
- 文本重写
- 对比学习样本构造

## 3. 训练策略

### 3.1 损失函数设计
1. **对比损失**
```python
class ContrastiveLoss:
    def __init__(self, temperature=0.07):
        self.temperature = temperature
        
    def forward(self, text_embeddings, graph_embeddings):
        # 计算对比损失
        pass
```

2. **InfoNCE损失**
```python
class InfoNCELoss:
    def __init__(self, temperature=0.07):
        self.temperature = temperature
        
    def forward(self, query, positive_key, negative_keys):
        # 计算InfoNCE损失
        pass
```

3. **三元组损失**
```python
class TripletLoss:
    def __init__(self, margin=0.3):
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        # 计算三元组损失
        pass
```

### 3.2 优化策略
- AdamW优化器
- 余弦学习率调度
- 梯度裁剪
- 早停策略

### 3.3 训练流程
1. 数据加载和批处理
2. 双通道编码
3. 损失计算
4. 反向传播和优化
5. 验证和模型保存

## 4. 评估指标

### 4.1 检索指标
- Hits@K (K=1,5,10)
- MRR@K
- Precision@K
- Recall@K

### 4.2 语义相似度指标
- 余弦相似度
- 语义对齐准确率
- 特征空间分布

### 4.3 效率指标
- 查询延迟
- 内存使用
- 吞吐量

## 5. 配置管理

### 5.1 模型配置
```yaml
model:
  text_model_name: "bert-base-uncased"
  node_dim: 256
  edge_dim: 64
  hidden_dim: 256
  output_dim: 768
  num_heads: 8
  dropout: 0.1
```

### 5.2 训练配置
```yaml
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 1e-4
  weight_decay: 1e-5
  warmup_steps: 1000
  max_grad_norm: 1.0
```

### 5.3 数据配置
```yaml
data:
  max_text_length: 512
  max_node_size: 100
  max_edge_size: 200
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```

## 6. 下一步计划

### 6.1 LLM集成
- 选择合适的LLM模型
- 设计提示工程策略
- 实现查询意图识别
- 开发Cypher查询生成器

### 6.2 Agent系统
- 设计Agent架构
- 实现任务规划
- 开发工具调用接口
- 集成对话管理 