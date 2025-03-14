# RAG知识库设计文档

## 1. 系统架构

### 1.1 核心组件
- **特征提取器** (FeatureExtractor)
  - 负责从Neo4j数据库中提取节点和关系特征
  - 支持静态特征和动态特征的提取
  - 实现批量处理以提高效率
  - 生成文本描述用于训练

- **双通道编码器** (DualEncoder)
  - 文本编码器：基于预训练中文BERT模型
  - 图编码器：基于图注意力网络(GAT)
  - 实现对比学习对齐文本和图表示
  - 支持温度缩放的相似度计算

- **混合索引** (HybridIndex)
  - 向量索引：使用FAISS进行语义相似搜索
  - 结构索引：使用字典索引快速过滤
  - 支持多条件组合查询
  - 实现批量构建和增量更新

- **查询处理器** (QueryProcessor)
  - 意图识别：基于规则和模式匹配
  - 实体提取：支持节点类型、ID等关键信息
  - 查询路由：映射到对应的Cypher模板
  - 结果整合：组合向量搜索和图查询结果

### 1.2 数据流
1. 特征提取阶段
   - 从Neo4j提取节点和关系数据
   - 生成结构化特征和文本描述
   - 批量处理并保存到中间文件

2. 模型训练阶段
   - 加载特征数据和文本描述
   - 训练双通道编码器对齐表示
   - 保存训练好的模型和配置

3. 索引构建阶段
   - 使用训练好的模型编码所有节点
   - 构建FAISS向量索引
   - 建立结构化索引

4. 查询处理阶段
   - 解析自然语言查询意图
   - 向量检索相似节点
   - 生成和执行Cypher查询
   - 返回组合查询结果

## 2. 详细设计

### 2.1 特征提取器
```python
class FeatureExtractor:
    def extract_node_static_features(self):
        # 提取节点类型、属性、度数等静态特征
        
    def extract_node_dynamic_features(self):
        # 提取性能指标、日志等动态特征
        
    def extract_relationship_features(self):
        # 提取关系类型、权重等特征
        
    def extract_subgraph_features(self):
        # 提取局部子图结构特征
        
    def generate_text_description(self):
        # 生成节点的文本描述
```

### 2.2 双通道编码器
```python
class DualEncoder(nn.Module):
    def encode_text(self, texts):
        # 使用BERT编码文本
        
    def encode_graph(self, node_features, edge_index):
        # 使用GAT编码图结构
        
    def compute_similarity(self, graph_embeddings, text_embeddings):
        # 计算余弦相似度
        
    def compute_loss(self, similarity):
        # 计算对比学习损失
```

### 2.3 混合索引
```python
class HybridIndex:
    def add_item(self, item_id, vector, type_label, attributes):
        # 添加向量和结构化索引
        
    def build_index(self):
        # 构建FAISS索引
        
    def search(self, query_vector, filters):
        # 组合检索
```

### 2.4 查询处理器
```python
class QueryProcessor:
    def process_query(self, query):
        # 处理自然语言查询
        
    def _extract_intent_entities(self, query):
        # 提取意图和实体
        
    def _process_node_info(self, entities, vector):
        # 处理节点信息查询
```

## 3. 实现进展

### 3.1 已完成功能
- [x] 特征提取器核心功能
  - 节点静态特征提取
  - 节点动态特征提取
  - 关系特征提取
  - 子图特征提取
  - 文本描述生成

- [x] 双通道编码器
  - BERT文本编码器
  - GAT图编码器
  - 对比学习损失
  - 温度参数优化

- [x] 混合索引系统
  - FAISS向量索引
  - 结构化过滤索引
  - 批量构建功能
  - 多条件搜索

- [x] 查询处理器
  - 意图识别模块
  - 实体提取功能
  - Cypher模板系统
  - 查询路由逻辑

### 3.2 待优化项目
- [ ] 特征提取性能优化
- [ ] 增量更新支持
- [ ] 分布式训练支持
- [ ] 查询缓存机制
- [ ] 错误恢复机制

## 4. 部署说明

### 4.1 环境依赖
```
                   +----------------------+
                   |  User Query          |
                   +----------+-----------+
                              |
                              v
+-----------------+  +-------+--------+  +-----------------+
| Cypher Translator|->| Hybrid Index   |->| Knowledge Graph |
+-----------------+  +-------+--------+  +--------+--------+
                              |                   |
                              v                   v
                    +---------+---------+ +-------+-------+
                    | Vector Retrieval  | | Graph Traversal|
                    +-------------------+ +----------------+
```

### 4.2 模型存储
- 模型检查点：保存在`checkpoints/`目录
- 向量索引：保存在`indices/`目录
- 中间特征：保存在`features/`目录

### 4.3 注意事项
- 大文件(>100MB)使用Git LFS管理
- 模型检查点单独存储
- 特征文件按需生成