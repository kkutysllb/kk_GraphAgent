# 文本编码器设计文档

## 1. 设计目标

文本编码器是双通道编码器架构中的关键组件，负责将自然语言查询转换为向量表示。设计目标包括：

1. 基于预训练中文语言模型，充分利用其语义理解能力
2. 支持多种池化策略，适应不同的语义表示需求
3. 提供灵活的特征维度配置，确保与图编码器的兼容性
4. 实现高效的批处理和设备迁移支持，提升处理效率

## 2. 架构设计

```
                                ┌─────────────────┐
                                │     输入文本     │
                                └────────┬────────┘
                                         │
                                ┌────────▼────────┐
                                │     分词器      │
                                └────────┬────────┘
                                         │
                                ┌────────▼────────┐
                                │   BERT编码器    │
                                └────────┬────────┘
                                         │
                     ┌───────────────────┴───────────────────┐
                     │                                       │
             ┌───────▼───────┐                     ┌─────────▼─────────┐
             │  序列表示      │                     │    池化策略        │
             └───────┬───────┘                     │  - CLS            │
                     │                             │  - Mean           │
                     │                             │  - Max            │
                     │                             │  - Attention      │
                     │                             │  - Weighted Layer │
                     │                             └─────────┬─────────┘
                     │                                       │
                     │                                       │
             ┌───────▼───────┐                     ┌─────────▼─────────┐
             │  序列投影层    │                     │    池化投影层      │
             └───────┬───────┘                     └─────────┬─────────┘
                     │                                       │
                     └───────────────────┬───────────────────┘
                                         │
                                ┌────────▼────────┐
                                │    输出表示      │
                                └─────────────────┘
```

## 3. 核心组件

### 3.1 预训练模型

- 使用`bert-base-chinese`作为基础模型
- 支持模型参数冻结选项，适应不同的训练策略
- 提供完整的隐藏状态访问，支持高级特征提取

### 3.2 池化策略

实现了五种池化策略，各有优缺点：

1. **CLS池化**
   - 使用[CLS]标记的表示作为整个序列的表示
   - 优点：简单高效，BERT预训练时优化过的表示
   - 缺点：可能丢失句子中间的重要信息

2. **Mean池化**
   - 对所有token的表示取平均
   - 优点：考虑所有token的贡献
   - 缺点：对所有token赋予相同权重，可能受无意义token影响

3. **Max池化**
   - 对每个维度取所有token表示的最大值
   - 优点：能捕获最显著的特征
   - 缺点：可能过度关注个别特征，忽略整体语义

4. **Attention池化**
   - 学习注意力权重，加权组合token表示
   - 优点：自适应关注重要token
   - 缺点：增加了模型复杂度和参数量

5. **Weighted Layer池化**
   - 学习不同层表示的权重，加权组合多层表示
   - 优点：利用不同层次的语义信息
   - 缺点：增加了训练难度和过拟合风险

### 3.3 特征投影

- 实现了灵活的特征投影层，支持任意输出维度
- 使用LayerNorm和Dropout提高泛化能力
- 当输入维度等于输出维度时，使用Identity层减少计算开销

### 3.4 直接编码接口

- 提供了`encode_text`方法，支持直接输入文本
- 自动处理分词、填充、截断等预处理步骤
- 自动处理设备迁移，确保与模型在同一设备上

## 4. 实现细节

### 4.1 初始化参数

```python
def __init__(
    self,
    model_name: str = "bert-base-chinese",
    output_dim: int = 768,
    dropout: float = 0.1,
    freeze_base: bool = False,
    pooling_strategy: str = "cls",
    max_length: int = 512,
    use_layer_weights: bool = False,
    num_hidden_layers: Optional[int] = None
):
    # 实现代码...
```

### 4.2 前向传播

```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    # 实现代码...
    return {
        'embeddings': sequence_output,
        'pooled': pooled_output,
        'hidden_states': outputs.hidden_states
    }
```

### 4.3 池化实现

```python
def _mean_pooling(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # 实现代码...

def _max_pooling(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # 实现代码...

def _attention_pooling(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # 实现代码...

def _weighted_layer_pooling(self, hidden_states: Tuple[torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
    # 实现代码...
```

## 5. 性能评估

### 5.1 池化策略比较

在5G核心网资源查询文本上的测试结果：

| 池化策略 | 平均相似度 | 区分能力 | 计算开销 | 适用场景 |
|---------|-----------|---------|---------|---------|
| CLS | 0.789 | 高 | 低 | 一般文本分类和语义匹配 |
| Mean | 0.774 | 中 | 低 | 文档级别的语义匹配 |
| Max | 0.906 | 中 | 低 | 关键词检索，特征词匹配 |
| Attention | 0.778 | 高 | 中 | 复杂语义理解，需要关注特定部分的场景 |
| Weighted | 0.964 | 低 | 高 | 需要高召回率的场景 |

### 5.2 推荐配置

基于测试结果，推荐以下配置：

1. **检索阶段**：使用CLS或Attention池化策略，提供更好的语义区分能力
2. **重排序阶段**：可以考虑结合多种策略的结果
3. **特定场景**：
   - 资源查询类文本：CLS策略
   - 状态监控类文本：Attention策略
   - 需要高召回率：Weighted策略

## 6. 与其他组件的集成

### 6.1 与双通道编码器的集成

- 确保文本编码器和图编码器的输出维度一致
- 使用投影层对齐特征空间
- 支持批处理和设备迁移

### 6.2 与训练流程的集成

- 支持梯度传播和参数更新
- 提供完整的隐藏状态，支持高级特征提取
- 支持模型参数冻结选项，适应不同的训练策略

## 7. 未来优化方向

1. **领域适应**：针对5G核心网领域数据微调BERT模型
2. **多模型集成**：支持多种预训练模型，如RoBERTa、ERNIE等
3. **动态池化**：实现自适应选择最佳池化策略的机制
4. **跨语言支持**：添加多语言模型支持，处理中英文混合查询
5. **知识增强**：集成领域知识图谱，提升语义理解能力 