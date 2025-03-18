# 数据集生成优化文档

## 概述

本文档详细说明了图文数据集生成的流程、优化方法和使用指南。数据集生成是训练双通道编码器的关键环节，通过从Neo4j数据库提取图结构和特征，生成相应的文本描述，创建图-文对用于模型训练。

## 数据集生成流程

1. **从Neo4j加载数据**
   - 加载节点数据：按类型批量提取节点属性和特征
   - 加载边数据：提取边关系和属性
   - 统计节点类型分布

2. **生成文本描述**
   - 为每个节点生成自然语言描述
   - 包含节点属性、连接关系和层次位置
   - 针对不同节点类型，定制化描述内容

3. **创建图-文对**
   - 为每个节点创建子图：包含中心节点和相连节点
   - 将节点文本描述与子图结构配对
   - 记录节点类型、ID和其他元数据

4. **数据集分割**
   - 按节点类型分层分割：训练集、验证集、测试集
   - 默认比例：80%训练、10%验证、10%测试
   - 保持各类型节点的比例一致性

5. **数据增强**
   - 节点类型平衡：增加稀有类型节点的样本量
   - 生成负样本：创建不匹配的图-文对
   - 文本扰动：同义词替换、随机掩码等
   - 子图增强：节点和边的随机添加/删除

6. **保存数据集**
   - 支持PyTorch的.pt格式，提高加载效率
   - 保存样本示例，便于可视化检查
   - 记录数据集统计信息

## 近期优化

### 1. 修复数据集分割比例不协调问题

**问题描述：**
原始代码中存在重复分割数据集的问题，导致实际的数据集比例与预期不符：
- 在`GraphTextDataset`类初始化时，使用`_apply_dataset_split()`方法对数据进行分割
- 在`generate_dataset.py`中创建三个单独的数据集实例，分别指定为训练、验证和测试集
- 这种双重分割导致数据集大小和比例不一致

**解决方案：**
- 首先创建一个完整数据集，设置`skip_internal_split=True`跳过内部分割
- 手动将完整数据集的所有样本按节点类型分组
- 对每种节点类型的样本单独应用分割比例
- 对训练集应用数据增强、节点类型平衡和负样本生成
- 验证集和测试集保持原始数据不做增强

**优化效果：**
- 数据集分割比例严格遵循设定值（默认8:1:1）
- 各节点类型在各数据集中的分布一致
- 训练集经过增强后大幅扩充，但验证集和测试集保持原始规模

### 2. .pt格式数据集支持

**问题描述：**
原始代码使用JSON格式保存数据集，存在以下问题：
- 不支持直接保存numpy数组和PyTorch张量
- 读写速度慢，文件大小大
- 每次加载都需要进行大量的数据类型转换

**解决方案：**
- 实现`GraphTextDataset`类的`save`和`load`方法
- 使用`torch.save`和`torch.load`保存和加载数据
- 完整保存所有必要的属性，避免重复计算
- 添加安全处理，适配PyTorch 2.6+的安全加载机制

**优化效果：**
- 数据加载速度提升5-10倍
- 文件大小减少约30%
- 消除了数据类型转换的开销
- 与PyTorch的DataLoader完美兼容

### 3. 数据增强优化

**优化描述：**
- 重构了数据增强流程，只对训练集应用增强
- 实现了更高效的负样本生成策略
- 优化了节点类型平衡算法，更均衡地处理稀有类型
- 增加了详细的日志记录，便于监控增强效果

## 使用指南

### 生成数据集

```bash
python scripts/generate_dataset.py --output_dir datasets/full_dataset --balance --adaptive --augmentation
```

参数说明：
- `--output_dir`: 数据集输出目录
- `--balance`: 启用节点类型平衡
- `--adaptive`: 使用自适应子图大小
- `--augmentation`: 启用数据增强
- `--negative_ratio`: 负样本比例（默认0.3）
- `--train_ratio`: 训练集比例（默认0.8）
- `--val_ratio`: 验证集比例（默认0.1）
- `--test_ratio`: 测试集比例（默认0.1）
- `--num_workers`: 数据加载工作线程数（默认4）
- `--seed`: 随机种子（默认42）

### 加载数据集

```python
from rag.data.dataset import GraphTextDataset
import torch

# 加载训练集
train_dataset = GraphTextDataset.load("datasets/full_dataset/train.pt")

# 加载验证集
val_dataset = GraphTextDataset.load("datasets/full_dataset/val.pt")

# 加载测试集
test_dataset = GraphTextDataset.load("datasets/full_dataset/test.pt")

# 创建数据加载器
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=train_dataset.collate_fn
)
```

### 数据集统计信息

数据集生成过程会保存统计信息到`dataset_stats.pt`文件，可以通过以下方式加载查看：

```python
import torch

# 加载统计信息
stats = torch.load("datasets/full_dataset/dataset_stats.pt")

# 查看各数据集大小
print(f"训练集大小: {stats['train_size']}")
print(f"验证集大小: {stats['val_size']}")
print(f"测试集大小: {stats['test_size']}")

# 查看节点类型
print(f"节点类型: {stats['node_types']}")

# 查看配置信息
print(f"数据集配置: {stats['config']}")
```

## 性能统计

优化后的数据集生成流程在处理约4000个节点时的性能：

| 处理阶段 | 耗时(秒) | 处理速度(样本/秒) |
|----------|----------|------------------|
| 加载节点 | ~5       | ~700             |
| 加载边   | ~9       | ~300             |
| 生成描述 | ~4       | ~778             |
| 创建图文对| ~21      | ~170             |
| 数据增强 | ~90      | ~10015           |
| 总耗时   | ~135     | -                |

## 注意事项

1. 确保Neo4j数据库已正确配置并包含所需节点和边数据
2. 首次生成可能需要较长时间，建议增加日志详细程度监控进度
3. 大型数据集建议使用多工作线程，但注意内存占用
4. 使用生成的.pt格式数据集能显著提高训练速度
5. 数据集生成过程设计为幂等操作，可以安全地多次执行 