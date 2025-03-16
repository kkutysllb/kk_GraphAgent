# 测试结果系统设计与使用指南

## 1. 概述

测试结果系统是为动态异构图编码器及其组件设计的自动化测试和结果可视化系统。该系统能够：

- 自动执行测试并记录详细的测试结果
- 将测试结果以结构化JSON格式保存
- 生成测试日志文件
- 提供测试结果可视化工具
- 生成测试通过率和结果分析图表

## 2. 系统架构

测试结果系统由以下几个部分组成：

1. **测试脚本**：位于`tests/`目录，包含各种测试用例
2. **测试结果存储**：位于`test_results/`目录，用于存储JSON格式的测试结果和日志文件
3. **可视化工具**：位于`tests/visualize_results.py`，用于可视化测试结果
4. **图表输出**：位于`test_results/charts/`目录，包含生成的测试结果图表

## 3. 目录结构

```
project_root/
├── test_results/                  # 测试结果目录
│   ├── charts/                    # 图表输出目录
│   │   ├── pass_rate_trend.png    # 测试通过率趋势图
│   │   └── latest_test_results.png # 最新测试结果饼图
│   ├── *.log                      # 测试日志文件
│   ├── *.json                     # 测试结果JSON文件
│   └── README.md                  # 测试结果目录说明文件
└── tests/                         # 测试脚本目录
    ├── visualize_results.py       # 测试结果可视化工具
    └── test_dynamic_heterogeneous_graph_encoder.py  # 动态异构图编码器测试脚本
```

## 4. 测试结果格式

测试结果以JSON格式保存，包含以下信息：

```json
{
  "test_name": "动态异构图编码器测试",
  "timestamp": "2025-03-22 11:30:00",
  "tests": [
    {
      "name": "节点级注意力层测试",
      "status": "通过",
      "details": {
        "input_dim": 16,
        "edge_dim": 8,
        "hidden_dim": 32,
        "output_shape": "[20, 32] for each edge type"
      }
    },
    {
      "name": "动态异构图编码器测试",
      "status": "通过",
      "details": {},
      "subtests": [
        {
          "name": "使用所有特征",
          "status": "通过",
          "details": {
            "output_shape": "[20, 64]"
          }
        },
        {
          "name": "不使用时间序列特征",
          "status": "通过",
          "details": {
            "output_shape": "[20, 64]"
          }
        }
      ]
    }
  ]
}
```

## 5. 使用方法

### 5.1 运行测试

要运行测试并将结果保存到测试结果目录，请执行以下命令：

```bash
python tests/test_dynamic_heterogeneous_graph_encoder.py
```

测试结果将在控制台输出，并保存为JSON文件和日志文件。

### 5.2 可视化测试结果

要可视化测试结果，请使用以下命令：

```bash
python tests/visualize_results.py
```

可选参数：

- `--dir`: 指定测试结果目录，默认为 `test_results`
- `--charts`: 指定图表输出目录，默认为 `test_results/charts`
- `--latest`: 只显示最新的测试结果详情

示例：

```bash
# 显示所有测试结果摘要和图表
python tests/visualize_results.py

# 显示最新测试结果的详细信息
python tests/visualize_results.py --latest

# 指定不同的测试结果目录和图表输出目录
python tests/visualize_results.py --dir /path/to/results --charts /path/to/charts
```

## 6. 测试脚本设计

测试脚本使用Python的`unittest`框架，并进行了扩展以支持结果记录和可视化。主要特点包括：

1. **自动化测试**：使用`unittest`框架自动执行测试用例
2. **结果记录**：将测试结果记录到控制台和JSON文件中
3. **日志记录**：使用Python的`logging`模块记录测试过程
4. **异常处理**：捕获并记录测试过程中的异常
5. **子测试支持**：支持在一个测试用例中执行多个子测试

### 6.1 测试用例示例

```python
def test_node_level_attention(self):
    """测试节点级注意力层"""
    test_result = {"name": "节点级注意力层测试", "status": "失败", "details": {}}
    
    try:
        logger.info("测试节点级注意力层...")
        
        # 创建节点级注意力层
        node_level_attention = NodeLevelAttention(
            input_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            edge_types=self.edge_types,
            dropout=0.1,
            use_edge_features=True
        )
        
        # 前向传播
        edge_type_embeddings = node_level_attention(
            self.node_features,
            self.edge_indices_dict,
            self.edge_features_dict
        )
        
        # 验证输出
        self.assertIsInstance(edge_type_embeddings, dict)
        self.assertEqual(len(edge_type_embeddings), len(self.edge_types))
        
        for edge_type in self.edge_types:
            self.assertIn(edge_type, edge_type_embeddings)
            self.assertEqual(edge_type_embeddings[edge_type].shape, (self.num_nodes, self.hidden_dim))
        
        logger.info("节点级注意力层测试通过！")
        test_result["status"] = "通过"
        test_result["details"] = {
            "input_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "hidden_dim": self.hidden_dim,
            "output_shape": f"[{self.num_nodes}, {self.hidden_dim}] for each edge type"
        }
    except Exception as e:
        logger.error(f"节点级注意力层测试失败: {str(e)}")
        test_result["details"]["error"] = str(e)
    
    self.test_results["tests"].append(test_result)
```

## 7. 可视化工具设计

可视化工具使用Python的`rich`库提供丰富的终端输出，使用`matplotlib`和`pandas`生成图表。主要功能包括：

1. **测试结果摘要**：显示所有测试结果的摘要信息
2. **测试详情**：显示单个测试结果的详细信息
3. **通过率趋势图**：生成测试通过率随时间变化的趋势图
4. **测试结果饼图**：生成最新测试结果的通过/失败比例饼图

## 8. 依赖项

测试结果系统依赖以下Python包：

- unittest (Python标准库)
- logging (Python标准库)
- json (Python标准库)
- matplotlib
- pandas
- rich

可以使用以下命令安装所需的依赖：

```bash
pip install matplotlib pandas rich
```

## 9. 最佳实践

1. **定期运行测试**：定期运行测试以确保代码质量
2. **保存测试结果**：保存测试结果以便追踪问题
3. **分析测试趋势**：使用可视化工具分析测试通过率趋势
4. **详细记录测试信息**：在测试用例中记录详细的测试信息
5. **使用子测试**：对于复杂的测试用例，使用子测试进行组织

## 10. 扩展与定制

测试结果系统设计为可扩展的，可以通过以下方式进行定制：

1. **添加新的测试用例**：在`tests/`目录中添加新的测试脚本
2. **扩展测试结果格式**：在测试脚本中扩展测试结果的JSON格式
3. **定制可视化工具**：修改`tests/visualize_results.py`以添加新的可视化功能
4. **集成CI/CD**：将测试系统集成到CI/CD流程中

## 11. 故障排除

如果遇到问题，请检查：

1. **测试环境**：确保测试环境正确设置
2. **依赖项**：确保所有依赖项已正确安装
3. **权限**：确保有权限写入测试结果目录
4. **日志文件**：查看日志文件以获取详细的错误信息 