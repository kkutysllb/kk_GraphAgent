# 测试结果目录

本目录用于存储动态异构图编码器及其组件的测试结果。

## 目录结构

```
test_results/
├── README.md                 # 本文件
├── charts/                   # 图表输出目录
│   ├── pass_rate_trend.png   # 测试通过率趋势图
│   └── latest_test_results.png # 最新测试结果饼图
├── *.log                     # 测试日志文件
└── *.json                    # 测试结果JSON文件
```

## 使用方法

### 运行测试

要运行测试并将结果保存到此目录，请执行以下命令：

```bash
python tests/test_dynamic_heterogeneous_graph_encoder.py
```

测试结果将在控制台输出，并保存为JSON文件和日志文件。

### 可视化测试结果

要可视化测试结果，请使用以下命令：

```bash
python tests/visualize_results.py
```

可选参数：

- `--dir`: 指定测试结果目录，默认为 `test_results`
- `--charts`: 指定图表输出目录，默认为 `test_results/charts`
- `--latest`: 只显示最新的测试结果详情

更多详细信息，请参阅项目文档：`docs/test_system.md` 