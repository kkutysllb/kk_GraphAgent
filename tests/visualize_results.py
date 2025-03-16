#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2025-03-22 11:30
# @Desc   : 测试结果可视化脚本
# --------------------------------------------------------
"""

import os
import json
import glob
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统推荐字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建控制台对象
console = Console()


def load_test_results(results_dir):
    """加载测试结果文件"""
    # 只加载动态异构图编码器测试结果文件
    json_files = glob.glob(os.path.join(results_dir, "dynamic_heterogeneous_graph_encoder_test_*.json"))
    results = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 验证文件格式
                if 'test_name' in data and 'tests' in data and isinstance(data['tests'], list):
                    data['file_path'] = file_path
                    data['file_name'] = os.path.basename(file_path)
                    results.append(data)
                else:
                    console.print(f"[yellow]文件格式不符: {file_path}[/yellow]")
        except Exception as e:
            console.print(f"[red]无法加载文件 {file_path}: {str(e)}[/red]")
    
    # 按时间戳排序
    results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return results


def display_test_summary(results):
    """显示测试结果摘要"""
    if not results:
        console.print("[yellow]没有找到测试结果文件[/yellow]")
        return
    
    table = Table(title="测试结果摘要")
    table.add_column("测试名称", style="cyan")
    table.add_column("时间戳", style="green")
    table.add_column("通过测试", style="green")
    table.add_column("失败测试", style="red")
    table.add_column("总测试数", style="blue")
    
    for result in results:
        test_name = result.get('test_name', '未知测试')
        timestamp = result.get('timestamp', '未知时间')
        tests = result.get('tests', [])
        
        passed = sum(1 for test in tests if test.get('status') == '通过')
        failed = sum(1 for test in tests if test.get('status') == '失败')
        total = len(tests)
        
        table.add_row(
            test_name,
            timestamp,
            str(passed),
            str(failed),
            str(total)
        )
    
    console.print(table)


def display_test_details(result):
    """显示单个测试结果的详细信息"""
    if not result:
        console.print("[yellow]没有选择测试结果[/yellow]")
        return
    
    test_name = result.get('test_name', '未知测试')
    timestamp = result.get('timestamp', '未知时间')
    
    console.print(Panel(f"[bold cyan]{test_name}[/bold cyan] - [green]{timestamp}[/green]"))
    
    tests = result.get('tests', [])
    for test in tests:
        name = test.get('name', '未知测试项')
        status = test.get('status', '未知')
        details = test.get('details', {})
        
        status_color = "green" if status == "通过" else "red"
        
        test_panel = Panel(
            Text.from_markup(f"[bold]{name}[/bold] - [{status_color}]{status}[/{status_color}]"),
            title="测试项",
            border_style=status_color
        )
        console.print(test_panel)
        
        # 显示详细信息
        if details:
            details_table = Table(show_header=True, header_style="bold magenta")
            details_table.add_column("参数", style="cyan")
            details_table.add_column("值", style="yellow")
            
            for key, value in details.items():
                if key != "error":
                    details_table.add_row(key, str(value))
            
            console.print(details_table)
        
        # 显示错误信息
        if "error" in details:
            console.print(Panel(f"[red]{details['error']}[/red]", title="错误信息"))
        
        # 显示子测试
        subtests = test.get('subtests', [])
        if subtests:
            subtests_table = Table(title="子测试结果")
            subtests_table.add_column("子测试名称", style="cyan")
            subtests_table.add_column("状态", style="green")
            
            for subtest in subtests:
                subtest_name = subtest.get('name', '未知子测试')
                subtest_status = subtest.get('status', '未知')
                status_style = "green" if subtest_status == "通过" else "red"
                subtests_table.add_row(subtest_name, f"[{status_style}]{subtest_status}[/{status_style}]")
            
            console.print(subtests_table)
        
        console.print()


def generate_test_charts(results, output_dir):
    """生成测试结果图表"""
    if not results:
        console.print("[yellow]没有找到测试结果文件，无法生成图表[/yellow]")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    data = []
    for result in results:
        test_name = result.get('test_name', '未知测试')
        timestamp = result.get('timestamp', '未知时间')
        tests = result.get('tests', [])
        
        # 跳过无效数据
        if timestamp == '未知时间' or not tests:
            continue
            
        passed = sum(1 for test in tests if test.get('status') == '通过')
        failed = sum(1 for test in tests if test.get('status') == '失败')
        total = len(tests)
        
        if total > 0:  # 只添加有效数据
            data.append({
                '测试名称': test_name,
                '时间戳': timestamp,
                '通过': passed,
                '失败': failed,
                '总数': total,
                '通过率': passed / total * 100 if total > 0 else 0
            })
    
    if not data:
        console.print("[yellow]没有有效的测试数据，无法生成图表[/yellow]")
        return
        
    df = pd.DataFrame(data)
    
    # 生成通过率趋势图（只有在有多个数据点时才生成）
    if len(df) > 1:
        plt.figure(figsize=(12, 6))
        plt.plot(df['时间戳'], df['通过率'], marker='o', linestyle='-', color='green')
        plt.title('测试通过率趋势')
        plt.xlabel('测试时间')
        plt.ylabel('通过率 (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        chart_path = os.path.join(output_dir, 'pass_rate_trend.png')
        plt.savefig(chart_path)
        console.print(f"[green]通过率趋势图已保存到: {chart_path}[/green]")
    else:
        console.print("[yellow]只有一个测试结果，无法生成趋势图[/yellow]")
    
    # 生成最新测试结果的详细图表
    if len(df) > 0:
        latest = df.iloc[0]
        
        # 如果所有测试都通过，生成按测试类型的柱状图而不是饼图
        latest_result = results[0]
        tests = latest_result.get('tests', [])
        
        # 按测试类型统计
        test_types = []
        test_statuses = []
        
        for test in tests:
            test_name = test.get('name', '未知')
            test_status = test.get('status', '未知')
            test_types.append(test_name)
            test_statuses.append(1 if test_status == '通过' else 0)
        
        # 生成柱状图
        plt.figure(figsize=(12, 6))
        bars = plt.bar(test_types, test_statuses, color='#66b3ff')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    '通过' if height == 1 else '失败',
                    ha='center', va='bottom', rotation=0)
        
        plt.title(f'最新测试结果详情: {latest["测试名称"]} ({latest["时间戳"]})')
        plt.xlabel('测试类型')
        plt.ylabel('状态 (1=通过, 0=失败)')
        plt.ylim(0, 1.2)  # 设置y轴范围，留出空间显示标签
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        bar_chart_path = os.path.join(output_dir, 'latest_test_details.png')
        plt.savefig(bar_chart_path)
        console.print(f"[green]最新测试详情图已保存到: {bar_chart_path}[/green]")
        
        # 如果有子测试，为动态异构图编码器测试生成子测试结果图
        for test in tests:
            if test.get('name') == '动态异构图编码器测试' and 'subtests' in test:
                subtests = test.get('subtests', [])
                if subtests:
                    subtest_names = []
                    subtest_statuses = []
                    
                    for subtest in subtests:
                        subtest_name = subtest.get('name', '未知')
                        subtest_status = subtest.get('status', '未知')
                        subtest_names.append(subtest_name)
                        subtest_statuses.append(1 if subtest_status == '通过' else 0)
                    
                    # 生成子测试柱状图
                    plt.figure(figsize=(12, 6))
                    bars = plt.bar(subtest_names, subtest_statuses, color='#99cc99')
                    
                    # 添加数值标签
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                '通过' if height == 1 else '失败',
                                ha='center', va='bottom', rotation=0)
                    
                    plt.title('动态异构图编码器子测试结果')
                    plt.xlabel('子测试类型')
                    plt.ylabel('状态 (1=通过, 0=失败)')
                    plt.ylim(0, 1.2)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    subtest_chart_path = os.path.join(output_dir, 'encoder_subtests.png')
                    plt.savefig(subtest_chart_path)
                    console.print(f"[green]子测试结果图已保存到: {subtest_chart_path}[/green]")
                    break
        
        # 生成测试参数比较图
        param_names = []
        param_values = []
        
        for test in tests:
            if 'details' in test and test['details']:
                for param, value in test['details'].items():
                    if param in ['hidden_dim', 'input_dim', 'output_dim'] and isinstance(value, (int, float)):
                        param_names.append(f"{test.get('name', '未知')} - {param}")
                        param_values.append(value)
        
        if param_names:
            plt.figure(figsize=(12, 6))
            bars = plt.bar(param_names, param_values, color='#ff9999')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        str(height),
                        ha='center', va='bottom', rotation=0)
            
            plt.title('测试参数比较')
            plt.xlabel('参数名称')
            plt.ylabel('参数值')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            param_chart_path = os.path.join(output_dir, 'parameter_comparison.png')
            plt.savefig(param_chart_path)
            console.print(f"[green]参数比较图已保存到: {param_chart_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description='测试结果可视化工具')
    parser.add_argument('--dir', type=str, default='test_results', help='测试结果目录')
    parser.add_argument('--charts', type=str, default='test_results/charts', help='图表输出目录')
    parser.add_argument('--latest', action='store_true', help='只显示最新的测试结果详情')
    args = parser.parse_args()
    
    console.print(Panel.fit("[bold cyan]测试结果可视化工具[/bold cyan]", border_style="green"))
    
    # 加载测试结果
    results = load_test_results(args.dir)
    
    # 显示测试摘要
    display_test_summary(results)
    
    # 显示最新测试详情
    if args.latest and results:
        console.print("\n[bold cyan]最新测试结果详情:[/bold cyan]")
        display_test_details(results[0])
    
    # 生成图表
    generate_test_charts(results, args.charts)


if __name__ == "__main__":
    main() 