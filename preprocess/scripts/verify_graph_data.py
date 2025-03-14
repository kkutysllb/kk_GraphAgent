#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : kkutysllb
# @E-mail : libing1@sn.chinamobile.com, 31468130@qq.com
# @Date   : 2024-03-11 16:30
# @Desc   : Graph Data Verification Script
# --------------------------------------------------------
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

# Create thread lock for synchronizing plot operations
plot_lock = threading.Lock()

class GraphDataVerifier:
    def __init__(self, graph_data_path, output_dir, max_workers=24):
        """Initialize graph data verifier
        
        Args:
            graph_data_path: Path to graph data file
            output_dir: Output directory
            max_workers: Maximum number of threads
        """
        self.graph_data = self._load_data(graph_data_path)
        self.output_dir = output_dir
        self.max_workers = max_workers
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plot style
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
    def _load_data(self, graph_data_path):
        """Load graph data"""
        print(f"\nLoading graph data from: {graph_data_path}")
        with open(graph_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Print data structure information
        print("\nData structure information:")
        print(f"Number of nodes: {len(data.get('nodes', []))}")
        print(f"Number of edges: {len(data.get('edges', []))}")
        
        # Check first node for metrics
        if data.get('nodes'):
            first_node = data['nodes'][0]
            print("\nFirst node structure:")
            print(f"Node type: {first_node.get('type')}")
            print(f"Has metrics: {'metrics' in first_node}")
            if 'metrics' in first_node:
                print("Metrics structure:", json.dumps(first_node['metrics'], indent=2)[:500] + "...")
            
            print(f"Has logs: {'logs' in first_node}")
            if 'logs' in first_node:
                print("Logs structure:", json.dumps(first_node['logs'], indent=2)[:500] + "...")
        
        # Check first edge for dynamics
        if data.get('edges'):
            first_edge = data['edges'][0]
            print("\nFirst edge structure:")
            print(f"Edge type: {first_edge.get('type')}")
            print(f"Has dynamics: {'dynamics' in first_edge}")
            if 'dynamics' in first_edge:
                print("Dynamics structure:", json.dumps(first_edge['dynamics'], indent=2)[:500] + "...")
        
        return data
    
    def _process_node_metrics(self, node):
        """Process metrics for a single node"""
        metrics_data = defaultdict(list)
        node_id = node.get('id')
        
        try:
            # Check if metrics exists and has the correct structure
            if 'metrics' in node and isinstance(node['metrics'], dict):
                metrics_dict = node['metrics'].get('metrics_data', {})
                if node_id in metrics_dict:
                    for entry in metrics_dict[node_id]:
                        timestamp = entry.get('timestamp')
                        values = entry.get('values', {})
                        for metric_name, value in values.items():
                            metrics_data[metric_name].append(value)
                            self.all_timestamps.add(timestamp)
            
            return {
                'node_id': node_id,
                'node_type': node.get('type'),
                'metrics_data': dict(metrics_data),
            }
        except Exception as e:
            print(f"Error processing metrics for node {node_id}: {str(e)}")
            return None
    
    def _process_node_logs(self, node):
        """Process logs for a single node"""
        try:
            if 'logs' not in node or not isinstance(node['logs'], dict):
                return None
            
            node_type = node['type']
            logs_data = defaultdict(list)
            timestamps = set()
            
            # Check if logs exists and has the correct structure
            if 'logs' in node and isinstance(node['logs'], dict):
                log_data_list = node['logs'].get('log_data', {}).get(node['id'], [])
                if isinstance(log_data_list, list):
                    for log_data in log_data_list:
                        if isinstance(log_data, dict):
                            timestamps.add(log_data.get('timestamp', ''))
                            for status_name, value in log_data.get('status', {}).items():
                                logs_data[status_name].append(value)
            
            if not logs_data:
                return None
            
            return {
                'node_type': node_type,
                'logs_data': dict(logs_data),
                'timestamps': list(timestamps)
            }
        except Exception as e:
            print(f"Error processing logs for node {node.get('id', 'unknown')}: {str(e)}")
            return None
    
    def _process_edge_dynamics(self, edge):
        """Process dynamics for a single edge"""
        try:
            if 'dynamics' not in edge or not isinstance(edge['dynamics'], dict):
                return None
            
            edge_type = edge.get('type', '')
            source_id = edge.get('source', '')
            target_id = edge.get('target', '')
            dynamics_data = defaultdict(lambda: defaultdict(list))
            state_changes = []
            timestamps = set()
            
            # Check if dynamics exists and has the correct structure
            if isinstance(edge['dynamics'], dict):
                propagation_data = edge['dynamics'].get('propagation_data', [])
                if isinstance(propagation_data, list):
                    prev_states = {}  # Track previous states for each effect
                    
                    for data in propagation_data:
                        if isinstance(data, dict):
                            timestamp = data.get('timestamp', '')
                            timestamps.add(timestamp)
                            
                            # Process effects and detect state changes
                            for effect_name, effect_data in data.get('effects', {}).items():
                                if isinstance(effect_data, dict):
                                    curr_state = effect_data.get('source_status')
                                    prev_state = prev_states.get(effect_name)
                                    
                                    # Record state change if detected
                                    if prev_state is not None and curr_state != prev_state:
                                        state_changes.append({
                                            'timestamp': timestamp,
                                            'edge_type': edge_type,
                                            'source': source_id,
                                            'target': target_id,
                                            'effect': effect_name,
                                            'from_state': prev_state,
                                            'to_state': curr_state
                                        })
                                    
                                    # Update state tracking
                                    prev_states[effect_name] = curr_state
                                    
                                    # Record probability and state
                                    dynamics_data['probability'][effect_name].append(
                                        effect_data.get('probability', 0))
                                    dynamics_data['source_status'][effect_name].append(curr_state)
            
            if not dynamics_data and not state_changes:
                return None
            
            return {
                'edge_id': f"{source_id}->{target_id}",
                'edge_type': edge_type,
                'dynamics_data': dict(dynamics_data),
                'state_changes': state_changes,
                'timestamps': list(timestamps)
            }
        except Exception as e:
            print(f"Error processing dynamics for edge {edge.get('source', '')}->{edge.get('target', '')}: {str(e)}")
            return None
    
    def analyze_node_metrics(self):
        """Analyze node metrics"""
        print("\nAnalyzing node metrics...")
        
        # Initialize data structures
        metrics_by_type = defaultdict(lambda: defaultdict(list))
        self.all_timestamps = set()
        
        # Process metrics for all nodes in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_node_metrics, node): node 
                      for node in self.graph_data['nodes']}
            
            with tqdm(total=len(futures), desc="Processing metrics") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        node_type = result['node_type']
                        for metric_name, values in result['metrics_data'].items():
                            metrics_by_type[node_type][metric_name].extend(values)
                pbar.update(1)
        
        # Generate statistics and plots for each node type
        node_types = list(metrics_by_type.keys())
        for node_type in node_types:
            metrics = metrics_by_type[node_type]
            if not metrics:
                continue
            
            # Create subplots for this node type
            n_metrics = len(metrics)
            if n_metrics == 0:
                continue
            
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            plt.figure(figsize=(6*n_cols, 4*n_rows))
            fig = plt.gcf()
            fig.suptitle(f'Metrics Distribution - {node_type}', fontsize=16, y=1.02)
            
            for idx, (metric_name, values) in enumerate(metrics.items(), 1):
                plt.subplot(n_rows, n_cols, idx)
                sns.histplot(values, kde=True)
                plt.title(f'{metric_name} Distribution')
                plt.xlabel(metric_name)
                plt.ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{node_type}_metrics_distribution.png'))
            plt.close()
        
        print(f"Found {len(self.all_timestamps)} unique timestamps in metrics data")
        
        # Save statistics to JSON
        stats = {
            'total_timestamps': len(self.all_timestamps),
            'metrics_by_type': {
                node_type: {
                    metric_name: {
                        'min': float(min(values)),
                        'max': float(max(values)),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    }
                    for metric_name, values in metrics.items()
                }
                for node_type, metrics in metrics_by_type.items()
            }
        }
        
        with open(os.path.join(self.output_dir, 'metrics_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    def analyze_node_logs(self):
        """Analyze node logs"""
        print("\nAnalyzing node logs...")
        
        # Use thread pool to process nodes
        logs_by_type = defaultdict(lambda: defaultdict(list))
        all_timestamps = set()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_node_logs, node): node 
                      for node in self.graph_data['nodes']}
            
            with tqdm(total=len(futures), desc="Processing logs") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        node_type = result['node_type']
                        for status_name, values in result['logs_data'].items():
                            logs_by_type[node_type][status_name].extend(values)
                        all_timestamps.update(result['timestamps'])
                    pbar.update(1)
        
        # Create multi-subplot figures for different node types
        with plot_lock:
            node_types = list(logs_by_type.keys())
            for node_type in node_types:
                statuses = logs_by_type[node_type]
                if not statuses:
                    continue
                
                n_statuses = len(statuses)
                if n_statuses == 0:
                    continue
                
                # Calculate subplot layout
                n_cols = min(3, n_statuses)
                n_rows = (n_statuses + n_cols - 1) // n_cols
                
                fig = plt.figure(figsize=(15, 5 * n_rows))
                fig.suptitle(f'Log Status Distribution - {node_type}', fontsize=16, y=1.02)
                
                for idx, (status_name, values) in enumerate(statuses.items(), 1):
                    plt.subplot(n_rows, n_cols, idx)
                    sns.histplot(values, kde=True)
                    plt.title(f'{status_name}')
                    plt.xlabel('Status Value')
                    plt.ylabel('Count')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{node_type}_logs_distribution.png'))
                plt.close()
        
        print(f"Found {len(all_timestamps)} unique timestamps in log data")
        
        # Save statistics
        stats = {
            'total_timestamps': len(all_timestamps),
            'logs_by_type': {
                node_type: {
                    status_name: {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
                    for status_name, values in statuses.items()
                }
                for node_type, statuses in logs_by_type.items()
            }
        }
        
        with open(os.path.join(self.output_dir, 'logs_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    def analyze_edge_dynamics(self):
        """Analyze edge dynamics"""
        print("\nAnalyzing edge dynamics...")
        
        # Use thread pool to process edges
        dynamics_data = defaultdict(lambda: defaultdict(list))
        state_changes = []
        all_timestamps = set()
        edges_with_dynamics = 0
        total_edges = len(self.graph_data['edges'])
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_edge_dynamics, edge): edge 
                      for edge in self.graph_data['edges']}
            
            with tqdm(total=len(futures), desc="Processing edge dynamics") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        edges_with_dynamics += 1
                        for effect_type, effect_data in result['dynamics_data'].items():
                            for effect_name, values in effect_data.items():
                                dynamics_data[effect_type][effect_name].extend(values)
                        
                        state_changes.extend(result['state_changes'])
                        all_timestamps.update(result['timestamps'])
                    pbar.update(1)
        
        print(f"Edge dynamics coverage: {edges_with_dynamics/total_edges*100:.2f}% ({edges_with_dynamics}/{total_edges})")
        print(f"Found {len(all_timestamps)} unique timestamps in dynamics data")
        print(f"Total state changes: {len(state_changes)}")
        
        # Convert state changes to DataFrame for analysis
        if state_changes:
            df = pd.DataFrame(state_changes)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create state change visualization
            with plot_lock:
                plt.figure(figsize=(20, 15))
                gs = GridSpec(3, 2, figure=plt.gcf())
                
                # 1. State transitions by edge type
                ax1 = plt.subplot(gs[0, :])
                transitions = df.groupby(['edge_type', 'from_state', 'to_state']).size().unstack(fill_value=0)
                transitions.plot(kind='bar', stacked=True, ax=ax1)
                ax1.set_title('State Transitions by Edge Type')
                ax1.set_xlabel('Edge Type')
                ax1.set_ylabel('Number of Transitions')
                plt.xticks(rotation=45)
                
                # 2. State changes over time
                ax2 = plt.subplot(gs[1, :])
                changes_over_time = df.groupby(['timestamp', 'to_state']).size().unstack(fill_value=0)
                changes_over_time.plot(kind='line', marker='o', ax=ax2)
                ax2.set_title('State Changes Over Time')
                ax2.set_xlabel('Timestamp')
                ax2.set_ylabel('Number of Changes')
                plt.xticks(rotation=45)
                
                # 3. State transition matrix heatmap
                ax3 = plt.subplot(gs[2, 0])
                transition_matrix = pd.crosstab(df['from_state'], df['to_state'])
                sns.heatmap(transition_matrix, annot=True, fmt='.0f', ax=ax3)
                ax3.set_title('State Transition Matrix')
                ax3.set_xlabel('To State')
                ax3.set_ylabel('From State')
                
                # 4. Edge type distribution
                ax4 = plt.subplot(gs[2, 1])
                edge_type_counts = df['edge_type'].value_counts()
                sns.barplot(x=edge_type_counts.index, y=edge_type_counts.values, ax=ax4)
                ax4.set_title('Edge Types with State Changes')
                ax4.set_xlabel('Edge Type')
                ax4.set_ylabel('Number of Changes')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'state_changes_analysis.png'))
                plt.close()
                
                # Create state change pattern analysis
                plt.figure(figsize=(15, 8))
                state_patterns = df.groupby(['edge_type', 'from_state', 'to_state']).size().reset_index(name='count')
                pivot_patterns = state_patterns.pivot_table(
                    values='count',
                    index=['edge_type'],
                    columns=['from_state', 'to_state'],
                    fill_value=0
                )
                sns.heatmap(pivot_patterns, annot=True, fmt='.0f')
                plt.title('State Change Patterns by Edge Type')
                plt.xlabel('State Transition (From->To)')
                plt.ylabel('Edge Type')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'state_change_patterns.png'))
                plt.close()
                
                # Print summary statistics
                print("\nState change summary:")
                print(f"Number of unique edge types: {len(df['edge_type'].unique())}")
                print(f"Number of unique states: {len(df['to_state'].unique())}")
                print("\nMost common state transitions:")
                top_transitions = (
                    df.groupby(['from_state', 'to_state'])
                    .size()
                    .sort_values(ascending=False)
                    .head(5)
                )
                print(top_transitions)
                
                print("\nMost active edge types:")
                top_edges = (
                    df['edge_type']
                    .value_counts()
                    .head(5)
                )
                print(top_edges)
            
            # Save detailed statistics
            stats = {
                'total_edges': total_edges,
                'edges_with_dynamics': edges_with_dynamics,
                'coverage_rate': edges_with_dynamics/total_edges,
                'total_timestamps': len(all_timestamps),
                'total_state_changes': len(state_changes),
                'state_transitions': {
                    f"{edge_type}_{from_state}_{to_state}": count
                    for (edge_type, from_state, to_state), count in (
                        df.groupby(['edge_type', 'from_state', 'to_state'])
                        .size()
                        .to_dict()
                        .items()
                    )
                } if state_changes else {},
                'state_stats': {
                    'by_edge_type': {
                        f"{edge_type}_{state}": count
                        for (edge_type, state), count in (
                            df.groupby('edge_type')['to_state']
                            .value_counts()
                            .to_dict()
                            .items()
                        )
                    } if state_changes else {},
                    'transition_matrix': {
                        str(from_state): {
                            str(to_state): int(count)
                            for to_state, count in to_states.items()
                        }
                        for from_state, to_states in (
                            pd.crosstab(df['from_state'], df['to_state'])
                            .to_dict('index')
                            .items()
                        )
                    } if state_changes else {},
                    'temporal_patterns': {
                        'hourly': {
                            f"{hour}_{state}": int(count)
                            for (hour, state), count in (
                                df.groupby([df['timestamp'].dt.hour, 'to_state'])
                                .size()
                                .to_dict()
                                .items()
                            )
                        } if state_changes else {},
                        'daily': {
                            f"{date.strftime('%Y-%m-%d')}_{state}": int(count)
                            for (date, state), count in (
                                df.groupby([df['timestamp'].dt.date, 'to_state'])
                                .size()
                                .to_dict()
                                .items()
                            )
                        } if state_changes else {}
                    }
                }
            }
            
            with open(os.path.join(self.output_dir, 'edge_dynamics_stats.json'), 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
    
    def verify_all(self):
        """Run all verifications"""
        print("Starting graph data verification...")
        self.analyze_node_metrics()
        self.analyze_node_logs()
        self.analyze_edge_dynamics()
        print("\nVerification complete, results saved in:", self.output_dir)

def main():
    """Main function"""
    # Set input and output paths
    graph_data_path = 'datasets/processed/graph_data_complete.json'
    output_dir = 'datasets/verification'
    
    # Create verifier and run verification
    verifier = GraphDataVerifier(graph_data_path, output_dir)
    verifier.verify_all()

if __name__ == '__main__':
    main() 