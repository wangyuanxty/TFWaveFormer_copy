"""
Comparison and visualization utilities for adaptive wavelet modes

Usage:
    # Compare all modes on a dataset
    python scripts/compare_adaptive_modes.py --dataset wikipedia

    # Visualize learned scales
    python scripts/compare_adaptive_modes.py --dataset wikipedia --visualize

Author: Research Implementation
Date: 2026-06-16
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_results(dataset_name, wavelet_modes=['continuous', 'hyper', 'implicit', 'gumbel']):
    """Load results for all adaptive modes"""
    results = {}

    for mode in wavelet_modes:
        result_file = f'./saved_results/AdaptiveTFWaveFormer_{mode}_{dataset_name}.json'
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results[mode] = json.load(f)
        else:
            print(f"Warning: {result_file} not found")

    # Also load original TFWaveFormer results if available
    original_file = f'./saved_results/TFWaveFormer_{dataset_name}.json'
    if os.path.exists(original_file):
        with open(original_file, 'r') as f:
            results['original'] = json.load(f)

    return results


def print_comparison_table(results, dataset_name):
    """Print comparison table"""
    print("\n" + "=" * 100)
    print(f"Adaptive Wavelet Mode Comparison on {dataset_name}")
    print("=" * 100)

    print(f"\n{'Mode':<20} {'Test AP':<20} {'Test AUC':<20} {'New Node AP':<20} {'New Node AUC':<20}")
    print("-" * 100)

    modes_order = ['original', 'continuous', 'hyper', 'implicit', 'gumbel']

    for mode in modes_order:
        if mode not in results:
            continue

        r = results[mode]
        test_ap = f"{r['test_ap']['mean']:.4f} ± {r['test_ap']['std']:.4f}"
        test_auc = f"{r['test_auc']['mean']:.4f} ± {r['test_auc']['std']:.4f}"
        new_node_ap = f"{r['new_node_test_ap']['mean']:.4f} ± {r['new_node_test_ap']['std']:.4f}"
        new_node_auc = f"{r['new_node_test_auc']['mean']:.4f} ± {r['new_node_test_auc']['std']:.4f}"

        print(f"{mode:<20} {test_ap:<20} {test_auc:<20} {new_node_ap:<20} {new_node_auc:<20}")

    print("=" * 100)

    # Highlight improvements
    if 'original' in results:
        print("\nImprovements over Original TFWaveFormer:")
        print("-" * 100)

        original_ap = results['original']['test_ap']['mean']

        for mode in ['continuous', 'hyper', 'implicit', 'gumbel']:
            if mode not in results:
                continue

            mode_ap = results[mode]['test_ap']['mean']
            improvement = ((mode_ap - original_ap) / original_ap) * 100

            symbol = "+" if improvement > 0 else ""
            print(f"{mode:<20} Test AP: {symbol}{improvement:.2f}%")

        print("=" * 100)


def plot_comparison(results, dataset_name, output_dir='./plots'):
    """Plot comparison bar chart"""
    os.makedirs(output_dir, exist_ok=True)

    modes = []
    test_ap_means = []
    test_ap_stds = []

    modes_order = ['original', 'continuous', 'hyper', 'implicit', 'gumbel']

    for mode in modes_order:
        if mode not in results:
            continue
        modes.append(mode)
        test_ap_means.append(results[mode]['test_ap']['mean'])
        test_ap_stds.append(results[mode]['test_ap']['std'])

    # Create bar plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(modes))
    bars = plt.bar(x, test_ap_means, yerr=test_ap_stds, capsize=5, alpha=0.7)

    # Color bars
    colors = ['gray', 'steelblue', 'orange', 'green', 'purple']
    for bar, color in zip(bars, colors[:len(bars)]):
        bar.set_color(color)

    plt.xlabel('Wavelet Mode', fontsize=12)
    plt.ylabel('Test AP', fontsize=12)
    plt.title(f'Adaptive Wavelet Mode Comparison on {dataset_name}', fontsize=14)
    plt.xticks(x, modes, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_file = os.path.join(output_dir, f'{dataset_name}_mode_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def visualize_learned_scales(dataset_name, mode='continuous', output_dir='./plots'):
    """Visualize learned scales from trained model"""
    import torch

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model_path = f'./saved_models/AdaptiveTFWaveFormer_{mode}/{dataset_name}/AdaptiveTFWaveFormer_{mode}_seed0_best_model.pkl'

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model = checkpoint['model'] if 'model' in checkpoint else checkpoint

        # Extract backbone
        if hasattr(model, 'module'):
            backbone = model.module[0]
        else:
            backbone = model[0]

        # Get learned scales
        num_layers = len(backbone.wavelet_transformers)

        fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))
        if num_layers == 1:
            axes = [axes]

        for layer_id in range(num_layers):
            wavelet_filter = backbone.wavelet_transformers[layer_id].wavelet_filter

            if hasattr(wavelet_filter, 'get_learned_scales'):
                scales = wavelet_filter.get_learned_scales()

                axes[layer_id].hist(scales, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                axes[layer_id].axvline(np.mean(scales), color='red', linestyle='--',
                                      linewidth=2, label=f'Mean: {np.mean(scales):.2f}')
                axes[layer_id].set_xlabel('Learned Scale', fontsize=12)
                axes[layer_id].set_ylabel('Number of Channels', fontsize=12)
                axes[layer_id].set_title(f'Layer {layer_id}', fontsize=14)
                axes[layer_id].legend()
                axes[layer_id].grid(axis='y', alpha=0.3)

        plt.suptitle(f'Learned Wavelet Scales - {dataset_name} ({mode} mode)', fontsize=16)
        plt.tight_layout()

        output_file = os.path.join(output_dir, f'{dataset_name}_{mode}_learned_scales.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Scale visualization saved to: {output_file}")
        plt.close()

    except Exception as e:
        print(f"Error loading model or visualizing scales: {e}")


def compare_across_datasets(datasets, mode='continuous', output_dir='./plots'):
    """Compare learned scales across multiple datasets"""
    os.makedirs(output_dir, exist_ok=True)

    import torch

    dataset_scales = {}

    for dataset in datasets:
        model_path = f'./saved_models/AdaptiveTFWaveFormer_{mode}/{dataset}/AdaptiveTFWaveFormer_{mode}_seed0_best_model.pkl'

        if not os.path.exists(model_path):
            print(f"Model not found for {dataset}, skipping...")
            continue

        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            model = checkpoint['model'] if 'model' in checkpoint else checkpoint

            if hasattr(model, 'module'):
                backbone = model.module[0]
            else:
                backbone = model[0]

            # Get scales from first layer
            wavelet_filter = backbone.wavelet_transformers[0].wavelet_filter
            if hasattr(wavelet_filter, 'get_learned_scales'):
                scales = wavelet_filter.get_learned_scales()
                dataset_scales[dataset] = scales

        except Exception as e:
            print(f"Error loading {dataset}: {e}")

    if not dataset_scales:
        print("No valid models found for comparison")
        return

    # Create violin plot
    fig, ax = plt.subplots(figsize=(12, 6))

    positions = []
    data_to_plot = []
    labels = []

    for i, (dataset, scales) in enumerate(dataset_scales.items()):
        positions.append(i)
        data_to_plot.append(scales)
        labels.append(dataset)

    parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)

    # Color violins
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Learned Scale', fontsize=12)
    ax.set_title(f'Learned Wavelet Scales Across Datasets ({mode} mode)', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_file = os.path.join(output_dir, f'scales_across_datasets_{mode}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nCross-dataset comparison saved to: {output_file}")
    plt.close()


def generate_latex_table(results, dataset_name):
    """Generate LaTeX table for paper"""
    print("\n" + "=" * 80)
    print("LaTeX Table (copy to paper):")
    print("=" * 80)

    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Adaptive Wavelet Mode Comparison on " + dataset_name.capitalize() + "}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Mode & Test AP & Test AUC & New Node AP & New Node AUC \\\\")
    print("\\midrule")

    modes_order = ['original', 'continuous', 'hyper', 'implicit', 'gumbel']
    mode_names = {
        'original': 'Original (Fixed)',
        'continuous': 'Continuous',
        'hyper': 'HyperNet',
        'implicit': 'Implicit',
        'gumbel': 'Gumbel-Softmax'
    }

    for mode in modes_order:
        if mode not in results:
            continue

        r = results[mode]
        name = mode_names[mode]
        test_ap = f"{r['test_ap']['mean']:.3f} $\\pm$ {r['test_ap']['std']:.3f}"
        test_auc = f"{r['test_auc']['mean']:.3f} $\\pm$ {r['test_auc']['std']:.3f}"
        new_node_ap = f"{r['new_node_test_ap']['mean']:.3f} $\\pm$ {r['new_node_test_ap']['std']:.3f}"
        new_node_auc = f"{r['new_node_test_auc']['mean']:.3f} $\\pm$ {r['new_node_test_auc']['std']:.3f}"

        print(f"{name} & {test_ap} & {test_auc} & {new_node_ap} & {new_node_auc} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare adaptive wavelet modes')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--visualize', action='store_true', help='Visualize learned scales')
    parser.add_argument('--cross_dataset', nargs='+', help='Compare across multiple datasets')
    parser.add_argument('--latex', action='store_true', help='Generate LaTeX table')
    parser.add_argument('--output_dir', type=str, default='./plots', help='Output directory for plots')

    args = parser.parse_args()

    # Load and compare results
    results = load_results(args.dataset)

    if results:
        print_comparison_table(results, args.dataset)
        plot_comparison(results, args.dataset, args.output_dir)

        if args.latex:
            generate_latex_table(results, args.dataset)
    else:
        print(f"No results found for dataset: {args.dataset}")

    # Visualize learned scales
    if args.visualize:
        for mode in ['continuous', 'hyper', 'implicit', 'gumbel']:
            print(f"\nVisualizing {mode} mode...")
            visualize_learned_scales(args.dataset, mode, args.output_dir)

    # Cross-dataset comparison
    if args.cross_dataset:
        print(f"\nComparing across datasets: {args.cross_dataset}")
        compare_across_datasets(args.cross_dataset, mode='continuous', output_dir=args.output_dir)
