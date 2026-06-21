"""python run_all_modes.py --epochs 50 --runs 5 --gpu 0  (default: all datasets in processed_data/)"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
import argparse


def run_training(dataset, mode, epochs, runs, batch_size, gpu):
    """
    Run training for a specific mode and return results

    Args:
        dataset: Dataset name
        mode: Wavelet mode (continuous, implicit, gumbel)
        epochs: Number of epochs
        runs: Number of runs
        batch_size: Batch size
        gpu: GPU id

    Returns:
        dict: Results including metrics and learned scales
    """
    print(f"\n{'='*80}")
    print(f"Running {mode} mode on {dataset}")
    print(f"{'='*80}\n")

    # Build command
    model_name = f'TFWaveFormer{mode.capitalize()}'
    cmd = [
        sys.executable,
        "train_link_prediction.py",
        "--dataset_name", dataset,
        "--model_name", model_name,
        "--num_epochs", str(epochs),
        "--num_runs", str(runs),
        "--batch_size", str(batch_size),
        "--gpu", str(gpu),
        "--load_best_configs"
    ]

    start_time = time.time()

    try:
        # Run training with real-time output
        print(f"Command: {' '.join(cmd)}\n")

        # Use Popen for real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Capture output while showing it in real-time
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # Show real-time
            output_lines.append(line)  # Save for parsing

        process.wait(timeout=7200)
        output = ''.join(output_lines)
        elapsed_time = time.time() - start_time

        # Read metrics from saved JSON
        result_file = f"./saved_results/{model_name}/{dataset}/{model_name}_seed0.json"
        metrics = {}
        scales = []
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                saved = json.load(f)
                for k, v in saved.items():
                    if k == 'learned_scales':
                        scales = v
                    elif isinstance(v, dict):
                        metrics[k] = v
        else:
            # fallback: parse from stdout
            metrics = parse_metrics_from_output(output)
            scales = parse_scales_from_output(output)

        return {
            "status": "success",
            "mode": mode,
            "dataset": dataset,
            "epochs": epochs,
            "runs": runs,
            "elapsed_time": elapsed_time,
            "metrics": metrics,
            "learned_scales": scales,
            "timestamp": datetime.now().isoformat()
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "mode": mode,
            "dataset": dataset,
            "elapsed_time": time.time() - start_time,
            "error": "Training timeout after 2 hours"
        }
    except Exception as e:
        return {
            "status": "error",
            "mode": mode,
            "dataset": dataset,
            "elapsed_time": time.time() - start_time,
            "error": str(e)
        }


def parse_metrics_from_output(output):
    """Parse metrics from training output"""
    metrics = {}

    # Parse final metrics
    lines = output.split('\n')
    for line in lines:
        # Test AP/AUC
        if 'Test AP:' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'AP:':
                    metrics['test_ap_mean'] = float(parts[i+1].replace(',', ''))
                    if '±' in parts:
                        idx = parts.index('±')
                        metrics['test_ap_std'] = float(parts[idx+1])
                elif part == 'AUC:':
                    metrics['test_auc_mean'] = float(parts[i+1].replace(',', ''))
                    if '±' in parts[i:]:
                        for j, p in enumerate(parts[i:]):
                            if p == '±':
                                metrics['test_auc_std'] = float(parts[i+j+1])
                                break

        # New node test AP/AUC
        if 'New Node Test AP:' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'AP:':
                    metrics['new_node_test_ap_mean'] = float(parts[i+1].replace(',', ''))
                    if '±' in parts:
                        idx = parts.index('±')
                        metrics['new_node_test_ap_std'] = float(parts[idx+1])

        if 'New Node Test AUC:' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'AUC:':
                    metrics['new_node_test_auc_mean'] = float(parts[i+1].replace(',', ''))
                    if '±' in parts[i:]:
                        for j, p in enumerate(parts[i:]):
                            if p == '±':
                                metrics['new_node_test_auc_std'] = float(parts[i+j+1])
                                break

    return metrics


def parse_scales_from_output(output):
    """Parse learned scales from training output"""
    scales = []

    lines = output.split('\n')
    capturing = False

    for line in lines:
        if 'Learned scales per layer:' in line:
            capturing = True
            continue

        if capturing:
            # Parse line like: "  Layer 0: mean=6.45, std=1.23, range=[4.21, 9.87]"
            if 'Layer' in line and 'mean=' in line:
                try:
                    parts = line.split(':')
                    layer_num = int(parts[0].strip().split()[1])

                    # Parse statistics
                    stats_str = parts[1]
                    mean = float(stats_str.split('mean=')[1].split(',')[0])
                    std = float(stats_str.split('std=')[1].split(',')[0])
                    range_str = stats_str.split('range=')[1]
                    range_min = float(range_str.split('[')[1].split(',')[0])
                    range_max = float(range_str.split(',')[1].split(']')[0])

                    scales.append({
                        'layer': layer_num,
                        'mean': mean,
                        'std': std,
                        'min': range_min,
                        'max': range_max
                    })
                except:
                    pass
            elif line.strip() == '' or 'Run' in line:
                # Stop capturing when empty line or new section
                capturing = False

    return scales


def save_results(all_results, output_dir):
    """Save results to JSON files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save overall results
    overall_file = os.path.join(output_dir, "all_modes_results.json")
    with open(overall_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Saved overall results to: {overall_file}")

    # Save individual mode results
    for result in all_results['results']:
        mode = result['mode']
        dataset = result['dataset']
        mode_file = os.path.join(output_dir, f"{dataset}_{mode}_results.json")
        with open(mode_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✅ Saved {mode} results to: {mode_file}")


def print_summary(all_results):
    """Print summary of all results"""
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL RUNS")
    print(f"{'='*80}\n")

    dataset = all_results['dataset']
    print(f"Dataset: {dataset}")
    print(f"Total time: {all_results['total_time']:.2f} seconds ({all_results['total_time']/60:.2f} minutes)\n")

    # Print comparison table
    print(f"{'Mode':<12} {'Test AP':<15} {'Test AUC':<15} {'New Node AP':<15} {'Time (s)':<10} {'Status':<10}")
    print("-" * 90)

    for result in all_results['results']:
        mode = result['mode']
        status = result['status']
        elapsed = result['elapsed_time']

        if status == 'success' and 'metrics' in result:
            metrics = result['metrics']
            test_ap = f"{metrics.get('test_ap_mean', 0):.4f} ± {metrics.get('test_ap_std', 0):.4f}"
            test_auc = f"{metrics.get('test_auc_mean', 0):.4f} ± {metrics.get('test_auc_std', 0):.4f}"
            new_ap = f"{metrics.get('new_node_test_ap_mean', 0):.4f} ± {metrics.get('new_node_test_ap_std', 0):.4f}"

            print(f"{mode:<12} {test_ap:<15} {test_auc:<15} {new_ap:<15} {elapsed:<10.1f} {status:<10}")
        else:
            print(f"{mode:<12} {'N/A':<15} {'N/A':<15} {'N/A':<15} {elapsed:<10.1f} {status:<10}")

    print()

    # Print learned scales summary
    print(f"\n{'='*80}")
    print("LEARNED SCALES SUMMARY")
    print(f"{'='*80}\n")

    for result in all_results['results']:
        if result['status'] == 'success' and 'learned_scales' in result and result['learned_scales']:
            mode = result['mode']
            scales = result['learned_scales']

            print(f"{mode} mode:")
            for scale_info in scales:
                print(f"  Layer {scale_info['layer']}: mean={scale_info['mean']:.2f}, "
                      f"std={scale_info['std']:.2f}, range=[{scale_info['min']:.2f}, {scale_info['max']:.2f}]")
            print()


def main():
    parser = argparse.ArgumentParser(description='Run all adaptive wavelet modes')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Dataset names to run (default: all in processed_data/). Can specify multiple: --datasets wikipedia reddit')
    parser.add_argument('--modes', type=str, nargs='+',
                        default=['continuous', 'implicit', 'gumbel'],
                        help='Modes to run (default: all three)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs per run (default: 50)')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs per mode (default: 5)')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size (default: 200)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (default: 0)')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help='Output directory for results (default: comparison_results)')

    args = parser.parse_args()

    if args.datasets is None:
        data_dir = 'processed_data'
        args.datasets = sorted([d for d in os.listdir(data_dir)
                                if os.path.isdir(os.path.join(data_dir, d))])
        if not args.datasets:
            args.datasets = ['wikipedia']

    print(f"\n{'='*80}")
    print("AUTOMATED ADAPTIVE WAVELET MODE COMPARISON")
    print(f"{'='*80}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Modes: {', '.join(args.modes)}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Runs per mode: {args.runs}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPU: {args.gpu}")
    print(f"Output directory: {args.output_dir}")
    print(f"Total experiments: {len(args.datasets)} datasets × {len(args.modes)} modes = {len(args.datasets) * len(args.modes)}")
    print(f"{'='*80}\n")

    # Confirm before starting
    confirm = input("Start training? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    # Run all datasets and modes
    global_start_time = time.time()
    all_datasets_results = []

    for dataset in args.datasets:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*80}\n")

        all_results = {
            'dataset': dataset,
            'epochs': args.epochs,
            'runs': args.runs,
            'batch_size': args.batch_size,
            'start_time': datetime.now().isoformat(),
            'results': []
        }

        dataset_start_time = time.time()

        for mode in args.modes:
            result = run_training(
                dataset=dataset,
                mode=mode,
                epochs=args.epochs,
                runs=args.runs,
                batch_size=args.batch_size,
                gpu=args.gpu
            )
            all_results['results'].append(result)

            # Save intermediate results after each mode
            all_results['total_time'] = time.time() - dataset_start_time
            output_dir = os.path.join(args.output_dir, dataset)
            save_results(all_results, output_dir)

        all_results['end_time'] = datetime.now().isoformat()
        all_results['total_time'] = time.time() - dataset_start_time

        # Save final results for this dataset
        output_dir = os.path.join(args.output_dir, dataset)
        save_results(all_results, output_dir)

        # Print summary for this dataset
        print_summary(all_results)

        all_datasets_results.append(all_results)

    # Save global summary
    global_summary = {
        'datasets': args.datasets,
        'modes': args.modes,
        'total_time': time.time() - global_start_time,
        'start_time': datetime.fromtimestamp(global_start_time).isoformat(),
        'end_time': datetime.now().isoformat(),
        'results_by_dataset': all_datasets_results
    }

    summary_file = os.path.join(args.output_dir, 'global_summary.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(global_summary, f, indent=2)

    print(f"\n{'='*80}")
    print("GLOBAL SUMMARY")
    print(f"{'='*80}")
    print(f"Datasets processed: {len(args.datasets)}")
    print(f"Modes per dataset: {len(args.modes)}")
    print(f"Total experiments: {len(args.datasets) * len(args.modes)}")
    print(f"Total time: {global_summary['total_time']/60:.2f} minutes ({global_summary['total_time']/3600:.2f} hours)")
    print(f"{'='*80}\n")

    print(f"\n✅ All runs completed!")
    print(f"📁 Results saved to: {args.output_dir}/")
    for dataset in args.datasets:
        print(f"   - {dataset}/ (contains all mode results)")
    print(f"   - global_summary.json (overall summary)")
    print(f"⏱️  Total time: {global_summary['total_time']/60:.2f} minutes\n")


if __name__ == "__main__":
    main()
