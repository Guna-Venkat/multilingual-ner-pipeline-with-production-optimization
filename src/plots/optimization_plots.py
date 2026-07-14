import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from src.configs.config import OptimizationConfig

def create_optimization_visualization(results: Dict[str, Any], config: OptimizationConfig) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Generate subplots comparing latencies, model sizes, throughput, and speedup factors,
    and save a summary CSV table.
    """
    models = ['PyTorch', 'ONNX', 'Quantized ONNX']
    latencies = [
        results['benchmark']['pytorch']['avg_latency_ms'],
        results['benchmark']['onnx']['avg_latency_ms'],
        results['benchmark']['quantized']['avg_latency_ms']
    ]
    sizes = [
        results['benchmark']['pytorch']['model_size_mb'],
        results['benchmark']['onnx']['model_size_mb'],
        results['benchmark']['quantized']['model_size_mb']
    ]
    throughput = [
        results['benchmark']['pytorch']['throughput_qps'],
        results['benchmark']['onnx']['throughput_qps'],
        results['benchmark']['quantized']['throughput_qps']
    ]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Latency comparison
    axes[0, 0].bar(models, latencies, color=['#4285F4', '#EA4335', '#34A853'])
    axes[0, 0].set_title('Inference Latency Comparison')
    axes[0, 0].set_ylabel('Latency (ms)')
    axes[0, 0].grid(True, alpha=0.3)
    for i, v in enumerate(latencies):
        axes[0, 0].text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    # Plot 2: Model size comparison
    axes[0, 1].bar(models, sizes, color=['#4285F4', '#EA4335', '#34A853'])
    axes[0, 1].set_title('Model Size Comparison')
    axes[0, 1].set_ylabel('Size (MB)')
    axes[0, 1].grid(True, alpha=0.3)
    for i, v in enumerate(sizes):
        axes[0, 1].text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    # Plot 3: Throughput comparison
    axes[1, 0].bar(models, throughput, color=['#4285F4', '#EA4335', '#34A853'])
    axes[1, 0].set_title('Throughput Comparison')
    axes[1, 0].set_ylabel('Queries per Second')
    axes[1, 0].grid(True, alpha=0.3)
    for i, v in enumerate(throughput):
        axes[1, 0].text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    # Plot 4: Speedup comparison
    speedup = [1.0, latencies[0]/latencies[1], latencies[0]/latencies[2]]
    axes[1, 1].bar(models, speedup, color=['#4285F4', '#EA4335', '#34A853'])
    axes[1, 1].set_title('Speedup vs PyTorch')
    axes[1, 1].set_ylabel('Speedup (x)')
    axes[1, 1].grid(True, alpha=0.3)
    for i, v in enumerate(speedup):
        axes[1, 1].text(i, v, f'{v:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(config.OUTPUT_DIR, "optimization_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create summary table
    summary_df = pd.DataFrame({
        'Model': models,
        'Size (MB)': sizes,
        'Latency (ms)': latencies,
        'Throughput (QPS)': throughput,
        'Speedup': speedup,
        'Accuracy': [
            results['accuracy']['original_accuracy'],
            results['accuracy']['onnx_accuracy'],
            results['accuracy']['quantized_accuracy']
        ]
    })
    
    print("\nOptimization Summary Table:")
    print("-"*80)
    print(summary_df.to_string(index=False))
    print("-"*80)
    
    # Save table
    csv_path = os.path.join(config.OUTPUT_DIR, "optimization_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    
    return fig, summary_df
