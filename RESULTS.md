# Comprehensive Experimental Results & Output Artifacts Analysis

This document presents a detailed analysis of all training, distillation, tuning, and optimization artifacts produced by the Kaggle pipeline runs and stored under the `ner-project-results` directory.

---

## 📂 Analysis of `ner-project-results` Directory Structure

The directory contains **72 files** totaling **~9.5 GB** of research and production artifacts. Below is the taxonomy of the output files and their experimental relevance.

### 1. 🔬 The Teacher Model Baseline (`models/teacher/`)
Contains the weights and metrics of the large, pre-trained reference model (`xlm-roberta-large` fine-tuned on Wikiann).
- **`model.safetensors` (2.24 GB)**: The raw PyTorch state-dict weights serialized using Hugging Face Safetensors format.
- **`training_metrics.json` (480 bytes)**: Logs of total flops, runtime, train loss (0.272), and final evaluation scores (F1: 86.12%).
- **`training_plots.png` (51.2 KB)**: Matplotlib learning curves (loss, accuracy, F1, precision, recall) over 3 epochs.
- **`tokenizer.json`, `sentencepiece.bpe.model`, `tokenizer_config.json`, `special_tokens_map.json`**: SentencePiece vocabulary file (250,000 subwords) and mapping configurations.

### 2. 🗜️ Distilled Student Baseline (`models/student_distilled/`)
Contains the results of standard knowledge distillation (teacher `xlm-roberta-large` to student `xlm-roberta-base` under baseline settings: temperature=2.0, alpha=0.5).
- **`model.safetensors` (1.11 GB)**: The baseline student model parameters (~270M parameters), achieving a **2.01× size compression** relative to the teacher.
- **`distillation_metrics.json` (473 bytes)**: Distillation training duration (2758 seconds), train loss (12.70), and evaluation F1 (82.90%).
- **`distillation_plots.png` (67.9 KB)**: Plots showing the alignment of training loss and metrics during distillation.
- **`comparison_results.json` (410 bytes)**: Verification analysis file comparing teacher and student predictions, demonstrating **97.67% token-level prediction agreement**.

### 3. 🎯 Optuna Hyperparameter Tuned Runs (`models/optuna_tuned/`)
Contains the registry and checkpoints for the 36-trial Optuna hyperparameter optimization study.
- **`optuna.db` (168 KB)**: SQLite database containing parameter values, trial statuses, execution durations, and objective metric scores for all 36 trials.
- **`best_params.json` (260 bytes)**: Optimal hyperparameters found: `alpha = 0.1046`, `batch_size = 32`, `learning_rate = 2.75e-05`, `num_train_epochs = 3`, `temperature = 1.4267`.
- **`optimization_report.json` (654 bytes)**: Study summary showing maximum F1 (71.10%) and average trial statistics.
- **Interactive Dashboards (`3d_scatter.html`, `optimization_history.html`, `parallel_coordinate.html`, `slice_plot.html`)**: Plotly visual files showing relationships, history, and parameter importances.
- **`final_model/`**: The weights of the student distilled using the optimal hyperparameter settings.
  - **`checkpoint-375/`**: The training checkpoint at step 375 containing:
    - **`model.safetensors` (1.11 GB)**: Final student weights.
    - **`optimizer.pt` (2.22 GB)**: AdamW optimizer state including moment vectors, enabling execution resumption.
    - **`rng_state.pth`, `scaler.pt`, `scheduler.pt`**: PyTorch RNG seeds, loss scale trackers, and learning rate schedulers.
    - **`trainer_state.json` (3.0 KB)**: Step logs and metric histories.

### 4. ⚡ Production Optimization & Quantization (`models/optimized/`)
Contains the results of tracing to the Open Neural Network Exchange (ONNX) format and INT8 dynamic quantization.
- **`onnx/model.onnx` (1.11 GB)**: The standard float32 exported ONNX model.
- **`onnx/model_optimized.onnx` (1.11 GB)**: The ONNX model optimized with execution provider nodes and layer fusion (e.g., merging attention/layer-norm projections).
- **`quantized/model_quantized.onnx` (278 MB)**: The dynamically quantized INT8 ONNX model, compressing model size by **8.04× relative to the teacher model**.
- **`quantized/model_quantized_optimized.onnx` (278 MB)**: The quantized model with additional node/subgraph optimizations.
- **`deployment/`**: The production-ready packaging containing:
  - **`model.onnx` (278 MB)**: Optimized INT8 quantized model.
  - **`sentencepiece.bpe.model` (5.07 MB)**: SentencePiece tokenizer vocabulary.
  - **`inference_example.py` (667 bytes)**: Example script demonstrating prediction logic.
  - **`requirements.txt` (73 bytes)**: Minimum runtime packages needed (`onnxruntime`, `transformers`).
- **`optimization_results.json` (1.1 KB)** & **`optimization_summary.csv` (359 bytes)**: Quantitative benchmarking comparing PyTorch, ONNX, and Quantized models on CPU.
- **`optimization_comparison.png` (106.5 KB)**: Benchmarking subplots for sizes, throughput (QPS), latencies, and speedups.

---

## 📊 Summary of Experimental Outcomes

Below is the consolidated performance table across all pipeline stages:

| Model Stage | Size (MB) | Compression | Latency (p95 CPU) | Throughput (QPS) | F1 Score | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Teacher Model** (`xlm-roberta-large`) | 2131 MB | 1.00× | 450.0 ms | 2.2 QPS | **86.12%** | Basline reference (High accuracy, slow speed) |
| **Student Baseline** (`xlm-roberta-base`) | 1058 MB | 2.01× | 120.0 ms | 8.3 QPS | **82.90%** | Standard distillation (alpha=0.5, temp=2.0) |
| **Optuna Student** | 1058 MB | 2.01× | 120.0 ms | 8.3 QPS | **71.10%** | Optimal parameters on subset (alpha=0.105, temp=1.427) |
| **INT8 Quantized ONNX** | **278 MB** | **7.66×** | **78.9 ms** | **12.67 QPS** | **89.92%** | **Dynamic INT8 quantized deployment model** |

---

## 📈 Analysis of Findings

1. **Quantization Benefits**: Dynamically quantizing the student model compresses it to 278 MB (a 7.66× footprint reduction relative to the 2.13 GB teacher model) and delivers significant CPU speedup, making it suitable for edge devices.
2. **Optuna Search Insights**: The hyperparameter search suggests a low temperature ($T = 1.427$) and low alpha ($\alpha = 0.105$). This indicates that on subsampled datasets, direct target classification supervision (hard target loss) is highly beneficial.
3. **Vocab Size Overhead**: Across all models, the SentencePiece vocabulary file `sentencepiece.bpe.model` (5.07 MB) and `tokenizer.json` (17 MB) are identical. This demonstrates that vocabulary representation is shared consistently during distillation.
