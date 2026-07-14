# Experimental Setup and Hyperparameter Runs

This document details the configuration, parameters, and experimental setup for all model training, distillation, and tuning experiments.

---

## 🧪 Experimental Roadmap

We systematically conduct four core experiments:

```
[Teacher Fine-tuning] ──> [Student Distillation] ──> [Optuna Tuning] ──> [ONNX INT8 Quantization]
```

---

## 📝 Experiment Configurations

### 1. Teacher Fine-tuning (`xlm-roberta-large`)
* **Objective**: Train a high-performing multilingual reference model.
* **Dataset**: Hugging Face `unimelb-nlp/wikiann` (Languages: `en`, `de`, `fr`).
* **Hyperparameters**:
  * **Learning Rate**: $2 \times 10^{-5}$
  * **Batch Size**: 16
  * **Weight Decay**: 0.01
  * **Max Epochs**: 3
  * **Seed**: 42
  * **Optimizer**: AdamW
  * **Scheduler**: Linear with warmup (warmup ratio: 0.0)
  * **Precision**: FP16 (if GPU available)
* **W&B Log**: Enabled in offline/online mode.

### 2. Student Distillation Baseline (`xlm-roberta-base`)
* **Objective**: Distill knowledge from the teacher into a base student model.
* **Student Foundation**: `xlm-roberta-base`
* **Teacher Reference**: Path to the fine-tuned `xlm-roberta-large` model.
* **Distillation Parameters**:
  * **Temperature ($T$)**: 2.0 (Controls smoothing of logits)
  * **Alpha ($\alpha$)**: 0.5 (Weight of KL-divergence vs hard cross-entropy)
  * **Learning Rate**: $2 \times 10^{-5}$
  * **Batch Size**: 16
  * **Weight Decay**: 0.01
  * **Epochs**: 3

### 3. Hyperparameter Optimization Run (Optuna Study)
* **Objective**: Optimize distillation loss parameters and learning rates for best student F1 score.
* **Search Space**:
  * **Learning Rate**: Log-uniform range $[1 \times 10^{-5}, 5 \times 10^{-5}]$
  * **Batch Size**: Categorical $[8, 16, 32]$
  * **Epochs**: Integer range $[2, 4]$
  * **Temperature ($T$)**: Uniform range $[1.0, 5.0]$
  * **Alpha ($\alpha$)**: Uniform range $[0.1, 0.9]$
* **Pruner**: Optuna MedianPruner (startup trials: 5, warmup steps: 10).
* **Database**: SQLite registry (`ner_optuna.db`) for process stability and resumption.

### 4. Production Benchmarking & Quantization
* **Objective**: Export student to ONNX format and execute INT8 dynamic quantization.
* **ONNX Config**:
  * **Execution Provider**: `CPUExecutionProvider`
  * **ONNX OP Set Version**: 14
  * **Input Names**: `input_ids`, `attention_mask`
  * **Dynamic Axes**: Sequence length and batch size.
* **INT8 Quantization Mode**: Dynamic quantization mapping weights to INT8.
* **Performance Validation**: Runs on CPU testing 100 sentences to report average latencies, p95 latencies, throughput (QPS), and file sizes.
