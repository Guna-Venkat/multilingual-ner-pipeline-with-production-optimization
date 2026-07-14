# Evaluation Methodology and Metrics

This document details the metrics used to evaluate Named Entity Recognition (NER) quality and model optimization performance.

---

## 📈 Model Quality (NER) Metrics

We use the standard **seqeval** framework, which evaluates token classification tasks at the **entity span level** (rather than individual token classification accuracy). 

### 1. Precision
The ratio of correctly predicted entity spans to the total predicted entity spans.
$$\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$$

### 2. Recall
The ratio of correctly predicted entity spans to the total actual ground-truth entity spans.
$$\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$$

### 3. F1-Score
The harmonic mean of precision and recall.
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 4. Accuracy
Token-level classification accuracy, computed only on valid tokens (ignoring subword padding tokens marked with `-100`).

---

## ⚡ Production Optimization Metrics

We benchmark models across four performance axes:

### 1. Model Footprint (Size)
Measured in megabytes (MB). Compressing the model reduces disk and memory overhead, allowing deployment on low-cost edge instances.

### 2. Inference Latency (p95)
Measured in milliseconds (ms) per sentence. The 95th percentile latency (p95) represents the maximum response time for 95% of incoming queries, which is critical for meeting strict Service Level Agreements (SLAs).

### 3. Throughput (Queries Per Second)
Measured in QPS (number of classification requests handled per second).
$$\text{Throughput} = \frac{\text{Total Requests}}{\text{Total Execution Time (seconds)}}$$

### 4. Compression Ratio
Shows the relative size reduction.
$$\text{Compression} = \frac{\text{FP32 Teacher Model Size}}{\text{Quantized Model Size}}$$
