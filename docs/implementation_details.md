# Technical Implementation Details

This document provides mathematical formulations, code mappings, and initialization details.

---

## 🧮 Knowledge Distillation Formulation

The distillation process leverages a combined loss function to transfer representation characteristics from the teacher to the student.

```
Logits (Teacher) ──> Softmax (T) ──┐
                                   ├──> KL Divergence Loss ──┐
Logits (Student) ──> Softmax (T) ──┘                         │
                                                             ├──> Total Loss (alpha)
Ground Truth     ──────────────────> Cross Entropy Loss ─────┘
```

### Soft Target Loss (KL Divergence)
The softened probability distributions for student ($p_s$) and teacher ($p_t$) logits at temperature $T$ are:
$$p_i = \text{softmax}\left(\frac{z_i}{T}\right)$$

The soft loss is the Kullback-Leibler divergence scaled by $T^2$:
$$\mathcal{L}_{\text{soft}} = T^2 \sum \text{softmax}\left(\frac{z_t}{T}\right) \log \left( \frac{\text{softmax}(z_t / T)}{\text{softmax}(z_s / T)} \right)$$

### Hard Target Loss (Cross Entropy)
$$\mathcal{L}_{\text{hard}} = -\sum y \log(\text{softmax}(z_s))$$

### Total Combined Loss
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{soft}} + (1 - \alpha) \mathcal{L}_{\text{hard}}$$

---

## ⚡ ONNX Session Initialization Details

When loading the quantized model for production inference, we initialize `onnxruntime.InferenceSession` with optimized execution parameters:

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4  # Matches core allocation
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "models/optimized/deployment/model.onnx",
    sess_options=opts,
    providers=["CPUExecutionProvider"]
)
```

### Dynamic Axes Configuration
To support batch inference and variable sentence lengths:
```python
dynamic_axes = {
    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
    'logits': {0: 'batch_size', 1: 'sequence_length'}
}
```
This configuration avoids session re-initialization overhead when processing sentences of varying lengths.
