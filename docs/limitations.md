# Pipeline Limitations and Assumptions

This document outlines the limitations and assumptions of the current Multilingual NER pipeline.

---

## ⚠️ Key Limitations

### 1. Hardware-Bound Throughput
* **Issue**: The optimized INT8 ONNX model achieves ~40 QPS on local CPU hardware. However, this throughput is CPU-core bounded.
* **Tradeoff**: While Dynamic Quantization is highly optimized for CPUs, massive multi-tenant production traffic still requires horizontal scaling or GPU acceleration (using TensorRT/CUDA runtimes).

### 2. Information Loss from Quantization
* **Issue**: Scaling float32 weights to INT8 parameters introduces quantization noise.
* **F1 Drop**: This results in a ~0.8% absolute F1 degradation. For safety-critical domains (e.g., medical clinical notes), even minor accuracy loss may be unacceptable.

### 3. Cross-Lingual Generalization Cap
* **Issue**: The distilled model is trained on Germanic and Romance languages (`en`, `de`, `fr`). Zero-shot transfer to languages with different scripts (e.g., Cyrillic in Russian `ru` or Arabic) shows a larger performance drop compared to supervised in-language models.

### 4. Sentence Truncation Limits
* **Issue**: The configuration enforces a `MAX_LENGTH` of 128 tokens.
* **Impact**: Document paragraphs that exceed this limit are truncated, resulting in loss of entity contexts and boundary errors at the tail end.

---

## 📋 Architectural Assumptions

1. **Static Vocabulary**: Assumes that the SentencePiece vocabulary of 250,000 tokens in `xlm-roberta-base` covers all target languages without introducing excessive out-of-vocabulary (`<unk>`) splits.
2. **Homogeneous Domain**: Assumes training data (Wikipedia-based WikiANN) generalizes to corporate text, customer support requests, or chat messages. Domain adaptation training is required if applying to highly specialized domains (e.g., legal contracts).
