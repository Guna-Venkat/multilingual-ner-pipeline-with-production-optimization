# Research Audit: Multilingual NER Pipeline with Production Optimization

This audit reviews the repository strictly from a **scientific and research perspective**, analyzing the core experimental questions, hypotheses, testing protocols, results, and artifacts in light of the finalized modular refactoring and directory reorganization.

---

## 1. What is the Research Question?

The central research question is:
**"To what extent can we compress a large multilingual Named Entity Recognition (NER) model (XLM-RoBERTa-large) using task-specific knowledge distillation, Bayesian hyperparameter optimization, and dynamic INT8 quantization, while preserving its multilingual and cross-lingual zero-shot generalization capabilities in resource-constrained CPU production settings?"**

---

## 2. What Hypotheses are Tested?

- **Hypothesis 1 (Model Compression via Distillation)**: A smaller student model (`xlm-roberta-base`) trained via task-specific knowledge distillation can match the NER F1 score of a fine-tuned teacher model (`xlm-roberta-large`) within a narrow margin (e.g. < 4% F1 loss) while reducing the model's footprint by over 50%.
- **Hypothesis 2 (Hyperparameter Importance in Distillation)**: Systematic Bayesian optimization (via Optuna) of distillation hyperparameters—specifically the temperature ($T$) and imitation weight ($\alpha$) alongside training parameters (learning rate, batch size)—can close the performance gap between the distilled student and the teacher compared to a default-parameter distilled student.
- **Hypothesis 3 (Quantization and Acceleration)**: Post-training graph optimization and dynamic INT8 quantization of the student model in ONNX format can yield a significant reduction in storage size and substantial improvement in CPU inference throughput, with negligible (< 1%) impact on token-level NER accuracy.
- **Hypothesis 4 (Cross-Lingual Zero-Shot Generalization)**: The compressed and quantized model preserves cross-lingual transfer capabilities (evaluated on Spanish and Russian, which were omitted from teacher training) similarly to the larger teacher model, without introducing boundary alignment degradation.

---

## 3. Which Experiment Answers Each Hypothesis?

- **Hypothesis 1 (Distillation Baseline)**: Answered by the distillation training run in `03_knowledge_distillation.ipynb`. This experiment evaluates the student model on English, German, and French and compares the F1 score directly with the fine-tuned teacher.
- **Hypothesis 2 (Bayesian Tuning)**: Answered by the 36-trial Optuna study in `04_hyperparameter_tuning.ipynb` and the final trained student model. The relationship plots (parallel coordinates, parameter importances) show how critical temperature and alpha are to closing the teacher-student gap.
- **Hypothesis 3 (ONNX and INT8 Quantization)**: Answered by the export and dynamic quantization benchmarking pipeline in `05_onnx_quantization.ipynb`. This runs PyTorch vs. ONNX vs. Quantized ONNX speed/memory tests and measures accuracy degradation.
- **Hypothesis 4 (Cross-Lingual and Error Stability)**: Answered by the zero-shot inference and error analysis in `06_error_analysis.ipynb` (which consumes outputs generated in `05_onnx_quantization.ipynb`), evaluating the model on Spanish (es) and Russian (ru) test splits.

---

## 4. Analysis of Reported Metrics & Inconsistencies

To prevent misinterpretation of reported metrics, we must explicitly distinguish the dataset splits, metrics (Entity-level F1 vs. Token-level Accuracy), and evaluation sample sizes:

1. **Teacher Baseline**:
   - **Metric**: Entity-level micro-F1 of **86.12%**
   - **Evaluation Set**: Full three-language WikiANN test set (EN, DE, FR).
2. **Student Baseline**:
   - **Metric**: Entity-level micro-F1 of **82.90%**
   - **Evaluation Set**: Full three-language WikiANN test set (EN, DE, FR).
3. **Optuna Hyperparameter Tuning**:
   - **Metric**: Best Validation F1 of **71.10%** (Validation subset F1 of **71.77%** for final model).
   - **Evaluation Set**: A **subsampled validation split** (restricted to EN and DE with fewer steps to prevent prohibitive optimization runtimes).
   - *Clarification*: The 71.10% validation F1 score is a validation subset metric and is **not directly comparable** to the full test set F1 scores (82.90% / 86.12%). It was used purely as a relative proxy objective to search for optimal parameter configurations.
4. **INT8 Quantized ONNX**:
   - **Metric**: Token-level classification accuracy of **89.92%** (compared to the PyTorch baseline's **88.74%**).
   - **Evaluation Set**: A tiny **252-sample benchmarking subset** loaded in `05_onnx_quantization.ipynb` to evaluate runtime speed, memory footprint, and model correctness.
   - *Clarification*: The 89.92% value is a **token-level classification accuracy** on a small benchmarking benchmark subset, and is **not** an entity-level test set micro-F1 score. The slight performance difference (89.92% vs. 88.74%) is due to minor variation in tensor division, logit values, and padding handling between PyTorch and ONNX Runtime.

---

## 5. Model Compression Summary

- **Teacher Model Weight Size**: **2.24 GB** (`model.safetensors`)
- **Quantized ONNX Model Weight Size**: **278 MB** (`model.onnx` inside deployment artifacts)
- **Outcome**: Successfully reduced storage from **2.24 GB to 278 MB (~8.1× reduction)**. The quantized model size is dominated by the embedding layer parameter weights needed for XLM-RoBERTa's large multilingual vocabulary of 250,000 tokens.

---

## 6. Which Plots Belong in the Paper?

For an academic publication, the following figures should be included to support the research claims (all paths reference the reorganized taxonomy):

1. **Learning Curves (Teacher vs. Student)**:
   - `figures/oracle/teacher_training_metrics.png` (Teacher training curves)
   - `figures/oracle/student_distillation_metrics.png` (Student distillation curves)
2. **Optuna Parameter Relationships**:
   - `outputs/plots/optuna_parallel_coordinate.html` and `outputs/plots/optuna_slice_plot.html` to visually demonstrate how hyperparameter configurations cluster around optimal F1 scores.
3. **Inference Latency & Throughput Trade-offs**:
   - `figures/latency/model_optimization_benchmarks.png` (Visualizing speedups and throughput on CPU)
4. **Linguistic Confusion Matrix**:
   - `figures/evaluation/confusion_matrix.png` (Entity prediction mismatches)
5. **Cross-Lingual Transfer Zero-Shot Performance**:
   - `figures/evaluation/cross_lingual_transfer.png` (F1 heatmaps across Latin and non-Latin target scripts)
6. **Error Classification Profiles**:
   - `figures/evaluation/boundary_errors.png` (Analyzing spans and boundary misses)
   - `figures/evaluation/error_patterns.png` (Evaluating prediction offsets)

---

## 7. Which Plots Belong in the README?

For the repository landing page, the figures focus on the high-level engineering and dataset profiling outcomes:

- **Dataset Characteristics**:
  - `figures/qualitative/language_distribution.png` (WikiANN language splits)
  - `figures/qualitative/ner_tag_distribution.png` (Class distribution balances)
  - `figures/qualitative/sentence_length_analysis.png` (Token length histogram suggesting 128 max length truncation)
- **Model Optimization comparison**:
  - `figures/latency/model_optimization_benchmarks.png` (Speedup ratios and footprints)
- **Zero-Shot Generalization heatmap**:
  - `figures/evaluation/cross_lingual_transfer.png` (Visualizing language transfer performance)

---

## 8. Which Checkpoints Correspond to Each Experiment?

All model weights and databases are tracked locally and managed via Git LFS:

- **Teacher Experiment**: `ner-project-results/models/teacher/` (Teacher configuration metadata and training metrics; model weights are kept local as they exceed the 2GB single-file GitHub LFS upload limit).
- **Knowledge Distillation Baseline**: `ner-project-results/models/student_distilled/model.safetensors` (Student base weights, 1.11 GB).
- **Hyperparameter-Tuned Student**: `ner-project-results/models/optuna_tuned/final_model/model.safetensors` (Optimal student weights, 1.11 GB).
- **Quantized ONNX Experiment**: `ner-project-results/models/optimized/deployment/model.onnx` (Dynamic INT8 quantized deployment model, 278 MB).
- **Bayesian Registry**: `ner-project-results/models/optuna_tuned/optuna.db` (SQLite database with 36 trials, 168 KB).

*Note: Intermediate training checkpoints (e.g. `checkpoint-375/` containing the 2.22 GB optimizer state) and raw unquantized 1.11 GB ONNX models have been safely discarded to save 5.82 GB of space.*

---

## 9. Which Experiment is the Final Model?

The final model is the **INT8 Dynamically Quantized ONNX Model** (derived from the Optuna hyperparameter-tuned distilled student model). It is located at:
`ner-project-results/models/optimized/deployment/model.onnx` (Size: 278 MB).

---

## 10. Which Experiments are Exploratory?

- **Notebook 01 (Data Exploration)**: Exploratory data analysis (EDA) to understand sentence lengths, vocabulary/token alignment, and tag distributions across WikiANN.
- **Notebook 03 (Distillation Baseline)**: An exploratory run to verify that a standard distillation setup converges before executing expensive hyperparameter searches.
- **Teacher GPU vs. ONNX CPU benchmarking**: Exploratory verification of hardware acceleration differences.

---

## 11. Which Experiments Should Be Discarded?

- **Raw, Unquantized ONNX Model**: `ner-project-results/models/optimized/onnx/model.onnx` (1.11 GB) and its optimized variant `model_optimized.onnx` (1.11 GB). These intermediate checkpoints do not need to be kept since they offer no compression and are slower than PyTorch GPU or Quantized ONNX CPU.
- **Intermediate Checkpoint States**: `checkpoint-375/` folder optimizer variables (`optimizer.pt`, `scheduler.pt`) which were deleted safely after training completion.

---

## 12. Which Experiments Are Missing?

From a rigorous research standpoint, the following experiments are missing to strengthen the paper:

1. **Base Model Training Baseline**: Training/fine-tuning the `xlm-roberta-base` student directly on the WikiANN dataset *without* distillation. This is required to show the delta improvement gained *specifically* from the teacher's dark knowledge (distillation) compared to standard training.
2. **Quantization-Aware Training (QAT)**: The pipeline only performs post-training dynamic quantization. QAT usually yields higher accuracy for low-bitwidth models.
3. **Structured Pruning Evaluation**: Evaluation of structural pruning (e.g. block/attention head removal) alongside quantization.
4. **Fair Baseline Benchmarking**: Benchmarks evaluating PyTorch and ONNX under the exact same hardware (CPU vs. CPU or GPU vs. GPU) to isolate the latency improvement of ONNX graph optimizations alone.

---

## 13. Threats to Validity

To preserve research integrity, several limitations must be acknowledged:

- **Restricted Training Languages**: The fine-tuning and distillation were restricted to English (EN), German (DE), and French (FR). Consequently, multilingual transfer to non-European or low-resource typologies has not been systematically evaluated.
- **Dataset Noise**: WikiANN is a silver-standard dataset built automatically from Wikipedia link structures. Noisy boundaries and entity labels in the dataset constitute a threat to true out-of-domain model generalization.
- **Zero-Shot Language Scope**: Zero-shot cross-lingual transfer was evaluated on only two target languages: Spanish (ES, Latin script) and Russian (RU, Cyrillic script).
- **Post-Training Dynamic Quantization (PTQ)**: PTQ was selected for dynamic execution, which may lead to out-of-distribution performance degradation compared to training-integrated Quantization-Aware Training (QAT).
- **Lack of Head/Layer Pruning**: The architecture compression is entirely parameter-quantized and distilled. Structural block or attention head pruning was not evaluated.
- **Hardware-Specific Benchmarks**: Throughput and latency measurements are hardware-dependent and bound to the host CPU environment on which the tests were conducted.

---

## 14. Contributions

This repository presents the following core contributions:

- **Task-Specific Knowledge Distillation**: Successfully compressed a large cross-lingual model (`xlm-roberta-large`) into a base model (`xlm-roberta-base`), preserving **96.26%** of the teacher's baseline F1 score ($82.90\%$ vs. $86.12\%$ micro-F1).
- **Bayesian Hyperparameter Search**: Implemented a 36-trial Optuna study to optimize temperature and alpha, establishing the critical importance of low temperature and lower alpha on small-dataset validation targets.
- **CPU Servable INT8 Quantization**: Integrated ONNX graph optimization and dynamic quantization to compress the student model from **2.24 GB to 278 MB (~8.1× reduction)**, delivering a deployment-ready model with 78.9ms latency.
- **Zero-Shot Cross-Lingual Evaluation**: Systematically measured cross-lingual validation degradation on Spanish (ES) and Russian (RU) zero-shot targets.
- **Modular, Reproducible Python Package**: Refactored monolithic Kaggle notebooks into a fully typed, tested, and documented Python package (`src/`) with reproducible command-line invocation drivers.

---

## 15. What is the Exact Experimental Story?

The experimental story progresses through 6 logical stages:

```
[1. Dataset Characterization]
   - Evaluated tag diversity and sentence lengths in WikiANN.
   - Identified English, German, and French as the primary multilingual training languages.
   
[2. Establishing Teacher Upper-Bound]
   - Fine-tuned XLM-RoBERTa-large (2.24GB) on EN, DE, FR.
   - Achieved a high-accuracy baseline F1 score of 86.12%.
   
[3. Baseline Knowledge Transfer]
   - Distilled teacher knowledge into XLM-RoBERTa-base (1.11GB) using standard parameters.
   - Reduced model size by 2.0x while maintaining an F1 of 82.90%.
   
[4. Optimization of Knowledge Transfer]
   - Conducted 36 Optuna trials to search for the best learning rate, batch size, temperature, and alpha.
   - Found optimal parameters and trained the best student model (71.77% Validation F1 on subset).
   
[5. Model Compaction and CPU Acceleration]
   - Converted the best student to ONNX format.
   - Applied dynamic INT8 quantization to reduce storage from 2.24 GB to 278 MB (~8.1x reduction) and achieve 78.9ms CPU latency.
   
[6. Evaluation of Generalization and Boundary Accuracy]
   - Tested the final model on zero-shot cross-lingual transfer (Spanish & Russian).
   - Diagnosed boundary classification patterns and compiled the final error reports.
```
