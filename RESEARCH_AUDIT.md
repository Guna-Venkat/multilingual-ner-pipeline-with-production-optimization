# Research Audit: Multilingual NER Pipeline with Production Optimization

This audit reviews the repository strictly from a **scientific and research perspective**, analyzing the core experimental questions, hypotheses, testing protocols, results, and artifacts.

---

## 1. What is the Research Question?

The central research question is:
**"To what extent can we compress a large multilingual Named Entity Recognition (NER) model (XLM-RoBERTa-large) using task-specific knowledge distillation, Bayesian hyperparameter optimization, and dynamic INT8 quantization, while preserving its multilingual and cross-lingual zero-shot generalization capabilities in resource-constrained CPU production settings?"**

---

## 2. What Hypotheses are Tested?

- **Hypothesis 1 (Model Compression via Distillation)**: A smaller student model (`xlm-roberta-base`) trained via task-specific knowledge distillation can match the NER F1 score of a fine-tuned teacher model (`xlm-roberta-large`) within a narrow margin (e.g. < 2% F1 loss) while reducing the model's footprint by over 50%.
- **Hypothesis 2 (Hyperparameter Importance in Distillation)**: Systematic Bayesian optimization (via Optuna) of distillation hyperparameters—specifically the temperature ($T$) and imitation weight ($\alpha$) alongside training parameters (learning rate, batch size)—can close the performance gap between the distilled student and the teacher compared to a default-parameter distilled student.
- **Hypothesis 3 (Quantization and Acceleration)**: Post-training graph optimization and dynamic INT8 quantization of the student model in ONNX format can yield a further 4x reduction in storage size and >4x improvement in CPU inference throughput, with negligible (< 1%) impact on token-level NER accuracy.
- **Hypothesis 4 (Cross-Lingual Zero-Shot Generalization)**: The compressed and quantized model preserves cross-lingual transfer capabilities (evaluated on Spanish and Russian, which were omitted from teacher training) similarly to the larger teacher model, without introducing boundary alignment degradation.

---

## 3. Which Experiment Answers Each Hypothesis?

- **Hypothesis 1 (Distillation Baseline)**: Answered by the distillation training run in `03_knowledge_distillation.ipynb`. This experiment evaluates the student model on English, German, and French and compares the F1 score directly with the fine-tuned teacher.
- **Hypothesis 2 (Bayesian Tuning)**: Answered by the 25-trial Optuna study in `04_hyperparameter_tuning.ipynb` and the final trained student model. The relationship plots (parallel coordinates, parameter importances) show how critical temperature and alpha are to closing the teacher-student gap.
- **Hypothesis 3 (ONNX and INT8 Quantization)**: Answered by the export and dynamic quantization benchmarking pipeline in `05_onnx_quantization.ipynb`. This runs PyTorch vs. ONNX vs. Quantized ONNX speed/memory tests and measures accuracy degradation.
- **Hypothesis 4 (Cross-Lingual and Error Stability)**: Answered by the zero-shot inference and error analysis in the latter half of `05_onnx_quantization.ipynb` (copied in `06_error_analysis.ipynb`), evaluating the model on Spanish (es) and Russian (ru) test splits.

---

## 4. Which Notebook Generated Each Reported Result?

The key results reported in the `README.md` were generated across the following notebooks:

- **Teacher F1 (0.892)**: Generated in `02_teacher_training.ipynb` (fine-tuning XLM-RoBERTa-large on WikiANN EN, DE, FR for 3 epochs).
- **Distilled Student Baseline F1 (0.876)**: Generated in `03_knowledge_distillation.ipynb` (distilling to XLM-RoBERTa-base with default parameters $T=2.0$, $\alpha=0.5$).
- **Optuna Tuning Improvement (+2.3% F1)**: Generated in `04_hyperparameter_tuning.ipynb` (25-trial Optuna search, yielding final tuned student weights).
- **Quantized ONNX Latency (25ms), Size (125MB/278MB), and F1 (0.868)**: Generated in `05_onnx_quantization.ipynb` (via ONNX runtime export, INT8 quantization, and validation accuracy code blocks).
  - *Note on Discrepancy*: The README reports the quantized size as 125MB, but the actual file size of the quantized ONNX model in `ner-project-results/` is **278 MB** (due to the large embedding parameters of XLM-RoBERTa's 250k vocabulary).

---

## 5. Which Plots Belong in the Paper?

For an academic publication, the following figures should be included to support the research claims:

1. **Learning Curves (Teacher vs. Student)**: Comparison of training/validation loss and F1 progression over epochs (`distillation_plots.png` from `03_knowledge_distillation.ipynb`).
2. **Optuna Parameter Relationships**: The **Parallel Coordinate Plot** (`parallel_coordinate.html`) and **Parameter Importance Chart** (`slice_plot.html`) to visually demonstrate how hyperparameter configurations cluster around optimal F1 scores.
3. **Inference Latency & Throughput Trade-offs**: The bar charts comparing PyTorch vs. ONNX vs. Quantized ONNX (`optimization_comparison.png` from `05_onnx_quantization.ipynb`).
4. **Linguistic Confusion Matrix**: The confusion matrix for NER tags (B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, O) across languages to identify token classification boundaries (`plot_confusion_matrix` output in error analysis).
5. **Cross-Lingual Transfer zero-shot performance**: Bar chart comparing performance drop on Spanish and Russian vs. trained languages (English, German, French).

---

## 6. Which Plots Belong in the README?

For a project landing page, the figures should focus on the high-level engineering outcomes:

- **Optimization Comparison Chart** (`ner-project-results/models/optimized/optimization_comparison.png`): Shows the 18x speedup and 4x model compression at a glance.
- **Parallel Coordinate Plot** (static export of `parallel_coordinate.html`): Demonstrates the systematic MLOps hyperparameter optimization approach.
- **Distillation Comparison Plot** (`ner-project-results/models/student_distilled/distillation_plots.png`): Summarizes the convergence behavior during knowledge distillation.

---

## 7. Which Checkpoints Correspond to Each Experiment?

The checkpoints are saved under the `ner-project-results/` directory:

- **Teacher Base Experiment**: `ner-project-results/models/teacher/` (safetensors weights of XLM-R Large, 2.24 GB).
- **Knowledge Distillation Baseline**: `ner-project-results/models/student_distilled/` (safetensors weights of XLM-R Base, 1.11 GB).
- **Hyperparameter-Tuned Student**: `ner-project-results/models/optuna_tuned/final_model/` (safetensors weights of tuned XLM-R Base, 1.11 GB) and its corresponding training checkpoint `checkpoint-375/`.
- **Quantized ONNX Experiment**: `ner-project-results/models/optimized/quantized/model_quantized.onnx` (INT8 quantized model, 278 MB).
- **Production Deployment Artifact**: `ner-project-results/models/optimized/deployment/model.onnx` (the quantized weights packaged with tokenizer configurations, 278 MB).

---

## 8. Which Experiment is the Final Model?

The final model is the **INT8 Dynamically Quantized ONNX Model** (derived from the Optuna hyperparameter-tuned distilled student model). It is located at:
`ner-project-results/models/optimized/deployment/model.onnx` (Size: 278 MB).

---

## 9. Which Experiments are Exploratory?

- **Notebook 01 (Data Exploration)**: Exploratory data analysis (EDA) to understand sentence lengths, vocabulary/token alignment, and tag distributions across WikiANN.
- **Notebook 03 (Distillation Baseline)**: An exploratory run to verify that a standard distillation setup converges before executing expensive hyperparameter searches.
- **Teacher GPU vs. ONNX CPU benchmarking**: Exploratory verification of hardware acceleration differences.

---

## 10. Which Experiments Should Be Discarded?

- **Raw, Unquantized ONNX Model**: `ner-project-results/models/optimized/onnx/model.onnx` (1.11 GB) and its optimized variant `model_optimized.onnx` (1.11 GB). These intermediate checkpoints do not need to be kept since they offer no compression and are slower than PyTorch GPU or Quantized ONNX CPU.
- **Notebook `06_error_analysis.ipynb`**: This file is a duplicate of `05_onnx_quantization.ipynb`. It should be refactored to remove the duplicate training, distillation, tuning, and quantization cells, leaving only the test dataset prediction loader and linguistic analysis metrics.

---

## 11. Which Experiments Are Missing?

From a rigorous research standpoint, the following experiments are missing to strengthen the paper:

1. **Base Model Training Baseline**: Training/fine-tuning the `xlm-roberta-base` student directly on the WikiANN dataset *without* distillation. This is required to show the delta improvement gained *specifically* from the teacher's dark knowledge (distillation) compared to standard training.
2. **Quantization-Aware Training (QAT)**: The pipeline only performs post-training dynamic quantization. QAT usually yields higher accuracy for low-bitwidth models.
3. **Pruning-ready Architecture Evaluation**: The README lists "pruning-ready architecture" as a feature, but there are no pruning experiments (structured or unstructured) implemented.
4. **Fair Baseline Benchmarking**: Benchmarks evaluating PyTorch and ONNX under the exact same hardware (CPU vs. CPU or GPU vs. GPU) to isolate the latency improvement of ONNX graph optimizations alone.

---

## 12. What is the Exact Experimental Story?

The experimental story progresses through 6 logical stages:

```
[1. Dataset Characterization]
   - Evaluated tag diversity and sentence lengths in WikiANN.
   - Identified English, German, and French as the primary multilingual training languages.
   
[2. Establishing Teacher Upper-Bound]
   - Fine-tuned XLM-RoBERTa-large (1.2GB) on EN, DE, FR.
   - Achieved a high-accuracy baseline F1 score of 0.892.
   
[3. Baseline Knowledge Transfer]
   - Distilled teacher knowledge into XLM-RoBERTa-base (500MB) using standard parameters.
   - Reduced model size by 2.4x while maintaining an F1 of 0.876.
   
[4. Optimization of Knowledge Transfer]
   - Conducted 25 Optuna trials to search for the best learning rate, batch size, temperature, and alpha.
   - Found optimal parameters and trained the best student model.
   
[5. Model Compaction and CPU Acceleration]
   - Converted the best student to ONNX format.
   - Applied dynamic INT8 quantization (278MB) to achieve 25ms CPU latency (an 18x speedup vs. teacher).
   
[6. Evaluation of Generalization and Boundary Accuracy]
   - Tested the final model on zero-shot cross-lingual transfer (Spanish & Russian).
   - Diagnosed boundary classification patterns and compiled the final error reports.
```
