# Repository Mapping Document: REPO_MAP.md

This document maps every file and core output group in the repository, explaining its research purpose, code dependencies, and recommended path for refactoring.

---

## Workspace Notebooks

### 1. `notebooks/01_data_exploration.ipynb`
- **Purpose**: Performs exploratory data analysis (EDA) on the WikiANN dataset splits (EN, DE, FR, ES, RU), analyzing tag frequencies, sentence lengths, and cross-lingual differences.
- **Dependencies**: `transformers`, `datasets`, `pandas`, `numpy`, `matplotlib`.
- **Can be deleted?**: No, it contains the baseline research exploration.
- **Should become module?**: Partly. The data cleaning, token alignment, and preprocessing function (`preprocess_dataset` / `tokenize_and_align_labels`) should be extracted to `src/data/dataset.py`.
- **Should remain notebook?**: Yes, the data visualization plots and EDA reporting should remain as a notebook for interactive presentation.
- **Used in final experiments?**: Yes, it defines the label alignment pre-processing logic used by all subsequent stages.
- **Mentioned in README?**: Yes, under project structure.

### 2. `notebooks/02_teacher_training.ipynb`
- **Purpose**: Fine-tunes the large teacher model (`xlm-roberta-large`) on EN, DE, and FR subsets of the WikiANN dataset.
- **Dependencies**: `transformers`, `datasets`, `evaluate`, `seqeval`, `accelerate`, `wandb`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `sentencepiece`, `sacremoses`.
- **Can be deleted?**: No, it is the code that establishes the upper-bound teacher baseline.
- **Should become module?**: Yes. The training execution loop, metric evaluation (`compute_metrics`), and checkpoint savers should move to `src/training/teacher.py` and `src/models/inference.py`.
- **Should remain notebook?**: Yes, but only as a slim execution interface that imports the modular functions rather than copy-pasting the raw setup code.
- **Used in final experiments?**: Yes, it generates the teacher checkpoint (`ner-project-results/models/teacher`) which is required for distillation.
- **Mentioned in README?**: Yes.

### 3. `notebooks/03_knowledge_distillation.ipynb`
- **Purpose**: Performs task-specific knowledge distillation from `xlm-roberta-large` to `xlm-roberta-base`.
- **Dependencies**: Same as teacher, plus custom `DistillationTrainer` and KL divergence loss.
- **Can be deleted?**: No, it is the primary distillation experiment.
- **Should become module?**: Yes. `DistillationTrainer` and distillation configs must move to `src/training/distillation.py`.
- **Should remain notebook?**: Yes, as a slim runner notebook importing from `src/`.
- **Used in final experiments?**: Yes, it validates that the distilled model structure works.
- **Mentioned in README?**: Yes.

### 4. `notebooks/04_hyperparameter_tuning.ipynb`
- **Purpose**: Runs Optuna trials to search for the best learning rate, batch size, temperature, and alpha for distillation.
- **Dependencies**: `optuna`, `plotly`, `kaleido` plus distillation dependencies.
- **Can be deleted?**: No, it optimizes the student model's transfer performance.
- **Should become module?**: Yes. The Optuna study setup and objective function should be moved to `src/optimization/tuning.py` or `src/training/distillation.py`.
- **Should remain notebook?**: Yes, as an interactive runner to display Plotly plots.
- **Used in final experiments?**: Yes, it finds the parameters that trained the final student model.
- **Mentioned in README?**: Yes.

### 5. `notebooks/05_onnx_quantization.ipynb`
- **Purpose**: Exports the student model to ONNX, applies INT8 dynamic quantization, benchmarks execution latency/memory, and runs error analysis on test splits.
- **Dependencies**: `onnx`, `onnxruntime`, `onnxruntime.quantization`, `psutil` plus evaluation packages.
- **Can be deleted?**: No, it contains the core production optimization and benchmarking experiments.
- **Should become module?**: Yes. The ONNX exporter, benchmark loops, and `MultilingualNER` prediction wrapper class should move to `src/optimization/quantization.py` and `src/models/inference.py`.
- **Should remain notebook?**: Yes, to visualize benchmarking metrics and run high-level latency comparisons.
- **Used in final experiments?**: Yes, it compiles the final deployable INT8 model and benchmarks.
- **Mentioned in README?**: Yes.

### 6. `notebooks/06_error_analysis.ipynb`
- **Purpose**: Performs detailed linguistic, zero-shot, and boundary error analysis on the models.
- **Dependencies**: Identical to Notebook 05 (they are duplicate files).
- **Can be deleted?**: Yes. The duplicate copy itself should be deleted, and a new, clean `06_error_analysis.ipynb` should be created that *only* contains the error analysis cells (Cells 87 to 104) and imports the model/data functions rather than duplicating the entire quantization pipeline.
- **Should become module?**: Yes, the error parsing, per-language accuracy grouping, and confusion matrix plotting should be in `src/evaluation/error_analysis.py`.
- **Should remain notebook?**: Yes, as a clean visual report.
- **Used in final experiments?**: Yes, it performs the final error diagnostic on the quantized model.
- **Mentioned in README?**: Yes.

---

## Infrastructure and Documentation Files

### 7. `README.md`
- **Purpose**: Project documentation, metrics reporting, folder structures, and API usage guides.
- **Dependencies**: None.
- **Can be deleted?**: No.
- **Should become module?**: No.
- **Should remain notebook?**: No.
- **Used in final experiments?**: No (metadata only).
- **Mentioned in README?**: N/A.

### 8. `LICENSE`
- **Purpose**: MIT License document.
- **Dependencies**: None.
- **Can be deleted?**: No.
- **Should become module?**: No.
- **Should remain notebook?**: No.
- **Used in final experiments?**: No.
- **Mentioned in README?**: Yes.

### 9. `.gitignore`
- **Purpose**: Excludes the large Kaggle output directory `ner-project-results/` from Git tracking.
- **Dependencies**: None.
- **Can be deleted?**: No.
- **Should become module?**: No.
- **Should remain notebook?**: No.
- **Used in final experiments?**: No.
- **Mentioned in README?**: No.

---

## Experiment Results (git-ignored `ner-project-results/`)

### 10. `ner-project-results/models/teacher/`
- **Purpose**: Weights and config file for the fine-tuned XLM-RoBERTa-large teacher model.
- **Dependencies**: None (serialized safetensors).
- **Can be deleted?**: No, this is the teacher baseline model used for distillation.
- **Should become module?**: No, it is a model checkpoint.
- **Should remain notebook?**: No.
- **Used in final experiments?**: Yes, acts as the teacher weights provider.
- **Mentioned in README?**: Yes.

### 11. `ner-project-results/models/student_distilled/`
- **Purpose**: Weights for the default baseline distilled student model (XLM-R base).
- **Dependencies**: None (safetensors).
- **Can be deleted?**: No, it represents the baseline distillation experiment checkpoint.
- **Should become module?**: No.
- **Should remain notebook?**: No.
- **Used in final experiments?**: Yes, compared against the final tuned model.
- **Mentioned in README?**: Yes.

### 12. `ner-project-results/models/optuna_tuned/`
- **Purpose**: Contains `optuna.db` (Bayesian search logs) and `final_model/` checkpoint (tuned student weights).
- **Dependencies**: SQLite, Safetensors.
- **Can be deleted?**: No. The tuning log and final trained model weights are essential.
- **Should become module?**: No.
- **Should remain notebook?**: No.
- **Used in final experiments?**: Yes, this is the peak PyTorch student model.
- **Mentioned in README?**: Yes.

### 13. `ner-project-results/models/optimized/`
- **Purpose**: Houses the ONNX and Quantized ONNX model files, deployment package, and custom requirements.
- **Dependencies**: ONNX format.
- **Can be deleted?**: The raw ONNX files (`model.onnx` at 1.1GB) can be deleted to save space, but the quantized ONNX models (`model_quantized.onnx`) and the `deployment/` subfolder must be preserved.
- **Should become module?**: No, these are production optimized artifacts.
- **Should remain notebook?**: No.
- **Used in final experiments?**: Yes, this contains the final deployment model.
- **Mentioned in README?**: Yes.

### 14. `ner-project-results/iframe_figures/` & `__results___files/`
- **Purpose**: Pre-computed plotly HTML reports and matplotlib figures from the Kaggle execution.
- **Dependencies**: HTML/PNG formats.
- **Can be deleted?**: No, these represent the scientific output plots from the training run.
- **Should become module?**: No.
- **Should remain notebook?**: No.
- **Used in final experiments?**: Yes, these are the visual evidence of the results.
- **Mentioned in README?**: Yes.
