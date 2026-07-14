# Function Extraction Plan: FUNCTION_EXTRACTION_PLAN.md

This document details the plan to extract and refactor functions, classes, and configurations from the Jupyter notebooks into the modular `src/` directory. 

> [!IMPORTANT]
> **Constraint Check**: In alignment with research preservation rules, this plan **does not invent new architectures** or create placeholder modules. It only outlines the extraction of real, functional code blocks already implemented in the notebooks.

---

## 1. Data Exploration & Loading Functions (`notebooks/01_data_exploration.ipynb`)

### `load_multilingual_data(languages)`
- **Move to**: `src/data/dataset.py`
- **Description**: Loads the Hugging Face `wikiann` datasets for multiple languages.
- **Dependencies**: `datasets.load_dataset`
- **Difficulty**: Easy (simple loop wrapper)
- **Risk**: Low

### `preprocess_dataset(dataset, tokenizer, max_length)`
- **Move to**: `src/data/dataset.py`
- **Description**: High-level wrapper that maps aligned tokenization over a dataset.
- **Dependencies**: `transformers.AutoTokenizer`
- **Difficulty**: Medium (encloses tokenization mappings)
- **Risk**: Medium (any shift in default parameters can alter tokenization shapes)

### `tokenize_and_align_labels(examples)` (Inner helper)
- **Move to**: `src/data/dataset.py`
- **Description**: Tokenizes inputs while maintaining matching tag indexes for sub-words and padding tokens (-100).
- **Dependencies**: `transformers.AutoTokenizer`
- **Difficulty**: High (crucial logic maps word boundaries to sub-word token splits)
- **Risk**: High (any bug will cause shape mismatches or corrupt loss functions during training)

### `create_train_val_split(dataset, train_ratio, seed)`
- **Move to**: `src/data/dataset.py`
- **Description**: Creates train/validation splits if not provided.
- **Dependencies**: `datasets.DatasetDict`
- **Difficulty**: Easy
- **Risk**: Low

---

## 2. Model Training & Distillation (`notebooks/02_teacher_training.ipynb` & `03_knowledge_distillation.ipynb`)

### `class TeacherConfig`
- **Move to**: `src/configs/teacher_config.py`
- **Description**: Configuration variables for training the XLM-RoBERTa-large teacher model.
- **Dependencies**: None
- **Difficulty**: Easy
- **Risk**: Low

### `class DistillationConfig`
- **Move to**: `src/configs/distillation_config.py`
- **Description**: Hyperparameters and paths for student model training and distillation.
- **Dependencies**: None
- **Difficulty**: Easy
- **Risk**: Low

### `class DistillationTrainer(Trainer)`
- **Move to**: `src/training/distillation.py`
- **Description**: Custom Hugging Face `Trainer` subclass that overrides `compute_loss` to inject KL divergence loss between teacher and student logit distributions.
- **Dependencies**: `transformers.Trainer`, `torch.nn.functional`
- **Difficulty**: High (manages parallel models, dynamic temperature scaling, and imitation loss balancing)
- **Risk**: High (critical path; device mapping bugs will throw CUDA memory or execution errors)

### `load_multilingual_dataset(languages, max_train_samples)` (Concatenated version)
- **Move to**: `src/data/dataset.py`
- **Description**: Loads, subsamples, and concatenates datasets for multi-language training.
- **Dependencies**: `datasets.load_dataset`, `datasets.concatenate_datasets`
- **Difficulty**: Medium (handles dataset down-sampling and shuffling logic)
- **Risk**: Medium (needs to be combined with exploration data loaders to prevent duplicate loading routines)

---

## 3. Optimization & Tuning (`notebooks/04_hyperparameter_tuning.ipynb` & `05_onnx_quantization.ipynb`)

### `class OptunaConfig`
- **Move to**: `src/configs/tuning_config.py`
- **Description**: Config class containing bounds for Bayesian trials, paths, and Optuna settings.
- **Dependencies**: None
- **Difficulty**: Easy
- **Risk**: Low

### `objective(trial, tokenized_dataset)`
- **Move to**: `src/training/distillation.py`
- **Description**: Core Optuna trial target; instantiates a student model, loads hyperparameters (learning rate, batch size, temperature, alpha), and runs distillation for 3 epochs.
- **Dependencies**: `optuna`, custom `DistillationTrainer`, `torch`
- **Difficulty**: High (coordinates resources; must prevent GPU memory leaks across trials)
- **Risk**: High (trial memory leaks will crash the entire tuning session midway)

### `train_final_model(best_trial, full_dataset)`
- **Move to**: `src/training/distillation.py`
- **Description**: Trains the final distilled student model using the optimal parameter set identified during Optuna trials.
- **Dependencies**: Custom `DistillationTrainer`
- **Difficulty**: Medium
- **Risk**: High (must output exact safetensors structure corresponding to the peak optimized student)

### `class OptimizationConfig`
- **Move to**: `src/configs/optimization_config.py`
- **Description**: Stores paths and parameters for ONNX export, graph optimizations, and dynamic quantization.
- **Dependencies**: None
- **Difficulty**: Easy
- **Risk**: Low

### `export_to_onnx(model, tokenizer, output_path)`
- **Move to**: `src/optimization/quantization.py`
- **Description**: Converts the PyTorch student model to ONNX format with dynamic axis configurations.
- **Dependencies**: `torch.onnx`, dummy tensor trace creators
- **Difficulty**: High (requires precise configuration of model inputs: `input_ids`, `attention_mask`)
- **Risk**: High (shape trace mismatches can export an invalid graph that fails inference)

### `quantize_onnx_model(onnx_path, output_path)`
- **Move to**: `src/optimization/quantization.py`
- **Description**: Applies Dynamic INT8 quantization to an exported ONNX model graph.
- **Dependencies**: `onnxruntime.quantization.quantize_dynamic`, `onnxruntime.quantization.QuantType`
- **Difficulty**: Medium (wraps ONNX library calls)
- **Risk**: Medium (must ensure compatibility with dynamic axis setups)

### `class MultilingualNER`
- **Move to**: `src/models/inference.py`
- **Description**: Production deployment class. Loads the quantized ONNX graph and tokenizer to perform end-to-end token predictions on raw strings.
- **Dependencies**: `onnxruntime`, `transformers.AutoTokenizer`, `numpy`, `json`
- **Difficulty**: High (must implement full aligned decoding since the original notebook class only contains a `pass` placeholder)
- **Risk**: High (this is the core wrapper exposed by the production REST API)

---

## 4. Evaluation & Metrics (`All Notebooks`)

### `set_seed(seed)`
- **Move to**: `src/utils/helpers.py`
- **Description**: Sets seeds for torch, numpy, and random.
- **Dependencies**: `torch`, `numpy`, `random`
- **Difficulty**: Easy
- **Risk**: Low

### `get_model_size(model, model_name)`
- **Move to**: `src/utils/helpers.py`
- **Description**: Counts total parameters and computes storage size in MB.
- **Dependencies**: `torch`
- **Difficulty**: Easy
- **Risk**: Low

### `compute_metrics(p)`
- **Move to**: `src/evaluation/metrics.py`
- **Description**: Aligns model logits with label sequences (skipping -100 padding) and runs seqeval F1/precision/recall tests.
- **Dependencies**: `evaluate.load("seqeval")`, `numpy`
- **Difficulty**: Medium
- **Risk**: Low

### `evaluate_by_language(dataset, trainer, tokenizer)`
- **Move to**: `src/evaluation/metrics.py`
- **Description**: Runs evaluations on distinct target language splits to check multi-lingual generalization.
- **Dependencies**: `seqeval`, HF `Trainer`
- **Difficulty**: Medium
- **Risk**: Low

### `predict_entities(text, model, tokenizer)`
- **Move to**: `src/models/inference.py`
- **Description**: Performs token-level inference on a single string using PyTorch.
- **Dependencies**: `torch`, `transformers`
- **Difficulty**: Medium
- **Risk**: Medium

### `compare_models(teacher_model, student_model, tokenizer, dataset, num_samples)`
- **Move to**: `src/evaluation/metrics.py`
- **Description**: Evaluates latency, throughput, and parameter metrics side-by-side between the teacher and student.
- **Dependencies**: `time`, `numpy`, `torch`
- **Difficulty**: Medium
- **Risk**: Low

### `class ModelBenchmark`
- **Move to**: `src/evaluation/metrics.py`
- **Description**: Benchmarking tool for CPU latency, memory utilization (via psutil), size, and throughput.
- **Dependencies**: `psutil`, `time`, `torch`, `onnxruntime`, `os`
- **Difficulty**: Medium
- **Risk**: Medium (needs to be corrected so both models run on the same baseline device execution provider)

### `validate_accuracy(original_model, onnx_session, quantized_session, tokenizer, test_samples)`
- **Move to**: `src/evaluation/metrics.py`
- **Description**: Compares PyTorch, raw ONNX, and quantized ONNX accuracy scores on test splits.
- **Dependencies**: `datasets`, `torch`, `onnxruntime`, `numpy`
- **Difficulty**: Medium
- **Risk**: Medium

---

## 5. Visualizations & Plots (`All Notebooks`)

### `analyze_sentence_lengths(dataset)`
- **Move to**: `src/plots/data_plots.py`
- **Description**: Generates histograms and box plots for sentence length distributions.
- **Dependencies**: `matplotlib.pyplot`, `numpy`
- **Difficulty**: Easy
- **Risk**: Low

### `compare_languages(multilingual_data)`
- **Move to**: `src/plots/data_plots.py`
- **Description**: Plots bar and pie charts comparing sample volumes and lengths across languages.
- **Dependencies**: `matplotlib.pyplot`, `pandas`
- **Difficulty**: Easy
- **Risk**: Low

### `plot_training_metrics(state_file)` & `plot_knowledge_transfer(state_file)`
- **Move to**: `src/plots/training_plots.py`
- **Description**: Plots training/validation loss curves from HF trainer history JSON files.
- **Dependencies**: `matplotlib.pyplot`, `pandas`, `json`
- **Difficulty**: Easy
- **Risk**: Low

### `plot_optimization_history(study)`, `plot_param_importances(study)`, `plot_parallel_coordinate(study)`, `plot_slice_plot(study)`
- **Move to**: `src/plots/tuning_plots.py`
- **Description**: Creates Plotly interactive plots for Optuna parameter tuning histories.
- **Dependencies**: `plotly`, `optuna.visualization`
- **Difficulty**: Medium (configures Plotly render outputs)
- **Risk**: Low

### `create_optimization_visualization(results, config)`
- **Move to**: `src/plots/optimization_plots.py`
- **Description**: Generates visual comparisons of latency, throughput, model size, and memory for PyTorch vs ONNX vs Quantized ONNX.
- **Dependencies**: `matplotlib.pyplot`
- **Difficulty**: Easy
- **Risk**: Low

---

## 6. Error & Linguistic Analysis (`notebooks/06_error_analysis.ipynb`)

### `class ErrorAnalysisConfig`
- **Move to**: `src/configs/evaluation_config.py`
- **Description**: Configuration limits, paths, and sample counts for evaluating prediction errors.
- **Dependencies**: None
- **Difficulty**: Easy
- **Risk**: Low

### `load_test_data(languages, sample_size)`
- **Move to**: `src/data/dataset.py`
- **Description**: Loads test subsets across languages for prediction harvesting.
- **Dependencies**: `datasets.load_dataset`
- **Difficulty**: Easy
- **Risk**: Low

### `predict_entities(model, tokenizer, tokens, model_type)` (List overload)
- **Move to**: `src/models/inference.py`
- **Description**: Overloaded prediction helper that processes pre-tokenized lists instead of raw text.
- **Dependencies**: `torch`, `onnxruntime`, `numpy`
- **Difficulty**: Medium
- **Risk**: High (must merge cleanly with raw-string prediction utilities)

### `collect_predictions(...)`
- **Move to**: `src/evaluation/error_analysis.py`
- **Description**: Iterates through datasets, makes model predictions, and records results into a prediction DataFrame.
- **Dependencies**: `pandas`
- **Difficulty**: Medium
- **Risk**: Low

### `calculate_overall_metrics(df)`, `analyze_by_language(df)`, `analyze_by_entity(df)`
- **Move to**: `src/evaluation/error_analysis.py`
- **Description**: Grouping utilities to compute metrics across entity types and languages from prediction dataframes.
- **Dependencies**: `pandas`, `seqeval`
- **Difficulty**: Easy
- **Risk**: Low

### `plot_confusion_matrix(true_labels, pred_labels)`
- **Move to**: `src/plots/evaluation_plots.py`
- **Description**: Generates normalized heatmaps for confusion matrices (e.g. tracking B-LOC to I-LOC or B-PER confusions).
- **Dependencies**: `seaborn`, `matplotlib.pyplot`, `sklearn.metrics.confusion_matrix`
- **Difficulty**: Medium
- **Risk**: Low

### `analyze_error_patterns(df)`
- **Move to**: `src/evaluation/error_analysis.py`
- **Description**: Isolates token sequences where the model makes systematic categorization mistakes.
- **Dependencies**: `pandas`
- **Difficulty**: Medium
- **Risk**: Low

### `visualize_error_patterns(error_df)`
- **Move to**: `src/plots/evaluation_plots.py`
- **Description**: Generates bar charts summarizing the most frequent sequence predictions errors.
- **Dependencies**: `matplotlib.pyplot`, `seaborn`
- **Difficulty**: Easy
- **Risk**: Low

### `find_hard_examples(df, n_examples)`
- **Move to**: `src/evaluation/error_analysis.py`
- **Description**: Filters prediction records to locate sentences with the lowest F1 scores.
- **Dependencies**: `pandas`
- **Difficulty**: Easy
- **Risk**: Low

### `analyze_cross_lingual_transfer(df)` & `analyze_boundary_errors(df)`
- **Move to**: `src/evaluation/error_analysis.py`
- **Description**: Extracts error frequencies specifically related to language zero-shot transfer and boundary token labeling (B- vs I- tags).
- **Dependencies**: `pandas`
- **Difficulty**: Easy
- **Risk**: Low

### `create_error_report(...)`
- **Move to**: `src/evaluation/error_analysis.py`
- **Description**: Compiles all error metrics and structures them into a comprehensive JSON report.
- **Dependencies**: `json`, `os`
- **Difficulty**: Easy
- **Risk**: Low
