import os
import time
import psutil
import torch
import numpy as np
import random
import wandb
from evaluate import load
from typing import Dict, List, Any, Optional

# Initialize seqeval metric
try:
    seqeval_metric = load("seqeval")
except Exception:
    seqeval_metric = None

def compute_metrics_fn(p: Any, label_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Compute seqeval NER metrics (Precision, Recall, F1, Accuracy) on predictions.
    Filters out padding/special tokens marked with -100.
    """
    if label_names is None:
        label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [label_names[p_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    if seqeval_metric is None:
        # Fallback if evaluate isn't fully set up or offline
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
        
    results = seqeval_metric.compute(
        predictions=true_predictions,
        references=true_labels
    )
    
    try:
        if wandb.run is not None:
            wandb.log({
                "eval_precision": results["overall_precision"],
                "eval_recall": results["overall_recall"],
                "eval_f1": results["overall_f1"],
                "eval_accuracy": results["overall_accuracy"]
            })
    except Exception:
        pass
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }

def evaluate_by_language(
    dataset: Any,
    trainer: Any,
    tokenizer: Any,
    languages: List[str],
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Evaluate trainer model performance separately on each language slice.
    """
    language_results = {}
    cols_to_remove = dataset["train"].column_names

    for lang in languages:
        print(f"\nEvaluating on {lang}...")
        
        # Filter the dataset by language metadata
        lang_dataset = dataset.filter(lambda x: x["language"] == lang)
        
        # Check if the test split is present and not empty
        if "test" not in lang_dataset or len(lang_dataset["test"]) == 0:
            print(f"Skipping {lang} evaluation: no samples in test split.")
            continue
            
        # Preprocess dataset slice using alignment functions
        from src.data.dataset import preprocess_dataset
        tokenized_lang_dataset = preprocess_dataset(lang_dataset, tokenizer, max_length)
        
        # Evaluate on the test split
        results = trainer.evaluate(tokenized_lang_dataset["test"])
        language_results[lang] = results
        
        print(f"{lang} results: {results}")
        
        try:
            if wandb.run is not None:
                wandb.log({
                    f"{lang}/f1": results.get("eval_f1", 0.0),
                    f"{lang}/precision": results.get("eval_precision", 0.0),
                    f"{lang}/recall": results.get("eval_recall", 0.0)
                })
        except Exception:
            pass
            
    return language_results

def compare_models(
    teacher_model: Any,
    student_model: Any,
    tokenizer: Any,
    dataset: Any,
    num_samples: int = 100,
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Compare predictions of the teacher model and student model side-by-side on validation samples.
    """
    samples = random.sample(list(dataset["validation"]), min(num_samples, len(dataset["validation"])))
    
    teacher_correct = 0
    student_correct = 0
    agreement = 0
    total_tokens = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)
    
    teacher_model.eval()
    student_model.eval()
    
    for sample in samples:
        tokens = sample["tokens"]
        true_labels = sample["ner_tags"]
        
        inputs_obj = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        
        word_ids = inputs_obj.word_ids(batch_index=0)
        inputs = {k: v.to(device) for k, v in inputs_obj.items()}
        
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
            student_outputs = student_model(**inputs)
        
        teacher_preds = torch.argmax(teacher_outputs.logits, dim=-1)[0].cpu().numpy()
        student_preds = torch.argmax(student_outputs.logits, dim=-1)[0].cpu().numpy()
        
        previous_word_idx = None
        teacher_aligned = []
        student_aligned = []
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                if word_idx < len(true_labels):
                    teacher_aligned.append(teacher_preds[idx])
                    student_aligned.append(student_preds[idx])
            previous_word_idx = word_idx
        
        if len(teacher_aligned) == len(true_labels):
            teacher_correct += sum(1 for t, tl in zip(teacher_aligned, true_labels) if t == tl)
            student_correct += sum(1 for s, tl in zip(student_aligned, true_labels) if s == tl)
            agreement += sum(1 for t, s in zip(teacher_aligned, student_aligned) if t == s)
            total_tokens += len(true_labels)
            
    total_tokens = max(total_tokens, 1)
    
    return {
        "teacher_accuracy": teacher_correct / total_tokens,
        "student_accuracy": student_correct / total_tokens,
        "prediction_agreement": agreement / total_tokens,
        "total_tokens_evaluated": total_tokens
    }

class ModelBenchmark:
    """
    Benchmark memory, latency, throughput, and size for PyTorch and ONNX model sessions.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results = {}
        
    def measure_memory(self) -> float:
        """Measure current process memory usage in MB."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024**2
    
    def benchmark_pytorch(self, model: Any, inputs: Dict[str, torch.Tensor], n_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark a PyTorch model's execution performance."""
        print(f"\nBenchmarking PyTorch model ({self.model_name})...")
        
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(**inputs)
        
        # Latency benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(n_iterations):
                _ = model(**inputs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_latency = (end_time - start_time) * 1000 / n_iterations
        
        # Memory benchmark
        memory_before = self.measure_memory()
        with torch.no_grad():
            _ = model(**inputs)
        memory_after = self.measure_memory()
        memory_usage = max(0.0, memory_after - memory_before)
        
        # Model size
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        model_size_mb = param_size / 1024**2
        
        self.results['pytorch'] = {
            'avg_latency_ms': avg_latency,
            'memory_usage_mb': memory_usage,
            'model_size_mb': model_size_mb,
            'throughput_qps': 1000 / avg_latency
        }
        
        print(f"  Average latency: {avg_latency:.2f} ms")
        print(f"  Model size: {model_size_mb:.2f} MB")
        print(f"  Memory usage: {memory_usage:.2f} MB")
        print(f"  Throughput: {1000/avg_latency:.2f} QPS")
        
        return self.results['pytorch']
    
    def _benchmark_onnx_session(self, session: Any, inputs: Dict[str, Any], n_iterations: int, key: str) -> Dict[str, Any]:
        """Generic runner to benchmark ONNX run sessions."""
        ort_inputs = {
            'input_ids': inputs['input_ids'].cpu().numpy() if hasattr(inputs['input_ids'], 'cpu') else inputs['input_ids'],
            'attention_mask': inputs['attention_mask'].cpu().numpy() if hasattr(inputs['attention_mask'], 'cpu') else inputs['attention_mask']
        }
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, ort_inputs)
        
        # Latency benchmark
        start_time = time.time()
        for _ in range(n_iterations):
            _ = session.run(None, ort_inputs)
        end_time = time.time()
        
        avg_latency = (end_time - start_time) * 1000 / n_iterations
        
        # Memory benchmark
        memory_before = self.measure_memory()
        _ = session.run(None, ort_inputs)
        memory_after = self.measure_memory()
        memory_usage = max(0.0, memory_after - memory_before)
        
        # Model size on disk
        model_size_mb = 0.0
        try:
            if hasattr(session, '_model_path') and session._model_path:
                model_size_mb = os.path.getsize(session._model_path) / 1024**2
        except Exception:
            pass
            
        self.results[key] = {
            'avg_latency_ms': avg_latency,
            'memory_usage_mb': memory_usage,
            'model_size_mb': model_size_mb,
            'throughput_qps': 1000 / avg_latency
        }
        
        print(f"  Average latency: {avg_latency:.2f} ms")
        print(f"  Model size: {model_size_mb:.2f} MB")
        print(f"  Memory usage: {memory_usage:.2f} MB")
        print(f"  Throughput: {1000/avg_latency:.2f} QPS")
        
        return self.results[key]
        
    def benchmark_onnx(self, session: Any, inputs: Any, n_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark ONNX runtime inference session."""
        print(f"\nBenchmarking ONNX model...")
        return self._benchmark_onnx_session(session, inputs, n_iterations, 'onnx')
        
    def benchmark_quantized(self, session: Any, inputs: Any, n_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark quantized ONNX runtime inference session."""
        print(f"\nBenchmarking Quantized ONNX model...")
        return self._benchmark_onnx_session(session, inputs, n_iterations, 'quantized')

def validate_accuracy(
    original_model: Any,
    onnx_session: Any,
    quantized_session: Any,
    tokenizer: Any,
    test_samples: int = 50,
    max_length: int = 128
) -> Dict[str, float]:
    """
    Validate sequence labeling accuracy alignment across PyTorch, ONNX, and Quantized models on English WikiANN splits.
    """
    from datasets import load_dataset
    
    device = next(original_model.parameters()).device
    original_model.eval()
    
    dataset = load_dataset("unimelb-nlp/wikiann", "en", trust_remote_code=True)
    test_data = dataset['test'].select(range(min(test_samples, len(dataset['test']))))
    
    correct_original = 0
    correct_onnx = 0
    correct_quantized = 0
    total_tokens = 0
    
    for i in range(len(test_data)):
        sample = test_data[i]
        tokens = sample['tokens']
        true_labels = sample['ner_tags']
        
        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length"
        ).to(device)
        
        # 1. Original PyTorch prediction
        with torch.no_grad():
            outputs = original_model(**inputs)
        original_preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
        
        # 2. ONNX prediction
        ort_inputs = {
            'input_ids': inputs['input_ids'].cpu().numpy(),
            'attention_mask': inputs['attention_mask'].cpu().numpy()
        }
        onnx_outputs = onnx_session.run(None, ort_inputs)
        onnx_preds = np.argmax(onnx_outputs[0], axis=-1)[0]
        
        # 3. Quantized ONNX prediction
        quantized_outputs = quantized_session.run(None, ort_inputs)
        quantized_preds = np.argmax(quantized_outputs[0], axis=-1)[0]
        
        # Extract word ids to align sub-words
        word_ids = tokenizer(tokens, is_split_into_words=True).word_ids(batch_index=0)
        previous_word_idx = None
        original_aligned = []
        onnx_aligned = []
        quantized_aligned = []
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                if word_idx < len(true_labels):
                    original_aligned.append(original_preds[idx])
                    onnx_aligned.append(onnx_preds[idx])
                    quantized_aligned.append(quantized_preds[idx])
            previous_word_idx = word_idx
            
        if len(original_aligned) == len(true_labels):
            correct_original += sum(1 for o, tl in zip(original_aligned, true_labels) if o == tl)
            correct_onnx += sum(1 for o, tl in zip(onnx_aligned, true_labels) if o == tl)
            correct_quantized += sum(1 for q, tl in zip(quantized_aligned, true_labels) if q == tl)
            total_tokens += len(true_labels)
            
    total_tokens = max(total_tokens, 1)
    
    scores = {
        "original_accuracy": correct_original / total_tokens,
        "onnx_accuracy": correct_onnx / total_tokens,
        "quantized_accuracy": correct_quantized / total_tokens
    }
    
    print("\n" + "="*60)
    print("ACCURACY VALIDATION RESULTS")
    print("="*60)
    print(f"Original Model Accuracy:  {scores['original_accuracy']:.4f}")
    print(f"ONNX Model Accuracy:      {scores['onnx_accuracy']:.4f}")
    print(f"Quantized Model Accuracy: {scores['quantized_accuracy']:.4f}")
    print("="*60)
    
    return scores
