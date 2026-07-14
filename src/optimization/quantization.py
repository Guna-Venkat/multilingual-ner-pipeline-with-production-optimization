import os
import time
import json
import psutil
import shutil
import torch
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from datasets import load_dataset
from typing import Dict, Any, List, Optional
from src.configs.config import OptimizationConfig

def export_to_onnx(model: torch.nn.Module, tokenizer: Any, output_path: str) -> onnx.ModelProto:
    """
    Export a PyTorch token classification model to ONNX format.
    """
    print(f"Exporting model to ONNX format...")
    
    model.eval()
    device = next(model.parameters()).device
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare input dummy on the same device
    text = "This is a sample sentence for ONNX export."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    dummy_input = (
        inputs["input_ids"], 
        inputs["attention_mask"]
    )
    
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'}
    }
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
    
    print(f"Model exported to {output_path}")
    
    # Verify exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed!")
    
    return onnx_model

def create_onnx_session(onnx_path: str, providers: Optional[List[str]] = None) -> ort.InferenceSession:
    """
    Create an ONNX Runtime InferenceSession with graph optimization enabled.
    """
    if providers is None:
        providers = ['CPUExecutionProvider']
    
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.optimized_model_filepath = onnx_path.replace(".onnx", "_optimized.onnx")
    
    session = ort.InferenceSession(
        onnx_path,
        sess_options=session_options,
        providers=providers
    )
    
    print(f"ONNX Runtime session created with providers: {session.get_providers()}")
    return session

def quantize_onnx_model(onnx_path: str, output_path: str) -> str:
    """
    Apply Dynamic Quantization (QInt8) to an ONNX model.
    """
    print(f"\nApplying dynamic quantization to ONNX model...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    quantize_dynamic(
        model_input=onnx_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )
    
    print(f"Quantized model saved to {output_path}")
    
    # Check size reduction
    original_size = os.path.getsize(onnx_path)
    quantized_size = os.path.getsize(output_path)
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"  Original size: {original_size / 1024**2:.2f} MB")
    print(f"  Quantized size: {quantized_size / 1024**2:.2f} MB")
    print(f"  Size reduction: {reduction:.1f}%")
    
    return output_path

class ModelBenchmark:
    """
    Benchmark suite to measure latency, throughput, and memory consumption.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results = {}
        
    def measure_memory(self) -> float:
        """Measure current RSS memory usage in MB."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024**2
    
    def benchmark_pytorch(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], n_iterations: int = 100) -> Dict[str, float]:
        """Benchmark PyTorch model latency and memory usage."""
        print(f"\nBenchmarking PyTorch model ({self.model_name})...")
        model.eval()
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(**inputs)
        
        # Benchmark time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(n_iterations):
                _ = model(**inputs)
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        avg_latency = (end_time - start_time) * 1000 / n_iterations
        
        # Measure memory
        memory_before = self.measure_memory()
        with torch.no_grad():
            _ = model(**inputs)
        memory_after = self.measure_memory()
        memory_usage = max(0.0, memory_after - memory_before)
        
        # Calculate size
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
    
    def _benchmark_onnx_session(self, session: ort.InferenceSession, inputs: Dict[str, torch.Tensor], n_iterations: int, key: str) -> Dict[str, float]:
        """Generic benchmark helper for ONNX session."""
        ort_inputs = {
            'input_ids': inputs['input_ids'].cpu().numpy(),
            'attention_mask': inputs['attention_mask'].cpu().numpy()
        }
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, ort_inputs)
        
        # Benchmark
        start_time = time.time()
        for _ in range(n_iterations):
            _ = session.run(None, ort_inputs)
        end_time = time.time()
        
        avg_latency = (end_time - start_time) * 1000 / n_iterations
        
        # Measure memory
        memory_before = self.measure_memory()
        _ = session.run(None, ort_inputs)
        memory_after = self.measure_memory()
        memory_usage = max(0.0, memory_after - memory_before)
        
        model_size_bytes = os.path.getsize(session._model_path)
        model_size_mb = model_size_bytes / 1024**2
        
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
        
    def benchmark_onnx(self, session: ort.InferenceSession, inputs: Dict[str, torch.Tensor], n_iterations: int = 100) -> Dict[str, float]:
        """Benchmark vanilla ONNX model latency and memory usage."""
        print(f"\nBenchmarking ONNX model...")
        return self._benchmark_onnx_session(session, inputs, n_iterations, 'onnx')
        
    def benchmark_quantized(self, session: ort.InferenceSession, inputs: Dict[str, torch.Tensor], n_iterations: int = 100) -> Dict[str, float]:
        """Benchmark Quantized ONNX model latency and memory usage."""
        print(f"\nBenchmarking Quantized ONNX model...")
        return self._benchmark_onnx_session(session, inputs, n_iterations, 'quantized')

def validate_accuracy(
    original_model: torch.nn.Module,
    onnx_session: ort.InferenceSession,
    quantized_session: ort.InferenceSession,
    tokenizer: Any,
    test_samples: int = 50,
    max_length: int = 128
) -> Dict[str, float]:
    """
    Validate and compare predictions across PyTorch, ONNX, and Quantized ONNX models on test dataset.
    """
    print("\n" + "="*60)
    print("ACCURACY VALIDATION")
    print("="*60)
    
    device = next(original_model.parameters()).device
    original_model.eval()
    
    # Load dataset test split
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
        
        # PyTorch prediction
        with torch.no_grad():
            outputs = original_model(**inputs)
        original_preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
        
        # ONNX prediction
        ort_inputs = {
            'input_ids': inputs['input_ids'].cpu().numpy(),
            'attention_mask': inputs['attention_mask'].cpu().numpy()
        }
        
        onnx_outputs = onnx_session.run(None, ort_inputs)
        onnx_preds = np.argmax(onnx_outputs[0], axis=-1)[0]
        
        # Quantized ONNX prediction
        quantized_outputs = quantized_session.run(None, ort_inputs)
        quantized_preds = np.argmax(quantized_outputs[0], axis=-1)[0]
        
        # Align predictions with ground truth word tokens
        word_ids = inputs.word_ids(batch_index=0)
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
            correct_original += sum(1 for o, t in zip(original_aligned, true_labels) if o == t)
            correct_onnx += sum(1 for o, t in zip(onnx_aligned, true_labels) if o == t)
            correct_quantized += sum(1 for q, t in zip(quantized_aligned, true_labels) if q == t)
            total_tokens += len(true_labels)
            
    accuracy_original = correct_original / total_tokens if total_tokens > 0 else 0.0
    accuracy_onnx = correct_onnx / total_tokens if total_tokens > 0 else 0.0
    accuracy_quantized = correct_quantized / total_tokens if total_tokens > 0 else 0.0
    
    print(f"Original Model Accuracy:  {accuracy_original:.4f}")
    print(f"ONNX Model Accuracy:      {accuracy_onnx:.4f}")
    print(f"Quantized Model Accuracy: {accuracy_quantized:.4f}")
    
    return {
        'original_accuracy': accuracy_original,
        'onnx_accuracy': accuracy_onnx,
        'quantized_accuracy': accuracy_quantized
    }

def save_optimization_results(
    benchmark_results: Dict[str, Any],
    accuracy_results: Dict[str, Any],
    config: OptimizationConfig
) -> Dict[str, Any]:
    """
    Save optimization benchmarking results and summaries to optimization_results.json.
    """
    results = {
        'benchmark': benchmark_results,
        'accuracy': accuracy_results,
        'config': {
            'model_path': config.MODEL_PATH,
            'max_length': config.MAX_LENGTH,
            'iterations': config.N_ITERATIONS
        },
        'summary': {
            'size_reduction': {
                'pytorch_to_onnx': (
                    benchmark_results['pytorch']['model_size_mb'] - 
                    benchmark_results['onnx']['model_size_mb']
                ),
                'pytorch_to_quantized': (
                    benchmark_results['pytorch']['model_size_mb'] - 
                    benchmark_results['quantized']['model_size_mb']
                ),
                'percentage_reduction': (
                    (1 - benchmark_results['quantized']['model_size_mb'] / 
                     benchmark_results['pytorch']['model_size_mb']) * 100
                )
            },
            'speedup': {
                'pytorch_to_onnx': (
                    benchmark_results['pytorch']['avg_latency_ms'] / 
                    benchmark_results['onnx']['avg_latency_ms']
                ),
                'pytorch_to_quantized': (
                    benchmark_results['pytorch']['avg_latency_ms'] / 
                    benchmark_results['quantized']['avg_latency_ms']
                )
            }
        }
    }
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUT_DIR, "optimization_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nOptimization results saved to {output_path}")
    
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Model size reduction: {results['summary']['size_reduction']['percentage_reduction']:.1f}%")
    print(f"Quantized model speedup: {results['summary']['speedup']['pytorch_to_quantized']:.2f}x")
    print(f"Accuracy drop: {(accuracy_results['original_accuracy'] - accuracy_results['quantized_accuracy']):.4f}")
    
    return results

def create_deployment_artifacts(
    config: OptimizationConfig,
    tokenizer: Any,
    quantized_path: str,
    final_results: Dict[str, Any]
) -> str:
    """
    Assemble and package all artifacts required for production model deployment.
    """
    artifacts_dir = os.path.join(config.OUTPUT_DIR, "deployment")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. Save tokenizer
    tokenizer.save_pretrained(artifacts_dir)
    
    # 2. Save model config details
    model_config = {
        "model_type": "onnx_quantized",
        "max_length": config.MAX_LENGTH,
        "label_names": config.LABEL_NAMES,
        "num_labels": config.NUM_LABELS,
        "optimization_results": {
            "latency_ms": final_results['benchmark']['quantized']['avg_latency_ms'],
            "size_mb": final_results['benchmark']['quantized']['model_size_mb'],
            "accuracy": final_results['accuracy']['quantized_accuracy']
        }
    }
    
    with open(os.path.join(artifacts_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
        
    # 3. Copy quantized ONNX model file
    shutil.copy2(quantized_path, os.path.join(artifacts_dir, "model.onnx"))
    
    # 4. Write requirements.txt
    requirements = [
        "onnxruntime>=1.15.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
        "numpy>=1.24.0"
    ]
    with open(os.path.join(artifacts_dir, "requirements.txt"), "w") as f:
        f.write("\n".join(requirements))
        
    # 5. Create inference utility script
    inference_template = '''import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import json

class MultilingualNER:
    def __init__(self, model_path, tokenizer_path):
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        with open(f"{tokenizer_path}/config.json", "r") as f:
            self.config = json.load(f)
            
    def predict(self, text, language="en"):
        # Inference wrapper implementation
        tokens = text.split()
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="np",
            truncation=True,
            max_length=self.config.get("max_length", 128),
            padding="max_length"
        )
        ort_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        outputs = self.session.run(None, ort_inputs)
        predictions = np.argmax(outputs[0], axis=-1)[0]
        
        word_ids = inputs.word_ids(batch_index=0)
        previous_word_idx = None
        predictions_aligned = []
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                predictions_aligned.append(predictions[idx])
            previous_word_idx = word_idx
            
        entities = []
        current_entity = None
        current_start = None
        current_label = None
        label_names = self.config.get("label_names")
        
        for i, (token, pred_idx) in enumerate(zip(tokens, predictions_aligned)):
            label = label_names[pred_idx]
            if label.startswith("B-"):
                if current_entity:
                    entities.append({
                        "entity": " ".join(tokens[current_start:i]),
                        "label": current_label,
                        "start": current_start,
                        "end": i
                    })
                current_label = label[2:]
                current_start = i
                current_entity = [token]
            elif label.startswith("I-") and current_label == label[2:]:
                current_entity.append(token)
            elif current_entity:
                entities.append({
                    "entity": " ".join(current_entity),
                    "label": current_label,
                    "start": current_start,
                    "end": i
                })
                current_entity = None
                current_label = None
                current_start = None
                
        if current_entity:
            entities.append({
                "entity": " ".join(current_entity),
                "label": current_label,
                "start": current_start,
                "end": len(tokens)
            })
            
        return entities

if __name__ == "__main__":
    model = MultilingualNER("model.onnx", ".")
    result = model.predict("Apple was founded by Steve Jobs in Cupertino.")
    print(result)
'''
    
    with open(os.path.join(artifacts_dir, "inference_example.py"), "w") as f:
        f.write(inference_template)
        
    print(f"\nDeployment artifacts created in {artifacts_dir}")
    return artifacts_dir
