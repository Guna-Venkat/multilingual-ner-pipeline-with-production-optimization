import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from evaluate import load
from sklearn.metrics import precision_recall_fscore_support, f1_score
from typing import List, Dict, Any, Tuple
from src.configs.config import ErrorAnalysisConfig

def load_test_data(languages: List[str], sample_size: int = 500, dataset_name: str = "unimelb-nlp/wikiann") -> Any:
    """
    Load test data splits for multiple languages from Hugging Face datasets.
    """
    all_data = []
    
    for lang in languages:
        try:
            print(f"Loading {lang} test data...")
            dataset = load_dataset(dataset_name, lang, trust_remote_code=True)
            test_data = dataset['test']
            
            # Subsample if sample_size is smaller than dataset size
            if sample_size < len(test_data):
                indices = np.random.choice(len(test_data), sample_size, replace=False)
                test_data = test_data.select(indices)
            
            # Append language tag column
            test_data = test_data.add_column("language", [lang] * len(test_data))
            all_data.append(test_data)
            
        except Exception as e:
            print(f"Failed to load {lang}: {e}")
            
    if not all_data:
        print("❌ Error: No test data loaded.")
        return None
        
    combined_data = concatenate_datasets(all_data)
    print(f"Total test samples loaded: {len(combined_data)}")
    return combined_data

def predict_entities(
    model: Any,
    tokenizer: Any,
    tokens: List[str],
    model_type: str = "PyTorch",
    max_length: int = 128
) -> List[int]:
    """
    Predict entities for a single tokenized sentence using PyTorch or ONNX model.
    """
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt" if model_type == "PyTorch" else "np",
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    
    if model_type == "PyTorch":
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
    else:
        # ONNX Inference Session
        ort_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        outputs = model.run(None, ort_inputs)
        predictions = np.argmax(outputs[0], axis=-1)[0]
        
    # Align predictions with original word tokens
    word_ids = inputs.word_ids(batch_index=0)
    aligned_predictions = []
    
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            aligned_predictions.append(predictions[idx])
        previous_word_idx = word_idx
        
    # Align lengths
    if len(aligned_predictions) > len(tokens):
        aligned_predictions = aligned_predictions[:len(tokens)]
    elif len(aligned_predictions) < len(tokens):
        aligned_predictions.extend([0] * (len(tokens) - len(aligned_predictions)))
        
    return aligned_predictions

def collect_predictions(
    model: Any,
    tokenizer: Any,
    test_data: Any,
    model_type: str = "PyTorch",
    max_samples: int = 1000,
    max_length: int = 128
) -> pd.DataFrame:
    """
    Generate predictions over the test dataset and return results in a Pandas DataFrame.
    """
    predictions_data = []
    
    for i, sample in enumerate(test_data):
        if i >= max_samples:
            break
            
        tokens = sample['tokens']
        true_labels = sample['ner_tags']
        language = sample['language']
        
        try:
            pred_labels = predict_entities(model, tokenizer, tokens, model_type, max_length)
            
            predictions_data.append({
                'id': i,
                'language': language,
                'tokens': tokens,
                'true_labels': true_labels,
                'pred_labels': pred_labels,
                'text': ' '.join(tokens[:50]) + ('...' if len(tokens) > 50 else ''),
                'length': len(tokens)
            })
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
            
    return pd.DataFrame(predictions_data)

def calculate_overall_metrics(
    df: pd.DataFrame,
    label_names: List[str]
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Compute overall evaluation metrics (precision, recall, F1, accuracy) using seqeval.
    """
    all_true = []
    all_pred = []
    
    for _, row in df.iterrows():
        true = row['true_labels']
        pred = row['pred_labels']
        min_len = min(len(true), len(pred))
        all_true.extend(true[:min_len])
        all_pred.extend(pred[:min_len])
        
    true_labels = [label_names[t] for t in all_true]
    pred_labels = [label_names[p] for p in all_pred]
    
    seqeval_metric = load("seqeval")
    
    true_sequences = [[label_names[l] for l in row['true_labels']] for _, row in df.iterrows()]
    pred_sequences = [[label_names[l] for l in row['pred_labels'][:len(row['true_labels'])]] for _, row in df.iterrows()]
    
    results = seqeval_metric.compute(
        predictions=pred_sequences,
        references=true_sequences
    )
    
    return results, true_labels, pred_labels

def analyze_by_language(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute classification performance metrics grouped per language.
    """
    language_results = {}
    
    for lang in df['language'].unique():
        lang_df = df[df['language'] == lang]
        
        all_true = []
        all_pred = []
        for _, row in lang_df.iterrows():
            true = row['true_labels']
            pred = row['pred_labels']
            min_len = min(len(true), len(pred))
            all_true.extend(true[:min_len])
            all_pred.extend(pred[:min_len])
            
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true, all_pred, average='weighted', zero_division=0
        )
        
        language_results[lang] = {
            'samples': len(lang_df),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true) if all_true else 0.0
        }
        
    return pd.DataFrame(language_results).T

def analyze_by_entity(df: pd.DataFrame, label_names: List[str]) -> pd.DataFrame:
    """
    Evaluate precision, recall, and F1 score per entity type.
    """
    from collections import defaultdict
    entity_results = defaultdict(lambda: {'true': 0, 'pred': 0, 'correct': 0})
    
    for _, row in df.iterrows():
        true = row['true_labels']
        pred = row['pred_labels']
        min_len = min(len(true), len(pred))
        
        for i in range(min_len):
            true_label = label_names[true[i]]
            pred_label = label_names[pred[i]]
            
            if true_label != 'O':
                entity_type = true_label[2:] if '-' in true_label else true_label
                entity_results[entity_type]['true'] += 1
                
                if pred_label != 'O':
                    pred_type = pred_label[2:] if '-' in pred_label else pred_label
                    entity_results[pred_type]['pred'] += 1
                    
                    if true_label == pred_label:
                        entity_results[entity_type]['correct'] += 1
                        
    entity_metrics = {}
    for entity_type, counts in entity_results.items():
        precision = counts['correct'] / counts['pred'] if counts['pred'] > 0 else 0
        recall = counts['correct'] / counts['true'] if counts['true'] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        entity_metrics[entity_type] = {
            'true_count': counts['true'],
            'pred_count': counts['pred'],
            'correct_count': counts['correct'],
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    return pd.DataFrame(entity_metrics).T

def analyze_error_patterns(df: pd.DataFrame, label_names: List[str]) -> pd.DataFrame:
    """
    Identify and extract incorrect prediction spans for analysis.
    """
    error_patterns = []
    
    for _, row in df.iterrows():
        tokens = row['tokens']
        true = row['true_labels']
        pred = row['pred_labels']
        min_len = min(len(tokens), len(true), len(pred))
        
        i = 0
        while i < min_len:
            true_label = label_names[true[i]]
            pred_label = label_names[pred[i]]
            
            if true_label != pred_label:
                start = i
                error_type = f"{true_label}->{pred_label}"
                
                while i < min_len and label_names[true[i]] != label_names[pred[i]]:
                    i += 1
                end = i
                error_span = ' '.join(tokens[start:end])
                
                error_patterns.append({
                    'pattern': error_type,
                    'span': error_span,
                    'length': end - start,
                    'language': row['language'],
                    'true_label': true_label,
                    'pred_label': pred_label
                })
            else:
                i += 1
                
    return pd.DataFrame(error_patterns)

def find_hard_examples(df: pd.DataFrame, n_examples: int = 10) -> List[Dict[str, Any]]:
    """
    Rank and return test instances with the lowest entity-level validation scores.
    """
    example_scores = []
    
    for idx, row in df.iterrows():
        true = row['true_labels']
        pred = row['pred_labels']
        min_len = min(len(true), len(pred))
        
        correct = sum(1 for t, p in zip(true[:min_len], pred[:min_len]) if t == p)
        accuracy = correct / min_len if min_len > 0 else 0.0
        
        example_scores.append({
            'id': idx,
            'language': row['language'],
            'text': row['text'],
            'accuracy': accuracy,
            'length': row['length'],
            'tokens': row['tokens'],
            'true_labels': row['true_labels'],
            'pred_labels': row['pred_labels']
        })
        
    example_scores.sort(key=lambda x: x['accuracy'])
    return example_scores[:n_examples]

def analyze_cross_lingual_transfer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate F1 metric for each target language under cross-lingual transfer conditions.
    """
    transfer_matrix = {}
    languages = sorted(df['language'].unique())
    
    for lang in languages:
        lang_df = df[df['language'] == lang]
        all_true = []
        all_pred = []
        
        for _, row in lang_df.iterrows():
            true = row['true_labels']
            pred = row['pred_labels']
            min_len = min(len(true), len(pred))
            all_true.extend(true[:min_len])
            all_pred.extend(pred[:min_len])
            
        f1 = f1_score(all_true, all_pred, average='weighted', zero_division=0)
        
        transfer_matrix[lang] = {
            'samples': len(lang_df),
            'f1_score': f1
        }
        
    return pd.DataFrame(transfer_matrix).T

def analyze_boundary_errors(df: pd.DataFrame, label_names: List[str]) -> pd.DataFrame:
    """
    Categorize entity start/continuation detection boundary errors (B-I tag confusion).
    """
    boundary_errors = []
    
    for _, row in df.iterrows():
        tokens = row['tokens']
        true = row['true_labels']
        pred = row['pred_labels']
        min_len = min(len(tokens), len(true), len(pred))
        
        for i in range(min_len):
            true_label = label_names[true[i]]
            pred_label = label_names[pred[i]]
            
            if true_label.startswith('B-') and pred_label.startswith('I-'):
                boundary_errors.append({
                    'type': 'B->I',
                    'entity': true_label[2:],
                    'token': tokens[i],
                    'language': row['language']
                })
            elif true_label.startswith('I-') and pred_label.startswith('B-'):
                boundary_errors.append({
                    'type': 'I->B',
                    'entity': true_label[2:],
                    'token': tokens[i],
                    'language': row['language']
                })
                
    return pd.DataFrame(boundary_errors)

def create_error_report(
    df: pd.DataFrame,
    error_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    language_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    overall_results: Dict[str, Any],
    config: ErrorAnalysisConfig
) -> Dict[str, Any]:
    """
    Generate, save, and return a comprehensive JSON and Markdown diagnostic report.
    """
    report = {
        'summary': {
            'total_samples': len(df),
            'languages_analyzed': list(df['language'].unique()),
            'overall_f1': overall_results.get('overall_f1', 0.0),
            'overall_precision': overall_results.get('overall_precision', 0.0),
            'overall_recall': overall_results.get('overall_recall', 0.0)
        },
        'per_language': language_df.to_dict('index'),
        'per_entity': entity_df.to_dict('index'),
        'error_patterns': {
            'total_errors': len(error_df),
            'top_patterns': error_df['pattern'].value_counts().head(10).to_dict() if not error_df.empty else {},
            'errors_by_language': error_df['language'].value_counts().to_dict() if not error_df.empty else {}
        },
        'cross_lingual': transfer_df.to_dict('index'),
        'hard_examples': [
            {
                'id': ex['id'],
                'language': ex['language'],
                'accuracy': ex['accuracy'],
                'text_preview': ex['text'][:100]
            }
            for ex in find_hard_examples(df, n_examples=5)
        ]
    }
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(config.OUTPUT_DIR, "error_analysis_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\nError analysis report saved to {report_path}")
    
    # Generate markdown report
    md_report = f"""# Multilingual NER Error Analysis Report

## Summary
- **Total Samples Analyzed**: {report['summary']['total_samples']}
- **Languages**: {', '.join(report['summary']['languages_analyzed'])}
- **Overall F1 Score**: {report['summary']['overall_f1']:.4f}
- **Overall Precision**: {report['summary']['overall_precision']:.4f}
- **Overall Recall**: {report['summary']['overall_recall']:.4f}

## Top Error Patterns
"""
    
    for pattern, count in report['error_patterns']['top_patterns'].items():
        md_report += f"- `{pattern}`: {count} occurrences\n"
        
    md_report += "\n## Recommendations\n"
    
    # Add smart diagnostic recommendations
    top_patterns = report['error_patterns']['top_patterns']
    if 'PER->ORG' in top_patterns:
        md_report += "- **Person vs Organization confusion**: Consider adding more diverse examples of organizations vs person names in training.\n"
    if 'LOC->ORG' in top_patterns:
        md_report += "- **Location vs Organization confusion**: Location names are being misclassified as organizations. Consider entity-context enhancement.\n"
    if any('B->I' in pattern for pattern in top_patterns):
        md_report += "- **Boundary detection issues**: Model struggles with multi-word entity boundaries. Consider adding span-focused training data.\n"
        
    lowest_lang = min(report['per_language'].items(), key=lambda x: x[1]['f1'])
    md_report += f"\n- **Lowest performing language**: {lowest_lang[0]} (F1: {lowest_lang[1]['f1']:.3f}). Consider adding more {lowest_lang[0]} training data.\n"
    
    md_report_path = os.path.join(config.OUTPUT_DIR, "error_analysis_report.md")
    with open(md_report_path, 'w') as f:
        f.write(md_report)
        
    print(f"Markdown report saved to {md_report_path}")
    return report
