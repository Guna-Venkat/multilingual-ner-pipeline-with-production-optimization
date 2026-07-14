import random
import numpy as np
import torch
import json
from typing import Dict, Any

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def export_statistics(dataset: Any, output_file: str = 'dataset_statistics.json') -> Dict[str, Any]:
    """Export basic dataset statistics to a JSON file."""
    stats = {
        'total_samples': len(dataset['train']) + len(dataset['validation']) + len(dataset['test']),
        'train_samples': len(dataset['train']),
        'val_samples': len(dataset['validation']),
        'test_samples': len(dataset['test']),
        'sample_metadata': {
            'tokens_per_sample': {
                'min': min([len(s['tokens']) for s in dataset['train']]),
                'max': max([len(s['tokens']) for s in dataset['train']]),
                'mean': float(np.mean([len(s['tokens']) for s in dataset['train']]))
            }
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics exported to {output_file}")
    return stats

import os

def save_model_artifacts(model: Any, tokenizer: Any, output_dir: str, config_dict: Dict[str, Any]) -> None:
    """Save model weight checkpoints, tokenizers, and parameter configurations."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    with open(os.path.join(output_dir, "config.json"), "w", encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Model artifacts saved to {output_dir}")

