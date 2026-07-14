import numpy as np
from datasets import load_dataset, DatasetDict, concatenate_datasets
from typing import Dict, Any, List
from transformers import AutoTokenizer

def load_multilingual_data(languages: List[str] = ['en', 'de', 'fr', 'es', 'ru']) -> Dict[str, Any]:
    """
    Load individual datasets from Hugging Face for multiple languages.
    Used for exploratory data analysis (EDA).
    """
    multilingual_data = {}
    for lang in languages:
        try:
            print(f"Loading {lang} data...")
            dataset = load_dataset("unimelb-nlp/wikiann", lang, trust_remote_code=True)
            multilingual_data[lang] = dataset
        except Exception as e:
            print(f"Failed to load {lang}: {e}")
    return multilingual_data

def load_multilingual_dataset(
    languages: List[str], 
    max_train_samples: int = 20000, 
    dataset_name: str = "unimelb-nlp/wikiann", 
    seed: int = 42
) -> DatasetDict:
    """
    Load, shuffle, subsample, and concatenate multiple languages into a single DatasetDict.
    Ensures language metadata columns are added to all splits.
    """
    datasets_dict = {}
    
    # 1. Load individual datasets from Hugging Face
    for lang in languages:
        try:
            print(f"Loading {lang} dataset...")
            datasets_dict[lang] = load_dataset(dataset_name, lang, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading {lang}: {e}")
    
    train_list, val_list, test_list = [], [], []
    
    # 2. Calculate how many samples to take per language to be fair
    samples_per_lang = max_train_samples // len(languages)
    eval_samples_per_lang = samples_per_lang // 4  # Keep validation smaller for speed

    for lang, ds in datasets_dict.items():
        # Shuffle and select a subset for each language to keep it representative
        train_sub = ds['train'].shuffle(seed=seed).select(range(min(samples_per_lang, len(ds['train']))))
        val_sub = ds['validation'].shuffle(seed=seed).select(range(min(eval_samples_per_lang, len(ds['validation']))))
        
        # Select a small portion of the test set for the specific language
        test_sub = ds['test'].shuffle(seed=seed).select(range(min(500, len(ds['test']))))
        
        # Add 'language' column to ALL splits to avoid ValueError during tokenization (.map)
        train_list.append(train_sub.add_column("language", [lang] * len(train_sub)))
        val_list.append(val_sub.add_column("language", [lang] * len(val_sub)))
        test_list.append(test_sub.add_column("language", [lang] * len(test_sub)))
        
    # 3. Combine everything into a single DatasetDict
    combined_dataset = DatasetDict({
        'train': concatenate_datasets(train_list).shuffle(seed=seed),
        'validation': concatenate_datasets(val_list).shuffle(seed=seed),
        'test': concatenate_datasets(test_list).shuffle(seed=seed)
    })
    
    print(f"\nCombined dataset sizes:")
    print(f"  Train: {len(combined_dataset['train'])}")
    print(f"  Validation: {len(combined_dataset['validation'])}")
    print(f"  Test: {len(combined_dataset['test'])}")
    
    return combined_dataset

def preprocess_dataset(dataset: Any, tokenizer: AutoTokenizer, max_length: int = 128) -> Any:
    """
    Preprocess and align labels for token classification.
    """
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
            padding="max_length"
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    # Same word as previous token (sub-word token) -> assign -100 to ignore in loss
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    return dataset.map(tokenize_and_align_labels, batched=True)

def create_train_val_split(dataset: Any, train_ratio: float = 0.8, seed: int = 42) -> DatasetDict:
    """
    Create train/validation split from 'train' split if not already provided.
    """
    split_dataset = dataset['train'].train_test_split(
        test_size=1-train_ratio, 
        seed=seed
    )
    return DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })
