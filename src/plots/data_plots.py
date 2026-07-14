import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Any, List

def plot_ner_distribution(dataset: Any, output_path: str = 'ner_tag_distribution.png') -> None:
    """Plot distribution of NER tags in train, validation, and test splits."""
    all_tags = []
    for split in ['train', 'validation', 'test']:
        tags = [tag for sample in dataset[split] for tag in sample['ner_tags']]
        all_tags.extend(tags)
    
    tag_counts = Counter(all_tags)
    
    # Map tag indices to names (WikiANN uses CoNLL-2003 format)
    tag_names = {
        0: 'O',
        1: 'B-PER', 2: 'I-PER',
        3: 'B-ORG', 4: 'I-ORG',
        5: 'B-LOC', 6: 'I-LOC'
    }
    
    tag_labels = [tag_names.get(tag, str(tag)) for tag in tag_counts.keys()]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(tag_counts)), tag_counts.values())
    plt.xticks(range(len(tag_counts)), tag_labels, rotation=45)
    plt.title('NER Tag Distribution')
    plt.xlabel('Tag')
    plt.ylabel('Count')
    
    # Add count labels on bars
    for bar, count in zip(bars, tag_counts.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"NER tag distribution saved to {output_path}")

def analyze_sentence_lengths(dataset: Any, output_path: str = 'sentence_length_analysis.png') -> List[int]:
    """Plot sentence length histograms, boxplots, and log distributions."""
    lengths = []
    for split in ['train', 'validation', 'test']:
        split_lengths = [len(sample['tokens']) for sample in dataset[split]]
        lengths.extend(split_lengths)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Sentence Length Distribution')
    plt.xlabel('Number of tokens')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(lengths), color='r', linestyle='--', 
                label=f'Mean: {np.mean(lengths):.1f}')
    plt.axvline(np.median(lengths), color='g', linestyle='--', 
                label=f'Median: {np.median(lengths):.1f}')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.boxplot(lengths)
    plt.title('Box Plot of Sentence Lengths')
    plt.ylabel('Number of tokens')
    
    plt.subplot(1, 3, 3)
    plt.hist(np.log1p(lengths), bins=50, edgecolor='black', alpha=0.7)
    plt.title('Log-transformed Length Distribution')
    plt.xlabel('log(1 + Number of tokens)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Sentence length analysis saved to {output_path}")
    print(f"Statistics:")
    print(f"  Min length: {min(lengths)}")
    print(f"  Max length: {max(lengths)}")
    print(f"  Mean length: {np.mean(lengths):.2f}")
    print(f"  Std length: {np.std(lengths):.2f}")
    print(f"  95th percentile: {np.percentile(lengths, 95):.2f}")
    
    return lengths

def compare_languages(multilingual_data: Dict[str, Any], output_path: str = 'language_comparison.png') -> pd.DataFrame:
    """Compare dataset volumes, average sentence lengths, and tag diversities across languages."""
    comparison_stats = []
    
    for lang, dataset in multilingual_data.items():
        train_size = len(dataset['train'])
        val_size = len(dataset['validation'])
        test_size = len(dataset['test'])
        
        # Calculate average sentence length using column-first indexing (from original fix)
        tokens_slice = dataset['train']['tokens'][:1000]
        avg_length = np.mean([len(t) for t in tokens_slice])
        
        # Calculate tag diversity using column-first indexing (from original fix)
        tags_slice = dataset['train']['ner_tags'][:1000]
        all_tags = [tag for sentence_tags in tags_slice for tag in sentence_tags]
        tag_diversity = len(set(all_tags))
        
        comparison_stats.append({
            'language': lang,
            'train_samples': train_size,
            'val_samples': val_size,
            'test_samples': test_size,
            'avg_sentence_length': avg_length,
            'tag_diversity': tag_diversity
        })
    
    df = pd.DataFrame(comparison_stats)
    
    # Visualization Code
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.bar(df['language'], df['train_samples'], color='skyblue')
    plt.title('Training Samples per Language')
    plt.ylabel('Number of samples')
    
    plt.subplot(2, 2, 2)
    plt.bar(df['language'], df['avg_sentence_length'], color='salmon')
    plt.title('Average Sentence Length per Language')
    plt.ylabel('Average tokens')
    
    plt.subplot(2, 2, 3)
    plt.bar(df['language'], df['tag_diversity'], color='lightgreen')
    plt.title('NER Tag Diversity per Language')
    plt.ylabel('Unique tags')
    
    plt.subplot(2, 2, 4)
    plt.pie(df['train_samples'], labels=df['language'], autopct='%1.1f%%', startangle=140)
    plt.title('Data Distribution Across Languages')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Language comparison saved to {output_path}")
    return df
