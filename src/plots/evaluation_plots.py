import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Tuple

def plot_confusion_matrix(
    true_labels: List[str], 
    pred_labels: List[str], 
    output_dir: str, 
    title: str = "Confusion Matrix"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate and save a normalized seaborn heatmap representation of the confusion matrix.
    """
    # Filter out 'O' labels for clearer visualization of entity confusions
    filtered_true = []
    filtered_pred = []
    for t, p in zip(true_labels, pred_labels):
        if t != 'O' or p != 'O':
            filtered_true.append(t)
            filtered_pred.append(p)
            
    unique_labels = sorted(set(filtered_true + filtered_pred))
    cm = confusion_matrix(filtered_true, filtered_pred, labels=unique_labels)
    
    # Normalize confusion matrix rows
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return cm, cm_normalized

def visualize_error_patterns(error_df: pd.DataFrame, output_dir: str):
    """
    Plot Horizontal Bar chart, histogram, and bar plots representing top error patterns.
    """
    if error_df.empty:
        print("⚠️ Warning: Error patterns DataFrame is empty. Skipping visualization.")
        return

    plt.figure(figsize=(14, 10))
    
    # 1. Top 10 error patterns
    plt.subplot(2, 2, 1)
    top_patterns = error_df['pattern'].value_counts().head(10)
    top_patterns.plot(kind='barh', color='#EA4335')
    plt.title('Top 10 Error Patterns')
    plt.xlabel('Count')
    plt.gca().invert_yaxis()
    
    # 2. Total errors by language
    plt.subplot(2, 2, 2)
    errors_by_lang = error_df.groupby('language')['pattern'].count()
    errors_by_lang.plot(kind='bar', color='#4285F4')
    plt.title('Total Errors by Language')
    plt.xlabel('Language')
    plt.ylabel('Error Count')
    plt.xticks(rotation=45)
    
    # 3. Error length distribution
    plt.subplot(2, 2, 3)
    error_df['length'].plot(kind='hist', bins=20, edgecolor='black', color='#FBBC05')
    plt.title('Error Span Length Distribution')
    plt.xlabel('Span Length (tokens)')
    plt.ylabel('Frequency')
    
    # 4. Error types by entity
    plt.subplot(2, 2, 4)
    error_df['true_label'].value_counts().head(10).plot(kind='bar', color='#34A853')
    plt.title('Most Error-Prone True Labels')
    plt.xlabel('Entity Type')
    plt.ylabel('Error Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "error_patterns.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_cross_lingual_transfer(transfer_df: pd.DataFrame, output_dir: str):
    """
    Generate and save a heatmap of cross-lingual transfer metrics.
    """
    if transfer_df.empty:
        print("⚠️ Warning: Transfer DataFrame is empty. Skipping visualization.")
        return

    plt.figure(figsize=(10, 6))
    sns.heatmap(transfer_df[['f1_score']].T, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'F1 Score'})
    
    plt.title('Cross-Lingual Transfer Performance', fontsize=16)
    plt.xlabel('Target Language', fontsize=14)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "cross_lingual_transfer.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_boundary_errors(boundary_df: pd.DataFrame, output_dir: str):
    """
    Generate and save boundary error analysis plots.
    """
    if boundary_df.empty:
        return

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    boundary_df['type'].value_counts().plot(kind='bar', color=['#EA4335', '#4285F4'])
    plt.title('Boundary Error Types')
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    boundary_df['entity'].value_counts().head(10).plot(kind='bar', color='#34A853')
    plt.title('Boundary Errors by Entity Type')
    plt.xlabel('Entity Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "boundary_errors.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_interactive_error_visualizations(
    true_labels: List[str], 
    pred_labels: List[str], 
    language_df: pd.DataFrame, 
    output_dir: str
):
    """
    Create and export interactive Plotly-based diagnostic error dashboards to HTML format.
    """
    try:
        import plotly.express as px
        unique_labels = sorted(set(true_labels + pred_labels))
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
        
        # Interactive confusion matrix
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="True", color="Count"),
            x=unique_labels,
            y=unique_labels,
            title="Interactive Confusion Matrix",
            color_continuous_scale='Blues'
        )
        fig.update_xaxes(tickangle=45)
        
        os.makedirs(output_dir, exist_ok=True)
        cm_path = os.path.join(output_dir, "interactive_confusion_matrix.html")
        fig.write_html(cm_path)
        
        # Interactive bar plot of per-language performance
        lang_fig = px.bar(
            language_df.reset_index(),
            x='index',
            y='f1',
            title='F1 Score by Language',
            labels={'index': 'Language', 'f1': 'F1 Score'},
            color='f1',
            color_continuous_scale='Viridis'
        )
        
        lang_path = os.path.join(output_dir, "performance_by_language.html")
        lang_fig.write_html(lang_path)
        
        print("\nInteractive visualizations created:")
        print(f"  - {cm_path}")
        print(f"  - {lang_path}")
        
    except ImportError:
        print("Plotly not installed. Skipping interactive visualizations.")
