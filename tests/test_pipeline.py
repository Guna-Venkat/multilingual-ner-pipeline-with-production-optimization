import os
import pytest
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.configs.config import (
    TeacherConfig, 
    OptunaConfig, 
    DistillationConfig, 
    OptimizationConfig, 
    ErrorAnalysisConfig
)
from src.utils.helpers import set_seed
from src.evaluation.metrics import compute_metrics_fn
from src.evaluation.error_analysis import predict_entities, calculate_overall_metrics

def test_configs_loading():
    """Test that all YAML configurations load successfully and expose expected properties."""
    teacher_cfg = TeacherConfig()
    assert isinstance(teacher_cfg.LANGUAGES, list)
    assert teacher_cfg.MAX_TRAIN_SAMPLES > 0
    
    optuna_cfg = OptunaConfig()
    assert optuna_cfg.N_TRIALS > 0
    
    distill_cfg = DistillationConfig()
    assert distill_cfg.ALPHA >= 0.0
    
    opt_cfg = OptimizationConfig()
    assert opt_cfg.MAX_LENGTH > 0
    
    err_cfg = ErrorAnalysisConfig()
    assert err_cfg.SAMPLE_SIZE > 0

def test_compute_metrics():
    """Test token classification evaluation metrics computation helper."""
    # Define class labels mapping
    label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    
    # Dummy predictions and labels
    predictions = np.array([
        [[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0], [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    ]) # Preds shape (2, 2, 7) -> indices: [0, 3] and [5, 0]
    
    labels = np.array([
        [0, 3],
        [5, -100] # -100 should be ignored in metric calculation
    ])
    
    metrics = compute_metrics_fn((predictions, labels), label_list)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 1.0

def test_set_seed():
    """Test random seed initialization stability."""
    set_seed(42)
    val1 = np.random.rand()
    set_seed(42)
    val2 = np.random.rand()
    assert val1 == val2

def test_error_analysis_calculation():
    """Test error analysis metrics helper functions."""
    label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    
    # Create dummy prediction records DataFrame
    df = pd.DataFrame([
        {
            'true_labels': [0, 1, 2, 0],
            'pred_labels': [0, 1, 2, 0]
        },
        {
            'true_labels': [0, 3, 4, 0],
            'pred_labels': [0, 3, 0, 0]  # One mismatch (I-ORG predicted as O)
        }
    ])
    
    results, true_labels, pred_labels = calculate_overall_metrics(df, label_names)
    assert 'overall_f1' in results
    assert len(true_labels) == 8
    assert len(pred_labels) == 8
