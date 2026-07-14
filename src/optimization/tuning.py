import os
import gc
import json
import shutil
import numpy as np
import pandas as pd
import optuna
import torch
import wandb
from optuna.trial import TrialState
from functools import partial
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from typing import Dict, Any, Optional, List
from src.configs.config import OptunaConfig
from src.training.distillation import DistillationTrainer
from src.evaluation.metrics import compute_metrics_fn
from datetime import datetime

def objective(
    trial: optuna.trial.Trial,
    config: OptunaConfig,
    teacher_model: torch.nn.Module,
    tokenizer: Any,
    tokenized_dataset: Any
) -> float:
    """
    Optuna objective function for tuning student model distillation hyperparameters.
    """
    # 1. Initialize wandb in OFFLINE mode for this specific trial
    run = wandb.init(
        project="ner_distillation_optuna",
        name=f"trial_{trial.number}",
        mode="offline",
        reinit=True,
        anonymous="allow"
    )

    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 4)
    
    # Distillation specific parameters
    temperature = trial.suggest_float("temperature", 1.0, 5.0)
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    
    # Initialize Student Model
    model = AutoModelForTokenClassification.from_pretrained(
        config.STUDENT_MODEL,
        num_labels=config.NUM_LABELS,
        id2label={i: label for i, label in enumerate(config.LABEL_NAMES)},
        label2id={label: i for i, label in enumerate(config.LABEL_NAMES)}
    )
    
    # Ensure student model is on the same device as teacher
    device = next(teacher_model.parameters()).device
    model.to(device)
    
    # Training arguments
    trial_tmp_dir = f"./tmp_trial_{trial.number}"
    training_args = TrainingArguments(
        output_dir=trial_tmp_dir,
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        logging_steps=50,
        report_to="none",
        skip_memory_metrics=True,
        save_strategy="no",
        fp16=torch.cuda.is_available() and config.FP16,
        disable_tqdm=True 
    )
    
    # Use partial function for compute_metrics
    compute_metrics = partial(compute_metrics_fn, label_names=config.LABEL_NAMES)
    
    # Instantiate DistillationTrainer with compatibility signature checking
    import inspect
    trainer_kwargs = {
        "model": model,
        "teacher_model": teacher_model,
        "temperature": temperature,
        "alpha": alpha,
        "args": training_args,
        "train_dataset": tokenized_dataset["train"],
        "eval_dataset": tokenized_dataset["validation"],
        "data_collator": DataCollatorForTokenClassification(tokenizer),
        "compute_metrics": compute_metrics,
    }
    sig = inspect.signature(DistillationTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
        
    trainer = DistillationTrainer(**trainer_kwargs)
    
    try:
        trainer.train()
        eval_results = trainer.evaluate()
        score = eval_results.get("eval_f1", 0.0)
        
        # Cleanup
        run.finish()
        del model
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        shutil.rmtree(trial_tmp_dir, ignore_errors=True)
        if os.path.exists("./wandb"):
            shutil.rmtree("./wandb", ignore_errors=True)
        
        return score
        
    except Exception as e:
        if 'run' in locals():
            run.finish(exit_code=1)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutil.rmtree(trial_tmp_dir, ignore_errors=True)
        print(f"Trial {trial.number} failed: {e}")
        return 0.0

def run_optuna_study(
    config: OptunaConfig,
    teacher_model: torch.nn.Module,
    tokenizer: Any,
    tokenized_dataset: Any
) -> optuna.study.Study:
    """
    Initialize and run the Optuna hyperparameter tuning study.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Check if the DB is in memory or disk
    if config.DB_PATH.startswith("sqlite"):
        # Resolve absolute path for database file to prevent relative cwd issues
        db_file = config.DB_PATH.replace("sqlite:///", "")
        abs_db_path = f"sqlite:///{os.path.abspath(db_file)}"
    else:
        abs_db_path = config.DB_PATH
        
    print(f"Optuna database path: {abs_db_path}")
    
    study = optuna.create_study(
        direction="maximize",
        study_name=config.STUDY_NAME,
        storage=abs_db_path,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    
    objective_partial = partial(
        objective,
        config=config,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        tokenized_dataset=tokenized_dataset
    )
    
    print(f"Starting Optuna study '{config.STUDY_NAME}' with {config.N_TRIALS} trials...")
    study.optimize(
        objective_partial,
        n_trials=config.N_TRIALS,
        show_progress_bar=True
    )
    
    return study

def export_optimization_report(
    study: optuna.study.Study,
    df_trials: pd.DataFrame,
    config: OptunaConfig
) -> Optional[Dict[str, Any]]:
    """
    Export study optimization statistics and configuration details to a JSON file.
    """
    if df_trials.empty:
        print("❌ Error: No completed trials found. Cannot export report.")
        return None

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = {
        "study_info": {
            "study_name": study.study_name,
            "n_trials": len(study.trials),
            "best_value": study.best_value if study.best_trials else None,
            "best_params": study.best_params,
            "datetime": current_time
        },
        "config": {
            "student_model": config.STUDENT_MODEL,
            "languages": config.LANGUAGES,
            "n_trials": config.N_TRIALS
        },
        "trials_summary": {
            "mean_f1": float(df_trials['f1_score'].mean()),
            "std_f1": float(df_trials['f1_score'].std()) if len(df_trials) > 1 else 0.0,
            "min_f1": float(df_trials['f1_score'].min()),
            "max_f1": float(df_trials['f1_score'].max())
        }
    }
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(config.OUTPUT_DIR, "optimization_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Optimization report saved to {report_path}")
    
    print("\n" + "="*30)
    print("      OPTIMIZATION SUMMARY      ")
    print("="*30)
    print(f"Best F1 Score:    {study.best_value:.4f}")
    print(f"Average F1 Score: {df_trials['f1_score'].mean():.4f} ± {df_trials['f1_score'].std():.4f}")
    print(f"Number of trials: {len(study.trials)}")
    
    print(f"\nBest parameters found:")
    for param, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {param:15}: {value:.6f}")
        else:
            print(f"  {param:15}: {value}")
    print("="*30)
    
    return report

def create_parameter_importance_table(study: optuna.study.Study) -> Dict[str, float]:
    """
    Analyze and display hyperparameter importance ranking using Random Forests.
    """
    evaluator = optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
    try:
        importances = optuna.importance.get_param_importances(study, evaluator=evaluator)
        
        print("\n" + "="*40)
        print(f"{'PARAMETER':<20} | {'IMPORTANCE SCORE'}")
        print("="*40)
        
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        for param, importance in sorted_importances:
            bar = "█" * int(importance * 20)
            print(f"{param:<20} | {importance:.4f}  {bar}")
            
        print("="*40)
        return importances
    except Exception as e:
        print(f"❌ Could not calculate importance: {e}")
        return {}

def analyze_hyperparameter_relationships(df_trials: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlations between parameters and print top performing trials.
    """
    numeric_cols = df_trials.select_dtypes(include=[np.number]).columns
    cols_to_corr = [col for col in numeric_cols if col != 'trial']
    corr_matrix = df_trials[cols_to_corr].corr()
    
    print("\n" + "="*30)
    print("TOP 5 PERFORMING TRIALS")
    print("="*30)
    
    actual_params = [col for col in df_trials.columns if col not in ['trial', 'f1_score']]
    top_trials = df_trials.nlargest(5, 'f1_score')
    print(top_trials[['trial', 'f1_score'] + actual_params].to_string(index=False))
    
    return corr_matrix
