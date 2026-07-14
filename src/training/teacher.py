import os
import torch
import wandb
from functools import partial
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from typing import Dict, Any, Optional
from src.configs.config import TeacherConfig
from src.data.dataset import load_multilingual_dataset, preprocess_dataset
from src.evaluation.metrics import compute_metrics_fn
from src.utils.helpers import set_seed, save_model_artifacts

def run_teacher_training(
    config: TeacherConfig,
    dataset: Optional[Any] = None,
    report_to: str = "none"
) -> Dict[str, Any]:
    """
    Initialize, configure, and train the teacher model on the multilingual dataset.
    """
    # 1. Set seed for reproducibility
    set_seed(config.SEED)
    
    # 2. Initialize WandB if requested
    if report_to == "wandb":
        try:
            wandb.init(
                project="multilingual-ner-teacher",
                config=config.to_dict()
            )
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}. Defaulting report_to='none'.")
            report_to = "none"
            
    # 3. Load Multilingual Dataset if not provided
    if dataset is None:
        print("Loading multilingual dataset...")
        dataset = load_multilingual_dataset(
            languages=config.LANGUAGES,
            max_train_samples=config.MAX_TRAIN_SAMPLES,
            dataset_name=config.DATASET_NAME,
            seed=config.SEED
        )
        
    # 4. Load Tokenizer & Model
    print(f"Loading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    print(f"Loading model: {config.MODEL_NAME}")
    model = AutoModelForTokenClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        id2label={i: label for i, label in enumerate(config.LABEL_NAMES)},
        label2id={label: i for i, label in enumerate(config.LABEL_NAMES)},
        ignore_mismatched_sizes=True
    )
    
    # 5. Preprocess/Tokenize datasets
    print("Tokenizing and aligning datasets...")
    tokenized_dataset = preprocess_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    # 6. Setup Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # 7. Device handling (FP16 is only supported on CUDA)
    use_fp16 = config.FP16 and torch.cuda.is_available()
    if config.FP16 and not torch.cuda.is_available():
        print("Warning: FP16 training requested but CUDA is not available. Running in FP32.")
        
    # 8. Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE * 2,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        logging_dir=config.LOGGING_DIR,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=report_to,
        save_total_limit=2,
        fp16=use_fp16,
        group_by_length=True,
        push_to_hub=False,
    )
    
    # Create partial function of compute_metrics using labels from config
    compute_metrics = partial(compute_metrics_fn, label_names=config.LABEL_NAMES)
    
    # 9. Instantiate Trainer
    print("Initializing Trainer...")
    import inspect
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_dataset["train"],
        "eval_dataset": tokenized_dataset["validation"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics
    }
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
        
    trainer = Trainer(**trainer_kwargs)
    
    # 10. Execute Training (Optional: skip if running validation check on CPU with mock/empty params)
    print("Starting teacher training...")
    trainer.train()
    
    # 11. Save model artifacts
    save_model_artifacts(
        model=trainer.model,
        tokenizer=tokenizer,
        output_dir=config.OUTPUT_DIR,
        config_dict={
            "model_name": config.MODEL_NAME,
            "languages": config.LANGUAGES,
            "max_length": config.MAX_LENGTH,
            "label_names": config.LABEL_NAMES,
            "training_params": {
                "batch_size": config.BATCH_SIZE,
                "learning_rate": config.LEARNING_RATE,
                "epochs": config.NUM_EPOCHS
            }
        }
    )
    
    return {
        "trainer": trainer,
        "tokenizer": tokenizer,
        "tokenized_dataset": tokenized_dataset
    }
