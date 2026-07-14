import yaml
from typing import Any, Dict, List

def load_yaml(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class TeacherConfig:
    def __init__(self, yaml_path: str = None, **kwargs):
        data = load_yaml(yaml_path) if yaml_path else {}
        data.update(kwargs)
        self.MODEL_NAME: str = data.get("model_name", "xlm-roberta-large")
        self.LANGUAGES: List[str] = data.get("languages", ["en", "de", "fr"])
        self.MAX_LENGTH: int = data.get("max_length", 128)
        self.BATCH_SIZE: int = data.get("batch_size", 16)
        self.EPOCHS: int = data.get("epochs", 3)
        self.NUM_EPOCHS: int = self.EPOCHS
        self.LEARNING_RATE: float = float(data.get("learning_rate", 2e-5))
        self.WEIGHT_DECAY: float = data.get("weight_decay", 0.01)
        self.SEED: int = data.get("seed", 42)
        self.OUTPUT_DIR: str = data.get("output_dir", "./models/teacher")
        self.MAX_TRAIN_SAMPLES: int = data.get("max_train_samples", 20000)
        
        # Add properties for teacher training compatibility
        self.NUM_LABELS: int = data.get("num_labels", 7)
        self.LABEL_NAMES: List[str] = data.get("label_names", ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"])
        self.FP16: bool = data.get("fp16", True)
        self.DATASET_NAME: str = data.get("dataset_name", "unimelb-nlp/wikiann")
        self.GRADIENT_ACCUMULATION_STEPS: int = data.get("gradient_accumulation_steps", 1)
        self.WARMUP_RATIO: float = data.get("warmup_ratio", 0.0)
        self.LOGGING_DIR: str = data.get("logging_dir", "./logs")

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class DistillationConfig:
    def __init__(self, yaml_path: str = None, **kwargs):
        data = load_yaml(yaml_path) if yaml_path else {}
        data.update(kwargs)
        self.TEACHER_MODEL_PATH: str = data.get("teacher_model_path", "./models/teacher")
        self.STUDENT_MODEL_NAME: str = data.get("student_model_name", "xlm-roberta-base")
        self.LANGUAGES: List[str] = data.get("languages", ["en", "de", "fr"])
        self.MAX_LENGTH: int = data.get("max_length", 128)
        self.BATCH_SIZE: int = data.get("batch_size", 16)
        self.EPOCHS: int = data.get("epochs", 3)
        self.NUM_EPOCHS: int = self.EPOCHS
        self.LEARNING_RATE: float = float(data.get("learning_rate", 2e-5))
        self.WEIGHT_DECAY: float = data.get("weight_decay", 0.01)
        self.TEMPERATURE: float = float(data.get("temperature", 2.0))
        self.ALPHA: float = float(data.get("alpha", 0.5))
        self.SEED: int = data.get("seed", 42)
        self.OUTPUT_DIR: str = data.get("output_dir", "./models/student_distilled")
        self.MAX_TRAIN_SAMPLES: int = data.get("max_train_samples", 20000)
        
        # Add properties for training compatibility
        self.NUM_LABELS: int = data.get("num_labels", 7)
        self.LABEL_NAMES: List[str] = data.get("label_names", ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"])
        self.FP16: bool = data.get("fp16", True)
        self.DATASET_NAME: str = data.get("dataset_name", "unimelb-nlp/wikiann")
        self.GRADIENT_ACCUMULATION_STEPS: int = data.get("gradient_accumulation_steps", 1)
        self.WARMUP_RATIO: float = data.get("warmup_ratio", 0.0)
        self.LOGGING_DIR: str = data.get("logging_dir", "./logs")

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class OptunaConfig:
    def __init__(self, yaml_path: str = None, **kwargs):
        data = load_yaml(yaml_path) if yaml_path else {}
        data.update(kwargs)
        self.STUDENT_MODEL: str = data.get("student_model", "xlm-roberta-base")
        self.TEACHER_MODEL: str = data.get("teacher_model", "./models/teacher")
        self.DATASET_NAME: str = data.get("dataset_name", "unimelb-nlp/wikiann")
        self.LANGUAGES: List[str] = data.get("languages", ["en", "de"])
        self.N_TRIALS: int = data.get("n_trials", 10)
        self.OUTPUT_DIR: str = data.get("output_dir", "./models/optuna_tuned")
        self.MAX_LENGTH: int = data.get("max_length", 128)
        
        # Add properties for training/tuning compatibility
        self.LABEL_NAMES: List[str] = data.get("label_names", ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"])
        self.NUM_LABELS: int = data.get("num_labels", 7)
        self.SEED: int = data.get("seed", 42)
        self.MAX_TRAIN_SAMPLES: int = data.get("max_train_samples", 20000)
        self.FP16: bool = data.get("fp16", True)
        self.GRADIENT_ACCUMULATION_STEPS: int = data.get("gradient_accumulation_steps", 1)
        self.WARMUP_RATIO: float = data.get("warmup_ratio", 0.0)
        self.LOGGING_DIR: str = data.get("logging_dir", "./logs")
        self.DB_PATH: str = data.get("db_path", "sqlite:///optuna.db")
        self.STUDY_NAME: str = data.get("study_name", "multilingual_ner_distillation")

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class OptimizationConfig:
    def __init__(self, yaml_path: str = None, **kwargs):
        data = load_yaml(yaml_path) if yaml_path else {}
        data.update(kwargs)
        self.MODEL_PATH: str = data.get("model_path", "./models/student_distilled")
        self.OUTPUT_DIR: str = data.get("output_dir", "./models/optimized")
        self.MAX_LENGTH: int = data.get("max_length", 128)
        self.N_ITERATIONS: int = data.get("n_iterations", 100)
        self.LABEL_NAMES: List[str] = data.get("label_names", ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"])
        self.NUM_LABELS: int = data.get("num_labels", 7)
        self.BATCH_SIZE: int = data.get("batch_size", 1)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class ErrorAnalysisConfig:
    def __init__(self, yaml_path: str = None, **kwargs):
        data = load_yaml(yaml_path) if yaml_path else {}
        data.update(kwargs)
        self.MODEL_PATH: str = data.get("model_path", "./models/optimized/deployment")
        self.ORIGINAL_MODEL_PATH: str = data.get("original_model_path", "./models/optuna_tuned/final_model")
        self.DATASET_NAME: str = data.get("dataset_name", "unimelb-nlp/wikiann")
        self.LANGUAGES: List[str] = data.get("languages", ["en", "de", "fr", "es", "ru"])
        self.MAX_LENGTH: int = data.get("max_length", 128)
        self.SAMPLE_SIZE: int = data.get("sample_size", 1000)
        self.LABEL_NAMES: List[str] = data.get("label_names", [
            "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"
        ])
        self.OUTPUT_DIR: str = data.get("output_dir", "./error_analysis_results")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.MODEL_PATH,
            "original_model_path": self.ORIGINAL_MODEL_PATH,
            "dataset_name": self.DATASET_NAME,
            "languages": self.LANGUAGES,
            "max_length": self.MAX_LENGTH,
            "sample_size": self.SAMPLE_SIZE,
            "label_names": self.LABEL_NAMES,
            "output_dir": self.OUTPUT_DIR
        }
