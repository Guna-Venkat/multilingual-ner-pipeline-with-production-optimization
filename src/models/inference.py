import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List, Dict, Any

class MultilingualNER:
    """
    Production-grade inference pipeline for running token classification using an optimized ONNX model.
    """
    def __init__(self, model_path: str, tokenizer_path: str):
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load configurations (such as label names) from the tokenizer config.json or a custom file
        with open(f"{tokenizer_path}/config.json", "r") as f:
            self.config = json.load(f)
            
        self.label_names = self.config.get("label_names", [
            "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"
        ])
        
    def predict(self, text: str, language: str = "en") -> List[Dict[str, Any]]:
        """
        Run inference on the input text to extract NER entities.
        """
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
        
        # Align predictions back to words
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
        
        for i, (token, pred_idx) in enumerate(zip(tokens, predictions_aligned)):
            if pred_idx >= len(self.label_names):
                label = "O"
            else:
                label = self.label_names[pred_idx]
                
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
                if current_entity:
                    current_entity.append(token)
            else:
                if current_entity:
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
