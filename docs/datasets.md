# Dataset Profile: WikiANN Multilingual NER

This document details the characteristics, tokenization strategy, and preprocessing details of the dataset used.

---

## 📊 Dataset Profile

* **Name**: WikiANN (Hugging Face Namespace: `unimelb-nlp/wikiann`)
* **Task**: Named Entity Recognition (NER) / Token Classification
* **Languages**:
  * **Fine-Tuning/Distillation**: English (`en`), German (`de`), French (`fr`)
  * **Zero-Shot / Cross-Lingual Evaluation**: Spanish (`es`), Russian (`ru`)
* **Total Output Labels**: 7
  * `O`: Outside of any entity
  * `B-PER` / `I-PER`: Beginning / Inside of a Person entity
  * `B-ORG` / `I-ORG`: Beginning / Inside of an Organization entity
  * `B-LOC` / `I-LOC`: Beginning / Inside of a Location entity

---

## 🔀 Label Mapping Schema

The label names are mapped to integer values consistently:

| Label ID | Label Name | Description |
| :--- | :--- | :--- |
| **0** | `O` | Non-entity tokens |
| **1** | `B-PER` | Start of Person's name |
| **2** | `I-PER` | Continuation of Person's name |
| **3** | `B-ORG` | Start of Organization name |
| **4** | `I-ORG` | Continuation of Organization name |
| **5** | `B-LOC` | Start of Location name |
| **6** | `I-LOC` | Continuation of Location name |

---

## 🔤 Tokenization & Label Alignment

Because XLM-RoBERTa uses a SentencePiece tokenizer, it splits words into multiple subwords (e.g., `"Steve"` $\rightarrow$ `["_Steve"]`, `"Jobs"` $\rightarrow$ `["_Jobs"]`). This introduces alignment challenges during training.

### Subword Alignment Algorithm
To align token-level annotations with subword tokens:
1. Tokenize inputs, returning word IDs for each subword.
2. For special tokens (like `<s>` or `</s>`), set label ID to `-100` (ignored by PyTorch Loss and metrics).
3. For normal subwords:
   - Assign the label of the word to the **first subword token**.
   - Assign `-100` to subsequent subword tokens of the same word (or assign `I-` labels depending on token classification setup; here, the first subword receives the label, and others are ignored to avoid inflating evaluation stats).

---

## 📥 Dataset Subsampling for CPU Validation

To enable local pipeline testing on a standard CPU machine:
- The raw datasets are subsampled using `src/data/dataset.py`.
- Training splits are downsampled (e.g., 20,000 max samples).
- Test splits are downsampled (e.g., 100 or 500 samples per language) during local regression/error checks.
- Set-seed guarantees consistent, reproducible splits.
