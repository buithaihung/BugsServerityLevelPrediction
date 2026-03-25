# Bug Severity Level Prediction (BSP)

> Automatically classify the severity level of software bug reports using a range of NLP techniques — from classical ML to fine-tuned Large Language Models.

---

## Table of Contents

- [Overview](#overview)
- [Severity Labels](#severity-labels)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Data Augmentation](#data-augmentation)
- [Models](#models)
- [Evaluation Metric](#evaluation-metric)
- [Getting Started](#getting-started)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)

---

## Overview

Bug triaging is a critical but labor-intensive step in software engineering. This project explores automated **bug severity level prediction** from raw bug report text (summary + description), framing it as a 5-class text classification problem.

The project systematically benchmarks multiple model families — from TF-IDF + classical ML baselines up to decoder-only LLM fine-tuning — and evaluates the impact of various data augmentation strategies on an imbalanced dataset.

---

## Severity Labels

| Label | Description |
|---|---|
| **Blocker** | Blocks development or testing progress; system crashes |
| **Critical** | Major functionality broken, no workaround available |
| **Major** | Significant functionality affected, but workaround exists |
| **Minor** | Minor issues, cosmetic problems, or low-impact bugs |
| **Trivial** | Very minor cosmetic issues or documentation typos |

---

## Dataset

Bug reports collected from **4 Apache open-source projects**:

| Project | Domain |
|---|---|
| **Ambari** | Hadoop cluster management |
| **Camel** | Enterprise integration framework |
| **Derby** | Relational database (Java) |
| **Wicket** | Component-based web framework |

Each sample contains: `summary`, `description`, `text` (summary + description concatenated), `source` (project name), and `priority` (label).

**Class distribution (train set — before augmentation):**

| Label | Count |
|---|---|
| Major | 2,078 |
| Minor | 802 |
| Trivial | 147 |
| Critical | 132 |
| Blocker | 39 |

The dataset is notably **imbalanced**, which motivates the augmentation strategies explored in this project.

---

## Project Structure

```
BugSeverityLevelPrediction/
├── datasets/
│   ├── raw_datasets_processed/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── eda_aug/                        # EDA-augmented training data
│   ├── aeda_aug/                       # AEDA-augmented training data
│   ├── ease_aug/                       # EASE-augmented training data
│   └── llm_aug/                        # LLM-augmented training data
│
└── notebooks/
    ├── data_preprocessing.ipynb        # Raw data processing & train/val/test split
    ├── augmentations/
    │   ├── eda_text_augmentation.ipynb
    │   ├── aeda_text_augmentation.ipynb
    │   ├── ease_augmentation.ipynb
    │   └── llm-augmentation/
    │       └── llm-augmentaion-gpt-5.ipynb
    └── training/
        ├── ml-based-models/
        │   └── train-ml-based-models.ipynb
        ├── dl-based-models/
        │   ├── train-cnn-glove-6b-300d.ipynb
        │   └── train-lstm-glove-6b-300d.ipynb
        └── transformer-based-models/
            ├── train-transformer-encoder-only-models.ipynb
            ├── train-transformer-decoder-only-qwen.ipynb
            └── train-transformer-decoder-only-llama.ipynb
```

---

## Data Augmentation

To address class imbalance, four augmentation strategies are explored and compared:

### 1. EDA — Easy Data Augmentation
Applies word-level perturbations: **synonym replacement** (`alpha_sr=0.05`) and **random deletion** (`alpha_rd=0.1`), generating `num_aug=16` variants per sample. Implementation via [jasonwei20/eda_nlp](https://github.com/jasonwei20/eda_nlp).

### 2. AEDA — An Easier Data Augmentation
Randomly inserts punctuation marks (`. , ! ? ; :`) into the original text at a rate of `PUNC_RATIO=0.3`, generating up to 8 variants per sample. Simpler than EDA with no external resources required.

### 3. EASE — Extract, Acquire, Sift, Employ
Treats each **sentence** within a bug report as an independent training sample:
1. **Extract** — Tokenize the document into sentences (`nltk.sent_tokenize`)
2. **Acquire** — Inherit the document-level label (label propagation)
3. **Sift** — Filter sentences shorter than `min_length=15` characters
4. **Employ** — Merge filtered sentences with the original dataset

Achieves a **2.26× augmentation rate** (3,198 → 10,413 samples), keeping 64.2% of extracted sentences.

### 4. LLM Augmentation
Uses **GPT-5** to generate fully synthetic bug reports that mirror the severity characteristics of real samples. Features:
- **Per-class, per-project prompts** — tailored system/user messages for each of the 5 severity levels × 4 Apache projects (20 distinct prompt templates)
- **Strict generation requirements** — summary length, description length, severity characteristics, and variation strategy all specified
- Output in JSON format, parsed and merged into the training set

This is the most semantically rich augmentation method in the project.

---

## Models

### ML-based Models
- **Feature**: TF-IDF (`max_features=5000`, English stopwords removed)
- **Models**: Logistic Regression, XGBoost
- **Tuning**: GridSearchCV with 5-fold cross-validation, optimizing Macro F1

### DL-based Models (PyTorch)
Pre-trained **GloVe 6B 300d** embeddings with custom neural architectures:
- **TextCNN** — multiple parallel convolutional filters (`kernel_sizes=[2,3,4,5]`, `num_filters=128`)
- **BiLSTM** — 2-layer bidirectional LSTM (`hidden_dim=256`, `dropout=0.5`)

Both models use:
- Adam optimizer + `ReduceLROnPlateau` scheduler
- Early stopping (patience=5)
- Dual checkpointing: best validation loss & best Macro F1

### Transformer-based Models

| Type | Model | Framework |
|---|---|---|
| Encoder-only | `bert-base-cased` (BERT) | HuggingFace `Trainer` |
| Encoder-only | `roberta-base` (RoBERTa) | HuggingFace `Trainer` |
| Decoder-only | `Qwen/Qwen3-0.6B` | Unsloth + SFTTrainer |
| Decoder-only | `meta-llama/Llama-3.2-1B-Instruct` | Unsloth + SFTTrainer |

**Encoder models** are fine-tuned for sequence classification with standard `TrainingArguments` (cosine LR scheduler, warmup, best model selection by `eval_f1`).

**Decoder models** are fine-tuned using **LoRA** (r=16) via [Unsloth](https://github.com/unslothai/unsloth) with 4-bit quantization, using the following instruction format:

```
<|im_start|>system
You are an expert software engineer with extensive experience in bug triaging and severity classification...
<|im_start|>user
Bug Report:
{bug_content}

Classify into: Blocker / Critical / Major / Minor / Trivial
Provide your answer:
<|im_start|>assistant
<think>

</think>

{priority}
```

---

## Evaluation Metric

The primary metric is **Macro F1-score**, computed across all 5 classes equally (unweighted by class frequency). This is appropriate given the class imbalance in the dataset.

Additional metrics reported: Accuracy, Micro/Weighted Precision, Recall, F1, per-class classification report, and confusion matrix.

---

## Getting Started

### Prerequisites

```bash
# Core dependencies
pip install pandas numpy scikit-learn xgboost
pip install nltk torch transformers datasets
pip install unsloth trl

# For DL models — download GloVe embeddings
# https://nlp.stanford.edu/data/glove.6B.zip
# Place glove.6B.300d.txt at /root/glove.6B.300d.txt (or update GLOVE_PATH in config)
```

### Data Preparation

```bash
# Run preprocessing notebook first
notebooks/data_preprocessing.ipynb
```

### Training

Each model family has its own notebook. To switch augmentation strategies, uncomment the corresponding line in the config section:

```python
df_train = pd.read_csv("train.csv")             # No augmentation (baseline)
# df_train = pd.read_csv("train_eda.csv")       # EDA
# df_train = pd.read_csv("train_aeda.csv")      # AEDA
# df_train = pd.read_csv("train_ease.csv")      # EASE
# df_train = pd.read_csv("train-llm-aug-*.csv") # LLM augmentation
```

---

## Experiments

The project conducts a **grid of experiments** crossing model families × augmentation strategies:

| Augmentation | Logistic Reg | XGBoost | CNN | BiLSTM | BERT | RoBERTa | Qwen3 | LLaMA |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| None (baseline) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| EDA | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| AEDA | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| EASE | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| LLM Aug | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Results

All metrics are reported on the **test set** (Accuracy, Macro Precision, Macro Recall, Macro F1). Three sampling strategies are evaluated for each combination: no sampling, undersampling, and oversampling. **Bold + underline** = best in group.

### Full Results by Model × Augmentation

<details>
<summary><b>Baseline (no augmentation)</b></summary>

| Model | w/o sampling Acc/Pre/Rec/F1 | w undersampling Acc/Pre/Rec/F1 | w oversampling Acc/Pre/Rec/F1 |
|---|---|---|---|
| Logistic Regression + TF-IDF | 64.00 / 19.47 / 20.90 / 18.68 | **33.75 / 26.95 / 38.99 / 24.00** | 51.75 / 30.23 / 35.25 / 31.09 |
| XGBoost + TF-IDF | 64.50 / 22.11 / 22.87 / 21.84 | 30.00 / 24.40 / 38.13 / 21.93 | 54.00 / 30.95 / 30.60 / 30.59 |
| CNN + GloVe | 65.75 / 22.57 / 22.89 / 21.63 | 31.50 / 25.00 / 31.26 / 19.40 | 54.25 / 33.46 / 26.67 / 26.01 |
| LSTM + GloVe | 65.00 / 13.00 / 20.00 / 15.76 | 26.00 / 21.06 / 20.11 / 8.67 | 28.00 / 21.41 / 22.85 / 17.09 |
| BERT | **66.25 / 37.42 / 28.88 / 28.98** | 22.50 / 24.63 / 27.80 / 17.00 | 56.50 / **42.30** / 40.48 / **38.79** |
| RoBERTa | 65.50 / 34.76 / 27.44 / 28.36 | 25.25 / 5.05 / 20.00 / 20.00 | **57.25** / 37.78 / **40.91** / 37.19 |
| Qwen3-0.6B | 59.75 / 19.09 / 20.69 / 19.37 | 9.00 / 17.17 / 27.61 / 9.56 | 36.00 / 18.46 / 20.04 / 15.15 |
| LLaMA-3.2-1B | 51.00 / 18.23 / 20.29 / 19.07 | 7.68 / 16.40 / 27.08 / 9.41 | 30.73 / 17.63 / 19.65 / 14.92 |

</details>

<details>
<summary><b>EDA augmentation</b></summary>

| Model | w/o sampling Acc/Pre/Rec/F1 | w undersampling Acc/Pre/Rec/F1 | w oversampling Acc/Pre/Rec/F1 |
|---|---|---|---|
| Logistic Regression + TF-IDF | 64.00 / 22.96 / 24.05 / 23.32 | 43.25 / **28.40 / 37.87 / 28.25** | 59.00 / 30.44 / 29.72 / 29.84 |
| XGBoost + TF-IDF | **65.75 / 36.05** / 25.69 / 26.27 | 42.75 / 26.61 / 35.23 / 27.22 | 56.25 / 26.25 / 26.54 / 26.29 |
| CNN + GloVe | 58.00 / 26.25 / 25.13 / 25.02 | **51.75** / 25.38 / 26.49 / 25.70 | 59.50 / 24.58 / 23.89 / 23.91 |
| LSTM + GloVe | 65.00 / 13.00 / 20.00 / 15.76 | 4.25 / 7.55 / 16.92 / 1.82 | 50.75 / 27.40 / 28.96 / 25.98 |
| BERT | 65.50 / 33.31 / **26.34 / 26.98** | 28.75 / 24.93 / 28.69 / 19.44 | 58.25 / 31.36 / **30.12** / 28.10 |
| RoBERTa | 63.75 / 26.10 / 22.53 / 21.74 | 28.00 / 26.38 / 34.79 / 21.05 | **64.50 / 35.75** / 29.77 / **31.35** |
| Qwen3-0.6B | 21.75 / 5.94 / 19.79 / 9.14 | 4.75 / 0.95 / 20.00 / 1.81 | 25.25 / 5.05 / 20.00 / 8.06 |
| LLaMA-3.2-1B | 20.75 / 5.94 / 19.85 / 9.12 | 4.75 / 0.95 / 20.00 / 1.81 | 24.09 / 5.05 / 20.06 / 8.04 |

</details>

<details>
<summary><b>AEDA augmentation</b></summary>

| Model | w/o sampling Acc/Pre/Rec/F1 | w undersampling Acc/Pre/Rec/F1 | w oversampling Acc/Pre/Rec/F1 |
|---|---|---|---|
| Logistic Regression + TF-IDF | 63.50 / 22.24 / 23.41 / 22.58 | 43.75 / 28.37 / **36.89** / 28.52 | 58.25 / 28.63 / 27.67 / 28.00 |
| XGBoost + TF-IDF | **64.75** / 26.96 / 23.68 / 23.32 | 47.00 / 28.87 / 36.30 / **29.16** | 59.00 / 26.76 / 26.05 / 26.17 |
| CNN + GloVe | 63.25 / 32.52 / 26.50 / 26.89 | 51.75 / **29.48** / 29.88 / 29.06 | 62.75 / 27.02 / 25.26 / 25.39 |
| LSTM + GloVe | 63.50 / 17.94 / 19.90 / 16.56 | 61.25 / 12.96 / 18.85 / 15.36 | 47.00 / 26.35 / 29.34 / 25.32 |
| BERT | 60.50 / **35.36** / 26.26 / 26.63 | 17.75 / 23.46 / 29.61 / 11.91 | **65.25 / 36.70 / 31.86 / 33.47** |
| RoBERTa | 62.50 / 32.90 / **29.92 / 30.89** | 34.50 / 26.88 / 35.74 / 24.17 | 64.50 / 35.75 / 29.77 / 31.35 |
| Qwen3-0.6B | 25.25 / 5.05 / 20.00 / 8.06 | **63.50** / 20.26 / 20.75 / 18.54 | 25.25 / 5.05 / 20.00 / 8.06 |
| LLaMA-3.2-1B | 11.00 / 6.16 / 20.68 / 7.38 | **65.00** / 13.00 / 20.00 / 15.76 | 25.85 / 3.24 / 19.28 / 6.85 |

</details>

<details>
<summary><b>EASE augmentation</b></summary>

| Model | w/o sampling Acc/Pre/Rec/F1 | w undersampling Acc/Pre/Rec/F1 | w oversampling Acc/Pre/Rec/F1 |
|---|---|---|---|
| Logistic Regression + TF-IDF | 64.50 / 21.89 / 22.27 / 20.95 | 27.50 / 25.04 / **40.07 / 22.17** | 54.25 / 27.34 / 28.50 / 27.55 |
| XGBoost + TF-IDF | 65.25 / 22.50 / 22.62 / 21.40 | 27.50 / 25.41 / 36.74 / 21.01 | 52.25 / 34.49 / 33.52 / 33.59 |
| CNN + GloVe | 60.25 / **40.73** / 23.87 / 23.72 | **54.75 / 27.16** / 21.58 / 19.16 | 47.50 / 32.12 / 26.82 / 23.65 |
| LSTM + GloVe | 65.00 / 13.00 / 20.00 / 15.76 | 22.50 / 21.71 / 22.06 / 14.50 | 49.50 / 23.62 / 23.52 / 23.23 |
| BERT | **65.50** / 33.49 / 26.82 / 26.48 | 19.00 / 25.28 / 30.20 / 13.37 | 55.50 / 33.59 / 33.01 / 32.75 |
| RoBERTa | 64.75 / 37.67 / **29.71 / 30.92** | 15.00 / 21.59 / 30.49 / 12.47 | **62.50** / 40.28 / **40.28 / 34.86** |
| Qwen3-0.6B | 51.75 / 17.86 / 17.86 / 18.75 | 9.70 / 14.60 / 14.66 / 5.85 | 28.25 / **45.62** / 28.25 / 19.85 |
| LLaMA-3.2-1B | 56.82 / 16.16 / 20.68 / 17.38 | 10.75 / 13.21 / 16.97 / 5.42 | 31.02 / 41.28 / 32.71 / 18.40 |

</details>

<details>
<summary><b>LLM Augmentation</b></summary>

| Model | w/o sampling Acc/Pre/Rec/F1 | w undersampling Acc/Pre/Rec/F1 | w oversampling Acc/Pre/Rec/F1 |
|---|---|---|---|
| Logistic Regression + TF-IDF | 62.75 / 25.91 / 23.80 / 23.64 | 25.75 / 28.03 / 37.98 / 20.70 | 39.25 / 25.26 / 29.34 / 23.53 |
| XGBoost + TF-IDF | 62.50 / 35.82 / 25.02 / 25.89 | **28.25** / 28.38 / **43.66 / 22.85** | 46.75 / 33.94 / **38.66** / 33.90 |
| CNN + GloVe | 61.25 / 34.52 / 28.60 / 30.11 | 28.00 / **29.28** / 38.23 / 22.46 | 55.00 / 31.73 / 33.23 / 31.35 |
| LSTM + GloVe | **65.00** / 13.00 / 20.00 / 15.76 | 5.25 / 10.94 / 18.40 / 3.47 | **57.75** / 13.60 / 18.94 / 15.79 |
| BERT | 60.50 / 38.59 / 32.17 / 32.30 | 20.00 / 9.77 / 34.72 / 13.31 | 47.25 / 33.59 / 33.01 / 32.75 |
| RoBERTa | 63.25 / **50.07 / 32.73 / 35.05** | 22.75 / 23.77 / 35.31 / 18.18 | 55.75 / **38.50** / 38.19 / **37.59** |
| Qwen3-0.6B | 30.68 / 19.50 / 16.88 / 15.39 | 15.75 / 5.74 / 18.46 / 8.29 | 27.98 / 19.96 / 17.29 / 12.26 |
| LLaMA-3.2-1B | 34.00 / 17.64 / 19.54 / 14.26 | 21.75 / 21.75 / 19.79 / 9.14 | 29.25 / 18.06 / 20.02 / 11.36 |

</details>

---

### Average Performance by Augmentation Method

| Method | w/o sampling Acc/Pre/Rec/F1 | w undersampling Acc/Pre/Rec/F1 | w oversampling Acc/Pre/Rec/F1 |
|---|---|---|---|
| Baseline | **62.72** / 23.33 / 23.00 / 21.71 | 23.21 / 20.08 / 28.87 / 16.25 | 46.06 / 29.03 / 29.56 / 26.35 |
| EDA | 53.06 / 21.19 / 22.92 / 19.67 | 26.03 / 17.64 / 27.50 / 15.89 | 49.70 / 23.24 / 26.13 / 22.70 |
| AEDA | 51.78 / 22.39 / 23.79 / 20.29 | **48.06 / 22.91** / 28.50 / **21.56** | **50.98** / 23.69 / 26.15 / 23.08 |
| EASE | 61.73 / 25.41 / 22.98 / 21.92 | 23.34 / 21.75 / 26.60 / 14.24 | 47.60 / **34.79 / 30.83 / 26.74** |
| LLM Aug | 54.99 / **29.38 / 24.84 / 24.05** | 20.94 / 19.71 / **30.82** / 14.80 | 44.87 / 26.83 / 28.59 / 24.82 |

---

### Average Performance by Sampling Strategy

| Sampling Strategy | Accuracy | Precision | Recall | Macro F1 |
|---|---|---|---|---|
| w/o sampling | **62.67** | 27.83 | 24.81 | 23.70 |
| Undersampling | 29.85 | 21.75 | **29.90** | 19.52 |
| Oversampling | 52.77 | **30.17** | 31.04 | **28.89** |

---

### Average Performance by Model

| Model | w/o sampling Acc/Pre/Rec/F1 | w undersampling Acc/Pre/Rec/F1 | w oversampling Acc/Pre/Rec/F1 |
|---|---|---|---|
| Logistic Regression + TF-IDF | 63.75 / 22.49 / 22.89 / 21.83 | 34.80 / **27.36 / 38.36 / 24.73** | 52.50 / 28.38 / 30.10 / 28.00 |
| XGBoost + TF-IDF | **64.55** / 28.69 / 23.98 / 23.74 | 35.10 / 26.73 / 38.01 / 24.43 | 53.65 / 30.48 / 31.07 / 30.11 |
| CNN + GloVe | 61.70 / 31.32 / 25.40 / 25.47 | **43.55** / 27.26 / 29.49 / 23.16 | 55.80 / 29.78 / 27.17 / 26.06 |
| LSTM + GloVe | 64.70 / 13.99 / 19.98 / 15.92 | 23.85 / 14.84 / 19.27 / 8.76 | 46.60 / 22.48 / 24.72 / 21.48 |
| BERT | 63.65 / 35.63 / 28.09 / 28.27 | 21.60 / 21.61 / 30.20 / 15.01 | 56.55 / 35.51 / 33.70 / 33.17 |
| RoBERTa | 63.95 / **36.30 / 28.47 / 29.39** | 25.10 / 20.73 / 31.27 / 19.17 | **60.90 / 37.61 / 35.78 / 34.47** |
| Qwen3-0.6B | 37.84 / 13.49 / 19.04 / 14.14 | 20.54 / 11.74 / 20.30 / 8.81 | 28.55 / 18.83 / 21.12 / 12.68 |
| LLaMA-3.2-1B | 34.71 / 12.83 / 20.21 / 13.44 | 21.99 / 13.06 / 20.77 / 8.31 | 28.19 / 17.05 / 22.34 / 11.91 |

---

### Key Takeaways

- **RoBERTa** consistently achieves the highest Macro F1, especially when combined with LLM Augmentation (F1 = **35.05** w/o sampling) or oversampling (F1 = **34.47** on average).
- **Oversampling** outperforms both no-sampling and undersampling strategies on average Macro F1 (28.89 vs 23.70 vs 19.52).
- **LLM Augmentation** yields the best average Macro F1 in the no-sampling setting (24.05), suggesting synthetic data from GPT-5 provides higher-quality diversity than word-level perturbation methods.
- **EASE augmentation** shows the strongest boost for oversampling scenarios (avg F1 = 26.74), indicating sentence-level decomposition helps with minority class coverage.
- **Decoder-only LLMs** (Qwen3-0.6B, LLaMA-3.2-1B) underperform encoder models on this task at their current scale, suggesting that larger parameter counts may be needed for competitive results.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{bsp2026,
  author       = {Thai-Hung Bui},
  title        = {Bug Severity Level Prediction},
  year         = {2026},
  howpublished = {\url{https://github.com/buithaihung/BugServerityLevelPrediction}}
}
```
