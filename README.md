# Humor Detection Benchmark & Experiments

This repository contains a complete experimental pipeline for text classification (humor detection), comparing traditional machine learning methods (SVM with TF-IDF and sentence embeddings) against fine-tuning modern transformer-based language models (BERT).

The project automates data splitting across multiple training set sizes and random seeds, runs systematic evaluations, logs all metrics, and generates learning curves showing how performance scales with data size.

# Humor Project structure

```text
├── datasets/
│   └── colbert_dataset.csv       # Source dataset (you will need to uploade one)
├── split/                        # Generated data splits
│   ├── val.csv
│   ├── test.csv
│   └── train_slices/             # Training subsets by size and seed
│       ├── Seed_7/
│       ├── Seed_10/
│       └── Seed_35/
├── test/
│   └── predictions.csv           # Model predictions on the test set
├── experiments_log.csv           # Consolidated log of all experiments and metrics
├── best_model.pth                # Best transformer model checkpoint
└── main.py                       # Main experiment script
```


## ⚙️ Configuration

Global parameters are defined at the top of `main.py`:

* **Transformer Model:** `bert-base-uncased`,
* **Embedding model for SVM** `sentence-transformers/all-MiniLM-L6-v2`.
* **Transformer model Hyperparameters:** `BATCH_SIZE = 16`, `EPOCHS = 10`, `LEARNING_RATE = 5e-5`, `MAX_LENGTH = 512`.
* **SVM Hyperparameters:** RBF kernel, `C = 1.0`, `gamma = 'scale'`.
* **Experiment Grid (`TRAIN_SPLIT_SIZES`):** Training subset sizes ranging from 25 to 1100 samples.
* **Random Seeds (`RANDOM_SEEDS`):** `(7, 10, 35)` for robust evaluation and variance reduction.
