# Humor Detection Benchmark & Experiments

This repository contains a complete experimental pipeline for text classification (humor detection), comparing traditional machine learning methods (SVM with TF-IDF and sentence embeddings) against fine-tuning modern transformer-based language models (BERT).

The project automates data splitting across multiple training set sizes and random seeds, runs systematic evaluations, logs all metrics, and generates learning curves showing how performance scales with data size.

# Humor Project structure

├── datasets/
│   └── colbert_dataset.csv       # Source dataset (text + humor label)
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
├── best_model.pth                # Best model checkpoint
└── main.py                       # Main experiment script
