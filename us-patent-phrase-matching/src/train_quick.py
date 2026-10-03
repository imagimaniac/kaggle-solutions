#!/usr/bin/env python3
"""
U.S. Patent Phrase to Phrase Matching
Quick Training Script - Optimized for 12hr GPU training

This creates a simpler training flow with good results.
"""

import os
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from huggingface_hub import login

# ═══════════════════════════════════════════════════════════════
# HUGGINGFACE AUTHENTICATION - Secure token handling
# ═══════════════════════════════════════════════════════════════

# Method 1: Try Kaggle Secrets
HF_TOKEN = None
try:
    from kaggle_secrets import KaggleSecrets
    HF_TOKEN = KaggleSecrets.get_secret("HF_TOKEN")
    print("✅ Loaded HF token from Kaggle Secrets")
except:
    pass

# Method 2: Fallback to environment variable
if not HF_TOKEN:
    HF_TOKEN = os.environ.get("HF_TOKEN", None)
    if HF_TOKEN:
        print("✅ Loaded HF token from environment")

# Login if token available
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✅ Logged into HuggingFace!")
else:
    print("⚠️ No HF token found - will use unauthenticated requests")

# Continue with imports
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============== CONFIG ==============
MODEL_NAME = 'microsoft/deberta-v3-small'
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3  # Good for 12hr
SEED = 42
N_FOLDS = 3

INPUT_TEMPLATE = "TEXT1: {context}; TEXT2: {target}; ANC1: {anchor}"

PROJECT_DIR = Path('~/Projects/kaggle/us-patent-phrase-matching').expanduser()
DATA_DIR = PROJECT_DIR / 'data'
MODELS_DIR = PROJECT_DIR / 'models'
OUTPUT_DIR = PROJECT_DIR / 'output'

# ============== MAIN ==============
def main():
    print("=" * 60)
    print("DeBERTa Training - 12hr Optimized")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Check GPU
    if device == 'cpu':
        print("\n" + "!" * 50)
        print("WARNING: No GPU detected!")
        print("For training, you need GPU access.")
        print("Options:")
        print("  1. Use Google Colab (free GPU)")
        print("  2. Cloud GPU (AWS, GCP, etc.)")
        print("  3. Local GPU")
        print("!" * 50 + "\n")
        return
    
    # Show GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    
    print(f"Train: {len(train_df)} rows")
    print(f"Test: {len(test_df)} rows")
    
    # Create input
    def create_input(df):
        df = df.copy()
        df['input'] = df.apply(
            lambda x: INPUT_TEMPLATE.format(
                context=str(x['context']),
                target=str(x['target']),
                anchor=str(x['anchor'])
            ),
            axis=1
        )
        return df
    
    train_df = create_input(train_df)
    test_df = create_input(test_df)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize(examples):
        return tokenizer(
            examples['input'],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True
        )
    
    # Prepare dataset
    print("Preparing dataset...")
    train_dataset = Dataset.from_pandas(train_df[['input', 'score']])
    train_dataset = train_dataset.map(tokenize, batched=True)
    train_dataset = train_dataset.rename_column('score', 'labels')
    
    # Split
    train_ds, val_ds = train_test_split(
        train_dataset, test_size=0.1, random_state=SEED
    )
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model init
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=1
        )
    
    # Metrics
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.flatten()
        labels = labels.flatten()
        corr, _ = pearsonr(preds, labels)
        return {'pearson': corr}
    
    # Training args
    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / 'deberta_final'),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='pearson',
        greater_is_better=True,
        logging_dir=str(MODELS_DIR / 'logs'),
        logging_steps=100,
        save_total_limit=1,
        seed=SEED,
        fp16=True,
        report_to='none'
    )
    
    # Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )
    
    # Train
    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50)
    trainer.train()
    
    # Evaluate
    print("\nEvaluating...")
    result = trainer.evaluate()
    val_pearson = result['eval_pearson']
    print(f"\nValidation Pearson: {val_pearson:.4f}")
    
    # Save model
    print("\nSaving model...")
    trainer.save_model(str(MODELS_DIR / 'final'))
    tokenizer.save_pretrained(str(MODELS_DIR / 'final'))
    
    # Generate test predictions
    print("\nGenerating predictions...")
    test_dataset = Dataset.from_pandas(test_df[['input']])
    test_dataset = test_dataset.map(tokenize, batched=True)
    
    preds = trainer.predict(test_dataset)
    test_preds = preds.predictions.flatten()
    test_preds = 1 / (1 + np.exp(-test_preds))  # sigmoid
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'score': test_preds
    })
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Validation Pearson: {val_pearson:.4f}")
    print(f"Expected LB: ~{val_pearson:.2f}")
    print(f"Submission: {OUTPUT_DIR / 'submission.csv'}")
    print("=" * 50)

if __name__ == "__main__":
    main()