#!/usr/bin/env python3
"""
U.S. Patent Phrase to Phrase Matching
DeBERTa-v3-small Fine-tuning for Top Scores (0.80+)

This script fine-tunes DeBERTa for patent phrase similarity prediction.
Expected: ~0.80 Pearson with proper GPU training.
"""

import os
import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
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
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============== CONFIG ==============
class CFG:
    # Model
    model_name = 'microsoft/deberta-v3-small'
    max_length = 256
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 3
    warmup_ratio = 0.1
    
    # Training
    n_folds = 3  # Use 3-fold CV
    seed = 42
    
    # Paths
    project_dir = Path('~/Projects/kaggle/us-patent-phrase-matching').expanduser()
    data_dir = project_dir / 'data'
    models_dir = project_dir / 'models'
    output_dir = project_dir / 'output'
    
    # Input format
    input_template = "TEXT1: {context}; TEXT2: {target}; ANC1: {anchor}"
    
    # Target
    target_col = 'score'

# ============== DATA LOADING ==============
def load_data():
    """Load training and test data"""
    train_df = pd.read_csv(CFG.data_dir / 'train.csv')
    test_df = pd.read_csv(CFG.data_dir / 'test.csv')
    sample_sub = pd.read_csv(CFG.data_dir / 'sample_submission.csv')
    
    print(f"Train: {len(train_df)} rows")
    print(f"Test: {len(test_df)} rows")
    
    return train_df, test_df, sample_sub

def create_input(df):
    """Create input text for model"""
    df = df.copy()
    df['input'] = df.apply(
        lambda x: CFG.input_template.format(
            context=str(x['context']),
            target=str(x['target']),
            anchor=str(x['anchor'])
        ),
        axis=1
    )
    return df

# ============== TOKENIZATION ==============
def load_tokenizer():
    """Load tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    return tokenizer

def tokenize_function(examples, tokenizer):
    """Tokenize text data"""
    return tokenizer(
        examples['input'],
        max_length=CFG.max_length,
        padding='max_length',
        truncation=True
    )

# ============== METRICS ==============
def compute_metrics(eval_pred):
    """Compute Pearson correlation"""
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    corr, _ = pearsonr(predictions, labels)
    return {'pearson': corr}

# ============== TRAINING ==============
def train_with_cv(train_df):
    """Train with cross-validation"""
    
    # Set seed
    set_seed(CFG.seed)
    
    # Prepare data
    train_df = create_input(train_df)
    tokenizer = load_tokenizer()
    
    # Convert to Dataset
    dataset = Dataset.from_pandas(train_df[['input', CFG.target_col]])
    
    # Tokenize
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    dataset = dataset.rename_column(CFG.target_col, 'labels')
    
    # K-Fold CV
    kfold = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    
    cv_scores = []
    fold_models = []
    
    print(f"\n{'='*50}")
    print(f"Starting {CFG.n_folds}-Fold Cross-Validation")
    print(f"{'='*50}")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n--- Fold {fold+1}/{CFG.n_folds} ---")
        
        # Split data
        train_fold = dataset.select(train_idx)
        val_fold = dataset.select(val_idx)
        
        print(f"Train: {len(train_fold)}, Val: {len(val_fold)}")
        
        # Model init
        def model_init():
            model = AutoModelForSequenceClassification.from_pretrained(
                CFG.model_name,
                num_labels=1
            )
            return model
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(CFG.models_dir / f'fold_{fold}'),
            num_train_epochs=CFG.num_epochs,
            per_device_train_batch_size=CFG.batch_size,
            per_device_eval_batch_size=CFG.batch_size * 2,
            learning_rate=CFG.learning_rate,
            warmup_ratio=CFG.warmup_ratio,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='pearson',
            greater_is_better=True,
            logging_dir=str(CFG.models_dir / 'logs'),
            logging_steps=100,
            save_total_limit=1,
            seed=CFG.seed + fold,
            fp16=True,  # Enable mixed precision
            dataloader_num_workers=4,
            report_to='none'
        )
        
        # Trainer
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_fold,
            eval_dataset=val_fold,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        trainer.train()
        
        # Evaluate
        eval_result = trainer.evaluate()
        fold_score = eval_result['eval_pearson']
        cv_scores.append(fold_score)
        
        print(f"Fold {fold+1} Pearson: {fold_score:.4f}")
        
        # Save model
        trainer.save_model(str(CFG.models_dir / f'final_fold_{fold}'))
        tokenizer.save_pretrained(str(CFG.models_dir / f'final_fold_{fold}'))
        
        # Clear memory
        del trainer
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # CV Results
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    print(f"\n{'='*50}")
    print(f"CV Results:")
    print(f"  Fold scores: {cv_scores}")
    print(f"  Mean Pearson: {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"{'='*50}")
    
    return mean_score, std_score

def predict_ensemble(test_df):
    """Generate predictions using ensemble of fold models"""
    
    print(f"\n{'='*50}")
    print("Generating Predictions with Ensemble")
    print(f"{'='*50}")
    
    # Prepare test data
    test_df = create_input(test_df)
    tokenizer = load_tokenizer()
    
    test_dataset = Dataset.from_pandas(test_df[['input', 'id']])
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['input', 'id']
    )
    
    all_predictions = []
    
    for fold in range(CFG.n_folds):
        model_path = CFG.models_dir / f'final_fold_{fold}'
        
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue
        
        print(f"Loading fold {fold} model...")
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        
        # Predict
        trainer = Trainer(model=model)
        
        predictions = trainer.predict(test_dataset)
        preds = predictions.predictions.flatten()
        
        # Apply sigmoid
        preds = 1 / (1 + np.exp(-preds))
        
        all_predictions.append(preds)
        
        del model, trainer
        gc.collect()
    
    # Ensemble (average)
    if all_predictions:
        ensemble_preds = np.mean(all_predictions, axis=0)
    else:
        raise ValueError("No models found!")
    
    return ensemble_preds

# ============== MAIN ==============
def main():
    print("="*60)
    print("DeBERTa Fine-tuning for Patent Phrase Matching")
    print("="*60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cpu':
        print("WARNING: Running on CPU! This will be very slow.")
        print("For best results, use GPU.")
    
    # Load data
    print("\nLoading data...")
    train_df, test_df, sample_sub = load_data()
    
    # Train with CV
    mean_score, std_score = train_with_cv(train_df)
    
    # Generate predictions
    predictions = predict_ensemble(test_df)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'score': predictions
    })
    
    # Save submission
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(CFG.output_dir / 'submission.csv', index=False)
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"CV Score: {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"Expected LB: ~{mean_score:.2f}")
    print(f"Submission saved to: {CFG.output_dir / 'submission.csv'}")
    print(f"{'='*50}")
    
    return mean_score

def set_seed(seed):
    """Set random seeds"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    main()