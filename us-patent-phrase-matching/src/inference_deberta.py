#!/usr/bin/env python3
"""
U.S. Patent Phrase to Phrase Matching
DeBERTa Inference Script

Generates predictions using trained DeBERTa models.
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

# ============== CONFIG ==============
class CFG:
    model_name = 'microsoft/deberta-v3-small'
    max_length = 256
    batch_size = 16
    
    project_dir = Path('~/Projects/kaggle/us-patent-phrase-matching').expanduser()
    data_dir = project_dir / 'data'
    models_dir = project_dir / 'models'
    output_dir = project_dir / 'output'
    
    input_template = "TEXT1: {context}; TEXT2: {target}; ANC1: {anchor}"

def load_test_data():
    """Load test data"""
    test_df = pd.read_csv(CFG.data_dir / 'test.csv')
    print(f"Test: {len(test_df)} rows")
    return test_df

def create_input(df):
    """Create input text"""
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

def tokenize(examples, tokenizer):
    """Tokenize"""
    return tokenizer(
        examples['input'],
        max_length=CFG.max_length,
        padding='max_length',
        truncation=True
    )

def predict_with_model(model_path, test_df):
    """Predict with a single model"""
    print(f"Loading model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    
    # Prepare data
    test_df = create_input(test_df)
    test_dataset = Dataset.from_pandas(test_df[['input', 'id']])
    test_dataset = test_dataset.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=['input', 'id']
    )
    
    # Get trainer
    from transformers import Trainer
    trainer = Trainer(model=model)
    
    # Predict
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.flatten()
    
    # Apply sigmoid
    preds = 1 / (1 + np.exp(-preds))
    
    return preds

def main():
    print("="*60)
    print("DeBERTa Inference")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load test data
    test_df = load_test_data()
    
    # Find all fold models
    model_dirs = sorted(CFG.models_dir.glob('final_fold_*'))
    print(f"Found {len(model_dirs)} models")
    
    if not model_dirs:
        print("ERROR: No trained models found!")
        print("Please run train_deberta.py first")
        return
    
    # Ensemble predictions
    all_preds = []
    
    for model_dir in model_dirs:
        preds = predict_with_model(model_dir, test_df.copy())
        all_preds.append(preds)
    
    # Average ensemble
    final_preds = np.mean(all_preds, axis=0)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'score': final_preds
    })
    
    # Save
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(CFG.output_dir / 'submission.csv', index=False)
    
    print(f"\nSaved: {CFG.output_dir / 'submission.csv'}")
    print(f"Score range: {final_preds.min():.4f} - {final_preds.max():.4f}")
    print(f"\nFirst 10 predictions:")
    print(submission.head(10))

if __name__ == "__main__":
    main()