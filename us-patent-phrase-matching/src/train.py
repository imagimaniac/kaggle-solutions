"""Training logic for Patent Phrase Matching"""

import os
import gc
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import KFold
import sys
sys.path.append(str(Path(__file__).parent.parent))

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import evaluate

from src.config import (
    DATA_DIR, MODELS_DIR, OUTPUT_DIR,
    MODEL_NAME, MAX_LENGTH, BATCH_SIZE,
    LEARNING_RATE, EPOCHS, WARMUP_RATIO,
    TARGET_COLUMN, SEED
)
from src.dataset import load_data, create_input
from src.model import set_seed


def set_seed(seed: int = SEED):
    """Set random seeds"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_function(examples, tokenizer):
    """Tokenize text data"""
    return tokenizer(
        examples["input"],
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True
    )


def compute_metrics(eval_pred):
    """Compute Pearson correlation metric"""
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Pearson correlation
    from scipy.stats import pearsonr
    corr, _ = pearsonr(predictions, labels)
    
    return {"pearson": corr}


def train_model(
    train_df: pd.DataFrame,
    model_name: str = MODEL_NAME,
    output_dir: str = None,
    n_folds: int = 3
) -> Dict[str, Any]:
    """Train model with cross-validation"""
    
    set_seed(SEED)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare dataset
    train_df = create_input(train_df)
    train_dataset = Dataset.from_pandas(train_df[["input", TARGET_COLUMN]])
    
    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    # Rename column for labels
    train_dataset = train_dataset.rename_column(TARGET_COLUMN, "labels")
    
    # Model init
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir or str(MODELS_DIR / "fold_0"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pearson",
        greater_is_better=True,
        logging_dir=str(MODELS_DIR / "logs"),
        logging_steps=100,
        save_total_limit=2,
        seed=SEED,
        data_seed=SEED,
        report_to="none"
    )
    
    # Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model(str(MODELS_DIR / "final"))
    tokenizer.save_pretrained(str(MODELS_DIR / "final"))
    
    print(f"Model saved to {MODELS_DIR / 'final'}")
    
    return {"status": "trained"}


def train_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Train baseline with Sentence-BERT and cosine similarity"""
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing sentence-transformers...")
        os.system("pip install sentence-transformers")
        from sentence_transformers import SentenceTransformer
    
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create inputs
    train_df = create_input(train_df)
    test_df = create_input(test_df)
    
    # Encode
    print("Encoding training data...")
    train_embeddings = model.encode(
        train_df["input"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print("Encoding test data...")
    test_embeddings = model.encode(
        test_df["input"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # For baseline, we need both anchor/target separately
    # Create combined embeddings based on the text
    print("Computing baseline predictions...")
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Simple baseline: average of anchor and target as separate inputs
    train_anchor_emb = model.encode(train_df["anchor"].tolist())
    train_target_emb = model.encode(train_df["target"].tolist())
    
    # Cosine similarity
    test_anchor_emb = model.encode(test_df["anchor"].tolist())
    test_target_emb = model.encode(test_df["target"].tolist())
    
    # Calculate similarity
    baseline_scores = []
    for i in range(len(test_df)):
        anchor_emb = test_anchor_emb[i:i+1]
        target_emb = test_target_emb[i:i+1]
        sim = cosine_similarity(anchor_emb, target_emb)[0, 0]
        baseline_scores.append(sim)
    
    # Create submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "score": baseline_scores
    })
    
    return submission


def main():
    """Main training function"""
    print("=" * 60)
    print("Patent Phrase Matching - Training")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_df, test_df = load_data()
    print(f"Train: {len(train_df)} rows")
    print(f"Test: {len(test_df)} rows")
    
    # Set seed
    set_seed(SEED)
    
    # Train baseline first
    print("\n--- Training Baseline ---")
    submission = train_baseline(train_df, test_df)
    
    # Save baseline submission
    baseline_path = OUTPUT_DIR / "submission_baseline.csv"
    submission.to_csv(baseline_path, index=False)
    print(f"\nBaseline saved to {baseline_path}")
    
    # Train main model
    print("\n--- Training Main Model ---")
    train_result = train_model(train_df)
    print(f"Training result: {train_result}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()