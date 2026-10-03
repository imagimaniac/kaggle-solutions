"""Inference logic for Patent Phrase Matching"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    DATA_DIR, MODELS_DIR, OUTPUT_DIR,
    MODEL_NAME, MAX_LENGTH, BATCH_SIZE
)
from src.dataset import load_data, create_input
from src.model import set_seed


def load_trained_model(model_path: str = None):
    """Load trained model"""
    path = model_path or str(MODELS_DIR / "final")
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    
    return model, tokenizer


def predict_batch(
    texts: List[str],
    model,
    tokenizer,
    device: str = "cpu",
    batch_size: int = BATCH_SIZE
) -> np.ndarray:
    """Predict similarity scores for a batch of texts"""
    model.eval()
    model.to(device)
    
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Apply sigmoid for probability
            scores = torch.sigmoid(outputs.logits).cpu().numpy().flatten()
            predictions.extend(scores)
    
    return np.array(predictions)


def generate_submission(
    model_path: str = None,
    test_df: pd.DataFrame = None
) -> pd.DataFrame:
    """Generate final submission"""
    
    print("Loading model...")
    model, tokenizer = load_trained_model(model_path)
    
    # Load test data if not provided
    if test_df is None:
        _, test_df = load_data()
    
    # Prepare inputs
    test_df = create_input(test_df)
    texts = test_df["input"].tolist()
    
    print(f"Generating predictions for {len(texts)} samples...")
    
    # Predict
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions = predict_batch(texts, model, tokenizer, device=device)
    
    # Create submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "score": predictions
    })
    
    return submission


def ensemble_predictions(model_paths: List[str], test_df: pd.DataFrame) -> np.ndarray:
    """Ensemble predictions from multiple models"""
    
    all_predictions = []
    
    for model_path in model_paths:
        print(f"Predicting with {model_path}...")
        model, tokenizer = load_trained_model(model_path)
        
        test_df = create_input(test_df)
        texts = test_df["input"].tolist()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        preds = predict_batch(texts, model, tokenizer, device=device)
        
        all_predictions.append(preds)
    
    # Average ensemble
    ensemble = np.mean(all_predictions, axis=0)
    
    return ensemble


def main():
    """Main inference function"""
    print("=" * 60)
    print("Patent Phrase Matching - Inference")
    print("=" * 60)
    
    # Check for trained model
    model_path = str(MODELS_DIR / "final")
    
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please run training first!")
        return
    
    # Load test data
    _, test_df = load_data()
    print(f"Test data: {len(test_df)} rows")
    
    # Generate submission
    submission = generate_submission(model_path, test_df)
    
    # Save submission
    output_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Shape: {submission.shape}")
    print("\nFirst 10 predictions:")
    print(submission.head(10))
    
    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()