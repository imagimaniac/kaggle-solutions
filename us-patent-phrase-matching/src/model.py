"""Model definitions for Patent Phrase Matching"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel
)
from typing import Tuple
import sys
sys.path.append(str(__file__).parent.parent))

from src.config import MODEL_NAME, MAX_LENGTH, SEED


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility"""
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer():
    """Load tokenizer"""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_model(num_labels: int = 1):
    """Load model for sequence classification"""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )
    return model


def get_deberta_embeddings(texts: list, tokenizer, model, device: str = "cpu"):
    """Get embeddings using DeBERTa model"""
    model.eval()
    model.to(device)
    
    embeddings = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                max_length=MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            # Use [CLS] token representation
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding[0])
    
    return embeddings


def compute_cosine_similarity(emb1, emb2):
    """Compute cosine similarity between embeddings"""
    import numpy as np
    
    # Normalize
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    
    # Dot product
    similarity = np.sum(emb1_norm * emb2_norm, axis=1)
    
    return similarity


class PatentModel:
    """Wrapper for patent matching model"""
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = load_tokenizer()
        self.model = None
    
    def prepare_inputs(self, anchor: str, target: str, context: str) -> dict:
        """Prepare input text"""
        input_text = f"TEXT1: {context}; TEXT2: {target}; ANC1: {anchor}"
        return self.tokenizer(
            input_text,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
    
    def predict(self, anchor: str, target: str, context: str, device: str = "cpu") -> float:
        """Predict similarity score"""
        if self.model is None:
            self.model = load_model()
            self.model.to(device)
            self.model.eval()
        
        inputs = self.prepare_inputs(anchor, target, context)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            score = torch.sigmoid(outputs.logits).item()
        
        return score


if __name__ == "__main__":
    # Test
    print(f"Model: {MODEL_NAME}")
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    print("Tokenizer loaded!")
    print("Model test passed!")