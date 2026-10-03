"""Data loading and processing utilities"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import DATA_DIR, INPUT_TEMPLATE, TARGET_COLUMN


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test data"""
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def create_input(df: pd.DataFrame) -> pd.DataFrame:
    """Create input text for model"""
    df = df.copy()
    df["input"] = df.apply(
        lambda x: INPUT_TEMPLATE.format(
            context=str(x.get("context", "")),
            target=str(x.get("target", "")),
            anchor=str(x.get("anchor", ""))
        ),
        axis=1
    )
    return df


def get_score_distribution(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Get score distribution statistics"""
    scores = train_df[TARGET_COLUMN]
    
    return {
        "mean": scores.mean(),
        "std": scores.std(),
        "min": scores.min(),
        "max": scores.max(),
        "median": scores.median(),
        "count": len(scores),
        "unique": scores.nunique()
    }


def get_context_counts(train_df: pd.DataFrame) -> Dict[str, int]:
    """Get context (CPC class) distribution"""
    return train_df["context"].value_counts().to_dict()


def analyze_data() -> Dict[str, Any]:
    """Perform full data analysis"""
    train_df, test_df = load_data()
    
    analysis = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_columns": list(train_df.columns),
        "test_columns": list(test_df.columns),
        "score_dist": get_score_distribution(train_df),
        "context_counts": get_context_counts(train_df)[:10],  # Top 10
    }
    
    return analysis


if __name__ == "__main__":
    analysis = analyze_data()
    print("Data Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")