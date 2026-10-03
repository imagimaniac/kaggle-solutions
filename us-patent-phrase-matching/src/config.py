"""Configuration for Patent Phrase Matching"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Model config
MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
WARMUP_RATIO = 0.1

# Input format
INPUT_TEMPLATE = "TEXT1: {context}; TEXT2: {target}; ANC1: {anchor}"

# Label
TARGET_COLUMN = "score"  # Similarity score 0-1

# Competition
COMPETITION_NAME = "us-patent-phrase-to-phrase-matching"

# Random seed
SEED = 42