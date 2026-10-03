#!/bin/bash
# DeBERTa GPU Training Setup & Run Script

echo "=========================================="
echo "GPU Training Setup for Patent Phrase Matching"
echo "=========================================="

# Check if GPU is available
python3 -c "import torch; print('GPU:', torch.cuda.is_available())"

if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch not installed"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Install accelerate if needed
pip install accelerate -q

# Run training
echo ""
echo "Starting training..."
echo ""

python3 src/train_quick.py

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="