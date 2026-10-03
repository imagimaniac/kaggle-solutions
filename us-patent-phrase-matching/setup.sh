#!/bin/bash
# Setup script for Patent Phrase Matching competition

echo "=========================================="
echo "Patent Phrase Matching - Data Download"
echo "=========================================="

PROJECT_DIR="$HOME/Projects/kaggle/us-patent-phrase-matching"
cd "$PROJECT_DIR"

echo ""
echo "Checking for data..."

# Check if data exists
if [ -f "data/train.csv" ] && [ -f "data/test.csv" ]; then
    echo "Data already exists!"
    ls -la data/
else
    echo ""
    echo "Data not found. Options to download:"
    echo ""
    echo "Option 1: Using Kaggle CLI"
    echo "  kaggle competitions download -c us-patent-phrase-to-phrase-matching -p ./data/"
    echo ""
    echo "Option 2: Manual download"
    echo "  1. Go to: https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/data"
    echo "  2. Download all files"
    echo "  3. Extract to data/ folder"
    echo ""
    echo "Current data folder:"
    ls -la data/ 2>/dev/null || echo "(empty)"
fi

echo ""
echo "=========================================="
echo "To download manually:"
echo "=========================================="
echo "1. Go to: https://www.kaggle.com/c/us-patent-phrase-to-phrase-matching/data"
echo "2. Click 'Download All'"
echo "3. Move zip file to: $PROJECT_DIR/data/"
echo "4. Unzip and extract"
echo ""
echo "Then run:"
echo "  cd notebooks"
echo "  jupyter lab"
echo ""
echo "Open: 01_eda.ipynb -> 02_baseline.ipynb -> 03_training.ipynb"
echo "=========================================="