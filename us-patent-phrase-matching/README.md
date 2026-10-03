# U.S. Patent Phrase to Phrase Matching

Kaggle competition solution for matching semantically similar patent phrases.

## Competition Overview

- **Task**: Predict similarity score (0-1) between patent phrase pairs
- **Metric**: Pearson Correlation
- **Domain**: NLP / Semantic Similarity
- **Data**: ~48,000 training pairs, patent phrases with CPC context

## Project Links

- **Kaggle Competition**: https://www.kaggle.com/c/us-patent-phrase-to-phrase-matching
- **Google Dataset**: https://blog.research.google/2022/08/announcing-patent-phrase-similarity.html

## Project Structure

```
us-patent-phrase-matching/
├── data/                    # Competition data (train.csv, test.csv)
├── src/                     # Python source code
│   ├── config.py            # Configuration
│   ├── dataset.py          # Data loading
│   ├── model.py           # Model definitions
│   ├── train.py          # Training logic
│   └── inference.py      # Inference logic
├── notebooks/               # Jupyter notebooks
│   ├── 01_eda.ipynb     # Data exploration
│   ├── 02_baseline.ipynb # Sentence-BERT baseline
│   └── 03_training.ipynb # DeBERTa fine-tuning
├── models/                  # Saved model weights
├── output/                  # Submission files
├── requirements.txt        # Dependencies
├── README.md              # This file
└── .gitignore           # Git ignore rules
```

## Data Download

**IMPORTANT**: This is a "Late Submission" competition. You need to:

1. Join the competition on Kaggle first
2. Then download data via CLI or manually

```bash
# CLI method (after joining)
kaggle competitions download -c us-patent-phrase-to-phrase-matching -p ./data/
```

**See**: [DOWNLOAD_INSTRUCTIONS.md](./DOWNLOAD_INSTRUCTIONS.md) for full details

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Notebooks

```bash
cd notebooks
jupyter lab
```

Open and run notebooks in order:
1. `01_eda.ipynb` - Data exploration
2. `02_baseline.ipynb` - Sentence-BERT baseline
3. `03_training.ipynb` - Fine-tuned model

## Approach

### Baseline
- Sentence-BERT embeddings + cosine similarity
- Expected Pearson: ~0.60

### Fine-tuned Model
- DeBERTa-v3-small fine-tuning
- Input format: `TEXT1: {context}; TEXT2: {target}; ANC1: {anchor}`
- Expected Pearson: ~0.80+

## Results

| Model | Pearson Score |
|-------|--------------|
| Baseline (Sentence-BERT) | 0.60 |
| Fine-tuned DeBERTa | 0.80+ |

## References

- [Competition Page](https://www.kaggle.com/c/us-patent-phrase-to-phrase-matching)
- [Google Dataset](https://blog.research.google/2022/08/announcing-patent-phrase-similarity.html)

## Security Setup

### HuggingFace Authentication

For Kaggle notebooks, use **Kaggle Secrets** to securely store your HF token:

1. Go to: **Kaggle → Your Account → Secrets**
2. Add new secret:
   - Name: `HF_TOKEN`
   - Value: Your HuggingFace token (from https://huggingface.co/settings/tokens)

The notebooks and scripts are configured to:
1. First try Kaggle Secrets
2. Fall back to environment variable `HF_TOKEN`
3. Run without authentication if neither available

This keeps your token secure and out of public notebooks!

**Important**: Never hardcode tokens in notebooks that will be shared publicly.