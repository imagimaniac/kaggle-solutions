# Competition Status

## Kaggle API: 401 Unauthorized Error

**Issue**: Competition data download requires accepting rules on the Kaggle website first.

## Manual Download Instructions

### Step 1: Accept Competition Rules
1. Go to: https://www.kaggle.com/c/us-patent-phrase-to-phrase-matching
2. Click **"Join Competition"** button (if available)
3. Accept any rules/terms

### Step 2: Download Data
1. Go to: https://www.kaggle.com/c/us-patent-phrase-to-phrase-matching/data
2. Click **"Download All"** button
3. Save the zip file

### Step 3: Extract Data
```bash
# Move downloaded file to data folder
mv ~/Downloads/us-patent-phrase-to-phrase-matching.zip ~/Projects/kaggle/us-patent-phrase-matching/data/

# Unzip
cd ~/Projects/kaggle/us-patent-phrase-matching/data/
unzip us-patent-phrase-to-phrase-matching.zip

# Verify
ls -la
```

### Expected Files
- `train.csv` (~48k rows)
- `test.csv`
- `sample_submission.csv`

### Once Data is Downloaded

```bash
# Install dependencies
cd ~/Projects/kaggle/us-patent-phrase-matching
pip install -r requirements.txt

# Run notebooks in order:
# 1. 01_eda.ipynb    - Explore data
# 2. 02_baseline.ipynb - Baseline model  
# 3. 03_training.ipynb - Fine-tuned model
```

---

## Expected Results

| Model | Pearson Score |
|-------|------------|
| Baseline (Sentence-BERT) | ~0.60 |
| Fine-tuned DeBERTa | ~0.80+ |

---

## Alternative: Use Sample Data

If you can't access the competition data, you can still build the solution using the Google Patent Phrase Similarity dataset:

- **GitHub**: https://github.com/google/patent-phrase-similarity
- **Paper**: https://blog.research.google/2022/08/announcing-patent-phrase-similarity.html

This dataset can be used to develop and test the solution. Replace the competition data with the sample data for testing.