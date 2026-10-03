# U.S. Patent Phrase to Phrase Matching - Training Guide

## Current Status - Need GPU

Your machine doesn't have a GPU. To achieve **0.80+** scores, you need GPU access.

---

## Options for GPU Training

### Option 1: Google Colab (Free)
1. Go to: https://colab.research.google.com
2. Create new notebook
3. Upload this project folder (or clone from GitHub)
4. Runtime → Change runtime type → GPU
5. Run the training notebooks

### Option 2: Cloud GPU (Paid)
- **Gradient** (free GPU): https://gradient.ai
- **Google Cloud**: $300 free credits
- **AWS**: EC2 g4dn instances

### Option 3: Local GPU
- NVIDIA GPU with 8GB+ VRAM

---

## Training Files Created

| File | Purpose |
|------|---------|
| `src/train_quick.py` | Quick training (recommended) |
| `src/train_deberta.py` | Full CV training |
| `src/inference_deberta.py` | Generate predictions |
| `notebooks/03_deberta_training.ipynb` | Jupyter version |
| `run_training.sh` | Run script |

---

## How to Run (With GPU)

### Method 1: Python Script
```bash
cd ~/Projects/kaggle/us-patent-phrase-matching
source venv/bin/activate
python3 src/train_quick.py
```

### Method 2: Jupyter Notebook
```bash
cd ~/Projects/kaggle/us-patent-phrase-matching
jupyter lab notebooks/03_deberta_training.ipynb
```

### Method 3: Shell Script
```bash
cd ~/Projects/kaggle/us-patent-phrase-matching
bash run_training.sh
```

---

## Expected Results (with GPU)

| Model | Pearson Score |
|-------|---------------|
| mpnet (current) | 0.62 |
| **DeBERTa fine-tuned** | **0.80+** |
| Ensemble | 0.85+ |

---

## Training Time Estimates

| GPU | Time for 3 epochs |
|-----|-----------------|
| T4 (Free Colab) | ~30-45 min |
| V100 | ~15-20 min |
| A100 | ~10 min |
| RTX 3090 | ~10-15 min |

---

## After Training

1. Submission saved to: `output/submission.csv`
2. Upload to Kaggle
3. Check leaderboard

---

## Need Help?

For questions about GPU setup, ask anytime!