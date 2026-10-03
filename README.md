# Kaggle Solutions

Personal solutions to select Kaggle competitions. Each subfolder is self-contained with its own notebooks and README.

| Folder | Competition | Task | Highlights |
|--------|------------|------|------------|
| `playground-s6e3-churn` | Playground Series S6E3 | Customer churn prediction (binary classification, ROC-AUC) | 95+ engineered features, Optuna tuning, 4-model ensemble (XGBoost/LightGBM/CatBoost/MLP), stacking + rank averaging, multi-seed training |
| `us-patent-phrase-matching` | US Patent Phrase-to-Phrase Matching | Semantic similarity of patent phrases | DeBERTa fine-tuning pipelines + lightweight TF-IDF baselines |

## Notes
- Competition data files are **not committed** (large / licence terms). See each folder's README for download instructions.
- `playground-s6e3-churn` notebooks auto-detect Kaggle vs local environments and fall back to local paths when data is present.

## License
MIT