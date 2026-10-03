# Customer Churn Prediction — Playground Series S6E3

Kaggle Playground Series challenge: predict customer churn (`Yes`/`No`) from synthetic Telco data, scored on ROC-AUC.

## Approach (`notebook_v2.ipynb`, `notebook_v3.ipynb`)
- **Feature engineering**: 95+ features — tenure/charge interactions, service bundles, risk flags, frequency + target encoding (out-of-fold).
- **Models**: XGBoost, LightGBM, CatBoost, MLP — each tuned with Optuna and trained across multiple seeds.
- **Ensemble**: simple average, rank average, weight-optimised blend, and a stacking meta-learner.
- **Result**: OOF ROC-AUC ~0.917, targeting the top ~10% of the leaderboard.

## Data
Not committed. Download from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e3/data) into `data/`. The optional original [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset can be placed in `original_data/` and is merged/deduplicated automatically when present.

## Run
Notebooks auto-detect Kaggle vs local environment. Locally, run with a Python 3.10+ kernel and the notebooks handle paths relative to the repo.