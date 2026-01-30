# Early Academic Burnout Signal Detection (Real Dataset: OULAD)

This project builds an **early-warning burnout risk model** using *real* multi-source student behaviour data from the **Open University Learning Analytics Dataset (OULAD)**.

⚠️ **Important**: OULAD does not contain a direct “burnout” label. In this repo, we model **burnout risk as a proxy outcome**:
- **burnout_label = 1** if the student’s `final_result` is **Withdrawn** or **Fail**
- **burnout_label = 0** if `final_result` is **Pass** or **Distinction**

This is common in learning analytics: disengagement/withdrawal is used as an observable outcome related to burnout-like patterns.

Dataset source and license:
- OULAD is available via **UCI Machine Learning Repository** and is licensed under **CC BY 4.0**.  
- The original dataset paper is by Kuzilek et al. (2017) in *Scientific Data*.

Cite:
- Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics dataset*. Scientific Data, 4, 170171.

---

## What you get in this repo
- ✅ Download + extract **real OULAD dataset** (no Kaggle login)
- ✅ Multi-source join: clickstream + assessments + registrations + demographics
- ✅ Early-window feature engineering (first **N weeks** only)
- ✅ Train / Evaluate (LogReg baseline; optional RandomForest)
- ✅ Predict risk scores for new students (same schema)
- ✅ Reproducible scripts + tests + documentation

---

## Project structure
```
burnout-oulad-burnout-detection/
  data/
    raw/oulad/                 # auto-downloaded here
    processed/
  models/
  reports/
  notebooks/
  src/
  tests/
  train.py
  predict.py
```

---

## Quickstart

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Download the real dataset
```bash
python -m src.download_oulad --out_dir data/raw/oulad
```

### 3) Build features (first 4 weeks as default)
```bash
python -m src.build_features_oulad --raw_dir data/raw/oulad --out_dir data/processed --weeks 4
```

### 4) Train + evaluate
```bash
python train.py --data_path data/processed/features_w4.parquet --out_dir reports --model_dir models
```

### 5) Predict risk scores
```bash
python predict.py --data_path data/processed/features_w4.parquet --model_dir models --out reports/predictions.csv --latest_only
```

---

## How “early detection” works here
The model only uses behaviour **up to a cutoff** (e.g., first 4 weeks = 28 days from course start) and tries to predict the final outcome. This simulates early intervention.

Try different windows:
- 2 weeks, 4 weeks, 8 weeks
```bash
python -m src.build_features_oulad --raw_dir data/raw/oulad --out_dir data/processed --weeks 8
python train.py --data_path data/processed/features_w8.parquet --out_dir reports --model_dir models
```

---

## Ethics note
This should be used to **support** learners (check-ins, extra resources, counselling), not penalize them. Avoid sensitive attributes unless you have strong justification and fairness checks.

---

## References / Sources
- UCI Machine Learning Repository: OULAD dataset page (download + CC BY 4.0 license).
- OU Analyse dataset description & citation recommendation.
- Kuzilek et al., 2017 (Scientific Data) dataset paper.

