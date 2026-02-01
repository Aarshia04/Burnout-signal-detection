# 🚀 Quick Start Guide
## Early Academic Burnout Detection Project

This guide will help you get the project up and running in **5 minutes**!

---

## ⚡ Fast Track (3 Steps)

### Step 1: Install Dependencies (1 minute)

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib --break-system-packages
```

Or use the requirements file:

```bash
pip install -r requirements.txt --break-system-packages
```

### Step 2: Generate Data (30 seconds)

```bash
cd burnout_detection
python src/generate_dataset.py
```

You should see output like:
```
Dataset generated successfully!
Total records: 16000
Unique students: 1000
Burnout cases: 305
Non-burnout cases: 695
```

### Step 3: Run Analysis (Open Jupyter) (1 minute)

```bash
jupyter notebook notebooks/burnout_detection_analysis.ipynb
```

Then click: **Cell → Run All**

The notebook will:
- ✅ Load the generated data
- ✅ Perform exploratory analysis
- ✅ Train 4 different ML models
- ✅ Generate visualizations
- ✅ Save the best model

**Total time: ~2-3 minutes to complete**

---

## 📊 What You'll Get

After running the notebook, you'll have:

### 1. Trained Models
- `models/best_burnout_model.pkl` - Ready-to-use model
- `models/scaler.pkl` - Feature scaler
- `models/label_encoder_major.pkl` - Category encoder
- `models/feature_names.txt` - Feature list

### 2. Visualizations (in `results/`)
- Burnout distribution charts
- Temporal trend analysis
- Correlation heatmaps
- Model performance comparisons
- ROC curves
- Confusion matrices
- Feature importance plots

### 3. Performance Metrics
- AUC-ROC Score: ~0.85-0.95
- F1-Score: ~0.75-0.90
- Classification reports
- Cross-validation scores

---

## 🎯 Next Steps

### Test the Prediction System

```bash
python src/predict.py
```

This will:
1. Load your trained model
2. Create an example student profile
3. Predict burnout risk
4. Show top risk factors

### Modify the Data

Edit `src/generate_dataset.py` to:
- Change number of students: `n_students=1000` → `n_students=2000`
- Change time period: `n_weeks=16` → `n_weeks=12`
- Adjust burnout rate: `np.random.random() < 0.30` → `< 0.25`

Then regenerate:
```bash
python src/generate_dataset.py
```

### Explore Different Models

In the notebook, try:
- Changing model parameters
- Adding new features
- Testing different algorithms
- Adjusting train-test split

---

## 🔍 Understanding the Results

### Reading the Output

**Burnout Probability**: 0-100%
- `0-40%` = LOW RISK (green zone)
- `40-70%` = MEDIUM RISK (yellow zone)
- `70-100%` = HIGH RISK (red zone)

**AUC-ROC Score**: Model quality
- `> 0.9` = Excellent
- `0.8-0.9` = Good
- `0.7-0.8` = Fair
- `< 0.7` = Poor

### Key Visualizations

1. **Temporal Patterns** - Shows how burnout develops over time
2. **Feature Importance** - Which factors matter most
3. **ROC Curves** - Model discrimination ability
4. **Confusion Matrix** - True vs False predictions

---

## 💡 Common Use Cases

### For Learning/Students
```bash
# 1. Generate data
python src/generate_dataset.py

# 2. Open notebook
jupyter notebook notebooks/burnout_detection_analysis.ipynb

# 3. Run all cells and explore!
```

### For Research
```bash
# 1. Replace synthetic data with real data
# 2. Update feature engineering in notebook
# 3. Re-train models
# 4. Analyze results
```

### For Deployment
```bash
# 1. Train model on real data
# 2. Use predict.py as API endpoint
# 3. Integrate with student information system
# 4. Set up monitoring dashboard
```

---

## 🛠️ Troubleshooting

### Issue: "No module named 'sklearn'"
```bash
pip install scikit-learn --break-system-packages
```

### Issue: "Jupyter not found"
```bash
pip install jupyter --break-system-packages
python -m ipykernel install --user
```

### Issue: "Permission denied"
```bash
# Use --break-system-packages flag
pip install package_name --break-system-packages
```

### Issue: Kernel dies when running notebook
- Reduce dataset size in `generate_dataset.py`
- Change `n_students=1000` to `n_students=500`

---

## 📚 File Overview

**Essential Files:**
- `README.md` - Complete documentation
- `QUICKSTART.md` - This file (fast setup)
- `src/generate_dataset.py` - Data generation
- `notebooks/burnout_detection_analysis.ipynb` - Main analysis
- `src/predict.py` - Prediction utility

**Generated Files:**
- `data/student_behavior_data.csv` - Dataset
- `models/*.pkl` - Trained models
- `results/*.png` - Visualizations

---

## 🎓 Learning Path

### Beginner (1-2 hours)
1. Read README.md overview
2. Run Quick Start steps
3. Explore generated visualizations
4. Run predict.py demo

### Intermediate (3-4 hours)
1. Study the Jupyter notebook
2. Understand feature engineering
3. Compare model performances
4. Modify data generation parameters

### Advanced (5+ hours)
1. Implement new features
2. Try different algorithms
3. Optimize hyperparameters
4. Build deployment pipeline

---

## 📞 Getting Help

1. **Check the README** - Most questions answered there
2. **Review notebook comments** - Detailed explanations in code
3. **Examine visualizations** - Pictures tell the story
4. **Test with predict.py** - Understand model behavior

---

## ✅ Success Checklist

Before you finish, make sure you have:

- [ ] Installed all dependencies
- [ ] Generated the dataset (16,000 records)
- [ ] Ran the complete Jupyter notebook
- [ ] Saw model training results (AUC > 0.80)
- [ ] Generated all visualizations
- [ ] Trained model saved in `models/`
- [ ] Tested prediction with `predict.py`

---

## 🎉 You're Ready!

You now have a complete, working burnout detection system!

**What you can do:**
- ✅ Understand the methodology
- ✅ Analyze student behavioral patterns
- ✅ Train machine learning models
- ✅ Make predictions on new students
- ✅ Identify key risk factors
- ✅ Generate professional visualizations

**Next level:**
- Integrate with real student data
- Deploy as web application
- Build monitoring dashboard
- Conduct research studies

---

**Happy analyzing! 🚀**

Questions? Review the main README.md for comprehensive details.
