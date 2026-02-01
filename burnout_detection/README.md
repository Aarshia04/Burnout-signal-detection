# Early Academic Burnout Signal Detection
## Using Multi-Source Student Behaviour Data

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📋 Project Overview

This project implements a **machine learning system** to detect early signals of academic burnout by analyzing multi-source student behavioral data. The system integrates data from 8 different sources to provide comprehensive burnout risk assessment.

### 🎯 Objectives

1. **Early Detection**: Identify students at risk of burnout before it becomes severe
2. **Multi-Source Analysis**: Leverage diverse data sources for holistic assessment
3. **Actionable Insights**: Provide interpretable results for intervention planning
4. **Scalable Solution**: Build a system that can be deployed in real educational settings

---

## 📊 Data Sources

The system analyzes **8 categories** of student behavioral data:

| Category | Features | Examples |
|----------|----------|----------|
| **Academic Performance** | GPA, Assignment Scores | Current GPA, score trends, decline rates |
| **Attendance Patterns** | Class Attendance | Attendance rate, classes missed |
| **Assignment Submission** | On-time, Late, Missing | Submission patterns, deadline proximity |
| **Online Learning (LMS)** | Engagement Metrics | Logins, time spent, video completion, forum activity |
| **Library Usage** | Study Patterns | Visits, study hours |
| **Social Engagement** | Campus Activities | Participation, peer interactions |
| **Health & Wellbeing** | Self-reported Data | Sleep quality/hours, stress levels, exercise |
| **Help-Seeking Behavior** | Support Services | Office hours, tutoring, counseling visits |

**Total Features**: 70+ features engineered from raw data

---

## 🏗️ Project Structure

```
burnout_detection/
│
├── data/
│   └── student_behavior_data.csv          # Generated synthetic dataset
│
├── notebooks/
│   └── burnout_detection_analysis.ipynb   # Complete analysis notebook
│
├── src/
│   ├── generate_dataset.py                # Data generation script
│   └── predict.py                         # Prediction utility
│
├── models/
│   ├── best_burnout_model.pkl            # Trained model
│   ├── scaler.pkl                         # Feature scaler
│   ├── label_encoder_major.pkl           # Major encoder
│   └── feature_names.txt                  # Feature list
│
├── results/
│   ├── burnout_distribution.png
│   ├── temporal_patterns.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── feature_importance_*.png
│
├── docs/
│   └── methodology.md                     # Detailed methodology
│
└── README.md                              # This file
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8 or higher
pip (Python package manager)
```

### Installation

1. **Clone or download the project**
   ```bash
   cd burnout_detection
   ```

2. **Install required packages**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib --break-system-packages
   ```

### Quick Start

#### Option 1: Run the Complete Analysis

```bash
# Start Jupyter Notebook
jupyter notebook notebooks/burnout_detection_analysis.ipynb
```

Then run all cells in the notebook to:
- Generate synthetic data
- Perform exploratory analysis
- Train multiple ML models
- Evaluate performance
- Generate visualizations

#### Option 2: Generate Data Only

```bash
python src/generate_dataset.py
```

This creates `data/student_behavior_data.csv` with 1,000 students × 16 weeks = 16,000 records.

---

## 📈 Dataset Details

### Synthetic Data Generation

The dataset is synthetically generated with realistic patterns:

- **Total Students**: 1,000
- **Time Period**: 16 weeks (one semester)
- **Burnout Rate**: ~30% (reflects realistic academic burnout rates)
- **Total Records**: 16,000 (1,000 students × 16 weeks)

### Data Characteristics

**Burnout Students** exhibit:
- Declining GPA over time
- Reduced attendance (< 80%)
- Increasing assignment delays/missing work
- Decreasing LMS engagement
- Higher stress levels (7-10/10)
- Lower sleep quality (< 5/10)
- Reduced social engagement
- Initial help-seeking followed by withdrawal

**Non-Burnout Students** show:
- Stable or improving GPA
- Consistent attendance (> 85%)
- Regular assignment completion
- Steady LMS engagement
- Moderate stress levels (3-6/10)
- Good sleep quality (> 6/10)
- Active social participation
- Proactive help-seeking when needed

---

## 🤖 Machine Learning Models

### Models Trained

1. **Logistic Regression** - Baseline linear model
2. **Random Forest** - Ensemble tree-based model
3. **Gradient Boosting** - Advanced ensemble method
4. **Support Vector Machine (SVM)** - Kernel-based classifier

### Model Selection Criteria

- **Primary Metric**: AUC-ROC (Area Under ROC Curve)
- **Secondary Metric**: F1-Score
- **Validation**: 5-fold cross-validation

### Expected Performance

With the synthetic dataset, you should see:

- **AUC-ROC**: 0.85 - 0.95
- **F1-Score**: 0.75 - 0.90
- **Accuracy**: 80% - 90%

The model with the highest AUC-ROC is automatically selected as the best model.

---

## 🔍 Key Features for Prediction

Based on feature importance analysis, the most predictive features typically include:

1. **Stress Level** (mean, max)
2. **GPA Decline** (from baseline)
3. **Sleep Quality** (mean, minimum)
4. **Assignment Missing** (count)
5. **Attendance Rate** (mean)
6. **LMS Engagement** (logins, time)
7. **Social Withdrawal** (campus activities, peer interactions)
8. **Academic Distress** (composite indicator)

---

## 📊 Results and Visualizations

The analysis generates multiple visualizations:

1. **Burnout Distribution**
   - Overall burnout rate
   - Distribution by academic major
   - Distribution by year

2. **Temporal Patterns**
   - GPA trends over time
   - Attendance changes
   - LMS activity patterns
   - Stress level progression
   - Sleep quality changes

3. **Feature Analysis**
   - Correlation heatmap
   - Feature importance rankings
   - Model comparison charts

4. **Model Performance**
   - Confusion matrix
   - ROC curves (all models)
   - Precision-Recall curves

All visualizations are saved in the `results/` directory.

---

## 💡 Using the Trained Model

### Making Predictions

After training, you can use the saved model to predict burnout risk for new students:

```python
import joblib
import pandas as pd
import numpy as np

# Load the model
model = joblib.load('models/best_burnout_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load feature names
with open('models/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

# Prepare new student data (with same features)
new_student_data = pd.DataFrame({...})  # Your aggregated features

# Scale features
new_student_scaled = scaler.transform(new_student_data[feature_names])

# Predict
burnout_probability = model.predict_proba(new_student_scaled)[:, 1]
burnout_prediction = model.predict(new_student_scaled)

print(f"Burnout Risk: {burnout_probability[0]*100:.1f}%")
print(f"Classification: {'At Risk' if burnout_prediction[0] == 1 else 'Not At Risk'}")
```

### Risk Thresholds

- **High Risk**: Probability > 0.7 (70%)
- **Medium Risk**: Probability 0.4 - 0.7
- **Low Risk**: Probability < 0.4

---

## 🎓 How to Use This Project

### For Students

1. **Understanding the Analysis**: Open and read through the Jupyter notebook
2. **Running the Code**: Execute cells step-by-step to see how each component works
3. **Modifying Parameters**: Try changing model parameters, data size, or features
4. **Experimenting**: Add new features or try different algorithms

### For Educators/Institutions

1. **Adapt to Real Data**: Replace synthetic data with actual student data
2. **Privacy Considerations**: Ensure FERPA/GDPR compliance
3. **Integration**: Connect to existing student information systems
4. **Dashboard**: Build a monitoring dashboard using the trained model
5. **Intervention Programs**: Use predictions to trigger support services

---

## 📚 Methodology Details

### Feature Engineering Process

1. **Raw Data Collection** (simulated from 8 sources)
2. **Temporal Aggregation** (per-student statistics across 16 weeks)
3. **Derived Features**:
   - GPA decline rate
   - Completion ratio
   - Engagement score
   - Academic distress index
   - Wellbeing score
   - Total help-seeking

### Model Training Pipeline

1. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Create student-level aggregates

2. **Feature Scaling**
   - StandardScaler for numerical features
   - Maintains distribution while normalizing

3. **Train-Test Split**
   - 80% training, 20% testing
   - Stratified by burnout status

4. **Cross-Validation**
   - 5-fold CV for robust evaluation
   - Prevents overfitting

5. **Model Selection**
   - Compare multiple algorithms
   - Select based on AUC-ROC

---

## 🔬 Research Applications

This project can be extended for:

1. **Academic Research**
   - Studying burnout patterns
   - Validating intervention strategies
   - Comparing across institutions

2. **Practical Implementation**
   - Early warning systems
   - Automated alerts
   - Intervention planning

3. **Further Analysis**
   - Temporal pattern mining
   - Subgroup analysis (by major, year)
   - Causal inference

---

## ⚠️ Important Considerations

### Ethical Considerations

- **Privacy**: Protect student data (anonymization, secure storage)
- **Consent**: Obtain informed consent for data collection
- **Transparency**: Explain to students how data is used
- **Bias**: Monitor for algorithmic bias across demographic groups
- **Human Oversight**: Always involve human judgment in interventions

### Limitations

- Synthetic data may not capture all real-world complexities
- Model requires sufficient historical data
- Prediction accuracy depends on data quality
- Cultural and institutional differences affect generalizability

### Data Privacy

When using real student data:
- Comply with FERPA (Family Educational Rights and Privacy Act)
- Follow GDPR if applicable
- Implement proper access controls
- Anonymize data for research purposes
- Obtain IRB approval for research studies

---

## 🛠️ Troubleshooting

### Common Issues

**Issue**: Package installation errors
```bash
# Solution: Use --break-system-packages flag
pip install package_name --break-system-packages
```

**Issue**: Jupyter kernel not found
```bash
# Solution: Install ipykernel
python -m pip install ipykernel --break-system-packages
python -m ipykernel install --user
```

**Issue**: Missing visualizations
```bash
# Solution: Ensure matplotlib backend is set
import matplotlib
matplotlib.use('Agg')  # for non-interactive backend
```

---

## 📖 Additional Resources

### Academic Papers

- Maslach Burnout Inventory (MBI) for Students
- Academic Burnout Scale
- Early Warning Systems in Higher Education

### Useful Links

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Student Mental Health Resources](https://www.activeminds.org/)
- [FERPA Guidelines](https://www2.ed.gov/policy/gen/guid/fpco/ferpa/)

---

## 🤝 Contributing

Suggestions for improvement:

1. **Data Enhancement**
   - Add more data sources (e.g., financial stress, commute time)
   - Include temporal features (time of day patterns)

2. **Model Improvements**
   - Try deep learning approaches (LSTMs for temporal patterns)
   - Implement ensemble methods
   - Add interpretability tools (SHAP values)

3. **Deployment**
   - Create REST API
   - Build web dashboard
   - Mobile app integration

---

## 📞 Support

For questions or issues:

1. Review the Jupyter notebook for detailed explanations
2. Check the troubleshooting section
3. Examine the code comments
4. Review the methodology documentation

---

## 📄 License

This project is created for educational purposes. Feel free to use and modify for academic or research purposes.

---

## 🙏 Acknowledgments

- Synthetic data generation inspired by real academic burnout research
- Machine learning techniques from scikit-learn
- Visualization best practices from seaborn and matplotlib communities

---

## 📝 Citation

If you use this project in your research, please cite:

```
Early Academic Burnout Signal Detection Using Multi-Source Student Behaviour Data
[Your Name/Institution]
[Year]
```

---

**Last Updated**: February 2026

**Version**: 1.0

**Status**: ✅ Complete and Ready to Use
