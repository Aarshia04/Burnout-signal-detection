"""
Automated Execution Script
Runs the complete burnout detection analysis pipeline
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def install_dependencies():
    """Install missing dependencies"""
    print("Installing required packages...")
    packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'jupyter', 'joblib']
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                             package, '--break-system-packages', '--quiet'])
    
    print("✅ All packages installed successfully!\n")

def run_analysis():
    """Execute the complete analysis pipeline"""
    
    print("="*70)
    print("EARLY ACADEMIC BURNOUT DETECTION - AUTOMATED EXECUTION")
    print("="*70)
    print()
    
    # Step 1: Check dependencies
    print("Step 1: Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"Missing packages: {missing}")
        print("Installing missing packages...")
        install_dependencies()
    else:
        print("✅ All dependencies installed\n")
    
    # Step 2: Generate dataset
    print("Step 2: Generating synthetic dataset...")
    print("-"*70)
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        import random
        
        # Set random seed
        np.random.seed(42)
        random.seed(42)
        
        # Import the data generation function
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from generate_dataset import generate_student_data
        
        # Generate data
        df = generate_student_data(n_students=1000, n_weeks=16)
        
        # Save data
        os.makedirs('../data', exist_ok=True)
        df.to_csv('../data/student_behavior_data.csv', index=False)
        
        print(f"✅ Dataset generated: {len(df)} records")
        print(f"   - Students: {df['student_id'].nunique()}")
        print(f"   - Burnout cases: {df[df['burnout_status']==1]['student_id'].nunique()}")
        print()
        
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        return False
    
    # Step 3: Run analysis
    print("Step 3: Running machine learning analysis...")
    print("-"*70)
    
    try:
        # Import required libraries
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.metrics import roc_auc_score, f1_score, classification_report
        import joblib
        
        print("📊 Loading data...")
        df = pd.read_csv('../data/student_behavior_data.csv')
        
        print("🔧 Engineering features...")
        # Create student-level features
        student_features = df.groupby('student_id').agg({
            'current_gpa': ['mean', 'std', 'min'],
            'assignment_score': ['mean', 'std', 'min'],
            'attendance_rate': ['mean', 'std', 'min'],
            'classes_missed': ['sum', 'mean', 'max'],
            'assignments_on_time': ['sum', 'mean'],
            'assignments_late': ['sum', 'mean'],
            'assignments_missing': ['sum', 'mean'],
            'lms_logins': ['sum', 'mean', 'std'],
            'time_on_lms_hours': ['sum', 'mean', 'std'],
            'video_completion_rate': ['mean', 'std', 'min'],
            'forum_posts': ['sum', 'mean'],
            'days_since_last_login': ['mean', 'max'],
            'library_visits': ['sum', 'mean'],
            'library_study_hours': ['sum', 'mean', 'std'],
            'campus_activities': ['sum', 'mean'],
            'peer_interactions': ['sum', 'mean', 'std'],
            'sleep_quality': ['mean', 'std', 'min'],
            'sleep_hours': ['mean', 'std', 'min'],
            'stress_level': ['mean', 'max', 'std'],
            'exercise_frequency': ['sum', 'mean'],
            'office_hours_visits': ['sum', 'mean'],
            'tutoring_sessions': ['sum', 'mean'],
            'counseling_visits': ['sum'],
            'burnout_status': 'first',
            'year': 'first',
            'major': 'first'
        }).reset_index()
        
        student_features.columns = ['_'.join(col).strip('_') for col in student_features.columns.values]
        student_features.rename(columns={'student_id_': 'student_id'}, inplace=True)
        
        # Create derived features
        student_features['gpa_decline'] = student_features['current_gpa_mean'] - student_features['current_gpa_min']
        total_assignments = student_features['assignments_on_time_sum'] + student_features['assignments_late_sum'] + student_features['assignments_missing_sum']
        student_features['completion_ratio'] = (student_features['assignments_on_time_sum'] + student_features['assignments_late_sum']) / (total_assignments + 1e-6)
        student_features['engagement_score'] = (student_features['lms_logins_mean'] + student_features['library_visits_mean'] + student_features['campus_activities_mean']) / 3
        student_features['academic_distress'] = ((student_features['current_gpa_mean'] < 3.0).astype(int) + (student_features['assignments_missing_sum'] > 5).astype(int) + (student_features['attendance_rate_mean'] < 0.8).astype(int))
        student_features['wellbeing_score'] = (student_features['sleep_quality_mean'] - student_features['stress_level_mean'] + student_features['exercise_frequency_mean']) / 3
        student_features['total_help_seeking'] = student_features['office_hours_visits_sum'] + student_features['tutoring_sessions_sum'] + student_features['counseling_visits_sum']
        
        # Encode major
        le_major = LabelEncoder()
        student_features['major_encoded'] = le_major.fit_transform(student_features['major_first'])
        
        # Prepare features
        feature_cols = [col for col in student_features.columns if col not in ['student_id', 'burnout_status_first', 'major_first', 'year_first']]
        X = student_features[feature_cols]
        y = student_features['burnout_status_first']
        
        print(f"   Features: {X.shape[1]} features, {X.shape[0]} students")
        
        print("🚂 Training models...")
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        results = {}
        best_auc = 0
        best_model_name = None
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'auc': auc
            }
            
            if auc > best_auc:
                best_auc = auc
                best_model_name = name
            
            print(f"   - {name}: AUC = {auc:.4f}")
        
        print(f"\n🏆 Best Model: {best_model_name} (AUC = {best_auc:.4f})")
        
        # Save best model
        print("\n💾 Saving models and artifacts...")
        os.makedirs('../models', exist_ok=True)
        os.makedirs('../results', exist_ok=True)
        
        best_model = results[best_model_name]['model']
        joblib.dump(best_model, '../models/best_burnout_model.pkl')
        joblib.dump(scaler, '../models/scaler.pkl')
        joblib.dump(le_major, '../models/label_encoder_major.pkl')
        
        with open('../models/feature_names.txt', 'w') as f:
            f.write('\n'.join(X.columns.tolist()))
        
        print("   ✅ Models saved to models/")
        
        # Generate a quick report
        print("\n" + "="*70)
        print("ANALYSIS SUMMARY")
        print("="*70)
        print(f"Dataset: {len(df)} records, {df['student_id'].nunique()} students")
        print(f"Burnout Rate: {(y.sum() / len(y)) * 100:.1f}%")
        print(f"Features Engineered: {len(feature_cols)}")
        print(f"Best Model: {best_model_name}")
        print(f"Performance (AUC-ROC): {best_auc:.4f}")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ Analysis complete!")
    print("\nNext steps:")
    print("1. Open Jupyter notebook for detailed visualizations:")
    print("   jupyter notebook notebooks/burnout_detection_analysis.ipynb")
    print("2. Test predictions:")
    print("   python src/predict.py")
    print("3. Review results in the results/ directory")

if __name__ == "__main__":
    success = run_analysis()
    
    if success:
        print("\n" + "🎉 "*10)
        print("PROJECT SETUP COMPLETE!")
        print("🎉 "*10)
    else:
        print("\n❌ Setup encountered errors. Please check the output above.")
        sys.exit(1)
