"""
Prediction Script for Burnout Detection
Use this script to predict burnout risk for new students
"""

import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class BurnoutPredictor:
    """
    A class to handle burnout predictions for students
    """
    
    def __init__(self, model_path='../models/best_burnout_model.pkl',
                 scaler_path='../models/scaler.pkl',
                 encoder_path='../models/label_encoder_major.pkl',
                 features_path='../models/feature_names.txt'):
        """
        Initialize the predictor with trained model and preprocessors
        """
        print("Loading model and preprocessors...")
        
        # Load model
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        
        # Load feature names
        with open(features_path, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        print(f"Model loaded successfully!")
        print(f"Expected features: {len(self.feature_names)}")
    
    def predict_single_student(self, student_data):
        """
        Predict burnout risk for a single student
        
        Parameters:
        -----------
        student_data : dict or pd.DataFrame
            Student features (must match training features)
        
        Returns:
        --------
        dict : Prediction results with probability and classification
        """
        
        # Convert to DataFrame if dict
        if isinstance(student_data, dict):
            student_data = pd.DataFrame([student_data])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(student_data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select and order features
        X = student_data[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        probability = self.model.predict_proba(X_scaled)[0, 1]
        prediction = self.model.predict(X_scaled)[0]
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "HIGH RISK"
            recommendation = "Immediate intervention recommended"
        elif probability >= 0.4:
            risk_level = "MEDIUM RISK"
            recommendation = "Monitor closely and offer support"
        else:
            risk_level = "LOW RISK"
            recommendation = "Continue regular monitoring"
        
        return {
            'burnout_probability': round(probability * 100, 2),
            'prediction': 'At Risk of Burnout' if prediction == 1 else 'Not At Risk',
            'risk_level': risk_level,
            'recommendation': recommendation
        }
    
    def predict_batch(self, students_data):
        """
        Predict burnout risk for multiple students
        
        Parameters:
        -----------
        students_data : pd.DataFrame
            DataFrame with student features
        
        Returns:
        --------
        pd.DataFrame : Predictions for all students
        """
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(students_data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select and order features
        X = students_data[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = self.model.predict(X_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'burnout_probability': probabilities * 100,
            'prediction': ['At Risk' if p == 1 else 'Not At Risk' for p in predictions],
            'risk_level': [self._get_risk_level(p) for p in probabilities]
        })
        
        return results
    
    def _get_risk_level(self, probability):
        """Helper function to determine risk level"""
        if probability >= 0.7:
            return "HIGH RISK"
        elif probability >= 0.4:
            return "MEDIUM RISK"
        else:
            return "LOW RISK"
    
    def get_top_risk_factors(self, student_data, top_n=5):
        """
        Identify top risk factors for a student
        (Only works with tree-based models that have feature_importances_)
        
        Parameters:
        -----------
        student_data : dict or pd.DataFrame
            Student features
        top_n : int
            Number of top factors to return
        
        Returns:
        --------
        pd.DataFrame : Top risk factors
        """
        
        if not hasattr(self.model, 'feature_importances_'):
            print("Feature importance not available for this model type")
            return None
        
        # Convert to DataFrame if dict
        if isinstance(student_data, dict):
            student_data = pd.DataFrame([student_data])
        
        # Get feature importances
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_,
            'student_value': student_data[self.feature_names].values[0]
        }).sort_values('importance', ascending=False)
        
        return importances.head(top_n)


def create_example_student():
    """
    Create an example student profile for demonstration
    """
    
    # This is a simplified example - in practice, you'd have aggregated data
    example = {
        'current_gpa_mean': 2.8,
        'current_gpa_std': 0.3,
        'current_gpa_min': 2.5,
        'assignment_score_mean': 72.5,
        'assignment_score_std': 8.2,
        'assignment_score_min': 60.0,
        'attendance_rate_mean': 0.75,
        'attendance_rate_std': 0.08,
        'attendance_rate_min': 0.65,
        'classes_missed_sum': 12,
        'classes_missed_mean': 0.75,
        'classes_missed_max': 3,
        'assignments_on_time_sum': 32,
        'assignments_on_time_mean': 2.0,
        'assignments_late_sum': 8,
        'assignments_late_mean': 0.5,
        'assignments_missing_sum': 8,
        'assignments_missing_mean': 0.5,
        'lms_logins_sum': 120,
        'lms_logins_mean': 7.5,
        'lms_logins_std': 2.5,
        'time_on_lms_hours_sum': 96,
        'time_on_lms_hours_mean': 6.0,
        'time_on_lms_hours_std': 2.0,
        'video_completion_rate_mean': 0.55,
        'video_completion_rate_std': 0.15,
        'video_completion_rate_min': 0.35,
        'forum_posts_sum': 12,
        'forum_posts_mean': 0.75,
        'days_since_last_login_mean': 3.5,
        'days_since_last_login_max': 7,
        'library_visits_sum': 24,
        'library_visits_mean': 1.5,
        'library_study_hours_sum': 48,
        'library_study_hours_mean': 3.0,
        'library_study_hours_std': 1.5,
        'campus_activities_sum': 12,
        'campus_activities_mean': 0.75,
        'peer_interactions_sum': 64,
        'peer_interactions_mean': 4.0,
        'peer_interactions_std': 2.0,
        'sleep_quality_mean': 4.5,
        'sleep_quality_std': 1.2,
        'sleep_quality_min': 3.0,
        'sleep_hours_mean': 5.5,
        'sleep_hours_std': 0.8,
        'sleep_hours_min': 4.5,
        'stress_level_mean': 7.5,
        'stress_level_max': 9.0,
        'stress_level_std': 1.0,
        'exercise_frequency_sum': 16,
        'exercise_frequency_mean': 1.0,
        'office_hours_visits_sum': 4,
        'office_hours_visits_mean': 0.25,
        'tutoring_sessions_sum': 3,
        'tutoring_sessions_mean': 0.19,
        'counseling_visits_sum': 1,
        'major_encoded': 2,  # Encoded major value
        'gpa_decline': 0.5,
        'completion_ratio': 0.83,
        'engagement_score': 4.42,
        'academic_distress': 2,
        'wellbeing_score': -0.17,
        'total_help_seeking': 8
    }
    
    return example


if __name__ == "__main__":
    
    print("="*70)
    print("BURNOUT PREDICTION DEMO")
    print("="*70)
    
    # Initialize predictor
    predictor = BurnoutPredictor()
    
    # Create example student
    print("\nCreating example student profile...")
    student = create_example_student()
    
    # Make prediction
    print("\nPredicting burnout risk...")
    result = predictor.predict_single_student(student)
    
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"Burnout Probability: {result['burnout_probability']:.2f}%")
    print(f"Classification: {result['prediction']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")
    print("="*70)
    
    # Show top risk factors (if available)
    print("\nIdentifying top risk factors...")
    risk_factors = predictor.get_top_risk_factors(student, top_n=5)
    
    if risk_factors is not None:
        print("\nTop 5 Risk Factors:")
        print("-"*70)
        for idx, row in risk_factors.iterrows():
            print(f"{row['feature']:<40} | Value: {row['student_value']:.2f}")
        print("="*70)
    
    print("\n✅ Demo completed successfully!")
    print("\nTo use with your own data:")
    print("1. Create a dictionary or DataFrame with student features")
    print("2. Call predictor.predict_single_student(your_data)")
    print("3. Review the results and recommendations")
