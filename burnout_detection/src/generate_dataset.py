"""
Generate synthetic multi-source student behavioral data for burnout detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_student_data(n_students=1000, n_weeks=16):
    """
    Generate comprehensive student behavioral data from multiple sources
    """
    
    students = []
    
    for student_id in range(1, n_students + 1):
        # Determine burnout status (30% will experience burnout)
        is_burnout = np.random.random() < 0.30
        
        # Base characteristics
        year = np.random.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.25, 0.15])
        major = np.random.choice(['Engineering', 'Sciences', 'Arts', 'Business', 'Medicine'])
        initial_gpa = np.random.uniform(2.5, 4.0)
        
        # Generate temporal patterns (burnout develops over time)
        for week in range(1, n_weeks + 1):
            # Burnout intensity increases over time for burnout students
            burnout_factor = 0
            if is_burnout:
                burnout_factor = min((week / n_weeks) * 1.5, 1.0)
            
            # 1. ACADEMIC PERFORMANCE
            # GPA degrades for burnout students
            gpa_decline = burnout_factor * np.random.uniform(0.3, 0.8)
            current_gpa = max(initial_gpa - gpa_decline + np.random.normal(0, 0.1), 1.0)
            
            # Assignment scores
            base_score = 85 if not is_burnout else 85 - (burnout_factor * 20)
            assignment_score = np.clip(base_score + np.random.normal(0, 10), 0, 100)
            
            # 2. ATTENDANCE PATTERNS
            # Burnout students miss more classes
            base_attendance = 0.92 if not is_burnout else 0.92 - (burnout_factor * 0.35)
            attendance_rate = np.clip(base_attendance + np.random.normal(0, 0.05), 0, 1)
            classes_missed = int((1 - attendance_rate) * 5)  # 5 classes per week
            
            # 3. ASSIGNMENT SUBMISSION PATTERNS
            total_assignments = 3  # per week
            on_time = int(total_assignments * (1 - burnout_factor * 0.7))
            late_submissions = total_assignments - on_time - int(burnout_factor * 1.5)
            late_submissions = max(0, late_submissions)
            missing_assignments = total_assignments - on_time - late_submissions
            
            # 4. ONLINE LEARNING ACTIVITY (LMS)
            # Burnout students show declining engagement
            base_logins = 15 if not is_burnout else 15 - (burnout_factor * 10)
            lms_logins = max(int(base_logins + np.random.normal(0, 3)), 0)
            
            base_time = 12 if not is_burnout else 12 - (burnout_factor * 8)
            time_on_lms = max(base_time + np.random.normal(0, 2), 0)
            
            # Video completion rate
            video_completion = np.clip((0.85 - burnout_factor * 0.5) + np.random.normal(0, 0.1), 0, 1)
            
            # Forum participation
            forum_posts = max(int((3 - burnout_factor * 2.5) + np.random.poisson(1)), 0)
            
            # 5. LIBRARY USAGE
            base_visits = 4 if not is_burnout else 4 - (burnout_factor * 3)
            library_visits = max(int(base_visits + np.random.normal(0, 1)), 0)
            
            base_study_hours = 10 if not is_burnout else 10 - (burnout_factor * 7)
            study_hours = max(base_study_hours + np.random.normal(0, 2), 0)
            
            # 6. SOCIAL ENGAGEMENT
            base_activities = 2 if not is_burnout else 2 - (burnout_factor * 1.5)
            campus_activities = max(int(base_activities + np.random.normal(0, 0.5)), 0)
            
            peer_interactions = max(int((8 - burnout_factor * 5) + np.random.normal(0, 2)), 0)
            
            # 7. HEALTH & WELLBEING
            # Sleep quality (1-10 scale)
            sleep_quality = np.clip(7 - burnout_factor * 3 + np.random.normal(0, 1), 1, 10)
            
            # Hours of sleep
            sleep_hours = np.clip(7 - burnout_factor * 2 + np.random.normal(0, 0.5), 3, 10)
            
            # Stress level (1-10 scale)
            stress_level = np.clip(4 + burnout_factor * 5 + np.random.normal(0, 1), 1, 10)
            
            # Exercise frequency (times per week)
            exercise_freq = max(int(3 - burnout_factor * 2 + np.random.normal(0, 0.5)), 0)
            
            # 8. HELP-SEEKING BEHAVIOR
            # Burnout students may initially seek help, then withdraw
            if burnout_factor < 0.3:
                help_seeking_multiplier = 1.5  # Early recognition
            else:
                help_seeking_multiplier = 0.3  # Withdrawal
            
            office_hours = max(int(1 * help_seeking_multiplier + np.random.poisson(0.5)), 0)
            tutoring_sessions = max(int(1 * help_seeking_multiplier + np.random.poisson(0.5)), 0)
            counseling_visits = 1 if (is_burnout and burnout_factor > 0.4 and np.random.random() < 0.3) else 0
            
            # 9. TEMPORAL PATTERNS
            # Days since last login (burnout = increasing gaps)
            days_since_login = int(burnout_factor * 5 + np.random.exponential(1))
            
            # Assignment submission time (hours before deadline)
            if missing_assignments == 0:
                hours_before_deadline = max(int(48 - burnout_factor * 40 + np.random.normal(0, 10)), -24)
            else:
                hours_before_deadline = -999  # Missing
            
            student_record = {
                'student_id': student_id,
                'week': week,
                'year': year,
                'major': major,
                
                # Academic
                'current_gpa': round(current_gpa, 2),
                'assignment_score': round(assignment_score, 1),
                
                # Attendance
                'attendance_rate': round(attendance_rate, 3),
                'classes_missed': classes_missed,
                
                # Assignments
                'assignments_on_time': on_time,
                'assignments_late': late_submissions,
                'assignments_missing': missing_assignments,
                'hours_before_deadline': hours_before_deadline,
                
                # LMS Activity
                'lms_logins': lms_logins,
                'time_on_lms_hours': round(time_on_lms, 1),
                'video_completion_rate': round(video_completion, 3),
                'forum_posts': forum_posts,
                'days_since_last_login': days_since_login,
                
                # Library
                'library_visits': library_visits,
                'library_study_hours': round(study_hours, 1),
                
                # Social
                'campus_activities': campus_activities,
                'peer_interactions': peer_interactions,
                
                # Health
                'sleep_quality': round(sleep_quality, 1),
                'sleep_hours': round(sleep_hours, 1),
                'stress_level': round(stress_level, 1),
                'exercise_frequency': exercise_freq,
                
                # Help-seeking
                'office_hours_visits': office_hours,
                'tutoring_sessions': tutoring_sessions,
                'counseling_visits': counseling_visits,
                
                # Target
                'burnout_status': 1 if is_burnout else 0,
                'burnout_severity': round(burnout_factor, 2) if is_burnout else 0
            }
            
            students.append(student_record)
    
    df = pd.DataFrame(students)
    return df


if __name__ == "__main__":
    print("Generating multi-source student behavioral dataset...")
    
    # Generate dataset
    df = generate_student_data(n_students=1000, n_weeks=16)
    
    # Save to CSV
    output_path = '/home/claude/burnout_detection/data/student_behavior_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated successfully!")
    print(f"Total records: {len(df)}")
    print(f"Unique students: {df['student_id'].nunique()}")
    print(f"Burnout cases: {df[df['burnout_status'] == 1]['student_id'].nunique()}")
    print(f"Non-burnout cases: {df[df['burnout_status'] == 0]['student_id'].nunique()}")
    print(f"\nDataset saved to: {output_path}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nDataset info:")
    print(df.info())
