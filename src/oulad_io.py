import os
import pandas as pd

def read_csv(raw_dir: str, name: str, **kwargs) -> pd.DataFrame:
    path = os.path.join(raw_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, **kwargs)

def load_core_tables(raw_dir: str):
    # Smaller tables
    student_info = read_csv(raw_dir, "studentInfo.csv")
    student_reg = read_csv(raw_dir, "studentRegistration.csv")
    assessments = read_csv(raw_dir, "assessments.csv")
    student_assessment = read_csv(raw_dir, "studentAssessment.csv")
    vle = read_csv(raw_dir, "vle.csv")
    # Big table: studentVle.csv handled elsewhere (chunked)
    return student_info, student_reg, assessments, student_assessment, vle
