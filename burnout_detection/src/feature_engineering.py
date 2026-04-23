"""
Feature engineering helpers for the burnout detection pipeline.

The primary goal of this module is to take a raw DataFrame (either the
synthetic multi-week data or a pre-aggregated OULAD table) and convert it
into a set of numeric features suitable for model training.  The functions are
written defensively so that missing columns are ignored, which simplifies
experimentation with different data sources.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def aggregate_temporal_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn a time-series style dataframe into one row per student.

    The original synthetic generator produces 16 rows per student (one per
    week).  OULAD data, on the other hand, is typically already at the
    student level.  This function detects which case we're dealing with and
    either performs the grouping or simply returns a copy of the input.

    The resulting DataFrame has a column called ``student_id`` (if it exists
    in the input) along with all of the original columns collapsed via
    simple statistics (mean, std, min, sum, max depending on the field).
    """

    df = raw_df.copy()
    if "week" in df.columns and "student_id" in df.columns:
        # replicate the grouping logic from the old pipeline
        agg = df.groupby("student_id").agg({
            "current_gpa": ["mean", "std", "min"],
            "assignment_score": ["mean", "std", "min"],
            "attendance_rate": ["mean", "std", "min"],
            "classes_missed": ["sum", "mean", "max"],
            "assignments_on_time": ["sum", "mean"],
            "assignments_late": ["sum", "mean"],
            "assignments_missing": ["sum", "mean"],
            "lms_logins": ["sum", "mean", "std"],
            "time_on_lms_hours": ["sum", "mean", "std"],
            "video_completion_rate": ["mean", "std", "min"],
            "forum_posts": ["sum", "mean"],
            "days_since_last_login": ["mean", "max"],
            "library_visits": ["sum", "mean"],
            "library_study_hours": ["sum", "mean", "std"],
            "campus_activities": ["sum", "mean"],
            "peer_interactions": ["sum", "mean", "std"],
            "sleep_quality": ["mean", "std", "min"],
            "sleep_hours": ["mean", "std", "min"],
            "stress_level": ["mean", "max", "std"],
            "exercise_frequency": ["sum", "mean"],
            "office_hours_visits": ["sum", "mean"],
            "tutoring_sessions": ["sum", "mean"],
            "counseling_visits": ["sum"],
            "burnout_status": "first",
            "year": "first",
            "major": "first",
        }).reset_index()
        # flatten column hierarchy
        agg.columns = ["_".join(col).strip("_") for col in agg.columns.values]
        agg.rename(columns={"student_id_": "student_id"}, inplace=True)
        return agg
    else:
        # assume already aggregated or single‑row per student
        return df.copy()


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional features that are useful for prediction.

    The function checks whether the required input columns exist before
    adding each derived column; this allows the same code to work with both
    the synthetic dataset and the limited feature set produced from OULAD.
    """

    df = df.copy()

    if "current_gpa_mean" in df.columns and "current_gpa_min" in df.columns:
        df["gpa_decline"] = df["current_gpa_mean"] - df["current_gpa_min"]

    # completion ratio only makes sense if we have the three assignment fields
    if all(col in df.columns for col in [
        "assignments_on_time_sum",
        "assignments_late_sum",
        "assignments_missing_sum",
    ]):
        total = (
            df["assignments_on_time_sum"]
            + df["assignments_late_sum"]
            + df["assignments_missing_sum"]
        )
        df["completion_ratio"] = (
            df["assignments_on_time_sum"] + df["assignments_late_sum"]
        ) / (total + 1e-6)

    if all(col in df.columns for col in [
        "lms_logins_mean",
        "library_visits_mean",
        "campus_activities_mean",
    ]):
        df["engagement_score"] = (
            df["lms_logins_mean"]
            + df["library_visits_mean"]
            + df["campus_activities_mean"]
        ) / 3

    if all(col in df.columns for col in [
        "current_gpa_mean",
        "assignments_missing_sum",
        "attendance_rate_mean",
    ]):
        df["academic_distress"] = (
            (df["current_gpa_mean"] < 3.0).astype(int)
            + (df["assignments_missing_sum"] > 5).astype(int)
            + (df["attendance_rate_mean"] < 0.8).astype(int)
        )

    if all(col in df.columns for col in [
        "sleep_quality_mean",
        "stress_level_mean",
        "exercise_frequency_mean",
    ]):
        df["wellbeing_score"] = (
            df["sleep_quality_mean"]
            - df["stress_level_mean"]
            + df["exercise_frequency_mean"]
        ) / 3

    if all(col in df.columns for col in [
        "office_hours_visits_sum",
        "tutoring_sessions_sum",
        "counseling_visits_sum",
    ]):
        df["total_help_seeking"] = (
            df["office_hours_visits_sum"]
            + df["tutoring_sessions_sum"]
            + df["counseling_visits_sum"]
        )

    return df


def prepare_features(raw_df: pd.DataFrame) -> tuple:
    """
    Complete transformation from raw data to feature matrix / label vector.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw input returned by :func:`data_loader.load_data`.

    Returns
    -------
    X : pd.DataFrame
        Design matrix containing numeric features.
    y : pd.Series or None
        Target variable (``burnout_status``) if present in ``raw_df``.
    student_ids : pd.Series or None
        ``student_id`` values if available (useful for later analysis).
    """

    df = aggregate_temporal_data(raw_df)
    df = add_derived_features(df)

    # extract target if available
    y = df["burnout_status"] if "burnout_status" in df.columns else None

    # drop identifier / target columns from features
    drop_cols = ["student_id", "burnout_status"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()

    # encode object dtypes and keep encoders so they can be saved later
    encoders = {}
    for col in X.select_dtypes(include=[object]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    student_ids = df["student_id"] if "student_id" in df.columns else None
    return X, y, student_ids, encoders
