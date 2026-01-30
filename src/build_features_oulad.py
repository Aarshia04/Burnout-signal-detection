import argparse
import os
import numpy as np
import pandas as pd

from .oulad_io import load_core_tables, read_csv

def make_key(df: pd.DataFrame) -> pd.Series:
    # unique per module-presentation
    return df["id_student"].astype(str) + "_" + df["code_module"].astype(str) + "_" + df["code_presentation"].astype(str)

def build_labels(student_info: pd.DataFrame) -> pd.DataFrame:
    df = student_info.copy()
    df["student_key"] = make_key(df)
    df["burnout_label"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
    # Keep a few fields that are useful but not overly sensitive
    keep = ["student_key", "id_student", "code_module", "code_presentation", "final_result", "burnout_label", "studied_credits", "num_of_prev_attempts"]
    return df[keep]

def build_registration_features(student_reg: pd.DataFrame) -> pd.DataFrame:
    df = student_reg.copy()
    df["student_key"] = make_key(df)
    # date_registration can be negative (registered before start)
    out = df.groupby("student_key", as_index=False).agg(
        date_registration=("date_registration", "min"),
        date_unregistration=("date_unregistration", "min"),
    )
    out["unregistered"] = out["date_unregistration"].notna().astype(int)
    return out

def build_assessment_features(labels: pd.DataFrame, assessments: pd.DataFrame, student_assessment: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """
    Build early-window assessment features.

    Fixes OULAD quirks:
    - assessments.date can contain '?' → coerce to numeric
    - studentAssessment.date_submitted can contain '?' → coerce to numeric
    """
    a = assessments.copy()

    a["date"] = pd.to_numeric(a["date"], errors="coerce")
    a = a.dropna(subset=["date"]).copy()
    a["date"] = a["date"].astype(int)

    a["weight"] = pd.to_numeric(a.get("weight", 0), errors="coerce").fillna(0)

    a_early = a[a["date"] <= cutoff_day][["id_assessment", "assessment_type", "date", "weight"]].copy()

    sa = student_assessment.copy()
    sa["date_submitted"] = pd.to_numeric(sa["date_submitted"], errors="coerce").fillna(-999).astype(int)
    sa["score"] = pd.to_numeric(sa["score"], errors="coerce")

    sa = sa.merge(a_early, on="id_assessment", how="inner")

    sa["late_days"] = (sa["date_submitted"] - sa["date"]).clip(lower=0)
    sa["is_late"] = (sa["late_days"] > 0).astype(int)

    sa = sa.merge(labels[["student_key", "id_student"]], on="id_student", how="inner")

    out = sa.groupby("student_key", as_index=False).agg(
        assessments_submitted=("score", "count"),
        mean_score=("score", "mean"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        late_count=("is_late", "sum"),
        avg_late_days=("late_days", "mean"),
    )

    return out.fillna(0)


def build_vle_daily_aggregate(raw_dir: str, cutoff_day: int, out_path: str, chunksize: int = 1_000_000) -> str:
    """Chunk-read studentVle.csv and produce daily totals per student_key and day."""
    student_vle_path = os.path.join(raw_dir, "studentVle.csv")
    if not os.path.exists(student_vle_path):
        raise FileNotFoundError(f"Missing required file: {student_vle_path}")

    parts = []
    for chunk in pd.read_csv(student_vle_path, chunksize=chunksize):
        # Filter early window
        chunk = chunk[(chunk["date"] >= 0) & (chunk["date"] <= cutoff_day)].copy()
        if chunk.empty:
            continue
        chunk["student_key"] = (chunk["id_student"].astype(str) + "_" + chunk["code_module"].astype(str) + "_" + chunk["code_presentation"].astype(str))
        # daily total clicks
        g = chunk.groupby(["student_key", "date"], as_index=False)["sum_click"].sum()
        parts.append(g)

    if not parts:
        raise RuntimeError("No VLE interactions found in the specified cutoff window.")

    daily = pd.concat(parts, ignore_index=True)
    daily = daily.groupby(["student_key", "date"], as_index=False)["sum_click"].sum()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    daily.to_parquet(out_path, index=False)
    return out_path

def build_vle_features(daily_path: str, cutoff_day: int) -> pd.DataFrame:
    daily = pd.read_parquet(daily_path)
    daily = daily.sort_values(["student_key", "date"]).copy()

    # weekly aggregation
    daily["week"] = (daily["date"] // 7).astype(int)
    weekly = daily.groupby(["student_key", "week"], as_index=False)["sum_click"].sum()

    # per-student features
    g = daily.groupby("student_key")
    out = g.agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        last_active_day=("date", "max"),
        avg_clicks_per_active_day=("sum_click", "mean"),
    ).reset_index()

    # weekly slope + volatility
    def slope(vals: np.ndarray) -> float:
        if len(vals) < 2:
            return 0.0
        x = np.arange(len(vals))
        return float(np.polyfit(x, vals, 1)[0])

    wg = weekly.groupby("student_key")["sum_click"]
    out["weekly_clicks_mean"] = wg.mean().values
    out["weekly_clicks_std"] = wg.std().fillna(0).values
    out["weekly_clicks_slope"] = wg.apply(lambda s: slope(s.values)).values

    # Drop flag: last week vs mean of previous weeks
    def drop_flag(s: pd.Series) -> int:
        if len(s) < 2:
            return 0
        last = s.iloc[-1]
        prev_mean = s.iloc[:-1].mean()
        return int(prev_mean > 0 and last < 0.5 * prev_mean)

    out["engagement_drop_flag"] = weekly.groupby("student_key")["sum_click"].apply(drop_flag).values

    # Normalize to window length
    out["window_days"] = cutoff_day + 1
    out["clicks_per_day"] = out["total_clicks"] / out["window_days"]
    out["activity_rate"] = out["active_days"] / out["window_days"]

    return out.fillna(0)

def assemble_dataset(raw_dir: str, out_dir: str, weeks: int) -> str:
    cutoff_day = weeks * 7 - 1
    student_info, student_reg, assessments, student_assessment, vle = load_core_tables(raw_dir)

    labels = build_labels(student_info)
    regf = build_registration_features(student_reg)
    assf = build_assessment_features(labels, assessments, student_assessment, cutoff_day)

    daily_path = os.path.join(out_dir, f"tmp_vle_daily_w{weeks}.parquet")
    daily_path = build_vle_daily_aggregate(raw_dir, cutoff_day, daily_path)

    vlf = build_vle_features(daily_path, cutoff_day)

    # Merge all features
    X = labels.merge(regf, on="student_key", how="left").merge(assf, on="student_key", how="left").merge(vlf, on="student_key", how="left")
    X = X.fillna(0)

    # Minimal cleanup: remove tmp daily to save space
    try:
        os.remove(daily_path)
    except OSError:
        pass

    out_path = os.path.join(out_dir, f"features_w{weeks}.parquet")
    os.makedirs(out_dir, exist_ok=True)
    X.to_parquet(out_path, index=False)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw/oulad")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--weeks", type=int, default=4)
    args = ap.parse_args()

    out_path = assemble_dataset(args.raw_dir, args.out_dir, args.weeks)
    print(f"Wrote features to: {out_path}")

if __name__ == "__main__":
    main()

