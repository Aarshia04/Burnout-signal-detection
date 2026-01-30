import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.modeling import train_logreg, train_rf, predict_proba, ModelBundle
from src.eval import metrics, save_metrics, save_plots

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="Path to features_w*.parquet")
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--model", choices=["logreg","rf"], default="logreg")
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_parquet(args.data_path)
    # ---- Clean any stray '?' strings in numeric columns ----
    df = df.replace("?", pd.NA)

# Convert all non-ID, non-label columns to numeric
    non_features = {"student_key","id_student","code_module","code_presentation","final_result","burnout_label"}
    for c in df.columns:
        if c in non_features:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.fillna(0)

    df = df.dropna(subset=["burnout_label"]).copy()
    y = df["burnout_label"].astype(int)

    # Group split by (module,presentation) to reduce leakage across course runs
    groups = df["code_module"].astype(str) + "_" + df["code_presentation"].astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    train_idx, test_idx = next(gss.split(df, y, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    if args.model == "logreg":
        bundle = train_logreg(train_df, y.iloc[train_idx])
    else:
        bundle = train_rf(train_df, y.iloc[train_idx])

    y_score = predict_proba(bundle, test_df)
    m = metrics(y.iloc[test_idx].to_numpy(), y_score)

    os.makedirs(args.out_dir, exist_ok=True)
    save_metrics(m, args.out_dir)
    save_plots(y.iloc[test_idx].to_numpy(), y_score, args.out_dir)

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(bundle, os.path.join(args.model_dir, "model.joblib"))

    print("Saved model to", os.path.join(args.model_dir, "model.joblib"))
    print("Metrics:", m)

if __name__ == "__main__":
    main()
