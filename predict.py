import argparse
import os
import joblib
import pandas as pd
from src.modeling import predict_proba, ModelBundle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--out", default="reports/predictions.csv")
    ap.add_argument("--latest_only", action="store_true", help="Return one row per student_key")
    args = ap.parse_args()

    bundle: ModelBundle = joblib.load(os.path.join(args.model_dir, "model.joblib"))
    df = pd.read_parquet(args.data_path)
    # ---- Clean any stray '?' strings in numeric columns ----
    df = df.replace("?", pd.NA)

    non_features = {"student_key","id_student","code_module","code_presentation","final_result","burnout_label"}
    for c in df.columns:
        if c in non_features:
            continue
    df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.fillna(0)


    scores = predict_proba(bundle, df)
    out = df[["student_key","id_student","code_module","code_presentation"]].copy()
    out["risk_score"] = scores

    # Optional: keep only latest (here latest_only just de-dupes student_key)
    if args.latest_only:
        out = out.drop_duplicates("student_key", keep="last")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
