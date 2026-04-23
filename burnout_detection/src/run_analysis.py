"""
Automated Execution Script
Runs the complete burnout detection analysis pipeline.

The script is deliberately flexible: it can work with the original
synthetic generator (used during early development) or with a real dataset
such as the Open University Learning Analytics (OULAD) collection.  The
behaviour is controlled via command‑line flags so that the same pipeline can
be reused in demonstrations or when the project is put in front of a
stakeholder.
"""

import subprocess
import sys
import os
import argparse


def check_dependencies():
    """Check if required packages are installed"""
    required = [
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "seaborn",
    ]
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    return missing


def install_dependencies():
    """Install missing dependencies via pip"""
    print("Installing required packages...")
    packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "jupyter",
        "joblib",
    ]
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                package,
                "--break-system-packages",
                "--quiet",
            ]
        )
    print("✅ All packages installed successfully!\n")


def run_analysis(dataset_type="synthetic", data_path=None, **kwargs):
    """Execute the full analysis pipeline.

    Parameters
    ----------
    dataset_type : str
        ``'synthetic'`` or ``'oulad'``.
    data_path : str or None
        Location of the dataset when using ``'oulad'``.  Ignored for
        synthetic data.
    **kwargs :
        Passed through to :func:`data_loader.load_data` (e.g. n_students,
        n_weeks for the synthetic generator).

    Returns
    -------
    bool
        ``True`` on success, ``False`` if an exception occurred.
    """

    print("=" * 70)
    print("EARLY ACADEMIC BURNOUT DETECTION - AUTOMATED EXECUTION")
    print("=" * 70)
    print()

    # 1. dependencies
    print("Step 1: Checking dependencies...")
    missing = check_dependencies()
    if missing:
        print(f"Missing packages: {missing}")
        print("Installing missing packages...")
        install_dependencies()
    else:
        print("✅ All dependencies installed\n")

    # 2. load / generate data
    print(f"Step 2: Loading dataset (type={dataset_type})")
    print("-" * 70)
    try:
        import pandas as pd
        # make sure local src directory is on the path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data_loader import load_data

        df = load_data(dataset_type=dataset_type, path=data_path, **kwargs)

        os.makedirs("../data", exist_ok=True)
        out_path = "../data/student_behavior_data.csv"
        df.to_csv(out_path, index=False)

        print(f"✅ Raw dataset saved to {out_path} ({len(df)} rows)")
        if "student_id" in df.columns:
            print(f"   - Students: {df['student_id'].nunique()}")
        if "burnout_status" in df.columns:
            print(f"   - Burnout cases: {df['burnout_status'].sum()}")
        print()

    except Exception as exc:
        print(f"❌ Error loading dataset: {exc}")
        import traceback

        traceback.print_exc()
        return False

    # 3. feature preparation & training
    print("Step 3: Running machine learning analysis...")
    print("-" * 70)
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import (
            RandomForestClassifier,
            GradientBoostingClassifier,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.metrics import roc_auc_score
        import joblib

        from feature_engineering import prepare_features

        print("📊 Preparing features...")
        X, y, _, encoders = prepare_features(df)
        if y is None:
            raise ValueError("Target column 'burnout_status' not found in data")

        print(f"   Features: {X.shape[1]} features, {X.shape[0]} samples")

        print("🚂 Training models...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(
                random_state=42, max_iter=1000
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=42
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        }

        results = {}
        best_auc = 0
        best_model_name = None

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            results[name] = {"model": model, "auc": auc}
            if auc > best_auc:
                best_auc = auc
                best_model_name = name
            print(f"   - {name}: AUC = {auc:.4f}")

        print(f"\n🏆 Best Model: {best_model_name} (AUC = {best_auc:.4f})")

        print("\n💾 Saving models and artifacts...")
        os.makedirs("../models", exist_ok=True)
        os.makedirs("../results", exist_ok=True)

        best_model = results[best_model_name]["model"]
        joblib.dump(best_model, "../models/best_burnout_model.pkl")
        joblib.dump(scaler, "../models/scaler.pkl")

        # persist any categorical encoders returned by the feature pipeline
        for col, enc in encoders.items():
            enc_path = f"../models/label_encoder_{col}.pkl"
            joblib.dump(enc, enc_path)
            print(f"   - encoder saved: {enc_path}")

        with open("../models/feature_names.txt", "w") as f:
            f.write("\n".join(X.columns.tolist()))

        print("   ✅ Models saved to models/")

        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Dataset: {len(df)} records")
        if "student_id" in df.columns:
            print(f"Students: {df['student_id'].nunique()}")
        if y is not None:
            print(f"Burnout rate: {(y.sum() / len(y)) * 100:.1f}%")
        print(f"Features engineered: {X.shape[1]}")
        print(f"Best model: {best_model_name}")
        print(f"Performance (AUC-ROC): {best_auc:.4f}")
        print("=" * 70)

        return True

    except Exception as exc:
        print(f"❌ Error during analysis: {exc}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the burnout detection pipeline"
    )
    parser.add_argument(
        "--data-type",
        choices=["synthetic", "oulad"],
        default="synthetic",
        help="Which source of data to use",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Directory or file path for the dataset (required for oulad)",
    )
    parser.add_argument(
        "--n-students",
        type=int,
        default=1000,
        help="Number of students (synthetic only)",
    )
    parser.add_argument(
        "--n-weeks",
        type=int,
        default=16,
        help="Number of weeks (synthetic only)",
    )
    args = parser.parse_args()

    success = run_analysis(
        dataset_type=args.data_type,
        data_path=args.data_path,
        n_students=args.n_students,
        n_weeks=args.n_weeks,
    )
    if success:
        print("\n" + "🎉 " * 10)
        print("PROJECT SETUP COMPLETE!")
        print("🎉 " * 10)
    else:
        print("\n❌ Setup encountered errors. Please check the output above.")
        sys.exit(1)
