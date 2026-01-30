import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

@dataclass
class ModelBundle:
    model_name: str
    model: object
    feature_cols: list[str]

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    # Exclude identifiers + labels
    drop = {"student_key","id_student","code_module","code_presentation","final_result","burnout_label"}
    return [c for c in df.columns if c not in drop]

def train_logreg(X: pd.DataFrame, y: pd.Series) -> ModelBundle:
    cols = get_feature_cols(X)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
    ])
    pipe.fit(X[cols], y)
    return ModelBundle("logreg", pipe, cols)

def train_rf(X: pd.DataFrame, y: pd.Series) -> ModelBundle:
    cols = get_feature_cols(X)
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    rf.fit(X[cols], y)
    return ModelBundle("random_forest", rf, cols)

def predict_proba(bundle: ModelBundle, X: pd.DataFrame) -> np.ndarray:
    return bundle.model.predict_proba(X[bundle.feature_cols])[:, 1]
