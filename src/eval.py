import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

def save_plots(y_true: np.ndarray, y_score: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # score distribution
    plt.figure()
    plt.hist(y_score, bins=30)
    plt.title("Risk score distribution")
    plt.xlabel("risk_score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "risk_score_hist.png"), dpi=150)
    plt.close()

    # precision-recall curve
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec)
    plt.title("Precision-Recall curve")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=150)
    plt.close()

    # confusion matrix at 0.5
    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

def metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    out = {}
    if len(set(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["avg_precision"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
        out["avg_precision"] = float("nan")
    out["pos_rate"] = float(np.mean(y_true))
    return out

def save_metrics(m: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(m, f, indent=2)
