"""
Evaluation metrics for binary fake-review classification.
Reports: Accuracy, Precision, Recall, F1, AUC, and Cross-Generator Performance Drop.
Also runs McNemar's test for statistical significance between classifiers.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from scipy.stats import chi2_contingency


def evaluate(y_true, y_pred, y_prob=None, label: str = "") -> dict:
    results = {
        "label": label,
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_true, y_prob) if y_prob is not None else None,
    }
    cm = confusion_matrix(y_true, y_pred)
    results["confusion_matrix"] = cm.tolist()
    _print_results(results)
    return results


def cross_gen_drop(in_dist_f1: float, cross_gen_f1: float) -> float:
    """Equation (5) from the paper."""
    drop = in_dist_f1 - cross_gen_f1
    print(f"Cross-Generator Performance Drop: {drop:.4f} "
          f"(in-dist F1={in_dist_f1:.4f}, cross-gen F1={cross_gen_f1:.4f})")
    return drop


def mcnemar_test(y_true, y_pred_a, y_pred_b) -> dict:
    """McNemar's test to compare two classifiers' predictions."""
    a_right = (y_pred_a == y_true)
    b_right = (y_pred_b == y_true)
    n01 = np.sum(a_right & ~b_right)   # A correct, B wrong
    n10 = np.sum(~a_right & b_right)   # A wrong, B correct
    table = [[0, n01], [n10, 0]]
    chi2, p, _, _ = chi2_contingency(table, correction=True)
    result = {"chi2": chi2, "p_value": p, "n01": int(n01), "n10": int(n10)}
    print(f"McNemar's test: chi2={chi2:.4f}, p={p:.4f}")
    return result


def results_to_dataframe(results_list: list) -> pd.DataFrame:
    rows = []
    for r in results_list:
        rows.append({
            "Classifier": r["label"],
            "Accuracy":   round(r["accuracy"], 4),
            "Precision":  round(r["precision"], 4),
            "Recall":     round(r["recall"], 4),
            "F1":         round(r["f1"], 4),
            "AUC":        round(r["auc"], 4) if r["auc"] else "-",
        })
    return pd.DataFrame(rows)


def _print_results(r: dict):
    print(f"\n{'='*40}")
    print(f"  {r['label']}")
    print(f"  Accuracy : {r['accuracy']:.4f}")
    print(f"  Precision: {r['precision']:.4f}")
    print(f"  Recall   : {r['recall']:.4f}")
    print(f"  F1       : {r['f1']:.4f}")
    if r["auc"]:
        print(f"  AUC      : {r['auc']:.4f}")
    print(f"{'='*40}")
