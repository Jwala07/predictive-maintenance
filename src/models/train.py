"""Train and compare models with a time-based split. Metrics logged to MLflow."""
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.engineering import feature_columns

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "features.parquet"
TARGET_RECALL = 0.90
GAP = pd.Timedelta(hours=24)

TRAIN_END = pd.Timestamp("2015-08-31 23:00")
VAL_END = pd.Timestamp("2015-10-31 23:00")


def split(df: pd.DataFrame):
    t = df["datetime"]
    train = df[t <= TRAIN_END - GAP]
    val = df[(t > TRAIN_END) & (t <= VAL_END - GAP)]
    test = df[t > VAL_END]
    return train, val, test


def pick_threshold(y, p, target_recall=TARGET_RECALL) -> float:
    """Highest threshold that still reaches target recall (chosen on validation)."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target_recall)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def evaluate(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": thr,
    }


def rule_score(df: pd.DataFrame) -> np.ndarray:
    """Naive baseline: any error in the last 1h/24h => alert score."""
    now = sum(df[f"error{i}_now"] for i in range(1, 6))
    return (now > 0).astype(float).to_numpy()

def rule_any_error_24h(df: pd.DataFrame) -> np.ndarray:
    """Alert if any error occurred in the last 24h."""
    total = sum(df[f"error{i}_cnt_24h"] for i in range(1, 6))
    return (total > 0).astype(float).to_numpy()


def rule_signature_24h(df: pd.DataFrame) -> np.ndarray:
    """Alert on error1/4/5, or error2+error3 together, in the last 24h.
    Lone error2 or error3 never preceded a failure in the EDA, so they are ignored."""
    hit = ((df["error1_cnt_24h"] > 0) | (df["error4_cnt_24h"] > 0)
           | (df["error5_cnt_24h"] > 0) | (df["err23_hours_since"] <= 23))
    return hit.astype(float).to_numpy()


def main() -> None:
    df = pd.read_parquet(DATA)
    cols = feature_columns(df)
    df[cols] = df[cols].astype("float32")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    train, val, test = split(df)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        print(f"{name}: {len(d)} rows, positives {int(d.target.sum())} "
              f"({100 * d.target.mean():.2f}%), {d.datetime.min()} -> {d.datetime.max()}")

    Xtr, ytr = train[cols], train["target"]
    Xva, yva = val[cols], val["target"]
    Xte, yte = test[cols], test["target"]

    models = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=20, max_samples=0.5,
            class_weight="balanced_subsample", n_jobs=2, random_state=42),
        "lightgbm": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(),
            random_state=42, verbose=-1),
    }

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("predictive-maintenance")

    # ---- rule baseline: threshold fixed at "any error now"
    for rule_name, rule_fn in [("rule_error_now", rule_score),
                               ("rule_any_error_24h", rule_any_error_24h),
                               ("rule_signature_24h", rule_signature_24h)]:
        with mlflow.start_run(run_name=rule_name):
            m = evaluate(yte, rule_fn(test), 0.5)
            mlflow.log_metrics(m)
            print(f"\n{rule_name} (test):", {k: round(float(v), 4) for k, v in m.items()})

    results = {}
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(Xtr, ytr)
            p_val = model.predict_proba(Xva)[:, 1]
            p_te = model.predict_proba(Xte)[:, 1]
            thr = 0.5   # chosen from the validation cost sweep (flat plateau 0.05-0.999)
            m = evaluate(yte, p_te, thr)
            mlflow.log_param("target_recall_on_val", TARGET_RECALL)
            mlflow.log_param("n_features", len(cols))
            mlflow.log_metrics(m)
            results[name] = m
            print(f"\n{name} (test, threshold from validation):",
                  {k: round(float(v), 4) for k, v in m.items()})
            if name == "lightgbm":
                imp = pd.Series(model.booster_.feature_importance(importance_type="gain"),
                                index=cols).sort_values(ascending=False)
                (ROOT / "models").mkdir(exist_ok=True)
                joblib.dump({"model": model, "features": cols, "threshold": thr},
                            ROOT / "models" / "lightgbm.joblib")
                parts = []
                for split_name, d, p in [("val", val, p_val), ("test", test, p_te)]:
                    o = d[["machine_id", "datetime", "target"]].copy()
                    o["p"] = p
                    o["rule_sig"] = rule_signature_24h(d)
                    o["split"] = split_name
                    parts.append(o)
                out = pd.concat(parts)
                out["thr"] = thr
                out.to_parquet(ROOT / "data" / "processed" / "predictions.parquet", index=False)
                print("\nTop 15 LightGBM features (share of total gain):")
                print((imp / imp.sum()).head(15).round(4))

    print("\nSummary (test):")
    print(pd.DataFrame(results).T[["pr_auc", "roc_auc", "precision", "recall", "f1"]].round(4))


if __name__ == "__main__":
    main()