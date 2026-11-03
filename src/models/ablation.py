"""Feature-group ablation with LightGBM (time-based split, threshold from validation)."""
import pandas as pd
from lightgbm import LGBMClassifier

from src.features.engineering import feature_columns
from src.models.train import DATA, evaluate, pick_threshold, split

SENSOR_PREFIX = ("volt", "rotate", "pressure", "vibration")


def main() -> None:
    df = pd.read_parquet(DATA)
    cols = feature_columns(df)
    df[cols] = df[cols].astype("float32")
    train, val, test = split(df)

    groups = {
        "all": cols,
        "no_maintenance": [c for c in cols if "maint" not in c],
        "no_sensors": [c for c in cols if not c.startswith(SENSOR_PREFIX)],
        "no_machine_attrs": [c for c in cols if c != "age" and not c.startswith("is_model")],
        "errors_only": [c for c in cols if c.startswith(("error", "err23", "n_error"))],
    }

    rows = {}
    for name, use in groups.items():
        model = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            scale_pos_weight=(train.target == 0).sum() / (train.target == 1).sum(),
            random_state=42, verbose=-1)
        model.fit(train[use], train["target"])
        thr = pick_threshold(val["target"], model.predict_proba(val[use])[:, 1])
        m = evaluate(test["target"], model.predict_proba(test[use])[:, 1], thr)
        rows[name] = {"n_features": len(use), **{k: m[k] for k in
                      ("pr_auc", "precision", "recall", "fp", "fn")}}
        print(name, "done")

    print(pd.DataFrame(rows).T.round(4))


if __name__ == "__main__":
    main()