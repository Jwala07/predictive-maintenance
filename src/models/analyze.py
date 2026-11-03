"""Event-level evaluation of the saved predictions."""
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.models.train import ROOT

PRED = ROOT / "data" / "processed" / "predictions.parquet"


def precision_at_recall(y, p, targets=(1.0, 0.99, 0.95, 0.90)) -> dict:
    prec, rec, _ = precision_recall_curve(y, p)
    return {t: round(float(prec[rec >= t].max()), 4) for t in targets}


def event_report(d: pd.DataFrame, score_col: str, thr: float, name: str) -> None:
    """An event = a run of consecutive positive hours (24h before a failure).
    Overlapping failures merge into one event. Detected = any alert inside it."""
    d = d.sort_values(["machine_id", "datetime"])
    alert = d[score_col] >= thr
    pos = d[d["target"] == 1].copy()
    pos["alert"] = alert[pos.index]
    new = ((pos["machine_id"] != pos["machine_id"].shift())
           | (pos["datetime"].diff() != pd.Timedelta(hours=1)))
    pos["event"] = new.cumsum()
    g = pos.groupby("event")
    detected = g["alert"].any()
    end = g["datetime"].max()
    first = pos[pos["alert"]].groupby("event")["datetime"].min()
    lead = ((end + pd.Timedelta(hours=1) - first) / pd.Timedelta(hours=1)).dropna()
    fp_rows = int((alert & (d["target"] == 0)).sum())

    print(f"\n{name}")
    print(f"  events: {len(detected)} | detected: {int(detected.sum())} "
          f"({100 * detected.mean():.1f}%) | missed: {int((~detected).sum())}")
    print(f"  lead time (h) before failure: median {lead.median():.0f}, "
          f"min {lead.min():.0f}, max {lead.max():.0f}")
    print(f"  false-alarm rows: {fp_rows}")
    missed = pos[pos["event"].isin(detected[~detected].index)]
    if len(missed):
        print("  missed events (machine, failure time):")
        for ev, m in missed.groupby("event"):
            print(f"    machine {int(m.machine_id.iloc[0])}, "
                  f"{m.datetime.max() + pd.Timedelta(hours=1)}")


def main() -> None:
    df = pd.read_parquet(PRED)
    test = df[df["split"] == "test"]
    thr = float(test["thr"].iloc[0])

    print("LightGBM precision at fixed recall (test):",
          precision_at_recall(test["target"], test["p"]))
    event_report(test, "p", thr, f"LightGBM (threshold {thr:.4f})")
    event_report(test, "rule_sig", 0.5, "rule_signature_24h")


if __name__ == "__main__":
    main()