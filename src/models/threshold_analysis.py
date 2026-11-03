"""Threshold sweep + event-level cost analysis. Threshold is chosen on validation only."""
import numpy as np
import pandas as pd

from src.models.train import ROOT

PRED = ROOT / "data" / "processed" / "predictions.parquet"

# ASSUMPTIONS (not in the dataset, change freely)
FAIL_COST = 16000         # unplanned failure: 8 h downtime x 2000 per hour
PLANNED_COST = 3000       # planned repair after a correct alert
FALSE_ALARM_COST = 500    # inspection after a false alarm
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99,
              0.999, 0.9999, 0.99999]


def runs(d: pd.DataFrame, mask) -> np.ndarray:
    """Run id for consecutive hourly rows (same machine) where mask is True; 0 elsewhere."""
    mask = np.asarray(mask, dtype=bool)
    mid = d["machine_id"].to_numpy()
    dt = d["datetime"].to_numpy()
    cont = np.zeros(len(d), dtype=bool)
    cont[1:] = (mid[1:] == mid[:-1]) & ((dt[1:] - dt[:-1]) == np.timedelta64(1, "h"))
    prev = np.zeros(len(d), dtype=bool)
    prev[1:] = mask[:-1]
    start = mask & ~(prev & cont)
    return np.where(mask, np.cumsum(start), 0)


def evaluate(d: pd.DataFrame, score, thr: float) -> dict:
    alert = np.asarray(score) >= thr
    y = d["target"].to_numpy() == 1
    ev = runs(d, y)                      # failure events (24h windows, overlaps merged)
    al = runs(d, alert)                  # alert episodes
    n_events = int(ev.max())
    detected = int(np.unique(ev[y & alert]).size)
    missed = n_events - detected
    all_ep = np.unique(al[al > 0]).size
    true_ep = np.unique(al[(al > 0) & y]).size
    false_ep = all_ep - true_ep
    cost = missed * FAIL_COST + detected * PLANNED_COST + false_ep * FALSE_ALARM_COST
    return {"thr": thr, "events": n_events, "detected": detected, "missed": missed,
            "false_alarm_episodes": false_ep,
            "row_recall": round(float((alert & y).sum() / max(y.sum(), 1)), 4),
            "row_precision": round(float((alert & y).sum() / max(alert.sum(), 1)), 4),
            "cost": cost}


def main() -> None:
    df = pd.read_parquet(PRED)
    val = df[df["split"] == "val"].sort_values(["machine_id", "datetime"]).reset_index(drop=True)
    test = df[df["split"] == "test"].sort_values(["machine_id", "datetime"]).reset_index(drop=True)

    thr_old = float(test["thr"].iloc[0])
    print(f"Threshold used in train.py (full precision): {thr_old!r}")
    pos = test.loc[test["target"] == 1, "p"]
    print("Probability on positive test rows, quantiles 5/25/50/75%:",
          pos.quantile([0.05, 0.25, 0.5, 0.75]).round(6).tolist())
    print(f"Positive rows with p >= old threshold: {(pos >= thr_old).mean():.3f}")

    grid = THRESHOLDS + [thr_old]
    val_tab = pd.DataFrame([evaluate(val, val["p"], t) for t in grid])
    test_tab = pd.DataFrame([evaluate(test, test["p"], t) for t in grid])
    print("\nVALIDATION sweep:\n", val_tab.to_string(index=False))
    print("\nTEST sweep:\n", test_tab.to_string(index=False))

    best = float(val_tab.loc[val_tab["cost"].idxmin(), "thr"])
    print(f"\nThreshold with the lowest cost on VALIDATION: {best}")
    print("Applied to TEST:", evaluate(test, test["p"], best))

    print("\nBaselines on TEST:")
    print(" rule_signature_24h:", evaluate(test, test["rule_sig"], 0.5))
    n_events = int(runs(test, test["target"] == 1).max())
    print(f" no alerts at all : cost {n_events * FAIL_COST} ({n_events} failures)")


if __name__ == "__main__":
    main()