"""Leakage-safe feature engineering, shared by training and serving.

Rule: the features of row (machine, t) use only data with datetime <= t.
Input tables use the same column names as the database tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SENSORS = ["volt", "rotate", "pressure", "vibration"]
SENSOR_WINDOWS = [3, 12, 24]
ERROR_IDS = [f"error{i}" for i in range(1, 6)]
ERROR_WINDOWS = [24, 48]
COMPS = [f"comp{i}" for i in range(1, 5)]
MODELS = [f"model{i}" for i in range(1, 5)]
HOURS_CAP = 168            # "hours since last error" is capped here (also = never seen)
NO_MAINT_HOURS = 24 * 365  # used when no maintenance of that component was seen yet


def _norm_time(df: pd.DataFrame) -> pd.DataFrame:
    """Copy and force datetime to datetime64[ns] so merges never mismatch."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"]).astype("datetime64[ns]")
    return df


def _hours_since(flag: pd.Series, cap: int = HOURS_CAP) -> pd.Series:
    """Rows since the last row where flag > 0 (0 on that row). Needs hourly rows."""
    pos = pd.Series(np.arange(len(flag)), index=flag.index, dtype="float64")
    last = pos.where(flag.to_numpy() > 0).ffill()
    return (pos - last).fillna(cap).clip(upper=cap)


def build_features(
    telemetry: pd.DataFrame,
    errors: pd.DataFrame,
    maintenance: pd.DataFrame,
    machines: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per (machine_id, datetime) of telemetry with all features."""
    tele = (_norm_time(telemetry)
            .sort_values(["machine_id", "datetime"])
            .reset_index(drop=True))
    out = tele[["machine_id", "datetime"]].copy()
    g = tele.groupby("machine_id", sort=False)

    # ---- 1) sensors: current value + rolling mean/std (window includes current row)
    for s in SENSORS:
        out[s] = tele[s]
        for w in SENSOR_WINDOWS:
            out[f"{s}_mean_{w}h"] = g[s].transform(
                lambda x: x.rolling(w, min_periods=1).mean())
            out[f"{s}_std_{w}h"] = g[s].transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0))

    # ---- 2) errors: per-type counts on the hourly grid
    err = _norm_time(errors)
    counts = (err.groupby(["machine_id", "datetime", "error_id"]).size()
                 .unstack("error_id", fill_value=0)
                 .reindex(columns=ERROR_IDS, fill_value=0)
                 .reset_index())
    cur = out[["machine_id", "datetime"]].merge(
        counts, on=["machine_id", "datetime"], how="left")
    cur[ERROR_IDS] = cur[ERROR_IDS].fillna(0).astype("int16")

    for e in ERROR_IDS:
        cg = cur.groupby("machine_id", sort=False)[e]
        out[f"{e}_now"] = cur[e]
        for w in ERROR_WINDOWS:
            out[f"{e}_cnt_{w}h"] = cg.transform(
                lambda x: x.rolling(w, min_periods=1).sum())
        out[f"{e}_hours_since"] = cg.transform(_hours_since)

    # combo: error2 AND error3 in the same hour (comp2 signature)
    both = ((cur["error2"] > 0) & (cur["error3"] > 0)).astype("int8")
    out["err23_now"] = both
    out["err23_hours_since"] = both.groupby(cur["machine_id"], sort=False).transform(
        _hours_since)
    out["n_error_types_now"] = (cur[ERROR_IDS] > 0).sum(axis=1).astype("int8")

    # ---- 3) maintenance: hours since last maintenance of each component (<= t)
    maint = _norm_time(maintenance)
    base = out[["machine_id", "datetime"]].copy()
    base["_row"] = np.arange(len(base))
    base_sorted = base.sort_values("datetime")      # merge_asof needs sorted key
    for c in COMPS:
        mc = (maint.loc[maint["comp"] == c, ["machine_id", "datetime"]]
                   .rename(columns={"datetime": "maint_ts"})
                   .sort_values("maint_ts"))
        merged = pd.merge_asof(
            base_sorted, mc, left_on="datetime", right_on="maint_ts",
            by="machine_id", direction="backward", allow_exact_matches=True)
        merged = merged.sort_values("_row")
        hrs = (merged["datetime"] - merged["maint_ts"]) / pd.Timedelta(hours=1)
        out[f"{c}_hours_since_maint"] = hrs.fillna(NO_MAINT_HOURS).to_numpy()

    # ---- 4) machine attributes
    out = out.merge(machines[["machine_id", "model", "age"]], on="machine_id", how="left")
    for m in MODELS:
        out[f"is_{m}"] = (out["model"] == m).astype("int8")
    out = out.drop(columns="model")

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All model input columns (everything except keys and target)."""
    return [c for c in df.columns if c not in ("machine_id", "datetime", "target")]