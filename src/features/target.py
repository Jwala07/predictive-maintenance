"""Target (label) creation for the 24h failure-prediction task."""
import numpy as np
import pandas as pd

HORIZON_HOURS = 24


def make_target(
    keys: pd.DataFrame,
    failures: pd.DataFrame,
    horizon_hours: int = HORIZON_HOURS,
) -> pd.Series:
    """Label = 1 if any failure happens in (t, t + horizon], bounds: open left, closed right.

    keys: DataFrame with columns machine_id, datetime (one row per prediction time).
    failures: DataFrame with columns machine_id, datetime.
    Returns an int8 Series aligned with keys.index.
    """
    out = pd.Series(0, index=keys.index, dtype="int8", name="target")
    horizon = np.timedelta64(horizon_hours, "h")

    fail_by_machine = {
        m: np.sort(g["datetime"].to_numpy().astype("datetime64[ns]"))
        for m, g in failures.groupby("machine_id")
    }

    for machine_id, g in keys.groupby("machine_id"):
        f = fail_by_machine.get(machine_id)
        if f is None:
            continue
        t = g["datetime"].to_numpy().astype("datetime64[ns]")
        nxt = np.searchsorted(f, t, side="right")      # first failure strictly after t
        has_next = nxt < len(f)
        nxt_ts = f[np.minimum(nxt, len(f) - 1)]
        out.loc[g.index] = (has_next & (nxt_ts <= t + horizon)).astype("int8")
    return out