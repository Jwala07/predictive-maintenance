import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.features.engineering import COMPS, ERROR_IDS, build_features, feature_columns
from src.streaming.scoring import StreamScorer


def _toy(n_hours=400, seed=1):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2015-01-01 06:00", periods=n_hours, freq="h")
    tele = pd.concat([pd.DataFrame({
        "machine_id": m, "datetime": hours,
        "volt": rng.normal(170, 15, n_hours), "rotate": rng.normal(450, 50, n_hours),
        "pressure": rng.normal(100, 10, n_hours), "vibration": rng.normal(40, 5, n_hours)})
        for m in (1, 2)], ignore_index=True)
    errors = pd.DataFrame({"machine_id": rng.choice([1, 2], 80),
                           "datetime": rng.choice(hours.to_numpy(), 80),
                           "error_id": rng.choice(ERROR_IDS, 80)})
    maint = pd.DataFrame({"machine_id": rng.choice([1, 2], 12),
                          "datetime": rng.choice(hours[::24].to_numpy(), 12),
                          "comp": rng.choice(COMPS, 12)})
    machines = pd.DataFrame({"machine_id": [1, 2], "model": ["model1", "model3"], "age": [5, 12]})
    return tele, errors, maint, machines


def test_stream_features_equal_batch_features():
    tele, errors, maint, machines = _toy()
    batch = build_features(tele, errors, maint, machines)
    cols = feature_columns(batch)
    scorer = StreamScorer(machines, model_path=None)

    hours = sorted(pd.to_datetime(tele["datetime"].unique()))
    checks = {hours[10], hours[199], hours[250], hours[399]}   # 250/399 test the buffer trimming
    err_g = {k: list(g["error_id"]) for k, g in errors.groupby(["machine_id", "datetime"])}
    mnt_g = {k: list(g["comp"]) for k, g in maint.groupby(["machine_id", "datetime"])}

    for ts in hours:
        for m in (1, 2):
            row = tele[(tele.machine_id == m) & (tele.datetime == ts)].iloc[0]
            scorer.update({"machine_id": m, "ts": ts, "volt": row.volt, "rotate": row.rotate,
                           "pressure": row.pressure, "vibration": row.vibration,
                           "errors": err_g.get((m, ts), []), "maintenance": mnt_g.get((m, ts), [])})
            if ts in checks:
                got = scorer.features(m)[cols].reset_index(drop=True)
                want = batch[(batch.machine_id == m) & (batch.datetime == ts)][cols].reset_index(drop=True)
                assert_frame_equal(got, want, check_dtype=False)