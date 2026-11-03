import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.features.engineering import (
    COMPS, ERROR_IDS, NO_MAINT_HOURS, build_features, feature_columns,
)


def _toy_tables(n_hours: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2015-01-01 06:00", periods=n_hours, freq="h")
    tele = pd.concat(
        [pd.DataFrame({
            "machine_id": m, "datetime": hours,
            "volt": rng.normal(170, 15, n_hours),
            "rotate": rng.normal(450, 50, n_hours),
            "pressure": rng.normal(100, 10, n_hours),
            "vibration": rng.normal(40, 5, n_hours)}) for m in (1, 2)],
        ignore_index=True)
    errors = pd.DataFrame({
        "machine_id": rng.choice([1, 2], 60),
        "datetime": rng.choice(hours.to_numpy(), 60),
        "error_id": rng.choice(ERROR_IDS, 60)})
    maint = pd.DataFrame({
        "machine_id": rng.choice([1, 2], 12),
        "datetime": rng.choice(hours[::24].to_numpy(), 12),
        "comp": rng.choice(COMPS, 12)})
    machines = pd.DataFrame({"machine_id": [1, 2],
                             "model": ["model1", "model3"], "age": [5, 12]})
    return tele, errors, maint, machines


def test_features_use_only_past_data():
    """Cutting off the future must not change the features at time t."""
    tele, errors, maint, machines = _toy_tables()
    full = build_features(tele, errors, maint, machines)
    cols = feature_columns(full)

    for cut in pd.to_datetime(["2015-01-03 00:00", "2015-01-09 12:00",
                               "2015-01-15 05:00"]):
        part = build_features(
            tele[tele.datetime <= cut], errors[errors.datetime <= cut],
            maint[maint.datetime <= cut], machines)
        a = full[full.datetime == cut].sort_values("machine_id")[cols].reset_index(drop=True)
        b = part[part.datetime == cut].sort_values("machine_id")[cols].reset_index(drop=True)
        assert len(a) == 2
        assert_frame_equal(a, b, check_dtype=False)


def test_error_features_semantics():
    hours = pd.date_range("2015-01-01 06:00", periods=100, freq="h")
    tele = pd.DataFrame({"machine_id": 1, "datetime": hours, "volt": 170.0,
                         "rotate": 450.0, "pressure": 100.0, "vibration": 40.0})
    t0 = hours[10]
    errors = pd.DataFrame({"machine_id": [1, 1, 1],
                           "datetime": [t0, t0, hours[50]],
                           "error_id": ["error2", "error3", "error2"]})
    maint = pd.DataFrame({"machine_id": pd.Series(dtype="int64"),
                          "datetime": pd.Series(dtype="datetime64[ns]"),
                          "comp": pd.Series(dtype="object")})
    machines = pd.DataFrame({"machine_id": [1], "model": ["model2"], "age": [7]})

    f = build_features(tele, errors, maint, machines).set_index("datetime")

    assert f.loc[t0, "err23_now"] == 1                  # error2 + error3 together
    assert f.loc[hours[50], "err23_now"] == 0           # error2 alone is not the combo
    assert f.loc[hours[34], "error2_hours_since"] == 24
    assert f.loc[hours[9], "error2_hours_since"] == 168  # nothing seen yet -> cap
    assert f.loc[hours[33], "error2_cnt_24h"] == 1       # window covers hours 10..33
    assert f.loc[hours[34], "error2_cnt_24h"] == 0       # error has left the window
    assert f.loc[t0, "comp1_hours_since_maint"] == NO_MAINT_HOURS
    assert f.loc[t0, "is_model2"] == 1 and f.loc[t0, "is_model1"] == 0