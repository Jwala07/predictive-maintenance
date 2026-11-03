import pandas as pd

from src.features.target import make_target


def test_target_boundaries():
    fail_ts = pd.Timestamp("2015-01-02 10:00")
    failures = pd.DataFrame({"machine_id": [1], "datetime": [fail_ts]})

    hours = pd.date_range("2015-01-01 08:00", "2015-01-02 12:00", freq="h")
    keys = pd.concat(
        [pd.DataFrame({"machine_id": m, "datetime": hours}) for m in (1, 2)],
        ignore_index=True,
    )
    keys["y"] = make_target(keys, failures)

    m1 = keys[keys.machine_id == 1].set_index("datetime")["y"]
    assert m1[pd.Timestamp("2015-01-01 09:00")] == 0   # 25h before: outside window
    assert m1[pd.Timestamp("2015-01-01 10:00")] == 1   # exactly 24h before: inclusive
    assert m1[pd.Timestamp("2015-01-02 09:00")] == 1   # 1h before
    assert m1[fail_ts] == 0                            # at failure hour: failure is not in the future
    assert m1[pd.Timestamp("2015-01-02 11:00")] == 0   # after failure
    assert m1.sum() == 24                              # exactly 24 positive hours per failure

    assert (keys[keys.machine_id == 2]["y"] == 0).all()  # other machine unaffected