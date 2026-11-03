"""Streaming scorer: same build_features + same saved model as batch (no train/serve skew)."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import joblib
import pandas as pd

from src.features.engineering import build_features

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "lightgbm.joblib"
BUFFER_HOURS = 200          # longest feature window is 168h
SENTINEL = pd.Timestamp("1990-01-01")


class StreamScorer:
    def __init__(self, machines: pd.DataFrame, model_path=MODEL_PATH,
                 buffer_hours: int = BUFFER_HOURS):
        self.machines = machines[["machine_id", "model", "age"]]
        self.buffer_hours = buffer_hours
        self.state: dict[int, dict] = {}
        self.model, self.cols, self.threshold = None, None, 0.5
        if model_path is not None:
            bundle = joblib.load(model_path)
            self.model = bundle["model"]
            self.cols = bundle["features"]
            self.threshold = float(bundle["threshold"])

    def _new_state(self) -> dict:
        return {"tele": deque(maxlen=self.buffer_hours), "errors": [], "maint": [], "last": None}

    def last_seen(self, m: int):
        st = self.state.get(m)
        return None if st is None else st["last"]

    def load_history(self, m: int, tele_rows, error_rows, maint_rows) -> None:
        """Reset a machine and preload history (all timestamps strictly before the stream)."""
        st = self._new_state()
        for r in tele_rows:
            r = dict(r)
            r["datetime"] = pd.Timestamp(r["datetime"])
            st["tele"].append(r)
        st["errors"] = [(pd.Timestamp(t), e) for t, e in error_rows]
        st["maint"] = [(pd.Timestamp(t), c) for t, c in maint_rows]
        st["last"] = st["tele"][-1]["datetime"] if st["tele"] else None
        self.state[m] = st

    def update(self, rec: dict) -> None:
        m, ts = rec["machine_id"], rec["ts"]
        st = self.state.setdefault(m, self._new_state())
        st["tele"].append({"machine_id": m, "datetime": ts, "volt": rec["volt"],
                           "rotate": rec["rotate"], "pressure": rec["pressure"],
                           "vibration": rec["vibration"]})
        for e in rec.get("errors", []):
            st["errors"].append((ts, e))
        for c in rec.get("maintenance", []):
            st["maint"].append((ts, c))
        cutoff = ts - pd.Timedelta(hours=self.buffer_hours)
        st["errors"] = [x for x in st["errors"] if x[0] > cutoff]
        st["last"] = ts

    def features(self, m: int) -> pd.DataFrame:
        st = self.state[m]
        tele = pd.DataFrame(list(st["tele"]))
        err = pd.DataFrame(st["errors"], columns=["datetime", "error_id"])
        if err.empty:   # sentinel row outside the telemetry grid keeps the code path simple
            err = pd.DataFrame({"datetime": [SENTINEL], "error_id": ["error1"]})
        err["machine_id"] = m
        maint = pd.DataFrame(st["maint"], columns=["datetime", "comp"])
        maint["datetime"] = pd.to_datetime(maint["datetime"]).astype("datetime64[ns]")
        maint["comp"] = maint["comp"].astype(object)
        maint["machine_id"] = m
        feats = build_features(tele, err, maint,
                               self.machines[self.machines["machine_id"] == m])
        return feats.iloc[[-1]]

    def score(self, m: int) -> float:
        row = self.features(m)
        X = row[self.cols].astype("float32")      # same cast as in training
        return float(self.model.predict_proba(X)[:, 1][0])