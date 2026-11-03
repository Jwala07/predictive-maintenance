import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from src.api.main import app
from src.models.train import ROOT
from src.streaming.common import get_engine

PRED = ROOT / "data" / "processed" / "predictions.parquet"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_rejects_short_or_gappy_history(client):
    rec = {"datetime": "2015-11-01T00:00:00", "volt": 170, "rotate": 450,
           "pressure": 100, "vibration": 40}
    assert client.post("/predict", json={"machine_id": 1, "history": [rec] * 5}).status_code == 422
    hours = pd.date_range("2015-11-01", periods=24, freq="h").delete(5)   # one hour missing
    hist = [{**rec, "datetime": t.isoformat()} for t in hours]
    assert client.post("/predict", json={"machine_id": 1, "history": hist}).status_code == 422


def test_unknown_machine_is_404(client):
    assert client.get("/machines/9999/risk").status_code == 404


def test_kpis_has_expected_keys(client):
    body = client.get("/kpis").json()
    for k in ("failures_by_component", "mtbf_hours_by_model", "alerts_raised"):
        assert k in body


@pytest.mark.skipif(not PRED.exists(), reason="run src.models.train first")
def test_predict_matches_batch_score(client):
    m, ts = 1, pd.Timestamp("2015-11-02 00:00")
    start = ts - pd.Timedelta(hours=199)
    p = {"m": m, "s": start.to_pydatetime(), "t": ts.to_pydatetime()}
    eng = get_engine()
    tele = pd.read_sql(text("SELECT datetime, volt, rotate, pressure, vibration FROM telemetry "
                            "WHERE machine_id=:m AND datetime>=:s AND datetime<=:t "
                            "ORDER BY datetime"), eng, params=p)
    err = pd.read_sql(text("SELECT datetime, error_id FROM errors WHERE machine_id=:m "
                           "AND datetime>=:s AND datetime<=:t"), eng, params=p)
    mnt = pd.read_sql(text("SELECT datetime, comp FROM maintenance WHERE machine_id=:m "
                           "AND datetime>=:s AND datetime<=:t"), eng, params=p)
    past = pd.read_sql(text("SELECT datetime, comp FROM maintenance WHERE machine_id=:m "
                            "AND datetime<:s"), eng, params=p)
    errs = err.groupby("datetime")["error_id"].apply(list).to_dict()
    mnts = mnt.groupby("datetime")["comp"].apply(list).to_dict()
    history = [{"datetime": r.datetime.isoformat(), "volt": r.volt, "rotate": r.rotate,
                "pressure": r.pressure, "vibration": r.vibration,
                "errors": errs.get(r.datetime, []), "maintenance": mnts.get(r.datetime, [])}
               for r in tele.itertuples()]
    body = {"machine_id": m, "history": history,
            "past_maintenance": [{"datetime": r.datetime.isoformat(), "comp": r.comp}
                                 for r in past.itertuples()]}
    got = client.post("/predict", json=body).json()["risk"]

    batch = pd.read_parquet(PRED)
    want = batch[(batch.machine_id == m) & (batch.datetime == ts)]["p"].iloc[0]
    assert abs(got - want) < 1e-6