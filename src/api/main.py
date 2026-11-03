"""FastAPI service: health, predict, per-machine risk, KPIs."""
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from sqlalchemy import text

from src.api.schemas import PredictRequest, PredictResponse
from src.streaming.common import get_engine
from src.streaming.scoring import StreamScorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_engine()
    machines = pd.read_sql("SELECT * FROM machines", engine)
    app.state.engine = engine
    app.state.machines = machines
    app.state.base = StreamScorer(machines)      # model is loaded once at startup
    yield


app = FastAPI(title="Predictive Maintenance API", version="1.0", lifespan=lifespan)


def _risk_level(risk: float, thr: float) -> str:
    # medium cut-off (0.2) is an arbitrary display choice, not tuned
    return "high" if risk >= thr else ("medium" if risk >= 0.2 else "low")


@app.get("/health")
def health(request: Request):
    db_ok = True
    try:
        with request.app.state.engine.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "database": db_ok,
            "model_loaded": request.app.state.base.model is not None,
            "threshold": request.app.state.base.threshold}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    st = request.app.state
    if req.machine_id not in set(st.machines["machine_id"]):
        raise HTTPException(404, f"unknown machine_id {req.machine_id}")

    ts = [pd.Timestamp(r.datetime) for r in req.history]
    if any(t.tzinfo is not None for t in ts):
        raise HTTPException(422, "timestamps must be timezone-naive (like the dataset)")
    steps = {b - a for a, b in zip(ts, ts[1:])}
    if steps != {pd.Timedelta(hours=1)}:
        raise HTTPException(422, "history must be sorted and strictly hourly with no gaps")
    last = ts[-1]
    if any(pd.Timestamp(p.datetime) > last for p in req.past_maintenance):
        raise HTTPException(422, "past_maintenance must not be later than the last history row")

    scorer = StreamScorer(st.machines, model_path=None)   # cheap, shares the loaded model
    scorer.model, scorer.cols, scorer.threshold = (
        st.base.model, st.base.cols, st.base.threshold)
    m = req.machine_id
    scorer.load_history(m, [], [], [(p.datetime, p.comp) for p in req.past_maintenance])
    for r, t in zip(req.history, ts):
        scorer.update({"machine_id": m, "ts": t, "volt": r.volt, "rotate": r.rotate,
                       "pressure": r.pressure, "vibration": r.vibration,
                       "errors": r.errors, "maintenance": r.maintenance})
    risk = scorer.score(m)

    warnings = []
    if len(req.history) < 168:
        warnings.append("history shorter than 168h: 'hours since error' features are capped "
                        "and may differ from the batch pipeline")
    if not req.past_maintenance and not any(r.maintenance for r in req.history):
        warnings.append("no maintenance records given: maintenance features default to 'never "
                        "maintained', which may differ from the real history")
    return PredictResponse(machine_id=m, datetime=last.to_pydatetime(), risk=risk,
                           alert=risk >= scorer.threshold,
                           risk_level=_risk_level(risk, scorer.threshold),
                           threshold=scorer.threshold, history_hours=len(req.history),
                           warnings=warnings)


@app.get("/machines/{machine_id}/risk")
def machine_risk(machine_id: int, request: Request, limit: int = Query(24, ge=1, le=500)):
    st = request.app.state
    if machine_id not in set(st.machines["machine_id"]):
        raise HTTPException(404, f"unknown machine_id {machine_id}")
    df = pd.read_sql(
        text("SELECT datetime, risk, alert FROM predictions WHERE machine_id=:m "
             "ORDER BY datetime DESC LIMIT :n"), st.engine, params={"m": machine_id, "n": limit})
    if df.empty:
        raise HTTPException(404, f"no predictions stored yet for machine {machine_id}")
    latest = df.iloc[0]
    return {"machine_id": machine_id, "latest": {
                "datetime": latest["datetime"].isoformat(), "risk": float(latest["risk"]),
                "alert": bool(latest["alert"])},
            "history": [{"datetime": r.datetime.isoformat(), "risk": float(r.risk),
                         "alert": bool(r.alert)} for r in df.itertuples()]}


@app.get("/kpis")
def kpis(request: Request):
    eng = request.app.state.engine
    by_comp = pd.read_sql("SELECT failure AS component, COUNT(*) AS n FROM failures "
                          "GROUP BY failure ORDER BY failure", eng)
    mtbf = pd.read_sql("""
        WITH ev AS (SELECT DISTINCT machine_id, datetime FROM failures),
        gaps AS (SELECT machine_id,
                 EXTRACT(EPOCH FROM (datetime - LAG(datetime) OVER
                   (PARTITION BY machine_id ORDER BY datetime))) / 3600.0 AS gap_hours FROM ev)
        SELECT m.model, ROUND(AVG(g.gap_hours)::numeric, 1) AS mtbf_hours
        FROM gaps g JOIN machines m USING (machine_id)
        WHERE g.gap_hours IS NOT NULL GROUP BY m.model ORDER BY m.model""", eng)
    with eng.connect() as c:
        n_pred = c.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
        n_alert = c.execute(text("SELECT COUNT(*) FROM alerts")).scalar()
        top = c.execute(text("SELECT machine_id, MAX(risk) FROM predictions "
                             "GROUP BY machine_id ORDER BY 2 DESC, 1 LIMIT 5")).fetchall()
    return {"failures_by_component": dict(zip(by_comp["component"], by_comp["n"].astype(int))),
            "mtbf_hours_by_model": {r.model: float(r.mtbf_hours) for r in mtbf.itertuples()},
            "mtbf_note": "mean gap between consecutive failures only; machines with <2 failures excluded",
            "predictions_stored": int(n_pred), "alerts_raised": int(n_alert),
            "highest_risk_machines": [{"machine_id": int(m), "max_risk": float(r)} for m, r in top]}