"""Consume telemetry, validate, store, score with the saved model, raise alerts."""
import argparse
import json
import math
import time

import pandas as pd
from confluent_kafka import Consumer
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from src.streaming.common import BROKER, TOPIC, get_engine, get_logger
from src.streaming.scoring import BUFFER_HOURS, StreamScorer

log = get_logger("consumer")
SENSORS = ["volt", "rotate", "pressure", "vibration"]
REQUIRED = ["machine_id", "datetime"] + SENSORS


def validate(rec: dict) -> dict:
    for k in REQUIRED:
        if k not in rec or rec[k] is None:
            raise ValueError(f"missing field {k}")
    out = {"machine_id": int(rec["machine_id"]), "ts": pd.Timestamp(rec["datetime"]),
           "errors": list(rec.get("errors", [])), "maintenance": list(rec.get("maintenance", []))}
    for s in SENSORS:
        v = float(rec[s])
        if not math.isfinite(v) or not (0 < v < 1000):
            raise ValueError(f"bad value {s}={v}")
        out[s] = v
    return out


def warm_start(engine, scorer: StreamScorer, m: int, ts: pd.Timestamp) -> None:
    p = {"m": m, "s": (ts - pd.Timedelta(hours=BUFFER_HOURS - 1)).to_pydatetime(),
         "t": ts.to_pydatetime()}
    tele = pd.read_sql(text("SELECT machine_id, datetime, volt, rotate, pressure, vibration "
                            "FROM telemetry WHERE machine_id=:m AND datetime>=:s AND datetime<:t "
                            "ORDER BY datetime"), engine, params=p)
    err = pd.read_sql(text("SELECT datetime, error_id FROM errors "
                           "WHERE machine_id=:m AND datetime>=:s AND datetime<:t"), engine, params=p)
    mnt = pd.read_sql(text("SELECT datetime, comp FROM maintenance "
                           "WHERE machine_id=:m AND datetime<:t"), engine, params=p)
    scorer.load_history(m, tele.to_dict("records"),
                        list(zip(err["datetime"], err["error_id"])),
                        list(zip(mnt["datetime"], mnt["comp"])))


def write_result(engine, rec: dict, risk: float, alert: bool, thr: float) -> None:
    p = {"m": rec["machine_id"], "t": rec["ts"].to_pydatetime(), "risk": risk,
         "alert": alert, "thr": thr, **{s: rec[s] for s in SENSORS}}
    for attempt in range(1, 4):
        try:
            with engine.begin() as conn:
                conn.execute(text(
                    "INSERT INTO live_telemetry (machine_id, datetime, volt, rotate, pressure, vibration) "
                    "VALUES (:m, :t, :volt, :rotate, :pressure, :vibration) ON CONFLICT DO NOTHING"), p)
                conn.execute(text(
                    "INSERT INTO predictions (machine_id, datetime, risk, alert) "
                    "VALUES (:m, :t, :risk, :alert) ON CONFLICT (machine_id, datetime) "
                    "DO UPDATE SET risk = EXCLUDED.risk, alert = EXCLUDED.alert"), p)
                if alert:
                    conn.execute(text(
                        "INSERT INTO alerts (machine_id, datetime, risk, threshold) "
                        "VALUES (:m, :t, :risk, :thr) ON CONFLICT DO NOTHING"), p)
            return
        except OperationalError as e:
            log.warning("db_retry attempt=%d error=%s", attempt, str(e)[:100])
            time.sleep(2 ** attempt)
    raise RuntimeError("database unavailable after 3 retries")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument("--group", default="pdm-consumer")
    ap.add_argument("--idle-exit", type=float, default=30, help="stop after N idle seconds")
    a = ap.parse_args()

    engine = get_engine()
    machines = pd.read_sql("SELECT * FROM machines", engine)
    scorer = StreamScorer(machines)
    log.info("model_loaded threshold=%s n_features=%d", scorer.threshold, len(scorer.cols))

    c = Consumer({"bootstrap.servers": BROKER, "group.id": a.group,
                  "auto.offset.reset": "earliest"})
    c.subscribe([a.topic])
    n = alerts = bad = dup = 0
    last_msg = time.time()
    try:
        while True:
            msg = c.poll(1.0)
            if msg is None:
                if n and time.time() - last_msg > a.idle_exit:
                    break
                continue
            if msg.error():
                log.warning("kafka_error detail=%s", msg.error())
                continue
            last_msg = time.time()
            try:
                rec = validate(json.loads(msg.value()))
            except Exception as e:
                bad += 1
                log.warning("invalid_message reason=%s", str(e)[:100])
                continue
            m, ts = rec["machine_id"], rec["ts"]
            last = scorer.last_seen(m)
            if last is None or (ts - last) > pd.Timedelta(hours=1):
                if last is not None:
                    log.warning("gap machine=%d last=%s now=%s reloading history", m, last, ts)
                warm_start(engine, scorer, m, ts)
            elif ts <= last:
                dup += 1
                continue
            scorer.update(rec)
            risk = scorer.score(m)
            alert = risk >= scorer.threshold
            write_result(engine, rec, risk, alert, scorer.threshold)
            n += 1
            alerts += int(alert)
            if n % 200 == 0:
                log.info("progress processed=%d alerts=%d invalid=%d duplicates=%d", n, alerts, bad, dup)
    finally:
        c.close()
    log.info("done processed=%d alerts=%d invalid=%d duplicates=%d", n, alerts, bad, dup)


if __name__ == "__main__":
    main()