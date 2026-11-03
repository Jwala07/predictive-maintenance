"""Replay historical telemetry hour by hour into Kafka (simulated IoT sensors)."""
import argparse
import json
import time
from collections import defaultdict

import pandas as pd
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
from sqlalchemy import text

from src.streaming.common import BROKER, TOPIC, get_engine, get_logger

log = get_logger("producer")


def ensure_topic(topic: str) -> None:
    admin = AdminClient({"bootstrap.servers": BROKER})
    for fut in admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)]).values():
        try:
            fut.result()
        except Exception as e:       # already exists is fine
            log.info("topic_note detail=%s", str(e)[:80])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-11-01 00:00")
    ap.add_argument("--hours", type=int, default=72)
    ap.add_argument("--machines", type=int, default=20, help="machine_id 1..N")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between simulated hours")
    ap.add_argument("--topic", default=TOPIC)
    a = ap.parse_args()

    start = pd.Timestamp(a.start)
    end = start + pd.Timedelta(hours=a.hours)
    p = {"s": start.to_pydatetime(), "e": end.to_pydatetime(), "n": a.machines}
    eng = get_engine()
    where = "datetime >= :s AND datetime < :e AND machine_id <= :n"
    tele = pd.read_sql(text(f"SELECT * FROM telemetry WHERE {where} ORDER BY datetime, machine_id"),
                       eng, params=p)
    errs = pd.read_sql(text(f"SELECT machine_id, datetime, error_id FROM errors WHERE {where}"), eng, params=p)
    maint = pd.read_sql(text(f"SELECT machine_id, datetime, comp FROM maintenance WHERE {where}"), eng, params=p)

    err_map, maint_map = defaultdict(list), defaultdict(list)
    for r in errs.itertuples():
        err_map[(r.machine_id, r.datetime)].append(r.error_id)
    for r in maint.itertuples():
        maint_map[(r.machine_id, r.datetime)].append(r.comp)

    ensure_topic(a.topic)
    prod = Producer({"bootstrap.servers": BROKER})
    log.info("replay_start rows=%d start=%s hours=%d machines=%d", len(tele), start, a.hours, a.machines)

    current, sent = None, 0
    for r in tele.itertuples():
        if current is not None and r.datetime != current and a.delay > 0:
            time.sleep(a.delay)
        current = r.datetime
        rec = {"machine_id": int(r.machine_id), "datetime": r.datetime.isoformat(),
               "volt": float(r.volt), "rotate": float(r.rotate),
               "pressure": float(r.pressure), "vibration": float(r.vibration),
               "errors": err_map.get((r.machine_id, r.datetime), []),
               "maintenance": maint_map.get((r.machine_id, r.datetime), [])}
        prod.produce(a.topic, key=str(r.machine_id), value=json.dumps(rec))
        sent += 1
        if sent % 500 == 0:
            prod.poll(0)
    prod.flush(30)
    log.info("replay_done sent=%d", sent)


if __name__ == "__main__":
    main()