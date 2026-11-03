"""Export KPI tables as CSV for Power BI / Tableau (Get Data > Text/CSV)."""
from pathlib import Path

import pandas as pd

from src.streaming.common import get_engine

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "exports"
HOURS_PER_FAILURE, COST_PER_HOUR = 8, 2000        # ASSUMPTIONS, not in the dataset


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eng = get_engine()
    pd.read_sql("SELECT failure AS component, COUNT(*) AS failures FROM failures "
                "GROUP BY failure ORDER BY failure", eng).to_csv(OUT / "failures_by_component.csv", index=False)
    pd.read_sql("SELECT DATE_TRUNC('month', datetime)::date AS month, failure AS component, "
                "COUNT(*) AS failures FROM failures GROUP BY 1, 2 ORDER BY 1, 2",
                eng).to_csv(OUT / "monthly_failures.csv", index=False)

    fm = pd.read_sql("""
        WITH ev AS (SELECT DISTINCT machine_id, datetime FROM failures)
        SELECT m.model, COUNT(DISTINCT m.machine_id) AS machines, COUNT(ev.datetime) AS failure_events
        FROM machines m LEFT JOIN ev USING (machine_id) GROUP BY m.model ORDER BY m.model""", eng)
    mt = pd.read_sql("""
        WITH ev AS (SELECT DISTINCT machine_id, datetime FROM failures),
        gaps AS (SELECT machine_id, EXTRACT(EPOCH FROM (datetime - LAG(datetime) OVER
                 (PARTITION BY machine_id ORDER BY datetime))) / 3600.0 AS gap FROM ev)
        SELECT m.model, AVG(g.gap)::float AS mtbf_hours
        FROM gaps g JOIN machines m USING (machine_id) WHERE g.gap IS NOT NULL GROUP BY m.model""", eng)
    fm = fm.merge(mt, on="model", how="left")
    fm["assumed_downtime_hours"] = fm["failure_events"] * HOURS_PER_FAILURE
    fm["assumed_cost"] = fm["assumed_downtime_hours"] * COST_PER_HOUR
    fm.to_csv(OUT / "model_kpis.csv", index=False)

    pred = pd.read_parquet(ROOT / "data" / "processed" / "predictions.parquet")
    test = pred[pred["split"] == "test"].copy()
    test["alert"] = test["p"] >= test["thr"]
    test["date"] = test["datetime"].dt.date
    (test.groupby(["machine_id", "date"])
         .agg(max_risk=("p", "max"), alert_hours=("alert", "sum")).reset_index()
         .to_csv(OUT / "machine_risk_daily.csv", index=False))

    machines = pd.read_sql("SELECT * FROM machines", eng)
    summ = test.groupby("machine_id").agg(test_max_risk=("p", "max"),
                                          test_alert_hours=("alert", "sum")).reset_index()
    machines.merge(summ, on="machine_id", how="left").to_csv(OUT / "machine_summary.csv", index=False)
    print("Exported to", OUT)


if __name__ == "__main__":
    main()