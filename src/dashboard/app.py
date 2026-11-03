"""Streamlit dashboard: live stream tab (auto-refresh from DB) + test-period analysis tab."""
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import text

from src.streaming.common import get_engine

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "data" / "processed" / "predictions.parquet"

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance Dashboard")
st.caption("Simulated Azure PdM data. 'Live' = replay of historical telemetry through Kafka, "
           "not real sensors. Costs are assumptions, not dataset values.")


@st.cache_resource
def engine():
    return get_engine()


def q(sql: str, **params) -> pd.DataFrame:
    return pd.read_sql(text(sql), engine(), params=params)


# ---------------------------------------------------------------- live tab
@st.fragment(run_every=5)
def live_view():
    stats = q("SELECT COUNT(*) AS n, MAX(datetime) AS last_ts, "
              "COUNT(DISTINCT machine_id) AS machines FROM predictions").iloc[0]
    n_alerts = int(q("SELECT COUNT(*) AS n FROM alerts").iloc[0]["n"])
    c = st.columns(4)
    c[0].metric("Predictions stored", int(stats["n"]))
    c[1].metric("Machines scored", int(stats["machines"]))
    c[2].metric("Alerts raised", n_alerts)
    c[3].metric("Latest simulated hour", str(stats["last_ts"])[:16])
    if int(stats["n"]) == 0:
        st.info("No predictions yet. Start the consumer and producer (see the steps).")
        return

    left, right = st.columns(2)
    latest = q("SELECT DISTINCT ON (machine_id) machine_id, datetime, risk, alert "
               "FROM predictions ORDER BY machine_id, datetime DESC")
    with left:
        st.subheader("Latest risk per machine")
        st.bar_chart(latest.set_index("machine_id")["risk"])
    with right:
        st.subheader("Recent alerts")
        st.dataframe(q("SELECT machine_id, datetime, risk FROM alerts "
                       "ORDER BY datetime DESC, machine_id LIMIT 15"),
                     hide_index=True, use_container_width=True)

    st.subheader("Machine detail (live)")
    m = st.selectbox("Machine", sorted(latest["machine_id"].tolist()), key="live_m")
    d = q("SELECT t.datetime, t.volt, t.rotate, t.pressure, t.vibration, p.risk "
          "FROM live_telemetry t JOIN predictions p USING (machine_id, datetime) "
          "WHERE t.machine_id = :m ORDER BY t.datetime DESC LIMIT 72", m=int(m))
    d = d.sort_values("datetime").set_index("datetime")
    a, b = st.columns(2)
    a.caption("Risk (last 72 simulated hours)")
    a.line_chart(d["risk"])
    b.caption("Sensors relative to their window mean")
    b.line_chart(d[["volt", "rotate", "pressure", "vibration"]] /
                 d[["volt", "rotate", "pressure", "vibration"]].mean())


# ------------------------------------------------------ test-period tab
@st.cache_data
def load_test():
    df = pd.read_parquet(PRED)
    return df[df["split"] == "test"].copy()


@st.cache_data(ttl=600)
def failures_by_model():
    return q("""
        WITH ev AS (SELECT DISTINCT machine_id, datetime FROM failures)
        SELECT m.model, COUNT(DISTINCT m.machine_id) AS machines, COUNT(ev.datetime) AS failure_events
        FROM machines m LEFT JOIN ev USING (machine_id) GROUP BY m.model ORDER BY m.model""")


@st.cache_data(ttl=600)
def mtbf_by_model():
    return q("""
        WITH ev AS (SELECT DISTINCT machine_id, datetime FROM failures),
        gaps AS (SELECT machine_id, EXTRACT(EPOCH FROM (datetime - LAG(datetime) OVER
                 (PARTITION BY machine_id ORDER BY datetime))) / 3600.0 AS gap FROM ev)
        SELECT m.model, ROUND(AVG(g.gap)::numeric, 1)::float AS mtbf_hours
        FROM gaps g JOIN machines m USING (machine_id) WHERE g.gap IS NOT NULL
        GROUP BY m.model ORDER BY m.model""")


@st.cache_data(ttl=600)
def monthly_failures():
    return q("SELECT DATE_TRUNC('month', datetime)::date AS month, failure, COUNT(*) AS n "
             "FROM failures GROUP BY 1, 2 ORDER BY 1, 2")


def test_view():
    df = load_test()
    thr = float(df["thr"].iloc[0])
    alert = df["p"] >= thr
    y = df["target"] == 1
    tp, fp, fn = int((alert & y).sum()), int((alert & ~y).sum()), int((~alert & y).sum())
    st.subheader("Model performance (test period, Nov-Dec, threshold %.2f)" % thr)
    c = st.columns(5)
    c[0].metric("Row precision", f"{tp / max(tp + fp, 1):.3f}")
    c[1].metric("Row recall", f"{tp / max(tp + fn, 1):.3f}")
    c[2].metric("False-alarm rows", fp)
    c[3].metric("Missed rows", fn)
    c[4].metric("Test rows", len(df))
    st.info("Event level (from src.models.threshold_analysis): 114 / 114 failure events detected, "
            "3 false-alarm episodes vs 239 for the rule baseline. Synthetic data: real factories "
            "will not be this clean.")

    st.subheader("Highest-risk machines (test period)")
    top = (df.assign(alert=alert).groupby("machine_id")
             .agg(max_risk=("p", "max"), alert_hours=("alert", "sum")).reset_index()
             .sort_values(["alert_hours", "max_risk"], ascending=False).head(10))
    st.dataframe(top, hide_index=True, use_container_width=True)

    l, r = st.columns(2)
    with l:
        st.subheader("Failures by month and component")
        mf = monthly_failures().pivot(index="month", columns="failure", values="n").fillna(0)
        st.bar_chart(mf)
    with r:
        st.subheader("MTBF by machine model (hours)")
        st.bar_chart(mtbf_by_model().set_index("model")["mtbf_hours"])
        st.caption("Mean gap between consecutive failures; machines with <2 failures excluded.")

    st.subheader("Estimated downtime cost (ASSUMPTIONS, adjust below)")
    a, b = st.columns(2)
    hours = a.slider("Downtime hours per failure", 1, 24, 8)
    rate = b.slider("Cost per downtime hour", 500, 5000, 2000, step=100)
    fm = failures_by_model()
    fm["est_cost"] = fm["failure_events"] * hours * rate
    fm["cost_per_machine"] = (fm["est_cost"] / fm["machines"]).round(0)
    st.dataframe(fm, hide_index=True, use_container_width=True)

    st.subheader("Machine explorer (test period)")
    m = st.selectbox("Machine", list(range(1, 101)), key="test_m")
    t0, t1 = df["datetime"].min(), df["datetime"].max()
    tele = q("SELECT datetime, volt, rotate, pressure, vibration FROM telemetry "
             "WHERE machine_id=:m AND datetime BETWEEN :a AND :b ORDER BY datetime",
             m=int(m), a=t0.to_pydatetime(), b=t1.to_pydatetime())
    risk = df[df["machine_id"] == m][["datetime", "p"]]
    d = tele.merge(risk, on="datetime", how="left").set_index("datetime")
    fails = q("SELECT datetime, failure FROM failures WHERE machine_id=:m AND datetime BETWEEN :a AND :b "
              "ORDER BY datetime", m=int(m), a=t0.to_pydatetime(), b=t1.to_pydatetime())
    x, y2 = st.columns(2)
    x.caption("Risk score")
    x.line_chart(d["p"])
    y2.caption("Sensors relative to their mean")
    sens = ["volt", "rotate", "pressure", "vibration"]
    y2.line_chart(d[sens] / d[sens].mean())
    st.caption("Failures of this machine in the test period")
    st.dataframe(fails, hide_index=True, use_container_width=True)


tab1, tab2 = st.tabs(["Live stream", "Test-period analysis"])
with tab1:
    live_view()
with tab2:
    test_view()