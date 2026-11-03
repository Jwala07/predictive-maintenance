-- =========================================================
-- Q1: Failures by component
-- =========================================================
SELECT failure AS component,
       COUNT(*) AS failure_count,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_total
FROM failures
GROUP BY failure
ORDER BY failure_count DESC;

-- =========================================================
-- Q2: Failures by machine model (normalised per machine)
-- =========================================================
SELECT m.model,
       COUNT(DISTINCT m.machine_id)                              AS machines,
       COUNT(f.id)                                               AS failures,
       ROUND(COUNT(f.id)::numeric / COUNT(DISTINCT m.machine_id), 2) AS failures_per_machine
FROM machines m
LEFT JOIN failures f ON f.machine_id = m.machine_id
GROUP BY m.model
ORDER BY failures_per_machine DESC;

-- =========================================================
-- Q3: Failures by machine age group
-- =========================================================
SELECT CASE
         WHEN m.age < 5  THEN '0-4 yrs'
         WHEN m.age < 10 THEN '5-9 yrs'
         WHEN m.age < 15 THEN '10-14 yrs'
         ELSE '15+ yrs'
       END AS age_group,
       COUNT(DISTINCT m.machine_id) AS machines,
       COUNT(f.id)                  AS failures,
       ROUND(COUNT(f.id)::numeric / COUNT(DISTINCT m.machine_id), 2) AS failures_per_machine
FROM machines m
LEFT JOIN failures f ON f.machine_id = m.machine_id
GROUP BY 1
ORDER BY MIN(m.age);

-- =========================================================
-- Q4: MTBF per machine (mean time between failures, hours)
-- Failure events are de-duplicated to one per machine per timestamp.
-- =========================================================
WITH ev AS (
    SELECT DISTINCT machine_id, datetime FROM failures
),
gaps AS (
    SELECT machine_id,
           EXTRACT(EPOCH FROM (datetime - LAG(datetime) OVER (
               PARTITION BY machine_id ORDER BY datetime))) / 3600.0 AS gap_hours
    FROM ev
)
SELECT machine_id,
       COUNT(gap_hours)            AS gaps,
       ROUND(AVG(gap_hours)::numeric, 1) AS mtbf_hours
FROM gaps
WHERE gap_hours IS NOT NULL
GROUP BY machine_id
ORDER BY mtbf_hours
LIMIT 15;

-- =========================================================
-- Q5: MTBF per machine model
-- =========================================================
WITH ev AS (
    SELECT DISTINCT machine_id, datetime FROM failures
),
gaps AS (
    SELECT machine_id,
           EXTRACT(EPOCH FROM (datetime - LAG(datetime) OVER (
               PARTITION BY machine_id ORDER BY datetime))) / 3600.0 AS gap_hours
    FROM ev
)
SELECT m.model,
       ROUND(AVG(g.gap_hours)::numeric, 1) AS mtbf_hours,
       ROUND((AVG(g.gap_hours) / 24)::numeric, 1) AS mtbf_days
FROM gaps g
JOIN machines m USING (machine_id)
WHERE g.gap_hours IS NOT NULL
GROUP BY m.model
ORDER BY mtbf_hours;

-- =========================================================
-- Q6: Errors in the 24 / 48 / 72 hours before a failure
-- Compare with the baseline: average errors per machine per 24h.
-- =========================================================
WITH fe AS (
    SELECT DISTINCT machine_id, datetime AS fail_ts FROM failures
)
SELECT
    COUNT(*)                                                              AS failure_events,
    ROUND(AVG((SELECT COUNT(*) FROM errors e
        WHERE e.machine_id = fe.machine_id
          AND e.datetime >  fe.fail_ts - INTERVAL '24 hours'
          AND e.datetime <= fe.fail_ts))::numeric, 2)                     AS avg_errors_24h,
    ROUND(AVG((SELECT COUNT(*) FROM errors e
        WHERE e.machine_id = fe.machine_id
          AND e.datetime >  fe.fail_ts - INTERVAL '48 hours'
          AND e.datetime <= fe.fail_ts))::numeric, 2)                     AS avg_errors_48h,
    ROUND(AVG((SELECT COUNT(*) FROM errors e
        WHERE e.machine_id = fe.machine_id
          AND e.datetime >  fe.fail_ts - INTERVAL '72 hours'
          AND e.datetime <= fe.fail_ts))::numeric, 2)                     AS avg_errors_72h,
    ROUND((SELECT COUNT(*) FROM errors)::numeric
          / (SELECT COUNT(*) FROM machines)
          / (SELECT COUNT(DISTINCT datetime::date) FROM telemetry), 3)    AS baseline_errors_per_machine_per_day
FROM fe;

-- =========================================================
-- Q7: Time since last maintenance of the SAME component before a failure
-- =========================================================
WITH fe AS (
    SELECT machine_id, datetime AS fail_ts, failure AS comp FROM failures
),
last_maint AS (
    SELECT fe.machine_id, fe.fail_ts, fe.comp,
           (SELECT MAX(mt.datetime) FROM maintenance mt
             WHERE mt.machine_id = fe.machine_id
               AND mt.comp = fe.comp
               AND mt.datetime <= fe.fail_ts) AS last_maint_ts
    FROM fe
)
SELECT comp,
       COUNT(*)                                   AS failures,
       COUNT(last_maint_ts)                       AS with_prior_maint,
       ROUND(AVG(EXTRACT(EPOCH FROM (fail_ts - last_maint_ts)) / 86400.0)::numeric, 1)
                                                  AS avg_days_since_maint
FROM last_maint
GROUP BY comp
ORDER BY comp;

-- =========================================================
-- Q8: Failures per month (trend)
-- =========================================================
SELECT DATE_TRUNC('month', datetime)::date AS month,
       COUNT(*) AS failures
FROM failures
GROUP BY 1
ORDER BY 1;

-- =========================================================
-- Q9: Estimated downtime cost (ASSUMPTIONS, change as needed)
--   downtime_hours_per_failure = 8
--   cost_per_downtime_hour     = 2000
-- =========================================================
WITH params AS (
    SELECT 8 AS downtime_hours_per_failure, 2000 AS cost_per_downtime_hour
),
ev AS (
    SELECT DISTINCT machine_id, datetime FROM failures
)
SELECT m.model,
       COUNT(*) AS failure_events,
       COUNT(*) * p.downtime_hours_per_failure                          AS downtime_hours,
       COUNT(*) * p.downtime_hours_per_failure * p.cost_per_downtime_hour AS est_cost
FROM ev
JOIN machines m USING (machine_id)
CROSS JOIN params p
GROUP BY m.model, p.downtime_hours_per_failure, p.cost_per_downtime_hour
ORDER BY est_cost DESC;