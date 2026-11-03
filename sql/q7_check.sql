-- How many failures have a maintenance record at the exact same timestamp and component?
SELECT COUNT(*) AS same_timestamp_records
FROM failures f
JOIN maintenance m
  ON m.machine_id = f.machine_id
 AND m.datetime   = f.datetime
 AND m.comp       = f.failure;

-- Corrected Q7: strictly BEFORE the failure
WITH lm AS (
    SELECT f.machine_id, f.datetime AS fail_ts, f.failure AS comp,
           (SELECT MAX(mt.datetime) FROM maintenance mt
             WHERE mt.machine_id = f.machine_id
               AND mt.comp = f.failure
               AND mt.datetime < f.datetime) AS prev_maint_ts
    FROM failures f
)
SELECT comp,
       COUNT(*)           AS failures,
       COUNT(prev_maint_ts) AS with_prior_maint,
       ROUND(AVG(EXTRACT(EPOCH FROM (fail_ts - prev_maint_ts)) / 86400.0)::numeric, 1)
                          AS avg_days_since_prev_maint
FROM lm
GROUP BY comp
ORDER BY comp;