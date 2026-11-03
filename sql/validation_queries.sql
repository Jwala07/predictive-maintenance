-- 1) Row counts of every table
SELECT 'machines' AS tbl, COUNT(*) AS n FROM machines
UNION ALL SELECT 'telemetry', COUNT(*) FROM telemetry
UNION ALL SELECT 'errors', COUNT(*) FROM errors
UNION ALL SELECT 'maintenance', COUNT(*) FROM maintenance
UNION ALL SELECT 'failures', COUNT(*) FROM failures;

-- 2) Nulls in telemetry (all should be 0)
SELECT
  COUNT(*) FILTER (WHERE volt IS NULL)      AS null_volt,
  COUNT(*) FILTER (WHERE rotate IS NULL)    AS null_rotate,
  COUNT(*) FILTER (WHERE pressure IS NULL)  AS null_pressure,
  COUNT(*) FILTER (WHERE vibration IS NULL) AS null_vibration
FROM telemetry;

-- 3) Timestamp continuity: every machine should have exactly 8761 hourly rows
--    (empty result = good)
SELECT machine_id, COUNT(*) AS hours, MIN(datetime) AS first_ts, MAX(datetime) AS last_ts
FROM telemetry
GROUP BY machine_id
HAVING COUNT(*) <> 8761
LIMIT 10;

-- 4) Duplicate failure records (empty result = good)
SELECT machine_id, datetime, failure, COUNT(*)
FROM failures
GROUP BY machine_id, datetime, failure
HAVING COUNT(*) > 1;

-- 5) Date range of the data
SELECT MIN(datetime) AS first_ts, MAX(datetime) AS last_ts FROM telemetry;

-- 6) Failures per component
SELECT failure, COUNT(*) AS n FROM failures GROUP BY failure ORDER BY failure;