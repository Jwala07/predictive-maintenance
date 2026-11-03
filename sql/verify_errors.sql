-- 1) How many failure events have an error exactly 24h before?
WITH fe AS (SELECT DISTINCT machine_id, datetime AS fail_ts FROM failures)
SELECT COUNT(*) AS failure_events,
       COUNT(*) FILTER (WHERE EXISTS (
           SELECT 1 FROM errors e
           WHERE e.machine_id = fe.machine_id
             AND e.datetime = fe.fail_ts - INTERVAL '24 hours')) AS with_error_at_24h
FROM fe;

-- 2) How many errors are exactly 24h before some failure?
SELECT COUNT(*) AS total_errors,
       COUNT(*) FILTER (WHERE EXISTS (
           SELECT 1 FROM failures f
           WHERE f.machine_id = e.machine_id
             AND f.datetime = e.datetime + INTERVAL '24 hours')) AS errors_24h_before_a_failure
FROM errors e;