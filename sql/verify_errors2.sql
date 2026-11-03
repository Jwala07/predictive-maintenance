-- Which error types appear exactly 24h before a failure, per component?
SELECT f.failure AS comp, e.error_id, COUNT(*) AS n
FROM failures f
JOIN errors e
  ON e.machine_id = f.machine_id
 AND e.datetime   = f.datetime - INTERVAL '24 hours'
GROUP BY 1, 2
ORDER BY 1, 2;

-- Background errors: type mix of errors NOT exactly 24h before a failure
SELECT e.error_id, COUNT(*) AS n
FROM errors e
WHERE NOT EXISTS (
    SELECT 1 FROM failures f
    WHERE f.machine_id = e.machine_id
      AND f.datetime = e.datetime + INTERVAL '24 hours')
GROUP BY 1 ORDER BY 1;