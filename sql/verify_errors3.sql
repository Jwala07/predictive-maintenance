-- Group errors by (machine, timestamp) and see which combinations precede a failure
WITH ev AS (
    SELECT machine_id, datetime,
           string_agg(error_id, '+' ORDER BY error_id) AS pattern
    FROM errors
    GROUP BY machine_id, datetime
),
flagged AS (
    SELECT pattern,
           EXISTS (SELECT 1 FROM failures f
                   WHERE f.machine_id = ev.machine_id
                     AND f.datetime = ev.datetime + INTERVAL '24 hours') AS before_failure
    FROM ev
)
SELECT pattern,
       COUNT(*) AS events,
       COUNT(*) FILTER (WHERE before_failure) AS followed_by_failure_24h,
       ROUND(100.0 * COUNT(*) FILTER (WHERE before_failure) / COUNT(*), 1) AS pct
FROM flagged
GROUP BY pattern
ORDER BY events DESC;