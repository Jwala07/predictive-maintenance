DROP TABLE IF EXISTS telemetry, errors, maintenance, failures, machines CASCADE;

CREATE TABLE machines (
    machine_id INT PRIMARY KEY,
    model      VARCHAR(10) NOT NULL,
    age        INT NOT NULL
);

CREATE TABLE telemetry (
    machine_id INT NOT NULL REFERENCES machines(machine_id),
    datetime   TIMESTAMP NOT NULL,
    volt       DOUBLE PRECISION,
    rotate     DOUBLE PRECISION,
    pressure   DOUBLE PRECISION,
    vibration  DOUBLE PRECISION,
    PRIMARY KEY (machine_id, datetime)
);

CREATE TABLE errors (
    id         BIGSERIAL PRIMARY KEY,
    machine_id INT NOT NULL REFERENCES machines(machine_id),
    datetime   TIMESTAMP NOT NULL,
    error_id   VARCHAR(10) NOT NULL
);

CREATE TABLE maintenance (
    id         BIGSERIAL PRIMARY KEY,
    machine_id INT NOT NULL REFERENCES machines(machine_id),
    datetime   TIMESTAMP NOT NULL,
    comp       VARCHAR(10) NOT NULL
);

CREATE TABLE failures (
    id         BIGSERIAL PRIMARY KEY,
    machine_id INT NOT NULL REFERENCES machines(machine_id),
    datetime   TIMESTAMP NOT NULL,
    failure    VARCHAR(10) NOT NULL
);

CREATE INDEX idx_errors_machine_time      ON errors (machine_id, datetime);
CREATE INDEX idx_maintenance_machine_time ON maintenance (machine_id, datetime);
CREATE INDEX idx_failures_machine_time    ON failures (machine_id, datetime);