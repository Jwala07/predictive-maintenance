CREATE TABLE IF NOT EXISTS live_telemetry (
    machine_id INT NOT NULL REFERENCES machines(machine_id),
    datetime   TIMESTAMP NOT NULL,
    volt DOUBLE PRECISION, rotate DOUBLE PRECISION,
    pressure DOUBLE PRECISION, vibration DOUBLE PRECISION,
    PRIMARY KEY (machine_id, datetime)
);

CREATE TABLE IF NOT EXISTS predictions (
    machine_id INT NOT NULL REFERENCES machines(machine_id),
    datetime   TIMESTAMP NOT NULL,
    risk       DOUBLE PRECISION NOT NULL,
    alert      BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (machine_id, datetime)
);

CREATE TABLE IF NOT EXISTS alerts (
    id         BIGSERIAL PRIMARY KEY,
    machine_id INT NOT NULL REFERENCES machines(machine_id),
    datetime   TIMESTAMP NOT NULL,
    risk       DOUBLE PRECISION NOT NULL,
    threshold  DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (machine_id, datetime)
);