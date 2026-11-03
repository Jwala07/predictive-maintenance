"""Load the 5 Azure PdM CSV files into PostgreSQL."""
import os
from io import StringIO
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SCHEMA_FILE = ROOT / "sql" / "schema.sql"

# table -> (csv file, column rename map). Order matters: machines first (foreign key).
TABLES = {
    "machines": (
        "PdM_machines.csv",
        {"machineID": "machine_id", "model": "model", "age": "age"},
    ),
    "telemetry": (
        "PdM_telemetry.csv",
        {"machineID": "machine_id", "datetime": "datetime", "volt": "volt",
         "rotate": "rotate", "pressure": "pressure", "vibration": "vibration"},
    ),
    "errors": (
        "PdM_errors.csv",
        {"machineID": "machine_id", "datetime": "datetime", "errorID": "error_id"},
    ),
    "maintenance": (
        "PdM_maint.csv",
        {"machineID": "machine_id", "datetime": "datetime", "comp": "comp"},
    ),
    "failures": (
        "PdM_failures.csv",
        {"machineID": "machine_id", "datetime": "datetime", "failure": "failure"},
    ),
}


def get_connection():
    """Create a PostgreSQL connection using values from the .env file."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def create_schema(cur) -> None:
    """Run schema.sql to (re)create all tables."""
    cur.execute(SCHEMA_FILE.read_text(encoding="utf-8"))
    print("Schema created.")


def copy_dataframe(cur, table: str, df: pd.DataFrame) -> None:
    """Bulk-load a DataFrame into a table using PostgreSQL COPY."""
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    columns = ", ".join(df.columns)
    cur.copy_expert(f"COPY {table} ({columns}) FROM STDIN WITH CSV", buffer)


def main() -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            create_schema(cur)
            for table, (filename, rename_map) in TABLES.items():
                df = pd.read_csv(RAW_DIR / filename)
                df = df.rename(columns=rename_map)[list(rename_map.values())]
                copy_dataframe(cur, table, df)
                print(f"Loaded {len(df):>8} rows into {table}")
        conn.commit()
        print("Done. All data committed.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()