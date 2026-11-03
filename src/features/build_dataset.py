"""Build the model-ready table (features + target) and save it as parquet."""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.features.engineering import ERROR_IDS, build_features, feature_columns
from src.features.target import HORIZON_HOURS, make_target

load_dotenv()
ROOT = Path(__file__).resolve().parents[2]
OUT_FILE = ROOT / "data" / "processed" / "features.parquet"


def get_engine():
    url = (f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
           f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
    return create_engine(url)


def main() -> None:
    engine = get_engine()
    tele = pd.read_sql("SELECT * FROM telemetry", engine)
    errors = pd.read_sql("SELECT machine_id, datetime, error_id FROM errors", engine)
    maint = pd.read_sql("SELECT machine_id, datetime, comp FROM maintenance", engine)
    fails = pd.read_sql("SELECT machine_id, datetime FROM failures", engine)
    machines = pd.read_sql("SELECT * FROM machines", engine)

    feats = build_features(tele, errors, maint, machines)

    landed = int(feats[[f"{e}_now" for e in ERROR_IDS]].sum().sum())
    print(f"Errors in table: {len(errors)} | landed on telemetry grid: {landed}")

    feats["target"] = make_target(feats[["machine_id", "datetime"]], fails)

    # last 24h: label window is incomplete, so these rows cannot be used
    cutoff = feats["datetime"].max() - pd.Timedelta(hours=HORIZON_HOURS)
    feats = feats[feats["datetime"] <= cutoff].reset_index(drop=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT_FILE, index=False)

    n_pos = int(feats["target"].sum())
    print(f"Rows: {len(feats)} | features: {len(feature_columns(feats))}")
    print(f"Positives: {n_pos} ({100 * n_pos / len(feats):.2f}%)")
    print(f"Time range: {feats['datetime'].min()} -> {feats['datetime'].max()}")
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()