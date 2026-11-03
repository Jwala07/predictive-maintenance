"""Compare streaming predictions (DB) with batch predictions (parquet) on the same rows."""
import pandas as pd

from src.models.train import ROOT
from src.streaming.common import get_engine

live = pd.read_sql("SELECT machine_id, datetime, risk, alert FROM predictions", get_engine())
batch = pd.read_parquet(ROOT / "data" / "processed" / "predictions.parquet")
batch = batch[batch["split"] == "test"]
j = live.merge(batch, on=["machine_id", "datetime"], how="inner")
print(f"live rows: {len(live)} | matched with batch: {len(j)}")
print(f"max |risk - p|: {(j['risk'] - j['p']).abs().max():.2e}")
print(f"alert agreement: {(j['alert'] == (j['p'] >= j['thr'])).mean():.4f}")
print(f"alerts raised: {int(live['alert'].sum())} | true-positive rows among them: "
      f"{int(((j['alert']) & (j['target'] == 1)).sum())}")