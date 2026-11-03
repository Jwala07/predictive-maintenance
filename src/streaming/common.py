"""Shared config, DB engine and logging for the streaming pipeline."""
import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
BROKER = os.getenv("KAFKA_BROKER", "127.0.0.1:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "telemetry")


def get_engine():
    url = (f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
           f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
    return create_engine(url, pool_pre_ping=True)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s")
    return logging.getLogger(name)