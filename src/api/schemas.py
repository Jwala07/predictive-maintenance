"""Pydantic request/response models for the API."""
from datetime import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field

ErrorId = Literal["error1", "error2", "error3", "error4", "error5"]
CompId = Literal["comp1", "comp2", "comp3", "comp4"]


class HourRecord(BaseModel):
    datetime: dt
    volt: float = Field(gt=0, lt=1000)
    rotate: float = Field(gt=0, lt=1000)
    pressure: float = Field(gt=0, lt=1000)
    vibration: float = Field(gt=0, lt=1000)
    errors: list[ErrorId] = []
    maintenance: list[CompId] = []


class PastMaintenance(BaseModel):
    datetime: dt
    comp: CompId


class PredictRequest(BaseModel):
    machine_id: int
    history: list[HourRecord] = Field(min_length=24, max_length=336)
    past_maintenance: list[PastMaintenance] = []


class PredictResponse(BaseModel):
    machine_id: int
    datetime: dt
    risk: float
    alert: bool
    risk_level: Literal["low", "medium", "high"]
    threshold: float
    history_hours: int
    warnings: list[str]