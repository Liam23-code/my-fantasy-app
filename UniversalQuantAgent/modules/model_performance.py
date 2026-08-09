"""Persist and summarize settled predictions without requiring a database."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from modules.data_quality import coerce_numeric

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "model_performance.csv"

def get_model_performance_summary(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path else DATA_FILE
    if not source.exists():
        return {"categories":[], "rolling_performance":[], "sample_size":0,
                "note":"No settled predictions yet. Add results to data/model_performance.csv to begin tracking."}
    data = pd.read_csv(source)
    required = {"category","projection","actual","sportsbook_line","date"}
    missing = required - set(data.columns)
    if missing: raise ValueError("Performance file is missing: " + ", ".join(sorted(missing)))
    data = coerce_numeric(data, ("projection","actual","sportsbook_line"))
    if "game_date" in data and "date" not in data:
        data = data.rename(columns={"game_date":"date"})
    data = data.dropna(subset=list(required - {"category","date"}))
    data["error"] = data["projection"] - data["actual"]
    data["hit"] = ((data["projection"]-data["sportsbook_line"])*(data["actual"]-data["sportsbook_line"]) > 0).astype(float)
    categories = []
    for category, group in data.groupby("category"):
        categories.append({"category":category, "mae":round(group["error"].abs().mean(),2),
                           "bias":round(group["error"].mean(),2), "hit_rate":round(group["hit"].mean()*100,1),
                           "sample_size":len(group)})
    ordered = data.sort_values("date").copy()
    ordered["rolling_mae"] = ordered["error"].abs().rolling(20,min_periods=1).mean()
    ordered["rolling_hit_rate"] = ordered["hit"].rolling(20,min_periods=1).mean()*100
    rolling = ordered[["date","rolling_mae","rolling_hit_rate"]].round(2).to_dict("records")
    return {"categories":categories, "rolling_performance":rolling, "sample_size":len(data), "note":""}
