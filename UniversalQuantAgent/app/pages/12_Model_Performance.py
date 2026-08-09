from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
_loaded_app = sys.modules.get("app")
if _loaded_app is not None and not hasattr(_loaded_app, "__path__"):
    del sys.modules["app"]
"""Display accuracy metrics from settled, locally recorded predictions."""
import pandas as pd
import streamlit as st
from app.page_runtime import apply_global_theme, records_table
from modules.model_performance import get_model_performance_summary

apply_global_theme()

st.title("Model Performance")
summary = get_model_performance_summary()
if summary["note"]: st.info(summary["note"])
st.metric("Settled predictions", summary["sample_size"])
if summary["categories"]:
    st.dataframe(records_table(summary["categories"]), use_container_width=True, hide_index=True)
    table = pd.DataFrame(summary["categories"]).set_index("category")
    st.bar_chart(table[["mae","bias","hit_rate"]])
if summary["rolling_performance"]:
    rolling = pd.DataFrame(summary["rolling_performance"]).set_index("date")
    st.line_chart(rolling)
