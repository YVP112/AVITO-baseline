from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


st.set_page_config(page_title="Hack Dashboard", layout="wide")
st.title("Baseline Comparison Dashboard")

comparison_df = load_csv(OUTPUTS_DIR / "baseline_comparison.csv")
val_predictions_df = load_csv(OUTPUTS_DIR / "val_predictions.csv")

if comparison_df.empty:
    st.warning("Файл outputs/baseline_comparison.csv пока не найден. Сначала запустите run_compare_baselines.py")
else:
    st.subheader("Сравнение baseline'ов")
    st.dataframe(comparison_df, use_container_width=True)

    metric_names = [
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "should_split_accuracy",
    ]

    selected_metric = st.selectbox("Метрика", metric_names, index=2)
    metric_df = comparison_df[["baseline", selected_metric]].set_index("baseline")
    st.bar_chart(metric_df)

if val_predictions_df.empty:
    st.info("Файл outputs/val_predictions.csv пока не найден. Сначала запустите run_baseline.py")
else:
    st.subheader("Просмотр предсказаний на validation")
    visible_columns = [
        "itemId",
        "sourceMcTitle",
        "description",
        "targetSplitMcIds",
        "shouldSplit",
        "predShouldSplit",
        "predDrafts",
        "caseType",
    ]
    existing_columns = [column for column in visible_columns if column in val_predictions_df.columns]
    st.dataframe(val_predictions_df[existing_columns], use_container_width=True, height=500)
