"""Project overview dashboard page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from masr.config_parser import pick_default_param_keys
from masr.filters import collect_dynamic_keys, collect_values
from masr.storage import AblationStore
from utils.display import flatten_experiments


def render_overview(store: AblationStore, project_id: str) -> None:
    experiments = store.list_experiments(project_id)
    datasets = collect_values(experiments, "dataset")
    models = collect_values(experiments, "model")
    param_keys = pick_default_param_keys(collect_dynamic_keys(experiments, "params"))
    metric_keys = collect_dynamic_keys(experiments, "metrics")

    cols = st.columns(4)
    cols[0].metric("实验数", len(experiments))
    cols[1].metric("数据集", len(datasets))
    cols[2].metric("模型", len(models))
    cols[3].metric("指标", len(metric_keys))

    st.subheader("最近实验")
    if experiments:
        st.dataframe(
            pd.DataFrame(flatten_experiments(experiments[:10], param_keys, metric_keys)),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("暂无实验。")

    st.subheader("数据位置")
    project_dir = store.project_dir(project_id).resolve()
    st.markdown(f"<div class='masr-path'>{project_dir}</div>", unsafe_allow_html=True)
