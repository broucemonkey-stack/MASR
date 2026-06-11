"""Comparison display components: table, image grid, config viewer."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd
import streamlit as st

from masr.models import Experiment
from masr.storage import AblationStore
from utils.display import flatten_experiments


def render_comparison_table(
    experiments: list[Experiment],
    param_keys: list[str],
    metric_keys: list[str],
) -> None:
    st.subheader("参数与指标对比")
    st.dataframe(
        pd.DataFrame(flatten_experiments(experiments, param_keys, metric_keys)),
        use_container_width=True,
        hide_index=True,
    )


def render_image_grid(store: AblationStore, project_id: str, experiments: list[Experiment]) -> None:
    st.subheader("结果图片")
    images_by_label: dict[str, list[tuple[Experiment, Any]]] = defaultdict(list)
    for experiment in experiments:
        for image in experiment.images:
            images_by_label[image.label or "result"].append((experiment, image))

    if not images_by_label:
        st.info("所选实验暂无图片。")
        return

    tabs = st.tabs(sorted(images_by_label.keys()))
    for tab, label in zip(tabs, sorted(images_by_label.keys())):
        with tab:
            records = images_by_label[label]
            for row_start in range(0, len(records), 3):
                cols = st.columns(3)
                for column, (experiment, image) in zip(cols, records[row_start : row_start + 3]):
                    with column:
                        st.markdown(f"**{experiment.name}**")
                        path = store.image_path(project_id, experiment.id, image)
                        thumb = store.thumbnail_path(project_id, experiment.id, image)
                        if thumb.exists():
                            st.image(str(thumb), caption=image.note or image.filename, width=200)
                            with st.expander("查看原图"):
                                if path.exists():
                                    st.image(str(path), use_column_width=True)
                        elif path.exists():
                            st.image(str(path), caption=image.note or image.filename, width=200)
                        else:
                            st.warning(f"缺失图片：{image.filename}")


def render_config_view(store: AblationStore, project_id: str, experiments: list[Experiment]) -> None:
    st.subheader("配置文件")
    for experiment in experiments:
        title = experiment.name
        if experiment.config_original_name:
            title = f"{title} · {experiment.config_original_name}"
        with st.expander(title):
            content = store.read_config_text(project_id, experiment)
            if content is None:
                st.info("暂无配置文件。")
            else:
                st.code(content, language="python")


def render_log_view(store: AblationStore, project_id: str, experiments: list[Experiment]) -> None:
    st.subheader("训练日志")
    for experiment in experiments:
        title = experiment.name
        if experiment.log_original_name:
            title = f"{title} · {experiment.log_original_name}"
        with st.expander(title):
            content = store.read_log_text(project_id, experiment)
            if content is None:
                st.info("暂无训练日志。")
            else:
                st.code(content, language="text")
