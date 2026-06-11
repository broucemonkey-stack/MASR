"""Experiment filtering and comparison page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from masr.config_parser import pick_default_param_keys
from masr.filters import (
    ExperimentFilters,
    collect_dynamic_keys,
    collect_tags,
    collect_values,
    filter_experiments,
    metric_range,
)
from masr.models import Experiment
from masr.storage import AblationStore
from components.comparison import render_comparison_table, render_config_view, render_image_grid, render_log_view
from utils.display import experiment_label, flatten_experiments

# Session-state keys used by the filter widgets — cleared on reset.
_FILTER_KEYS = [
    "filter_search",
    "filter_datasets",
    "filter_models",
    "filter_strategies",
    "filter_tags",
    "filter_param_key",
    "filter_param_value",
    "filter_metric_key",
    "filter_metric_range",
]


def render_compare_page(store: AblationStore, project_id: str) -> None:
    experiments = store.list_experiments(project_id)
    if not experiments:
        st.info("暂无实验。")
        return

    filters = render_filters(experiments)
    filtered = filter_experiments(experiments, filters)
    param_keys = collect_dynamic_keys(filtered, "params")
    metric_keys = collect_dynamic_keys(filtered, "metrics")
    display_param_keys, display_metric_keys = render_display_options(param_keys, metric_keys)

    st.subheader("筛选结果")
    st.caption(f"{len(filtered)} / {len(experiments)} 个实验")
    if filtered:
        st.dataframe(
            pd.DataFrame(flatten_experiments(filtered, display_param_keys, display_metric_keys)),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("没有匹配的实验。")
        return

    labels = {experiment.id: experiment_label(experiment) for experiment in filtered}
    default_selection = [experiment.id for experiment in filtered[: min(4, len(filtered))]]
    selected_ids = st.multiselect(
        "选择对比实验",
        [experiment.id for experiment in filtered],
        default=default_selection,
        format_func=lambda experiment_id: labels.get(experiment_id, experiment_id),
    )
    selected = [experiment for experiment in filtered if experiment.id in selected_ids]
    if not selected:
        st.info("请选择实验。")
        return

    render_comparison_table(selected, display_param_keys, display_metric_keys)
    render_image_grid(store, project_id, selected)
    render_config_view(store, project_id, selected)
    render_log_view(store, project_id, selected)


def render_filters(experiments: list[Experiment]) -> ExperimentFilters:
    with st.expander("筛选条件", expanded=True):
        # --- Reset button ---
        reset_cols = st.columns([1, 5])
        if reset_cols[0].button("重置筛选", use_container_width=True, key="reset_filters_btn"):
            for key in _FILTER_KEYS:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        top = st.columns([2, 1, 1, 1])
        search = top[0].text_input("搜索", key="filter_search")
        datasets = set(top[1].multiselect(
            "数据集", collect_values(experiments, "dataset"), key="filter_datasets",
        ))
        models = set(top[2].multiselect(
            "模型", collect_values(experiments, "model"), key="filter_models",
        ))
        strategies = set(top[3].multiselect(
            "策略", collect_values(experiments, "strategy"), key="filter_strategies",
        ))

        bottom = st.columns([1, 1, 1, 1])
        tags = set(bottom[0].multiselect(
            "标签", collect_tags(experiments), key="filter_tags",
        ))
        param_keys = collect_dynamic_keys(experiments, "params")
        param_key = bottom[1].selectbox(
            "参数", [""] + param_keys,
            format_func=lambda value: value or "不限",
            key="filter_param_key",
        )
        param_value = bottom[2].text_input("参数值包含", key="filter_param_value")

        metric_keys = collect_dynamic_keys(experiments, "metrics")
        metric_key = bottom[3].selectbox(
            "指标", [""] + metric_keys,
            format_func=lambda value: value or "不限",
            key="filter_metric_key",
        )

        metric_min = None
        metric_max = None
        if metric_key:
            bounds = metric_range(experiments, metric_key)
            if bounds is not None:
                min_value, max_value = bounds
                if min_value == max_value:
                    st.caption(f"{metric_key}: {min_value:g}")
                    metric_min = min_value
                    metric_max = max_value
                else:
                    metric_min, metric_max = st.slider(
                        "指标范围",
                        min_value=float(min_value),
                        max_value=float(max_value),
                        value=(float(min_value), float(max_value)),
                        key="filter_metric_range",
                    )

    return ExperimentFilters(
        search=search.strip(),
        datasets=datasets,
        models=models,
        strategies=strategies,
        tags=tags,
        param_key=param_key,
        param_value=param_value.strip(),
        metric_key=metric_key,
        metric_min=metric_min,
        metric_max=metric_max,
    )


def render_display_options(param_keys: list[str], metric_keys: list[str]) -> tuple[list[str], list[str]]:
    with st.expander("显示列", expanded=False):
        display_param_keys = st.multiselect(
            "参数列",
            param_keys,
            default=pick_default_param_keys(param_keys),
        )
        display_metric_keys = st.multiselect(
            "指标列",
            metric_keys,
            default=metric_keys,
        )
    return display_param_keys, display_metric_keys
