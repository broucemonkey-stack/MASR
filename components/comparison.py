"""Comparison display components: table, image grid, config viewer, curves."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd
import streamlit as st

from masr.log_parser import extract_epoch_curves
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


def render_curve_view(store: AblationStore, project_id: str, experiments: list[Experiment]) -> None:
    """Show training curves for experiments that have log files."""
    candidates = [e for e in experiments if e.log_file]
    if not candidates:
        return

    st.subheader("训练曲线")

    if len(candidates) == 1:
        selected_exp = candidates[0]
    else:
        labels = {e.id: e.name for e in candidates}
        selected_id = st.selectbox(
            "选择实验",
            [e.id for e in candidates],
            format_func=lambda eid: labels.get(eid, eid),
            key="curve_select_exp",
        )
        selected_exp = next(e for e in candidates if e.id == selected_id)

    content = store.read_log_text(project_id, selected_exp)
    if content is None:
        st.error(f"没有找到「{selected_exp.name}」的训练日志。")
        return

    curves = extract_epoch_curves(content)
    if not curves or len(curves) <= 1:
        st.warning("未能从日志中提取到 epoch 级别的指标。")
        return

    metric_keys = [k for k in curves.keys() if k != "epoch"]
    # 自动勾选验证集指标（accuracy, f1 等），训练集 loss 也默认勾选
    default_keywords = ["accuracy", "f1", "loss"]
    default_metrics = [k for k in metric_keys if any(kw in k.lower() for kw in default_keywords)]
    if not default_metrics:
        default_metrics = list(metric_keys)
    selected = st.multiselect(
        "选择要显示的指标",
        options=metric_keys,
        default=default_metrics,
        key="curve_compare_metrics",
    )
    if not selected:
        return

    import plotly.graph_objects as go

    fig = go.Figure()
    for key in selected:
        fig.add_trace(go.Scatter(
            x=curves["epoch"],
            y=curves[key],
            mode="lines+markers",
            name=key,
            hovertemplate="%{y:.4f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="指标值",
        hovermode="x unified",
        hoverdistance=20,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=10, b=30),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)
