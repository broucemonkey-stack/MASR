"""Sidebar project selector and empty-state UI components."""

from __future__ import annotations

from typing import Any

import streamlit as st

from masr.storage import AblationStore


def render_project_selector(store: AblationStore, projects: list) -> Any:
    with st.sidebar.expander("新建项目", expanded=not projects):
        with st.form("project_create_form", clear_on_submit=True):
            name = st.text_input("项目名称")
            description = st.text_area("项目描述", height=90)
            submitted = st.form_submit_button("创建项目", use_container_width=True)
        if submitted:
            if not name.strip():
                st.sidebar.error("请填写项目名称。")
            else:
                store.create_project(name, description)
                st.sidebar.success("项目已创建。")
                st.rerun()

    if not projects:
        return None

    labels = {
        project.id: f"{project.name} · {len(store.list_experiments(project.id))} 个实验"
        for project in projects
    }
    selected_id = st.sidebar.selectbox(
        "项目",
        [project.id for project in projects],
        format_func=lambda project_id: labels.get(project_id, project_id),
    )
    return store.get_project(selected_id)


def render_empty_state(store: AblationStore) -> None:
    st.title("消融实验结果管理")
    st.info("暂无项目。")
    with st.form("empty_project_create_form"):
        name = st.text_input("项目名称", placeholder="例如：SAM 消融实验")
        description = st.text_area("项目描述", height=100)
        submitted = st.form_submit_button("创建项目")
    if submitted:
        if not name.strip():
            st.error("请填写项目名称。")
        else:
            store.create_project(name, description)
            st.success("项目已创建。")
            st.rerun()
