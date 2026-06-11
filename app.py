"""MASR — 消融实验结果管理 (Streamlit entry point)."""

from __future__ import annotations

import streamlit as st

from components.project_selector import render_empty_state, render_project_selector
from masr.storage import AblationStore
from views.compare import render_compare_page
from views.experiment_form import render_experiment_form
from views.maintenance import render_maintenance_page
from views.overview import render_overview

st.set_page_config(
    page_title="MASR 消融实验结果管理",
    page_icon="MASR",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    inject_style()
    store = get_store()
    projects = store.list_projects()

    st.sidebar.title("MASR")
    project = render_project_selector(store, projects)
    page = st.sidebar.radio(
        "页面",
        ["项目概览", "实验录入", "筛选与对比", "数据维护"],
        label_visibility="collapsed",
    )

    if project is None:
        render_empty_state(store)
        return

    st.title("消融实验结果管理")
    st.caption(f"当前项目：{project.name}")

    if page == "项目概览":
        render_overview(store, project.id)
    elif page == "实验录入":
        render_experiment_form(store, project.id)
    elif page == "筛选与对比":
        render_compare_page(store, project.id)
    else:
        render_maintenance_page(store, project.id)


@st.cache_resource
def get_store() -> AblationStore:
    store = AblationStore()
    store.ensure()
    return store


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.85rem 1rem;
        }
        div[data-testid="stExpander"] {
            border-radius: 8px;
        }
        .masr-muted {
            color: #64748b;
            font-size: 0.92rem;
        }
        .masr-path {
            color: #475569;
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
