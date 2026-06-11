"""Data maintenance page: project update, experiment edit, and deletion."""

from __future__ import annotations

import streamlit as st

from masr.config_parser import extract_params_from_config_text
from masr.models import Experiment
from masr.parsing import format_key_value_lines, parse_key_value_lines, parse_metrics_text, parse_tags
from masr.storage import AblationStore
from utils.display import experiment_label


def render_maintenance_page(store: AblationStore, project_id: str) -> None:
    project = store.get_project(project_id)
    experiments = store.list_experiments(project_id)
    if project is None:
        st.error("项目不存在。")
        return

    st.subheader("项目信息")
    with st.form("project_update_form"):
        name = st.text_input("项目名称", value=project.name)
        description = st.text_area("项目描述", value=project.description, height=100)
        submitted = st.form_submit_button("更新项目")
    if submitted:
        if not name.strip():
            st.error("请填写项目名称。")
        else:
            project.name = name.strip()
            project.description = description.strip()
            store.update_project(project)
            st.success("项目已更新。")
            st.rerun()

    st.subheader("实验维护")
    if not experiments:
        st.info("暂无实验。")
    for experiment in experiments:
        with st.expander(experiment_label(experiment)):
            render_experiment_editor(store, project_id, experiment)

    st.subheader("项目删除")
    confirm_project_name = st.text_input("输入项目名称")
    if st.button("删除当前项目", disabled=confirm_project_name != project.name):
        store.delete_project(project_id)
        st.success("项目已删除。")
        st.rerun()


def render_experiment_editor(store: AblationStore, project_id: str, experiment: Experiment) -> None:
    if experiment.config_file:
        action_cols = st.columns([1, 3])
        if action_cols[0].button("从配置重建参数", key=f"extract_params_{experiment.id}"):
            content = store.read_config_text(project_id, experiment)
            if content is None:
                st.error("没有找到配置文件。")
            else:
                try:
                    experiment.params = extract_params_from_config_text(content)
                except SyntaxError as exc:
                    st.error(f"配置文件解析失败：{exc}")
                else:
                    store.save_experiment(project_id, experiment)
                    st.success(f"已从配置文件提取 {len(experiment.params)} 个参数。")
                    st.rerun()
        action_cols[1].caption("会用当前 config.py 的解析结果替换参数，适合修正手动填写造成的截断或混合字段。")

    with st.form(f"experiment_edit_form_{experiment.id}", clear_on_submit=False):
        left, right = st.columns(2)
        with left:
            name = st.text_input("实验名称", value=experiment.name, key=f"edit_name_{experiment.id}")
            dataset = st.text_input("数据集", value=experiment.dataset, key=f"edit_dataset_{experiment.id}")
            model = st.text_input("模型", value=experiment.model, key=f"edit_model_{experiment.id}")
            strategy = st.text_input("策略", value=experiment.strategy, key=f"edit_strategy_{experiment.id}")
        with right:
            tags_text = st.text_input(
                "标签",
                value=", ".join(experiment.tags),
                key=f"edit_tags_{experiment.id}",
            )
            seed = st.text_input("随机种子", value=experiment.seed, key=f"edit_seed_{experiment.id}")
            config_file = st.file_uploader(
                "替换 Python 配置文件",
                type=["py"],
                accept_multiple_files=False,
                key=f"edit_config_{experiment.id}",
            )
            log_file = st.file_uploader(
                "替换训练日志",
                type=["log", "txt"],
                accept_multiple_files=False,
                key=f"edit_log_{experiment.id}",
            )
            rebuild_from_new_config = st.checkbox(
                "用新配置文件重建参数",
                value=True,
                key=f"edit_rebuild_config_{experiment.id}",
                help="上传新配置文件时生效。勾选后会忽略下方参数文本，直接使用新配置解析结果。",
            )

        description = st.text_area(
            "实验描述",
            value=experiment.description,
            height=90,
            key=f"edit_description_{experiment.id}",
        )
        params_text = st.text_area(
            "参数",
            value=format_key_value_lines(experiment.params),
            height=220,
            key=f"edit_params_{experiment.id}",
            help="每行一个 key=value，支持数字、布尔值、列表和字典。",
        )
        metrics_text = st.text_area(
            "结果指标",
            value=format_key_value_lines(experiment.metrics),
            height=150,
            key=f"edit_metrics_{experiment.id}",
            help="每行一个 key=value。数字指标可以在对比页进行范围筛选。",
        )

        image_updates: list[tuple[int, str, str, bool]] = []
        if experiment.images:
            st.markdown("<div class='masr-muted'>已有图片</div>", unsafe_allow_html=True)
            for index, image in enumerate(experiment.images):
                cols = st.columns([2, 1, 2, 1])
                cols[0].write(image.filename)
                label = cols[1].text_input(
                    "标签",
                    value=image.label,
                    key=f"edit_image_label_{experiment.id}_{index}",
                    label_visibility="collapsed",
                )
                note = cols[2].text_input(
                    "说明",
                    value=image.note,
                    key=f"edit_image_note_{experiment.id}_{index}",
                    label_visibility="collapsed",
                )
                remove = cols[3].checkbox(
                    "移除",
                    key=f"edit_image_remove_{experiment.id}_{index}",
                    help="只从实验记录中移除，不删除磁盘文件。",
                )
                image_updates.append((index, label, note, remove))

        new_image_files = st.file_uploader(
            "追加结果图片",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            accept_multiple_files=True,
            key=f"edit_new_images_{experiment.id}",
        )
        new_image_meta: list[tuple[str, str]] = []
        if new_image_files:
            st.markdown("<div class='masr-muted'>新图片标签</div>", unsafe_allow_html=True)
            for index, image in enumerate(new_image_files):
                cols = st.columns([2, 1, 2])
                cols[0].write(image.name)
                label = cols[1].text_input(
                    "标签",
                    value="result",
                    key=f"edit_new_image_label_{experiment.id}_{index}",
                    label_visibility="collapsed",
                )
                note = cols[2].text_input(
                    "说明",
                    key=f"edit_new_image_note_{experiment.id}_{index}",
                    label_visibility="collapsed",
                )
                new_image_meta.append((label, note))

        submitted = st.form_submit_button("保存修改", use_container_width=True)

    if submitted:
        if not name.strip():
            st.error("请填写实验名称。")
            return

        config_bytes = config_file.getvalue() if config_file is not None else None
        if config_bytes is not None and rebuild_from_new_config:
            try:
                params = extract_params_from_config_text(config_bytes.decode("utf-8", errors="replace"))
            except SyntaxError as exc:
                st.error(f"配置文件解析失败：{exc}")
                return
        else:
            params = parse_key_value_lines(params_text)

        experiment.name = name.strip()
        experiment.description = description.strip()
        experiment.tags = parse_tags(tags_text)
        experiment.dataset = dataset.strip()
        experiment.model = model.strip()
        experiment.strategy = strategy.strip()
        experiment.seed = seed.strip()
        experiment.params = params
        experiment.metrics = parse_metrics_text(metrics_text)

        if config_file is not None:
            filename, original_name = store.save_config_file(
                project_id,
                experiment.id,
                config_file.name,
                config_bytes or b"",
            )
            experiment.config_file = filename
            experiment.config_original_name = original_name

        if log_file is not None:
            log_bytes = log_file.getvalue()
            filename, original_name = store.save_log_file(
                project_id,
                experiment.id,
                log_file.name,
                log_bytes or b"",
            )
            experiment.log_file = filename
            experiment.log_original_name = original_name

        retained_images = []
        for index, label, note, remove in image_updates:
            if remove:
                continue
            image = experiment.images[index]
            image.label = label.strip() or "result"
            image.note = note.strip()
            retained_images.append(image)
        if not image_updates:
            retained_images = list(experiment.images)

        for image_file, (label, note) in zip(new_image_files or [], new_image_meta):
            retained_images.append(
                store.save_image_file(
                    project_id,
                    experiment.id,
                    image_file.name,
                    image_file.getvalue(),
                    label=label,
                    note=note,
                )
            )
        experiment.images = retained_images

        store.save_experiment(project_id, experiment)
        st.success("实验已更新。")
        st.rerun()

    cols = st.columns([3, 1])
    with cols[0]:
        if st.checkbox("查看原始 JSON", key=f"show_json_{experiment.id}"):
            st.json(experiment.to_dict(), expanded=False)
    with cols[1]:
        confirm = st.checkbox("确认删除", key=f"delete_confirm_{experiment.id}")
        if st.button("删除实验", key=f"delete_exp_{experiment.id}", disabled=not confirm):
            store.delete_experiment(project_id, experiment.id)
            st.success("实验已删除。")
            st.rerun()
