from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from masr.config_parser import extract_params_from_config_text, pick_default_param_keys
from masr.filters import (
    ExperimentFilters,
    collect_dynamic_keys,
    collect_tags,
    collect_values,
    filter_experiments,
    metric_range,
)
from masr.models import Experiment
from masr.parsing import format_key_value_lines, parse_key_value_lines, parse_metrics_text, parse_tags
from masr.storage import AblationStore


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


def render_overview(store: AblationStore, project_id: str) -> None:
    experiments = store.list_experiments(project_id)
    datasets = collect_values(experiments, "dataset")
    models = collect_values(experiments, "model")
    strategies = collect_values(experiments, "strategy")
    param_keys = pick_default_param_keys(collect_dynamic_keys(experiments, "params"))
    metric_keys = collect_dynamic_keys(experiments, "metrics")

    cols = st.columns(5)
    cols[0].metric("实验数", len(experiments))
    cols[1].metric("数据集", len(datasets))
    cols[2].metric("模型", len(models))
    cols[3].metric("策略", len(strategies))
    cols[4].metric("指标", len(metric_keys))

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


def render_experiment_form(store: AblationStore, project_id: str) -> None:
    st.subheader("录入实验")
    with st.form("experiment_create_form", clear_on_submit=False):
        left, right = st.columns(2)
        with left:
            name = st.text_input("实验名称")
            dataset = st.text_input("数据集")
            model = st.text_input("模型")
            strategy = st.text_input("策略")
        with right:
            tags_text = st.text_input("标签", placeholder="baseline, no_prompt, dice_loss")
            seed = st.text_input("随机种子")
            config_file = st.file_uploader("Python 配置文件", type=["py"], accept_multiple_files=False)
            auto_extract_params = st.checkbox("从配置文件自动提取参数", value=True)

        description = st.text_area("实验描述", height=90)
        params_text = st.text_area(
            "补充/覆盖参数",
            placeholder="learning_rate = 1e-4\nloss = dice\naugmentation = flip",
            height=130,
            help="上传配置文件时会自动提取参数；这里填写的同名参数会覆盖配置解析结果。",
        )
        metrics_text = st.text_area(
            "结果指标",
            placeholder="Dice = 0.871\nIoU = 0.792\nHD95 = 4.36",
            height=130,
        )

        image_files = st.file_uploader(
            "结果图片",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            accept_multiple_files=True,
        )
        image_meta: list[tuple[str, str]] = []
        if image_files:
            st.markdown("<div class='masr-muted'>图片标签</div>", unsafe_allow_html=True)
            for index, image in enumerate(image_files):
                cols = st.columns([1, 1, 2])
                cols[0].write(image.name)
                label = cols[1].text_input("标签", value="result", key=f"image_label_{index}", label_visibility="collapsed")
                note = cols[2].text_input("说明", key=f"image_note_{index}", label_visibility="collapsed")
                image_meta.append((label, note))

        submitted = st.form_submit_button("保存实验", use_container_width=True)

    if submitted:
        if not name.strip():
            st.error("请填写实验名称。")
            return
        config_bytes = config_file.getvalue() if config_file is not None else None
        auto_params: dict[str, Any] = {}
        if config_bytes is not None and auto_extract_params:
            try:
                config_text = config_bytes.decode("utf-8", errors="replace")
                auto_params = extract_params_from_config_text(config_text)
            except SyntaxError as exc:
                st.error(f"配置文件解析失败：{exc}")
                return
        manual_params = parse_key_value_lines(params_text)
        experiment = Experiment(
            id="",
            name=name.strip(),
            description=description.strip(),
            tags=parse_tags(tags_text),
            dataset=dataset.strip(),
            model=model.strip(),
            strategy=strategy.strip(),
            seed=seed.strip(),
            params={**auto_params, **manual_params},
            metrics=parse_metrics_text(metrics_text),
        )
        try:
            experiment = store.create_experiment(project_id, experiment)
            if config_file is not None:
                filename, original_name = store.save_config_file(
                    project_id,
                    experiment.id,
                    config_file.name,
                    config_bytes or b"",
                )
                experiment.config_file = filename
                experiment.config_original_name = original_name

            for image_file, (label, note) in zip(image_files or [], image_meta):
                image_record = store.save_image_file(
                    project_id,
                    experiment.id,
                    image_file.name,
                    image_file.getvalue(),
                    label=label,
                    note=note,
                )
                experiment.images.append(image_record)

            store.save_experiment(project_id, experiment)
            st.success("实验已保存。")
            st.rerun()
        except FileExistsError:
            st.error("实验目录已存在，请重试。")


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


def render_filters(experiments: list[Experiment]) -> ExperimentFilters:
    with st.expander("筛选条件", expanded=True):
        top = st.columns([2, 1, 1, 1])
        search = top[0].text_input("搜索")
        datasets = set(top[1].multiselect("数据集", collect_values(experiments, "dataset")))
        models = set(top[2].multiselect("模型", collect_values(experiments, "model")))
        strategies = set(top[3].multiselect("策略", collect_values(experiments, "strategy")))

        bottom = st.columns([1, 1, 1, 1])
        tags = set(bottom[0].multiselect("标签", collect_tags(experiments)))
        param_keys = collect_dynamic_keys(experiments, "params")
        param_key = bottom[1].selectbox("参数", [""] + param_keys, format_func=lambda value: value or "不限")
        param_value = bottom[2].text_input("参数值包含")

        metric_keys = collect_dynamic_keys(experiments, "metrics")
        metric_key = bottom[3].selectbox("指标", [""] + metric_keys, format_func=lambda value: value or "不限")

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
                        if path.exists():
                            st.image(str(path), caption=image.note or image.filename, use_column_width=True)
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


def _clean_number(value: Any) -> str:
    """Convert a value to a clean string with no trailing zeros.

    Floats use :g formatting (strips trailing zeros), ints stay as-is,
    everything else is stringified.
    """
    if isinstance(value, float):
        # :g removes trailing zeros and switches to scientific for tiny/large numbers
        return f"{value:g}"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    return str(value)


def flatten_experiments(
    experiments: list[Experiment],
    param_keys: list[str] | None = None,
    metric_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    param_keys = param_keys if param_keys is not None else collect_dynamic_keys(experiments, "params")
    metric_keys = metric_keys if metric_keys is not None else collect_dynamic_keys(experiments, "metrics")
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        row: dict[str, Any] = {
            "实验名称": experiment.name,
            "数据集": experiment.dataset,
            "模型": experiment.model,
        }
        for key in metric_keys:
            row[f"指标:{key}"] = _clean_number(experiment.metrics.get(key, ""))
        row.update({
            "策略": experiment.strategy,
            "随机种子": experiment.seed,
            "标签": ", ".join(experiment.tags),
            "更新时间": experiment.updated_at,
        })
        for key in param_keys:
            row[f"参数:{key}"] = _clean_number(experiment.params.get(key, ""))
        rows.append(row)
    return rows


def experiment_label(experiment: Experiment) -> str:
    parts = [experiment.name]
    detail = " / ".join(part for part in [experiment.dataset, experiment.model, experiment.strategy] if part)
    if detail:
        parts.append(detail)
    return " · ".join(parts)


if __name__ == "__main__":
    main()
