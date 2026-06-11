"""Experiment creation form page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from masr.config_parser import extract_params_from_config_text
from masr.models import Experiment
from masr.parsing import parse_key_value_lines, parse_metrics_text, parse_tags
from masr.storage import AblationStore


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
            log_file = st.file_uploader("训练日志", type=["log", "txt"], accept_multiple_files=False)
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
