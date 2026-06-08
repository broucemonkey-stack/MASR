# MASR 消融实验结果管理

MASR 是一个本地 Streamlit 工具，用纯文件目录管理消融实验结果。它保存 Python 配置文件、关键参数、结果指标和多张结果图片，并支持按条件筛选后进行表格与图片对比。

上传 Python 配置文件时，应用会默认从配置中自动提取参数，并把嵌套配置拆成独立字段，例如 `optim_wrapper.optimizer.lr`、`model.backbone.depth`、`train_dataloader.batch_size`。手动参数输入只用于补充或覆盖少数字段。

## 运行

```powershell
streamlit run app.py
```

默认数据目录为 `data/projects/`。

已有实验可以在“数据维护”页面点击“从配置重建参数”，用 `config.py` 的解析结果替换旧的手动参数。

“数据维护”页面也支持直接修改实验名称、数据集、模型、策略、标签、参数、指标、配置文件和图片说明，并可以追加结果图片。

## 数据结构

```text
data/
  projects/
    <project_id>/
      project.json
      experiments/
        <experiment_id>/
          manifest.json
          config.py
          images/
```

## 测试

```powershell
pytest
```
