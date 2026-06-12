# MASR — 消融实验结果管理

**MASR**（Management of Ablation Study Results）是一个基于 Streamlit 的本地 Web 工具，用于管理深度学习消融实验的参数、指标和结果图片。面向 MMEngine/OpenMMLab 生态，纯文件目录存储，零数据库依赖。

## 项目定位

对标 MLflow Tracking 中消融实验管理需求的轻量替代方案。约 4600 行 Python 代码，覆盖实验元数据录入、配置文件安全解析、训练日志解析与曲线可视化、多维筛选对比和图片查看的全流程。

## 技术栈

| 技术 | 用途 |
|------|------|
| **Streamlit** ≥1.37 | Web UI 框架 |
| **Pandas** ≥2.2 | 表格展示与数据处理 |
| **Plotly** | 训练曲线交互式图表 |
| **Pillow** ≥10 | 图片读取、展示与缩略图生成 |
| **pytest** ≥8 | 单元测试（47 条用例） |
| Python 标准库 `ast` | 配置文件安全解析（不执行代码） |
| Python 标准库 `abc` | 插件式解析器抽象基类 |
| Python 标准库 `re` | 训练日志指标提取 |
| Python 标准库 `json` / `pathlib` / `uuid` | 数据序列化与文件管理 |

## 项目结构

```
MASR/
├── app.py                          # Streamlit 入口（导航、样式、页面路由，~60 行）
├── requirements.txt                # 依赖声明
├── pytest.ini                      # pytest 配置
├── README.md                       # 项目说明
├── masr/                           # 核心库（无状态，可独立引用）
│   ├── __init__.py                 # 导出公共 API
│   ├── models.py                   # 数据模型：Project / Experiment / ImageRecord，含验证
│   ├── storage.py                  # 文件存储层 AblationStore（纯 I/O）
│   ├── services.py                 # 业务逻辑服务层（校验 + 编排）
│   ├── filters.py                  # 实验筛选：多维组合过滤 + 辅助收集函数
│   ├── parsing.py                  # 文本解析：标签拆分、key=value 行解析、MMEngine 复合指标行解析
│   ├── config_parser.py            # 插件式配置解析器：AST 安全解析 + 注册表 + 自定义解析器支持
│   ├── log_parser.py               # 训练日志解析：最佳 epoch 指标提取 + epoch 级曲线数据提取
│   ├── image_utils.py              # 图片缩略图生成（Pillow）
│   └── utils.py                    # 通用工具：slugify、文件名清理、去重
├── views/                          # 页面层（UI 逻辑）
│   ├── overview.py                 # 项目概览仪表盘
│   ├── experiment_form.py          # 实验录入表单
│   ├── compare.py                  # 筛选与对比（含筛选器重置）
│   └── maintenance.py              # 数据维护（编辑、删除、日志指标提取、曲线查看）
├── components/                     # 可复用 UI 组件
│   ├── project_selector.py         # 侧边栏项目选择器
│   └── comparison.py               # 对比表格、图片网格（含缩略图）、配置文件查看、日志查看、训练曲线
├── utils/                          # 应用层工具
│   └── display.py                  # 实验扁平化、数值格式化、标签生成
├── tests/
│   ├── test_filters.py             # 筛选器 + 解析器 + config_parser + log_parser 测试（26 条）
│   ├── test_storage.py             # 存储层 CRUD + 文件名处理测试（5 条）
│   ├── test_models.py              # 模型验证 + 深度保护测试（12 条）
│   └── test_image_utils.py         # 缩略图生成测试（4 条）
├── data/projects/                  # 运行时数据目录（gitignore）
└── test_artifacts/                 # 测试临时数据（gitignore）
```

### 架构分层

```
┌─────────────────────────────────────────────────────────┐
│  app.py          (入口 + 路由)                           │
├─────────────────────────────────────────────────────────┤
│  views/          (页面逻辑)                              │
├─────────────────────────────────────────────────────────┤
│  components/     (UI 组件)                              │  ← Streamlit 依赖
├─────────────────────────────────────────────────────────┤
│  utils/          (显示工具)                              │
├─────────────────────────────────────────────────────────┤
│  masr/services   (业务逻辑 + 校验)                       │
├─────────────────────────────────────────────────────────┤
│  masr/storage    (文件 I/O)                             │  ← 纯 Python，无 UI 依赖
├─────────────────────────────────────────────────────────┤
│  masr/models     (数据模型)                              │
└─────────────────────────────────────────────────────────┘
```

层级间单向依赖：上层可引用下层，反之不可。

## 核心功能

### 1. 项目管理

- 创建/编辑/删除项目，自动生成唯一 ID（`name-timestamp-uuid8`）
- 侧边栏下拉切换，显示每个项目下的实验数量

### 2. 实验录入

- 基础信息：实验名称、数据集、模型、策略、随机种子
- **名称校验**：实验名称不能为空，创建/保存时自动验证
- 标签系统：逗号分隔（支持中英文逗号），如 `baseline, no_prompt, dice_loss`
- **配置文件上传**：上传 `.py` 配置文件并自动提取参数（可开关）
- **训练日志上传**：上传 `.log` / `.txt` 训练日志文件，支持后续自动提取最佳 epoch 指标和绘制训练曲线
- **手动补充参数**：以 `key = value` 格式录入，同名键覆盖自动提取值
- **结果指标**：支持两种录入方式——
  - 简单格式：`key = value` 或 `key: value`，每行一个，自动推断类型
  - **MMEngine 复合格式**：直接粘贴评估日志行，自动拆分为独立指标。例如输入 `验证集accuracy/top1: 68.5415  single-label/precision: 62.2472  single-label/recall: 56.6112  single-label/f1-score: 56.5116` 会自动解析为四个指标（验证集accuracy / precision / recall / f1），并归一化指标名（`f1-score`→`f1`，`top1`→`accuracy`）
- **多图上传**：每张图可设标签（label）和说明（note），文件名自动去重，上传时自动生成缩略图

### 3. 配置文件安全解析（核心特性）

不上传配置文件也能手动录入，但上传后会**安全解析**：只做 AST 语法树遍历，只求值字面量和 `dict(...)` 调用，**绝不 `exec` 或 `import` 用户代码**。

```python
# 输入（MMEngine 风格配置）
model = dict(
    type='ImageClassifier',
    backbone=dict(type='ResNet', depth=50),
    head=dict(num_classes=6, loss=dict(type='CrossEntropyLoss')),
)
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.001))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='PackClsInputs'),
]

# 输出（自动扁平化 + pipeline 摘要）
model.type               → "ImageClassifier"
model.backbone.type      → "ResNet"
model.backbone.depth     → 50
optim_wrapper.optimizer.lr → 0.001
train_pipeline_summary   → "LoadImageFromFile → RandomResizedCrop → PackClsInputs"
```

**关键参数自动优先展示**：筛选对比页的参数列默认按优先级选取前 12 个——学习率（`lr`）> 优化器（`optimizer`）> 学习率衰减（`param_scheduler`）> train_pipeline > test_pipeline > 其他常用键。

**插件式解析器架构**：内置 `MMEngineConfigParser` 处理 MMEngine 风格配置，可通过 `register_parser()` 注册自定义解析器支持其他配置格式（YAML、Hydra 等）。

### 4. 训练日志解析（核心特性）

上传训练日志后，系统可自动解析 MMEngine 格式的训练日志，提取：

- **最佳 epoch 指标**：自动识别日志中的 `The best checkpoint with ... at N epoch` 行确定最佳 epoch，或通过全量扫描 + 目标指标自动选取，提取验证集指标
- **Epoch 级曲线数据**：扫描所有 `Epoch(val)` 和 `Epoch(train)` 行，提取每个 epoch 的指标值（accuracy、f1、loss 等），支持训练曲线可视化
- **测试日志指标**：从 `Epoch(test)` 行提取测试集指标

```python
from masr.log_parser import extract_best_metrics, extract_epoch_curves, extract_test_metrics

# 从训练日志提取最佳 epoch 的验证集指标
best_metrics, best_epoch, summary = extract_best_metrics(log_text)
# → ({"验证集accuracy": 68.54, "验证集f1": 56.51, ...}, 141, "Epoch 141 — ...")

# 提取所有 epoch 的指标曲线（用于绘图）
curves = extract_epoch_curves(log_text)
# → {"epoch": [1,2,...], "验证集accuracy": [...], "训练集loss": [...], ...}

# 从测试日志提取指标
test_metrics = extract_test_metrics(test_log_text)
# → {"测试集accuracy": 72.31, "测试集f1": 61.24, ...}
```

**智能指标筛选**：自动过滤非指标噪声（`data_time`、`time`、`memory`、混淆矩阵 dump 等），仅保留真正的评估指标。指标名自动归一化并附加中文分组前缀（验证集/测试集/训练集）。

**训练曲线可视化**：在筛选对比页和数据维护页均可查看交互式 Plotly 曲线图，支持多指标自由选择和悬停查看精确数值。

### 5. 实验筛选与对比

**多维筛选器**：
| 维度 | 方式 |
|------|------|
| 全文搜索 | 名称/描述/数据集/模型/策略/标签/参数值/指标值 |
| 数据集/模型/策略 | 多选下拉（自动收集已有值） |
| 标签 | 多选，要求包含全部选中标签 |
| 参数 | 选参数名 + 输入值片段（包含匹配） |
| 指标 | 选指标名 + 范围滑块（自动计算 min/max） |

**筛选器重置**：一键清除所有筛选条件，恢复默认状态。

**对比展示**：
- 可自由选择显示列（参数列默认优先展示常用键）
- 对比表格列顺序：实验名称 → 数据集 → 模型 → 指标 → 策略 → 随机种子 → 标签 → 更新时间 → 参数
- **训练曲线对比**：选择一个实验查看其交互式 Plotly 曲线图，默认勾选 accuracy、f1、loss，可自由选择其他指标
- 结果图片按 label 分 Tab 展示，每行 3 列，优先加载缩略图，可展开查看原图
- 可展开查看原始配置文件内容（Python 语法高亮）
- 可展开查看训练日志原文

### 6. 数据维护

- 编辑项目信息
- **从配置重建参数**：一键重新解析 `config.py` 覆盖 `params`（修正手工填写造成的截断或混合字段）
- **从日志提取最佳指标**：一键解析训练日志，用验证集最佳 epoch 的指标替换当前 metrics
- **从测试日志提取指标**：一键解析测试日志，将 `Epoch(test)` 指标合并到当前 metrics
- **查看训练曲线**：在实验编辑器中直接查看单个实验的 epoch 级训练曲线（交互式 Plotly 图表）
- 查看实验完整 JSON 元数据
- 删除实验（需勾选确认）/ 删除项目（需输入项目名称确认）
- 实验编辑：直接修改名称、数据集、模型、策略、标签、参数、指标、配置文件、训练日志、测试日志、图片说明
- **测试日志自动提取**：上传测试日志时自动解析并合并 `Epoch(test)` 指标

### 7. 概览仪表盘

- 4 个指标卡片：实验总数 / 数据集种类 / 模型种类 / 指标种类
- 最近 10 个实验的表格预览
- 项目数据目录的绝对路径

## 数据模型

```python
Project
  id, name, description, created_at, updated_at

Experiment
  id, name, description, tags, dataset, model, strategy, seed
  params: dict[str, Any]      # 展平后的超参，如 {"lr": 0.001, "loss": "dice"}
  metrics: dict[str, Any]     # 结果指标，如 {"Dice": 0.871, "IoU": 0.792}
  test_metrics: dict[str, Any] # 测试集指标（上传测试日志时自动填充）
  images: list[ImageRecord]
  config_file, config_original_name     # 关联的配置文件
  log_file, log_original_name           # 关联的训练日志
  test_log_file, test_log_original_name # 关联的测试日志

ImageRecord
  filename, label, note, uploaded_at
```

所有模型在构造时自动校验必填字段非空；`from_dict` 反序列化支持 `max_depth` 参数（默认 10）防止恶意嵌套。

## 存储架构

纯 JSON 文件 + 原始文件目录，可 Git 追踪（`data/` 在 gitignore 中）：

```text
data/projects/<project_id>/
├── project.json                     # Project 序列化
└── experiments/<experiment_id>/
    ├── manifest.json                # Experiment 完整序列化
    ├── config.py                    # 上传的原始配置文件
    ├── train.log                    # 上传的训练日志
    ├── test.log                     # 上传的测试日志
    └── images/                      # 结果图片
        ├── result.png               # 原图
        └── thumbnails/              # 缩略图（自动生成）
            └── result.png
```

- `project.json` 的 `updated_at` 在实验增删时自动更新
- 加载时自动跳过损坏的 `manifest.json`，不影响其他实验
- 图片上传时自动生成 300px 宽度缩略图，对比页优先加载缩略图以提升性能

## 安全设计

- **配置文件**：AST 解析，不执行用户代码；支持输入大小限制（建议 1MB）
- **日志文件**：纯文本读取，使用 `errors='replace'` 处理编码异常
- **路径安全**：删除前校验目标路径在存储根目录内，防止路径穿越
- **文件名处理**：清理非法字符、检测 Windows 保留名（`CON`/`PRN` 等）、自动去重
- **深度保护**：JSON 反序列化默认限深 10 层，防止栈溢出
- **输入校验**：模型层 `__post_init__` 校验必填字段，服务层二次校验
- **无依赖注入**：无数据库、无网络请求，纯本地工具

## 运行

```powershell
# 安装依赖
pip install -r requirements.txt

# 启动（默认数据目录 data/projects/）
streamlit run app.py
```

## 测试

```powershell
pytest
```

测试覆盖（47 条用例）：
- `tests/test_filters.py`（26 条）— 筛选器多条件组合、collectors、指标范围、标签/键值对解析、MMEngine 复合指标行解析、百分号数值转换、配置文件 AST 解析、pipeline 摘要提取、默认参数优先级选取、解析器注册表、训练日志最佳指标提取、epoch 曲线提取、测试日志指标提取
- `tests/test_models.py`（12 条）— 模型必填字段校验、`from_dict` 深度保护、序列化往返、log_file / test_log_file / test_metrics 字段
- `tests/test_storage.py`（5 条）— 项目和实验 CRUD 生命周期、配置文件保存/读取、日志文件保存/读取、测试日志保存/读取、图片存储、破损 manifest 容错、文件名清理和去重、路径安全校验
- `tests/test_image_utils.py`（4 条）— 缩略图生成、小图透传、缺失文件容错、路径计算

## 扩展指南

### 添加自定义配置解析器

```python
from masr.config_parser import ConfigParser, register_parser

class YAMLConfigParser(ConfigParser):
    def parse(self, text: str) -> dict[str, Any]:
        import yaml
        return yaml.safe_load(text)

register_parser(YAMLConfigParser())
```

注册后，`extract_params_from_config_text()` 会自动按注册顺序尝试各解析器，返回首个成功的结果。

### 使用服务层

```python
from masr import AblationStore, ExperimentService, ProjectService

store = AblationStore()
store.ensure()

project_svc = ProjectService(store)
project = project_svc.create_project("My Project", "description")

experiment_svc = ExperimentService(store)
# 服务层自动校验名称非空等业务规则
experiment = experiment_svc.create_experiment(project.id, experiment)
```

### 使用日志解析

```python
from masr import extract_best_metrics, extract_epoch_curves, extract_test_metrics

# 读取训练日志
log_text = open("train.log").read()

# 提取最佳 epoch 指标
metrics, epoch, summary = extract_best_metrics(log_text)

# 提取所有 epoch 曲线数据（用于绘图）
curves = extract_epoch_curves(log_text)
# curves = {"epoch": [...], "验证集accuracy": [...], "训练集loss": [...]}

# 提取测试日志指标
test_metrics = extract_test_metrics(open("test.log").read())
```
