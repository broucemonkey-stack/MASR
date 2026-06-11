# MASR — 消融实验结果管理

**MASR**（Management of Ablation Study Results）是一个基于 Streamlit 的本地 Web 工具，用于管理深度学习消融实验的参数、指标和结果图片。面向 MMEngine/OpenMMLab 生态，纯文件目录存储，零数据库依赖。

## 项目定位

对标 MLflow Tracking 中消融实验管理需求的轻量替代方案。约 700 行 Python 代码，覆盖实验元数据录入、配置文件安全解析、多维筛选对比和图片查看的全流程。

## 技术栈

| 技术 | 用途 |
|------|------|
| **Streamlit** ≥1.37 | Web UI 框架 |
| **Pandas** ≥2.2 | 表格展示与数据处理 |
| **Pillow** ≥10 | 图片读取与展示 |
| **pytest** ≥8 | 单元测试 |
| Python 标准库 `ast` | 配置文件安全解析（不执行代码） |
| Python 标准库 `json` / `pathlib` / `uuid` | 数据序列化与文件管理 |

## 项目结构

```
MASR/
├── app.py                    # Streamlit 主入口，包含全部 4 个页面 UI 及布局逻辑
├── requirements.txt          # 依赖声明
├── pytest.ini                # pytest 配置
├── README.md                 # 项目说明
├── masr/                     # 核心库（无状态，可独立引用）
│   ├── __init__.py           # 导出公共 API：AblationStore, Experiment, ImageRecord, Project
│   ├── models.py             # 数据模型：Project / Experiment / ImageRecord 三个 dataclass
│   ├── storage.py            # 文件存储层 AblationStore（CRUD + 文件名安全 + 路径校验）
│   ├── filters.py            # 实验筛选：多维组合过滤 + 辅助收集函数
│   ├── parsing.py            # 文本解析：标签拆分、key=value 行解析、MMEngine 复合指标行解析
│   └── config_parser.py      # AST 安全解析 + 嵌套扁平化 + pipeline 摘要 + 关键参数优先选取
├── tests/
│   ├── test_filters.py       # 筛选器 + 解析器 + config_parser 测试
│   └── test_storage.py       # 存储层 CRUD 生命周期 + 容错 + 文件名处理测试
├── data/projects/            # 运行时数据目录（gitignore）
└── test_artifacts/           # 测试临时数据（gitignore，pytest 自动生成和清理）
```

## 核心功能

### 1. 项目管理

- 创建/编辑/删除项目，自动生成唯一 ID（`name-timestamp-uuid8`）
- 侧边栏下拉切换，显示每个项目下的实验数量

### 2. 实验录入

- 基础信息：实验名称、数据集、模型、策略、随机种子
- 标签系统：逗号分隔（支持中英文逗号），如 `baseline, no_prompt, dice_loss`
- **配置文件上传**：上传 `.py` 配置文件并自动提取参数（可开关）
- **手动补充参数**：以 `key = value` 格式录入，同名键覆盖自动提取值
- **结果指标**：支持两种录入方式——
  - 简单格式：`key = value` 或 `key: value`，每行一个，自动推断类型
  - **MMEngine 复合格式**：直接粘贴评估日志行，自动拆分为独立指标。例如输入 `验证集accuracy/top1: 68.5415  single-label/precision: 62.2472  single-label/recall: 56.6112  single-label/f1-score: 56.5116` 会自动解析为四个指标（验证集accuracy / precision / recall / f1），并归一化指标名（`f1-score`→`f1`，`top1`→`accuracy`）
- **多图上传**：每张图可设标签（label）和说明（note），文件名自动去重

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

### 4. 实验筛选与对比

**多维筛选器**：
| 维度 | 方式 |
|------|------|
| 全文搜索 | 名称/描述/数据集/模型/策略/标签/参数值/指标值 |
| 数据集/模型/策略 | 多选下拉（自动收集已有值） |
| 标签 | 多选，要求包含全部选中标签 |
| 参数 | 选参数名 + 输入值片段（包含匹配） |
| 指标 | 选指标名 + 范围滑块（自动计算 min/max） |

**对比展示**：
- 可自由选择显示列（参数列默认优先展示常用键）
- 对比表格列顺序：实验名称 → 数据集 → 模型 → 指标 → 策略 → 随机种子 → 标签 → 更新时间 → 参数
- 结果图片按 label 分 Tab 展示，每行 3 列
- 可展开查看原始配置文件内容（Python 语法高亮）

### 5. 数据维护

- 编辑项目信息
- **从配置重建参数**：一键重新解析 `config.py` 覆盖 `params`（修正手工填写造成的截断或混合字段）
- 查看实验完整 JSON 元数据
- 删除实验（需勾选确认）/ 删除项目（需输入项目名称确认）
- 实验编辑：直接修改名称、数据集、模型、策略、标签、参数、指标、配置文件、图片说明

### 6. 概览仪表盘

- 5 个指标卡片：实验总数 / 数据集种类 / 模型种类 / 策略种类 / 指标种类
- 最近 10 个实验的表格预览
- 项目数据目录的绝对路径

## 数据模型

```python
Project
  id, name, description, created_at, updated_at

Experiment
  id, name, description, tags, dataset, model, strategy, seed
  params: dict[str, Any]   # 展平后的超参，如 {"lr": 0.001, "loss": "dice"}
  metrics: dict[str, Any]  # 结果指标，如 {"Dice": 0.871, "IoU": 0.792}
  images: list[ImageRecord]
  config_file, config_original_name  # 关联的配置文件

ImageRecord
  filename, label, note, uploaded_at
```

## 存储架构

纯 JSON 文件 + 原始文件目录，可 Git 追踪（`data/` 在 gitignore 中）：

```text
data/projects/<project_id>/
├── project.json                     # Project 序列化
└── experiments/<experiment_id>/
    ├── manifest.json                # Experiment 完整序列化
    ├── config.py                    # 上传的原始配置文件
    └── images/                      # 结果图片
```

- `project.json` 的 `updated_at` 在实验增删时自动更新
- 加载时自动跳过损坏的 `manifest.json`，不影响其他实验

## 安全设计

- **配置文件**：AST 解析，不执行用户代码
- **路径安全**：删除前校验目标路径在存储根目录内，防止路径穿越
- **文件名处理**：清理非法字符、检测 Windows 保留名（`CON`/`PRN` 等）、自动去重
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

测试覆盖：
- `tests/test_storage.py` — 项目和实验 CRUD 生命周期、配置文件保存/读取、图片存储、破损 manifest 容错、文件名清理和去重、路径安全校验
- `tests/test_filters.py` — 筛选器多条件组合、collectors、指标范围、标签/键值对解析、MMEngine 复合指标行解析、百分号数值转换、配置文件 AST 解析、pipeline 摘要提取、默认参数优先级选取
