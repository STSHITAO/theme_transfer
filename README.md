# Theme Transfer & ITTE

面向移动 App 图标的真实主题学习、整包风格迁移与客观图像评测系统。

项目从设计师制作的 `original → style_ref` 配对中学习主题包的共享设计语言，由 Qwen 为每个目标 App 制定受约束的迁移策略，使用 Wan 生成多个候选并完成多模态质量筛选；独立的 ITTE（Icon Theme Transfer Evaluation）随后从图像层面评估风格忠实度、身份保持、整包一致性与视觉质量。

ITTE = Icon Theme Transfer Evaluation。

整个工作流强调三件事：主题规则来自真实设计样例，目标身份不会被参考 App 污染，评测证据不回灌生图链路。

## 核心能力

- **真实主题学习**：分析主题内全部有效 `original → style_ref` 配对，通过分批多模态分析与整包聚合提取色彩、描边、材质、构图、主体比例和结构变化规律。
- **逐 App 迁移决策**：结合主题规律、目标原图和中性应用语义，在生图前冻结 `preserve_major_structure` 或 `semantic_recompose` 路线。
- **身份隔离生图**：Wan 始终接收 `TARGET_IMAGE` 在前、`STYLE_REFERENCE` 在后；参考图只提供视觉处理，不提供 logo、文字、主体或内部结构。
- **多候选质量控制**：支持每个 App 生成多个候选，由 Qwen QC 综合主题匹配、身份识别、语义合理性、过度重构风险和画面缺陷选择结果。
- **断点续跑与故障隔离**：完整 case 可复用；单个输入被平台拒绝时记录原因并继续，避免整包任务因一个 App 中断。
- **ITTE 客观评测**：使用 DINOv3、VGG16 Gram、DISTS、LPIPS 及图像质量诊断，在 GPU/CPU 上运行并复用确定性磁盘特征缓存。
- **真实 Benchmark**：提供 91 张原图、231 张主题图和 158 组验证配对，用于在调整指标前后执行同划分对照实验。

## 系统架构

```text
dataset/                         规范化原图、主题图与 App 元数据
    │
    ├── scripts/prepare_generation_data.py
    ▼
data/styles + data/targets       生图入口数据
    │
    ├── Qwen 全主题配对分析
    ├── 逐 App 身份与结构策略
    ├── Wan 多候选生成
    └── Qwen 候选 QC
    ▼
data/packages/<package_id>/      候选、最终图、接触表与完整溯源
    │
    └── ITTE v1.4（图像证据 + 特征缓存）
    ▼
data/evaluations/<eval_id>/      分项分数、硬门、覆盖率与离群报告
```

生成阶段与评测阶段严格分离。Prompt、应用描述和 Qwen QC 只参与生成或诊断，不进入 ITTE 主分。

## ITTE 评测框架

ITTE v1.4 包含四个主维度：

| 维度 | 关注问题 | 主要证据 |
|---|---|---|
| Style Fidelity | 结果是否属于指定主题，而非自创风格 | VGG16 多层 Gram、主题参考分布 |
| Identity Preservation | 目标 App 是否仍然可识别 | DINOv3 身份检索与条件结构诊断 |
| Package Coherence | 整包是否统一且仍归属于参考主题 | 包内一致性与参考主题归属 |
| Visual Quality | 是否存在空图、曝光、边界、伪影等技术问题 | 可观察图像质量规则 |

结构身份指标的适用性在生成前确定：保留主要结构时进入身份主分；语义重构时仍记录结构诊断，但不会因未复刻原始几何而直接扣除主分。高平均分不能覆盖严重单图失败，报告会保留硬门与整包离群结论。

Package Coherence 同时衡量包内视觉统一性和参考主题归属，因此整包内部统一但远离参考主题，不能得到高分。

## 已验证实验

theme-learning-v2 使用同一组 40 个 App，每个 App 生成两个候选。两个主题均完成 100% 生成与 ITTE 覆盖：

| 主题 | 真实学习配对 | 候选 / Final | 结构保留 / 语义重构 | ITTE | 风格 | 身份 | 整包一致性 | 视觉质量 | 客观结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| theme_003 | 86 | 80 / 40 | 29 / 11 | 89.03 | 90.19 | 94.34 | 70.96 | 99.81 | `failed_hard_gate`：`xiaohongshu` 为整包离群点 |
| theme_004 | 36 | 80 / 40 | 26 / 14 | 87.27 | 86.08 | 89.90 | 76.69 | 98.88 | `failed_hard_gate`：`wechat` 为整包离群点 |

实验保留了硬门失败，没有用高均分掩盖严重离群。当前设置允许目标 App 出现在主题学习配对中，因此这些结果衡量的是**主题包内重建**，不等价于未见 App 泛化。完整实验协议、候选 QC 与逐 App 证据见 [Theme Learning v2 实验报告](docs/THEME_LEARNING_V2_EXPERIMENT.md)。

## 快速开始

### 1. 获取项目与模型文件

```powershell
git clone https://github.com/STSHITAO/theme_transfer.git
Set-Location theme_transfer
git lfs pull
```

仓库中的大型公开模型权重通过 Git LFS 管理。若只克隆到指针文件，ITTE 模型将无法加载。

### 2. 配置 Python 环境

推荐 Python 3.11。GPU 环境应先按照显卡驱动安装匹配的 PyTorch/TorchVision，再安装项目依赖，避免通用安装覆盖 CUDA 构建。

```powershell
conda create -n theme_transfer python=3.11 -y
conda activate theme_transfer

# 先从 https://pytorch.org/get-started/locally/ 安装匹配的 CUDA PyTorch
python -m pip install -r requirements.txt
```

当前开发验证环境为 Conda `pytorch`、PyTorch `2.5.0+cu118` 和 NVIDIA GTX 1650。CPU 接口保持可用。

### 3. 配置模型 API

在项目根目录创建本地 `.env`，填写以下变量；不要提交密钥：

```dotenv
MOCK_MODE=false

ALI_PLAN_BASE_URL=<Qwen API 地址>
ALI_PLAN_MODEL=<Qwen 模型 ID>
ALI_PLAN_API_KEY=<Qwen API Key>

ALI_IMAGE_BASE_URL=<Wan API 地址>
ALI_IMAGE_MODEL=<Wan 模型 ID>
ALI_IMAGE_API_KEY=<Wan API Key>
```

所有固定自然语言 Prompt 均使用中文。JSON 字段名、枚举值、App ID 以及 `TARGET_IMAGE`、`STYLE_REFERENCE` 等协议标识保持英文，以保证现有解析和图片角色绑定稳定。

### 4. 准备生图数据

规范化源数据结构：

```text
dataset/
├── originals/<app_id>.<ext>
├── themes/<theme_id>/<app_id>.<ext>
└── apps.json
```

生成入口结构：

```powershell
python scripts/prepare_generation_data.py --clean
```

脚本只读取 `dataset/`，将匹配成功的原图与主题图转换为：

```text
data/
├── styles/<theme_id>/
│   ├── theme.json
│   └── <app_id>/
│       ├── <app_id>.<ext>
│       └── <app_id>_style_ref.<ext>
└── targets/<app_id>/
    ├── target.json
    └── <original_file>
```

`target.json` 和 `theme.json` 只保存名称、分类、应用市场描述和核心功能等中性事实，不写死钱包、准星等视觉对象，也不预设结构保留路线。

### 5. 生成主题包

单主题、每个 App 两个候选：

```powershell
conda activate theme_transfer
python scripts/run_full_generation.py --theme-id theme_003 --candidate-count 2
```

多个主题可重复传入 `--theme-id`；省略该参数时依次处理四个主题。任务默认按完整 case 断点续跑，使用 `--no-resume` 可显式重新生成。

固定抽取覆盖完整列表的 40 个目标进行实验：

```powershell
python scripts/run_full_generation.py `
  --theme-id theme_003 `
  --theme-id theme_004 `
  --target-limit 40 `
  --candidate-count 2 `
  --package-prefix package_theme_learning_v2
```

生成结果写入 `data/packages/<package_id>/`，包括主题分析、逐 App 迁移计划、候选图、QC 报告、最终图、接触表、失败记录和元数据。

### 6. 运行 ITTE

GPU：

```powershell
$env:TPQS_DEVICE='cuda:0'
$env:TPQS_BATCH_SIZE='2'
python scripts/evaluate_full_packages.py --theme-id theme_003
```

CPU：

```powershell
$env:TPQS_DEVICE='cpu'
$env:TPQS_BATCH_SIZE='1'
python scripts/evaluate_full_packages.py --theme-id theme_003
```

评测结果写入 `data/evaluations/`。跨主题比较会额外记录共同成功 App 交集，避免因平台拒绝或缺图造成不公平比较。

## Benchmark 复现

真实评测集位于 [benchmark/evaluation_set_v1](benchmark/evaluation_set_v1)。先验证资产与映射：

```powershell
conda activate theme_transfer
python benchmark/tools/validate_evaluation_set.py
```

在身份不重叠的参考/查询划分上运行设计师正样本与未迁移原图控制对照：

```powershell
$env:TPQS_DEVICE='cuda:0'
$env:TPQS_BATCH_SIZE='2'
python benchmark/tools/evaluate_current_itte.py --references 8 --queries 4
```

仓库保留以下可复现实验证据：

- `benchmark/evaluation_set_v1/results/itte_v12_baseline_gpu_cv/`
- `benchmark/evaluation_set_v1/results/itte_v13_final_gpu_cv/`
- `benchmark/evaluation_set_v1/results/itte_v13_final_gpu_cv/comparison_to_v12/`
- `benchmark/evaluation_set_v1/results/identity_retrieval_dinov3/`
- `benchmark/evaluation_set_v1/results/identity_retrieval_perceptual/`

设备冒烟测试：

```powershell
python benchmark/tools/smoke_itte_device.py --device cuda:0
python benchmark/tools/smoke_itte_device.py --device cpu
```

## 独立应用元数据提取工具

[tools/app_metadata_extractor](tools/app_metadata_extractor) 用于验证“从爬虫应用描述中提取中性 schema”的可行性。它拥有自己的中文 Prompt、Qwen 客户端、稳定 ID 映射、断点文件和测试：

- 不导入 `backend` 或生图脚本；
- 不读取或发送主题图标；
- 不修改 `dataset`、`data` 或 ITTE 结果；
- 默认只写入工具自己的忽略目录；
- 输出必须经过人工审核，不会自动进入项目主流程。

详见 [独立工具说明](tools/app_metadata_extractor/README.md)。

## 测试

主项目测试：

```powershell
conda activate theme_transfer
python -m unittest discover -v
```

独立元数据工具测试：

```powershell
python -m unittest tools.app_metadata_extractor.tests.test_extractor -v
```

当前验证结果为主项目 `81/81`、独立工具 `5/5`。

## 目录结构

```text
backend/                         主题分析、生图编排、QC 与产物管理
benchmark/evaluation_set_v1/    真实 Benchmark、冻结划分和实验报告
data/                            生图入口、主题包、评测结果与特征缓存
dataset/                         规范化只读源数据
docs/                            产品、技术、验收、实验和进度文档
evaluation/                      ITTE 指标与包级评测工作流
models/                          官方模型权重与本地模型缓存
prompts/                         Qwen/Wan 中文 Prompt 契约
scripts/                         数据转换、批量生成、评测和实验清单工具
tests/                           主项目回归测试
tools/app_metadata_extractor/   与主流程隔离的应用描述提取工具
```

## 评测边界

- ITTE 主分只使用图像证据；Prompt 文图匹配、OpenCLIP 和生成阶段 Qwen QC 不进入主分。
- 评测结果不回灌生成链路，不再生成 `generation_feedback_prompt.md`。
- Benchmark 主题描述用于解释数据，不参与评分。
- 当前真实数据主要提供设计师正样本与自然未迁移控制，不包含大规模人工主观评分或主观合成退化标签。
- 指标变更必须先保存同一数据划分下的冻结基线，再用相同输入、随机种子和 split 复测。
- DISTS 与 LPIPS 保留为诊断证据，不替代已经过真实标签检索验证的身份主指标。
- 特征缓存键包含输入身份、预处理视图、模型 ID 与相关配置；CPU/GPU 选择不改变数值缓存身份。

## 文档

- [产品说明](docs/PRODUCT_SPEC.md)
- [技术设计](docs/TECH_DESIGN.md)
- [验收清单](docs/ACCEPTANCE.md)
- [Theme Learning v2 实验](docs/THEME_LEARNING_V2_EXPERIMENT.md)
- [当前进度与复现状态](docs/PROGRESS.md)
