# theme_transfer

## TPQS 整包量化评价（当前口径）

TPQS 是独立的后处理评价模块，入口在 `evaluation/evaluate_package.py`，不调用 Qwen 或 Wan，也不修改生成结果。当前评价语义是：

```text
data/styles/<theme_id>/<ref_app>/*_background.*   # 参考 App 原始系统预置图标背景层
data/styles/<theme_id>/<ref_app>/*_foreground.*   # 参考 App 原始系统预置图标前景层
data/styles/<theme_id>/<ref_app>/*_style_ref.*    # 参考 App 被设计师绘制后的主题风格图
data/targets/<app>/<app>.png|jpg|jpeg|webp        # 待迁移 App 原始系统预置图标
data/packages/<package_id>/final/*.png            # 生成后的主题图标
```

也就是说，TPQS 会把 `background + foreground` 合成为 `reference_raw`，再评价 `reference_raw -> style_ref` 与 `target -> generated` 这两种风格变化是否一致。

主要输出：

```text
data/evaluations/<eval_id>/
  tpqs_report.json
  metrics.csv
  style_pairwise_distances.json
  style_delta_distances.json
  dino_pairwise_distances.json
  inputs_manifest.json
```

主要指标：

- `style_delta_transfer_score` / `style_transfer_score`：生成图相对 target 的风格变化是否接近参考包中的 `reference_raw -> style_ref` 变化。
- `package_internal_style_consistency_score`：生成包内部风格是否统一，只看生成包自身。
- `reference_style_distribution_match_score`：生成包风格差异分布是否比原 target 更接近真实主题参考包。
- `theme_membership_score`：单个 App 是否偏离整包终态主题风格。
- `identity_separability_score`：DINOv3 判断生成图是否仍能和对应原 App 区分匹配。
- `visual_quality_score`：颜色、亮度、饱和度、对比度、边缘密度等低层视觉统计是否向主题参考靠拢。

默认模型来源是 ModelScope，DINOv3 权重会下载到项目本地 `models/modelscope/`。测试不会下载 DINOv3，也不会调用真实 API：

```powershell
$env:TPQS_EMBEDDING_BACKEND='stats'
python -m unittest tests.test_tpqs_evaluation -v
```

## TPQS 整包量化评价

TPQS 是独立的后处理评价模块，入口在 `evaluation/evaluate_package.py`，不调用 Qwen 或 Wan，也不修改生成结果。它读取：

```text
data/styles/<theme_id>/*/*_style_ref.*
data/packages/<package_id>/final/*.png
data/targets/<app>/<app>.png|jpg|jpeg|webp
```

默认配置固定在 `evaluation/evaluate_package.py`：

```python
THEME_ID = "theme_001"
PACKAGE_ID = "package_001_theme_001"
EVAL_ID = "eval_001_package_001_theme_001"
```

正式 TPQS 默认使用 Style Features + DINOv3：

```powershell
conda activate SEG
python evaluation/evaluate_package.py
```

默认环境变量为：

```env
TPQS_EMBEDDING_BACKEND=dinov3
TPQS_MODEL_SOURCE=modelscope
TPQS_MODEL_ID=facebook/dinov3-vitb16-pretrain-lvd1689m
TPQS_STYLE_FEATURE_BACKEND=color_edge_composition
TPQS_DEVICE=cpu
TPQS_POOLING=cls
TPQS_IMAGE_SIZE=224
TPQS_BATCH_SIZE=1
TPQS_USE_OPENCLIP=false
```

DINOv3 只用于 App 身份可分辨度、结构异常和主体塌缩风险，不再单独决定整包风格好坏。风格类指标主要使用 Style Features，包括颜色直方图、HSV/RGB 统计、边缘/描边、构图、复杂度等。

DINOv3 权重会下载到项目本地，不使用系统默认 Hugging Face 缓存：

```text
models/huggingface/hub/
models/modelscope/
models/torch/
```

默认模型来源是 ModelScope。需要切回 Hugging Face 时可设置：

```powershell
$env:TPQS_MODEL_SOURCE='huggingface'
python evaluation/evaluate_package.py
```

快速离线测试可以使用统计特征 backend，这个结果只用于开发验证，报告里会标记 `is_official_tpqs=false`：

```powershell
$env:TPQS_EMBEDDING_BACKEND='stats'
python evaluation/evaluate_package.py
```

评价输出写入：

```text
data/evaluations/<eval_id>/
  tpqs_report.json
  metrics.csv
  style_pairwise_distances.json
  dino_pairwise_distances.json
  inputs_manifest.json
```

TPQS 主要指标包括：

- `style_transfer_score`：生成图是否比原 target 更接近主题参考风格。
- `package_internal_style_consistency_score`：生成包内部风格是否统一，只看生成包自身。
- `reference_style_distribution_match_score`：生成包风格差异分布是否比原 target 更接近真实主题参考包。
- `theme_membership_score`：单个 App 是否偏离整包主题风格。
- `identity_separability_score`：DINOv3 判断生成图是否仍能和对应原 App 区分匹配。
- `visual_quality_score`：颜色、亮度、饱和度、对比度、边缘密度等低层视觉统计是否向主题参考靠拢。
- `tpqs` / `tpqs_total_score`：上述指标的几何平均分。

当前版本不输出 PCA 图，避免把可视化调试产物混入主评价结果。

测试不会下载 DINOv3，也不会调用真实 API：

```powershell
$env:TPQS_EMBEDDING_BACKEND='stats'
python -m unittest tests.test_tpqs_evaluation -v
```

图标主题风格迁移 MVP。项目从一个主题包里的多个参考 App 中学习共同的视觉迁移规则，再把这些规则迁移到目标 App 的原始图标上，生成候选图并用 Qwen 做质检选择。

当前工作流使用：

- Qwen 多模态模型：分析主题风格、做候选图 QC。
- Wan 图像生成模型：根据主题参考图和目标图生成迁移候选。
- 本地 mock 模式：不调用真实 API，用于快速验证流程。

## 目录结构

```text
backend/
  run_case.py              # 示例入口，当前默认跑 theme_001 -> bilibili
  workflow.py              # 端到端编排
  services/
    path_service.py        # 解析主题包和目标 App 路径
    image_service.py       # 生成参考 layout / 目标 layout
    qwen_client.py         # Qwen 分析和 QC
    wan_client.py          # Wan 候选图生成
    prompt_service.py      # 组装 Wan 生成 prompt
    qc_service.py          # 选择最佳候选图
    storage_service.py     # 保存 JSON 产物

data/
  styles/<theme_id>/<app>/ # 主题包参考样例
  styles/<theme_id>/theme.json # 参考 App 的轻量语义描述
  targets/<target_app>/    # 目标 App 原始图
  targets/<target_app>/target.json # 目标 App 的轻量语义描述
  cases/                   # 运行中间产物，默认不提交
  outputs/                 # 生成结果，默认不提交

prompts/
  qwen_theme_analysis.md   # 主题规则分析 prompt
  wan_generation.md        # Wan 生成 prompt 模板
  qwen_qc.md               # Qwen 质检 prompt
  qwen_target_identity.md  # 目标 App 身份锚点分析 prompt
  qwen_theme_design_analysis.md # V2 主题设计语言分析 prompt
  qwen_identity_strategy.md # V2 目标 App 表达策略 prompt
  qwen_transfer_plan.md    # 单图迁移计划 prompt
  qwen_package_qc.md       # 整包一致性质检 prompt

tests/                     # 单元测试和 mock workflow 测试
```

## 输入约定

主题包参考样例仍然是三张图：

```text
data/styles/theme_001/alipay/alipay_background.png
data/styles/theme_001/alipay/alipay_foreground.png
data/styles/theme_001/alipay/alipay_style_ref.jpg
```

含义：

- `background`：参考 App 的背景层。
- `foreground`：参考 App 的前景主体层。
- `style_ref`：已经完成主题迁移后的参考图，用来学习最终主题效果。

目标 App 现在只需要一张最终原始图：

```text
data/targets/bilibili/bilibili.png
data/targets/qq/qq.jpg
data/targets/tieba/<任意唯一图片名>.jpg
```

目标图解析规则：

- 优先寻找 `<target_app>.png/.jpg/.jpeg/.webp`。
- 其次寻找 `target.*`、`original.*`、`input.*`、`image.*`。
- 如果目录里只有一张图片，也会自动使用这张图。
- 如果目录里有多张图片且无法根据命名判断目标图，会抛出明确错误。
- 旧版 `background.png + foreground.png` 仍然兼容，但新结构推荐只放一张目标图。

`theme_id` 按字面路径解析，例如 `theme_001` 对应 `data/styles/theme_001`；不会把 `theme001` 自动映射成 `theme_001`。

V2 允许为主题包和目标 App 增加轻量语义 JSON。它们只描述“是什么、做什么”，不手写设计方案：

```text
data/styles/theme_001/theme.json
data/targets/xiaohongshu/target.json
```

示例：

```json
{
  "app": "xiaohongshu",
  "display_name": "小红书",
  "category": "内容社区 / 生活记录",
  "core_function": "发布和浏览图文、视频笔记，记录和分享生活经验、商品体验、旅行、美妆、穿搭等内容"
}
```

## 环境变量

在项目根目录放置 `.env`。该文件包含密钥，不要提交到 Git。

```env
MOCK_MODE=true

ALI_PLAN_BASE_URL=https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/api/v1
ALI_PLAN_MODEL=qwen3.7-plus
ALI_PLAN_API_KEY=sk-...

ALI_IMAGE_BASE_URL=https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/api/v1
ALI_IMAGE_MODEL=wan2.7-image-pro
ALI_IMAGE_API_KEY=sk-...
```

`MOCK_MODE=true` 时不调用真实 API；`MOCK_MODE=false` 时会真实调用 Qwen 和 Wan。

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 运行

默认示例入口在 `backend/run_case.py`：

```python
THEME_ID = "theme_001"
TARGET_APP = "bilibili"
CASE_ID = "case_001_theme_001_to_bilibili"
```

运行 mock 流程，不调用真实 API：

```powershell
$env:MOCK_MODE='true'
python backend/run_case.py
```

输出会写入：

```text
data/cases/<case_id>/
data/outputs/<case_id>/
```

典型产物：

- `theme_style_analysis.json`
- `generation_prompt.txt`
- `wan_response.json`
- `qc_report.json`
- `metadata.json`
- `candidates/candidate_01.png` 到 `candidate_03.png`
- `best_output.png`

## 端到端流程

1. `path_service` 解析主题参考样例和目标 App 单图输入。
2. `image_service` 生成可检查的 reference layout 和 target layout。
3. `qwen_client.analyze_theme` 把参考样例的 `background / foreground / style_ref` 发给 Qwen，归纳主题迁移规则。
4. `prompt_service` 把 Qwen 分析结果组装成 Wan 生成 prompt。
5. `wan_client.generate_candidates` 将 style_ref 参考图和目标图发给 Wan，默认生成 3 个候选。
6. `qwen_client.score_candidates` 用 Qwen 对 3 个候选图打分。
7. `qc_service` 选择最佳候选并复制为 `best_output.png`。
8. `storage_service` 保存 metadata 和各类中间产物路径。

## 批量主题包生成

单图 workflow：

```bash
python backend/run_case.py
```

批量主题包 workflow：

```bash
python backend/run_package.py
```

批量模式会自动扫描 `data/targets/` 下所有有效目标 App，并输出到：

```text
data/packages/package_001_theme_001/
```

批量模式的核心不是简单循环单图生成，而是尽量保证整包风格一致：

- `theme_001` 的风格只分析一次。
- 所有 target App 共用同一个 `theme_style_analysis.json` 和 `theme_rules.json`。
- 所有 target App 共用同一个 `generation_base_prompt.txt`。
- 每个 target App 会先生成 `target_identity.json`，分析原始图标必须保留的轮廓、符号、识别色和构图。
- 每个 target App 会再生成 `transfer_plan.json`，把整包主题规则和目标身份锚点合并成单图执行计划。
- 每个 target App 的 `generation_prompt.txt` 只追加迁移计划和身份保留要求，不重新定义主题风格。
- 每个 target App 默认生成 3 个候选图，并选择一个 `best_output.png`。
- 候选选择采用身份优先策略：目标身份分过低的候选不会因为整体风格分高而被优先选中；如果 3 个候选身份分都偏低，会在 `qc_report.json` 标记 `needs_retry=true`。
- 所有 `best_output` 会复制到 `final/`。
- `final/` 会合成为 `contact_sheet.png`，再用 Qwen 做整包一致性 QC。

批量输出结构：

```text
data/packages/package_001_theme_001/
  theme_rules.json
  theme_style_analysis.json
  theme_design_analysis.json
  generation_base_prompt.txt
  target_apps.json
  contact_sheet.png
  package_qc_report.json
  metadata.json

  cases/
    bilibili/
      target_layout.png
      target_identity.json
      identity_strategy.json
      transfer_plan.json
      generation_prompt.txt
      candidates/
        candidate_01.png
        candidate_02.png
        candidate_03.png
      best_output.png
      qc_report.json

  final/
    bilibili.png
    damai.png
    hanglvzongheng.png
    meituan.png
    qq.png
    tieba.png
    wps.png
    xiaohongshu.png
```

mock 测试批量流程：

```powershell
$env:MOCK_MODE='true'
python backend/run_package.py
```

真实 API 批量流程：

```powershell
$env:MOCK_MODE='false'
python backend/run_package.py
```

真实模式会对每个目标 App 调用 Wan 生成候选，并调用 Qwen 做单图 QC 和整包 QC，会消耗更多 API 额度和时间。

### V1 身份优先优化

当前 V1 仍然是单轮 workflow，不做复杂自动多轮 agent。它把整包迁移拆成三个稳定阶段：

1. `theme_rules.json`：只从 `background / foreground / style_ref` 学习整包共用主题规则。
2. `target_identity.json`：逐个分析目标 App 原图，明确哪些主体结构、关键符号和原色倾向必须保留。
3. `transfer_plan.json`：把主题规则套到目标身份上，告诉 Wan 哪些地方能风格化、哪些地方不能替换。

这样做的目标是避免整包风格一致但 App 主体丢失，例如 WPS、小红书这类图标被改成无法识别的通用主题形状。

### V2 语义辅助策略层

V2 第一阶段不做自动 retry，重点是把“原 logo 换皮”升级为“理解 App 功能语义后的主题化表达”。

新增流程：

1. 读取 `theme.json`，了解参考 App 的名称、分类和核心功能。
2. Qwen 结合参考样例的 `background / foreground / style_ref` 输出 `theme_design_analysis.json`。
3. `theme_design_analysis.json` 中包含整包共用的 `theme_board`，用于约束所有目标 App 的颜色、线条、材质、背景和构图语言。
4. 读取每个目标 App 的 `target.json`。
5. Qwen 根据目标图、`target.json`、`theme_design_analysis.json` 和 `theme_rules.json` 输出 `identity_strategy.json`。
6. `identity_strategy.json` 动态决定目标 App 使用 `logo_preserve`、`logo_simplify`、`semantic_recompose` 或 `symbolic_scene`，并给出 `identity_constraint_level`。
7. `transfer_plan.json` 只保留 Wan 可执行内容，不把长篇设计推理直接塞进生成 prompt。

V2 QC 指标包含：

- `style_match_score`
- `target_recognition_score`
- `semantic_fit_score`
- `identity_constraint_score`
- `over_recompose_risk`
- `artifact_score`

候选选择会优先过滤目标识别度低、身份约束不达标、重构过度风险高或画面伪影明显的结果。

## 测试

```bash
python -m unittest discover -v
```

测试覆盖：

- 新版目标 App 单图解析。
- 旧版 target background / foreground 兼容。
- `theme_001` 字面路径规则。
- layout 合成。
- mock Qwen / Wan 流程。
- 3 候选图生成和 QC 选择。
- V1 身份分析、迁移计划和身份优先候选选择。
- V2 主题设计分析、身份表达策略和语义辅助 QC 字段。
- 真实 API 客户端调用参数和错误处理。

## 注意事项

- 不要提交 `.env`。
- `data/cases/`、`data/outputs/` 和 `data/packages/` 是运行产物，默认不提交。
- prompt 模板写成规则型，不把某个主题的具体元素写死；具体风格由 Qwen 根据当前参考图提取。
- 真实端到端测试会消耗 API 额度，并可能需要较长等待时间。
