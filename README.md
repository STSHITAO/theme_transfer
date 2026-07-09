# theme_transfer

图标主题包风格迁移 MVP。项目目标不是简单给原 logo 换皮，而是从参考主题包中学习统一设计语言，再把目标 App 的身份和功能语义主题化重设计成一整套风格一致的图标包。

## 当前数据结构

参考主题包位于 `data/styles/<theme_id>/`。当前主结构是每个参考 App 两张图：

```text
data/styles/<theme_id>/<ref_app>/<ref_app>.jpg|png|webp
data/styles/<theme_id>/<ref_app>/<ref_app>_style_ref.jpg|png|webp
data/styles/<theme_id>/theme.json
```

- `original`：参考 App 的原始系统图标或未主题化图标，是身份和功能语义来源。
- `style_ref`：同一参考 App 已完成主题绘制后的风格图。
- `theme.json`：只写事实语义，例如 App 名称、分类、核心功能；不手写设计方案。

目标 App 位于 `data/targets/<app>/`：

```text
data/targets/<app>/<app>.png|jpg|jpeg|webp
data/targets/<app>/target.json
```

`target.json` 同样只写事实语义，不写设计方案。

## V2 Workflow

批量入口：

```powershell
conda activate SEG
python backend/run_package.py
```

主要流程：

1. 读取 `theme.json` 和多组 `original -> style_ref` 参考样例。
2. Qwen 输出 `theme_style_analysis.json` / `theme_rules.json`，总结整包共用主题规则。
3. Qwen 输出 `theme_design_analysis.json`，总结主题包的设计语言和身份处理策略。
4. 对每个目标 App 读取原图和 `target.json`，生成 `target_identity.json`。
5. Qwen 生成 `identity_strategy.json`，动态判断目标 App 哪些线索必须保留、哪些可以重构。
6. Qwen 生成可执行的 `transfer_plan.json`，再由 Wan 根据 style_ref 参考图和目标图生成 3 个候选。
7. Qwen QC 选择最佳候选，汇总为整包 `final/` 和 `contact_sheet.png`。

V2 生成优化方向：

- 目标不是只生成一套好看的图，也不是只追求包内部一致。
- 目标是让生成结果像 `theme_001` 主题包中缺失的 App 图标。
- `theme_design_analysis.json` 会显式分析 `original -> style_ref` 的颜色、背景、描边、构图、主体比例和细节复杂度迁移规律。
- `identity_strategy.json` 只决定当前 App 的身份表达方式，不能重新定义全局主题风格。
- `transfer_plan.json` 会把 theme fidelity 规则和 App 身份线索合并成 Wan 可执行指令。
- Wan prompt 会强约束 color / stroke / composition / identity，防止生成一套与 `theme_001` 不一致的新主题。

输出目录示例：

```text
data/packages/<package_id>/
  theme_rules.json
  theme_style_analysis.json
  theme_design_analysis.json
  generation_base_prompt.txt
  contact_sheet.png
  package_qc_report.json
  metadata.json
  cases/<app>/
    target_identity.json
    identity_strategy.json
    transfer_plan.json
    generation_prompt.txt
    candidates/
    best_output.png
    qc_report.json
  final/
    <app>.png
```

## 单 App Workflow

单 App 入口：

```powershell
conda activate SEG
python backend/run_case.py
```

默认参数在 `backend/run_case.py` 中修改：

```python
THEME_ID = "theme_001"
TARGET_APP = "bilibili"
CASE_ID = "case_001_theme_001_to_bilibili"
```

## 环境变量

在项目根目录放置 `.env`，不要提交该文件：

```env
MOCK_MODE=true

ALI_PLAN_BASE_URL=https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/api/v1
ALI_PLAN_MODEL=qwen3.7-plus
ALI_PLAN_API_KEY=sk-...

ALI_IMAGE_BASE_URL=https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/api/v1
ALI_IMAGE_MODEL=wan2.7-image-pro
ALI_IMAGE_API_KEY=sk-...
```

- `MOCK_MODE=true`：不调用真实 API，用于快速验证流程。
- `MOCK_MODE=false`：真实调用 Qwen 和 Wan，会消耗 API 额度。

## TPQS 评价模块

TPQS 是独立 evaluation 模块，不调用 Qwen / Wan，也不修改生成结果。它是业务诊断面板，不是绝对审美裁判。

运行：

```powershell
conda activate SEG
python evaluation/evaluate_package.py
```

输出：

```text
data/evaluations/<eval_id>/
  tpqs_report.json
  metrics.csv
  style_pairwise_distances.json
  style_delta_distances.json
  dino_pairwise_distances.json
  inputs_manifest.json
  tpqs_feedback_retry_prompt.md
```

新版 TPQS 是“终态主题包可用性评价器”，主分不再等同于严格 delta transfer。

主指标位于 `primary_scores`，并决定 `tpqs` / `tpqs_primary_score`：

- `theme_style_image_fit_score`：生成图最终是否接近当前主题 `style_ref` 的图像风格分布。
- `theme_style_text_fit_score`：可选 OpenCLIP 指标，比较生成图和 Qwen 从 `theme_design_analysis.json` 提取出的主题风格文本；默认关闭。
- `package_unity_score`：生成包内部是否像同一套主题包。
- `theme_membership_score`：生成图是否像当前主题包成员；这是 soft score，不作为硬失败条件。
- `visual_artifact_quality_score`：清晰度、过暗/过亮、主体面积、边缘复杂度等基础图像质量风险。

诊断指标位于 `diagnostic_scores`，只解释严格统计迁移路径，不再拉低主 TPQS：

- `style_delta_transfer_score`
- `color_delta_score`
- `edge_delta_score`
- `composition_delta_score`
- `complexity_delta_score`
- `visual_stats_transfer_score`

风险指标位于 `risk_scores`：

- `dino_identity_structure_risk_score`
- `dino_identity_top1_accuracy`
- `dino_identity_random_baseline`
- `dino_identity_warning_apps`

DINOv3 只作为身份结构风险雷达，不单独判断主题风格好坏。若 `data/packages/<package_id>/package_qc_report.json` 已存在，evaluation 会读取其中的 Qwen package QC 分数写入 `qwen_qc_scores`，但不会重新调用 Qwen。

当 `style_delta_transfer_score` 较低但主指标较好时，报告会输出 `package_usable_but_strict_delta_weak`，表示结果可能可用，只是严格低层统计迁移路径没有对齐。

TPQS 还会生成 `tpqs_feedback_retry_prompt.md`。第一阶段它只作为诊断型 retry prompt，不会自动调用 Wan：

- `color_delta_score` 低：强化 `theme_001` 色彩系统，避免引入不属于参考主题的新颜色。
- `edge_delta_score` 低：强化描边粗细、线条密度、边缘圆润程度与参考主题对齐。
- `composition_delta_score` 低：强化主体大小、居中程度、留白比例和背景占比。
- `identity_match_correct=false`：强化当前 App 的 identity anchor、brand cues 和 semantic cues。

启用 OpenCLIP：

```powershell
$env:TPQS_USE_OPENCLIP='true'
python evaluation/evaluate_package.py
```

OpenCLIP 模型缓存会放在项目本地 `models/openclip/` 和 `models/huggingface/` 下。默认 `TPQS_USE_OPENCLIP=false`，不会下载或加载 OpenCLIP。

DINOv3 默认从 ModelScope 下载到项目本地：

```text
models/modelscope/
models/huggingface/
models/torch/
```

离线开发可使用 stats backend：

```powershell
$env:TPQS_EMBEDDING_BACKEND='stats'
python evaluation/evaluate_package.py
```

## 测试

```powershell
conda activate SEG
python -m unittest discover -v
```

测试覆盖：

- 当前 `original + style_ref` 参考样例解析。
- 单目标和批量 mock workflow。
- Qwen / Wan 客户端参数和错误处理。
- TPQS 输入解析、报告生成和指标逻辑。

## 注意事项

- 不要提交 `.env`。
- `data/cases/`、`data/outputs/`、`data/packages/`、`data/evaluations/` 是运行结果目录，是否提交取决于你的实验记录需求。
- prompt 必须保持规则型，不能把某个主题的具体元素写死；具体风格应由 Qwen 根据当前参考图动态提取。
