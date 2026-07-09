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

## ITTE 评价模块

ITTE = Icon Theme Transfer Evaluation。

ITTE 是独立 evaluation 模块，不调用 Qwen / Wan，也不修改生成结果。它评价的不是单张图是否好看，而是图标主题包风格迁移是否成功。

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
  generation_feedback_prompt.md
  tpqs_feedback_retry_prompt.md
```

ITTE v1.1 报告会继续保留 `tpqs` / `tpqs_primary_score` 作为兼容字段，但它们等于新的 `itte_score`。v1.1 不引入 VLM，不做随机裁判；评估阶段只使用确定性图像特征、DINO/stats embedding、可选 OpenCLIP 文图相似度和生成阶段已有的 QC prior。核心结构包括：

1. `Style Transfer Effectiveness Score`

   判断 `generated_icons` 是否从 `target_originals` 的原始风格，靠近 `theme_refs/style_ref` 的终态主题风格。这是 ITTE 最核心指标。v1.1 使用加权可靠组件：

   - `style_cue_profile_match_score`：确定性高层风格 profile，自动从 style refs 学习背景色、pastel 程度、粗黑描边、主体比例、边缘密度、简洁度等可测 cue。
   - `theme_style_image_transfer_score`：整体图像风格是否更接近 `style_ref`。
   - `style_attribute_transfer_score`：属性级风格迁移分，覆盖 `color`、`background`、`stroke`、`texture_material`、`composition`、`complexity`。
   - `theme_prompt_image_alignment_score`：复用现有 OpenCLIP `theme_style_text_fit_score`，只比较主题风格提示词和生成图之间的文图对齐；仅当 `theme_style_text_fit_reliable=true` 时参与主分。

2. `Package Coherence Score`

   判断生成出来的一整包图标是否内部统一。它会参考 `package_unity_score`、`theme_membership_score` 和 `generated_internal_outlier_apps`。整包一致不等于风格迁移成功：一套图标可能内部很统一，但仍然不是指定 `theme_id` 的风格。

3. `App Identity Coherence Score`

   判断主题化后 App 身份是否仍然可识别。v1.1 不再让 Qwen package QC 直接覆盖身份分，而是把它作为 `generation_qwen_qc_prior`。主身份分由 `target_structure_retention_score`、`identity_recognition_prior_score`、`identity_separability_score` 和 `over_recomposition_penalty` 共同决定。这样可以惩罚“生成图很像主题，但原 App logo 结构被语义重画过头”的情况。

4. `Visual Quality Score`

   判断是否存在模糊、过暗、过亮、主体面积异常、边缘复杂度异常等基础 artifact 问题。`visual_stats_transfer_score` 不属于这个分数。

5. `Strict Delta Diagnostics`

   判断 `target->generated` 的低层统计变化方向是否复刻 `reference_raw->style_ref`。它只解释严格低层迁移路径，不作为 ITTE 主判定，也不会因为分数低直接导致风格迁移失败。

`style_transfer_score` 现在等于 `style_transfer_effectiveness.score`。旧的低层 delta 分保留为：

- `style_delta_transfer_score`
- `legacy_style_delta_transfer_score`
- `color_delta_score`
- `edge_delta_score`
- `composition_delta_score`
- `complexity_delta_score`
- `visual_stats_transfer_score`

ITTE 还会生成 `generation_feedback_prompt.md`，并保留旧兼容文件 `tpqs_feedback_retry_prompt.md`。它只作为诊断型生成反馈，不是自动 retry 机制，也不会自动调用 Wan：

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

可选 VGG Gram 辅助分：

```powershell
$env:TPQS_STYLE_FEATURE_BACKEND='vgg_gram_attribute'
python evaluation/evaluate_package.py
```

VGG Gram 在 ITTE 中只作为 `auxiliary_scores.vgg_gram_style_fit` 的材质/纹理辅助信号。默认不启用，不参与 `itte_score`，也不影响 `decision`。如果运行环境没有可用的 torchvision 或模型权重，evaluation 应回退到轻量属性特征，不应影响主评价流程。

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
- ITTE 输入解析、报告生成、属性级风格迁移分、decision 和兼容字段逻辑。

## 注意事项

- 不要提交 `.env`。
- `data/cases/`、`data/outputs/`、`data/packages/`、`data/evaluations/` 是运行结果目录，是否提交取决于你的实验记录需求。
- prompt 必须保持规则型，不能把某个主题的具体元素写死；具体风格应由 Qwen 根据当前参考图动态提取。
