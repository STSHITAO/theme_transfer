# theme_transfer

## Environment Setup

`SEG` 只是开发机上的 conda 环境名，新机器不需要也不应该依赖这个名字。任意 Python 环境只要安装 `requirements.txt` 即可运行项目。

推荐环境：

```powershell
conda create -n theme-transfer python=3.12 -y
conda activate theme-transfer
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果需要 GPU 加速，建议先按本机 CUDA 版本安装匹配的 PyTorch / TorchVision，再运行 `python -m pip install -r requirements.txt` 安装剩余依赖。CPU 也可以运行 ITTE v1.2，但 DINOv3、VGG、DISTS 和 LPIPS 会明显更慢。

快速验证：

```powershell
python -m pytest -q
```

官方 ITTE v1.2 评估：

```powershell
$env:TPQS_DEVICE='cuda'
$env:TPQS_BATCH_SIZE='4'
python evaluation/evaluate_package.py
```

DINOv3、VGG16、DISTS 和 LPIPS 构成 v1.2 的确定性纯图像评估链路。OpenCLIP、Prompt 文图分和生成阶段 Qwen QC 不参与任何主分、阈值或最终决策。

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
conda activate theme-transfer
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
conda activate theme-transfer
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

ITTE = Icon Theme Transfer Evaluation。ITTE v1.2 是独立的纯图像评估模块。它不会调用 Qwen、Wan 或其他 VLM，也不会修改生成结果。完整设计、边界和论文映射见：

[`docs/superpowers/specs/2026-07-22-itte-v12-image-only-evaluation-design.md`](docs/superpowers/specs/2026-07-22-itte-v12-image-only-evaluation-design.md)

### 评价边界

ITTE 只回答：生成图是否完成了可观测的主题视觉迁移、是否保留目标图像结构、整包是否一致、是否存在技术缺陷。

以下内容不进入主分：

- 设计师创作意图、文化叙事和主观审美。
- App 商业语义、法务和版权判断。
- OCR、文字拼写和“保留文字/禁止文字”规则。
- Prompt 文图对齐、OpenCLIP 分数。
- 生成阶段 Qwen QC 分数。Qwen 报告仅可出现在 `diagnostics.generation_qwen_qc`，并明确记录 `generation_qwen_qc_used_in_score=false`。

### 四维主分

校准前的工程先验公式为：

```text
ITTE = 35% Style Fidelity
     + 30% Identity Preservation
     + 20% Package Coherence
     + 15% Visual Quality
```

1. `Style Fidelity`

   - 40% VGG16 多层 Gram：采用 Gatys 风格表征思想。
   - 20% DISTS texture：采用 DISTS 纹理分量。
   - 25% 可解释视觉属性：颜色、背景、描边、材质、构图和复杂度。
   - 15% DINOv3 repeated motif：使用 dense patch 发现参考包反复出现的视觉局部。

2. `Identity Preservation`

   - 50% DINOv3 双向 dense patch 对应与空间一致性。
   - 30% DISTS structure。
   - 20% LPIPS content。
   - 所有身份距离都以参考包 `original -> style_ref` 的正常变形强度和冻结干扰基线归一化。

3. `Package Coherence`

   同时测量生成包内部 P90/MAD 离散度、到参考主题中心的距离和包内离群项。整包内部统一但远离参考主题，不能得到高分。

4. `Visual Quality`

   使用确定性规则检查黑边、异常透明、裁切、模糊、曝光、主体面积和边缘异常。它评价技术缺陷，不评价“是否好看”。

具体权重、`35/45/60/70/80` 等阈值均是待评测集校准的工程先验，不是论文给出的通用常数。

### 三视图预处理

- `Appearance View`：统一色彩和 Alpha 合成，用于风格与包级比较。
- `Structure View`：自动去除启动器截图背景、底部 App 名称和主题背景，用于 DINOv3/DISTS/LPIPS 身份比较。
- `Artifact View`：保留原始像素、Alpha 和边界，用于质量缺陷检测。

同一输入、权重、依赖和配置应得到完全一致的输出。报告记录输入 SHA256、模型 ID、代码 commit 和评分配置哈希。

### 运行与输出

在 `evaluation/evaluate_package.py` 设置 `THEME_ID`、`PACKAGE_ID` 和 `EVAL_ID`，然后运行：

```powershell
$env:TPQS_DEVICE='cuda'       # 没有 GPU 时使用 cpu
$env:TPQS_BATCH_SIZE='4'
python evaluation/evaluate_package.py
```

输出：

```text
data/evaluations/<eval_id>/
  itte_report.json
  tpqs_report.json                 # 兼容别名，内容与 itte_report.json 相同
  metrics.csv
  style_pairwise_distances.json
  style_delta_distances.json
  dino_pairwise_distances.json
  inputs_manifest.json
```

v1.2 不再生成 `generation_feedback_prompt.md` 或 `tpqs_feedback_retry_prompt.md`，因为评估模块没有自动 retry 机制。

### 环境变量

```text
TPQS_EMBEDDING_BACKEND=dinov3      # 官方；stats 仅用于离线轻量测试
TPQS_MODEL_SOURCE=modelscope       # 或 huggingface
TPQS_MODEL_ID=facebook/dinov3-vitb16-pretrain-lvd1689m
TPQS_DEVICE=cpu                    # 或 cuda
TPQS_BATCH_SIZE=1
TPQS_IMAGE_SIZE=224
ITTE_USE_PERCEPTUAL=true           # DISTS + LPIPS
ITTE_USE_VGG_GRAM=true
```

轻量测试模式不会下载或加载深度模型：

```powershell
$env:TPQS_EMBEDDING_BACKEND='stats'
$env:ITTE_USE_PERCEPTUAL='false'
$env:ITTE_USE_VGG_GRAM='false'
python evaluation/evaluate_package.py
```

该模式会降低 `evaluation_confidence`，不能代替正式 DINOv3 评估。

### 模型目录

模型统一放在项目内：

```text
models/modelscope/    # DINOv3
models/torch/         # VGG16
models/huggingface/   # 历史 OpenCLIP 缓存；不参与 v1.2 主分
```

仓库使用 Git LFS 管理大权重。克隆后运行 `git lfs pull`；如果 LFS 权重不可用，ModelScope/TorchVision 也会按配置下载到上述目录。

### 最新实测

`package_006_theme_001_fidelity_e2e` 的最终 v1.2 报告位于：

```text
data/evaluations/eval_012_package_006_theme_001_fidelity_e2e_itte_v12_final/
```

本次结果：总分 `74.39`，风格 `59.63`，身份 `74.05`，整包一致性 `86.65`，视觉质量 `93.13`，置信度 `medium`。最终决策为 `failed_hard_gate`，主要原因是航旅纵横视觉身份结构丢失、贴吧存在大面积黑色边角，以及包级身份 P10 低于临时门槛。该结论表示整包整体风格接近且一致，但仍有个别 App 阻止整包正式通过。

## 测试

```powershell
conda activate theme-transfer
python -m pytest -q
```

测试覆盖：

- 当前 `original + style_ref` 参考样例解析。
- 单目标和批量 mock workflow。
- Qwen / Wan 客户端参数和错误处理。
- ITTE v1.2 输入解析、三视图预处理、dense identity、组件重归一化、硬门槛和报告输出。

## 注意事项

- 不要提交 `.env`。
- 本仓库提交了用于复现实验的最新评估结果；新的临时运行结果应按实验记录策略选择是否提交。
- `models/` 包含公开模型缓存，但大文件必须通过 Git LFS 提交。
- prompt 必须保持规则型，不能把某个主题的具体元素写死；具体风格应由 Qwen 根据当前参考图动态提取。
