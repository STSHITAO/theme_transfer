# theme_transfer

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
  targets/<target_app>/    # 目标 App 原始图
  cases/                   # 运行中间产物，默认不提交
  outputs/                 # 生成结果，默认不提交

prompts/
  qwen_theme_analysis.md   # 主题规则分析 prompt
  wan_generation.md        # Wan 生成 prompt 模板
  qwen_qc.md               # Qwen 质检 prompt
  qwen_target_identity.md  # 目标 App 身份锚点分析 prompt
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
  generation_base_prompt.txt
  target_apps.json
  contact_sheet.png
  package_qc_report.json
  metadata.json

  cases/
    bilibili/
      target_layout.png
      target_identity.json
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
- 真实 API 客户端调用参数和错误处理。

## 注意事项

- 不要提交 `.env`。
- `data/cases/`、`data/outputs/` 和 `data/packages/` 是运行产物，默认不提交。
- prompt 模板写成规则型，不把某个主题的具体元素写死；具体风格由 Qwen 根据当前参考图提取。
- 真实端到端测试会消耗 API 额度，并可能需要较长等待时间。
