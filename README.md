# theme_transfer

图标主题风格迁移 MVP。项目目标是从一个主题包里的多个参考 App 图标中归纳共同的视觉规则，再把这些规则迁移到目标 App 图标上，输出候选图、质检结果和最终推荐图。

当前工作流使用：

- Qwen 多模态模型：分析主题风格、做候选图 QC。
- Wan 图像生成模型：根据主题参考图和目标图生成迁移候选。
- 本地 mock 模式：不调用真实 API，用于快速测试流程。

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
  targets/<target_app>/    # 目标 App 输入
  cases/                   # 运行中间产物，默认不提交
  outputs/                 # 生成结果，默认不提交

prompts/
  qwen_theme_analysis.md   # 主题规则分析 prompt
  wan_generation.md        # Wan 生成 prompt 模板
  qwen_qc.md               # Qwen 质检 prompt

tests/                     # 单元测试和 mock workflow 测试
```

## 输入约定

主题包路径：

```text
data/styles/theme_001/alipay/alipay_background.png
data/styles/theme_001/alipay/alipay_foreground.png
data/styles/theme_001/alipay/alipay_style_ref.jpg
```

目标 App 路径：

```text
data/targets/bilibili/background.png
data/targets/bilibili/foreground.png
```

说明：

- `theme_id` 会按字面路径解析，例如 `theme_001` 对应 `data/styles/theme_001`。
- 当前不会把 `theme001` 自动映射成 `theme_001`。
- 如果输入素材的 `background` 和 `foreground` 本身完全一样，流程仍可运行，但无法真正学习到分层差异。

## 环境变量

在项目根目录放置 `.env`。该文件包含密钥，不要提交到 Git。

需要的变量：

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

运行：

```bash
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
- `candidates/candidate_01.png` 等候选图
- `best_output.png`

## 端到端流程

1. `path_service` 解析主题参考样例和目标 App 输入。
2. `image_service` 生成可检查的 reference layout 和 target layout。
3. `qwen_client.analyze_theme` 把每个参考样例的 background / foreground / style_ref 分别发送给 Qwen，归纳主题规则。
4. `prompt_service` 把 Qwen 分析结果组装成 Wan 生成 prompt。
5. `wan_client.generate_candidates` 将 style_ref 参考图和目标 foreground 发给 Wan，生成候选图。
6. `qwen_client.score_candidates` 用 Qwen 对候选图打分。
7. `qc_service` 选择最佳候选并复制为 `best_output.png`。
8. `storage_service` 保存 metadata 和各类中间产物路径。

## 测试

```bash
python -m unittest discover -v
```

测试覆盖：

- 路径解析和 `theme_001` 字面路径规则。
- layout 合成。
- mock Qwen / Wan 流程。
- prompt 保存。
- QC 候选图选择。
- 真实 API 客户端调用参数和错误处理。

## 注意事项

- 不要提交 `.env`。
- `data/cases/` 和 `data/outputs/` 是运行产物，默认不提交。
- prompt 模板写成规则型，不把某个主题的具体元素写死；具体风格应由 Qwen 根据当前参考图提取。
- 真实端到端测试会消耗 API 额度，并可能需要较长等待时间。
