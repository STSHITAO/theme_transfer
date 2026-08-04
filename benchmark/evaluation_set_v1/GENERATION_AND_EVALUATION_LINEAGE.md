# 原生成与测评链路

## 原来如何生图

旧的整包生成入口是 `backend.package_workflow.run_package_workflow`：

1. 从 `data/styles/<theme_id>/<app>/` 读取“原图 + style_ref”成对参考图，并从 `data/targets/<app>/` 读取目标 App 原图。
2. Qwen 先生成整包级描述：`theme_style_analysis.json`、`theme_rules.json` 和 `theme_design_analysis.json`。
3. 每个目标 App 再生成 `target_identity.json`、`identity_strategy.json` 和 `transfer_plan.json`。
4. `prompt_service` 把整包规则和单 App 迁移计划组合为 `generation_prompt.txt`。
5. Wan 接收多张主题参考图和最后一张目标原图，生成多个候选图；原始返回保存在 `wan_response.json`。
6. Qwen 视觉 QC 对候选图打分并选出最佳图，随后执行整包 QC，最终图片发布到 `data/packages/<package_id>/final/`。

这些 JSON 的角色不同：主题描述和迁移计划是生成输入；`wan_response.json`、`qc_report.json` 和 `package_qc_report.json` 是生成日志或辅助筛选结果，不是 ITTE 的客观真值。

## 原来如何测评

旧测评入口是 `evaluation.tpqs_workflow.run_tpqs`。它读取主题参考图、目标原图和 `final/` 中的生成图，通过 ITTE v1.2 计算：

- 风格保真 35%：VGG Gram、DISTS texture、视觉属性和 DINO motif。
- 身份保持 30%：DINO dense、DISTS structure 和 LPIPS content。
- 整包一致性 20%。
- 视觉质量 15%。

ITTE v1.2 明确设置 `text_policy = out_of_scope`，OpenCLIP 和生成阶段的 Qwen QC 都不进入主分数。测评目录中的 JSON 主要是输入清单、距离矩阵和结果报告，不是图像描述驱动的评分。

## 与当前 Benchmark 的关系

当前 Benchmark 的四个主题来自另一批真实主题图标。它们的 `theme_001` 等 ID 只是本数据集内的编号，与 `data/styles/theme_001` 的旧生成主题没有语义对应关系。旧 JSON 不能直接复制到这里。

本数据集新增 `theme_descriptions.json`，用于人工理解、失败归因和可选的 text-assisted 实验；ITTE 主轨仍保持 image-only。若生成器使用描述，必须单独报告为 `image_plus_theme_description`，不能与主轨混算。
