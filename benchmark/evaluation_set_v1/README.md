# ITTE Benchmark: evaluation_set_v1

这是当前项目唯一保留的、可自包含使用的真实主题图标 Benchmark。

## 数据规模

- 固定原图身份库：91 张。
- 真实主题图：231 张（theme_001=30、theme_002=20、theme_003=127、theme_004=54）。
- 严格原图—主题图同应用配对：158 组（24、12、86、36）。
- 同应用跨主题配对：137 组。
- 资产总数：322；SHA-256 数据集指纹见 `evaluation_set_fingerprint.json`。

## 核心文件

- `assets/`：全部原图和主题图。
- `identity_gallery.csv`：固定 91-App 身份检索库。
- `theme_assets_manifest.csv`：主题资产、标签和裁切来源证据。
- `pair_manifest.csv`：原图—主题图正配对。
- `package_pair_manifest.csv`：同应用跨主题配对。
- `four_theme_core.csv`：5 个应用的四主题完整交集，只保存路径映射，不重复复制图片。
- `excluded_pairs.csv`：没有可靠原图、因而不能进入身份配对的主题资产。
- `theme_descriptions.json`：四套真实主题的人工复核描述，仅作元数据。
- `benchmark_protocol.json`：生成、测评、划分和防泄漏规则。
- `GENERATION_AND_EVALUATION_LINEAGE.md`：旧生图和 ITTE 测评链路说明。

## 描述使用规则

ITTE v1.2 主分数是 image-only，不读取描述。主题描述适合用于解释结果、按视觉属性分组以及建立独立的 text-assisted 生成轨；不得把使用描述的结果与 image-only 主轨混合。

不添加逐图片生成描述。真实主题图本身是 gold 图像，给生成器逐图描述会泄漏答案；应用身份由 `canonical_app_id`、`app_slug`、App Store ID 和固定原图共同定义。

## 生成与评测

对 `pair_manifest.csv` 中每一行：以 `original_asset_path` 为目标身份输入，从相同主题中选参考图，但必须排除该应用自身的 `themed_asset_path`。生成结果与被留出的真实主题图进行配对比较，同时使用固定身份库测试身份检索。

四个主题样本量不平衡，必须同时报告逐主题结果、宏平均和汇总微平均。校准/测试按 `canonical_app_id` 划分，不能按图片随机划分。

当前数据只包含原图与设计师完成的真实主题图，不包含人工评分和可控退化样本。它可以客观验证风格分类、身份检索、跨主题区分和真实主题混包识别，但不能单独确定视觉质量权重，也不能证明四个主维度之间唯一正确的总体权重。不要直接把模型自身分数当作权重训练真值。

## 验证

```powershell
conda run -n theme_transfer python benchmark/tools/validate_evaluation_set.py
```

验证器检查资产可读性、哈希、标签唯一性、全部配对关系、描述范围、协议约束和数据集指纹。
