# Theme Transfer

一个面向移动 App 图标的主题风格迁移与纯图像评测项目。生成链路从真实主题示例中分析共享设计规则，通过 Qwen 规划、Wan 生图和候选 QC 生成整包图标；ITTE 独立评估可观察的风格迁移、身份保持、整包一致性和技术缺陷。

ITTE = Icon Theme Transfer Evaluation。当前 v1.4 的四个主维度是 Style Fidelity、Identity Preservation、Package Coherence 和 Visual Quality。所有分值仍来自图像；风格诊断包含 VGG16 多层 Gram 特征。每个 App 在生图前冻结 `preserve_major_structure` 或 `semantic_recompose`，只用于决定 DINOv3 几何身份分进入主分还是仅保留为诊断。Package Coherence 同时衡量整包统一性与参考主题归属，因此整包内部统一但远离参考主题，不能得到高分。评测只输出诊断与报告，不把评测文字反馈接回生图链路。

## 项目状态

- 生成工作流：已实现。
- ITTE v1.3：图像指标已在真实设计师 Benchmark 上完成五折验证。v1.4 新增生成前结构适用性策略，正在通过四主题全量生图验证；DISTS/LPIPS 仍保留为诊断，不进入身份主分。
- Benchmark：`evaluation_set_v1` 已通过独立完整性校验。
- GPU：使用 Conda 环境 `pytorch`，支持 CUDA 与磁盘特征缓存；保留 CPU 接口。
- Web UI / 部署：尚未实现。

最新的 theme-learning-v2 实验已完成：Qwen 分批学习主题内全部真实 `original -> style_ref` 配对，再结合现有 `target.json` 和目标原图为每个主题/App 动态冻结结构保留或用途语义重构路线。theme_003、theme_004 使用同一组 40 个 App，每 App 生成两个候选，生成与 ITTE 覆盖均为 100%。

| 主题 | 学习配对 | 候选 / Final | 结构保留 / 语义重构 | ITTE | 风格 | 身份 | 整包一致性 | 视觉质量 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| theme_003 | 86 | 80 / 40 | 29 / 11 | 89.03 | 90.19 | 94.34 | 70.96 | 99.81 | `failed_hard_gate`：`xiaohongshu` 整包离群 |
| theme_004 | 36 | 80 / 40 | 26 / 14 | 87.27 | 86.08 | 89.90 | 76.69 | 98.88 | `failed_hard_gate`：`wechat` 整包离群 |

高平均分不代表实验通过：两个包都保留了真实硬门失败。当前实验允许目标 App 同时出现在主题学习配对中，因此衡量的是主题包内重建，而不是未见 App 泛化；详细边界、QC 失败列表和客观评测证据见实验报告。

进度与待办见 [docs/PROGRESS.md](docs/PROGRESS.md)，产品和技术说明见 [docs/PRODUCT_SPEC.md](docs/PRODUCT_SPEC.md) 与 [docs/TECH_DESIGN.md](docs/TECH_DESIGN.md)。
最新的 theme_003/theme_004 40-App 全配对学习、生图和 ITTE 结果见 [docs/THEME_LEARNING_V2_EXPERIMENT.md](docs/THEME_LEARNING_V2_EXPERIMENT.md)。

## 环境

当前机器推荐直接使用已配置的 GPU 环境：

```powershell
conda activate pytorch
$env:TPQS_DEVICE='cuda:0'
$env:TPQS_BATCH_SIZE='2'
```

CPU 兼容运行：

```powershell
conda activate pytorch
$env:TPQS_DEVICE='cpu'
$env:TPQS_BATCH_SIZE='1'
```

如需新建环境，先按照本机驱动安装匹配的 CUDA PyTorch/TorchVision，再安装 `requirements.txt`，避免通用安装覆盖 GPU 构建。

## 生成主题包

在 `.env` 中配置本地 API 参数后运行：

```powershell
conda activate pytorch
python backend/run_package.py
```

输入位于：

```text
data/styles/<theme_id>/<reference_app>/
data/targets/<target_app>/
```

每个 `data/targets/<target_app>/target.json` 提供 App 名称、类别、商店描述和核心功能等中性事实，不预先指定钱包、准星等视觉对象，也不写死结构保留模式。Qwen 先结合 `theme.json` 分批分析主题内全部 `original -> style_ref` 配对，归纳设计师在什么条件下保留主体结构、什么条件下按软件用途重构，再结合目标原图和 `target.json` 为当前“主题 × App”冻结生成路线。最终 Wan Prompt 会直接使用这些元数据及 Qwen 生成的执行 brief。

输出位于 `data/packages/<package_id>/`。`.env` 包含密钥，严禁提交。

规范化 `dataset/` 转换为生图入口，并显式清除旧入口数据：

```powershell
python scripts/prepare_generation_data.py --clean
```

当前四主题全量真实验证为每个 App 生成两个候选、再由多模态 QC 选择最终图，并按完整 case 断点续跑：

```powershell
python scripts/run_full_generation.py --candidate-count 2
```

若单个输入被 Wan 以 `DataInspectionFailed` 拒绝，批处理会保存 `case_failure.json` 和包级 `package_failures.json` 后继续下一 App。ITTE 对实际成功输出评分，并在报告中记录请求数、评测数、跳过 App 和覆盖率；四主题横向比较同时记录共同成功 App 交集。

只运行一个主题：

```powershell
python scripts/run_full_generation.py --theme-id theme_001 --candidate-count 2
```

固定抽取覆盖完整 App 列表的 40 个目标进行两候选实验：

```powershell
python scripts/run_full_generation.py --theme-id theme_003 --theme-id theme_004 --target-limit 40 --candidate-count 2 --package-prefix package_theme_learning_v2
```

生成完成后在 GPU 上批量评测：

```powershell
$env:TPQS_DEVICE='cuda:0'
$env:TPQS_BATCH_SIZE='2'
python scripts/evaluate_full_packages.py
```

后台监控全量任务（默认只读，每 10 分钟记录一次状态）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/monitor_full_pipeline.ps1
```

监控记录位于 `data/packages/_full_generation_logs/monitor_status.jsonl`；自动重启会再次调用外部生图 API，需明确确认后才使用 `-AutoRestart`。

实验结束后生成可复现上传清单（包含 SHA-256，排除可重建的特征缓存）：

```powershell
python scripts/build_experiment_manifest.py
```

## 运行已有包的 ITTE

在 `evaluation/evaluate_package.py` 中设置主题、包和评测 ID，然后运行：

```powershell
conda activate pytorch
$env:TPQS_DEVICE='cuda:0'
$env:TPQS_BATCH_SIZE='2'
python evaluation/evaluate_package.py
```

## 验证当前 ITTE 方案

真实 Benchmark 位于 [benchmark/evaluation_set_v1](benchmark/evaluation_set_v1)。当前版本基线 Runner 不修改 ITTE 算法：

```powershell
conda activate pytorch
$env:TPQS_DEVICE='cuda:0'
$env:TPQS_BATCH_SIZE='2'
python benchmark/tools/evaluate_current_itte.py --references 8 --queries 4
```

Runner 使用身份不重叠的参考/查询划分，对比设计师真实主题图与未迁移原图控制，并把报告写入 Benchmark 的 `results/`。冻结 v1.2、最终 v1.3 以及逐折对照分别位于：

- `benchmark/evaluation_set_v1/results/itte_v12_baseline_gpu_cv/`
- `benchmark/evaluation_set_v1/results/itte_v13_final_gpu_cv/`
- `benchmark/evaluation_set_v1/results/itte_v13_final_gpu_cv/comparison_to_v12/`

真实标签身份检索证据位于 `identity_retrieval_dinov3/` 与 `identity_retrieval_perceptual/`。

设备接口冒烟：

```powershell
python benchmark/tools/smoke_itte_device.py --device cuda:0
python benchmark/tools/smoke_itte_device.py --device cpu
```

## 测试

```powershell
conda activate pytorch
python -m pytest -q
python benchmark/tools/validate_evaluation_set.py
```

## 重要边界

- ITTE 主分只使用图像证据。
- OpenCLIP、Prompt 文图匹配和生成阶段 Qwen QC 不进入主分。
- 评测结果不回灌生成链路，不再生成 `generation_feedback_prompt.md`。
- Benchmark 主题描述只用于解释，不参与评分。
- 修改指标前必须保存相同数据划分下的原版本基线，修改后使用同一划分复测。
- 模型缓存位于 `models/`，特征缓存位于 `data/evaluations/_cache/`；不要提交临时或不完整下载文件。
